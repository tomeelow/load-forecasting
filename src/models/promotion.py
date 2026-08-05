"""The promotion gate: the one function standing between a scheduled retrain and production.

Deliberately free of MLflow, filesystem and network so it can be tested in isolation and
reasoned about completely. Automated retraining (Phase 7) calls it unattended, at an
hour when nobody is watching, which is exactly why it must be boring and correct.

Two conditions, both required:

1. **Beat the naive seasonal baseline.** A model that cannot beat `load[T - 168]` is
   broken, whatever else its metrics say.
2. **Do not materially regress against the current champion.** A retrain that produces
   a worse model must leave production alone; a slightly worse one is tolerated within
   `max_regression` so the champion is not frozen by noise.

Both are stated in terms of an error metric where lower is better (MAPE by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict, with the reasoning attached so it can be logged verbatim."""

    promote: bool
    reason: str
    candidate: float
    naive: float
    champion: float | None
    max_regression: float

    def __str__(self) -> str:
        verdict = "PROMOTE" if self.promote else "HOLD"
        return f"{verdict}: {self.reason}"


def _finite(value: float | None) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def promotion_decision(
    candidate: float,
    naive: float,
    champion: float | None = None,
    *,
    max_regression: float = 1.02,
) -> GateDecision:
    """Decide whether `candidate` should replace the current champion.

    `champion` is `None` the first time round, when nothing is in production yet; the
    regression check is then vacuous and only the naive baseline has to be beaten.

    A non-finite metric never promotes. A NaN slipping through a failed evaluation must
    not be able to reach production by accident — this fails closed on purpose.
    """
    if max_regression < 1:
        raise ValueError(f"max_regression must be at least 1.0, got {max_regression}")

    def decision(promote: bool, reason: str) -> GateDecision:
        return GateDecision(promote, reason, candidate, naive, champion, max_regression)

    if not _finite(candidate):
        return decision(False, f"candidate metric is not a finite number ({candidate})")
    if not _finite(naive):
        return decision(False, f"naive baseline metric is not a finite number ({naive})")
    if candidate >= naive:
        return decision(
            False, f"candidate {candidate:.4f} does not beat the naive baseline {naive:.4f}"
        )

    if champion is None:
        return decision(
            True, f"candidate {candidate:.4f} beats naive {naive:.4f} and there is no champion yet"
        )
    if not _finite(champion):
        return decision(False, f"champion metric is not a finite number ({champion})")

    ceiling = champion * max_regression
    if candidate > ceiling:
        return decision(
            False,
            f"candidate {candidate:.4f} regresses past the champion's tolerance "
            f"{ceiling:.4f} (= {champion:.4f} x {max_regression})",
        )
    return decision(
        True,
        f"candidate {candidate:.4f} beats naive {naive:.4f} and stays within "
        f"{ceiling:.4f} of the champion {champion:.4f}",
    )
