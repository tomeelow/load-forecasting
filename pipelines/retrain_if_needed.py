"""Retrain when it is due or when drift asked for it — and only promote through the gate.

    uv run python -m pipelines.retrain_if_needed
    uv run python -m pipelines.retrain_if_needed --force   # ignore the schedule

Two triggers, either sufficient: a scheduled cadence has elapsed, or `check_drift` left
the retrain flag raised. Retraining blindly every night burns compute and churns the
registry; retraining only on a timer ignores the world changing in between.

The promotion gate from `src/models/promotion.py` decides what happens next, and it is
the same function the tests cover in isolation. A candidate that fails is logged to
MLflow and left there — production is untouched, which is the entire reason the gate
exists. Unattended retraining that can silently make things worse is worse than no
unattended retraining.

The flag is cleared once a retrain has been *attempted*, not once one has been promoted:
a candidate that lost has answered the question the flag asked, and leaving it raised
would retrain the same losing model every night.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from pipelines.train import train_horizon
from src.config import Config, load_config
from src.models.tracking import configure
from src.pipeline_state import PipelineState

PIPELINE = "retrain_if_needed"

SKIPPED = "skipped"
PROMOTED = "promoted"
TRAINED_NOT_PROMOTED = "trained_not_promoted"


@dataclass(frozen=True)
class RetrainOutcome:
    status: str
    reason: str
    horizon: int | None = None
    gate_reason: str | None = None

    def format(self) -> str:
        line = f"Retrain: {self.status.upper()} — {self.reason}"
        return f"{line}\n  gate: {self.gate_reason}" if self.gate_reason else line


def _due(cfg: Config, state: PipelineState, now: pd.Timestamp) -> tuple[bool, str]:
    """Is a retrain owed, and why?"""
    flag = state.retrain_flag()
    if flag is not None:
        return True, f"drift flag raised at {flag.raised_at:%Y-%m-%d %H:%M}: {flag.reason}"

    elapsed = state.days_since(PIPELINE, now=now)
    cadence = cfg.retraining.cadence_days
    if elapsed is None:
        return True, "no model has been retrained by this pipeline yet"
    if elapsed >= cadence:
        return True, f"scheduled cadence elapsed ({elapsed:.1f} days >= {cadence})"
    return False, f"last retrain {elapsed:.1f} days ago, cadence is {cadence} days"


def run(
    cfg: Config,
    *,
    state: PipelineState | None = None,
    now: pd.Timestamp | None = None,
    force: bool = False,
) -> RetrainOutcome:
    """Retrain if warranted, then apply the promotion gate."""
    state = state or PipelineState(cfg.state.pipeline_state_path)
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")

    due, reason = (True, "forced from the command line") if force else _due(cfg, state, now)
    if not due:
        logger.info("Retrain skipped: {}", reason)
        return RetrainOutcome(SKIPPED, reason)

    logger.info("Retraining because: {}", reason)
    configure(cfg.mlflow)
    horizon = cfg.model.horizons[0]
    result = train_horizon(cfg, horizon, tune=cfg.retraining.tune)

    # Cleared on attempt, not on promotion: the flag asked whether a better model
    # exists, and "no" is an answer.
    state.clear_retrain_flag()
    state.record_success(PIPELINE, now)

    outcome = RetrainOutcome(
        status=PROMOTED if result.promoted else TRAINED_NOT_PROMOTED,
        reason=reason,
        horizon=horizon,
        gate_reason=result.gate_reason,
    )
    logger.log("INFO" if result.promoted else "WARNING", "\n{}", outcome.format())
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrain when due or when drift flagged it, and gate promotion."
    )
    parser.add_argument("--force", action="store_true", help="retrain regardless of schedule")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    load_dotenv()
    run(load_config(args.config), force=args.force)
    # A candidate that fails the gate is a healthy outcome, not a failed run.
    return 0


if __name__ == "__main__":
    sys.exit(main())
