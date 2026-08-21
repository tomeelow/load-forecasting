"""A row per drift check, kept so the signal can be read as a trend rather than a lamp.

The Evidently HTML report says everything about *one* comparison and nothing about how
that comparison has moved. On this target that distinction decides whether monitoring is
informative: input drift fires most nights (ADR-009), so a red/green indicator sits
permanently red and gets ignored, while the same numbers plotted over weeks show whether
the share is drifting upward, which features keep reappearing, and whether served error
is tracking any of it.

One CSV, appended once per check, in the state directory — so ADR-008's snapshot carries
it between ephemeral runs along with the prediction log. It is small (a few hundred bytes
a year) and it cannot be reconstructed: a drift check is a statement about the data as it
stood that day, and re-running it later on a revised dataset answers a different question.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from loguru import logger

COLUMNS = (
    "checked_at",
    "status",
    "drift_share",
    "drifted_features",
    "monitored_features",
    "drifted_names",
    "reference_strategy",
    "reference_rows",
    "current_rows",
    "rolling_mape",
    "tso_mape",
    "scored_predictions",
)


class DriftHistory:
    """Append-only record of what each drift check found."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: object, *, monitored_features: int | None = None) -> None:
        """Record one `DriftResult`. Typed loosely to keep monitoring off the import cycle."""
        row = {
            "checked_at": result.checked_at.isoformat(),
            "status": result.status,
            "drift_share": result.drift_share,
            "drifted_features": len(result.drifted_features),
            "monitored_features": monitored_features,
            "drifted_names": " ".join(result.drifted_features),
            "reference_strategy": result.reference_strategy,
            "reference_rows": result.reference_rows,
            "current_rows": result.current_rows,
            "rolling_mape": result.rolling_mape,
            "tso_mape": result.tso_mape,
            "scored_predictions": result.scored_predictions,
        }
        new = not self.path.exists()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        logger.debug("Appended drift check to {}", self.path)

    def read(self) -> pd.DataFrame:
        """The history as a frame indexed by check time; empty and typed when there is none."""
        if not self.path.exists():
            empty = pd.DataFrame(columns=[c for c in COLUMNS if c != "checked_at"])
            empty.index = pd.DatetimeIndex([], tz="UTC", name="checked_at")
            return empty

        frame = pd.read_csv(self.path)
        frame["checked_at"] = pd.to_datetime(frame["checked_at"], utc=True, format="mixed")
        return frame.set_index("checked_at").sort_index()
