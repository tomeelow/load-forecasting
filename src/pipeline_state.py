"""The small amount of state the scheduled loop must remember between runs.

Two facts, both of which cost real damage if lost:

* **When each pipeline last succeeded.** GitHub's cron is best-effort — runs are delayed
  under load and dropped entirely during incidents. A pipeline that assumes "yesterday"
  will silently leave a hole in the dataset the first time a run is skipped, and nothing
  will ever go back for it. Ingestion asks this file how far back to reach instead.
* **Whether a retrain is owed.** Drift detection and the retraining job are separate
  steps in a separate process; the flag is how the first tells the second, and it has to
  survive the gap between them.

JSON rather than SQLite because it is written once per run, read once per run, and being
readable in a pull request diff is worth more here than transactionality. Writes are
atomic (temp file plus rename) so a runner killed mid-write cannot leave a half-file that
breaks every subsequent run.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

STATE_VERSION = 1


@dataclass(frozen=True)
class RetrainFlag:
    """Why a retrain is owed, and since when."""

    reason: str
    raised_at: datetime

    def as_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "raised_at": self.raised_at.isoformat()}


class PipelineState:
    """Last-success markers and the retrain flag, in one small JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": STATE_VERSION, "last_success": {}, "retrain_flag": None}
        try:
            payload = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Pipeline state at {self.path} is not valid JSON ({exc}). Refusing to "
                "guess: delete it to start fresh, and expect a backfill on the next run."
            ) from exc
        payload.setdefault("last_success", {})
        payload.setdefault("retrain_flag", None)
        return payload

    def _write(self, payload: dict) -> None:
        payload["version"] = STATE_VERSION
        payload["updated_at"] = datetime.now(UTC).isoformat()
        # Atomic: a killed runner leaves either the old file or the new one, never half.
        handle, temporary = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def last_success(self, pipeline: str) -> pd.Timestamp | None:
        """When `pipeline` last completed, or None if it never has."""
        raw = self._read()["last_success"].get(pipeline)
        return None if raw is None else pd.Timestamp(raw).tz_convert("UTC")

    def record_success(self, pipeline: str, when: pd.Timestamp | None = None) -> pd.Timestamp:
        """Mark `pipeline` as having succeeded, and return the timestamp recorded."""
        moment = pd.Timestamp(when or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
        payload = self._read()
        payload["last_success"][pipeline] = moment.isoformat()
        self._write(payload)
        logger.debug("Recorded success for '{}' at {}", pipeline, moment)
        return moment

    def days_since(self, pipeline: str, now: pd.Timestamp | None = None) -> float | None:
        """Days since `pipeline` last succeeded, or None if it never has."""
        last = self.last_success(pipeline)
        if last is None:
            return None
        now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
        return (now - last).total_seconds() / 86_400

    def raise_retrain_flag(self, reason: str) -> RetrainFlag:
        """Record that a retrain is owed. The earliest reason wins and is kept.

        Repeated drift on consecutive days should not keep resetting the clock — what
        matters is that a retrain has been outstanding since some moment, and how long.
        """
        payload = self._read()
        existing = payload.get("retrain_flag")
        if existing:
            logger.info("Retrain flag already set: {}", existing["reason"])
            return RetrainFlag(existing["reason"], datetime.fromisoformat(existing["raised_at"]))

        flag = RetrainFlag(reason=reason, raised_at=datetime.now(UTC))
        payload["retrain_flag"] = flag.as_dict()
        self._write(payload)
        logger.warning("Retrain flag raised: {}", reason)
        return flag

    def retrain_flag(self) -> RetrainFlag | None:
        raw = self._read()["retrain_flag"]
        return (
            None
            if raw is None
            else RetrainFlag(raw["reason"], datetime.fromisoformat(raw["raised_at"]))
        )

    def clear_retrain_flag(self) -> None:
        """Clear the flag once a retrain has actually been attempted.

        Cleared on attempt rather than on promotion: a candidate that fails the gate has
        answered the question the flag asked, and leaving it raised would retrain the
        same losing model every night.
        """
        payload = self._read()
        if payload.get("retrain_flag") is None:
            return
        logger.info("Clearing retrain flag: {}", payload["retrain_flag"]["reason"])
        payload["retrain_flag"] = None
        self._write(payload)

    def snapshot(self) -> dict:
        """The whole state, for logging at the end of a run."""
        return self._read()
