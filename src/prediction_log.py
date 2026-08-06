"""Every served prediction, persisted — the record that makes monitoring possible.

Nothing downstream can be reconstructed without this. A forecast is a claim about a
future hour, and once that hour passes the claim is either right or wrong; if it was
never written down, the question cannot be asked again. Drift reports and rolling error
in Phase 8 are joins against this table, so it is a first-class component rather than a
side effect of the endpoint.

**Scoring against a moving target.** ENTSO-E revises published actuals, so a prediction
scored on Monday against a provisional value may deserve a different score on Friday.
Scores are therefore upserted against the *latest* available actual and stamped with
`scored_at`, rather than written once and frozen.

Timestamps are stored as ISO-8601 UTC text: sortable as strings, unambiguous across the
DST transitions this project is careful about, and readable when someone opens the file
with the sqlite3 CLI at an awkward hour.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_at    TEXT    NOT NULL,
    target_time     TEXT    NOT NULL,
    horizon_hours   INTEGER NOT NULL,
    load_mw         REAL    NOT NULL,
    p10             REAL,
    p50             REAL,
    p90             REAL,
    model_name      TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    run_id          TEXT,
    dataset_version TEXT,
    features        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS predictions_target ON predictions (target_time);
CREATE INDEX IF NOT EXISTS predictions_made   ON predictions (predicted_at);

CREATE TABLE IF NOT EXISTS prediction_scores (
    prediction_id     INTEGER PRIMARY KEY REFERENCES predictions (id) ON DELETE CASCADE,
    actual_mw         REAL NOT NULL,
    abs_error_mw      REAL NOT NULL,
    ape               REAL NOT NULL,
    tso_forecast_mw   REAL,
    tso_abs_error_mw  REAL,
    scored_at         TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class PredictionRecord:
    """One served prediction for one target hour."""

    predicted_at: pd.Timestamp
    target_time: pd.Timestamp
    horizon_hours: int
    load_mw: float
    model_name: str
    model_version: str
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    run_id: str | None = None
    dataset_version: str | None = None
    features: dict[str, float] = field(default_factory=dict)


def _utc_text(value: pd.Timestamp | datetime) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError(f"Refusing to store a naive timestamp: {value!r}")
    return timestamp.tz_convert("UTC").isoformat()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PredictionLog:
    """SQLite-backed store of served predictions and their scores.

    SQLite rather than Postgres because the whole point is that this survives on a
    laptop, in an ephemeral CI runner and in a container without anything to
    administer. It is one file, which is also what makes it easy to persist between
    scheduled runs (see ADR-008).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            stored = connection.execute("SELECT version FROM schema_version").fetchone()
            if stored is None:
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif stored["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Prediction log at {self.path} is schema v{stored['version']}, "
                    f"this code expects v{SCHEMA_VERSION}. Migrate or move the file aside."
                )

    @property
    def schema_version(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT version FROM schema_version").fetchone()[0])

    def log(self, records: Sequence[PredictionRecord]) -> int:
        """Append served predictions. Returns the number of rows written.

        Every call is kept, including repeat forecasts for the same target hour: what
        was served is a fact, and overwriting it would erase the audit trail. Monitoring
        picks the most recent one per target via `latest_per_target`.
        """
        if not records:
            return 0
        rows = [
            (
                _utc_text(r.predicted_at),
                _utc_text(r.target_time),
                int(r.horizon_hours),
                float(r.load_mw),
                r.p10,
                r.p50,
                r.p90,
                r.model_name,
                str(r.model_version),
                r.run_id,
                r.dataset_version,
                json.dumps({k: float(v) for k, v in r.features.items()}, sort_keys=True),
            )
            for r in records
        ]
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO predictions
                   (predicted_at, target_time, horizon_hours, load_mw, p10, p50, p90,
                    model_name, model_version, run_id, dataset_version, features)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        logger.info("Logged {} prediction(s) to {}", len(rows), self.path)
        return len(rows)

    def read(
        self,
        *,
        since: pd.Timestamp | None = None,
        until: pd.Timestamp | None = None,
        model_version: str | None = None,
        with_features: bool = False,
    ) -> pd.DataFrame:
        """Logged predictions as a frame indexed by target hour.

        `since`/`until` filter on the target hour, which is what monitoring windows are
        expressed in — not on when the prediction was made.
        """
        clauses, params = [], []
        if since is not None:
            clauses.append("target_time >= ?")
            params.append(_utc_text(since))
        if until is not None:
            clauses.append("target_time <= ?")
            params.append(_utc_text(until))
        if model_version is not None:
            clauses.append("model_version = ?")
            params.append(str(model_version))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as connection:
            frame = pd.read_sql_query(
                f"SELECT * FROM predictions {where} ORDER BY target_time, predicted_at",
                connection,
                params=params,
            )
        return self._shape(frame, with_features=with_features)

    def _shape(self, frame: pd.DataFrame, *, with_features: bool) -> pd.DataFrame:
        if frame.empty:
            columns = [
                "id",
                "predicted_at",
                "horizon_hours",
                "load_mw",
                "p10",
                "p50",
                "p90",
                "model_name",
                "model_version",
                "run_id",
                "dataset_version",
            ]
            empty = pd.DataFrame(columns=columns)
            empty.index = pd.DatetimeIndex([], tz="UTC", name="target_time")
            return empty

        frame["predicted_at"] = pd.to_datetime(frame["predicted_at"], utc=True)
        frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
        if with_features:
            features = pd.DataFrame([json.loads(f) for f in frame["features"]], index=frame.index)
            frame = pd.concat([frame.drop(columns=["features"]), features], axis=1)
        else:
            frame = frame.drop(columns=["features"])
        return frame.set_index("target_time").sort_index()

    def latest_per_target(self, **kwargs) -> pd.DataFrame:
        """One row per target hour: the most recent forecast made for it.

        Rolling error is a statement about the forecast that stood, not about how many
        times a client happened to ask for it.
        """
        frame = self.read(**kwargs)
        if frame.empty:
            return frame
        return (
            frame.reset_index()
            .sort_values("predicted_at")
            .drop_duplicates(subset=["target_time", "horizon_hours", "model_version"], keep="last")
            .set_index("target_time")
            .sort_index()
        )

    def score(self, actuals: pd.DataFrame) -> int:
        """Match logged predictions to actual load and upsert their scores.

        `actuals` is the ingested dataset — UTC-hourly with `load_mw` and optionally
        `tso_forecast_mw`, so the same join yields the benchmark's error on exactly the
        hours the model was scored on.

        Every scoreable prediction is re-scored on each call rather than only the
        unscored ones. That is deliberate: ENTSO-E revises actuals, so an old score can
        become wrong without anything in this table changing.
        """
        if "load_mw" not in actuals.columns:
            raise ValueError("Scoring needs an 'actual' frame with a 'load_mw' column")
        if actuals.index.tz is None:
            raise ValueError("Actuals must have a timezone-aware UTC index")

        known = actuals["load_mw"].dropna()
        if known.empty:
            logger.info("No actuals available yet; nothing to score")
            return 0
        tso = actuals.get("tso_forecast_mw", pd.Series(dtype="float64"))

        predictions = self.read()
        if predictions.empty:
            logger.info("No logged predictions to score")
            return 0

        joined = predictions.join(known.rename("actual_mw"), how="inner")
        if joined.empty:
            logger.info("None of the {} logged predictions have an actual yet", len(predictions))
            return 0
        joined = joined.join(tso.rename("tso_mw"), how="left")

        scored_at = _now()
        rows = [
            (
                int(row["id"]),
                float(row["actual_mw"]),
                abs(float(row["load_mw"]) - float(row["actual_mw"])),
                abs(float(row["load_mw"]) - float(row["actual_mw"])) / float(row["actual_mw"]),
                None if pd.isna(row.get("tso_mw")) else float(row["tso_mw"]),
                None
                if pd.isna(row.get("tso_mw"))
                else abs(float(row["tso_mw"]) - float(row["actual_mw"])),
                scored_at,
            )
            for _, row in joined.iterrows()
            if float(row["actual_mw"]) != 0
        ]
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO prediction_scores
                   (prediction_id, actual_mw, abs_error_mw, ape, tso_forecast_mw,
                    tso_abs_error_mw, scored_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(prediction_id) DO UPDATE SET
                     actual_mw        = excluded.actual_mw,
                     abs_error_mw     = excluded.abs_error_mw,
                     ape              = excluded.ape,
                     tso_forecast_mw  = excluded.tso_forecast_mw,
                     tso_abs_error_mw = excluded.tso_abs_error_mw,
                     scored_at        = excluded.scored_at""",
                rows,
            )
        logger.info("Scored {} prediction(s) against the latest actuals", len(rows))
        return len(rows)

    def scored(
        self, *, since: pd.Timestamp | None = None, latest_only: bool = True
    ) -> pd.DataFrame:
        """Predictions joined to their scores, indexed by target hour."""
        clauses, params = [], []
        if since is not None:
            clauses.append("p.target_time >= ?")
            params.append(_utc_text(since))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as connection:
            frame = pd.read_sql_query(
                f"""SELECT p.*, s.actual_mw, s.abs_error_mw, s.ape,
                           s.tso_forecast_mw AS tso_mw, s.tso_abs_error_mw, s.scored_at
                    FROM predictions p
                    JOIN prediction_scores s ON s.prediction_id = p.id
                    {where}
                    ORDER BY p.target_time, p.predicted_at""",
                connection,
                params=params,
            )
        if frame.empty:
            return self._shape(frame, with_features=False)

        frame["scored_at"] = pd.to_datetime(frame["scored_at"], utc=True)
        shaped = self._shape(frame, with_features=False)
        if latest_only and not shaped.empty:
            shaped = (
                shaped.reset_index()
                .sort_values("predicted_at")
                .drop_duplicates(
                    subset=["target_time", "horizon_hours", "model_version"], keep="last"
                )
                .set_index("target_time")
                .sort_index()
            )
        return shaped

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
