"""Assemble the canonical UTC-hourly dataset and persist it as parquet.

One row per hour, no holes: load and the TSO forecast from ENTSO-E, population-weighted
weather from Open-Meteo, and the Polish holiday flag. Gaps are represented as rows with
null values rather than as missing rows, so downstream code can tell "we have no data
for 03:00" apart from "03:00 does not exist", and so positional shifts in the feature
builder are also time shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.calendar_pl import DEFAULT_TZ, is_holiday
from src.ingestion.entsoe_client import FORECAST_COLUMN, LOAD_COLUMN

if TYPE_CHECKING:
    from src.config import Config

WEATHER_COLUMNS = ["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]
CANONICAL_COLUMNS = [
    LOAD_COLUMN,
    FORECAST_COLUMN,
    *WEATHER_COLUMNS,
    "is_holiday",
    "data_source_version",
]


def new_run_id() -> str:
    """Identifier for one ingestion run, stored per row as `data_source_version`.

    Provenance: a row tells you which run last wrote it, which pairs with the content
    fingerprint of the whole file to answer "what exactly was this model trained on".
    """
    return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")


def hourly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """A complete, closed UTC hourly index."""
    return pd.date_range(start, end, freq="1h", tz="UTC", name="timestamp_utc")


def _require_utc(frame: pd.DataFrame | pd.Series, what: str) -> None:
    if frame.index.tz is None:
        raise ValueError(f"{what} must have a timezone-aware index; this project stores UTC")


def build_dataset(
    load_frame: pd.DataFrame,
    weather: pd.DataFrame,
    run_id: str | None = None,
    tz: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Join load, TSO forecast, weather and holidays onto one complete hourly index.

    The index spans the union of both inputs, so weather reaching past the last known
    actual (the serving case) produces rows with a null target rather than being dropped.
    """
    _require_utc(load_frame, "load frame")
    _require_utc(weather, "weather frame")
    if load_frame.empty and weather.empty:
        raise ValueError("Cannot build a dataset from two empty frames")

    starts = [f.index.min() for f in (load_frame, weather) if not f.empty]
    ends = [f.index.max() for f in (load_frame, weather) if not f.empty]
    index = hourly_index(min(starts).tz_convert("UTC"), max(ends).tz_convert("UTC"))

    df = pd.DataFrame(index=index)
    for column in (LOAD_COLUMN, FORECAST_COLUMN):
        df[column] = load_frame[column].reindex(index) if column in load_frame else pd.NA
    for column in WEATHER_COLUMNS:
        df[column] = weather[column].reindex(index) if column in weather else pd.NA
    df = df.astype("float64")

    df["is_holiday"] = is_holiday(index, tz)
    df["data_source_version"] = run_id or new_run_id()

    logger.info(
        "Dataset: {} hourly rows {} to {} | load {} | tso forecast {} | temp {}",
        len(df),
        df.index.min(),
        df.index.max(),
        int(df[LOAD_COLUMN].notna().sum()),
        int(df[FORECAST_COLUMN].notna().sum()),
        int(df["temp_c"].notna().sum()),
    )
    return df[CANONICAL_COLUMNS]


def merge_datasets(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Overlay a freshly pulled window onto the stored dataset.

    Incoming rows win: ENTSO-E revises published actuals, so the newer pull is the
    more correct one. Rows outside the overlap keep their original values and their
    original `data_source_version`.
    """
    _require_utc(existing, "existing dataset")
    _require_utc(incoming, "incoming dataset")

    kept = existing[~existing.index.isin(incoming.index)]
    combined = pd.concat([kept, incoming]).sort_index()
    combined = combined.reindex(hourly_index(combined.index.min(), combined.index.max()))
    combined["is_holiday"] = combined["is_holiday"].astype(bool)

    logger.info(
        "Merged {} incoming rows into {} existing rows -> {} rows ({} replaced)",
        len(incoming),
        len(existing),
        len(combined),
        len(existing) - len(kept),
    )
    return combined[CANONICAL_COLUMNS]


def write_dataset(df: pd.DataFrame, path: Path) -> Path:
    """Write the dataset to parquet, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Wrote {} rows to {}", len(df), path)
    return path


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a previously written dataset, asserting the UTC index survived the round trip."""
    df = pd.read_parquet(path)
    _require_utc(df, f"dataset at {path}")
    return df


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """A short content hash of the dataset, for provenance on every training run.

    Content-addressed rather than a filename or a timestamp: ENTSO-E revises published
    actuals, so "the dataset" changes underneath a fixed path. Without this, a metric
    that moved between two runs cannot be attributed to the model or to the data.

    This detects a change; it does not preserve the old bytes. That trade — no DVC, no
    remote, no bill, in exchange for "we can tell you the data differed but not hand it
    back" — is ADR-003, and it holds only because the dataset is re-pullable for free.
    """
    digest = sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return digest.hexdigest()[:12]


@dataclass(frozen=True)
class TrainingData:
    """A frame to train on, plus where it came from."""

    frame: pd.DataFrame
    version: str
    synthetic: bool

    @property
    def description(self) -> str:
        origin = "synthetic" if self.synthetic else "ingested"
        return f"{origin}:{self.version}"


def load_training_frame(cfg: Config) -> TrainingData:
    """The ingested dataset if one exists, otherwise the synthetic generator.

    The ENTSO-E token takes days to arrive. Rather than blocking the whole pipeline on
    it, training falls back to invented data and says so loudly — every run built this
    way is tagged `synthetic` in MLflow, because metrics from invented data measure the
    plumbing and nothing else.
    """
    from src import synthetic

    path = cfg.data.dataset_path
    if path.exists():
        frame = read_dataset(path)
        logger.info("Training on the ingested dataset: {} rows from {}", len(frame), path)
        return TrainingData(frame, dataset_fingerprint(frame), synthetic=False)

    frame = synthetic.make_dataset(
        start=cfg.data.synthetic_start, end=cfg.data.synthetic_end, seed=cfg.model.seed
    )
    logger.warning(
        "No dataset at {} — falling back to SYNTHETIC data ({} rows, {} to {}). "
        "Every metric from this run measures the pipeline, not Polish demand.",
        path,
        len(frame),
        frame.index.min().date(),
        frame.index.max().date(),
    )
    return TrainingData(frame, dataset_fingerprint(frame), synthetic=True)
