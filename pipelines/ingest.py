"""Ingestion entrypoint: pull, assemble, validate, write.

    uv run python -m pipelines.ingest --full   # rebuild the configured history
    uv run python -m pipelines.ingest          # re-pull the trailing window and merge

The incremental form is the one that runs on a schedule. It re-pulls the last
`ingestion.trailing_repull_days` days rather than appending yesterday, because
ENTSO-E revises published actuals and some hours arrive late; those rows are
overwritten with the newer values on every run.

Weather in the incremental window comes from the archive where the archive reaches
(it lags real time by a few days) and from the forecast endpoint for the rest. Since
the re-pull window is wider than the archive lag, a row that was first written with
forecast weather is replaced by observed weather on a later run.

Exits non-zero if validation finds errors — the data is still written, so the report
can be read against the file that produced it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.config import Config, load_config
from src.ingestion.dataset import (
    build_dataset,
    merge_datasets,
    new_run_id,
    read_dataset,
    write_dataset,
)
from src.ingestion.entsoe_client import fetch_load_frame, make_client, trailing_window
from src.ingestion.validate import ValidationReport, validate_dataset
from src.ingestion.weather_client import fetch_national_archive, fetch_national_forecast

ARCHIVE_LAG_DAYS = 7  # generous margin on Open-Meteo's few-day archive delay
MAX_PAST_DAYS = 92  # Open-Meteo's documented maximum for `past_days`


def _window(cfg: Config, full: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not full:
        return trailing_window(cfg.ingestion.trailing_repull_days)
    tz = cfg.data.timezone_local
    start = pd.Timestamp(cfg.data.start_date, tz=tz).tz_convert("UTC")
    end = (
        pd.Timestamp(cfg.data.end_date, tz=tz).tz_convert("UTC")
        if cfg.data.end_date
        else pd.Timestamp.now(tz="UTC").ceil("h")
    )
    return start, end


def _weather(cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Observed weather where it exists, forecast weather for the recent past and future."""
    now = pd.Timestamp.now(tz="UTC").normalize()
    archive_end = min(end, now - pd.Timedelta(days=ARCHIVE_LAG_DAYS))
    archive = (
        fetch_national_archive(
            cfg.weather, start.strftime("%Y-%m-%d"), archive_end.strftime("%Y-%m-%d")
        )
        if archive_end >= start
        else pd.DataFrame()
    )
    if end <= archive_end:
        return archive

    forecast = fetch_national_forecast(
        cfg.weather,
        forecast_days=cfg.ingestion.weather_forecast_days,
        past_days=int(min(max((now - start.normalize()).days, 0), MAX_PAST_DAYS)),
    )
    return archive.combine_first(forecast) if not archive.empty else forecast


def run(cfg: Config, full: bool) -> tuple[Path, ValidationReport]:
    """Pull one window, merge it into the stored dataset, validate and write."""
    start, end = _window(cfg, full)
    run_id = new_run_id()
    logger.info(
        "Ingestion {} | {} | {} -> {}", run_id, "full" if full else "incremental", start, end
    )

    load_frame = fetch_load_frame(
        make_client(),
        cfg.data.country_code,
        start,
        end,
        cfg.ingestion.load_aggregation,
    )
    weather = _weather(cfg, start, end)
    incoming = build_dataset(load_frame, weather, run_id=run_id, tz=cfg.data.timezone_local)

    path = cfg.data.dataset_path
    if not full and path.exists():
        dataset = merge_datasets(read_dataset(path), incoming)
    else:
        dataset = incoming

    report = validate_dataset(dataset, cfg.validation)
    write_dataset(dataset, path)
    return path, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull ENTSO-E load and TSO forecast plus Open-Meteo weather into "
        "the canonical UTC-hourly dataset, validate it, and write parquet."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="rebuild the whole configured history instead of re-pulling the trailing window",
    )
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    load_dotenv()
    _, report = run(load_config(args.config), full=args.full)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
