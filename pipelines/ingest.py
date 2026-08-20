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
from loguru import logger

from src.config import Config, load_config, load_project_env
from src.evaluation.splits import minimum_training_hours
from src.ingestion.dataset import (
    build_dataset,
    merge_datasets,
    new_run_id,
    read_dataset,
    write_dataset,
)
from src.ingestion.entsoe_client import (
    LOAD_COLUMN,
    fetch_load_frame,
    make_client,
    trailing_window,
)
from src.ingestion.validate import ValidationReport, validate_dataset
from src.ingestion.weather_client import fetch_national_archive, fetch_national_forecast
from src.pipeline_state import PipelineState

PIPELINE = "ingest"

ARCHIVE_LAG_DAYS = 7  # generous margin on Open-Meteo's few-day archive delay
MAX_PAST_DAYS = 92  # Open-Meteo's documented maximum for `past_days`


def _window(
    cfg: Config, full: bool, *, last_success: pd.Timestamp | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The window to pull.

    A full run rebuilds the configured history. An incremental run re-pulls the trailing
    window — but stretches it back to the last successful run when that is further away,
    because GitHub's cron is best-effort and a skipped night must not leave a hole that
    nothing ever returns for. Reaching too far back costs a slower run; reaching too
    little costs data that is gone for good.
    """
    if not full:
        start, end = trailing_window(cfg.ingestion.trailing_repull_days)
        if last_success is not None and last_success < start:
            widened = last_success.tz_convert("UTC").floor("D")
            logger.warning(
                "Data ends at {} ({:.1f} days ago), beyond the {}-day trailing "
                "window — backfilling from {}",
                last_success,
                (end - last_success).total_seconds() / 86_400,
                cfg.ingestion.trailing_repull_days,
                widened,
            )
            start = widened
        return start, end

    tz = cfg.data.timezone_local
    start = pd.Timestamp(cfg.data.start_date, tz=tz).tz_convert("UTC")
    end = (
        pd.Timestamp(cfg.data.end_date, tz=tz).tz_convert("UTC")
        if cfg.data.end_date
        else pd.Timestamp.now(tz="UTC").ceil("h")
    )
    return start, end


def _rebuild_reason(cfg: Config, existing: pd.DataFrame | None) -> str | None:
    """Why this run must rebuild the configured history instead of topping it up.

    Two situations, neither of which an incremental pull can climb out of on its own:

    * **No dataset at all.** An incremental pull onto nothing stores only the trailing
      window, and because the next run re-pulls that same window it never grows. A
      scheduled runner that lost its state would sit there forever with a fortnight of
      data and a model that cannot be trained.
    * **A dataset too short to train on that also starts later than configured.** This
      is the first case one day later, and it is worse because it looks healthy: the
      file exists, every run merges a fresh window into it, and it grows by a day a
      night. It took the scheduled loop three weeks to accumulate 31 days, with every
      retrain failing on a 60-day validation split in the meantime.

    The second condition needs both halves. Short alone would rebuild nightly on a zone
    whose publication history is genuinely brief; later-than-configured alone would
    re-pull years because someone moved `start_date` back, which is what `--full` is
    for.
    """
    if existing is None:
        return f"no dataset at {cfg.data.dataset_path}"

    needed = minimum_training_hours(
        validation_days=cfg.model.validation_days,
        horizon=cfg.model.horizons[0],
        weekly_lag=cfg.features.weekly_lag_hours,
    )
    if len(existing) >= needed:
        return None

    configured_start = pd.Timestamp(cfg.data.start_date, tz=cfg.data.timezone_local).tz_convert(
        "UTC"
    )
    if existing.index.min() <= configured_start + pd.Timedelta(days=1):
        return None  # already back at the beginning; this is all the source has

    return (
        f"the stored dataset holds {len(existing)} hours, short of the {needed} a "
        f"{cfg.model.validation_days}-day validation split needs, and starts "
        f"{existing.index.min():%Y-%m-%d} rather than the configured {cfg.data.start_date}"
    )


def _resume_from(state: PipelineState, existing: pd.DataFrame | None) -> pd.Timestamp | None:
    """How far back this run has to reach: the marker, or the data itself.

    The marker is the better answer — it says when a run last *completed*. But the two
    are restored together (ADR-008) and can come apart: a state file deleted by hand, a
    partial restore, or a dataset ingested before the markers existed. The stored data
    then still knows where the hole starts, and asking it is the difference between
    backfilling the gap and re-pulling a fortnight over the top of it.
    """
    marker = state.last_success(PIPELINE)
    if marker is not None or existing is None or LOAD_COLUMN not in existing:
        return marker

    last_actual = existing[LOAD_COLUMN].last_valid_index()
    if last_actual is not None:
        logger.warning(
            "No last-success marker for '{}' but a dataset exists; resuming from its "
            "newest actual, {}",
            PIPELINE,
            last_actual,
        )
    return last_actual


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


def run(
    cfg: Config, full: bool, *, state: PipelineState | None = None
) -> tuple[Path, ValidationReport]:
    """Pull one window, merge it into the stored dataset, validate and write.

    Idempotent by construction: the window is merged onto whatever is stored, and
    `merge_datasets` replaces overlapping hours rather than appending them. Running
    twice for the same day produces the same file as running once.

    Loads `.env` itself rather than trusting `main()` to have done it. This is the only
    function in the project that needs the ENTSO-E token, and a caller that reaches it
    without going through the command line — a scheduler, a notebook, another module —
    would otherwise get "ENTSOE_API_KEY is not set" while a perfectly good `.env` sat
    unread beside it.
    """
    load_project_env()
    state = state or PipelineState(cfg.state.pipeline_state_path)
    path = cfg.data.dataset_path
    existing = read_dataset(path) if path.exists() else None

    if not full:
        reason = _rebuild_reason(cfg, existing)
        if reason is not None:
            logger.warning(
                "Rebuilding the full configured history instead of pulling the trailing window: {}",
                reason,
            )
            full = True

    start, end = _window(cfg, full, last_success=_resume_from(state, existing))
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

    dataset = merge_datasets(existing, incoming) if existing is not None else incoming

    report = validate_dataset(dataset, cfg.validation)
    write_dataset(dataset, path)
    # Recorded only once the data is on disk, so a crash mid-run leaves the marker
    # where it was and the next run backfills the window this one failed to store.
    state.record_success(PIPELINE)
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

    load_project_env()
    _, report = run(load_config(args.config), full=args.full)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
