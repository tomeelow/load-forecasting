"""Serve a day-ahead forecast on the schedule, and write it to the prediction log.

    uv run python -m pipelines.forecast

Without this step the prediction log stays empty forever, and everything Phase 8 measures
— rolling production error, prediction drift, the served-performance panel on the
dashboard — is silent. The log is also the one piece of state that cannot be
reconstructed after the fact (ADR-008): what the model said about tomorrow can only be
recorded before tomorrow happens.

It is deliberately not a second serving implementation. The champion is resolved through
the same registry alias the API uses, the features come from the same
`features_for_targets`, and the record written is the same `PredictionRecord`. The only
thing this module adds is *which hours to ask about*.

**Which hours.** Every target from the first hour with no published actual through the
furthest the recorded history supports — exactly `horizon` hours, and contiguous with
what the previous run covered. Anchoring to the last actual rather than to the clock is
what makes the series tile: anchoring to `now` would leave a two-hour hole every day,
right where ENTSO-E's publication lag sits, and holes are what a rolling MAPE cannot see
past.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.api.champion import Champion, load_champion
from src.config import Config, load_config, load_project_env
from src.features.inference import features_for_targets, servable_until
from src.ingestion.dataset import read_dataset
from src.ingestion.entsoe_client import LOAD_COLUMN
from src.ingestion.weather_client import fetch_national_forecast
from src.pipeline_state import PipelineState
from src.prediction_log import PredictionLog, PredictionRecord

PIPELINE = "forecast"


@dataclass(frozen=True)
class ForecastSummary:
    """What one scheduled forecast run produced."""

    logged: int
    requested: int
    model_version: str | None
    dataset_version: str | None
    first_target: pd.Timestamp | None
    last_target: pd.Timestamp | None
    note: str | None = None

    def format(self) -> str:
        if self.logged == 0:
            return f"Forecast: nothing logged — {self.note or 'no reachable target hours'}"
        line = (
            f"Forecast: logged {self.logged} of {self.requested} hour(s), "
            f"{self.first_target:%Y-%m-%d %H:%M} to {self.last_target:%Y-%m-%d %H:%M} UTC, "
            f"model {self.model_version}, dataset {self.dataset_version}"
        )
        return f"{line}\n  note: {self.note}" if self.note else line


def target_hours(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """The hours to forecast: everything after the last published actual, out to the limit.

    Empty when the dataset holds no actuals at all, which is a first-run condition rather
    than an error.
    """
    if LOAD_COLUMN not in history.columns:
        return pd.DatetimeIndex([], tz="UTC")
    latest = history[LOAD_COLUMN].last_valid_index()
    reachable = servable_until(history, horizon)
    if latest is None or reachable is None:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(latest + pd.Timedelta(hours=1), reachable, freq="1h", tz="UTC")


def _records(
    champion: Champion, features: pd.DataFrame, predicted: pd.DataFrame, made_at: datetime
) -> list[PredictionRecord]:
    """The same record shape the API writes, so monitoring cannot tell the two apart."""
    return [
        PredictionRecord(
            predicted_at=made_at,
            target_time=timestamp,
            horizon_hours=champion.horizon,
            load_mw=float(row["load_mw"]),
            p10=_optional(row.get("p10")),
            p50=_optional(row.get("p50")),
            p90=_optional(row.get("p90")),
            model_name=champion.name,
            model_version=champion.version,
            run_id=champion.run_id,
            dataset_version=champion.dataset_version,
            features=features.loc[timestamp].to_dict(),
        )
        for timestamp, row in predicted.iterrows()
    ]


def _optional(value: object) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 1)


def run(
    cfg: Config,
    *,
    log: PredictionLog | None = None,
    state: PipelineState | None = None,
    now: pd.Timestamp | None = None,
) -> ForecastSummary:
    """Forecast the reachable horizon from the current champion and log every hour.

    A missing champion is reported, not raised: on a fresh deployment the registry is
    empty until the first retrain promotes something, and failing the nightly job over a
    condition that resolves itself teaches whoever is on call to ignore a red loop.
    """
    log = log or PredictionLog(cfg.state.prediction_log_path)
    state = state or PipelineState(cfg.state.pipeline_state_path)
    made_at = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").to_pydatetime()

    try:
        champion = load_champion(cfg.mlflow)
    except Exception as exc:  # noqa: BLE001 — an empty registry is a state, not a fault
        logger.warning("No champion to forecast with: {}: {}", type(exc).__name__, exc)
        summary = ForecastSummary(0, 0, None, None, None, None, f"no champion ({exc})")
        state.record_success(PIPELINE, now)
        return summary

    history = read_dataset(cfg.data.dataset_path)
    targets = target_hours(history, champion.horizon)
    if targets.empty:
        logger.warning("Nothing to forecast: the dataset holds no usable actuals yet")
        state.record_success(PIPELINE, now)
        return ForecastSummary(
            0, 0, champion.uri, champion.dataset_version, None, None, "no actuals"
        )

    weather = fetch_national_forecast(
        cfg.weather, forecast_days=cfg.serving.weather_forecast_days, past_days=2
    )
    features = features_for_targets(
        history,
        weather,
        champion.horizon,
        targets,
        rolling_window=cfg.features.rolling_window_hours,
        weekly_lag=cfg.features.weekly_lag_hours,
        tz=cfg.data.timezone_local,
    )
    if features.empty:
        logger.warning("No requested hour could be built from the recorded history")
        state.record_success(PIPELINE, now)
        return ForecastSummary(
            0,
            len(targets),
            champion.uri,
            champion.dataset_version,
            None,
            None,
            "no buildable hours",
        )

    predicted = champion.predict(features)
    logged = log.log(_records(champion, features, predicted, made_at))

    dropped = len(targets) - logged
    summary = ForecastSummary(
        logged=logged,
        requested=len(targets),
        model_version=champion.uri,
        dataset_version=champion.dataset_version,
        first_target=predicted.index.min(),
        last_target=predicted.index.max(),
        note=(
            f"{dropped} requested hour(s) lacked the history a {champion.horizon}h horizon needs"
            if dropped
            else None
        ),
    )
    logger.info("\n{}", summary.format())
    state.record_success(PIPELINE, now)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forecast the reachable horizon from the champion and log every prediction."
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    load_project_env()
    run(load_config(args.config))
    # An empty registry or a stale dataset is a condition the next run resolves, not a
    # failure worth stopping the loop for.
    return 0


if __name__ == "__main__":
    sys.exit(main())
