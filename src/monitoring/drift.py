"""Drift and performance monitoring, wired to the retraining trigger.

Three questions, in increasing order of how much they should worry you:

1. **Data drift** — do the model's inputs still look like what it was trained on?
2. **Prediction drift** — has the shape of its output moved?
3. **Performance** — is it actually getting worse against reality?

Only the third is proof. Input drift is an early warning that may or may not matter, and
on a seasonal series it is the one most likely to be a false alarm (see `reference.py`).
Rolling error of served predictions has no seasonal confound and is therefore weighted
as the decisive signal — but it lags, because the actual for an hour arrives after that
hour, and it is silent until enough predictions have been served and scored.

**Bootstrapping.** For the first days of operation there are no served predictions at
all. Every check here reports `insufficient_data` rather than computing a MAPE over
three hours and pretending it means something.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import Config
from src.evaluation.metrics import mape
from src.features.builder import (
    LAG_COLUMN_PATTERN,
    ROLLING_COLUMN_PATTERN,
    WEATHER_FEATURE_COLUMNS,
    feature_columns,
    make_features,
)
from src.monitoring.reference import ReferenceWindow, choose_reference
from src.prediction_log import PredictionLog

INSUFFICIENT = "insufficient_data"
DRIFT = "drift"
OK = "ok"

_COLUMN_IN_METRIC = re.compile(r"ValueDrift\(column=(?P<column>[^,]+),")


def monitored_columns(columns: list[str]) -> list[str]:
    """The inputs that can meaningfully drift — weather and load history, not the clock.

    Calendar features are deterministic functions of the timestamp. `hour` cannot drift:
    testing it compares whether two windows happen to contain a whole number of days,
    and a 14-day window ending mid-afternoon fails that test every time. `month` is
    worse — the seasonal reference pads each historical window by a week, so the month
    mix differs by construction and drift is reported with certainty.

    Including them does not make monitoring more sensitive, it makes it wrong: nine
    guaranteed alarms drown the share-based trigger and hide the features that did move.
    """
    return [
        c
        for c in columns
        if LAG_COLUMN_PATTERN.match(c)
        or ROLLING_COLUMN_PATTERN.match(c)
        or c in WEATHER_FEATURE_COLUMNS
    ]


@dataclass(frozen=True)
class DriftResult:
    """What monitoring found, and whether it is enough to justify a retrain."""

    status: str
    reasons: list[str] = field(default_factory=list)
    data_drift: bool = False
    drifted_features: list[str] = field(default_factory=list)
    drift_share: float | None = None
    prediction_drift: bool = False
    rolling_mape: float | None = None
    tso_mape: float | None = None
    scored_predictions: int = 0
    reference_strategy: str | None = None
    reference_detail: str | None = None
    reference_rows: int = 0
    current_rows: int = 0
    report_path: Path | None = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def should_retrain(self) -> bool:
        return self.status == DRIFT

    def summary(self) -> str:
        lines = [
            f"Drift check {self.checked_at:%Y-%m-%d %H:%M} UTC — {self.status.upper()}",
            f"  reference   {self.reference_strategy} ({self.reference_rows} rows)"
            f" | current {self.current_rows} rows",
        ]
        if self.drift_share is not None:
            lines.append(
                f"  input drift {len(self.drifted_features)} feature(s), "
                f"share {self.drift_share:.2f}"
            )
            if self.drifted_features:
                lines.append(f"              {', '.join(self.drifted_features)}")
        lines.append(
            f"  served      {self.scored_predictions} scored prediction(s)"
            + (f", rolling MAPE {self.rolling_mape:.3f}%" if self.rolling_mape is not None else "")
            + (f" vs PSE {self.tso_mape:.3f}%" if self.tso_mape is not None else "")
        )
        for reason in self.reasons:
            lines.append(f"  -> {reason}")
        return "\n".join(lines)


def _drift_from_snapshot(payload: dict, threshold: float) -> tuple[list[str], float | None]:
    """Pull the drifted column names and the drifted share out of an Evidently result."""
    drifted: list[str] = []
    share: float | None = None
    for metric in payload.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            share = float(value.get("share", 0.0))
        match = _COLUMN_IN_METRIC.match(name)
        if match and isinstance(value, (int, float)):
            column_threshold = float(metric.get("config", {}).get("threshold", 0.05))
            if float(value) < column_threshold:
                drifted.append(match.group("column"))
    return sorted(drifted), share


def _evidently_report(
    current: pd.DataFrame, reference: pd.DataFrame, report_path: Path | None
) -> dict:
    """Run Evidently's drift preset and optionally save the shareable HTML."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=list(current.columns))
    snapshot = Report(metrics=[DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_path))
        logger.info("Wrote drift report to {}", report_path)
    return snapshot.dict()


def check_drift(
    cfg: Config,
    dataset: pd.DataFrame,
    log: PredictionLog,
    *,
    now: pd.Timestamp | None = None,
    write_report: bool = True,
) -> DriftResult:
    """Run all three checks and decide whether a retrain is owed."""
    settings = cfg.monitoring
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    current_start = now - pd.Timedelta(days=settings.current_days)

    horizon = cfg.model.horizons[0]
    features = make_features(
        dataset,
        horizon,
        rolling_window=cfg.features.rolling_window_hours,
        weekly_lag=cfg.features.weekly_lag_hours,
    )
    columns = monitored_columns(feature_columns(list(features.columns)))
    inputs = features[columns]

    current_index = inputs.index[(inputs.index >= current_start) & (inputs.index <= now)]
    if len(current_index) < 24:
        return DriftResult(
            status=INSUFFICIENT,
            reasons=[
                f"only {len(current_index)} feature row(s) in the last "
                f"{settings.current_days} days; ingest has not caught up"
            ],
            current_rows=len(current_index),
        )

    reference: ReferenceWindow = choose_reference(
        inputs.index,
        current_start,
        now,
        strategy=settings.reference.strategy,
        years_back=settings.reference.years_back,
        pad_days=settings.reference.pad_days,
        min_rows=settings.reference.min_rows,
        fallback_to_trailing=settings.reference.fallback_to_trailing,
        current_days=settings.current_days,
    )
    if reference.rows < 24:
        return DriftResult(
            status=INSUFFICIENT,
            reasons=[f"reference window has only {reference.rows} row(s)"],
            reference_strategy=reference.strategy,
            reference_detail=reference.detail,
            reference_rows=reference.rows,
            current_rows=len(current_index),
        )

    report_path = None
    if write_report:
        report_path = settings.reports_dir / f"drift_{now:%Y%m%dT%H%M%S}.html"

    payload = _evidently_report(inputs.loc[current_index], inputs.loc[reference.index], report_path)
    drifted, share = _drift_from_snapshot(payload, settings.drift_share_threshold)
    data_drift = share is not None and share >= settings.drift_share_threshold

    performance = _performance(cfg, log, current_start, now)
    reasons: list[str] = []
    if data_drift:
        reasons.append(
            f"{len(drifted)} of {len(columns)} input features drifted "
            f"(share {share:.2f} >= {settings.drift_share_threshold}) "
            f"against the {reference.strategy} reference"
        )
    if performance["breached"]:
        reasons.append(
            f"rolling MAPE {performance['mape']:.3f}% over "
            f"{settings.rolling_mape_threshold}% across {performance['n']} scored hour(s)"
        )
    if performance["insufficient"]:
        reasons.append(
            f"performance not assessed: {performance['n']} scored prediction(s), "
            f"need {settings.min_scored_predictions}"
        )

    status = DRIFT if (data_drift or performance["breached"]) else OK
    result = DriftResult(
        status=status,
        reasons=reasons or ["nothing exceeded its threshold"],
        data_drift=data_drift,
        drifted_features=drifted,
        drift_share=share,
        prediction_drift=_prediction_drift(log, current_start, now, settings.current_days),
        rolling_mape=performance["mape"],
        tso_mape=performance["tso_mape"],
        scored_predictions=performance["n"],
        reference_strategy=reference.strategy,
        reference_detail=reference.detail,
        reference_rows=reference.rows,
        current_rows=len(current_index),
        report_path=report_path,
    )
    logger.log("WARNING" if result.should_retrain else "INFO", "\n{}", result.summary())
    return result


def _performance(cfg: Config, log: PredictionLog, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Rolling error of *served* predictions — the signal without a seasonal confound."""
    scored = log.scored(since=start)
    scored = scored[scored.index <= end] if not scored.empty else scored
    count = len(scored)
    if count < cfg.monitoring.min_scored_predictions:
        return {"n": count, "mape": None, "tso_mape": None, "breached": False, "insufficient": True}

    rolling = float(scored["ape"].mean() * 100)
    tso = None
    with_tso = scored.dropna(subset=["tso_mw"])
    if not with_tso.empty:
        tso = mape(with_tso["actual_mw"], with_tso["tso_mw"])
    return {
        "n": count,
        "mape": rolling,
        "tso_mape": tso,
        "breached": rolling > cfg.monitoring.rolling_mape_threshold,
        "insufficient": False,
    }


def _prediction_drift(
    log: PredictionLog, start: pd.Timestamp, end: pd.Timestamp, window_days: int
) -> bool:
    """Has the distribution of served predictions moved against the window before it?

    Compared against previously *served* predictions rather than against training-time
    outputs, because that is the only like-for-like comparison available: both windows
    are the model's own output on real requests. Silent until two windows exist.
    """
    current = log.latest_per_target(since=start, until=end)
    previous = log.latest_per_target(since=start - pd.Timedelta(days=window_days), until=start)
    if len(current) < 24 or len(previous) < 24:
        return False

    payload = _evidently_report(
        current[["load_mw"]].astype("float64"), previous[["load_mw"]].astype("float64"), None
    )
    drifted, _ = _drift_from_snapshot(payload, 0.5)
    return "load_mw" in drifted


def latest_report(reports_dir: Path) -> Path | None:
    """The most recent drift report, for the dashboard to pick up without guessing."""
    reports = sorted(reports_dir.glob("drift_*.html"))
    return reports[-1] if reports else None
