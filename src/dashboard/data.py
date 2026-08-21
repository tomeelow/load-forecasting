"""Everything the dashboard reads, and nothing it displays.

Kept apart from the Streamlit app for two reasons. It is testable without a browser or a
Streamlit runtime, which is what lets the empty-data paths below be checked rather than
hoped for. And it makes a rule enforceable that a dashboard breaks quietly without: every
number on the page comes from a real artifact — the MLflow registry, the prediction log,
the drift history, the ingested dataset, the backtest reports — and none is written down
here. A figure typed into a chart is a figure nobody can trace to a run.

**Missing data is a normal state, not an error.** A fresh deployment has an empty
registry, an empty prediction log and no drift history at all, and the dashboard has to
say so rather than crash or invent. Every loader here returns a typed-empty result and
lets the page decide what to say about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import Config
from src.evaluation.metrics import mape, metrics_by_segment, point_metrics
from src.ingestion.dataset import read_dataset
from src.models.baselines import TSO_FORECAST
from src.prediction_log import PredictionLog

ACTUAL_COLUMN = "load_mw"
TSO_COLUMN = "tso_forecast_mw"

BACKTEST_MODEL = "lgbm_weather_tuned"
AUDIT_MODEL = "lgbm_gate_closure"


@dataclass(frozen=True)
class ModelCard:
    """Who is serving, since when, and on what data."""

    name: str | None = None
    version: str | None = None
    run_id: str | None = None
    horizon: int | None = None
    dataset_version: str | None = None
    promoted_at: datetime | None = None
    synthetic: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.version is not None

    @property
    def data_source(self) -> str:
        if self.synthetic is None:
            return "unknown"
        return "SYNTHETIC" if self.synthetic else "real ENTSO-E"


@dataclass(frozen=True)
class ServedPerformance:
    """Production error from the prediction log, and whether there is enough of it yet."""

    scored: pd.DataFrame
    required: int
    logged: int = 0

    @property
    def sufficient(self) -> bool:
        return len(self.scored) >= self.required

    @property
    def model_mape(self) -> float | None:
        if self.scored.empty:
            return None
        return float(self.scored["ape"].mean() * 100)

    @property
    def tso_mape(self) -> float | None:
        with_tso = self.scored.dropna(subset=["tso_mw"]) if not self.scored.empty else self.scored
        if with_tso.empty:
            return None
        return mape(with_tso["actual_mw"], with_tso["tso_mw"])

    @property
    def status(self) -> str:
        """What the served panel is allowed to claim right now."""
        if self.scored.empty:
            return "empty"
        return "measuring" if self.sufficient else "accumulating"


@dataclass(frozen=True)
class BacktestEvidence:
    """Out-of-sample predictions from a backtest, with the references on the same rows."""

    predictions: pd.DataFrame
    model_column: str
    source: Path | None = None
    label: str = ""

    @property
    def available(self) -> bool:
        return not self.predictions.empty and self.model_column in self.predictions.columns

    def overall(self) -> pd.DataFrame:
        actual = self.predictions["actual"]
        rows = {
            column: point_metrics(actual, self.predictions[column])
            for column in self.predictions.columns
            if column != "actual"
        }
        table = pd.DataFrame(rows).T
        if TSO_FORECAST in table.index:
            table["mape_vs_tso"] = table["mape"] - table.loc[TSO_FORECAST, "mape"]
        return table

    def rolling_mape(self, window_days: int = 30) -> pd.DataFrame:
        """Rolling error of the model and of PSE, on the same hours and the same axis."""
        actual = self.predictions["actual"]
        window = window_days * 24
        frame = pd.DataFrame(
            {
                "model": (self.predictions[self.model_column] - actual).abs() / actual * 100,
                "PSE": (self.predictions[TSO_FORECAST] - actual).abs() / actual * 100,
            }
        )
        return frame.rolling(window, min_periods=window // 2).mean().dropna()

    def by_segment(self, peak_hours: tuple[int, int]) -> pd.DataFrame:
        segments = metrics_by_segment(
            self.predictions["actual"],
            self.predictions[[self.model_column, TSO_FORECAST]],
            peak_hours=peak_hours,
        )
        wide = segments.pivot_table(
            index=["segment_kind", "segment", "n"], columns="model", values="mape"
        ).reset_index()
        wide["gap"] = wide[self.model_column] - wide[TSO_FORECAST]
        return wide.sort_values("gap", ascending=False).reset_index(drop=True)


def load_model_card(cfg: Config) -> ModelCard:
    """Resolve the champion through the registry and read everything logged about it.

    Imports MLflow lazily and swallows its failures into `error`: a dashboard that cannot
    reach the registry should say which model it cannot find, not fail to render.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = MlflowClient()
        version = client.get_model_version_by_alias(
            cfg.mlflow.registered_model_name, cfg.mlflow.champion_alias
        )
        run = client.get_run(version.run_id)
    except Exception as exc:  # noqa: BLE001 — reported on the page, never raised
        logger.warning("Dashboard could not resolve the champion: {}", exc)
        return ModelCard(error=f"{type(exc).__name__}: {exc}")

    synthetic = run.data.tags.get("synthetic")
    return ModelCard(
        name=cfg.mlflow.registered_model_name,
        version=str(version.version),
        run_id=version.run_id,
        horizon=int(run.data.params.get("horizon_hours", 0)) or None,
        dataset_version=run.data.params.get("dataset_version"),
        # The alias moved when the gate passed, so the version's own last update is when
        # this model started serving — closer to the truth than the run's start time.
        promoted_at=pd.Timestamp(version.last_updated_timestamp, unit="ms", tz="UTC"),
        synthetic=None if synthetic is None else synthetic == "true",
        metrics=dict(run.data.metrics),
        params=dict(run.data.params),
        importance=_load_importance(run.info.artifact_uri),
    )


def _load_importance(artifact_uri: str) -> pd.DataFrame:
    """Feature importance as logged by the training run, or an empty frame."""
    path = Path(artifact_uri.removeprefix("file://")) / "feature_importance.json"
    if not path.exists():
        logger.debug("No feature importance artifact at {}", path)
        return pd.DataFrame(columns=["feature", "gain", "split"])
    payload = json.loads(path.read_text())
    return pd.DataFrame(payload["data"], columns=payload["columns"])


def load_recent_actuals(cfg: Config, days: int = 14) -> pd.DataFrame:
    """Actual load and PSE's forecast over the last `days`, from the ingested dataset."""
    try:
        dataset = read_dataset(cfg.data.dataset_path)
    except FileNotFoundError:
        logger.warning("No dataset at {}", cfg.data.dataset_path)
        return pd.DataFrame(columns=[ACTUAL_COLUMN, TSO_COLUMN])

    latest = dataset[ACTUAL_COLUMN].last_valid_index()
    if latest is None:
        return dataset.iloc[:0][[ACTUAL_COLUMN, TSO_COLUMN]]
    columns = [c for c in (ACTUAL_COLUMN, TSO_COLUMN) if c in dataset.columns]
    return dataset.loc[latest - pd.Timedelta(days=days) :, columns]


def load_served_forecast(cfg: Config, days: int = 3) -> pd.DataFrame:
    """The most recent forecast standing for each target hour, with its band.

    `latest_per_target` rather than every row: a chart of what the service currently
    says should show one line, not one per time anybody asked.
    """
    log = PredictionLog(cfg.state.prediction_log_path)
    latest = log.read()
    if latest.empty:
        return latest
    since = latest.index.max() - pd.Timedelta(days=days)
    return log.latest_per_target(since=since)


def load_served_performance(cfg: Config, days: int | None = None) -> ServedPerformance:
    """Scored served predictions, and the count still needed before they mean anything."""
    log = PredictionLog(cfg.state.prediction_log_path)
    since = None if days is None else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return ServedPerformance(
        scored=log.scored(since=since),
        required=cfg.monitoring.min_scored_predictions,
        logged=log.count(),
    )


def load_drift_history(cfg: Config) -> pd.DataFrame:
    """Every drift check this deployment has run, oldest first."""
    from src.monitoring.history import DriftHistory

    return DriftHistory(cfg.state.drift_history_path).read()


def latest_drift_report(cfg: Config) -> Path | None:
    """The newest Evidently HTML report on disk, if any survived the run that made it."""
    reports = sorted(cfg.monitoring.reports_dir.glob("drift_*.html"))
    return reports[-1] if reports else None


def load_backtest(cfg: Config, horizon: int = 24, *, prefer_audit: bool = True) -> BacktestEvidence:
    """Out-of-sample predictions to plot the model against PSE with.

    Prefers the audited, gate-closure-aligned run over the flat-horizon one, because the
    flat-horizon figure compares two different products (see `docs/evaluation_notes.md`)
    and a dashboard should not show the flattering number just because it was computed
    first.
    """
    reports = cfg.backtest.reports_dir
    candidates: list[tuple[Path, str, str]] = []
    if prefer_audit:
        candidates.append(
            (reports / f"audit_c_h{horizon}.csv", AUDIT_MODEL, "gate-closure-aligned backtest")
        )
    candidates.append(
        (reports / f"predictions_h{horizon}.csv", BACKTEST_MODEL, "rolling-origin backtest")
    )

    for path, model_column, label in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if model_column not in frame.columns:
            continue
        frame.index = _as_utc(frame.index)
        return BacktestEvidence(frame.sort_index(), model_column, path, label)

    logger.warning("No backtest predictions found in {}", reports)
    return BacktestEvidence(pd.DataFrame(), BACKTEST_MODEL)


def _as_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def audit_summary(cfg: Config, horizon: int = 24) -> str | None:
    """The audit report as markdown, so the page can show it rather than restate it."""
    path = cfg.backtest.reports_dir / f"audit_h{horizon}.md"
    return path.read_text() if path.exists() else None
