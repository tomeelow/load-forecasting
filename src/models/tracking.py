"""MLflow: every run recorded, and the registry that decides what production means.

An untracked run does not exist. The point is not the metric — it is being able to
answer, months later, which data, which features and which parameters produced the model
currently serving, and to roll back by moving an alias rather than rebuilding anything.

The local setup uses a SQLite backend because the model registry and its aliases need a
database; a plain file store cannot hold them. Phase 10 swaps it for the containerised
Postgres server without touching this module's callers.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.config import MlflowConfig
from src.models.lgbm import TrainedModel, feature_importance, feature_importance_figure
from src.models.promotion import GateDecision


@dataclass(frozen=True)
class RunSpec:
    """Everything a training run must record to be reproducible later."""

    name: str
    horizon: int
    dataset_version: str
    feature_set: str
    synthetic: bool
    params: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def resolved_name(self) -> str:
        """Synthetic runs say so in the run name, not only in a tag nobody filters on."""
        return f"{self.name}_synthetic" if self.synthetic else self.name


def configure(cfg: MlflowConfig) -> None:
    """Point MLflow at the configured backend and experiment."""
    if cfg.tracking_uri.startswith("sqlite:///"):
        Path(cfg.tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    logger.debug("MLflow at {}, experiment '{}'", cfg.tracking_uri, cfg.experiment)


def log_run(
    spec: RunSpec,
    model: TrainedModel,
    X_sample: pd.DataFrame,
    *,
    registered_model_name: str | None = None,
    extra_artifacts: dict[str, pd.DataFrame] | None = None,
    extra_models: dict[str, TrainedModel] | None = None,
) -> str:
    """Log one training run and return its run id.

    The model goes in with an inferred signature so the serving layer cannot quietly
    hand it a frame with the wrong columns or dtypes.
    """
    with mlflow.start_run(run_name=spec.resolved_name()) as run:
        mlflow.log_params({k: v for k, v in spec.params.items() if v is not None})
        mlflow.log_param("horizon_hours", spec.horizon)
        mlflow.log_param("dataset_version", spec.dataset_version)
        mlflow.log_param("feature_set", spec.feature_set)
        mlflow.log_param("n_features", len(model.columns))
        mlflow.log_param("best_iteration", model.best_iteration)
        mlflow.log_metrics(spec.metrics)

        mlflow.set_tags(
            {
                "horizon": str(spec.horizon),
                "data_source": "synthetic" if spec.synthetic else "ingested",
                "synthetic": str(spec.synthetic).lower(),
                **spec.tags,
            }
        )
        if spec.synthetic:
            mlflow.set_tag(
                "mlflow.note.content",
                "Trained on SYNTHETIC data — no ENTSO-E token was available. These "
                "metrics measure the pipeline end to end, not Polish demand. Re-run "
                "once the token arrives before quoting any number from this run.",
            )

        figure = feature_importance_figure(model)
        mlflow.log_figure(figure, "feature_importance.png")
        figure.clf()
        mlflow.log_table(feature_importance(model), "feature_importance.json")
        for name, frame in (extra_artifacts or {}).items():
            mlflow.log_table(frame, f"{name}.json")

        predictions = model.predict(X_sample)
        signature = infer_signature(X_sample[model.columns], predictions)
        mlflow.lightgbm.log_model(
            model.booster,
            name="model",
            signature=signature,
            registered_model_name=registered_model_name,
        )
        # Quantile boosters ride along unregistered: the champion is the point model,
        # and the band is served from the same run so the two cannot drift apart.
        for name, companion in (extra_models or {}).items():
            mlflow.lightgbm.log_model(companion.booster, name=name, signature=signature)

        logger.info("Logged MLflow run '{}' ({})", spec.resolved_name(), run.info.run_id)
        return run.info.run_id


def latest_version_for_run(client: MlflowClient, model_name: str, run_id: str) -> str | None:
    """The registered version produced by a given run, if the run registered one."""
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        if version.run_id == run_id:
            return version.version
    return None


def champion_metric(client: MlflowClient, model_name: str, alias: str, metric: str) -> float | None:
    """The champion's value for `metric`, or None when nothing is in production yet.

    Read from the run that produced the champion, so the gate compares like with like
    rather than against a number typed into a config file.
    """
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except Exception:  # noqa: BLE001 — MLflow raises different types for "not found"
        logger.info("No '{}' alias on '{}' yet; nothing is in production", alias, model_name)
        return None

    run = client.get_run(version.run_id)
    value = run.data.metrics.get(metric)
    if value is None:
        logger.warning("Champion version {} has no '{}' metric logged", version.version, metric)
    return value


def prune_runs(
    client: MlflowClient,
    cfg: MlflowConfig,
    *,
    keep_days: int,
    now: pd.Timestamp | None = None,
) -> list[str]:
    """Soft-delete tracked runs older than `keep_days` that no alias points at.

    The artifact store grows by several megabytes per retrain, and the whole of it is
    carried between scheduled runs (ADR-008). Without retention that mechanism has a
    quiet expiry date — it works for months and then the state push starts timing out.

    Aliased versions are never touched: whatever is serving must remain loadable, and so
    must anything a rollback would reach for. Soft-deleted runs keep their metrics until
    `mlflow gc` reclaims the files, so the history stays readable in the meantime.
    """
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    cutoff_ms = int((now - pd.Timedelta(days=keep_days)).timestamp() * 1000)

    protected = set()
    for version in client.search_model_versions(f"name='{cfg.registered_model_name}'"):
        if version.aliases:
            protected.add(version.run_id)
    # No champion yet is a normal state, not an error worth handling loudly.
    with suppress(Exception):
        protected.add(
            client.get_model_version_by_alias(cfg.registered_model_name, cfg.champion_alias).run_id
        )

    experiment = client.get_experiment_by_name(cfg.experiment)
    if experiment is None:
        return []

    deleted = []
    for run in client.search_runs([experiment.experiment_id], max_results=5000):
        if run.info.run_id in protected or run.info.lifecycle_stage != "active":
            continue
        if run.info.start_time and run.info.start_time < cutoff_ms:
            client.delete_run(run.info.run_id)
            deleted.append(run.info.run_id)

    if deleted:
        logger.info(
            "Marked {} run(s) older than {} days for deletion; `mlflow gc` reclaims the "
            "artifacts. {} aliased run(s) protected.",
            len(deleted),
            keep_days,
            len(protected),
        )
    return deleted


def apply_gate(
    client: MlflowClient,
    model_name: str,
    version: str,
    alias: str,
    decision: GateDecision,
) -> bool:
    """Move the champion alias if — and only if — the gate said so.

    The decision itself is recorded on the model version either way, so a retrain that
    was held back leaves a trace explaining why rather than silently doing nothing.
    """
    client.set_model_version_tag(model_name, version, "gate_decision", str(decision.promote))
    client.set_model_version_tag(model_name, version, "gate_reason", decision.reason)

    if not decision.promote:
        logger.warning("Not promoting {} v{}: {}", model_name, version, decision.reason)
        return False

    client.set_registered_model_alias(model_name, alias, version)
    logger.info("Promoted {} v{} to @{}: {}", model_name, version, alias, decision.reason)
    return True
