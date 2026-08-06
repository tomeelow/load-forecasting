"""Loading the production model from the MLflow registry, by alias.

Never from a file path. The alias is the whole point of having a registry: rolling back
is moving a pointer, and a service that loads `model_final_v3.pkl` from disk cannot be
rolled back without a deploy. `/reload-model` re-resolves the same alias, so promoting a
new champion reaches the running service without a restart.

The P10/P50/P90 companions are logged inside the champion's own run, so they are loaded
by run id rather than registered separately — the band and the point forecast can never
be a version apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient

from src.config import MlflowConfig

QUANTILE_MODELS = {"p10": "model_p10", "p50": "model_p50", "p90": "model_p90"}


@dataclass(frozen=True)
class Champion:
    """The model currently serving, with everything needed to log what it did."""

    name: str
    version: str
    run_id: str
    horizon: int
    feature_columns: list[str]
    dataset_version: str | None
    loaded_at: datetime
    point: object  # mlflow.pyfunc.PyFuncModel
    quantiles: dict[str, object] = field(default_factory=dict)

    @property
    def uri(self) -> str:
        return f"{self.name}/{self.version}"

    @property
    def has_intervals(self) -> bool:
        return bool(self.quantiles)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Point forecast, plus the band when the champion carries one.

        Columns are selected in the model's own order: a frame with the right names in
        the wrong order would otherwise be silently mis-read.
        """
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns the model needs: {missing}")
        ordered = X[self.feature_columns]

        out = pd.DataFrame(index=X.index)
        out["load_mw"] = self.point.predict(ordered)
        for label, model in self.quantiles.items():
            out[label] = model.predict(ordered)
        return out


def load_champion(cfg: MlflowConfig) -> Champion:
    """Resolve the champion alias and load it, with its quantile companions."""
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = MlflowClient()

    version = client.get_model_version_by_alias(cfg.registered_model_name, cfg.champion_alias)
    uri = f"models:/{cfg.registered_model_name}@{cfg.champion_alias}"
    point = mlflow.pyfunc.load_model(uri)

    run = client.get_run(version.run_id)
    horizon = int(run.data.params.get("horizon_hours", 0))
    if horizon < 1:
        raise RuntimeError(
            f"Champion {cfg.registered_model_name} v{version.version} has no usable "
            "'horizon_hours' param; the service cannot know what it forecasts"
        )

    schema = point.metadata.get_input_schema()
    if schema is None:
        raise RuntimeError(f"Champion v{version.version} was logged without a signature")

    quantiles = {}
    for label, artifact in QUANTILE_MODELS.items():
        try:
            quantiles[label] = mlflow.pyfunc.load_model(f"runs:/{version.run_id}/{artifact}")
        except Exception:  # noqa: BLE001 — a missing companion is expected, not exceptional
            logger.debug("Champion run has no '{}' companion", artifact)

    champion = Champion(
        name=cfg.registered_model_name,
        version=str(version.version),
        run_id=version.run_id,
        horizon=horizon,
        feature_columns=list(schema.input_names()),
        dataset_version=run.data.params.get("dataset_version"),
        loaded_at=datetime.now(UTC),
        point=point,
        quantiles=quantiles,
    )
    logger.info(
        "Loaded champion {} (run {}), horizon {}h, {} features, intervals: {}",
        champion.uri,
        champion.run_id[:8],
        champion.horizon,
        len(champion.feature_columns),
        "yes" if champion.has_intervals else "no",
    )
    return champion
