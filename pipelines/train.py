"""Training entrypoint: baselines, model variants, MLflow, and the promotion gate.

    uv run python -m pipelines.train                 # every horizon in config
    uv run python -m pipelines.train --horizon 24    # just one
    uv run python -m pipelines.train --no-tune       # skip Optuna while iterating

Three LightGBM variants are trained per horizon so the benchmark table can say what the
weather and the tuning are each worth: calendar+lags only, plus weather, and plus weather
tuned. The tuned variant is the candidate that faces the promotion gate.

Every number here comes from a single chronological holdout — enough to gate on, not
enough to publish. `pipelines/backtest.py` is what produces the reported figures.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient

from src.config import Config, load_config, load_project_env
from src.evaluation.metrics import pinball_loss, point_metrics
from src.evaluation.splits import chronological_split
from src.features.builder import TARGET_COLUMN, feature_columns, feature_set_id, make_features
from src.ingestion.dataset import load_training_frame
from src.models import lgbm
from src.models.baselines import LinearBaseline, naive_seasonal, tso_forecast
from src.models.promotion import promotion_decision
from src.models.tracking import (
    RunSpec,
    apply_gate,
    champion_metric,
    configure,
    latest_version_for_run,
    log_run,
)

CALENDAR_ONLY = "lgbm_calendar"
WITH_WEATHER = "lgbm_weather"
TUNED = "lgbm_weather_tuned"


@dataclass(frozen=True)
class TrainingResult:
    horizon: int
    metrics: pd.DataFrame
    champion_run_id: str | None
    promoted: bool
    gate_reason: str


def _baseline_metrics(frame: pd.DataFrame, y_val: pd.Series) -> dict[str, dict]:
    """Score the references on the same validation block the model is scored on."""
    return {
        "naive_seasonal": point_metrics(y_val, naive_seasonal(frame).reindex(y_val.index)),
        "pse_day_ahead": point_metrics(y_val, tso_forecast(frame).reindex(y_val.index)),
    }


def train_horizon(cfg: Config, horizon: int, *, tune: bool = True) -> TrainingResult:
    """Train every variant for one horizon, log them all, and gate the best candidate."""
    data = load_training_frame(cfg)
    features = make_features(
        data.frame,
        horizon,
        rolling_window=cfg.features.rolling_window_hours,
        weekly_lag=cfg.features.weekly_lag_hours,
    )
    train_index, val_index = chronological_split(
        features.index, validation_days=cfg.model.validation_days, horizon=horizon
    )
    all_columns = feature_columns(list(features.columns))
    X = features[all_columns]
    y = features[TARGET_COLUMN]
    X_train, y_train = X.loc[train_index], y.loc[train_index]
    X_val, y_val = X.loc[val_index], y.loc[val_index]

    logger.info(
        "h={}: {} training rows to {}, {} validation rows from {}",
        horizon,
        len(X_train),
        train_index.max(),
        len(X_val),
        val_index.min(),
    )

    scores = _baseline_metrics(data.frame, y_val)
    linear = LinearBaseline().fit(X_train, y_train)
    scores["linear_calendar_weather"] = point_metrics(y_val, linear.predict(X_val))

    naive_mape = scores["naive_seasonal"]["mape"]
    tso_mape = scores["pse_day_ahead"]["mape"]

    fit_kwargs = {
        "num_boost_round": cfg.model.num_boost_round,
        "early_stopping_rounds": cfg.model.early_stopping_rounds,
        "seed": cfg.model.seed,
    }
    variants = {
        CALENDAR_ONLY: (feature_columns(all_columns, include_weather=False), None),
        WITH_WEATHER: (all_columns, None),
    }
    if tune:
        best_params = lgbm.tune(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=cfg.model.tuning.n_trials,
            seed=cfg.model.seed,
            timeout_s=cfg.model.tuning.timeout_s,
            num_boost_round=cfg.model.num_boost_round,
            early_stopping_rounds=cfg.model.early_stopping_rounds,
        )
        variants[TUNED] = (all_columns, best_params)

    candidate_name = TUNED if tune else WITH_WEATHER
    candidate_run_id: str | None = None
    candidate_mape = float("inf")

    for name, (columns, params) in variants.items():
        model = lgbm.train(
            X_train[columns], y_train, X_val[columns], y_val, params=params, **fit_kwargs
        )
        predictions = model.predict(X_val[columns])
        metrics = point_metrics(y_val, predictions)
        metrics["mape_vs_tso"] = metrics["mape"] - tso_mape  # negative means we beat PSE
        metrics["mape_vs_naive"] = metrics["mape"] - naive_mape
        scores[name] = metrics

        is_candidate = name == candidate_name
        companions = {}
        if is_candidate and cfg.model.quantiles:
            quantile_models = lgbm.train_quantiles(
                X_train[columns],
                y_train,
                X_val[columns],
                y_val,
                tuple(cfg.model.quantiles),
                params=params,
                **fit_kwargs,
            )
            for quantile, quantile_model in quantile_models.items():
                label = f"p{int(quantile * 100)}"
                companions[f"model_{label}"] = quantile_model
                metrics[f"pinball_{label}"] = pinball_loss(
                    y_val, quantile_model.predict(X_val[columns]), quantile
                )

        run_id = log_run(
            RunSpec(
                name=f"{name}_h{horizon}",
                horizon=horizon,
                dataset_version=data.version,
                feature_set=feature_set_id(columns),
                synthetic=data.synthetic,
                params={**lgbm.DEFAULT_PARAMS, **(params or {})},
                metrics={**metrics, "naive_mape": naive_mape, "tso_mape": tso_mape},
                tags={"variant": name, "candidate": str(is_candidate).lower()},
            ),
            model,
            X_val[columns],
            registered_model_name=cfg.mlflow.registered_model_name if is_candidate else None,
            extra_models=companions,
        )
        if is_candidate:
            candidate_run_id, candidate_mape = run_id, metrics["mape"]

    client = MlflowClient()
    champion = champion_metric(
        client,
        cfg.mlflow.registered_model_name,
        cfg.mlflow.champion_alias,
        cfg.promotion.metric,
        data_source="synthetic" if data.synthetic else "ingested",
    )
    decision = promotion_decision(
        candidate_mape, naive_mape, champion, max_regression=cfg.promotion.max_regression
    )
    logger.info("Gate: {}", decision)

    promoted = False
    version = latest_version_for_run(client, cfg.mlflow.registered_model_name, candidate_run_id)
    if version is not None:
        promoted = apply_gate(
            client, cfg.mlflow.registered_model_name, version, cfg.mlflow.champion_alias, decision
        )

    table = pd.DataFrame(scores).T.sort_values("mape")
    logger.info("\nValidation metrics (h={}):\n{}", horizon, table.round(3).to_string())
    return TrainingResult(horizon, table, candidate_run_id, promoted, decision.reason)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train baselines and LightGBM variants, log to MLflow, gate promotion."
    )
    parser.add_argument("--horizon", type=int, action="append", help="override config horizons")
    parser.add_argument("--no-tune", action="store_true", help="skip the Optuna search")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    load_project_env()
    cfg = load_config(args.config)
    configure(cfg.mlflow)

    for horizon in args.horizon or cfg.model.horizons:
        result = train_horizon(cfg, horizon, tune=not args.no_tune)
        logger.info(
            "h={} finished: {} — {}",
            horizon,
            "promoted" if result.promoted else "not promoted",
            result.gate_reason,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
