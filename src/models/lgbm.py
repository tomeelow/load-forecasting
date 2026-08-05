"""LightGBM: the production model, trained directly for one horizon.

Direct per-horizon, never recursive (ADR-002): the model is fitted on rows whose lag
set was derived from the horizon it will serve, so what it is scored on and what it is
asked to do are the same thing.

Every fit here validates on a period strictly *later* than it trains on. That is not a
detail — early stopping on a random split lets the model interpolate between hours it
has already seen and picks an iteration count that will not hold up in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from loguru import logger

matplotlib.use("Agg")  # pipelines and CI have no display
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "mape",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "min_child_samples": 40,
    "verbosity": -1,
}


@dataclass(frozen=True)
class TrainedModel:
    """A fitted booster plus the exact columns it expects, in order.

    Carrying the column list with the model is what stops a serving path from handing
    over a frame with the right names in the wrong order, or with an extra column.
    """

    booster: lgb.Booster
    columns: list[str]
    params: dict[str, object] = field(default_factory=dict)
    best_iteration: int = 0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns the model needs: {missing}")
        values = self.booster.predict(X[self.columns], num_iteration=self.best_iteration or None)
        return pd.Series(np.asarray(values), index=X.index, name="prediction")


def _seeded(params: dict[str, object], seed: int) -> dict[str, object]:
    """Pin every stochastic knob so a rerun reproduces the run exactly."""
    return {
        **params,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "num_threads": 0,
    }


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    params: dict[str, object] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
) -> TrainedModel:
    """Fit one booster, stopping early on a chronologically later validation block."""
    columns = list(X_train.columns)
    resolved = _seeded({**DEFAULT_PARAMS, **(params or {})}, seed)

    train_set = lgb.Dataset(X_train[columns], y_train)
    val_set = lgb.Dataset(X_val[columns], y_val, reference=train_set)
    booster = lgb.train(
        resolved,
        train_set,
        valid_sets=[val_set],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    logger.debug(
        "LightGBM fitted: {} rows x {} features, best iteration {}/{}",
        len(X_train),
        len(columns),
        booster.best_iteration,
        num_boost_round,
    )
    return TrainedModel(booster, columns, resolved, booster.best_iteration)


def train_quantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: tuple[float, ...],
    *,
    params: dict[str, object] | None = None,
    **kwargs,
) -> dict[float, TrainedModel]:
    """One booster per quantile, giving the P10/P50/P90 band a forecast is worth more with.

    A point estimate tells an operator what to schedule; a band tells them how much
    reserve to hold, which is the question they are actually asking.
    """
    base = params or {}
    models = {}
    for quantile in quantiles:
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        models[quantile] = train(
            X_train,
            y_train,
            X_val,
            y_val,
            params={**base, "objective": "quantile", "alpha": quantile, "metric": "quantile"},
            **kwargs,
        )
    return models


def tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_trials: int = 30,
    seed: int = 42,
    timeout_s: int | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
) -> dict[str, object]:
    """Search hyperparameters against the chronological validation block.

    The objective is validation MAPE, the same metric the run is reported on. The
    validation period is later than the training period, so a set of parameters that
    only works by memorising the training window scores badly here.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        model = train(
            X_train,
            y_train,
            X_val,
            y_val,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
        )
        predictions = model.predict(X_val)
        return float(np.mean(np.abs((y_val - predictions) / y_val)) * 100)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)

    logger.info(
        "Optuna: {} trials, best validation MAPE {:.3f}% with {}",
        len(study.trials),
        study.best_value,
        study.best_params,
    )
    return study.best_params


def feature_importance(model: TrainedModel) -> pd.DataFrame:
    """Gain-based importance, highest first."""
    return (
        pd.DataFrame(
            {
                "feature": model.booster.feature_name(),
                "gain": model.booster.feature_importance("gain"),
                "split": model.booster.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )


def feature_importance_figure(model: TrainedModel, top_n: int = 20) -> plt.Figure:
    """A horizontal bar chart of gain importance, for logging as a run artifact."""
    importance = feature_importance(model).head(top_n).iloc[::-1]
    figure, axes = plt.subplots(figsize=(8, max(3.0, 0.32 * len(importance))))
    axes.barh(importance["feature"], importance["gain"], color="#4c72b0")
    axes.set_xlabel("gain")
    axes.set_title("Feature importance")
    figure.tight_layout()
    return figure
