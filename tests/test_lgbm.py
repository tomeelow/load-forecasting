"""LightGBM training: reproducibility, column discipline, and beating the naive baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.metrics import mape, pinball_loss
from src.evaluation.splits import chronological_split
from src.features.builder import TARGET_COLUMN, feature_columns, make_features
from src.models.baselines import naive_seasonal
from src.models.lgbm import (
    TrainedModel,
    feature_importance,
    feature_importance_figure,
    train,
    train_quantiles,
    tune,
)

HORIZON = 24


@pytest.fixture(scope="module")
def training_data(request):
    """Features from ~21 months of synthetic data, split chronologically."""
    frame = request.getfixturevalue("dataset")
    features = make_features(frame, HORIZON)
    train_index, val_index = chronological_split(
        features.index, validation_days=45, horizon=HORIZON
    )
    columns = feature_columns(list(features.columns))
    return (
        features.loc[train_index, columns],
        features.loc[train_index, TARGET_COLUMN],
        features.loc[val_index, columns],
        features.loc[val_index, TARGET_COLUMN],
    )


@pytest.fixture(scope="module")
def fitted(training_data):
    X_train, y_train, X_val, y_val = training_data
    return train(X_train, y_train, X_val, y_val, num_boost_round=300, early_stopping_rounds=30)


def test_predictions_come_back_on_the_index_they_were_asked_for(fitted, training_data):
    _, _, X_val, _ = training_data

    predictions = fitted.predict(X_val)

    assert isinstance(predictions, pd.Series)
    assert predictions.index.equals(X_val.index)
    assert predictions.notna().all()


def test_the_model_beats_the_naive_seasonal_baseline(fitted, training_data, dataset):
    _, _, X_val, y_val = training_data

    model_error = mape(y_val, fitted.predict(X_val))
    naive_error = mape(y_val, naive_seasonal(dataset).reindex(y_val.index))

    assert model_error < naive_error, "a model that loses to load[t-168] is broken"


def test_a_missing_feature_column_is_refused_rather_than_guessed(fitted, training_data):
    _, _, X_val, _ = training_data

    with pytest.raises(ValueError, match="missing columns"):
        fitted.predict(X_val.drop(columns=["temp_c"]))


def test_the_model_carries_its_own_column_order(fitted, training_data):
    _, _, X_val, _ = training_data
    reordered = X_val[list(reversed(fitted.columns))]

    # Same values, different column order: the model must reorder rather than mis-read.
    pd.testing.assert_series_equal(fitted.predict(reordered), fitted.predict(X_val))


def test_the_same_seed_reproduces_the_same_model(training_data):
    X_train, y_train, X_val, y_val = training_data
    kwargs = {"num_boost_round": 60, "early_stopping_rounds": 20}

    first = train(X_train, y_train, X_val, y_val, seed=7, **kwargs)
    second = train(X_train, y_train, X_val, y_val, seed=7, **kwargs)

    pd.testing.assert_series_equal(first.predict(X_val), second.predict(X_val))


def test_a_calendar_only_model_never_sees_the_weather(training_data):
    X_train, y_train, X_val, y_val = training_data
    columns = feature_columns(list(X_train.columns), include_weather=False)

    model = train(
        X_train[columns],
        y_train,
        X_val[columns],
        y_val,
        num_boost_round=60,
        early_stopping_rounds=20,
    )

    assert "temp_c" not in model.columns
    assert "temp_sq" not in model.columns
    assert any(c.startswith("load_lag_") for c in model.columns)


def test_early_stopping_picks_an_iteration_inside_the_budget(fitted):
    assert 1 <= fitted.best_iteration <= 300


def test_quantile_models_produce_an_ordered_band(training_data):
    X_train, y_train, X_val, y_val = training_data

    models = train_quantiles(
        X_train,
        y_train,
        X_val,
        y_val,
        (0.1, 0.5, 0.9),
        num_boost_round=120,
        early_stopping_rounds=30,
    )
    band = pd.DataFrame({q: m.predict(X_val) for q, m in models.items()})

    assert set(models) == {0.1, 0.5, 0.9}
    # Quantile crossing is possible in principle; it must be rare, not absent.
    assert (band[0.1] <= band[0.9]).mean() > 0.99
    assert band[0.1].mean() < band[0.5].mean() < band[0.9].mean()


def test_each_quantile_model_is_best_at_its_own_pinball_loss(training_data):
    X_train, y_train, X_val, y_val = training_data

    models = train_quantiles(
        X_train,
        y_train,
        X_val,
        y_val,
        (0.1, 0.9),
        num_boost_round=120,
        early_stopping_rounds=30,
    )
    p10, p90 = models[0.1].predict(X_val), models[0.9].predict(X_val)

    assert pinball_loss(y_val, p10, 0.1) < pinball_loss(y_val, p90, 0.1)
    assert pinball_loss(y_val, p90, 0.9) < pinball_loss(y_val, p10, 0.9)


def test_an_impossible_quantile_is_rejected(training_data):
    X_train, y_train, X_val, y_val = training_data

    with pytest.raises(ValueError, match="quantile must be"):
        train_quantiles(X_train, y_train, X_val, y_val, (1.5,))


def test_tuning_returns_usable_parameters(training_data):
    X_train, y_train, X_val, y_val = training_data

    best = tune(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=3,
        seed=1,
        num_boost_round=60,
        early_stopping_rounds=20,
    )

    assert {"learning_rate", "num_leaves", "feature_fraction"} <= set(best)
    model = train(
        X_train, y_train, X_val, y_val, params=best, num_boost_round=60, early_stopping_rounds=20
    )
    assert isinstance(model, TrainedModel)


def test_feature_importance_is_ranked_and_covers_every_column(fitted):
    importance = feature_importance(fitted)

    assert list(importance.columns) == ["feature", "gain", "split"]
    assert set(importance["feature"]) == set(fitted.columns)
    assert importance["gain"].is_monotonic_decreasing
    # Lagged load should matter most for load forecasting; if it does not, look closer.
    assert importance.iloc[0]["feature"].startswith("load_")


def test_the_importance_figure_renders(fitted):
    figure = feature_importance_figure(fitted, top_n=5)

    assert figure.axes
    assert len(figure.axes[0].patches) == 5
