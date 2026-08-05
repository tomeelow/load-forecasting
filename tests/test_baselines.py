"""The references every model is judged against."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.metrics import mape
from src.features.builder import TARGET_COLUMN, feature_columns, make_features
from src.models.baselines import LinearBaseline, naive_seasonal, tso_forecast


def test_naive_seasonal_is_the_same_hour_one_week_earlier(short_dataset):
    predictions = naive_seasonal(short_dataset)
    row = short_dataset.index[500]

    assert predictions.loc[row] == pytest.approx(
        short_dataset.loc[row - pd.Timedelta(hours=168), "load_mw"]
    )
    assert predictions.iloc[:168].isna().all()  # no week of history yet


def test_naive_seasonal_rejects_an_unsorted_index(short_dataset):
    shuffled = short_dataset.iloc[::-1]

    with pytest.raises(ValueError, match="sorted, unique"):
        naive_seasonal(shuffled)


def test_the_tso_forecast_comes_straight_from_the_dataset(short_dataset):
    predictions = tso_forecast(short_dataset)

    pd.testing.assert_series_equal(
        predictions, short_dataset["tso_forecast_mw"].rename("pse_day_ahead")
    )


def test_the_benchmark_is_good_enough_to_be_worth_beating(short_dataset):
    """If PSE were easy to beat it would not be the benchmark that matters."""
    naive_error = mape(short_dataset["load_mw"], naive_seasonal(short_dataset))
    tso_error = mape(short_dataset["load_mw"], tso_forecast(short_dataset))

    assert tso_error < naive_error


def test_the_linear_baseline_fits_on_calendar_and_weather_but_never_on_lags(short_dataset):
    features = make_features(short_dataset, 24)
    X, y = features.drop(columns=[TARGET_COLUMN]), features[TARGET_COLUMN]

    model = LinearBaseline().fit(X, y)

    assert "temp_c" in model.columns
    assert "hour_sin" in model.columns
    assert not any(c.startswith("load_") for c in model.columns)
    assert model.columns == feature_columns(list(X.columns), include_lags=False)


def test_the_linear_baseline_predicts_on_the_index_it_was_given(short_dataset):
    features = make_features(short_dataset, 24)
    X, y = features.drop(columns=[TARGET_COLUMN]), features[TARGET_COLUMN]

    predictions = LinearBaseline().fit(X, y).predict(X)

    assert predictions.index.equals(X.index)
    assert predictions.notna().all()
    # The clock and the thermometer explain a lot of Polish demand, but not all of it.
    assert 1.0 < mape(y, predictions) < 15.0


def test_predicting_before_fitting_is_an_error(short_dataset):
    features = make_features(short_dataset, 24)

    with pytest.raises(RuntimeError, match="before fit"):
        LinearBaseline().predict(features)
