"""The leakage tests. If these are weak, every number this project ever reports is fiction.

Forecasting `load[t + H]` may use only what is known at `t`. Three independent angles:

1. **Structural** — read the generated columns and check every load-derived offset is at
   least `H` hours old, and that no *other* column is derived from the load series.
2. **Behavioural** — replace the future of the load series with an absurd sentinel and
   check that no feature value on the past side moves. This catches leakage the column
   names would not reveal.
3. **Arithmetic** — verify the rolling window and the target are the exact series they
   claim to be, computed by hand from the raw frame.

All of it runs at several horizons, because a guarantee that only holds at H=24 is an
accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.builder import (
    LAG_COLUMN_PATTERN,
    TARGET_COLUMN,
    lag_hours,
    make_features,
    target_derived_columns,
)

HORIZONS = [1, 6, 24, 48]

# Every column that may touch the load series. A new feature derived from the target
# must be added here deliberately — and then it needs its own leakage argument.
EXPECTED_TARGET_DERIVED = {"load_lag_", "load_roll_mean_", "load_roll_std_", TARGET_COLUMN}


def is_target_derived(column: str) -> bool:
    return column == TARGET_COLUMN or any(
        column.startswith(prefix) for prefix in EXPECTED_TARGET_DERIVED if prefix != TARGET_COLUMN
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_every_lag_column_is_at_least_the_horizon_old(short_dataset, horizon):
    features = make_features(short_dataset, horizon)

    lags = [int(m.group(1)) for c in features.columns if (m := LAG_COLUMN_PATTERN.match(c))]

    assert lags, "the feature set must actually contain lagged load"
    assert min(lags) >= horizon, f"lag {min(lags)}h is younger than the {horizon}h horizon"


@pytest.mark.parametrize("horizon", HORIZONS)
def test_no_unexpected_column_is_derived_from_the_load_series(short_dataset, horizon):
    features = make_features(short_dataset, horizon)

    declared = set(target_derived_columns(list(features.columns)))
    actual = {c for c in features.columns if is_target_derived(c)}

    assert declared == actual
    # Nothing else may carry the raw target: `load_mw` itself is never a feature.
    assert "load_mw" not in features.columns


@pytest.mark.parametrize("horizon", HORIZONS)
def test_future_load_cannot_influence_any_feature(short_dataset, horizon):
    """The behavioural check: poison the future, verify the past does not notice."""
    cut = short_dataset.index[len(short_dataset) // 2]
    poisoned = short_dataset.copy()
    poisoned.loc[poisoned.index > cut, "load_mw"] = 9.9e8  # a sentinel no real grid produces

    clean_features = make_features(short_dataset, horizon)
    poisoned_features = make_features(poisoned, horizon)

    past = clean_features.index <= cut
    feature_columns = [c for c in clean_features.columns if c != TARGET_COLUMN]
    pd.testing.assert_frame_equal(
        clean_features.loc[past, feature_columns],
        poisoned_features.loc[poisoned_features.index <= cut, feature_columns],
    )

    # And prove the test is not vacuous: the sentinel *does* reach the target column.
    poisoned_targets = poisoned_features.loc[poisoned_features.index <= cut, TARGET_COLUMN]
    assert (poisoned_targets > 1e8).any(), "sentinel never reached the target — test is vacuous"


@pytest.mark.parametrize("horizon", HORIZONS)
def test_features_use_nothing_newer_than_horizon_hours_before_the_row(short_dataset, horizon):
    """Poison one hour at a time, at the boundary where leakage would first appear."""
    features = make_features(short_dataset, horizon)
    row = features.index[len(features) // 2]

    for offset in range(0, horizon):
        poisoned = short_dataset.copy()
        poisoned.loc[row - pd.Timedelta(hours=offset), "load_mw"] = 9.9e8

        rebuilt = make_features(poisoned, horizon)
        feature_columns = [c for c in features.columns if c != TARGET_COLUMN]

        pd.testing.assert_series_equal(
            features.loc[row, feature_columns],
            rebuilt.loc[row, feature_columns],
            check_names=False,
        )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_rolling_stats_are_computed_on_an_already_shifted_series(short_dataset, horizon):
    features = make_features(short_dataset, horizon, rolling_window=24)
    row = features.index[len(features) // 2]

    # The window ends `horizon` hours before the row and reaches 24 hours further back.
    window_end = row - pd.Timedelta(hours=horizon)
    window_start = window_end - pd.Timedelta(hours=23)
    window = short_dataset.loc[window_start:window_end, "load_mw"]

    assert len(window) == 24
    assert features.loc[row, "load_roll_mean_24"] == pytest.approx(window.mean())
    assert features.loc[row, "load_roll_std_24"] == pytest.approx(window.std())


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_target_is_the_load_exactly_horizon_hours_later(short_dataset, horizon):
    features = make_features(short_dataset, horizon)

    for row in features.index[[0, len(features) // 2, -1]]:
        target_time = row + pd.Timedelta(hours=horizon)
        assert features.loc[row, TARGET_COLUMN] == pytest.approx(
            short_dataset.loc[target_time, "load_mw"]
        )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_lag_values_are_the_load_exactly_that_many_hours_earlier(short_dataset, horizon):
    features = make_features(short_dataset, horizon)
    row = features.index[len(features) // 2]

    for column in features.columns:
        match = LAG_COLUMN_PATTERN.match(column)
        if match:
            lag = int(match.group(1))
            source = short_dataset.loc[row - pd.Timedelta(hours=lag), "load_mw"]
            assert features.loc[row, column] == pytest.approx(source)


def test_a_lag_younger_than_the_horizon_is_dropped_not_clipped():
    # Beyond a week ahead, the 168-hour lag is itself in the future.
    assert lag_hours(200) == [200, 201, 224, 368]
    assert lag_hours(24) == [24, 25, 48, 168, 192]
    assert lag_hours(1) == [1, 2, 25, 168, 169]


def test_a_zero_horizon_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        lag_hours(0)


def test_gaps_in_the_input_cannot_turn_a_positional_shift_into_a_shorter_one(short_dataset):
    """A missing row would make `shift(24)` reach back 23 hours — that is leakage."""
    horizon = 24
    gapped = short_dataset.drop(short_dataset.index[300:305])

    features = make_features(gapped, horizon)
    row = features.index[-1]

    for column in features.columns:
        match = LAG_COLUMN_PATTERN.match(column)
        if match:
            lag = int(match.group(1))
            expected = short_dataset.loc[row - pd.Timedelta(hours=lag), "load_mw"]
            assert features.loc[row, column] == pytest.approx(expected)


def test_no_feature_correlates_suspiciously_with_the_target_beyond_what_lags_explain(dataset):
    """A blunt smell test: at a 24h horizon, nothing should track the target near-perfectly."""
    features = make_features(dataset, 24)
    correlations = features.drop(columns=[TARGET_COLUMN]).corrwith(features[TARGET_COLUMN]).abs()

    assert correlations.max() < 0.98, f"suspiciously predictive: {correlations.idxmax()}"
    assert not np.isnan(correlations).any()
