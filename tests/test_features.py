"""Feature builder behaviour beyond leakage: calendar, weather, horizons, serving."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.builder import TARGET_COLUMN, make_features
from tests.fixtures import synthetic

WARSAW = "Europe/Warsaw"


def test_output_has_no_nulls_left(short_dataset):
    features = make_features(short_dataset, 24)

    assert not features.isna().any().any()
    assert len(features) > 0


def test_leading_rows_without_enough_history_are_dropped(short_dataset):
    features = make_features(short_dataset, 24)

    # The deepest reach back is the 168+24 lag; the rolling window only needs 47 hours.
    assert features.index.min() == short_dataset.index.min() + pd.Timedelta(hours=192)
    # Rows are target-indexed, so the last hour of data is itself a usable target.
    assert features.index.max() == short_dataset.index.max()


def test_calendar_features_are_the_warsaw_clock_not_utc(short_dataset):
    features = make_features(short_dataset, 24)
    row = pd.Timestamp("2024-01-20 15:00", tz="UTC")  # a Saturday, 16:00 in Warsaw

    assert features.loc[row, "hour"] == 16  # not 15: demand follows the local clock
    assert features.loc[row, "dow"] == 5
    assert features.loc[row, "month"] == 1
    assert features.loc[row, "is_weekend"] == 1


def test_the_local_day_boundary_is_what_flips_the_weekend_flag(short_dataset):
    features = make_features(short_dataset, 24)

    # 23:00 UTC on Sunday is already Monday 00:00 in Warsaw, so the weekend is over.
    assert features.loc[pd.Timestamp("2024-01-21 22:00", tz="UTC"), "is_weekend"] == 1
    assert features.loc[pd.Timestamp("2024-01-21 23:00", tz="UTC"), "is_weekend"] == 0
    assert features.loc[pd.Timestamp("2024-01-21 23:00", tz="UTC"), "dow"] == 0


def test_cyclical_encodings_wrap_around(short_dataset):
    features = make_features(short_dataset, 24)

    on_circle = features["hour_sin"] ** 2 + features["hour_cos"] ** 2
    assert np.allclose(on_circle, 1.0)

    midnight = features[features["hour"] == 0].iloc[0]
    assert midnight["hour_sin"] == pytest.approx(0.0, abs=1e-12)
    assert midnight["hour_cos"] == pytest.approx(1.0)

    # Hour 23 is adjacent to hour 0, which is the whole point of the encoding.
    late = features[features["hour"] == 23].iloc[0]
    assert (
        np.hypot(late["hour_sin"] - midnight["hour_sin"], late["hour_cos"] - midnight["hour_cos"])
        < 0.3
    )


def test_holiday_flag_matches_the_polish_calendar(short_dataset):
    features = make_features(short_dataset, 24)
    local_dates = features.index.tz_convert(WARSAW).date.astype(str)

    assert features["is_holiday"][local_dates == "2024-02-05"].eq(0).all()
    assert features["is_holiday"].isin([0, 1]).all()


def test_weather_features_include_the_squared_temperature_term(short_dataset):
    features = make_features(short_dataset, 24)

    # U-shaped response: demand rises in both cold and heat, so temp alone is not enough.
    assert np.allclose(features["temp_sq"], features["temp_c"] ** 2)
    assert {"wind_ms", "cloud_cover"} <= set(features.columns)


@pytest.mark.parametrize("horizon", [1, 24, 48])
def test_weather_is_the_weather_of_the_hour_being_predicted(short_dataset, horizon):
    """Not the weather at the prediction moment — demand responds to the hour it is in.

    Legal because the forecast endpoint publishes it in advance; see the builder.
    """
    features = make_features(short_dataset, horizon)
    row = features.index[len(features) // 2]

    for column in ("temp_c", "wind_ms", "cloud_cover"):
        assert features.loc[row, column] == pytest.approx(short_dataset.loc[row, column])


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_the_lag_set_moves_with_the_horizon(short_dataset, horizon):
    features = make_features(short_dataset, horizon)

    assert f"load_lag_{horizon}" in features.columns
    assert f"load_lag_{horizon + 24}" in features.columns
    assert "load_lag_168" in features.columns  # always safe at day-ahead horizons


def test_the_rolling_window_is_reflected_in_the_column_name(short_dataset):
    features = make_features(short_dataset, 24, rolling_window=48)

    assert "load_roll_mean_48" in features.columns
    assert "load_roll_mean_24" not in features.columns


def test_serving_omits_the_target_column(short_dataset):
    features = make_features(short_dataset, 24, include_target=False)

    assert TARGET_COLUMN not in features.columns
    assert features.index.equals(make_features(short_dataset, 24).index)


def test_serving_can_build_features_for_tomorrow_from_weather_alone():
    """The real serving shape: load known up to now, weather forecast running ahead."""
    horizon = 24
    frame = synthetic.make_dataset(start="2024-01-01", end="2024-02-02")
    last_actual = pd.Timestamp("2024-02-01 00:00", tz="UTC")
    frame.loc[frame.index > last_actual, "load_mw"] = np.nan

    features = make_features(frame, horizon, include_target=False)

    # Every hour of the next day is predictable, and no hour beyond it pretends to be.
    assert features.index.max() == last_actual + pd.Timedelta(hours=horizon)
    assert not features.isna().any().any()


def test_missing_input_columns_are_rejected(short_dataset):
    with pytest.raises(ValueError, match="missing required columns"):
        make_features(short_dataset.drop(columns=["temp_c"]), 24)


def test_a_naive_index_is_rejected(short_dataset):
    naive = short_dataset.copy()
    naive.index = naive.index.tz_localize(None)

    with pytest.raises(ValueError, match="timezone-aware"):
        make_features(naive, 24)


def test_a_duplicated_index_is_rejected(short_dataset):
    duplicated = pd.concat([short_dataset, short_dataset.iloc[:2]]).sort_index()

    with pytest.raises(ValueError, match="unique index"):
        make_features(duplicated, 24)


@pytest.mark.parametrize("day", ["2024-03-31", "2024-10-27"])
def test_features_are_built_across_both_dst_transitions(dataset, day):
    window = dataset.loc[
        pd.Timestamp(day, tz=WARSAW) - pd.Timedelta(days=14) : pd.Timestamp(day, tz=WARSAW)
        + pd.Timedelta(days=2)
    ]

    features = make_features(window, 24)

    assert not features.isna().any().any()
    assert not features.index.has_duplicates
    assert features.index.is_monotonic_increasing
    # Every UTC day in the output still has 24 rows: the transition is a local-clock event.
    per_day = features.groupby(features.index.date).size()
    assert set(per_day.iloc[1:-1]) == {24}
