"""DST: Europe/Warsaw loses an hour each spring and repeats one each autumn.

Both are handled by computing in UTC, where neither transition exists. These tests
exist because "handled by construction" is a claim, and the failure mode it prevents
— NaNs, duplicate index entries, an off-by-one hour count — is silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import synthetic
from src.features.builder import make_features
from src.ingestion.dataset import build_dataset
from src.ingestion.entsoe_client import to_utc_hourly

WARSAW = "Europe/Warsaw"

SPRING_FORWARD = "2024-03-31"  # 02:00 local never happens: the local day has 23 hours
AUTUMN_BACK = "2024-10-27"  # 02:00 local happens twice: the local day has 25 hours


def local_day_hours(index: pd.DatetimeIndex, day: str) -> int:
    local = index.tz_convert(WARSAW)
    return int(np.sum(local.date.astype(str) == day))


@pytest.mark.parametrize(
    ("day", "expected_local_hours"),
    [(SPRING_FORWARD, 23), (AUTUMN_BACK, 25)],
)
def test_quarter_hourly_local_data_resamples_cleanly_across_a_transition(day, expected_local_hours):
    # As entsoe-py hands it back: 15-minute resolution, localised to Europe/Warsaw.
    start = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(day) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    series = synthetic.make_subhourly_load(start, end, tz=WARSAW)

    hourly = to_utc_hourly(series)

    assert hourly.notna().all(), "no NaN explosion across the transition"
    assert not hourly.index.has_duplicates
    assert hourly.index.is_monotonic_increasing
    assert str(hourly.index.tz) == "UTC"
    assert local_day_hours(hourly.index, day) == expected_local_hours


def test_every_utc_day_has_24_hours_including_the_transition_days():
    series = synthetic.make_subhourly_load("2024-03-30", "2024-04-02", tz=WARSAW)

    hourly = to_utc_hourly(series)
    per_utc_day = hourly.groupby(hourly.index.date).size()

    assert set(per_utc_day.iloc[:-1]) == {24}


def test_the_repeated_autumn_hour_is_two_distinct_utc_instants():
    series = synthetic.make_subhourly_load("2024-10-26", "2024-10-28", tz=WARSAW)

    hourly = to_utc_hourly(series)
    local = hourly.index.tz_convert(WARSAW)
    repeated = hourly.index[(local.date.astype(str) == AUTUMN_BACK) & (local.hour == 2)]

    assert len(repeated) == 2
    assert repeated[0] == pd.Timestamp("2024-10-27 00:00", tz="UTC")
    assert repeated[1] == pd.Timestamp("2024-10-27 01:00", tz="UTC")


def test_the_missing_spring_hour_has_no_utc_row_pretending_otherwise():
    series = synthetic.make_subhourly_load("2024-03-30", "2024-04-01", tz=WARSAW)

    hourly = to_utc_hourly(series)
    local = hourly.index.tz_convert(WARSAW)

    assert not ((local.date.astype(str) == SPRING_FORWARD) & (local.hour == 2)).any()
    assert ((local.date.astype(str) == SPRING_FORWARD) & (local.hour == 3)).sum() == 1


@pytest.mark.parametrize(
    ("day", "expected_local_hours"),
    [(SPRING_FORWARD, 23), (AUTUMN_BACK, 25)],
)
def test_dataset_assembly_survives_a_transition(day, expected_local_hours):
    start = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(day) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    source = synthetic.make_dataset(start=start, end=end)
    load = source[["load_mw", "tso_forecast_mw"]]
    weather = source[["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]]

    df = build_dataset(load, weather, run_id="run-dst")

    assert not df.index.has_duplicates
    assert df.index.equals(pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC"))
    assert df[["load_mw", "tso_forecast_mw", "temp_c"]].notna().all().all()
    assert df["is_holiday"].notna().all()
    assert local_day_hours(df.index, day) == expected_local_hours


@pytest.mark.parametrize("day", [SPRING_FORWARD, AUTUMN_BACK])
def test_calendar_features_carry_the_local_hour_across_a_transition(dataset, day):
    """`hour` is the Warsaw clock, so it must stay right where the Warsaw clock jumps."""
    window = dataset.loc[
        pd.Timestamp(day, tz=WARSAW) - pd.Timedelta(days=14) : pd.Timestamp(day, tz=WARSAW)
        + pd.Timedelta(days=2)
    ]

    features = make_features(window, 24)
    local = features.index.tz_convert(WARSAW)

    assert (features["hour"].to_numpy() == local.hour.to_numpy()).all()
    assert (features["dow"].to_numpy() == local.dayofweek.to_numpy()).all()
    assert (features["is_weekend"].to_numpy() == (local.dayofweek >= 5).astype(int)).all()
    # The UTC hour differs from the local hour on both sides of the jump, so a UTC-derived
    # feature would have been wrong here — this assertion has teeth.
    assert not (features["hour"].to_numpy() == features.index.hour.to_numpy()).all()


def test_the_spring_hour_that_never_happens_is_absent_from_the_hour_feature(dataset):
    window = dataset.loc["2024-03-01":"2024-04-05"]

    features = make_features(window, 24)
    local = features.index.tz_convert(WARSAW)
    on_the_day = features["hour"][local.date.astype(str) == SPRING_FORWARD]

    assert 2 not in set(on_the_day)  # 02:00 local does not exist that day
    assert sorted(set(on_the_day)) == [0, 1, *range(3, 24)]


def test_the_repeated_autumn_hour_appears_twice_in_the_hour_feature(dataset):
    window = dataset.loc["2024-10-01":"2024-11-05"]

    features = make_features(window, 24)
    local = features.index.tz_convert(WARSAW)
    on_the_day = features["hour"][local.date.astype(str) == AUTUMN_BACK]

    assert (on_the_day == 2).sum() == 2  # 02:00 local happens twice
    assert len(on_the_day) == 25


def test_a_year_of_hourly_rows_accounts_for_both_transitions(dataset):
    # 2024 is a leap year: 366 * 24 = 8784 UTC hours, transitions or not.
    rows_2024 = int((dataset.index.year == 2024).sum())
    assert rows_2024 == 8784
