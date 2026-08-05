"""The fixture is load-bearing: every other test trusts its shape, so check it."""

from __future__ import annotations

import pandas as pd

from src.ingestion.dataset import CANONICAL_COLUMNS
from tests.fixtures import synthetic

WARSAW = "Europe/Warsaw"


def test_index_is_complete_utc_hourly(dataset):
    assert dataset.index.tz is not None
    assert str(dataset.index.tz) == "UTC"
    expected = pd.date_range(dataset.index.min(), dataset.index.max(), freq="1h", tz="UTC")
    assert dataset.index.equals(expected)
    assert list(dataset.columns) == CANONICAL_COLUMNS


def test_spans_both_dst_transitions(dataset):
    local_days = set(dataset.index.tz_convert(WARSAW).date.astype(str))
    assert {"2024-03-31", "2024-10-27"} <= local_days


def test_load_stays_within_plausible_polish_bounds(dataset, cfg):
    assert dataset["load_mw"].between(cfg.validation.load_min_mw, cfg.validation.load_max_mw).all()


def test_daily_profile_has_two_peaks_and_a_night_trough(dataset):
    profile = dataset["load_mw"].groupby(dataset.index.tz_convert(WARSAW).hour).mean()

    assert profile.idxmin() in range(2, 5)  # overnight trough
    assert profile.idxmax() in range(17, 21)  # evening peak, the higher of the two
    assert profile[9] > profile[13] < profile[19]  # morning bump, midday dip, evening peak
    assert profile.max() - profile.min() > 3_000  # a real swing, not a ripple


def test_weekends_are_lower_than_weekdays(dataset):
    local = dataset.index.tz_convert(WARSAW)
    by_weekend = dataset["load_mw"].groupby(local.dayofweek >= 5).mean()
    assert by_weekend[True] < by_weekend[False] * 0.95


def test_winter_peaks_and_summer_troughs(dataset):
    by_month = dataset["load_mw"].groupby(dataset.index.tz_convert(WARSAW).month).mean()
    assert by_month[1] > by_month[7]
    assert by_month.idxmax() in (1, 12)


def test_temperature_is_seasonal_and_correlates_with_load(dataset):
    by_month = dataset["temp_c"].groupby(dataset.index.tz_convert(WARSAW).month).mean()
    assert by_month[1] < 3 < by_month[7]
    # Winter-peaking system: colder means more demand, over the year as a whole.
    assert dataset["temp_c"].corr(dataset["load_mw"]) < -0.3


def test_tso_forecast_is_a_credible_benchmark(dataset):
    error = dataset["tso_forecast_mw"] - dataset["load_mw"]
    mape = (error.abs() / dataset["load_mw"]).mean() * 100
    assert 0.3 < mape < 3.0, "the benchmark should be good, but not clairvoyant"
    # Its errors persist for hours, as an operational forecast's do.
    assert error.autocorr(lag=1) > 0.5


def test_generator_is_deterministic():
    a = synthetic.make_dataset(start="2024-01-01", end="2024-01-08")
    b = synthetic.make_dataset(start="2024-01-01", end="2024-01-08")
    pd.testing.assert_frame_equal(a, b)


def test_subhourly_series_resolution_and_timezone():
    series = synthetic.make_subhourly_load("2024-03-30", "2024-04-01", tz=WARSAW)
    assert str(series.index.tz) == WARSAW
    assert (series.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
