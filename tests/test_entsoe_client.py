"""ENTSO-E client: the conversions, exercised without a token or a network call."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from entsoe.exceptions import NoMatchingDataError

from src.ingestion.entsoe_client import (
    FORECAST_COLUMN,
    LOAD_COLUMN,
    fetch_actual_load,
    fetch_load_frame,
    to_utc_hourly,
    trailing_window,
)
from tests.fixtures import synthetic

WARSAW = "Europe/Warsaw"


class StubClient:
    """Stands in for EntsoePandasClient, returning what entsoe-py returns: frames in
    the area's local timezone, load often at 15-minute resolution."""

    def __init__(self, load=None, forecast=None, raises: type[Exception] | None = None):
        self._load = load
        self._forecast = forecast
        self._raises = raises
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def query_load(self, country_code, start, end):
        self.calls.append(("load", start, end))
        if self._raises:
            raise self._raises("no data")
        return self._load

    def query_load_forecast(self, country_code, start, end):
        self.calls.append(("forecast", start, end))
        if self._raises:
            raise self._raises("no data")
        return self._forecast


def quarter_hourly(start="2024-05-01", end="2024-05-03", tz=WARSAW) -> pd.Series:
    return synthetic.make_subhourly_load(start, end, tz=tz)


def test_resamples_quarter_hourly_to_hourly_means():
    index = pd.date_range("2024-05-01", periods=8, freq="15min", tz="UTC")
    series = pd.Series([100.0, 200.0, 300.0, 400.0, 10.0, 20.0, 30.0, 40.0], index=index)

    hourly = to_utc_hourly(series)

    assert len(hourly) == 2
    # Mean, not sum: the values are instantaneous MW, so the hourly figure is average power.
    assert hourly.iloc[0] == pytest.approx(250.0)
    assert hourly.iloc[1] == pytest.approx(25.0)


def test_converts_local_time_to_utc():
    series = quarter_hourly()

    hourly = to_utc_hourly(series)

    assert str(hourly.index.tz) == "UTC"
    assert hourly.index.is_monotonic_increasing
    assert not hourly.index.has_duplicates


def test_missing_hours_become_nulls_not_missing_rows():
    series = quarter_hourly("2024-05-01", "2024-05-02", tz="UTC")
    gapped = series.drop(series.index[8:12])  # drop 02:00–02:45 entirely

    hourly = to_utc_hourly(gapped)

    expected = pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h", tz="UTC")
    assert hourly.index.equals(expected)
    assert hourly.isna().sum() == 1


def test_naive_index_is_rejected():
    series = quarter_hourly().tz_localize(None)

    with pytest.raises(ValueError, match="timezone-aware"):
        to_utc_hourly(series)


def test_revised_duplicate_timestamps_resolve_to_the_last_value():
    index = pd.date_range("2024-05-01", periods=4, freq="15min", tz="UTC")
    first_publication = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    revision = pd.Series([200.0, 200.0, 200.0, 200.0], index=index)
    stitched = pd.concat([first_publication, revision])

    hourly = to_utc_hourly(stitched)

    assert len(hourly) == 1
    assert hourly.iloc[0] == pytest.approx(200.0)


def test_reads_the_named_column_out_of_an_entsoe_frame():
    series = quarter_hourly()
    frame = pd.DataFrame({"Actual Load": series, "Something Else": series * 2})

    hourly = fetch_actual_load(StubClient(load=frame), "PL", *trailing_window(3))

    assert hourly.name == LOAD_COLUMN
    assert hourly.max() < series.max() * 1.5  # took Actual Load, not Something Else


def test_load_and_tso_forecast_come_back_together():
    load = quarter_hourly()
    forecast = pd.DataFrame({"Forecasted Load": load.resample("1h").mean() + 150.0})
    client = StubClient(load=pd.DataFrame({"Actual Load": load}), forecast=forecast)

    frame = fetch_load_frame(client, "PL", *trailing_window(3))

    assert list(frame.columns) == [LOAD_COLUMN, FORECAST_COLUMN]
    assert frame[FORECAST_COLUMN].notna().any()
    assert [c[0] for c in client.calls] == ["load", "forecast"]


def test_absent_data_is_a_warning_not_a_crash():
    client = StubClient(raises=NoMatchingDataError)

    frame = fetch_load_frame(client, "PL", *trailing_window(3))

    assert frame.empty
    assert list(frame.columns) == [LOAD_COLUMN, FORECAST_COLUMN]


def test_trailing_window_covers_the_repull_period_and_tomorrow():
    now = pd.Timestamp("2026-08-05 06:30", tz="UTC")

    start, end = trailing_window(14, now=now)

    assert start == pd.Timestamp("2026-07-22 00:00", tz="UTC")
    # Tomorrow is included so the day-ahead forecast is picked up as soon as it publishes.
    assert end == pd.Timestamp("2026-08-07 00:00", tz="UTC")
    assert (end - start) > pd.Timedelta(days=14)


def test_trailing_window_rejects_a_zero_day_window():
    with pytest.raises(ValueError, match="at least 1"):
        trailing_window(0)


def test_hourly_aggregation_of_a_synthetic_day_is_the_quarter_hourly_mean():
    series = quarter_hourly("2024-05-01", "2024-05-02", tz="UTC")

    hourly = to_utc_hourly(series)

    first_hour = series.loc["2024-05-01 00:00":"2024-05-01 00:45"]
    assert hourly.iloc[0] == pytest.approx(np.mean(first_hour.to_numpy()))
