"""Assembling, merging and persisting the canonical table."""

from __future__ import annotations

import pandas as pd
import pytest

from src import synthetic
from src.ingestion.dataset import (
    CANONICAL_COLUMNS,
    build_dataset,
    merge_datasets,
    read_dataset,
    write_dataset,
)

WARSAW = "Europe/Warsaw"


def load_frame(start="2024-05-01", periods=48) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "load_mw": range(20_000, 20_000 + periods),
            "tso_forecast_mw": range(20_100, 20_100 + periods),
        },
        index=index,
        dtype="float64",
    )


def weather_frame(start="2024-05-01", periods=48) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"temp_c": 12.0, "wind_ms": 3.0, "cloud_cover": 40.0, "humidity_pct": 70.0},
        index=index,
    )


def test_builds_the_canonical_columns_on_a_complete_hourly_index():
    df = build_dataset(load_frame(), weather_frame(), run_id="run-test")

    assert list(df.columns) == CANONICAL_COLUMNS
    expected = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    assert df.index.equals(expected)
    assert (df["data_source_version"] == "run-test").all()


def test_gaps_become_null_rows_rather_than_absent_rows():
    load = load_frame()
    load = load.drop(load.index[5:9])

    df = build_dataset(load, weather_frame(), run_id="run-test")

    assert len(df) == 48
    assert df["load_mw"].isna().sum() == 4
    assert df["temp_c"].notna().all()


def test_weather_reaching_past_the_last_actual_is_kept_for_serving():
    df = build_dataset(load_frame(periods=24), weather_frame(periods=48), run_id="run-test")

    assert len(df) == 48
    assert df["load_mw"].isna().sum() == 24
    assert df["temp_c"].notna().all()


def test_holiday_flag_follows_the_warsaw_calendar_not_the_utc_one():
    # 23:00 UTC on Christmas Eve is already 00:00 on Christmas Day in Warsaw.
    index = pd.date_range("2024-12-24 21:00", periods=4, freq="1h", tz="UTC")
    load = pd.DataFrame({"load_mw": 20_000.0, "tso_forecast_mw": 20_000.0}, index=index)

    df = build_dataset(load, weather_frame(start="2024-12-24 21:00", periods=4), run_id="r")

    flags = df["is_holiday"].tolist()
    assert flags == [False, False, True, True]
    assert df.index[2] == pd.Timestamp("2024-12-24 23:00", tz="UTC")


def test_known_polish_holidays_are_flagged(dataset):
    local_dates = dataset.index.tz_convert(WARSAW).date.astype(str)
    for holiday in (
        "2024-05-03",
        "2024-11-11",
        "2024-12-25",
    ):  # Constitution, Independence, Christmas
        assert dataset["is_holiday"][local_dates == holiday].all()
    assert not dataset["is_holiday"][local_dates == "2024-05-06"].any()


def test_two_empty_frames_are_rejected():
    with pytest.raises(ValueError, match="empty frames"):
        build_dataset(
            pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")),
            pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")),
        )


def test_merge_lets_revised_values_win_and_keeps_the_rest():
    existing = build_dataset(load_frame(periods=48), weather_frame(periods=48), run_id="run-old")
    revised_load = load_frame(start="2024-05-02", periods=48) + 500
    incoming = build_dataset(
        revised_load, weather_frame(start="2024-05-02", periods=48), run_id="run-new"
    )

    merged = merge_datasets(existing, incoming)

    expected = pd.date_range(merged.index.min(), merged.index.max(), freq="1h", tz="UTC")
    assert merged.index.equals(expected)
    assert len(merged) == 72
    # Overlapping hours carry the newer pull; untouched hours keep their provenance.
    assert merged.loc["2024-05-02 00:00", "data_source_version"] == "run-new"
    assert merged.loc["2024-05-01 00:00", "data_source_version"] == "run-old"
    assert merged.loc["2024-05-02 00:00", "load_mw"] == pytest.approx(20_500.0)


def test_merge_preserves_the_boolean_holiday_dtype():
    existing = build_dataset(load_frame(), weather_frame(), run_id="run-old")
    incoming = build_dataset(
        load_frame(start="2024-05-03"), weather_frame(start="2024-05-03"), run_id="run-new"
    )

    merged = merge_datasets(existing, incoming)

    assert merged["is_holiday"].dtype == bool


def test_parquet_round_trip_preserves_the_utc_index_and_dtypes(tmp_path):
    df = synthetic.make_dataset(start="2024-01-01", end="2024-01-15")

    path = write_dataset(df, tmp_path / "processed" / "dataset.parquet")
    restored = read_dataset(path)

    assert path.exists()
    # freq is index metadata parquet does not carry; nothing downstream relies on it.
    pd.testing.assert_frame_equal(df, restored, check_freq=False)
    assert str(restored.index.tz) == "UTC"
