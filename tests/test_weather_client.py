"""Open-Meteo parsing and the population-weighted national combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import City
from src.ingestion.weather_client import national_weather, parse_hourly

VARIABLES = ("temperature_2m", "wind_speed_10m", "cloud_cover", "relative_humidity_2m")


def payload(times: list[str], **series) -> dict:
    return {"utc_offset_seconds": 0, "hourly": {"time": times, **series}}


def test_parses_hourly_response_into_utc_frame_with_our_column_names():
    response = payload(
        ["2024-05-01T00:00", "2024-05-01T01:00"],
        temperature_2m=[11.5, 12.0],
        wind_speed_10m=[3.1, 3.4],
        cloud_cover=[40, 55],
        relative_humidity_2m=[80, 78],
    )

    frame = parse_hourly(response, VARIABLES)

    assert list(frame.columns) == ["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2024-05-01 00:00", tz="UTC")
    assert frame["temp_c"].tolist() == [11.5, 12.0]


def test_non_utc_response_is_rejected():
    response = payload(["2024-05-01T02:00"], temperature_2m=[11.5])
    response["utc_offset_seconds"] = 7200

    with pytest.raises(ValueError, match="UTC"):
        parse_hourly(response, ("temperature_2m",))


def test_missing_variable_is_rejected():
    response = payload(["2024-05-01T00:00"], temperature_2m=[11.5])

    with pytest.raises(ValueError, match="missing requested variables"):
        parse_hourly(response, ("temperature_2m", "wind_speed_10m"))


def cities() -> tuple[City, ...]:
    return (
        City("Warsaw", 52.23, 21.01, 0.6),
        City("Gdansk", 54.35, 18.65, 0.4),
    )


def city_frame(temp: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-05-01", periods=len(temp), freq="1h", tz="UTC")
    return pd.DataFrame({"temp_c": temp, "wind_ms": [2.0] * len(temp)}, index=index)


def test_national_series_is_population_weighted_not_a_plain_mean():
    frames = {"Warsaw": city_frame([10.0, 10.0]), "Gdansk": city_frame([0.0, 0.0])}

    national = national_weather(frames, cities())

    assert national["temp_c"].tolist() == [6.0, 6.0]  # 0.6 * 10, not the 5.0 plain mean


def test_a_city_with_a_gap_renormalises_rather_than_punching_a_hole():
    warsaw = city_frame([10.0, 10.0])
    gdansk = city_frame([0.0, np.nan])

    national = national_weather({"Warsaw": warsaw, "Gdansk": gdansk}, cities())

    assert national["temp_c"].iloc[0] == pytest.approx(6.0)
    assert national["temp_c"].iloc[1] == pytest.approx(10.0)  # Warsaw's weight renormalised to 1


def test_the_weighted_mean_cannot_drift_outside_the_values_it_averages():
    """Seven cities all at 100% cloud must give 100%, not 100.00000000000001.

    Float division by renormalised weights overshoots by ~1e-14, which is invisible
    until a 0-100 range check rejects 4,369 hours of real data.
    """
    frames = {"Warsaw": city_frame([100.0, 0.0]), "Gdansk": city_frame([100.0, 0.0])}

    national = national_weather(frames, cities())

    assert national["temp_c"].tolist() == [100.0, 0.0]
    assert national["temp_c"].max() <= 100.0
    assert national["temp_c"].min() >= 0.0


def test_the_clip_does_not_distort_a_genuine_average():
    frames = {"Warsaw": city_frame([10.0]), "Gdansk": city_frame([0.0])}

    national = national_weather(frames, cities())

    assert national["temp_c"].iloc[0] == pytest.approx(6.0)  # 0.6 * 10, still weighted


def test_an_unweighted_city_is_rejected():
    frames = {"Warsaw": city_frame([10.0]), "Katowice": city_frame([5.0])}

    with pytest.raises(ValueError, match="No configured weight"):
        national_weather(frames, cities())


def test_configured_city_weights_are_normalised(cfg):
    assert sum(c.weight for c in cfg.weather.cities) == pytest.approx(1.0)
    assert {c.name for c in cfg.weather.cities} >= {"Warsaw", "Krakow", "Gdansk"}
