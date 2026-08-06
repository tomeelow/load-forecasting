"""Open-Meteo weather: historical archive for training, forecast for serving.

Free and keyless, with the same variables on both endpoints — which is what makes the
train–serve weather comparison in the README possible at all. City coordinates and
population weights come from config; nothing about Poland is hardcoded here.

Per-city series are combined into one population-weighted national series. Demand
follows where people live, so weighting Warsaw at 0.30 and Szczecin at 0.09 tracks
national demand far better than an unweighted mean of seven cities.
"""

from __future__ import annotations

import pandas as pd
import requests
from loguru import logger

from src.config import City, WeatherConfig

# Open-Meteo variable -> our column name. `wind_speed_unit=ms` below is what makes
# `wind_ms` true: Open-Meteo defaults to km/h.
COLUMN_NAMES = {
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_ms",
    "cloud_cover": "cloud_cover",
    "relative_humidity_2m": "humidity_pct",
}

REQUEST_TIMEOUT_S = 60


def parse_hourly(payload: dict, variables: tuple[str, ...]) -> pd.DataFrame:
    """Turn an Open-Meteo hourly response into a UTC-indexed frame with our column names."""
    offset = payload.get("utc_offset_seconds", 0)
    if offset != 0:
        raise ValueError(f"Expected a UTC response from Open-Meteo, got offset {offset}s")

    hourly = payload["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), name="timestamp_utc").tz_localize(
        "UTC"
    )
    missing = [v for v in variables if v not in hourly]
    if missing:
        raise ValueError(f"Open-Meteo response is missing requested variables: {missing}")

    frame = pd.DataFrame({v: hourly[v] for v in variables}, index=index, dtype="float64")
    return frame.rename(columns=COLUMN_NAMES).sort_index()


def _get(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def fetch_city_archive(
    city: City, start_date: str, end_date: str, cfg: WeatherConfig
) -> pd.DataFrame:
    """Observed hourly weather for one city over a closed date range."""
    payload = _get(
        cfg.archive_url,
        {
            "latitude": city.lat,
            "longitude": city.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(cfg.variables),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
    )
    return parse_hourly(payload, cfg.variables)


def fetch_city_forecast(
    city: City, forecast_days: int, cfg: WeatherConfig, past_days: int = 0
) -> pd.DataFrame:
    """Forecast hourly weather for one city.

    This is what serving gets. It is less accurate than the archive values training
    sees — the skew is real and is quantified rather than ignored (see README).

    `past_days` extends the response backwards, which incremental ingestion needs: the
    archive endpoint lags real time by several days, so without it every daily run
    would leave a weather hole behind the present.
    """
    payload = _get(
        cfg.forecast_url,
        {
            "latitude": city.lat,
            "longitude": city.lon,
            "forecast_days": forecast_days,
            "past_days": past_days,
            "hourly": ",".join(cfg.variables),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
    )
    return parse_hourly(payload, cfg.variables)


def national_weather(frames: dict[str, pd.DataFrame], cities: tuple[City, ...]) -> pd.DataFrame:
    """Combine per-city frames into one population-weighted national frame.

    Weights are renormalised per timestamp over the cities that actually reported, so
    a gap in one city shifts the average slightly rather than punching a hole in it.
    """
    if not frames:
        raise ValueError("No city weather frames to combine")

    weights = {c.name: c.weight for c in cities}
    unknown = set(frames) - set(weights)
    if unknown:
        raise ValueError(f"No configured weight for {sorted(unknown)}")

    columns = list(next(iter(frames.values())).columns)
    combined = {}
    for column in columns:
        wide = pd.DataFrame({name: frame[column] for name, frame in frames.items()})
        w = pd.Series({name: weights[name] for name in wide.columns})
        weighted_sum = wide.mul(w, axis=1).sum(axis=1, skipna=True)
        weight_present = wide.notna().mul(w, axis=1).sum(axis=1)
        average = (weighted_sum / weight_present).where(weight_present > 0)
        # A weighted mean with non-negative weights lies between the smallest and
        # largest contributing value, so clipping to them changes nothing mathematically
        # and removes float division error. Without it, seven cities all reporting 100%
        # cloud produce 100.00000000000001, which then fails a 0-100 range check.
        combined[column] = average.clip(wide.min(axis=1), wide.max(axis=1))

    national = pd.DataFrame(combined).sort_index()
    national.index.name = "timestamp_utc"
    return national


def fetch_national_archive(cfg: WeatherConfig, start_date: str, end_date: str) -> pd.DataFrame:
    """Population-weighted observed weather for the whole country."""
    frames = {c.name: fetch_city_archive(c, start_date, end_date, cfg) for c in cfg.cities}
    national = national_weather(frames, cfg.cities)
    logger.info(
        "Open-Meteo archive: {} cities, {} hourly rows, {} to {}",
        len(frames),
        len(national),
        national.index.min(),
        national.index.max(),
    )
    return national


def fetch_national_forecast(
    cfg: WeatherConfig, forecast_days: int, past_days: int = 0
) -> pd.DataFrame:
    """Population-weighted forecast weather for the whole country."""
    frames = {c.name: fetch_city_forecast(c, forecast_days, cfg, past_days) for c in cfg.cities}
    national = national_weather(frames, cfg.cities)
    logger.info(
        "Open-Meteo forecast: {} cities, {} hourly rows out to {}",
        len(frames),
        len(national),
        national.index.max(),
    )
    return national
