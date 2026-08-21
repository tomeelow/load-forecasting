"""Open-Meteo weather: historical archive for training, forecast for serving.

Free and keyless, with the same variables on both endpoints — which is what makes the
train–serve weather comparison in the README possible at all. City coordinates and
population weights come from config; nothing about Poland is hardcoded here.

Per-city series are combined into one population-weighted national series. Demand
follows where people live, so weighting Warsaw at 0.30 and Szczecin at 0.09 tracks
national demand far better than an unweighted mean of seven cities.
"""

from __future__ import annotations

import time

import pandas as pd
import requests
from loguru import logger

from src.config import City, RequestConfig, WeatherConfig

# Open-Meteo variable -> our column name. `wind_speed_unit=ms` below is what makes
# `wind_ms` true: Open-Meteo defaults to km/h.
COLUMN_NAMES = {
    "temperature_2m": "temp_c",
    "wind_speed_10m": "wind_ms",
    "cloud_cover": "cloud_cover",
    "relative_humidity_2m": "humidity_pct",
}

# Statuses worth trying again: a timeout, a rate limit, or a server that is briefly
# unwell. Everything else in 4xx says the request itself is wrong, and repeating a
# malformed request five times only delays the error by a minute.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Transport failures that are, by nature, worth another attempt: the connection never
# opened, the response never finished, or it took too long.
RETRYABLE_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

# The archive is asked for the whole configured history whenever a rebuild happens —
# seven years, seven cities. As one request per city that is a response Open-Meteo takes
# longer than any sane timeout to begin, and a retry restarts the whole thing: the first
# CI rebuild spent five attempts and three minutes failing on exactly that. A year per
# request keeps every response small, so a timeout costs one year rather than all of
# them. entsoe-py chunks its own long ranges for the same reason.
ARCHIVE_CHUNK_DAYS = 365


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


def _retry_after(error: requests.HTTPError) -> float | None:
    """The server's own `Retry-After`, in seconds, when it sent one.

    A rate limiter knows better than our backoff curve does how long it wants to be
    left alone, and Open-Meteo does rate-limit the free tier.
    """
    response = getattr(error, "response", None)
    if response is None:
        return None
    try:
        return float(response.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _get(url: str, params: dict, request: RequestConfig) -> dict:
    """GET one Open-Meteo endpoint, retrying transient failures with exponential backoff.

    The nightly loop makes fourteen of these calls (seven cities, two endpoints) against
    a free service with no availability promise. Without this, one connection reset ends
    the run, the last-success marker stays where it was, and the next run has a wider
    window to backfill — recoverable, but a night of monitoring is lost for a hiccup that
    a two-second pause would have absorbed.

    Only transient failures are retried; a 400 is raised on the first attempt.
    """
    last: Exception | None = None
    for attempt in range(1, request.max_attempts + 1):
        pause = request.pause_before(attempt + 1)
        try:
            response = requests.get(url, params=params, timeout=request.timeout_s)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS:
                raise
            last = exc
            pause = _retry_after(exc) or pause
        except RETRYABLE_ERRORS as exc:
            last = exc

        if attempt == request.max_attempts:
            break
        logger.warning(
            "Open-Meteo {} failed (attempt {}/{}): {}. Retrying in {:.1f}s",
            url,
            attempt,
            request.max_attempts,
            last,
            pause,
        )
        time.sleep(pause)

    logger.error("Open-Meteo {} failed after {} attempt(s): {}", url, request.max_attempts, last)
    raise last


def date_chunks(start_date: str, end_date: str, days: int = ARCHIVE_CHUNK_DAYS) -> list[tuple]:
    """Split a closed date range into consecutive windows of at most `days` days."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        return []
    chunks = []
    while start <= end:
        stop = min(start + pd.Timedelta(days=days - 1), end)
        chunks.append((start.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        start = stop + pd.Timedelta(days=1)
    return chunks


def fetch_city_archive(
    city: City, start_date: str, end_date: str, cfg: WeatherConfig
) -> pd.DataFrame:
    """Observed hourly weather for one city over a closed date range.

    Requested a year at a time; see `ARCHIVE_CHUNK_DAYS`. The windows are consecutive
    and closed on both ends, so the concatenation is the same continuous series a single
    request would have returned.
    """
    frames = [
        parse_hourly(
            _get(
                cfg.archive_url,
                {
                    "latitude": city.lat,
                    "longitude": city.lon,
                    "start_date": chunk_start,
                    "end_date": chunk_end,
                    "hourly": ",".join(cfg.variables),
                    "wind_speed_unit": "ms",
                    "timezone": "UTC",
                },
                cfg.request,
            ),
            cfg.variables,
        )
        for chunk_start, chunk_end in date_chunks(start_date, end_date)
    ]
    if not frames:
        raise ValueError(f"Archive range ends before it starts: {start_date} to {end_date}")

    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def fetch_city_forecast(
    city: City, forecast_days: int, cfg: WeatherConfig, past_days: int = 0
) -> pd.DataFrame:
    """Forecast hourly weather for one city.

    This is what serving gets. It is less accurate than the archive values training
    sees. How much less is not yet measurable from what this repository stores — the
    README says exactly what is missing and what it would take.

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
        cfg.request,
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


def _lead_suffix(lead_days: int) -> str:
    """Open-Meteo's suffix for "the value as it was forecast `lead_days` days earlier"."""
    if lead_days < 1:
        raise ValueError(f"lead_days must be at least 1, got {lead_days}")
    return f"_previous_day{lead_days}"


def fetch_city_day_ahead(
    city: City, start_date: str, end_date: str, cfg: WeatherConfig
) -> pd.DataFrame:
    """The weather that *was forecast* for each past hour, at a fixed lead time.

    Evaluation-only, and it exists to answer one question the archive cannot: how much
    of the model's advantage over PSE is the model, and how much is having been handed
    weather nobody had at forecast time. The archive endpoint returns what was later
    observed; this returns what the forecast said a day earlier, which is what serving
    actually gets.

    Open-Meteo exposes it as a `_previous_dayN` suffix on each variable, so the response
    carries our normal column set once the suffix is stripped.
    """
    if not cfg.historical_forecast_url:
        raise ValueError("No historical_forecast_url configured; cannot fetch day-ahead weather")

    suffix = _lead_suffix(cfg.historical_forecast_lead_days)
    requested = tuple(f"{v}{suffix}" for v in cfg.variables)
    frames = [
        parse_hourly(
            _get(
                cfg.historical_forecast_url,
                {
                    "latitude": city.lat,
                    "longitude": city.lon,
                    "start_date": chunk_start,
                    "end_date": chunk_end,
                    "hourly": ",".join(requested),
                    "wind_speed_unit": "ms",
                    "timezone": "UTC",
                },
                cfg.request,
            ),
            requested,
        ).rename(columns={v: COLUMN_NAMES[v.removesuffix(suffix)] for v in requested})
        for chunk_start, chunk_end in date_chunks(start_date, end_date)
    ]
    if not frames:
        raise ValueError(f"Day-ahead range ends before it starts: {start_date} to {end_date}")

    combined = pd.concat(frames).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def fetch_national_day_ahead(cfg: WeatherConfig, start_date: str, end_date: str) -> pd.DataFrame:
    """Population-weighted day-ahead *forecast* weather over a past range."""
    frames = {c.name: fetch_city_day_ahead(c, start_date, end_date, cfg) for c in cfg.cities}
    national = national_weather(frames, cfg.cities)
    logger.info(
        "Open-Meteo day-ahead archive (lead {}d): {} cities, {} hourly rows, {} to {}",
        cfg.historical_forecast_lead_days,
        len(frames),
        len(national),
        national.index.min(),
        national.index.max(),
    )
    return national
