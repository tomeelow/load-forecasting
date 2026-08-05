"""A synthetic Polish load series, so nothing here needs an API token or a network.

This is a supported component, not a test helper. The ENTSO-E token takes days to
arrive, and until it does this generator is what lets the whole pipeline — features,
training, tracking, backtest — run end to end. Anything trained on it is marked
`synthetic` in MLflow, because a metric from invented data is a plumbing check and
not a result. The test suite uses it as its fixture for the same reason.

The shape matters more than the realism: the tests below assert things about lags,
DST and validation, and a series without a daily double peak or a weekend dip cannot
exercise them meaningfully. What is reproduced here:

* a **daily double peak** — a morning ramp and a higher evening peak, positioned on
  the *Europe/Warsaw* clock, which is what makes the DST transitions visible in the
  data rather than only in the index;
* **weekly seasonality** — weekends roughly 12% below weekdays, holidays lower still;
* **annual seasonality** — a winter-peaking system, deepest in December, mildest in
  summer, in the 10–28 GW band the validator considers plausible for Poland;
* a **correlated temperature series** with its own annual and daily cycles and AR(1)
  noise, which the load responds to in a U shape (heating below ~15 °C, cooling above
  ~22 °C);
* `tso_forecast_mw` = actual load plus small **autocorrelated** error, because a real
  TSO forecast is good and its mistakes persist for hours rather than resampling
  independently every hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar_pl import DEFAULT_TZ, is_holiday
from src.ingestion.dataset import CANONICAL_COLUMNS

# Both DST transitions in one default span: 2024-03-31 (spring forward, hour 02:00
# local never happens) and 2024-10-27 (autumn back, hour 02:00 local happens twice).
DEFAULT_START = "2023-06-01"
DEFAULT_END = "2025-03-01"


def _ar1(n: int, rng: np.random.Generator, sd: float, rho: float = 0.85) -> np.ndarray:
    """Autocorrelated noise — errors that persist, not white noise."""
    innovations = rng.normal(0.0, sd * np.sqrt(1 - rho**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innovations[i]
    return out


def _daily_shape(local_hour: np.ndarray) -> np.ndarray:
    """A daytime plateau with two bumps on it: morning ~08:30, higher evening ~19:00.

    Centred on zero so the caller's amplitude sets the trough-to-peak swing directly.
    """
    u = (local_hour - 3.5) % 24  # hours since the overnight trough
    plateau = 1 / (1 + np.exp(-(u - 2.5))) / (1 + np.exp(u - 19.0))
    morning = 0.35 * np.exp(-0.5 * ((u - 5.0) / 1.8) ** 2)
    evening = 0.55 * np.exp(-0.5 * ((u - 15.5) / 2.2) ** 2)
    return plateau + morning + evening - 0.80  # ~zero-mean over a day


def synthetic_weather(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Temperature, wind, cloud and humidity on the given UTC hourly index."""
    local = index.tz_convert(DEFAULT_TZ)
    day_of_year = local.dayofyear.to_numpy()
    local_hour = local.hour.to_numpy() + local.minute.to_numpy() / 60

    # Coldest in mid-January, warmest in mid-July; daily swing peaks mid-afternoon.
    annual = -10.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    daily = 4.0 * np.sin(2 * np.pi * (local_hour - 9) / 24)
    temp = 8.5 + annual + daily + _ar1(len(index), rng, sd=3.0, rho=0.93)

    wind = np.clip(
        4.0 + 1.5 * np.sin(2 * np.pi * day_of_year / 365.25) + _ar1(len(index), rng, 2.0), 0, None
    )
    cloud = np.clip(55 + _ar1(len(index), rng, sd=25.0, rho=0.9), 0, 100)
    humidity = np.clip(75 - 0.8 * daily + _ar1(len(index), rng, sd=10.0, rho=0.9), 0, 100)

    return pd.DataFrame(
        {"temp_c": temp, "wind_ms": wind, "cloud_cover": cloud, "humidity_pct": humidity},
        index=index,
    )


def synthetic_load(
    index: pd.DatetimeIndex, temp_c: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Polish-shaped hourly demand in MW for the given UTC hourly index."""
    local = index.tz_convert(DEFAULT_TZ)
    local_hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    day_of_year = local.dayofyear.to_numpy()
    is_weekend = local.dayofweek.to_numpy() >= 5
    holiday = is_holiday(index)

    base = 18_500.0
    annual = 1_600.0 * np.cos(2 * np.pi * (day_of_year - 5) / 365.25)  # winter-peaking
    daily = 3_300.0 * _daily_shape(local_hour)
    weekly = np.where(is_weekend, -2_300.0, 0.0) + np.where(holiday, -1_500.0, 0.0)

    heating = 130.0 * np.clip(15.0 - temp_c, 0, None)
    cooling = 90.0 * np.clip(temp_c - 22.0, 0, None)

    noise = _ar1(len(index), rng, sd=350.0, rho=0.8)
    load = base + annual + daily + weekly + heating + cooling + noise
    return np.clip(load, 10_500, 27_500)


def make_dataset(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    seed: int = 20260805,
    run_id: str = "run-synthetic",
    forecast_error_sd: float = 260.0,
) -> pd.DataFrame:
    """A full canonical dataset: the same columns and dtypes ingestion produces.

    `tso_forecast_mw` is the actual load plus autocorrelated error, i.e. a benchmark
    that is genuinely good — as PSE's is — so nothing in the tests can beat it by
    accident.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="1h", tz="UTC", name="timestamp_utc")

    weather = synthetic_weather(index, rng)
    load = synthetic_load(index, weather["temp_c"].to_numpy(), rng)
    forecast = load + _ar1(len(index), rng, sd=forecast_error_sd, rho=0.9)

    df = weather.copy()
    df["load_mw"] = load
    df["tso_forecast_mw"] = forecast
    df["is_holiday"] = is_holiday(index)
    df["data_source_version"] = run_id
    return df[CANONICAL_COLUMNS]


def make_subhourly_load(
    start: str,
    end: str,
    freq: str = "15min",
    tz: str = "UTC",
    seed: int = 7,
) -> pd.Series:
    """A load series at sub-hourly resolution, as ENTSO-E often publishes it.

    `tz` exists so the DST tests can build the series the way entsoe-py hands it back
    — localised to Europe/Warsaw, with a missing hour in spring and a repeated one in
    autumn — and check that the conversion to UTC hourly survives it.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq=freq, tz="UTC", name="timestamp_utc")
    weather = synthetic_weather(index, rng)
    values = synthetic_load(index, weather["temp_c"].to_numpy(), rng)
    series = pd.Series(values, index=index, name="load_mw")
    return series if tz == "UTC" else series.tz_convert(tz)
