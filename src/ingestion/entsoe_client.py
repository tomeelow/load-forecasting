"""ENTSO-E Transparency Platform: actual total load and PSE's day-ahead forecast.

Both series are pulled together, always. The TSO forecast is the benchmark this
project is measured against — a dataset carrying only the actuals is incomplete.

Everything leaves this module on a UTC hourly index. entsoe-py returns data in the
area's local timezone (Europe/Warsaw for PL) and often at 15-minute resolution, so
the conversion and the resampling are explicit and tested rather than implied.
"""

from __future__ import annotations

import os
from typing import Protocol

import pandas as pd
import requests
from entsoe.exceptions import NoMatchingDataError
from loguru import logger
from requests.adapters import HTTPAdapter, Retry

LOAD_COLUMN = "load_mw"
FORECAST_COLUMN = "tso_forecast_mw"

# entsoe-py does not time out by default, and a request that hangs forever hangs the
# scheduled run with it. It does retry — but only `ConnectionError`, `gaierror` and
# `RemoteDisconnected` (see entsoe/decorators.py), and *not* a read timeout, which is
# precisely what the Transparency Platform produces when it is busy and the caller is
# asking for a year of 15-minute data. A full rebuild in CI died on exactly that.
#
# So: a bounded timeout, and retries mounted on the session, where they apply to each
# HTTP request entsoe-py makes rather than to the whole multi-year call. Retrying the
# call would re-fetch every year to recover one.
REQUEST_TIMEOUT_S = 120
RETRY_COUNT = 3
RETRY_BACKOFF_S = 5  # urllib3 doubles this per attempt: ~0s, 10s, 20s
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def _retrying_session() -> requests.Session:
    """A session that retries the transient failures entsoe-py's own decorator misses."""
    session = requests.Session()
    retries = Retry(
        total=RETRY_COUNT,
        read=RETRY_COUNT,
        connect=RETRY_COUNT,
        status=RETRY_COUNT,
        backoff_factor=RETRY_BACKOFF_S,
        status_forcelist=RETRYABLE_STATUS,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# Column names entsoe-py gives the parsed series, preferred over positional access.
_ACTUAL_LOAD_LABEL = "Actual Load"
_FORECAST_LOAD_LABEL = "Forecasted Load"


class LoadQueryClient(Protocol):
    """The slice of `entsoe.EntsoePandasClient` this module actually uses."""

    def query_load(self, country_code: str, start: pd.Timestamp, end: pd.Timestamp): ...

    def query_load_forecast(self, country_code: str, start: pd.Timestamp, end: pd.Timestamp): ...


def make_client(api_key: str | None = None) -> LoadQueryClient:
    """Build an ENTSO-E client, reading the security token from the environment."""
    from entsoe import EntsoePandasClient

    key = api_key or os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError(
            "ENTSOE_API_KEY is not set. Register at https://transparency.entsoe.eu/ and "
            "email transparency@entsoe.eu with subject 'Restful API access'; access is "
            "granted in ~3 working days. Then copy .env.example to .env and paste the token."
        )
    return EntsoePandasClient(
        api_key=key,
        session=_retrying_session(),
        timeout=REQUEST_TIMEOUT_S,
    )


def to_utc_hourly(series: pd.Series, aggregation: str = "mean") -> pd.Series:
    """Convert a timezone-aware series to a UTC hourly index.

    Resampling happens on the UTC index, never the local one: UTC has no DST, so the
    spring-forward and autumn-back transitions are ordinary hours here. `aggregation`
    is `mean` for load — the values are instantaneous MW, so the mean over the hour is
    the hourly average power; summing 15-minute samples would inflate it fourfold.

    Duplicate timestamps (ENTSO-E revisions, or an autumn fold in a local-time index
    that was localised ambiguously) resolve to the last value seen.
    """
    if series.index.tz is None:
        raise ValueError(
            "to_utc_hourly needs a timezone-aware index; a naive index cannot be "
            "disambiguated across a DST transition"
        )

    s = series.copy()
    s.index = s.index.tz_convert("UTC")
    s = s.sort_index()

    duplicated = s.index.duplicated(keep="last")
    if duplicated.any():
        logger.warning(
            "{} duplicate timestamps in '{}' resolved to the last value",
            int(duplicated.sum()),
            series.name,
        )
        s = s[~duplicated]

    hourly = s.resample("1h").agg(aggregation)
    logger.debug(
        "'{}': {} rows at source resolution -> {} hourly rows ({})",
        series.name,
        len(series),
        len(hourly),
        aggregation,
    )
    return hourly


def _as_series(obj: pd.DataFrame | pd.Series, preferred: str, name: str) -> pd.Series:
    """Pick the value column out of whatever entsoe-py returned."""
    if isinstance(obj, pd.Series):
        return obj.rename(name)
    if preferred in obj.columns:
        return obj[preferred].rename(name)
    logger.warning(
        "Column '{}' not in ENTSO-E response {}; using the first column",
        preferred,
        list(obj.columns),
    )
    return obj.iloc[:, 0].rename(name)


def _query(
    call, country_code: str, start: pd.Timestamp, end: pd.Timestamp, label: str
) -> pd.DataFrame | pd.Series:
    try:
        return call(country_code, start=start, end=end)
    except NoMatchingDataError:
        logger.warning("ENTSO-E has no {} for {} in [{}, {})", label, country_code, start, end)
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))


def fetch_actual_load(
    client: LoadQueryClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    aggregation: str = "mean",
) -> pd.Series:
    """Actual total load in MW on a UTC hourly index. This is the target."""
    raw = _query(client.query_load, country_code, start, end, "actual load")
    return to_utc_hourly(_as_series(raw, _ACTUAL_LOAD_LABEL, LOAD_COLUMN), aggregation)


def fetch_tso_forecast(
    client: LoadQueryClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    aggregation: str = "mean",
) -> pd.Series:
    """PSE's official day-ahead load forecast in MW on a UTC hourly index. The benchmark."""
    raw = _query(client.query_load_forecast, country_code, start, end, "day-ahead forecast")
    return to_utc_hourly(_as_series(raw, _FORECAST_LOAD_LABEL, FORECAST_COLUMN), aggregation)


def fetch_load_frame(
    client: LoadQueryClient,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Actual load and the TSO day-ahead forecast, joined on one UTC hourly index."""
    load = fetch_actual_load(client, country_code, start, end, aggregation)
    forecast = fetch_tso_forecast(client, country_code, start, end, aggregation)
    frame = pd.concat([load, forecast], axis=1).sort_index()
    logger.info(
        "ENTSO-E {}: {} hourly rows, {} with load, {} with TSO forecast",
        country_code,
        len(frame),
        int(frame[LOAD_COLUMN].notna().sum()),
        int(frame[FORECAST_COLUMN].notna().sum()),
    )
    return frame


def trailing_window(
    trailing_days: int, now: pd.Timestamp | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The re-pull window for an incremental ingestion run.

    ENTSO-E revises published actuals and some hours arrive late, so an incremental
    run re-pulls the last `trailing_days` days rather than trusting that yesterday is
    final. The end is pushed a day out so tomorrow's day-ahead forecast, which is
    published in the afternoon, is picked up as soon as it exists.
    """
    if trailing_days < 1:
        raise ValueError("trailing_days must be at least 1")
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    start = (now - pd.Timedelta(days=trailing_days)).floor("D")
    end = (now + pd.Timedelta(days=1)).ceil("D")
    return start, end
