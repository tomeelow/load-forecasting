"""Assembling the frame a forecast is made from — the serving half of the one feature path.

This module deliberately contains no feature logic. It stitches together the two things
serving has (recorded history and forecast weather) and hands them to the *same*
`make_features` training uses. A second feature implementation in the API layer is the
bug class this project is built to avoid, so the only thing here is data assembly.

**Why a request can return fewer hours than it asked for.** A row for target `T` needs
`load[T - horizon]`, and ENTSO-E publishes actuals two to three hours behind real time.
So the furthest target a 24-hour model can reach is roughly `now + 21h`, not `now + 24h`,
and it degrades further if ingestion has not run recently. Rows without their lags are
dropped rather than filled, and the caller is told which hours were lost — a forecast
invented from missing history would be worse than an absent one.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.calendar_pl import DEFAULT_TZ
from src.features.builder import make_features
from src.ingestion.dataset import WEATHER_COLUMNS, build_dataset
from src.ingestion.entsoe_client import FORECAST_COLUMN, LOAD_COLUMN


def inference_frame(
    history: pd.DataFrame, weather_forecast: pd.DataFrame, *, tz: str = DEFAULT_TZ
) -> pd.DataFrame:
    """Splice forecast weather onto recorded history on one continuous hourly index.

    Observed weather wins wherever both exist; the forecast only fills hours that have
    not happened yet. Load is left missing in the future, which is precisely the shape
    `make_features(include_target=False)` expects.
    """
    if history.empty:
        raise ValueError("Cannot build an inference frame without any history")
    for frame, name in ((history, "history"), (weather_forecast, "weather forecast")):
        if not frame.empty and frame.index.tz is None:
            raise ValueError(f"The {name} needs a timezone-aware UTC index")

    known_weather = history[[c for c in WEATHER_COLUMNS if c in history.columns]]
    weather = (
        known_weather.combine_first(weather_forecast)
        if not weather_forecast.empty
        else known_weather
    )
    load = history[[c for c in (LOAD_COLUMN, FORECAST_COLUMN) if c in history.columns]]
    return build_dataset(load, weather, run_id="serving", tz=tz)


def features_for_targets(
    history: pd.DataFrame,
    weather_forecast: pd.DataFrame,
    horizon: int,
    targets: pd.DatetimeIndex,
    *,
    rolling_window: int = 24,
    weekly_lag: int = 168,
    tz: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Feature rows for the requested target hours, built by the shared builder.

    Returns only the requested hours that could actually be built. The caller compares
    the result's index against `targets` to see what was dropped.
    """
    frame = inference_frame(history, weather_forecast, tz=tz)
    features = make_features(
        frame,
        horizon,
        rolling_window=rolling_window,
        weekly_lag=weekly_lag,
        include_target=False,
        tz=tz,
    )
    served = features.reindex(features.index.intersection(targets)).sort_index()

    missing = len(targets) - len(served)
    if missing:
        logger.warning(
            "{} of {} requested target hour(s) lack the history a {}h horizon needs; "
            "latest usable actual is {}",
            missing,
            len(targets),
            horizon,
            history[LOAD_COLUMN].last_valid_index() if LOAD_COLUMN in history else "unknown",
        )
    return served


def servable_until(history: pd.DataFrame, horizon: int) -> pd.Timestamp | None:
    """The furthest target hour the recorded history can support, or None if it cannot.

    Reported by `/health` so a stale dataset is visible before it silently shortens a
    forecast rather than after.
    """
    if LOAD_COLUMN not in history.columns:
        return None
    latest = history[LOAD_COLUMN].last_valid_index()
    return None if latest is None else latest + pd.Timedelta(hours=horizon)
