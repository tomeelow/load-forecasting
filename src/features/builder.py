"""The feature builder. One implementation, used by training and by serving.

**The indexing convention — read this before changing anything here.**

A row is indexed by the **target hour** `T`: the hour whose load is being predicted.
The prediction moment is therefore `T - H`, `H` hours earlier. So for the row at `T`:

    target        = load[T]
    load_lag_L    = load[T - L]        for every L in `lag_hours(H)`, and min(L) == H
    load_roll_*_W = stats over load[T - H - W + 1 .. T - H]
    weather       = weather at T       (the hour being predicted)
    calendar      = local calendar of T

**The leakage rule.** At the prediction moment `T - H` the newest load anyone can know
is `load[T - H]`. Every load-derived feature is therefore at least `H` hours old
*relative to the target*, which is exactly — not merely at least — what the horizon
requires. The lag set is derived from `horizon` rather than written down; a hardcoded
24 is correct once and silently wrong everywhere else. `tests/test_no_leakage.py`
re-checks this structurally, behaviourally and arithmetically across several horizons.

The alternative convention — indexing rows by prediction moment — was what this module
used first, and it quietly cost a factor of two: a lag of `H` relative to the row was
then `2H` relative to the target, discarding a full horizon of the most recent and most
informative history for no safety benefit.

The 168-hour lag (same hour, one week ago) is safe at any day-ahead horizon and is
the one feature that survives no matter how the horizon moves.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from loguru import logger

from src.calendar_pl import DEFAULT_TZ, is_holiday

TARGET_COLUMN = "target"
LAG_COLUMN_PATTERN = re.compile(r"^load_lag_(\d+)$")
ROLLING_COLUMN_PATTERN = re.compile(r"^load_roll_(mean|std)_(\d+)$")

DEFAULT_ROLLING_WINDOW = 24
DEFAULT_WEEKLY_LAG = 168

REQUIRED_COLUMNS = ("load_mw", "temp_c", "wind_ms", "cloud_cover")


def lag_hours(horizon: int, weekly_lag: int = DEFAULT_WEEKLY_LAG) -> list[int]:
    """The lag set for a horizon: recent-but-legal, a day earlier, and last week.

    Anything younger than the horizon is dropped rather than clipped — a lag shorter
    than the horizon is a bug, not a tuning choice, and this is where that rule is
    enforced instead of hoped for. (It bites for horizons beyond a week, where the
    weekly lag itself would be in the future.)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 hour, got {horizon}")

    candidates = {horizon, horizon + 1, horizon + 24, weekly_lag, weekly_lag + horizon}
    safe = sorted(lag for lag in candidates if lag >= horizon)
    dropped = sorted(candidates - set(safe))
    if dropped:
        logger.warning("Dropped lag(s) {} — younger than horizon {}h", dropped, horizon)
    return safe


def _hourly_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` on a complete, sorted, unique UTC hourly index.

    This is a leakage guard, not tidiness. Every lag below is a positional `shift`, so
    a missing row would make `shift(24)` reach back fewer than 24 hours and quietly
    pull in data newer than the horizon allows.
    """
    if df.index.tz is None:
        raise ValueError("make_features needs a timezone-aware UTC index")
    if df.index.has_duplicates:
        raise ValueError("make_features needs a unique index; deduplicate before building features")

    df = df.sort_index()
    complete = pd.date_range(
        df.index.min(), df.index.max(), freq="1h", tz="UTC", name=df.index.name
    )
    if len(complete) != len(df.index) or not complete.equals(df.index):
        logger.warning(
            "Input index is not complete hourly ({} rows over {} hours); reindexing so "
            "positional shifts stay time-accurate",
            len(df),
            len(complete),
        )
        df = df.reindex(complete)
    return df


def make_features(
    df: pd.DataFrame,
    horizon: int,
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    weekly_lag: int = DEFAULT_WEEKLY_LAG,
    include_target: bool = True,
    tz: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Build the feature frame for a given horizon.

    `df` is UTC-hourly with at least `load_mw`, `temp_c`, `wind_ms` and `cloud_cover`.
    The row indexed `T` predicts `load_mw[T]` from the standpoint of `T - horizon`, and
    uses no load newer than `load_mw[T - horizon]`. See the module docstring for the
    full convention.

    Set `include_target=False` at serving time, where the target does not exist yet.
    With load known up to `L`, rows out to `L + horizon` survive — precisely the hours
    a forecast run is asked for.

    Calendar fields are derived from the **Europe/Warsaw local** timestamp, not from
    UTC. Demand follows human behaviour and humans follow the local clock: the morning
    ramp is at 07:00 in Warsaw all year, which is 06:00 UTC in winter and 05:00 UTC in
    summer. Deriving `hour` from UTC would smear that ramp across two values and make
    the model relearn the DST offset twice a year. The index itself stays UTC.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"make_features is missing required columns: {missing}")

    df = _hourly_frame(df)
    f = pd.DataFrame(index=df.index)

    # --- Calendar: known arbitrarily far in advance, so no leakage risk at all ---
    # Computed on the local clock people actually live by, not on UTC.
    local = df.index.tz_convert(tz)
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["month"] = local.month
    f["is_weekend"] = (local.dayofweek >= 5).astype(int)
    f["is_holiday"] = is_holiday(df.index, tz).astype(int)

    # Cyclical encoding, so the model sees hour 23 as adjacent to hour 0.
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    f["dow_sin"] = np.sin(2 * np.pi * f["dow"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * f["dow"] / 7)

    # --- Lagged load: each one `lag` hours before the target, and lag >= horizon ---
    for lag in lag_hours(horizon, weekly_lag):
        f[f"load_lag_{lag}"] = df["load_mw"].shift(lag)

    # Rolling stats over a series already `horizon` hours behind the target, so the
    # freshest observation inside the window is the newest one the horizon allows.
    base = df["load_mw"].shift(horizon)
    f[f"load_roll_mean_{rolling_window}"] = base.rolling(rolling_window).mean()
    f[f"load_roll_std_{rolling_window}"] = base.rolling(rolling_window).std()

    # --- Weather at the target hour ---
    # Unshifted, so these are the conditions during the hour being forecast rather than
    # during the hour the forecast is made. That looks like leakage and is not: at
    # serving time these values come from the Open-Meteo *forecast* endpoint, which
    # publishes the weather for T well before T happens. The load series gets no such
    # treatment — nobody publishes tomorrow's demand in advance, which is the whole
    # reason this project exists. In training these are observed values while at serving
    # they are forecast values; see the README on that skew.
    f["temp_c"] = df["temp_c"]
    f["temp_sq"] = df["temp_c"] ** 2
    f["wind_ms"] = df["wind_ms"]
    f["cloud_cover"] = df["cloud_cover"]

    if include_target:
        f[TARGET_COLUMN] = df["load_mw"]

    built = f.dropna()
    logger.debug(
        "Features h={}: {} rows x {} columns ({} dropped for incomplete history)",
        horizon,
        len(built),
        built.shape[1],
        len(f) - len(built),
    )
    return built


def target_derived_columns(columns: list[str]) -> list[str]:
    """Feature columns computed from the load series — the ones leakage can hide in."""
    return [
        c
        for c in columns
        if LAG_COLUMN_PATTERN.match(c) or ROLLING_COLUMN_PATTERN.match(c) or c == TARGET_COLUMN
    ]
