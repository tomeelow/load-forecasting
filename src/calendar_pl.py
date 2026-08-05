"""Polish public-holiday flags.

Shared by ingestion and the feature builder so the flag is computed exactly once,
one way. A public holiday is a property of the *local* calendar date, not of a UTC
date: 2024-12-24 23:30 UTC is already 00:30 on Christmas Day in Warsaw.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

DEFAULT_TZ = "Europe/Warsaw"
_CALENDAR = holidays.country_holidays("PL")


def is_holiday(index: pd.DatetimeIndex, tz: str = DEFAULT_TZ) -> np.ndarray:
    """Boolean array: is each timestamp's local calendar date a Polish public holiday?

    The index must be timezone-aware — this project stores UTC and converts to `tz`
    only where the local calendar is semantically required, which is here.
    """
    if index.tz is None:
        raise ValueError("is_holiday needs a timezone-aware index; this project stores UTC")
    local_dates = index.tz_convert(tz).date
    return np.fromiter((d in _CALENDAR for d in local_dates), dtype=bool, count=len(index))
