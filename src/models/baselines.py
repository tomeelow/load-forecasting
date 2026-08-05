"""The three references every result in this project is reported against.

Built before the model, not after it. The naive seasonal is famously hard to beat; the
PSE forecast is the bar that actually matters, and it is free in the dataset already.
A LightGBM that cannot beat the first is broken, and one that approaches the second is
a real result.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from src.features.builder import DEFAULT_WEEKLY_LAG, feature_columns

NAIVE_SEASONAL = "naive_seasonal"
TSO_FORECAST = "pse_day_ahead"
LINEAR = "linear_calendar_weather"


def naive_seasonal(frame: pd.DataFrame, weekly_lag: int = DEFAULT_WEEKLY_LAG) -> pd.Series:
    """`load[T - 168]` — the same hour one week earlier, indexed by target hour.

    Safe at any day-ahead horizon: a week-old observation is older than any horizon
    this project serves, so this baseline never needs to know what the horizon is.
    """
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("naive_seasonal needs a sorted, unique hourly index")
    return frame["load_mw"].shift(weekly_lag).rename(NAIVE_SEASONAL)


def tso_forecast(frame: pd.DataFrame) -> pd.Series:
    """PSE's own day-ahead forecast, straight from the ingested column. The benchmark."""
    return frame["tso_forecast_mw"].rename(TSO_FORECAST)


class LinearBaseline:
    """Ordinary least squares on calendar and weather features — no lags.

    The plan's "simple statistical reference": it shows how much of Polish demand is
    explained by the clock and the thermometer alone, which is a useful thing to know
    before crediting a gradient booster for it.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearBaseline:
        self.columns = feature_columns(list(X.columns), include_weather=True, include_lags=False)
        self._model.fit(X[self.columns], y)
        logger.debug("Linear baseline fitted on {} columns, {} rows", len(self.columns), len(X))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.columns:
            raise RuntimeError("LinearBaseline.predict called before fit")
        return pd.Series(self._model.predict(X[self.columns]), index=X.index, name=LINEAR)
