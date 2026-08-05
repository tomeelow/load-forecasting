"""Forecast error metrics and the per-segment breakdown.

MAPE is the industry standard for load and it is not enough on its own: it hides
absolute magnitude and inflates at the overnight trough, where the denominator is
smallest. Every report here pairs it with RMSE and MAE in MW, and with the signed bias,
because a model that is 400 MW low every evening is a different problem from one that
is 400 MW out in both directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar_pl import DEFAULT_TZ, is_holiday

# Local clock hours counted as peak. Polish demand plateaus through the working day and
# peaks in the evening; the overnight trough is a genuinely different regime.
DEFAULT_PEAK_HOURS = (7, 22)

_SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def _aligned(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Line two series up on their index and drop pairs where either side is missing."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        frame = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    else:
        frame = pd.DataFrame({"true": np.asarray(y_true), "pred": np.asarray(y_pred)}).dropna()
    if frame.empty:
        raise ValueError("No overlapping non-null values to score")
    return frame["true"].to_numpy(), frame["pred"].to_numpy()


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error, in percent."""
    true, pred = _aligned(y_true, y_pred)
    nonzero = true != 0
    if not nonzero.all():
        true, pred = true[nonzero], pred[nonzero]
    return float(np.mean(np.abs((true - pred) / true)) * 100)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error, in MW. Punishes the large misses that cost money."""
    true, pred = _aligned(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error, in MW."""
    true, pred = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Signed mean error in MW: positive means the forecast runs high."""
    true, pred = _aligned(y_true, y_pred)
    return float(np.mean(pred - true))


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    """Pinball (quantile) loss in MW — the scoring rule a P10/P90 band is judged by.

    Under-predicting the q-quantile is penalised `q`, over-predicting `1 - q`, so a P90
    that is routinely too low is punished nine times harder than one that is too high.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    true, pred = _aligned(y_true, y_pred)
    error = true - pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def point_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """The four numbers every model in this project is reported with."""
    return {
        "mape": mape(y_true, y_pred),
        "rmse_mw": rmse(y_true, y_pred),
        "mae_mw": mae(y_true, y_pred),
        "bias_mw": bias(y_true, y_pred),
    }


def segment_labels(
    index: pd.DatetimeIndex,
    *,
    tz: str = DEFAULT_TZ,
    peak_hours: tuple[int, int] = DEFAULT_PEAK_HOURS,
) -> pd.DataFrame:
    """Label each target hour by the segments the evaluation breaks down over.

    Segments are cut on the *local* clock and calendar, because that is what makes them
    different from each other: an evening peak is an evening peak in Warsaw, not in UTC.
    """
    local = index.tz_convert(tz)
    start, end = peak_hours
    holiday = is_holiday(index, tz)

    christmas_week = ((local.month == 12) & (local.day >= 24)) | (
        (local.month == 1) & (local.day == 1)
    )

    return pd.DataFrame(
        {
            "period": np.where((local.hour >= start) & (local.hour < end), "peak", "off-peak"),
            "daytype": np.where(local.dayofweek >= 5, "weekend", "weekday"),
            "holiday": np.where(holiday, "holiday", "ordinary day"),
            "season": [_SEASONS[m] for m in local.month],
            "special": np.where(christmas_week, "christmas–new year", "rest of year"),
        },
        index=index,
    )


def metrics_by_segment(
    actual: pd.Series,
    predictions: pd.DataFrame,
    *,
    tz: str = DEFAULT_TZ,
    peak_hours: tuple[int, int] = DEFAULT_PEAK_HOURS,
) -> pd.DataFrame:
    """Long-form error table: one row per (segment kind, segment, model).

    Reported for every model including the PSE benchmark, so a segment where the model
    loses is as visible as one where it wins. Selecting flattering segments afterwards
    is a choice this shape makes obvious rather than easy.
    """
    labels = segment_labels(actual.index, tz=tz, peak_hours=peak_hours)
    rows = []
    for kind in labels.columns:
        for segment, group in labels.groupby(kind, observed=True):
            mask = actual.index.isin(group.index)
            for model in predictions.columns:
                rows.append(
                    {
                        "segment_kind": kind,
                        "segment": segment,
                        "model": model,
                        "n": int(mask.sum()),
                        **point_metrics(actual[mask], predictions.loc[mask, model]),
                    }
                )
    return pd.DataFrame(rows)
