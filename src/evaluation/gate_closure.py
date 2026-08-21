"""Evaluation aligned to PSE's gate closure, so the benchmark comparison is like-for-like.

**Why this exists.** The production model is a direct 24-hour model: for every target hour
`T` it stands at `T - 24` and uses nothing newer. PSE does not work that way. Its
day-ahead load forecast is published **once per day**, by 10:00 local on `D-1`, covering
all 24 hours of delivery day `D`. Its lead time therefore runs from about 14 hours (the
00:00 hour) to about 37 hours (the 23:00 hour) — short at the start of the day, long at
the end, and only ~24 hours on average by coincidence.

Comparing a rolling 24-hour model against that is not a comparison of forecast skill. For
every hour after mid-morning our model has load history PSE did not have when it
published, and for the small hours PSE has fresher information than we do.

**What this module does.** It reproduces PSE's product: for each target hour it computes
the horizon a forecaster standing at gate closure would have faced, and it scores that
hour with a direct model trained for exactly that horizon. `H(T) = 14 + local_hour(T)`,
give or take an hour at the DST transitions, which is why the horizon is derived from the
timestamps rather than written down.

The cost is 24 models instead of one. That is the honest price of the comparison, and it
is why this is a separate evaluation rather than a change to the served model — ADR-002
still holds, this simply instantiates it once per horizon the product needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.calendar_pl import DEFAULT_TZ
from src.evaluation.backtest import ACTUAL
from src.evaluation.splits import Split, chronological_split, rolling_origin_splits
from src.features.builder import TARGET_COLUMN, feature_columns, make_features
from src.models import lgbm
from src.models.baselines import NAIVE_SEASONAL, TSO_FORECAST
from src.models.baselines import naive_seasonal as naive_baseline
from src.models.baselines import tso_forecast as tso_baseline

MODEL = "lgbm_gate_closure"

# PSE's day-ahead load forecast must be published no later than two hours before
# day-ahead market gate closure, which is 12:00 local. The deadline, not the habit:
# PSE usually publishes earlier, so assuming 10:00 gives PSE the benefit of the doubt
# and this evaluation the harder side of the comparison.
DEFAULT_PUBLICATION_HOUR = 10


def gate_closure_horizons(
    targets: pd.DatetimeIndex,
    *,
    tz: str = DEFAULT_TZ,
    publication_hour: int = DEFAULT_PUBLICATION_HOUR,
) -> pd.Series:
    """Hours of lead time between PSE's publication deadline and each target hour.

    The publication moment for target `T` is `publication_hour` local on the day before
    `T`'s **local** date. Computed from the timestamps rather than as `14 + hour` so the
    two DST transitions each year come out right instead of being off by one.
    """
    if targets.tz is None:
        raise ValueError("gate_closure_horizons needs a timezone-aware UTC index")

    local = targets.tz_convert(tz)
    published_local = (local.normalize() - pd.Timedelta(days=1)) + pd.Timedelta(
        hours=publication_hour
    )
    lead = targets - published_local.tz_convert("UTC")
    return pd.Series((lead / pd.Timedelta(hours=1)).astype(int), index=targets, name="horizon")


@dataclass(frozen=True)
class GateClosureResult:
    """Accumulated gate-closure-aligned predictions, plus the references on the same rows."""

    predictions: pd.DataFrame  # index = target hour; `actual`, the model, and the references
    horizons: pd.Series  # the horizon each row was forecast at
    splits: list[Split]
    label: str = "observed weather"
    params: dict[str, object] = field(default_factory=dict)

    @property
    def actual(self) -> pd.Series:
        return self.predictions[ACTUAL]

    @property
    def coverage_days(self) -> int:
        return (self.predictions.index.max() - self.predictions.index.min()).days + 1


def _fit_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    split: Split,
    index: pd.DatetimeIndex,
    *,
    horizon: int,
    inner_validation_days: int,
    params: dict[str, object] | None,
    fit_kwargs: dict[str, object],
) -> pd.Series | None:
    """One origin, one horizon: fit on the training block and predict the test block."""
    train_index = split.train_index(index)
    test_index = split.test_index(index)
    if len(test_index) == 0 or len(train_index) == 0:
        return None

    inner_train, inner_val = chronological_split(
        train_index, validation_days=inner_validation_days, horizon=horizon
    )
    model = lgbm.train(
        X.loc[inner_train],
        y.loc[inner_train],
        X.loc[inner_val],
        y.loc[inner_val],
        params=params,
        **fit_kwargs,
    )
    return pd.Series(model.predict(X.loc[test_index]), index=test_index)


def run_gate_closure_backtest(
    frame: pd.DataFrame,
    *,
    initial_train_days: int,
    test_days: int,
    step_days: int,
    max_splits: int | None = None,
    first_test_start: pd.Timestamp | None = None,
    inner_validation_days: int = 45,
    params: dict[str, object] | None = None,
    tz: str = DEFAULT_TZ,
    publication_hour: int = DEFAULT_PUBLICATION_HOUR,
    seed: int = 42,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    rolling_window: int = 24,
    weekly_lag: int = 168,
    label: str = "observed weather",
) -> GateClosureResult:
    """Rolling-origin backtest where each hour is forecast at PSE's own lead time.

    Origins are computed **once**, on the most restrictive horizon's feature index, and
    reused for every horizon. Splits are timestamp-based, so every horizon predicts the
    same test blocks and the accumulated rows tile the coverage window exactly once.

    Hyperparameters are not searched here. Passing `params=None` trains every horizon on
    LightGBM defaults, which understates the model relative to the tuned production
    configuration — the conservative direction, and the one to be on when the result is
    the headline.
    """
    required = gate_closure_horizons(
        frame.index, tz=tz, publication_hour=publication_hour
    ).drop_duplicates()
    horizons = sorted({int(h) for h in required if h >= 1})
    logger.info("Gate-closure horizons in play: {}h to {}h", horizons[0], horizons[-1])

    # The largest horizon drops the most rows to missing lags, so its feature index is
    # the one every horizon can satisfy. Deriving the origins from it keeps the test
    # blocks identical across horizons.
    widest = make_features(
        frame, horizons[-1], rolling_window=rolling_window, weekly_lag=weekly_lag
    )
    splits = rolling_origin_splits(
        widest.index,
        horizon=horizons[-1],
        initial_train_days=initial_train_days,
        test_days=test_days,
        step_days=step_days,
        max_splits=max_splits,
        first_test_start=first_test_start,
    )

    fit_kwargs = {
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "seed": seed,
    }

    pieces: list[pd.Series] = []
    for number, horizon in enumerate(horizons, start=1):
        features = make_features(
            frame, horizon, rolling_window=rolling_window, weekly_lag=weekly_lag
        )
        columns = feature_columns(list(features.columns))
        X, y = features[columns], features[TARGET_COLUMN]

        for split in splits:
            predicted = _fit_and_predict(
                X,
                y,
                split,
                features.index,
                horizon=horizon,
                inner_validation_days=inner_validation_days,
                params=params,
                fit_kwargs=fit_kwargs,
            )
            if predicted is None:
                continue
            # Keep only the hours this horizon is the right one for. Every other hour in
            # the block is served by a different model, at its own lead time.
            wanted = gate_closure_horizons(
                predicted.index, tz=tz, publication_hour=publication_hour
            )
            pieces.append(predicted[wanted == horizon])

        logger.info("Horizon {}/{} done (H={}h)", number, len(horizons), horizon)

    combined = pd.concat(pieces).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    predictions = pd.DataFrame({ACTUAL: frame["load_mw"].reindex(combined.index)})
    predictions[MODEL] = combined
    predictions[NAIVE_SEASONAL] = naive_baseline(frame, weekly_lag).reindex(combined.index)
    predictions[TSO_FORECAST] = tso_baseline(frame).reindex(combined.index)
    predictions = predictions.dropna()

    logger.info(
        "Gate-closure backtest ({}): {} origins x {} horizons, {} scored hours from {} to {}",
        label,
        len(splits),
        len(horizons),
        len(predictions),
        predictions.index.min(),
        predictions.index.max(),
    )
    return GateClosureResult(
        predictions=predictions,
        horizons=gate_closure_horizons(
            predictions.index, tz=tz, publication_hour=publication_hour
        ),
        splits=splits,
        label=label,
        params=params or {},
    )
