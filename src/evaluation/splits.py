"""Chronological splits. Never a random one — that is the cardinal rule of Phase 5.

**The embargo.** Rows are indexed by target hour, and the label for target `T` only
becomes known at `T`. A model asked to predict target `T` is standing at `T - H`, so it
can only have been fitted on rows whose labels existed by then: `T_train <= T - H`.

Leaving training right up against the test block therefore leaks by exactly one
horizon — the model would be using labels that had not been published when it made the
prediction. Every split here inserts that `H`-hour gap. It costs a handful of rows and
buys the right to call the resulting numbers out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


class InsufficientHistoryError(ValueError):
    """There is not enough history to form both blocks of a chronological split.

    A `ValueError` subclass so existing callers and tests are unaffected, and its own
    type so the scheduled loop can tell "this deployment is too young to train yet"
    apart from "training broke". The first is a fact about a new deployment; the second
    is a failure.
    """


def minimum_training_hours(*, validation_days: int, horizon: int, weekly_lag: int) -> int:
    """Hours of raw history a chronological split needs before it yields both blocks.

    Three costs, in the order the data pays them: the feature builder drops the first
    `weekly_lag + horizon` rows for want of lags, the validation block claims
    `validation_days` of what is left, and the embargo takes `horizon` more between the
    two. Exactly this much leaves a single training row, so it is a floor to test
    against — "can this be split at all" — and never an amount to aim for.
    """
    return weekly_lag + horizon + validation_days * 24 + horizon


@dataclass(frozen=True)
class Split:
    """One train/test pair, with an embargo of `horizon` hours between them."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp  # inclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive
    horizon: int

    @property
    def embargo(self) -> pd.Timedelta:
        return self.test_start - self.train_end

    def train_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return index[(index >= self.train_start) & (index <= self.train_end)]

    def test_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return index[(index >= self.test_start) & (index <= self.test_end)]

    def __str__(self) -> str:
        return (
            f"train {self.train_start:%Y-%m-%d} -> {self.train_end:%Y-%m-%d} | "
            f"test {self.test_start:%Y-%m-%d} -> {self.test_end:%Y-%m-%d}"
        )


def chronological_split(
    index: pd.DatetimeIndex, *, validation_days: int, horizon: int
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split into an earlier training block and a later validation block.

    Used for early stopping and for Optuna. Validating on a *later* period than you
    train on is the only honest way to do it here; a random split lets the model
    interpolate between hours it has already seen and produces fantasy metrics.
    """
    if len(index) == 0:
        raise ValueError("Cannot split an empty index")

    validation_start = index.max() - pd.Timedelta(days=validation_days) + pd.Timedelta(hours=1)
    train_end = validation_start - pd.Timedelta(hours=horizon)

    train = index[index <= train_end]
    validation = index[index >= validation_start]
    if len(train) == 0 or len(validation) == 0:
        raise InsufficientHistoryError(
            f"validation_days={validation_days} leaves {len(train)} training and "
            f"{len(validation)} validation rows over {len(index)} available"
        )
    return train, validation


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    *,
    horizon: int,
    initial_train_days: int,
    test_days: int,
    step_days: int,
    max_splits: int | None = None,
    first_test_start: pd.Timestamp | None = None,
) -> list[Split]:
    """Expanding-window splits: the origin steps forward, training always starts at the top.

    Expanding rather than sliding, because more history is genuinely better here and a
    production retrain would use all of it. Each origin is a separate fit, so accumulated
    predictions across all splits are out-of-sample everywhere.

    `first_test_start` moves the first origin without shrinking the training data behind
    it. Left unset, coverage begins `initial_train_days` after the data does — which on a
    long series means the *oldest* year is the one reported. Setting it is how the
    coverage window becomes a choice rather than a side effect of where the data starts.
    """
    if len(index) == 0:
        raise ValueError("Cannot build splits from an empty index")
    for name, value in (
        ("initial_train_days", initial_train_days),
        ("test_days", test_days),
        ("step_days", step_days),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")

    start, last = index.min(), index.max()
    embargo = pd.Timedelta(hours=horizon)
    step = pd.Timedelta(days=step_days)
    test_span = pd.Timedelta(days=test_days) - pd.Timedelta(hours=1)

    splits: list[Split] = []
    train_end = start + pd.Timedelta(days=initial_train_days) - pd.Timedelta(hours=1)
    if first_test_start is not None:
        requested = pd.Timestamp(first_test_start) - embargo
        if requested < train_end:
            raise ValueError(
                f"first_test_start {first_test_start} leaves less than "
                f"{initial_train_days} training days after {start}"
            )
        train_end = requested
    while True:
        test_start = train_end + embargo
        test_end = min(test_start + test_span, last)
        if test_start > last or len(index[(index >= test_start) & (index <= test_end)]) == 0:
            break
        splits.append(Split(start, train_end, test_start, test_end, horizon))
        if max_splits is not None and len(splits) >= max_splits:
            break
        train_end = train_end + step

    if not splits:
        raise ValueError(
            f"No splits fit: {len(index)} rows spanning {(last - start).days} days cannot "
            f"hold {initial_train_days} training days plus a {test_days}-day test block"
        )

    logger.info(
        "Rolling origin: {} splits, test coverage {} -> {} ({} days), embargo {}h",
        len(splits),
        splits[0].test_start,
        splits[-1].test_end,
        (splits[-1].test_end - splits[0].test_start).days + 1,
        horizon,
    )
    return splits
