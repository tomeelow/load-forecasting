"""Splits, and the evaluation-level equivalent of the leakage test.

The feature builder guarantees a row never sees the future. This file guarantees the
*backtest* never does: no model may be fitted on a label that had not been published
when it made the prediction being scored.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.splits import chronological_split, rolling_origin_splits

HORIZONS = [1, 6, 24, 48]


@pytest.fixture
def year_index():
    return pd.date_range("2024-01-01", "2024-12-31 23:00", freq="1h", tz="UTC")


@pytest.mark.parametrize("horizon", HORIZONS)
def test_no_split_trains_on_a_label_published_after_the_prediction_moment(year_index, horizon):
    """The evaluation-level leakage guarantee.

    To predict target `T` the model stands at `T - horizon`, so every training label
    must belong to an hour at or before that moment.
    """
    splits = rolling_origin_splits(
        year_index, horizon=horizon, initial_train_days=90, test_days=7, step_days=14
    )

    assert splits
    for split in splits:
        earliest_prediction_moment = split.test_start - pd.Timedelta(hours=horizon)
        assert split.train_end <= earliest_prediction_moment
        # And, more bluntly: training never overlaps the period it is scored on.
        assert split.train_end < split.test_start
        train = split.train_index(year_index)
        test = split.test_index(year_index)
        assert len(train) and len(test)
        assert train.max() < test.min()


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_embargo_is_exactly_one_horizon(year_index, horizon):
    splits = rolling_origin_splits(
        year_index, horizon=horizon, initial_train_days=90, test_days=7, step_days=7
    )

    for split in splits:
        assert split.embargo == pd.Timedelta(hours=horizon)


def test_the_window_expands_rather_than_slides(year_index):
    splits = rolling_origin_splits(
        year_index, horizon=24, initial_train_days=90, test_days=7, step_days=7
    )

    assert all(s.train_start == year_index.min() for s in splits)
    train_lengths = [len(s.train_index(year_index)) for s in splits]
    assert train_lengths == sorted(train_lengths)
    assert train_lengths[-1] > train_lengths[0]


def test_origins_step_forward_by_the_configured_period(year_index):
    splits = rolling_origin_splits(
        year_index, horizon=24, initial_train_days=90, test_days=7, step_days=14
    )

    gaps = {splits[i + 1].train_end - splits[i].train_end for i in range(len(splits) - 1)}
    assert gaps == {pd.Timedelta(days=14)}


def test_test_blocks_together_cover_a_full_year(year_index):
    two_years = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="1h", tz="UTC")

    splits = rolling_origin_splits(
        two_years, horizon=24, initial_train_days=365, test_days=7, step_days=7
    )

    covered = splits[-1].test_end - splits[0].test_start
    assert covered >= pd.Timedelta(days=360), "a backtest must see every season"


def test_test_blocks_do_not_overlap_when_the_step_matches_the_block(year_index):
    splits = rolling_origin_splits(
        year_index, horizon=24, initial_train_days=90, test_days=7, step_days=7
    )

    for earlier, later in zip(splits, splits[1:], strict=False):
        assert earlier.test_end < later.test_start


def test_a_window_that_cannot_fit_is_an_error_not_an_empty_list(year_index):
    with pytest.raises(ValueError, match="No splits fit"):
        rolling_origin_splits(
            year_index, horizon=24, initial_train_days=400, test_days=7, step_days=7
        )


def test_max_splits_caps_the_work(year_index):
    splits = rolling_origin_splits(
        year_index, horizon=24, initial_train_days=90, test_days=7, step_days=7, max_splits=3
    )

    assert len(splits) == 3


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_validation_block_is_later_than_training_and_embargoed(year_index, horizon):
    train, validation = chronological_split(year_index, validation_days=30, horizon=horizon)

    assert train.max() < validation.min()
    assert validation.min() - train.max() == pd.Timedelta(hours=horizon)
    assert len(validation) == 30 * 24
    # The embargo costs exactly the horizon minus the boundary hour, which is legal to
    # train on: its label is known at the very moment the first prediction is made.
    assert len(train) + len(validation) == len(year_index) - (horizon - 1)


def test_a_validation_window_longer_than_the_data_is_rejected(year_index):
    with pytest.raises(ValueError, match="validation_days"):
        chronological_split(year_index, validation_days=400, horizon=24)


def test_an_empty_index_is_rejected():
    empty = pd.DatetimeIndex([], tz="UTC")

    with pytest.raises(ValueError, match="empty"):
        chronological_split(empty, validation_days=30, horizon=24)
    with pytest.raises(ValueError, match="empty"):
        rolling_origin_splits(empty, horizon=24, initial_train_days=1, test_days=1, step_days=1)
