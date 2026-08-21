"""Gate-closure-aligned evaluation: the horizons, the splits, and the weather swap.

The point of this module is that the PSE comparison is like-for-like, so the tests are
about *alignment* rather than accuracy: does each hour get the lead time PSE actually
had, do the DST days come out right, and does the forecast-weather variant genuinely
stop using observed values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.gate_closure import (
    DEFAULT_PUBLICATION_HOUR,
    gate_closure_horizons,
)
from src.evaluation.splits import rolling_origin_splits


def local_day(date: str, tz: str = "Europe/Warsaw") -> pd.DatetimeIndex:
    """Every local hour of one delivery day, as a UTC index.

    The two ends are localised separately rather than a day being added to the first,
    because on a DST day those are different things — and 23 or 25 hours is exactly what
    these tests are here to check.
    """
    start = pd.Timestamp(date).tz_localize(tz)
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize(tz)
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


def test_the_first_hour_of_the_day_has_the_shortest_lead():
    """00:00 local is 14 hours after a 10:00 publication on the day before."""
    horizons = gate_closure_horizons(local_day("2026-01-15"))

    assert horizons.iloc[0] == 24 - DEFAULT_PUBLICATION_HOUR
    assert horizons.iloc[0] == 14


def test_lead_time_grows_by_an_hour_through_the_delivery_day():
    horizons = gate_closure_horizons(local_day("2026-01-15"))

    assert list(horizons) == list(range(14, 38))


@pytest.mark.parametrize("date", ["2026-01-15", "2026-07-15"])
def test_the_lead_set_is_the_same_in_winter_and_summer(date):
    """The publication moment is local, so the UTC offset cancels out."""
    assert list(gate_closure_horizons(local_day(date))) == list(range(14, 38))


def test_the_spring_forward_day_is_an_hour_shorter():
    """Europe/Warsaw loses 02:00 on the last Sunday of March: 23 hours, ending at 36."""
    horizons = gate_closure_horizons(local_day("2026-03-29"))

    assert len(horizons) == 23
    assert horizons.max() == 36


def test_the_autumn_fall_back_day_is_an_hour_longer():
    """The repeated hour makes 25, and the last one is 38 hours after publication."""
    horizons = gate_closure_horizons(local_day("2026-10-25"))

    assert len(horizons) == 25
    assert horizons.max() == 38


def test_every_horizon_is_at_least_the_publication_gap():
    """No hour of any delivery day is knowable sooner than the deadline allows."""
    for month in range(1, 13):
        horizons = gate_closure_horizons(local_day(f"2026-{month:02d}-10"))
        assert horizons.min() >= 14


def test_a_naive_index_is_refused():
    naive = pd.date_range("2026-01-15", periods=24, freq="1h")

    with pytest.raises(ValueError, match="timezone-aware"):
        gate_closure_horizons(naive)


def test_an_earlier_publication_hour_lengthens_every_lead():
    early = gate_closure_horizons(local_day("2026-01-15"), publication_hour=8)
    deadline = gate_closure_horizons(local_day("2026-01-15"), publication_hour=10)

    assert (early - deadline == 2).all()


def test_the_flat_horizon_sits_inside_the_gate_closure_range():
    """The heart of the audit: a flat 24h model is neither uniformly harder nor easier.

    It has *less* information than PSE for the small hours and *more* for the evening,
    which is why the comparison had to be redone rather than argued about.
    """
    horizons = gate_closure_horizons(local_day("2026-01-15"))

    assert (horizons < 24).any(), "no hour where PSE knows more than a flat 24h model"
    assert (horizons > 24).any(), "no hour where a flat 24h model knows more than PSE"


class TestFirstTestStart:
    """`first_test_start` moves coverage without shrinking the history behind it."""

    @staticmethod
    def index() -> pd.DatetimeIndex:
        return pd.date_range("2020-01-01", "2024-01-01", freq="1h", tz="UTC")

    def test_coverage_begins_where_asked(self):
        wanted = pd.Timestamp("2023-06-01", tz="UTC")

        splits = rolling_origin_splits(
            self.index(),
            horizon=24,
            initial_train_days=730,
            test_days=14,
            step_days=14,
            max_splits=3,
            first_test_start=wanted,
        )

        assert splits[0].test_start == wanted

    def test_the_embargo_survives_the_move(self):
        splits = rolling_origin_splits(
            self.index(),
            horizon=24,
            initial_train_days=730,
            test_days=14,
            step_days=14,
            max_splits=2,
            first_test_start=pd.Timestamp("2023-06-01", tz="UTC"),
        )

        for split in splits:
            assert split.embargo == pd.Timedelta(hours=24)

    def test_training_still_starts_at_the_top_of_the_data(self):
        splits = rolling_origin_splits(
            self.index(),
            horizon=24,
            initial_train_days=730,
            test_days=14,
            step_days=14,
            max_splits=2,
            first_test_start=pd.Timestamp("2023-06-01", tz="UTC"),
        )

        assert splits[0].train_start == self.index().min()

    def test_asking_for_a_window_the_history_cannot_support_is_refused(self):
        with pytest.raises(ValueError, match="training days"):
            rolling_origin_splits(
                self.index(),
                horizon=24,
                initial_train_days=730,
                test_days=14,
                step_days=14,
                first_test_start=pd.Timestamp("2020-03-01", tz="UTC"),
            )

    def test_leaving_it_unset_keeps_the_old_behaviour(self):
        without = rolling_origin_splits(
            self.index(), horizon=24, initial_train_days=730, test_days=14, step_days=14
        )

        assert without[0].test_start == self.index().min() + pd.Timedelta(days=730, hours=23)
