"""The promotion gate.

This function decides, unattended, whether a retrained model reaches production. Every
branch is tested, including the ones that should never happen — a gate that fails open
on a NaN is worse than no gate, because it looks like one.
"""

from __future__ import annotations

import math

import pytest

from src.models.promotion import promotion_decision

NAIVE = 4.00
CHAMPION = 2.00


def test_a_better_candidate_is_promoted():
    decision = promotion_decision(candidate=1.80, naive=NAIVE, champion=CHAMPION)

    assert decision.promote
    assert "beats naive" in decision.reason


def test_a_marginally_worse_candidate_is_still_promoted():
    """Within tolerance, so the champion is not frozen in place by ordinary noise."""
    decision = promotion_decision(
        candidate=2.03, naive=NAIVE, champion=CHAMPION, max_regression=1.02
    )

    assert decision.promote


def test_the_regression_tolerance_boundary_is_inclusive():
    exactly_at_ceiling = CHAMPION * 1.02

    assert promotion_decision(exactly_at_ceiling, NAIVE, CHAMPION, max_regression=1.02).promote
    assert not promotion_decision(
        exactly_at_ceiling + 1e-9, NAIVE, CHAMPION, max_regression=1.02
    ).promote


def test_a_materially_worse_candidate_is_held_back():
    decision = promotion_decision(candidate=2.50, naive=NAIVE, champion=CHAMPION)

    assert not decision.promote
    assert "regresses past" in decision.reason


def test_a_much_worse_candidate_is_held_back():
    decision = promotion_decision(candidate=3.90, naive=NAIVE, champion=CHAMPION)

    assert not decision.promote


def test_a_candidate_that_loses_to_the_naive_baseline_never_promotes():
    """Even with no champion in the way, and even if it would pass every other check."""
    assert not promotion_decision(candidate=4.10, naive=NAIVE, champion=None).promote
    assert not promotion_decision(candidate=4.10, naive=NAIVE, champion=99.0).promote


def test_matching_the_naive_baseline_is_not_beating_it():
    assert not promotion_decision(candidate=NAIVE, naive=NAIVE, champion=CHAMPION).promote


def test_the_first_model_promotes_with_no_champion_to_compare_against():
    decision = promotion_decision(candidate=3.00, naive=NAIVE, champion=None)

    assert decision.promote
    assert "no champion yet" in decision.reason


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None])
def test_a_non_finite_candidate_metric_fails_closed(bad):
    decision = promotion_decision(candidate=bad, naive=NAIVE, champion=CHAMPION)

    assert not decision.promote
    assert "not a finite number" in decision.reason


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_champion_or_naive_metric_fails_closed(bad):
    assert not promotion_decision(candidate=1.0, naive=bad, champion=CHAMPION).promote
    assert not promotion_decision(candidate=1.0, naive=NAIVE, champion=bad).promote


def test_a_tolerance_below_one_would_mean_demanding_improvement_and_is_rejected():
    with pytest.raises(ValueError, match="at least 1.0"):
        promotion_decision(candidate=1.0, naive=NAIVE, champion=CHAMPION, max_regression=0.98)


def test_a_zero_tolerance_setting_demands_no_regression_at_all():
    assert promotion_decision(1.99, NAIVE, CHAMPION, max_regression=1.0).promote
    assert not promotion_decision(2.01, NAIVE, CHAMPION, max_regression=1.0).promote


def test_the_decision_carries_the_numbers_it_judged():
    decision = promotion_decision(candidate=1.80, naive=NAIVE, champion=CHAMPION)

    assert (decision.candidate, decision.naive, decision.champion) == (1.80, NAIVE, CHAMPION)
    assert str(decision).startswith("PROMOTE")
    assert str(promotion_decision(9.0, NAIVE, CHAMPION)).startswith("HOLD")
