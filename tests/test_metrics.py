"""Metrics, checked against values computed by hand on a tiny fixture.

A metric function that is subtly wrong invalidates every comparison in the project
while looking entirely plausible, so none of these assertions are computed by the code
under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    bias,
    mae,
    mape,
    metrics_by_segment,
    pinball_loss,
    point_metrics,
    rmse,
    segment_labels,
)

WARSAW = "Europe/Warsaw"


@pytest.fixture
def tiny():
    """Four hours with errors of +100, -200, +300, -400 MW against a 20 GW level."""
    index = pd.date_range("2024-01-15 06:00", periods=4, freq="1h", tz="UTC")
    actual = pd.Series([20_000.0, 20_000.0, 10_000.0, 10_000.0], index=index)
    predicted = pd.Series([20_100.0, 19_800.0, 10_300.0, 9_600.0], index=index)
    return actual, predicted


def test_mape_is_the_mean_of_percentage_errors(tiny):
    actual, predicted = tiny

    # |100|/20000, |200|/20000, |300|/10000, |400|/10000 = 0.5%, 1%, 3%, 4% -> mean 2.125%
    assert mape(actual, predicted) == pytest.approx(2.125)


def test_mae_is_the_mean_absolute_error_in_mw(tiny):
    actual, predicted = tiny

    assert mae(actual, predicted) == pytest.approx((100 + 200 + 300 + 400) / 4)


def test_rmse_punishes_the_large_misses(tiny):
    actual, predicted = tiny

    expected = np.sqrt((100**2 + 200**2 + 300**2 + 400**2) / 4)
    assert rmse(actual, predicted) == pytest.approx(expected)
    assert rmse(actual, predicted) > mae(actual, predicted)


def test_bias_is_signed_and_cancels(tiny):
    actual, predicted = tiny

    # +100 -200 +300 -400 = -200 over four hours
    assert bias(actual, predicted) == pytest.approx(-50.0)


def test_a_perfect_forecast_scores_zero_everywhere(tiny):
    actual, _ = tiny

    assert point_metrics(actual, actual) == pytest.approx(
        {"mape": 0.0, "rmse_mw": 0.0, "mae_mw": 0.0, "bias_mw": 0.0}
    )


def test_pinball_loss_penalises_the_two_directions_differently():
    actual = pd.Series([100.0])
    too_low = pd.Series([90.0])
    too_high = pd.Series([110.0])

    # At q=0.9, being 10 under costs 0.9*10; being 10 over costs 0.1*10.
    assert pinball_loss(actual, too_low, 0.9) == pytest.approx(9.0)
    assert pinball_loss(actual, too_high, 0.9) == pytest.approx(1.0)
    # At the median the penalty is symmetric and is half the absolute error.
    assert pinball_loss(actual, too_low, 0.5) == pytest.approx(5.0)
    assert pinball_loss(actual, too_high, 0.5) == pytest.approx(5.0)


def test_pinball_loss_rejects_a_quantile_outside_the_open_unit_interval():
    actual = pd.Series([100.0])
    with pytest.raises(ValueError, match="quantile must be"):
        pinball_loss(actual, actual, 1.0)


def test_metrics_align_on_the_index_rather_than_on_position(tiny):
    actual, predicted = tiny
    shuffled = predicted.iloc[::-1]

    assert mae(actual, shuffled) == pytest.approx(mae(actual, predicted))


def test_missing_values_are_dropped_pairwise(tiny):
    actual, predicted = tiny
    with_gap = predicted.copy()
    with_gap.iloc[0] = np.nan

    # The remaining three errors are 200, 300, 400.
    assert mae(actual, with_gap) == pytest.approx(300.0)


def test_scoring_nothing_is_an_error(tiny):
    actual, _ = tiny
    empty = pd.Series(np.nan, index=actual.index)

    with pytest.raises(ValueError, match="No overlapping"):
        mae(actual, empty)


def test_segments_are_cut_on_the_local_clock():
    # 05:00 UTC is 06:00 in Warsaw (off-peak); 06:00 UTC is 07:00 (peak).
    index = pd.date_range("2024-01-15 05:00", periods=2, freq="1h", tz="UTC")

    labels = segment_labels(index, peak_hours=(7, 22))

    assert labels["period"].tolist() == ["off-peak", "peak"]
    assert labels["daytype"].tolist() == ["weekday", "weekday"]  # a Monday
    assert labels["season"].tolist() == ["winter", "winter"]


def test_holidays_and_the_christmas_week_are_their_own_segments():
    index = pd.date_range("2024-12-25 12:00", periods=1, freq="1h", tz="UTC")

    labels = segment_labels(index)

    assert labels["holiday"].iloc[0] == "holiday"
    assert labels["special"].iloc[0] == "christmas–new year"


def test_segment_breakdown_covers_every_model_and_every_segment(dataset):
    actual = dataset["load_mw"].iloc[:2000]
    predictions = pd.DataFrame(
        {"model_a": actual + 200.0, "pse": dataset["tso_forecast_mw"].iloc[:2000]}
    )

    table = metrics_by_segment(actual, predictions)

    assert set(table["segment_kind"]) == {"period", "daytype", "holiday", "season", "special"}
    assert set(table["model"]) == {"model_a", "pse"}
    assert (table["n"] > 0).all()
    # A constant +200 MW offset must show up as exactly that bias in every segment.
    assert table.loc[table["model"] == "model_a", "bias_mw"].round(6).eq(200.0).all()


def test_segment_row_counts_add_up_to_the_whole(dataset):
    actual = dataset["load_mw"].iloc[:2000]
    predictions = pd.DataFrame({"model_a": actual})

    table = metrics_by_segment(actual, predictions)

    for kind, group in table.groupby("segment_kind"):
        assert group["n"].sum() == len(actual), f"{kind} segments do not partition the data"
