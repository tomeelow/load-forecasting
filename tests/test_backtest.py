"""The rolling-origin backtest.

The load-bearing test here is the same guarantee as `test_no_leakage.py`, one level up:
a model scored on a test block must never have been fitted on data from that block or
after it. Everything else is about the report being an honest summary of what happened.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.backtest import (
    ACTUAL,
    CALENDAR_ONLY,
    TSO_FORECAST,
    WITH_WEATHER,
    BacktestResult,
    benchmark_markdown,
    run_backtest,
)
from src.models.baselines import LINEAR, NAIVE_SEASONAL

HORIZON = 24
FAST = {
    "initial_train_days": 365,
    "test_days": 14,
    "step_days": 14,
    "max_splits": 3,
    "inner_validation_days": 30,
    "tune": False,
    "num_boost_round": 40,
    "early_stopping_rounds": 10,
}


@pytest.fixture(scope="module")
def two_years():
    from src import synthetic

    return synthetic.make_dataset(start="2022-01-01", end="2023-09-01")


@pytest.fixture(scope="module")
def result(two_years) -> BacktestResult:
    return run_backtest(two_years, horizon=HORIZON, **FAST)


def test_no_origin_is_scored_on_data_it_was_trained_on(result):
    """The evaluation-level leakage guarantee, restated on the splits that actually ran."""
    for split in result.splits:
        assert split.train_end < split.test_start
        assert split.test_start - split.train_end == pd.Timedelta(hours=HORIZON)


def test_predictions_only_ever_come_from_test_blocks(result):
    """Nothing in the accumulated frame may fall inside any origin's training window."""
    covered = pd.DatetimeIndex([])
    for split in result.splits:
        covered = covered.union(
            pd.date_range(split.test_start, split.test_end, freq="1h", tz="UTC")
        )

    assert result.predictions.index.isin(covered).all()


def test_every_predicted_hour_precedes_no_training_data_used_for_it(result, two_years):
    """For each predicted hour, the model that produced it stopped learning `H` hours earlier."""
    for split in result.splits:
        block = result.predictions.loc[
            (result.predictions.index >= split.test_start)
            & (result.predictions.index <= split.test_end)
        ]
        if block.empty:
            continue
        earliest_prediction_moment = block.index.min() - pd.Timedelta(hours=HORIZON)
        assert split.train_end <= earliest_prediction_moment


def test_test_blocks_do_not_overlap_so_no_hour_is_scored_twice(result):
    assert not result.predictions.index.has_duplicates
    assert result.predictions.index.is_monotonic_increasing


def test_the_backtest_accumulates_more_than_one_origin(result):
    assert len(result.splits) == 3
    assert len(result.predictions) > 500


def test_every_model_and_baseline_is_scored_on_identical_rows(result):
    """A comparison on different hours is not a comparison."""
    assert result.predictions.notna().all().all()
    for column in (NAIVE_SEASONAL, TSO_FORECAST, LINEAR, CALENDAR_ONLY, WITH_WEATHER, ACTUAL):
        assert column in result.predictions.columns


def test_the_overall_table_reports_more_than_mape(result):
    table = result.overall()

    assert {"mape", "rmse_mw", "mae_mw", "bias_mw", "mape_vs_tso"} <= set(table.columns)
    assert table.loc[TSO_FORECAST, "mape_vs_tso"] == pytest.approx(0.0)
    # The references come first, so the table reads as "here is the bar, here is us".
    assert list(table.index)[:3] == [NAIVE_SEASONAL, LINEAR, TSO_FORECAST]


def test_the_model_beats_the_naive_baseline_out_of_sample(result):
    table = result.overall()

    assert table.loc[WITH_WEATHER, "mape"] < table.loc[NAIVE_SEASONAL, "mape"]


def test_the_segment_breakdown_covers_every_segment_and_every_model(result):
    segments = result.by_segment()

    assert set(segments["segment_kind"]) == {"period", "daytype", "holiday", "season", "special"}
    assert set(segments["model"]) >= {NAIVE_SEASONAL, TSO_FORECAST, WITH_WEATHER}
    assert (segments["n"] > 0).all()


def test_the_comparison_against_pse_is_ordered_worst_first(result):
    comparison = result.versus_tso()

    assert comparison["gap_vs_tso"].is_monotonic_decreasing
    assert set(comparison["verdict"]) <= {"model wins", "PSE wins"}
    # The gap is the model's MAPE minus PSE's, per segment.
    row = comparison.iloc[0]
    assert row["gap_vs_tso"] == pytest.approx(row[result.primary_model] - row[TSO_FORECAST])


def test_the_primary_model_is_the_best_variant_that_actually_ran(result):
    # This run had tuning switched off, so the tuned variant does not exist.
    assert result.primary_model == WITH_WEATHER


def test_the_benchmark_markdown_contains_the_table_and_the_losses(result):
    markdown = benchmark_markdown(result, synthetic=True, dataset_version="abc123")

    assert "| Model | MAPE | RMSE (MW) | MAE (MW) | Bias (MW) | vs PSE forecast |" in markdown
    assert "PSE day-ahead forecast" in markdown
    assert "*(the benchmark)*" in markdown
    assert "wins and loses against PSE" in markdown
    assert "abc123" in markdown


def test_synthetic_results_are_marked_as_such_in_the_report(result):
    synthetic_report = benchmark_markdown(result, synthetic=True, dataset_version="v1")
    real_report = benchmark_markdown(result, synthetic=False, dataset_version="v1")

    assert "SYNTHETIC data" in synthetic_report
    assert "SYNTHETIC data" not in real_report


def test_quantile_predictions_produce_pinball_losses(two_years):
    result = run_backtest(
        two_years, horizon=HORIZON, **{**FAST, "max_splits": 1, "quantiles": (0.1, 0.9)}
    )

    losses = result.pinball()

    assert set(losses) == {"pinball_p10", "pinball_p90"}
    assert all(v > 0 for v in losses.values())
    assert "Pinball loss" in benchmark_markdown(result, synthetic=True, dataset_version="v1")


def test_a_backtest_without_quantiles_reports_none(result):
    assert result.pinball() == {}
