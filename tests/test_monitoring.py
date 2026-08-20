"""Drift monitoring: reference-window choice, bootstrapping, and wiring to action.

The two failure modes worth guarding are opposite. One is crying wolf — reporting drift
every autumn because the weather turned, which trains everyone to ignore the alert. The
other is crashing or emitting confident nonsense on day one, when nothing has been
served yet and there is nothing to measure.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import synthetic
from src.monitoring.drift import (
    INSUFFICIENT,
    OK,
    _drift_from_snapshot,
    _evidently_report,
    check_drift,
    latest_report,
    monitored_columns,
)
from src.monitoring.reference import SEASONAL, TRAILING, choose_reference, seasonal_reference
from src.prediction_log import PredictionLog, PredictionRecord

NOW = pd.Timestamp("2024-09-15 12:00", tz="UTC")


@pytest.fixture
def monitoring_cfg(cfg, tmp_path):
    return dataclasses.replace(
        cfg,
        monitoring=dataclasses.replace(cfg.monitoring, reports_dir=tmp_path / "monitoring"),
        state=dataclasses.replace(cfg.state, dir=tmp_path),
    )


@pytest.fixture
def log(tmp_path) -> PredictionLog:
    return PredictionLog(tmp_path / "predictions.db")


@pytest.fixture(scope="module")
def four_years():
    """Enough history for a seasonal reference to have somewhere to look."""
    return synthetic.make_dataset(start="2021-01-01", end="2024-10-01")


def test_only_features_that_can_actually_drift_are_monitored():
    columns = [
        "hour",
        "dow",
        "month",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "dow_cos",
        "temp_c",
        "temp_sq",
        "wind_ms",
        "cloud_cover",
        "load_lag_24",
        "load_roll_mean_24",
        "load_roll_std_24",
    ]

    monitored = monitored_columns(columns)

    assert "temp_c" in monitored
    assert "load_lag_24" in monitored
    # The clock does not drift, and testing it manufactures certain alarms.
    for deterministic in ("hour", "dow", "month", "is_weekend", "hour_sin"):
        assert deterministic not in monitored


def test_the_seasonal_reference_looks_at_the_same_weeks_in_earlier_years(four_years):
    window = seasonal_reference(
        four_years.index, NOW - pd.Timedelta(days=14), NOW, years_back=3, pad_days=7
    )

    assert window.strategy == SEASONAL
    assert window.rows > 0
    months = set(window.index.month)
    years = set(window.index.year)
    # September of previous years, and nothing from the current one.
    assert months <= {8, 9, 10}
    assert years == {2021, 2022, 2023}
    assert 2024 not in years


def test_a_trailing_reference_is_the_period_immediately_before(four_years):
    start = NOW - pd.Timedelta(days=14)

    window = choose_reference(
        four_years.index,
        start,
        NOW,
        strategy=TRAILING,
        years_back=3,
        pad_days=7,
        min_rows=10,
        fallback_to_trailing=True,
        current_days=14,
    )

    assert window.strategy == TRAILING
    assert window.index.max() < start


def test_year_one_falls_back_to_trailing_and_says_so(four_years):
    """No seasonal history yet is a fact to report, not a reason to crash."""
    first_year = four_years.loc["2021-01-01":"2021-06-01"]

    window = choose_reference(
        first_year.index,
        pd.Timestamp("2021-05-15", tz="UTC"),
        pd.Timestamp("2021-05-29", tz="UTC"),
        strategy=SEASONAL,
        years_back=3,
        pad_days=7,
        min_rows=500,
        fallback_to_trailing=True,
        current_days=14,
    )

    assert window.strategy == TRAILING
    assert window.rows > 0


@pytest.mark.parametrize(
    ("rows", "expected_method"),
    [(200, "p_value"), (3000, "distance")],
)
def test_the_drifted_column_named_is_the_one_that_actually_moved(rows, expected_method):
    """Evidently answers with a p-value on small samples and a distance on large ones.

    They point in opposite directions — drift is *below* a p-value threshold and *above*
    a distance threshold — and Evidently chooses between them by sample size rather than
    being told. Reading both the same way names the complement of the drifted columns,
    silently, and only on the larger samples where the answer matters.
    """
    rng = np.random.default_rng(0)
    current = pd.DataFrame({"unmoved": rng.normal(0, 1, rows), "moved": rng.normal(5, 1, rows)})
    reference = pd.DataFrame({"unmoved": rng.normal(0, 1, rows), "moved": rng.normal(0, 1, rows)})

    payload = _evidently_report(current, reference, None)
    drifted, share = _drift_from_snapshot(payload)

    method = next(
        m["config"]["method"]
        for m in payload["metrics"]
        if m["metric_name"].startswith("ValueDrift")
    )
    assert (expected_method == "p_value") == ("p_value" in method), f"Evidently used {method}"
    assert drifted == ["moved"]
    assert share == pytest.approx(0.5)


def test_the_named_columns_agree_with_evidentlys_own_count(four_years, monitoring_cfg, log):
    """The count comes from the library, the names from our parsing of the same payload."""
    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    assert result.drift_share == pytest.approx(len(result.drifted_features) / 11, abs=0.001)


def lag_features(result) -> set[str]:
    return {f for f in result.drifted_features if f.startswith("load")}


def trailing_config(monitoring_cfg):
    return dataclasses.replace(
        monitoring_cfg,
        monitoring=dataclasses.replace(
            monitoring_cfg.monitoring,
            reference=dataclasses.replace(monitoring_cfg.monitoring.reference, strategy=TRAILING),
        ),
    )


def test_a_seasonal_reference_reports_less_drift_than_a_trailing_one(
    monitoring_cfg, log, four_years
):
    """The comparison that justifies the seasonal choice, run explicitly.

    Late September against previous Septembers versus against late August: the load
    level in the trailing window has moved with the season, and every lag feature says
    so. Judging September by Septembers removes that, which is the whole decision in
    ADR-009.
    """
    seasonal = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)
    trailing = check_drift(
        trailing_config(monitoring_cfg), four_years, log, now=NOW, write_report=False
    )

    assert seasonal.reference_strategy == SEASONAL
    assert trailing.reference_strategy == TRAILING
    assert len(seasonal.drifted_features) < len(trailing.drifted_features)

    assert lag_features(seasonal) < lag_features(trailing), (
        "the seasonal reference should absorb the level shift"
    )


def test_a_fortnight_against_years_of_history_still_flags_the_weather(
    monitoring_cfg, log, four_years
):
    """A limitation this pins deliberately, not a property to be proud of.

    ADR-009 originally claimed no weather feature drifts against a seasonal reference.
    It does — on synthetic data and on the ingested series, at every date tried. The
    claim came from reading Evidently's distance metric as though it were a p-value,
    which named the complement of the drifted columns; see `_is_drifted`.

    The cause is the shape of the comparison, not the seasonality: fourteen days of one
    realised fortnight against a three-year mixture is a narrow distribution against a
    wide one, and a normed Wasserstein distance over 0.1 is nearly guaranteed. Input
    drift is therefore not specific enough to be the trigger on its own, which is why
    ADR-004 makes rolling performance the decisive signal. Fixing it means a like-sized
    reference or a different threshold — a monitoring design change, and an open
    question rather than a silent edit.
    """
    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    weather = {f for f in result.drifted_features if f.startswith(("temp", "wind", "cloud"))}
    assert weather, "if this now passes, the sensitivity was fixed — update ADR-009 with it"


def test_with_no_served_predictions_performance_is_not_assessed(monitoring_cfg, log, four_years):
    """Day one: nothing has been served, so there is nothing to measure."""
    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    assert result.scored_predictions == 0
    assert result.rolling_mape is None
    assert any("performance not assessed" in r for r in result.reasons)


def test_a_handful_of_predictions_is_still_not_enough(monitoring_cfg, log, four_years):
    """Three scored hours is not a MAPE, and reporting one would be worse than silence."""
    for offset in range(3):
        target = NOW - pd.Timedelta(hours=offset + 1)
        log.log(
            [
                PredictionRecord(
                    predicted_at=target - pd.Timedelta(hours=24),
                    target_time=target,
                    horizon_hours=24,
                    load_mw=20_000.0,
                    model_name="m",
                    model_version="1",
                )
            ]
        )
    log.score(four_years)

    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    assert 0 < result.scored_predictions < monitoring_cfg.monitoring.min_scored_predictions
    assert result.rolling_mape is None
    assert any("need 168" in r for r in result.reasons)


def test_too_little_recent_data_reports_insufficient_rather_than_failing(
    monitoring_cfg, log, four_years
):
    stale = four_years.loc[:"2024-01-01"]

    result = check_drift(monitoring_cfg, stale, log, now=NOW, write_report=False)

    assert result.status == INSUFFICIENT
    assert "ingest has not caught up" in result.reasons[0]
    assert not result.should_retrain


def test_a_bad_enough_rolling_error_triggers_a_retrain(monitoring_cfg, log, four_years):
    """The decisive signal: served predictions that are simply wrong."""
    actual = four_years["load_mw"]
    targets = actual.loc[NOW - pd.Timedelta(days=10) : NOW].index
    log.log(
        [
            PredictionRecord(
                predicted_at=target - pd.Timedelta(hours=24),
                target_time=target,
                horizon_hours=24,
                load_mw=float(actual.loc[target]) * 1.20,  # 20% high, every hour
                model_name="m",
                model_version="1",
            )
            for target in targets
        ]
    )
    log.score(four_years)

    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    assert result.scored_predictions >= monitoring_cfg.monitoring.min_scored_predictions
    assert result.rolling_mape == pytest.approx(20.0, abs=0.5)
    assert result.should_retrain
    assert any("rolling MAPE" in r for r in result.reasons)


def test_an_accurate_model_does_not_trigger_a_retrain(monitoring_cfg, log, four_years):
    actual = four_years["load_mw"]
    targets = actual.loc[NOW - pd.Timedelta(days=10) : NOW].index
    log.log(
        [
            PredictionRecord(
                predicted_at=target - pd.Timedelta(hours=24),
                target_time=target,
                horizon_hours=24,
                load_mw=float(actual.loc[target]) * 1.005,  # half a percent out
                model_name="m",
                model_version="1",
            )
            for target in targets
        ]
    )
    log.score(four_years)

    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    assert result.rolling_mape == pytest.approx(0.5, abs=0.2)
    assert not any("rolling MAPE" in r for r in result.reasons)


def test_the_benchmark_is_scored_on_the_same_hours(monitoring_cfg, log, four_years):
    actual = four_years["load_mw"]
    targets = actual.loc[NOW - pd.Timedelta(days=10) : NOW].index
    log.log(
        [
            PredictionRecord(
                predicted_at=target - pd.Timedelta(hours=24),
                target_time=target,
                horizon_hours=24,
                load_mw=float(actual.loc[target]) * 1.02,
                model_name="m",
                model_version="1",
            )
            for target in targets
        ]
    )
    log.score(four_years)

    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    assert result.tso_mape is not None
    assert 0 < result.tso_mape < 5


def test_a_dated_html_report_is_written_and_findable(monitoring_cfg, log, four_years):
    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=True)

    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report_path.name == "drift_20240915T120000.html"
    assert latest_report(monitoring_cfg.monitoring.reports_dir) == result.report_path


def test_no_report_directory_means_no_latest_report(tmp_path):
    assert latest_report(tmp_path / "nothing") is None


def test_the_summary_states_which_reference_was_used(monitoring_cfg, log, four_years):
    result = check_drift(monitoring_cfg, four_years, log, now=NOW, write_report=False)

    summary = result.summary()
    assert result.status.upper() in summary
    assert "reference" in summary
    assert result.reference_strategy in summary
    assert result.status in (OK, "drift")
