"""The three pipelines that close the loop, run offline against stub sources.

What matters here is the wiring between them: drift raises a flag, retraining consumes
it, and a candidate that loses the gate leaves production alone.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from pipelines import check_drift, evaluate, retrain_if_needed
from src import synthetic
from src.evaluation.splits import InsufficientHistoryError
from src.ingestion.dataset import write_dataset
from src.pipeline_state import PipelineState
from src.prediction_log import PredictionLog, PredictionRecord

NOW = pd.Timestamp("2024-09-15 12:00", tz="UTC")


@pytest.fixture(scope="module")
def dataset():
    return synthetic.make_dataset(start="2021-01-01", end="2024-10-01")


@pytest.fixture
def loop_cfg(cfg, tmp_path, dataset):
    write_dataset(dataset, tmp_path / "dataset.parquet")
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, processed_dir=tmp_path, dataset_filename="dataset.parquet"
        ),
        state=dataclasses.replace(cfg.state, dir=tmp_path),
        monitoring=dataclasses.replace(cfg.monitoring, reports_dir=tmp_path / "monitoring"),
    )


def log_predictions(log: PredictionLog, dataset: pd.DataFrame, *, error: float, days: int = 10):
    actual = dataset["load_mw"]
    targets = actual.loc[NOW - pd.Timedelta(days=days) : NOW].index
    log.log(
        [
            PredictionRecord(
                predicted_at=target - pd.Timedelta(hours=24),
                target_time=target,
                horizon_hours=24,
                load_mw=float(actual.loc[target]) * (1 + error),
                model_name="pl_load_lgbm",
                model_version="1",
            )
            for target in targets
        ]
    )


def test_evaluate_reports_insufficient_data_before_anything_was_served(loop_cfg, tmp_path):
    log = PredictionLog(tmp_path / "p.db")

    summary = evaluate.run(loop_cfg, log=log, now=NOW)

    assert not summary.sufficient
    assert summary.scored == 0
    assert summary.model_mape is None
    assert "insufficient data" in summary.format()


def test_evaluate_scores_served_predictions_against_pse_on_the_same_hours(
    loop_cfg, tmp_path, dataset
):
    log = PredictionLog(tmp_path / "p.db")
    log_predictions(log, dataset, error=0.02)

    summary = evaluate.run(loop_cfg, log=log, now=NOW)

    assert summary.sufficient
    assert summary.model_mape == pytest.approx(2.0, abs=0.2)
    assert summary.tso_mape is not None
    assert summary.versus_tso == pytest.approx(summary.model_mape - summary.tso_mape)
    assert "PSE day-ahead" in summary.format()


def test_evaluate_rescores_when_an_actual_is_revised(loop_cfg, tmp_path, dataset):
    log = PredictionLog(tmp_path / "p.db")
    log_predictions(log, dataset, error=0.02)
    first = evaluate.run(loop_cfg, log=log, now=NOW)

    revised = dataset.copy()
    revised["load_mw"] = revised["load_mw"] * 1.05
    write_dataset(revised, loop_cfg.data.dataset_path)
    second = evaluate.run(loop_cfg, log=log, now=NOW)

    assert second.model_mape != pytest.approx(first.model_mape)


def test_evaluate_records_its_marker(loop_cfg, tmp_path):
    state = PipelineState(tmp_path / "state.json")

    evaluate.run(loop_cfg, log=PredictionLog(tmp_path / "p.db"), state=state, now=NOW)

    assert state.last_success(evaluate.PIPELINE) is not None


def test_a_bad_rolling_error_raises_the_flag_retraining_consumes(loop_cfg, tmp_path, dataset):
    """The wiring that makes monitoring more than a report nobody reads."""
    log = PredictionLog(tmp_path / "p.db")
    log_predictions(log, dataset, error=0.20)
    log.score(dataset)
    state = PipelineState(tmp_path / "state.json")

    result = check_drift.run(loop_cfg, log=log, state=state, now=NOW, write_report=False)

    assert result.should_retrain
    assert state.retrain_flag() is not None
    assert "rolling MAPE" in state.retrain_flag().reason


def test_an_accurate_model_leaves_the_flag_alone(loop_cfg, tmp_path, dataset):
    log = PredictionLog(tmp_path / "p.db")
    log_predictions(log, dataset, error=0.005)
    log.score(dataset)
    state = PipelineState(tmp_path / "state.json")

    check_drift.run(loop_cfg, log=log, state=state, now=NOW, write_report=False)

    flag = state.retrain_flag()
    assert flag is None or "rolling MAPE" not in flag.reason


def test_check_drift_exits_zero_even_when_it_finds_drift(loop_cfg, monkeypatch, tmp_path, dataset):
    """Drift is a finding. Failing the step here would stop the retrain that answers it."""
    log = PredictionLog(tmp_path / "p.db")
    log_predictions(log, dataset, error=0.20)
    log.score(dataset)
    monkeypatch.setattr(check_drift, "load_config", lambda _: loop_cfg)

    assert check_drift.main(["--no-report"]) == 0


def test_retraining_is_skipped_when_neither_trigger_fires(loop_cfg, tmp_path):
    state = PipelineState(tmp_path / "state.json")
    state.record_success(retrain_if_needed.PIPELINE, NOW - pd.Timedelta(days=1))

    outcome = retrain_if_needed.run(loop_cfg, state=state, now=NOW)

    assert outcome.status == retrain_if_needed.SKIPPED
    assert "cadence is 7 days" in outcome.reason


def test_the_drift_flag_triggers_a_retrain_even_within_the_cadence(loop_cfg, tmp_path, monkeypatch):
    state = PipelineState(tmp_path / "state.json")
    state.record_success(retrain_if_needed.PIPELINE, NOW - pd.Timedelta(days=1))
    state.raise_retrain_flag("rolling MAPE 6.2% over 4.0%")

    trained = {}

    def fake_train(cfg, horizon, *, tune):
        trained["horizon"] = horizon
        return _TrainResult(promoted=True, gate_reason="beats naive, no regression")

    monkeypatch.setattr(retrain_if_needed, "train_horizon", fake_train)
    monkeypatch.setattr(retrain_if_needed, "configure", lambda _: None)

    outcome = retrain_if_needed.run(loop_cfg, state=state, now=NOW)

    assert outcome.status == retrain_if_needed.PROMOTED
    assert trained["horizon"] == loop_cfg.model.horizons[0]
    assert "rolling MAPE 6.2%" in outcome.reason


def test_a_candidate_that_fails_the_gate_leaves_production_untouched(
    loop_cfg, tmp_path, monkeypatch
):
    """The whole reason unattended retraining is safe to leave running."""
    state = PipelineState(tmp_path / "state.json")
    state.raise_retrain_flag("data drift")

    monkeypatch.setattr(
        retrain_if_needed,
        "train_horizon",
        lambda cfg, horizon, *, tune: _TrainResult(
            promoted=False, gate_reason="candidate 3.9 regresses past the champion's 2.1"
        ),
    )
    monkeypatch.setattr(retrain_if_needed, "configure", lambda _: None)

    outcome = retrain_if_needed.run(loop_cfg, state=state, now=NOW)

    assert outcome.status == retrain_if_needed.TRAINED_NOT_PROMOTED
    assert "regresses past" in outcome.gate_reason


def test_the_flag_is_cleared_once_a_retrain_has_been_attempted(loop_cfg, tmp_path, monkeypatch):
    """Otherwise the same losing candidate is retrained every single night."""
    state = PipelineState(tmp_path / "state.json")
    state.raise_retrain_flag("data drift")
    monkeypatch.setattr(
        retrain_if_needed,
        "train_horizon",
        lambda cfg, horizon, *, tune: _TrainResult(promoted=False, gate_reason="lost"),
    )
    monkeypatch.setattr(retrain_if_needed, "configure", lambda _: None)

    retrain_if_needed.run(loop_cfg, state=state, now=NOW)

    assert state.retrain_flag() is None


def test_forcing_a_retrain_ignores_the_schedule(loop_cfg, tmp_path, monkeypatch):
    state = PipelineState(tmp_path / "state.json")
    state.record_success(retrain_if_needed.PIPELINE, NOW - pd.Timedelta(hours=1))
    monkeypatch.setattr(
        retrain_if_needed,
        "train_horizon",
        lambda cfg, horizon, *, tune: _TrainResult(promoted=True, gate_reason="ok"),
    )
    monkeypatch.setattr(retrain_if_needed, "configure", lambda _: None)

    outcome = retrain_if_needed.run(loop_cfg, state=state, now=NOW, force=True)

    assert outcome.status == retrain_if_needed.PROMOTED
    assert "forced" in outcome.reason


def test_the_cadence_triggers_a_retrain_without_any_drift(loop_cfg, tmp_path, monkeypatch):
    state = PipelineState(tmp_path / "state.json")
    state.record_success(retrain_if_needed.PIPELINE, NOW - pd.Timedelta(days=9))
    monkeypatch.setattr(
        retrain_if_needed,
        "train_horizon",
        lambda cfg, horizon, *, tune: _TrainResult(promoted=True, gate_reason="ok"),
    )
    monkeypatch.setattr(retrain_if_needed, "configure", lambda _: None)

    outcome = retrain_if_needed.run(loop_cfg, state=state, now=NOW)

    assert outcome.status == retrain_if_needed.PROMOTED
    assert "cadence elapsed" in outcome.reason


def test_a_deployment_too_young_to_train_reports_it_instead_of_failing(
    loop_cfg, tmp_path, monkeypatch
):
    """Three weeks of scheduled runs died here, one traceback a night.

    A fresh deployment has weeks of data and a 60-day validation split; that is a fact
    about the data, not a broken pipeline. Failing the job over it turns the loop red
    every morning until someone stops reading it — and the steps that carry state out
    of the runner sit behind that failure.
    """
    state = PipelineState(tmp_path / "state.json")
    state.raise_retrain_flag("input drift")

    def too_young(cfg, horizon, *, tune):
        raise InsufficientHistoryError(
            "validation_days=60 leaves 0 training and 414 validation rows over 414 available"
        )

    monkeypatch.setattr(retrain_if_needed, "train_horizon", too_young)
    monkeypatch.setattr(retrain_if_needed, "configure", lambda _: None)

    outcome = retrain_if_needed.run(loop_cfg, state=state, now=NOW)

    assert outcome.status == retrain_if_needed.INSUFFICIENT_DATA
    assert "414 validation rows" in outcome.reason
    # The question was never answered, so the flag stays raised and the cadence clock
    # stays where it was: tomorrow asks again with a day more history.
    assert state.retrain_flag() is not None
    assert state.last_success(retrain_if_needed.PIPELINE) is None


def test_the_loop_step_still_exits_zero_when_there_is_too_little_history(
    loop_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(retrain_if_needed, "load_config", lambda _: loop_cfg)
    monkeypatch.setattr(retrain_if_needed, "configure", lambda _: None)
    monkeypatch.setattr(
        retrain_if_needed,
        "train_horizon",
        lambda cfg, horizon, *, tune: (_ for _ in ()).throw(InsufficientHistoryError("too young")),
    )

    assert retrain_if_needed.main([]) == 0


@dataclasses.dataclass
class _TrainResult:
    promoted: bool
    gate_reason: str


def test_every_drift_check_leaves_a_row_behind(loop_cfg, tmp_path, dataset):
    """The dashboard plots the share as a trend, which needs the checks to accumulate."""
    from src.monitoring.history import DriftHistory

    history = DriftHistory(tmp_path / "drift.csv")
    log = PredictionLog(tmp_path / "p.db")

    check_drift.run(loop_cfg, log=log, history=history, now=NOW, write_report=False)
    check_drift.run(
        loop_cfg, log=log, history=history, now=NOW + pd.Timedelta(days=1), write_report=False
    )

    recorded = history.read()
    assert len(recorded) == 2
    assert recorded["drift_share"].notna().all()
    assert recorded["monitored_features"].gt(0).all()


def test_a_quiet_check_is_recorded_as_well_as_a_noisy_one(loop_cfg, tmp_path, dataset):
    """A run of quiet nights is what makes a noisy one legible."""
    from src.monitoring.history import DriftHistory

    history = DriftHistory(tmp_path / "drift.csv")

    result = check_drift.run(
        loop_cfg,
        log=PredictionLog(tmp_path / "p.db"),
        history=history,
        now=NOW,
        write_report=False,
    )

    assert history.read()["status"].iloc[0] == result.status


def test_the_drift_history_lands_in_the_state_directory_the_workflow_carries(loop_cfg):
    """It cannot be recomputed later, so it has to travel with the prediction log."""
    assert loop_cfg.state.drift_history_path.parent == loop_cfg.state.dir
