"""Pipeline state: last-success markers and the retrain flag.

Small, but it is what stands between a missed cron run and a permanent hole in the
dataset, and between drift being detected and a retrain actually happening.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.pipeline_state import PipelineState

NOW = pd.Timestamp("2026-08-06 05:30", tz="UTC")


@pytest.fixture
def state(tmp_path) -> PipelineState:
    return PipelineState(tmp_path / "pipeline_state.json")


def test_a_pipeline_that_never_ran_has_no_marker(state):
    assert state.last_success("ingest") is None
    assert state.days_since("ingest") is None


def test_a_success_is_remembered_across_instances(state, tmp_path):
    state.record_success("ingest", NOW)

    reopened = PipelineState(tmp_path / "pipeline_state.json")

    assert reopened.last_success("ingest") == NOW


def test_each_pipeline_is_tracked_separately(state):
    state.record_success("ingest", NOW)
    state.record_success("evaluate", NOW - pd.Timedelta(days=3))

    assert state.last_success("ingest") == NOW
    assert state.last_success("evaluate") == NOW - pd.Timedelta(days=3)
    assert state.last_success("check_drift") is None


def test_days_since_measures_the_gap_a_missed_run_leaves(state):
    state.record_success("ingest", NOW - pd.Timedelta(days=4, hours=12))

    assert state.days_since("ingest", now=NOW) == pytest.approx(4.5)


def test_the_retrain_flag_survives_and_carries_its_reason(state):
    flag = state.raise_retrain_flag("rolling MAPE 4.1% over threshold 3.5%")

    assert state.retrain_flag() is not None
    assert state.retrain_flag().reason == flag.reason
    assert "4.1%" in state.retrain_flag().reason


def test_raising_the_flag_twice_keeps_the_original_moment(state):
    """Drift on three consecutive days is one outstanding retrain, not a reset clock."""
    first = state.raise_retrain_flag("data drift on temp_c")
    second = state.raise_retrain_flag("data drift on load_lag_24")

    assert second.raised_at == first.raised_at
    assert state.retrain_flag().reason == "data drift on temp_c"


def test_clearing_the_flag_makes_it_absent(state):
    state.raise_retrain_flag("drift")

    state.clear_retrain_flag()

    assert state.retrain_flag() is None


def test_clearing_an_unset_flag_is_harmless(state):
    state.clear_retrain_flag()

    assert state.retrain_flag() is None


def test_the_file_is_readable_by_a_human_reviewing_a_diff(state, tmp_path):
    state.record_success("ingest", NOW)
    state.raise_retrain_flag("drift")

    payload = json.loads((tmp_path / "pipeline_state.json").read_text())

    assert payload["version"] == 1
    assert payload["last_success"]["ingest"].startswith("2026-08-06")
    assert payload["retrain_flag"]["reason"] == "drift"


def test_a_corrupt_state_file_is_refused_rather_than_guessed(tmp_path):
    path = tmp_path / "pipeline_state.json"
    path.write_text("{ this is not json")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        PipelineState(path).last_success("ingest")


def test_a_write_leaves_no_partial_file_behind(state, tmp_path):
    state.record_success("ingest", NOW)

    leftovers = list(tmp_path.glob("*.tmp"))

    assert leftovers == []
