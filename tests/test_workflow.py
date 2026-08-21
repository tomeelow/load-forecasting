"""The scheduled workflow, read as data.

The loop's correctness is not only in the Python: it is in the order the steps run and
in what is carried between runs. Both live in YAML that no test would otherwise touch,
and both are quiet when they break — a reordered step still passes CI, and a state path
dropped from the snapshot only shows up as monitoring that mysteriously restarts from
nothing. See ADR-008.
"""

from __future__ import annotations

import yaml

from src.config import REPO_ROOT

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "daily-loop.yml"

# The order matters: nothing can be scored before the actuals are pulled, drift cannot
# be judged before the served predictions are scored, retraining answers drift, and the
# forecast goes last so it is made with whatever champion the run leaves in place.
CHAIN = [
    "pipelines.ingest",
    "pipelines.evaluate",
    "pipelines.check_drift",
    "pipelines.retrain_if_needed",
    "pipelines.forecast",
]


def workflow() -> dict:
    # `on:` is the YAML boolean True, which is a wart of the format rather than a bug.
    return yaml.safe_load(WORKFLOW.read_text())


def steps() -> list[dict]:
    return workflow()["jobs"]["loop"]["steps"]


def test_the_loop_runs_ingest_then_evaluate_then_drift_then_retrain_then_forecast():
    commands = [step.get("run", "") for step in steps()]
    positions = [
        next(i for i, command in enumerate(commands) if pipeline in command) for pipeline in CHAIN
    ]

    assert positions == sorted(positions), f"the loop's steps are out of order: {CHAIN}"


def test_every_unreproducible_path_is_carried_between_runs():
    """The prediction log especially: what the service said yesterday cannot be re-derived."""
    carried = workflow()["env"]["STATE_PATHS"].split()

    assert set(carried) >= {"state", "mlruns", "data/processed"}


def test_state_is_persisted_even_when_a_step_failed():
    persist = next(s for s in steps() if "push --force" in s.get("run", ""))

    assert persist["if"] == "always()"


def test_two_runs_cannot_race_on_the_state_branch():
    """Concurrent runs would force-push over each other's snapshot."""
    concurrency = workflow()["concurrency"]

    assert concurrency["group"]
    assert concurrency["cancel-in-progress"] is False


def test_the_workflow_writes_to_the_same_mlflow_store_the_code_reads(monkeypatch):
    """The scheduled loop sets no `MLFLOW_TRACKING_URI`, so it gets the configured store."""
    from src.config import load_config

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    configured = load_config().mlflow.tracking_uri.rsplit("/", 2)[-2:]

    assert "/".join(configured) == "mlruns/mlflow.db"
    assert "sqlite:///mlruns/mlflow.db" in WORKFLOW.read_text(), "mlflow gc must reclaim from it"


def test_the_loop_forecasts_so_the_prediction_log_is_never_empty():
    """Without this step nothing ever serves, and Phase 8 measures an empty table."""
    commands = [step.get("run", "") for step in steps()]

    assert any("pipelines.forecast" in command for command in commands)


def test_the_forecast_runs_after_the_gate_so_it_uses_the_promoted_champion():
    commands = [step.get("run", "") for step in steps()]
    retrain = next(i for i, c in enumerate(commands) if "pipelines.retrain_if_needed" in c)
    forecast = next(i for i, c in enumerate(commands) if "pipelines.forecast" in c)

    assert forecast > retrain
