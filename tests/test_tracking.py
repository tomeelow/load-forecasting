"""MLflow tracking and the registry loop, against a throwaway SQLite backend.

The gate's arithmetic is tested in `test_promotion.py`; what is tested here is the
wiring around it — that a run is recorded with the provenance it claims, that the
champion's metric can be read back, and that the alias only moves when the gate says so.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import MlflowConfig
from src.models.lgbm import train
from src.models.promotion import promotion_decision
from src.models.tracking import (
    RunSpec,
    apply_gate,
    champion_metric,
    configure,
    latest_version_for_run,
    log_run,
)

MODEL_NAME = "test_load_lgbm"
ALIAS = "champion"


@pytest.fixture
def tracking(tmp_path):
    """A private MLflow backend per test, so nothing touches the developer's runs."""
    cfg = MlflowConfig(
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        experiment="test-experiment",
        registered_model_name=MODEL_NAME,
        champion_alias=ALIAS,
    )
    configure(cfg)
    yield cfg
    mlflow.set_tracking_uri(None)


@pytest.fixture
def tiny_model(short_dataset):
    """A deliberately small booster: these tests are about the plumbing, not accuracy."""
    from src.features.builder import TARGET_COLUMN, feature_columns, make_features

    features = make_features(short_dataset, 24)
    columns = feature_columns(list(features.columns))
    X, y = features[columns], features[TARGET_COLUMN]
    split = len(X) - 200
    model = train(
        X.iloc[:split],
        y.iloc[:split],
        X.iloc[split:],
        y.iloc[split:],
        num_boost_round=10,
        early_stopping_rounds=5,
    )
    return model, X.iloc[split:]


def logged_model_names(run_id: str) -> set[str]:
    """MLflow 3 stores models as first-class entities, not as run artifacts."""
    models = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run_id}'", output_format="list"
    )
    return {m.name for m in models}


def spec(**overrides) -> RunSpec:
    base = {
        "name": "lgbm_h24",
        "horizon": 24,
        "dataset_version": "abc123",
        "feature_set": "fs0000",
        "synthetic": True,
        "params": {"learning_rate": 0.05},
        "metrics": {"mape": 2.0, "rmse_mw": 500.0},
    }
    return RunSpec(**{**base, **overrides})


def test_a_run_records_the_provenance_it_will_be_asked_about_later(tracking, tiny_model):
    model, X_sample = tiny_model

    run_id = log_run(spec(), model, X_sample)

    run = MlflowClient().get_run(run_id)
    assert run.data.params["horizon_hours"] == "24"
    assert run.data.params["dataset_version"] == "abc123"
    assert run.data.params["feature_set"] == "fs0000"
    assert run.data.params["learning_rate"] == "0.05"
    assert run.data.metrics["mape"] == pytest.approx(2.0)
    assert run.data.tags["horizon"] == "24"


def test_synthetic_runs_are_labelled_in_the_name_the_tags_and_a_note(tracking, tiny_model):
    model, X_sample = tiny_model

    run_id = log_run(spec(synthetic=True), model, X_sample)

    run = MlflowClient().get_run(run_id)
    assert run.info.run_name.endswith("_synthetic")
    assert run.data.tags["data_source"] == "synthetic"
    assert "SYNTHETIC" in run.data.tags["mlflow.note.content"]


def test_a_real_data_run_carries_no_synthetic_marking(tracking, tiny_model):
    model, X_sample = tiny_model

    run_id = log_run(spec(synthetic=False), model, X_sample)

    run = MlflowClient().get_run(run_id)
    assert not run.info.run_name.endswith("_synthetic")
    assert run.data.tags["data_source"] == "ingested"


def test_the_model_is_logged_with_a_signature_and_an_importance_artifact(tracking, tiny_model):
    model, X_sample = tiny_model

    run_id = log_run(spec(), model, X_sample)

    client = MlflowClient()
    assert "model" in logged_model_names(run_id)
    assert "feature_importance.png" in {a.path for a in client.list_artifacts(run_id)}

    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    assert loaded.metadata.signature is not None
    assert len(loaded.metadata.get_input_schema().input_names()) == len(model.columns)


def test_nothing_is_in_production_before_the_first_promotion(tracking):
    assert champion_metric(MlflowClient(), MODEL_NAME, ALIAS, "mape") is None


def test_a_promoted_model_becomes_the_champion_the_gate_compares_against(tracking, tiny_model):
    model, X_sample = tiny_model
    client = MlflowClient()

    run_id = log_run(spec(metrics={"mape": 2.0}), model, X_sample, registered_model_name=MODEL_NAME)
    version = latest_version_for_run(client, MODEL_NAME, run_id)
    decision = promotion_decision(candidate=2.0, naive=4.0, champion=None)

    assert apply_gate(client, MODEL_NAME, version, ALIAS, decision)
    assert champion_metric(client, MODEL_NAME, ALIAS, "mape") == pytest.approx(2.0)


def test_a_held_back_candidate_leaves_production_untouched(tracking, tiny_model):
    """The whole point of the gate: a scheduled retrain must not be able to make it worse."""
    model, X_sample = tiny_model
    client = MlflowClient()

    good = log_run(spec(metrics={"mape": 2.0}), model, X_sample, registered_model_name=MODEL_NAME)
    good_version = latest_version_for_run(client, MODEL_NAME, good)
    apply_gate(client, MODEL_NAME, good_version, ALIAS, promotion_decision(2.0, 4.0, None))

    worse = log_run(spec(metrics={"mape": 3.5}), model, X_sample, registered_model_name=MODEL_NAME)
    worse_version = latest_version_for_run(client, MODEL_NAME, worse)
    decision = promotion_decision(candidate=3.5, naive=4.0, champion=2.0)

    assert not apply_gate(client, MODEL_NAME, worse_version, ALIAS, decision)
    # The alias still points at the good model, and the champion metric is unchanged.
    assert client.get_model_version_by_alias(MODEL_NAME, ALIAS).version == good_version
    assert champion_metric(client, MODEL_NAME, ALIAS, "mape") == pytest.approx(2.0)


def test_the_gate_decision_is_recorded_on_the_version_that_was_refused(tracking, tiny_model):
    model, X_sample = tiny_model
    client = MlflowClient()

    run_id = log_run(spec(metrics={"mape": 9.0}), model, X_sample, registered_model_name=MODEL_NAME)
    version = latest_version_for_run(client, MODEL_NAME, run_id)
    decision = promotion_decision(candidate=9.0, naive=4.0, champion=2.0)
    apply_gate(client, MODEL_NAME, version, ALIAS, decision)

    tags = client.get_model_version(MODEL_NAME, version).tags
    assert tags["gate_decision"] == "False"
    assert "does not beat the naive baseline" in tags["gate_reason"]


def test_an_unregistered_run_has_no_version(tracking, tiny_model):
    model, X_sample = tiny_model

    run_id = log_run(spec(), model, X_sample)  # no registered_model_name

    assert latest_version_for_run(MlflowClient(), MODEL_NAME, run_id) is None


def test_quantile_companions_ride_along_with_the_point_model(tracking, tiny_model, short_dataset):
    from src.features.builder import TARGET_COLUMN, feature_columns, make_features
    from src.models.lgbm import train_quantiles

    model, X_sample = tiny_model
    features = make_features(short_dataset, 24)
    columns = feature_columns(list(features.columns))
    X, y = features[columns], features[TARGET_COLUMN]
    split = len(X) - 200
    band = train_quantiles(
        X.iloc[:split],
        y.iloc[:split],
        X.iloc[split:],
        y.iloc[split:],
        (0.1, 0.9),
        num_boost_round=10,
        early_stopping_rounds=5,
    )

    run_id = log_run(
        spec(),
        model,
        X_sample,
        extra_models={"model_p10": band[0.1], "model_p90": band[0.9]},
    )

    assert {"model", "model_p10", "model_p90"} <= logged_model_names(run_id)


def test_extra_tables_are_logged_as_artifacts(tracking, tiny_model):
    model, X_sample = tiny_model
    table = pd.DataFrame({"segment": ["peak"], "mape": [1.5]})

    run_id = log_run(spec(), model, X_sample, extra_artifacts={"segments": table})

    artifacts = {a.path for a in MlflowClient().list_artifacts(run_id)}
    assert "segments.json" in artifacts


def test_a_synthetic_champion_is_not_compared_against_a_real_data_candidate(tracking, tiny_model):
    """Different data, different problem. A 2.5% on real load is not a regression."""
    model, X_sample = tiny_model
    client = MlflowClient()

    run_id = log_run(
        spec(synthetic=True, metrics={"mape": 1.7}),
        model,
        X_sample,
        registered_model_name=MODEL_NAME,
    )
    version = latest_version_for_run(client, MODEL_NAME, run_id)
    apply_gate(client, MODEL_NAME, version, ALIAS, promotion_decision(1.7, 4.0, None))

    same_source = champion_metric(client, MODEL_NAME, ALIAS, "mape", data_source="synthetic")
    other_source = champion_metric(client, MODEL_NAME, ALIAS, "mape", data_source="ingested")

    assert same_source == pytest.approx(1.7)
    assert other_source is None, "metrics from different data sources must not be compared"


def test_without_a_declared_data_source_the_champion_metric_is_returned(tracking, tiny_model):
    model, X_sample = tiny_model
    client = MlflowClient()
    run_id = log_run(spec(metrics={"mape": 2.0}), model, X_sample, registered_model_name=MODEL_NAME)
    version = latest_version_for_run(client, MODEL_NAME, run_id)
    apply_gate(client, MODEL_NAME, version, ALIAS, promotion_decision(2.0, 4.0, None))

    assert champion_metric(client, MODEL_NAME, ALIAS, "mape") == pytest.approx(2.0)
