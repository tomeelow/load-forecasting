"""The scheduled forecast step: which hours it asks about, and what it writes down.

Everything monitoring later says about production performance is a join against the rows
this step writes, so the tests are about *coverage* and *provenance*: does the daily run
tile the hours without leaving a gap, and does every row carry the model version and
dataset hash needed to explain it months later.

The champion is stubbed. Loading a real one means an MLflow registry, and the thing worth
testing here is the hour selection and the record shape, not that MLflow can read its own
database.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from pipelines import forecast
from src import synthetic
from src.ingestion.dataset import write_dataset
from src.pipeline_state import PipelineState
from src.prediction_log import PredictionLog

HORIZON = 24


@pytest.fixture(scope="module")
def dataset():
    return synthetic.make_dataset(start="2024-01-01", end="2024-03-01")


@pytest.fixture
def forecast_cfg(cfg, tmp_path, dataset):
    write_dataset(dataset, tmp_path / "dataset.parquet")
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, processed_dir=tmp_path, dataset_filename="dataset.parquet"
        ),
        state=dataclasses.replace(cfg.state, dir=tmp_path),
    )


class StubChampion:
    """The registry's answer, without the registry."""

    name = "pl_load_lgbm"
    version = "7"
    run_id = "abc123"
    dataset_version = "deadbeef1234"
    horizon = HORIZON

    def __init__(self, columns: list[str] | None = None) -> None:
        self.feature_columns = columns or []

    @property
    def uri(self) -> str:
        return f"{self.name}/{self.version}"

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["load_mw"] = 20000.0
        out["p10"] = 19000.0
        out["p50"] = 20000.0
        out["p90"] = 21000.0
        return out


def future_weather(dataset: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    """Forecast weather running past the last actual, which is what serving really gets.

    Without hours beyond the end of the load series there is nothing to forecast: a
    target hour needs its own weather, and the archive stops where the past does.
    """
    columns = ["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]
    ahead = pd.date_range(
        dataset.index.max() + pd.Timedelta(hours=1), periods=hours, freq="1h", tz="UTC"
    )
    extension = pd.DataFrame(
        {c: float(dataset[c].iloc[-1]) for c in columns if c in dataset.columns}, index=ahead
    )
    recent = dataset[[c for c in columns if c in dataset.columns]].tail(72)
    return pd.concat([recent, extension])


@pytest.fixture
def served(monkeypatch, dataset):
    """Stub the champion and the weather call; leave the feature path real."""
    monkeypatch.setattr(forecast, "load_champion", lambda _: StubChampion())
    monkeypatch.setattr(
        forecast, "fetch_national_forecast", lambda *args, **kwargs: future_weather(dataset)
    )


def test_the_targets_start_the_hour_after_the_last_published_actual(dataset):
    targets = forecast.target_hours(dataset, HORIZON)

    assert targets.min() == dataset["load_mw"].last_valid_index() + pd.Timedelta(hours=1)


def test_the_targets_reach_exactly_one_horizon_past_the_last_actual(dataset):
    targets = forecast.target_hours(dataset, HORIZON)

    assert targets.max() == dataset["load_mw"].last_valid_index() + pd.Timedelta(hours=HORIZON)
    assert len(targets) == HORIZON


def test_consecutive_runs_tile_the_hours_without_a_gap(dataset):
    """Anchoring on the last actual rather than the clock is what makes this true.

    A hole in the served series is invisible in a rolling MAPE and cannot be filled in
    later, because the forecast for a past hour can no longer be made.
    """
    today = forecast.target_hours(dataset, HORIZON)

    # The next run happens after ingestion published actuals for the hours just forecast.
    tomorrow_data = dataset.reindex(dataset.index.union(today))
    tomorrow_data.loc[today, "load_mw"] = 20000.0
    tomorrow = forecast.target_hours(tomorrow_data, HORIZON)

    assert tomorrow.min() == today.max() + pd.Timedelta(hours=1)
    assert len(tomorrow) == HORIZON


def test_a_dataset_with_no_actuals_yields_no_targets(dataset):
    empty = dataset.copy()
    empty["load_mw"] = pd.NA

    assert forecast.target_hours(empty, HORIZON).empty


def test_a_frame_without_a_load_column_yields_no_targets(dataset):
    assert forecast.target_hours(dataset.drop(columns=["load_mw"]), HORIZON).empty


def test_a_run_logs_one_prediction_per_reachable_hour(forecast_cfg, tmp_path, served):
    log = PredictionLog(tmp_path / "p.db")

    summary = forecast.run(forecast_cfg, log=log, state=PipelineState(tmp_path / "s.json"))

    assert summary.logged > 0
    assert log.count() == summary.logged


def test_every_logged_row_carries_the_model_version_and_dataset_hash(
    forecast_cfg, tmp_path, served
):
    """A prediction nobody can attribute to a model and a dataset cannot be explained."""
    log = PredictionLog(tmp_path / "p.db")

    forecast.run(forecast_cfg, log=log, state=PipelineState(tmp_path / "s.json"))

    logged = log.read()
    assert (logged["model_version"] == "7").all()
    assert (logged["dataset_version"] == "deadbeef1234").all()
    assert logged["run_id"].notna().all()


def test_the_band_is_logged_alongside_the_point_forecast(forecast_cfg, tmp_path, served):
    log = PredictionLog(tmp_path / "p.db")

    forecast.run(forecast_cfg, log=log, state=PipelineState(tmp_path / "s.json"))

    logged = log.read()
    assert (logged["p10"] < logged["p90"]).all()


def test_the_feature_vector_that_produced_each_prediction_is_kept(forecast_cfg, tmp_path, served):
    log = PredictionLog(tmp_path / "p.db")

    forecast.run(forecast_cfg, log=log, state=PipelineState(tmp_path / "s.json"))

    with_features = log.read(with_features=True)
    assert "load_lag_24" in with_features.columns
    assert "temp_c" in with_features.columns


def test_logged_predictions_can_be_scored_once_the_actuals_arrive(
    forecast_cfg, tmp_path, served, dataset
):
    """The whole point: today's rows become tomorrow's production error."""
    log = PredictionLog(tmp_path / "p.db")
    forecast.run(forecast_cfg, log=log, state=PipelineState(tmp_path / "s.json"))

    later = dataset.copy()
    targets = log.read().index
    extended = later.reindex(later.index.union(targets))
    extended.loc[targets, "load_mw"] = 20500.0
    extended.loc[targets, "tso_forecast_mw"] = 20800.0

    assert log.score(extended) == len(targets)
    assert len(log.scored()) == len(targets)


def test_an_empty_registry_is_reported_not_raised(forecast_cfg, tmp_path, monkeypatch):
    """A fresh deployment has no champion until the first retrain promotes one."""

    def no_champion(_):
        raise RuntimeError("no model version with alias champion")

    monkeypatch.setattr(forecast, "load_champion", no_champion)
    state = PipelineState(tmp_path / "s.json")

    summary = forecast.run(forecast_cfg, log=PredictionLog(tmp_path / "p.db"), state=state)

    assert summary.logged == 0
    assert "no champion" in summary.note
    assert "nothing logged" in summary.format()


def test_the_marker_is_recorded_even_when_nothing_could_be_forecast(
    forecast_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(forecast, "load_champion", lambda _: StubChampion())
    monkeypatch.setattr(forecast, "target_hours", lambda *_: pd.DatetimeIndex([], tz="UTC"))
    state = PipelineState(tmp_path / "s.json")

    forecast.run(forecast_cfg, log=PredictionLog(tmp_path / "p.db"), state=state)

    assert state.last_success(forecast.PIPELINE) is not None


def test_the_run_records_its_marker(forecast_cfg, tmp_path, served):
    state = PipelineState(tmp_path / "s.json")

    forecast.run(forecast_cfg, log=PredictionLog(tmp_path / "p.db"), state=state)

    assert state.last_success(forecast.PIPELINE) is not None


def test_the_summary_names_the_model_and_the_window(forecast_cfg, tmp_path, served):
    summary = forecast.run(
        forecast_cfg, log=PredictionLog(tmp_path / "p.db"), state=PipelineState(tmp_path / "s.json")
    )

    text = summary.format()
    assert "pl_load_lgbm/7" in text
    assert "deadbeef1234" in text
