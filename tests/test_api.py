"""The forecast service, with a stub champion and stub weather.

No registry, no network, no token. What is under test is the contract: the response
shape, that `/health` is honest about being unable to serve, that `/reload-model` picks
up a new champion without a restart, and — most importantly — that every served
prediction reaches the log with the features that produced it.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.champion import Champion
from src.api.main import Service, create_app
from src.prediction_log import PredictionLog

TARGET_DATE = "2024-02-01"
HORIZON = 24


class StubModel:
    """Stands in for an MLflow pyfunc model: predict a frame, get an array back."""

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.seen: list[list[str]] = []

    def predict(self, X):
        self.seen.append(list(X.columns))
        return (20_000.0 + self.offset) + X["hour"].to_numpy() * 10.0


def make_champion(version: str = "1", *, with_intervals: bool = True) -> Champion:
    from src import synthetic
    from src.features.builder import feature_columns, make_features

    columns = feature_columns(
        list(
            make_features(
                synthetic.make_dataset(start="2024-01-01", end="2024-01-20"), HORIZON
            ).columns
        )
    )
    quantiles = (
        {"p10": StubModel(-900), "p50": StubModel(0), "p90": StubModel(900)}
        if with_intervals
        else {}
    )
    return Champion(
        name="pl_load_lgbm",
        version=version,
        run_id="run-stub",
        horizon=HORIZON,
        feature_columns=columns,
        dataset_version="ds-test",
        loaded_at=pd.Timestamp("2024-02-01 06:00", tz="UTC").to_pydatetime(),
        point=StubModel(),
        quantiles=quantiles,
    )


@pytest.fixture
def history():
    """Recorded history ending mid-morning, as a real dataset does.

    ENTSO-E publishes actuals a few hours behind real time, so the last known load is
    always older than 'now' — which is what bounds how far a day-ahead model can reach.
    """
    from src import synthetic

    return synthetic.make_dataset(start="2024-01-01", end="2024-02-01 09:00")


@pytest.fixture
def weather_forecast():
    """Forecast weather running days past the last actual, as Open-Meteo's does."""
    from src import synthetic

    ahead = synthetic.make_dataset(start="2024-01-01", end="2024-02-05")
    return ahead[["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]]


@pytest.fixture
def service_factory(cfg, tmp_path, history, weather_forecast):
    def build(champion: Champion | None = "default", *, load_error: str | None = None):
        if champion == "default":
            champion = make_champion()
        tmp_cfg = dataclasses.replace(cfg, state=dataclasses.replace(cfg.state, dir=tmp_path))
        service = Service(
            cfg=tmp_cfg,
            log=PredictionLog(tmp_cfg.state.prediction_log_path),
            champion=champion,
            load_error=load_error,
        )
        service.load_history = lambda: history
        service.fetch_weather = lambda: weather_forecast
        return service

    return build


@pytest.fixture
def client(service_factory):
    app = create_app(load_model_on_startup=False)
    app.state.service = service_factory()
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_model_and_the_data_it_can_reach(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "pl_load_lgbm/1"
    assert body["model_horizon_hours"] == HORIZON
    assert body["dataset_latest_actual"] is not None
    # The furthest servable target is one horizon past the last recorded actual.
    assert body["servable_until"] > body["dataset_latest_actual"]


def test_health_is_not_200_when_nothing_can_be_served(service_factory):
    """A health check that always says ok is worse than none at all."""
    app = create_app(load_model_on_startup=False)
    app.state.service = service_factory(None, load_error="RestException: alias not found")

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "no_model"
    assert "alias not found" in response.json()["detail"]


def test_a_forecast_returns_hourly_predictions_with_a_band(client):
    response = client.post("/forecast", json={"target_date": TARGET_DATE, "horizon_hours": 6})

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "pl_load_lgbm/1"
    assert body["model_horizon_hours"] == HORIZON
    assert body["intervals_available"] is True
    assert body["requested_hours"] == 6
    assert body["served_hours"] == len(body["predictions"])

    first = body["predictions"][0]
    assert set(first) == {"timestamp", "load_mw", "p10", "p50", "p90"}
    assert first["p10"] < first["load_mw"] < first["p90"]


def test_the_band_is_absent_when_the_champion_has_no_quantiles(service_factory):
    app = create_app(load_model_on_startup=False)
    app.state.service = service_factory(make_champion(with_intervals=False))

    with TestClient(app) as test_client:
        body = test_client.post(
            "/forecast", json={"target_date": TARGET_DATE, "horizon_hours": 3}
        ).json()

    assert body["intervals_available"] is False
    assert body["predictions"][0]["p10"] is None
    assert body["predictions"][0]["load_mw"] is not None


def test_hours_beyond_the_recorded_history_are_omitted_and_explained(client):
    """History ends at 09:00, so a 24h model cannot reach the whole of the next day."""
    response = client.post("/forecast", json={"target_date": "2024-02-02", "horizon_hours": 24})

    body = response.json()
    assert body["served_hours"] < body["requested_hours"]
    assert body["note"] is not None
    assert "beyond what the recorded history supports" in body["note"]


def test_a_request_entirely_beyond_reach_is_a_conflict_not_a_guess(client):
    response = client.post("/forecast", json={"target_date": "2024-03-01", "horizon_hours": 24})

    assert response.status_code == 409
    assert "run the ingest pipeline" in response.json()["detail"]


def test_forecasting_without_a_model_is_unavailable_not_a_crash(service_factory):
    app = create_app(load_model_on_startup=False)
    app.state.service = service_factory(None, load_error="no champion alias")

    with TestClient(app) as test_client:
        response = test_client.post("/forecast", json={"target_date": TARGET_DATE})

    assert response.status_code == 503
    assert "no champion alias" in response.json()["detail"]


def test_an_oversized_request_is_rejected(client):
    response = client.post("/forecast", json={"target_date": TARGET_DATE, "horizon_hours": 100})

    assert response.status_code == 422


def test_every_served_prediction_is_logged_with_its_features(client):
    """The linchpin: without this row, production error can never be computed."""
    service = client.app.state.service

    body = client.post("/forecast", json={"target_date": TARGET_DATE, "horizon_hours": 5}).json()

    logged = service.log.read(with_features=True)
    assert len(logged) == body["served_hours"] > 0
    row = logged.iloc[0]
    assert row["model_version"] == "1"
    assert row["dataset_version"] == "ds-test"
    assert row["horizon_hours"] == HORIZON
    # The exact inputs are recoverable, which is what makes drift analysis possible.
    assert "temp_c" in logged.columns
    assert "load_lag_24" in logged.columns
    assert logged.index[0] == pd.Timestamp(body["predictions"][0]["timestamp"])


def test_the_logged_point_forecast_matches_what_was_returned(client):
    service = client.app.state.service

    body = client.post("/forecast", json={"target_date": TARGET_DATE, "horizon_hours": 4}).json()

    logged = service.log.read()
    for prediction in body["predictions"]:
        stored = logged.loc[pd.Timestamp(prediction["timestamp"]), "load_mw"]
        assert float(stored) == pytest.approx(prediction["load_mw"], abs=0.05)


def test_the_model_receives_columns_in_the_order_it_was_trained_on(client):
    service = client.app.state.service

    client.post("/forecast", json={"target_date": TARGET_DATE, "horizon_hours": 3})

    assert service.champion.point.seen[0] == service.champion.feature_columns


def test_reload_switches_the_served_version_without_a_restart(service_factory):
    app = create_app(load_model_on_startup=False)
    service = service_factory(make_champion(version="1"))
    app.state.service = service

    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["model_version"] == "pl_load_lgbm/1"

        service.load_model = lambda: (
            setattr(service, "champion", make_champion(version="7")) or service.champion
        )
        response = test_client.post("/reload-model")

        assert response.status_code == 200
        assert response.json() == {
            **response.json(),
            "previous": "pl_load_lgbm/1",
            "current": "pl_load_lgbm/7",
        }
        assert test_client.get("/health").json()["model_version"] == "pl_load_lgbm/7"


def test_a_failed_reload_reports_unavailable(service_factory):
    app = create_app(load_model_on_startup=False)
    service = service_factory(make_champion())
    app.state.service = service

    with TestClient(app) as test_client:

        def fail():
            service.champion = None
            service.load_error = "RestException: registry unreachable"
            return None

        service.load_model = fail
        response = test_client.post("/reload-model")

    assert response.status_code == 503
    assert "registry unreachable" in response.json()["detail"]


def test_a_champion_refuses_a_frame_missing_a_column(client):
    champion = client.app.state.service.champion
    frame = pd.DataFrame({"hour": [1.0]}, index=pd.DatetimeIndex(["2024-02-01"], tz="UTC"))

    with pytest.raises(ValueError, match="missing columns"):
        champion.predict(frame)
