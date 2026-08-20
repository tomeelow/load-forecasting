"""The forecast service.

Three endpoints, one rule: everything it serves, it records. `/forecast` writes each
prediction to the log with the feature vector that produced it and the model version
that produced it, because Phase 8 measures production error by joining that log to
actuals as they arrive. A service that forecasts without logging looks identical from
the outside and makes monitoring impossible.

Run it with:

    uv run uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.champion import Champion, load_champion
from src.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HourlyForecast,
    ReloadResponse,
)
from src.config import Config, load_config
from src.features.inference import features_for_targets, servable_until
from src.ingestion.dataset import read_dataset
from src.ingestion.entsoe_client import LOAD_COLUMN
from src.ingestion.weather_client import fetch_national_forecast
from src.prediction_log import PredictionLog, PredictionRecord


@dataclass
class Service:
    """Everything the endpoints need, in one place so tests can substitute any of it."""

    cfg: Config
    log: PredictionLog
    champion: Champion | None = None
    load_error: str | None = None

    def load_history(self) -> pd.DataFrame:
        return read_dataset(self.cfg.data.dataset_path)

    def fetch_weather(self) -> pd.DataFrame:
        return fetch_national_forecast(
            self.cfg.weather,
            forecast_days=self.cfg.serving.weather_forecast_days,
            past_days=2,
        )

    def load_model(self) -> Champion | None:
        """Resolve the champion alias. A failure is recorded, not raised.

        The service must start even with an empty registry, so that `/health` can say
        *why* it is unhealthy instead of the container failing to boot with a stack
        trace nobody sees.
        """
        try:
            self.champion = load_champion(self.cfg.mlflow)
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 — reported through /health, not raised
            self.champion = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Could not load the champion model: {}", self.load_error)
        return self.champion


def create_app(cfg: Config | None = None, *, load_model_on_startup: bool = True) -> FastAPI:
    """Build the app. `cfg` is injectable so tests never touch the real registry."""
    load_dotenv()
    cfg = cfg or load_config()
    service = Service(cfg=cfg, log=PredictionLog(cfg.state.prediction_log_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Loading here rather than at import keeps the module importable without a
        # registry, which is what lets the tests run offline.
        if load_model_on_startup:
            app.state.service.load_model()
        yield

    app = FastAPI(
        title="PL day-ahead load forecast",
        description="Serves the registered champion model and logs every prediction.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:
        """Report readiness, and return non-200 when there is nothing to serve.

        A container health check that always says 200 tells the orchestrator nothing.
        """
        svc: Service = request.app.state.service
        if svc.champion is None:
            body = HealthResponse(status="no_model", detail=svc.load_error or "no model loaded")
            return JSONResponse(status_code=503, content=body.model_dump(mode="json"))

        latest_actual = None
        reachable = None
        try:
            history = svc.load_history()
            latest_actual = history[LOAD_COLUMN].last_valid_index()
            reachable = servable_until(history, svc.champion.horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Health check could not read the dataset: {}", exc)

        body = HealthResponse(
            status="ok",
            model_version=svc.champion.uri,
            model_horizon_hours=svc.champion.horizon,
            loaded_at=svc.champion.loaded_at,
            dataset_latest_actual=latest_actual,
            servable_until=reachable,
        )
        return JSONResponse(status_code=200, content=body.model_dump(mode="json"))

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
        svc: Service = request.app.state.service
        if svc.champion is None:
            raise HTTPException(status_code=503, detail=svc.load_error or "no model loaded")
        if payload.horizon_hours > svc.cfg.serving.max_request_hours:
            raise HTTPException(
                status_code=422,
                detail=f"At most {svc.cfg.serving.max_request_hours} hours per request",
            )

        tz = svc.cfg.data.timezone_local
        start_local = pd.Timestamp(payload.target_date, tz=tz)
        targets = pd.date_range(
            start_local, periods=payload.horizon_hours, freq="1h", tz=tz
        ).tz_convert("UTC")

        history = svc.load_history()
        weather = svc.fetch_weather()
        features = features_for_targets(
            history,
            weather,
            svc.champion.horizon,
            targets,
            rolling_window=svc.cfg.features.rolling_window_hours,
            weekly_lag=svc.cfg.features.weekly_lag_hours,
            tz=tz,
        )

        if features.empty:
            reachable = servable_until(history, svc.champion.horizon)
            raise HTTPException(
                status_code=409,
                detail=(
                    "No requested hour can be forecast from the recorded history. "
                    f"The dataset supports targets up to {reachable}; run the ingest "
                    "pipeline to bring it forward."
                ),
            )

        predicted = svc.champion.predict(features)
        _log_predictions(svc, features, predicted)

        note = None
        if len(predicted) < len(targets):
            note = (
                f"{len(targets) - len(predicted)} of {len(targets)} requested hour(s) fall "
                f"beyond what the recorded history supports for a "
                f"{svc.champion.horizon}h horizon and were omitted."
            )
        return ForecastResponse(
            predictions=[
                HourlyForecast(
                    timestamp=timestamp,
                    load_mw=round(float(row["load_mw"]), 1),
                    p10=_optional(row.get("p10")),
                    p50=_optional(row.get("p50")),
                    p90=_optional(row.get("p90")),
                )
                for timestamp, row in predicted.iterrows()
            ],
            model_version=svc.champion.uri,
            model_horizon_hours=svc.champion.horizon,
            dataset_version=svc.champion.dataset_version,
            intervals_available=svc.champion.has_intervals,
            requested_hours=len(targets),
            served_hours=len(predicted),
            note=note,
        )

    @app.post("/reload-model", response_model=ReloadResponse)
    def reload_model(request: Request) -> ReloadResponse:
        """Re-resolve the champion alias without restarting the process."""
        svc: Service = request.app.state.service
        previous = svc.champion.uri if svc.champion else None

        champion = svc.load_model()
        if champion is None:
            raise HTTPException(status_code=503, detail=svc.load_error or "reload failed")

        logger.info("Reloaded champion: {} -> {}", previous, champion.uri)
        return ReloadResponse(
            previous=previous, current=champion.uri, reloaded_at=datetime.now(UTC)
        )

    return app


def _optional(value: object) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 1)


def _log_predictions(service: Service, features: pd.DataFrame, predicted: pd.DataFrame) -> None:
    """Persist what was served, with the exact inputs that produced it."""
    champion = service.champion
    made_at = datetime.now(UTC)
    records = [
        PredictionRecord(
            predicted_at=made_at,
            target_time=timestamp,
            horizon_hours=champion.horizon,
            load_mw=float(row["load_mw"]),
            p10=_optional(row.get("p10")),
            p50=_optional(row.get("p50")),
            p90=_optional(row.get("p90")),
            model_name=champion.name,
            model_version=champion.version,
            run_id=champion.run_id,
            dataset_version=champion.dataset_version,
            features=features.loc[timestamp].to_dict(),
        )
        for timestamp, row in predicted.iterrows()
    ]
    service.log.log(records)


app = create_app()
