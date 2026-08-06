"""Request and response shapes for the forecast service."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """A request for one day's hourly forecasts.

    `target_date` is a **Europe/Warsaw** calendar date, because that is the operating
    day a grid is scheduled against. `horizon_hours` is how many hourly forecasts to
    return starting at local midnight — not the model's own forecast horizon, which is
    fixed when the model is trained and is reported back as `model_horizon_hours`.
    """

    target_date: date
    horizon_hours: int = Field(default=24, ge=1, le=168)


class HourlyForecast(BaseModel):
    timestamp: datetime
    load_mw: float
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None


class ForecastResponse(BaseModel):
    predictions: list[HourlyForecast]
    model_version: str
    model_horizon_hours: int
    dataset_version: str | None
    intervals_available: bool
    requested_hours: int
    served_hours: int
    note: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
    model_horizon_hours: int | None = None
    loaded_at: datetime | None = None
    dataset_latest_actual: datetime | None = None
    servable_until: datetime | None = None
    detail: str | None = None


class ReloadResponse(BaseModel):
    previous: str | None
    current: str
    reloaded_at: datetime
