"""Typed access to config/config.yaml.

Pipeline parameters live in YAML; secrets live in the environment. Nothing here
reads a secret — see `src/ingestion/entsoe_client.py` for the one token this
project needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    weight: float


@dataclass(frozen=True)
class DataConfig:
    country_code: str
    start_date: str
    end_date: str | None
    timezone_local: str
    raw_dir: Path
    processed_dir: Path
    dataset_filename: str

    @property
    def dataset_path(self) -> Path:
        return self.processed_dir / self.dataset_filename


@dataclass(frozen=True)
class IngestionConfig:
    trailing_repull_days: int
    load_aggregation: str
    weather_forecast_days: int


@dataclass(frozen=True)
class WeatherConfig:
    archive_url: str
    forecast_url: str
    variables: tuple[str, ...]
    cities: tuple[City, ...]


@dataclass(frozen=True)
class FeaturesConfig:
    horizon_hours: int
    rolling_window_hours: int
    weekly_lag_hours: int


@dataclass(frozen=True)
class ValidationConfig:
    load_min_mw: float
    load_max_mw: float
    temp_min_c: float
    temp_max_c: float


@dataclass(frozen=True)
class Config:
    data: DataConfig
    ingestion: IngestionConfig
    weather: WeatherConfig
    features: FeaturesConfig
    validation: ValidationConfig


def _resolve(path: str) -> Path:
    """Make a configured path absolute, relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path | None = None) -> Config:
    """Read and validate the pipeline configuration."""
    raw = yaml.safe_load((path or DEFAULT_CONFIG_PATH).read_text())

    data = raw["data"]
    weather = raw["weather"]
    cities = tuple(City(**c) for c in weather["cities"])
    total = sum(c.weight for c in cities)
    if total <= 0:
        raise ValueError("City weights must sum to a positive number")
    cities = tuple(
        City(name=c.name, lat=c.lat, lon=c.lon, weight=c.weight / total) for c in cities
    )

    return Config(
        data=DataConfig(
            country_code=data["country_code"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            timezone_local=data["timezone_local"],
            raw_dir=_resolve(data["raw_dir"]),
            processed_dir=_resolve(data["processed_dir"]),
            dataset_filename=data["dataset_filename"],
        ),
        ingestion=IngestionConfig(**raw["ingestion"]),
        weather=WeatherConfig(
            archive_url=weather["archive_url"],
            forecast_url=weather["forecast_url"],
            variables=tuple(weather["variables"]),
            cities=cities,
        ),
        features=FeaturesConfig(**raw["features"]),
        validation=ValidationConfig(**raw["validation"]),
    )
