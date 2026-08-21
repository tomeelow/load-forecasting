"""Typed access to config/config.yaml, and the one place `.env` is loaded.

Pipeline parameters live in YAML; secrets live in the environment. Nothing here
reads a secret — see `src/ingestion/entsoe_client.py` for the one token this
project needs. What lives here is where that token is *found*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"


def load_project_env() -> bool:
    """Load the repository's `.env`. Returns whether a file was found.

    Anchored to the repository root, like every other configured path (`_resolve`).
    `python-dotenv`'s own discovery walks up from whichever module happens to call it,
    which works but makes "which file" depend on where the call was written; naming the
    path leaves nothing to infer, from any working directory.

    **`override=False`, deliberately.** An exported variable is someone's decision — a
    different token for one run, a CI secret injected by the workflow — and a file on
    disk must not silently win against it. The consequence is worth knowing: an empty
    `ENTSOE_API_KEY` exported in a shell shadows a perfectly good `.env`, and the error
    then says the key is not set while the file plainly holds one.
    """
    found = load_dotenv(ENV_PATH, override=False)
    if not found:
        logger.debug("No .env at {}; relying on the ambient environment", ENV_PATH)
    return found


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
    synthetic_start: str
    synthetic_end: str

    @property
    def dataset_path(self) -> Path:
        return self.processed_dir / self.dataset_filename


@dataclass(frozen=True)
class IngestionConfig:
    trailing_repull_days: int
    load_aggregation: str
    weather_forecast_days: int


@dataclass(frozen=True)
class RequestConfig:
    """How hard an HTTP client tries before giving up on a transient failure."""

    timeout_s: float
    max_attempts: int
    backoff_s: float
    backoff_max_s: float

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")
        if self.backoff_s < 0 or self.backoff_max_s < self.backoff_s:
            raise ValueError(
                f"backoff_s must be non-negative and no larger than backoff_max_s, "
                f"got {self.backoff_s} and {self.backoff_max_s}"
            )

    def pause_before(self, attempt: int) -> float:
        """Seconds to wait before `attempt` (2 is the first retry): exponential, capped."""
        return min(self.backoff_s * 2 ** (attempt - 2), self.backoff_max_s)


@dataclass(frozen=True)
class WeatherConfig:
    archive_url: str
    forecast_url: str
    variables: tuple[str, ...]
    cities: tuple[City, ...]
    request: RequestConfig
    # Open-Meteo's archive of past forecast runs. Evaluation-only — see the note in
    # config.yaml and `fetch_national_day_ahead`.
    historical_forecast_url: str = ""
    historical_forecast_lead_days: int = 1


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
class TuningConfig:
    n_trials: int
    timeout_s: int | None


@dataclass(frozen=True)
class ModelConfig:
    horizons: tuple[int, ...]
    seed: int
    validation_days: int
    num_boost_round: int
    early_stopping_rounds: int
    quantiles: tuple[float, ...]
    tuning: TuningConfig


@dataclass(frozen=True)
class PromotionConfig:
    metric: str
    max_regression: float


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment: str
    registered_model_name: str
    champion_alias: str


@dataclass(frozen=True)
class BacktestConfig:
    initial_train_days: int
    test_days: int
    step_days: int
    max_splits: int | None
    quantiles: bool
    reports_dir: Path


@dataclass(frozen=True)
class EvaluationConfig:
    peak_hours: tuple[int, int]


@dataclass(frozen=True)
class ReferenceConfig:
    strategy: str
    years_back: int
    pad_days: int
    min_rows: int
    fallback_to_trailing: bool


@dataclass(frozen=True)
class MonitoringConfig:
    reports_dir: Path
    current_days: int
    reference: ReferenceConfig
    min_scored_predictions: int
    rolling_mape_threshold: float
    drift_share_threshold: float


@dataclass(frozen=True)
class RetrainingConfig:
    cadence_days: int
    tune: bool
    keep_runs_days: int


@dataclass(frozen=True)
class StateConfig:
    """Where the state that cannot be recomputed lives. See ADR-008."""

    dir: Path
    prediction_log: str
    pipeline_state: str

    @property
    def prediction_log_path(self) -> Path:
        return self.dir / self.prediction_log

    @property
    def pipeline_state_path(self) -> Path:
        return self.dir / self.pipeline_state


@dataclass(frozen=True)
class ServingConfig:
    max_request_hours: int
    weather_forecast_days: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    ingestion: IngestionConfig
    weather: WeatherConfig
    features: FeaturesConfig
    validation: ValidationConfig
    model: ModelConfig
    promotion: PromotionConfig
    mlflow: MlflowConfig
    backtest: BacktestConfig
    evaluation: EvaluationConfig
    state: StateConfig
    serving: ServingConfig
    monitoring: MonitoringConfig
    retraining: RetrainingConfig


def _resolve(path: str) -> Path:
    """Make a configured path absolute, relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


SQLITE_PREFIX = "sqlite:///"


def _resolve_tracking_uri(uri: str) -> str:
    """Anchor a relative SQLite tracking URI to the repository root.

    `sqlite:///mlruns/mlflow.db` is relative to the *working directory*, so the same
    command run from a subdirectory silently creates a second, empty store — a fresh
    registry with no champion in it, and a training run that promotes itself against
    nothing. Every other configured path is resolved against the repo root; this one
    was the exception, which is how a stray mlflow.db came to sit in the project root.
    """
    if not uri.startswith(SQLITE_PREFIX):
        return uri
    return f"{SQLITE_PREFIX}{_resolve(uri.removeprefix(SQLITE_PREFIX))}"


def load_config(path: Path | None = None) -> Config:
    """Read and validate the pipeline configuration."""
    raw = yaml.safe_load((path or DEFAULT_CONFIG_PATH).read_text())

    data = raw["data"]
    weather = raw["weather"]
    cities = tuple(City(**c) for c in weather["cities"])
    total = sum(c.weight for c in cities)
    if total <= 0:
        raise ValueError("City weights must sum to a positive number")
    cities = tuple(City(name=c.name, lat=c.lat, lon=c.lon, weight=c.weight / total) for c in cities)

    return Config(
        data=DataConfig(
            country_code=data["country_code"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            timezone_local=data["timezone_local"],
            raw_dir=_resolve(data["raw_dir"]),
            processed_dir=_resolve(data["processed_dir"]),
            dataset_filename=data["dataset_filename"],
            synthetic_start=data["synthetic_start"],
            synthetic_end=data["synthetic_end"],
        ),
        ingestion=IngestionConfig(**raw["ingestion"]),
        weather=WeatherConfig(
            archive_url=weather["archive_url"],
            forecast_url=weather["forecast_url"],
            variables=tuple(weather["variables"]),
            cities=cities,
            request=RequestConfig(**weather["request"]),
            historical_forecast_url=weather.get("historical_forecast_url", ""),
            historical_forecast_lead_days=int(weather.get("historical_forecast_lead_days", 1)),
        ),
        features=FeaturesConfig(**raw["features"]),
        validation=ValidationConfig(**raw["validation"]),
        model=ModelConfig(
            **{
                k: v
                for k, v in raw["model"].items()
                if k not in ("horizons", "quantiles", "tuning")
            },
            horizons=tuple(raw["model"]["horizons"]),
            quantiles=tuple(raw["model"]["quantiles"]),
            tuning=TuningConfig(**raw["model"]["tuning"]),
        ),
        promotion=PromotionConfig(**raw["promotion"]),
        mlflow=MlflowConfig(
            **{k: v for k, v in raw["mlflow"].items() if k != "tracking_uri"},
            tracking_uri=_resolve_tracking_uri(raw["mlflow"]["tracking_uri"]),
        ),
        backtest=BacktestConfig(
            **{k: v for k, v in raw["backtest"].items() if k != "reports_dir"},
            reports_dir=_resolve(raw["backtest"]["reports_dir"]),
        ),
        evaluation=EvaluationConfig(peak_hours=tuple(raw["evaluation"]["peak_hours"])),
        state=StateConfig(
            **{k: v for k, v in raw["state"].items() if k != "dir"},
            dir=_resolve(raw["state"]["dir"]),
        ),
        serving=ServingConfig(**raw["serving"]),
        monitoring=MonitoringConfig(
            **{k: v for k, v in raw["monitoring"].items() if k not in ("reports_dir", "reference")},
            reports_dir=_resolve(raw["monitoring"]["reports_dir"]),
            reference=ReferenceConfig(**raw["monitoring"]["reference"]),
        ),
        retraining=RetrainingConfig(**raw["retraining"]),
    )
