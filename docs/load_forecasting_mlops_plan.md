# Implementation Plan: Short-Term Electricity Load Forecasting with MLOps (Polish Power System)

> **Goal:** A production-shaped, portfolio-ready MLOps system — a real forecasting *service* with a live data loop, automated retraining, and drift monitoring. Not a notebook reproduction.
>
> **Headline target:** day-ahead hourly load forecasts for the Polish bidding zone that **match or approach PSE's own official forecast**, with every prediction logged and measured against reality as it arrives.

---

## Architecture Overview

```
                    ┌──────────────────────────────────────────────┐
                    │  GitHub Actions  (scheduled daily cron)       │
                    └───────────────────────┬──────────────────────┘
                                            │ triggers
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                     ▼
  Data ingestion                    Evaluate + drift check               Retrain (if gated)
  • ENTSO-E actual load             • rolling MAPE vs new actuals        • LightGBM + Optuna
  • ENTSO-E TSO day-ahead forecast  • Evidently: data/prediction drift   • only promote if it
  • Open-Meteo weather              │                                      beats the gate
        │                           │                                     │
        ▼                           │                                     ▼
  Versioned dataset (DVC) ──────────┘                              MLflow tracking
        │                                                          + model registry
        ▼                                                                 │ promote "champion"
  Feature builder (leakage-safe) ───────────────────────────────►  FastAPI  /forecast
        │                                                          loads Production model
        │                                                          logs every prediction
        ▼                                                                 │
   Streamlit dashboard ◄──────────────────────────────────────────────────┘
   forecast vs actual · rolling error · drift report · live model version
```

**Core design principle:** the system continuously ingests fresh grid + weather data, retrains only behind explicit quality gates, and is benchmarked against the TSO's *own* forecast. Because every served prediction is persisted, performance and drift are measured against ground truth as it arrives — this is what makes the "live retraining" story real rather than a timer looping over a frozen CSV.

---

## Data Sources

Collect and validate these before writing model code — data quality defines project quality.

| Data | What it is | Source | Access |
|---|---|---|---|
| **Actual total load (PL)** | Realised system demand in MW — your **target** | ENTSO-E Transparency Platform | Free; register + request API token |
| **Day-ahead load forecast (PL)** | PSE's official forecast in MW — your **benchmark to beat** | ENTSO-E Transparency Platform | Same token |
| **Historical weather** | Temperature, wind, cloud, humidity for major PL cities | Open-Meteo Archive API | Free, **no key** |
| **Forecast weather** | Same variables on the forecast horizon — for *inference* features | Open-Meteo Forecast API | Free, **no key** |
| **Polish calendar** | Public holidays + bridge days | `holidays` library (PL) | Free, offline |

### Getting the ENTSO-E token (do this first — it has a lead time)
1. Register an account at `https://transparency.entsoe.eu/`.
2. Email `transparency@entsoe.eu` with **"Restful API access"** in the subject and your registered email address in the body.
3. Access is granted within ~3 working days; the token then appears in your account under *Web API Security Token*.

This 3-day wait is the only real scheduling constraint in the whole project — request the token on day one so it's ready when you need it.

### Why this data
Temperature is the single dominant driver of electricity demand (heating and cooling), and the calendar (holidays, weekends, the dead week between Christmas and New Year) causes the largest swings. The decisive advantage here: ENTSO-E hands you the **TSO's own day-ahead forecast for free**, so you get a credible, hard benchmark that most forecasting portfolios completely lack. "My model beats a naive baseline" is weak; "my model approaches the operational forecast PSE actually dispatches against" is a real claim.

### Canonical record schema (use from day one)
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LoadRecord:
    timestamp_utc: datetime          # ALWAYS store/compute in UTC, convert only for display
    load_mw: float | None            # actual total load (target source)
    tso_forecast_mw: float | None    # PSE day-ahead forecast (the benchmark)
    temp_c: float | None             # population-weighted temperature across PL cities
    wind_ms: float | None
    cloud_cover: float | None
    is_holiday: bool
    data_source_version: str         # DVC hash / ingestion run id — provenance matters
```

---

## Phase 0 — Project Setup

**Goal:** clean scaffolding before any model code touches the repo.

### Repository structure
```
pl-load-forecasting/
├── data/
│   ├── raw/              # API pulls, DVC-tracked, not in git
│   └── processed/        # feature tables, DVC-tracked, not in git
├── src/
│   ├── ingestion/        # ENTSO-E + Open-Meteo clients, dataset assembly
│   ├── features/         # leakage-safe feature builder (shared by train + serve)
│   ├── models/           # training, tuning, baselines
│   ├── api/              # FastAPI app
│   └── monitoring/       # Evidently drift + performance reports
├── pipelines/            # ingest / evaluate / check_drift / retrain entrypoints
├── notebooks/            # exploration only — never production logic here
├── tests/
├── docker/
├── .github/workflows/    # CI + scheduled retraining
├── dvc.yaml
├── .env.example
├── docker-compose.yml
└── README.md
```

### Tooling setup
```bash
uv init && uv add \
    entsoe-py openmeteo-requests requests-cache retry-requests \
    pandas numpy scikit-learn \
    lightgbm xgboost prophet \
    mlflow optuna \
    evidently \
    fastapi uvicorn pydantic \
    streamlit plotly \
    holidays pandera python-dotenv loguru \
    dvc

# Dev tooling
uv add --dev ruff black pytest pytest-asyncio

pre-commit install
```

### Architecture Decision Record (write before coding)
Create `docs/ADR.md` capturing four decisions with explicit reasoning:
1. **LightGBM as the primary model, not Prophet/SARIMA.** It absorbs dozens of engineered calendar/weather/lag features, captures nonlinear temperature effects, and trains in seconds. Prophet and SARIMA stay in the repo as *interpretable baselines*, not the production model.
2. **Direct per-horizon forecasting, not recursive.** Recursive (feed predictions back as inputs) accumulates error across 24 steps and invites subtle leakage. A model trained for the exact horizon you serve is cleaner and easier to evaluate honestly.
3. **MLflow registry + DVC, not ad-hoc artifact files.** You need model lineage (which data + features + params produced this model), one-line rollback to a previous champion, and reproducible datasets. Hand-saved `model_final_v3.pkl` files are the thing this project is meant to prove you've outgrown.
4. **Evidently for drift, not a hand-rolled check.** Distribution drift, prediction drift, and rolling performance in one library that emits shareable HTML reports — and the report *is* a portfolio artifact.

Writing ADRs is what professional teams do; including one is an immediately visible signal of engineering maturity.

---

## Phase 1 — Data Ingestion & Storage

**Goal:** a clean, versioned historical dataset joining load, the TSO forecast, and weather on a single UTC hourly index.

### ENTSO-E: pull the target *and* the benchmark
```python
import os
import pandas as pd
from entsoe import EntsoePandasClient

client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])

start = pd.Timestamp("2019-01-01", tz="Europe/Warsaw")
end   = pd.Timestamp("2024-01-01", tz="Europe/Warsaw")

# entsoe-py auto-splits requests longer than one year (the API caps each call at 1 year)
actual_load  = client.query_load("PL", start=start, end=end)           # MW, your target
tso_forecast = client.query_load_forecast("PL", start=start, end=end)  # MW, PSE day-ahead — your benchmark
```
Immediately convert both to a single **UTC** hourly index and resample (load is often 15-min). Keep `Europe/Warsaw` only for human-facing display.

### Open-Meteo: free weather, no key
Pull historical weather for several major cities and combine into a **population-weighted** national temperature — a small touch that meaningfully improves accuracy and is easy to explain in an interview.

```python
import requests

CITIES = {  # (lat, lon, population weight)
    "Warsaw":  (52.23, 21.01, 0.30),
    "Krakow":  (50.06, 19.94, 0.16),
    "Lodz":    (51.76, 19.46, 0.12),
    "Wroclaw": (51.11, 17.04, 0.12),
    "Poznan":  (52.41, 16.93, 0.11),
    "Gdansk":  (54.35, 18.65, 0.10),
    "Szczecin":(53.43, 14.55, 0.09),
}

def fetch_weather_history(lat, lon, start, end):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m,cloud_cover,relative_humidity_2m",
        "timezone": "UTC",
    })
    return r.json()["hourly"]
```
For **serving**, the same variables come from the forecast endpoint `https://api.open-meteo.com/v1/forecast` — see the train-serve note in Phase 2.

### Validation + versioning
Write `validate_dataset.py` (use `pandera` for schema + range checks) that flags: timestamp gaps, load values outside plausible bounds (PL system load roughly 10–28 GW), and any hour where load is present but the TSO forecast is missing. Then track the raw and processed datasets with **DVC** so a change to ingestion logic produces a new, hashed dataset version.

**Deliverable:** `data/processed/dataset.parquet` — a UTC-hourly table of `load_mw`, `tso_forecast_mw`, weather, and `is_holiday`, DVC-tracked, with a validation report printing per-year row counts and gap locations.

---

## Phase 2 — Feature Engineering (read this carefully)

This is the step most tutorials get **catastrophically wrong**, and the single most likely thing to silently invalidate your whole project. The failure has a name: **target leakage**.

### The leakage rule
You are forecasting `load[t + H]` (e.g. H = 24 hours ahead). At prediction time `t`, you only know information available *up to and including* `t`. Therefore **every lagged-load feature must be at least H hours old relative to the target.** If you forecast 24h ahead but feed `load[t+1]` as a feature, your offline metrics will look spectacular and the model will be useless in production, because that value doesn't exist yet when you actually run it.

The one always-safe lag is **168 hours (same hour, one week ago)** — older than any reasonable day-ahead horizon.

```python
import numpy as np
import pandas as pd
import holidays

PL_HOLIDAYS = holidays.Poland()

def make_features(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """df is UTC-hourly with columns: load_mw, temp_c, wind_ms, cloud_cover.
    Returns a feature frame aligned so each row predicts load_mw[t + horizon]."""
    f = pd.DataFrame(index=df.index)

    # --- Calendar features: always known in advance, zero leakage risk ---
    f["hour"]  = df.index.hour
    f["dow"]   = df.index.dayofweek
    f["month"] = df.index.month
    f["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    f["is_holiday"] = df.index.normalize().isin(PL_HOLIDAYS).astype(int)

    # Cyclical encoding so the model knows hour 23 is next to hour 0
    f["hour_sin"] = np.sin(2*np.pi*f["hour"]/24)
    f["hour_cos"] = np.cos(2*np.pi*f["hour"]/24)
    f["dow_sin"]  = np.sin(2*np.pi*f["dow"]/7)
    f["dow_cos"]  = np.cos(2*np.pi*f["dow"]/7)

    # --- Lagged load: MUST be >= horizon old ---
    for lag in sorted({horizon, horizon + 1, horizon + 24, 168, 168 + horizon}):
        f[f"load_lag_{lag}"] = df["load_mw"].shift(lag)

    # Rolling stats computed only on data at least `horizon` old
    base = df["load_mw"].shift(horizon)
    f["load_roll_mean_24"] = base.rolling(24).mean()
    f["load_roll_std_24"]  = base.rolling(24).std()

    # --- Weather features ---
    # At inference these come from the FORECAST API, so train on forecast-aligned
    # weather (see note below), not perfect observed weather.
    f["temp_c"]      = df["temp_c"]
    f["temp_sq"]     = df["temp_c"]**2          # U-shaped: load rises in both cold and heat
    f["wind_ms"]     = df["wind_ms"]
    f["cloud_cover"] = df["cloud_cover"]

    # --- Target ---
    f["target"] = df["load_mw"].shift(-horizon)
    return f.dropna()
```

### The train–serve weather skew (a production-thinking differentiator)
At training time you have *observed* weather. At inference you only have *forecast* weather, which is less accurate. If you train on perfect observed weather and serve on forecasts, your live accuracy will be worse than your backtest — a classic, invisible degradation. Handle it honestly: either train on Open-Meteo's *historical forecast* (reforecast) data so train and serve distributions match, or at minimum **quantify** the gap by backtesting once with observed and once with forecast weather and reporting both. Naming and addressing this in your README is exactly the kind of nuance that separates serious work from a tutorial.

**Deliverable:** `features/builder.py` with a single `make_features()` used identically by training and serving (never two copies — that drift is its own bug class), plus `tests/test_no_leakage.py` that asserts no feature column correlates with the target through an impossibly-recent lag.

---

## Phase 3 — Baselines & Model Training

**Goal:** establish honest baselines first, then beat them.

### Baselines (build these before the fancy model)
1. **Naive seasonal:** `load[t+H] ≈ load[t+H-168]` (same hour last week). Shockingly hard to beat; if your ML model can't, something is wrong.
2. **The TSO forecast:** PSE's `query_load_forecast` values, already in your dataset. This is the bar that matters.
3. **Linear / SARIMA:** a simple statistical reference for the writeup.

### Primary model: LightGBM
```python
import lightgbm as lgb

def train_lgbm(X_train, y_train, X_val, y_val, params):
    train_set = lgb.Dataset(X_train, y_train)
    val_set   = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(
        params, train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
```

### Hyperparameter tuning with Optuna
```python
import optuna

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mape",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("mcs", 20, 200),
    }
    model = train_lgbm(X_tr, y_tr, X_val, y_val, params)
    return mape(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```
Crucially, the validation split here must be **chronological** (validate on a later period than you train on), never random — see Phase 5.

### Probabilistic forecasts (differentiator, do this if time allows)
Utilities care about *uncertainty*, not just a point estimate. Train LightGBM quantile models (`objective="quantile", alpha=0.1` and `0.9`) to produce P10/P50/P90 bands, or wrap the point model with conformal prediction. Evaluate with **pinball loss**. A forecast that says "load will be 21.4 GW, 90% confidence interval 20.1–22.7 GW" is a notably more mature deliverable than a bare number.

**Deliverable:** a training script that logs every run to MLflow (Phase 4) and a champion LightGBM model that beats the naive baseline and is benchmarked against the TSO forecast (Phase 5).

---

## Phase 4 — Experiment Tracking & Model Registry (MLflow)

This is where "I trained a model" becomes "I can tell you exactly which data, features, and parameters produced the model currently in production, and roll back in one command."

```python
import mlflow, mlflow.lightgbm
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("pl-load-forecasting")

with mlflow.start_run(run_name="lgbm_h24_weather_tuned"):
    mlflow.log_params(best_params)
    mlflow.log_param("horizon", 24)
    mlflow.log_param("dataset_version", dvc_hash)   # provenance link to DVC

    model = train_lgbm(X_tr, y_tr, X_val, y_val, best_params)

    mlflow.log_metric("mape", val_mape)
    mlflow.log_metric("rmse_mw", val_rmse)
    mlflow.log_metric("mape_vs_tso", val_mape - tso_mape)   # negative = we beat PSE

    # Log feature importance plot as an artifact
    mlflow.log_figure(plot_importance(model), "feature_importance.png")

    signature = infer_signature(X_val, model.predict(X_val))
    mlflow.lightgbm.log_model(
        model, "model", signature=signature,
        registered_model_name="pl_load_lgbm",
    )
```

### Model registry + quality gate
Promote a new model to the `Production`/`champion` alias **only** through an explicit gate, not by default:

```python
def passes_gate(candidate_metrics, naive_mape, current_prod_mape) -> bool:
    return (
        candidate_metrics["mape"] < naive_mape           # must beat naive
        and candidate_metrics["mape"] <= current_prod_mape * 1.02  # no big regression
    )
```
This gate is the heart of safe automated retraining (Phase 7): a scheduled retrain that produces a *worse* model must not silently replace a good one.

**Deliverable:** an MLflow tracking server (containerized in Phase 10) with a populated experiment, a registered model with stage transitions, and a documented promotion gate.

---

## Phase 5 — Backtesting & Evaluation Framework

The phase that turns a "cool project" into a "serious candidate" signal. The cardinal rule: **never use a random train/test split on a time series.** Random splits let the model peek at the future and produce fantasy metrics.

### Rolling-origin (expanding window) backtest
```
Train on  [────────────]                 → predict next block
Train on  [──────────────────]           → predict next block
Train on  [────────────────────────]     → predict next block
```
Step the origin forward (e.g. weekly), retraining periodically, and accumulate out-of-sample predictions across a full year so seasonality is represented.

### Metrics — report several, not just MAPE
- **MAPE** (industry standard for load) — but pair it with absolute error so MW context is visible.
- **RMSE / MAE in MW** — what the answer actually means physically.
- **Pinball loss** — if you built the probabilistic version.
- **Per-segment breakdown** — error on **peak vs off-peak hours**, **weekday vs weekend**, and **holidays**. This is where you find your model's real character (almost everyone's model is worst on holidays and the Christmas–New Year week).

### The benchmark table (your headline artifact)
Fill this with real numbers from your backtest. It belongs in the README and the blog post.

| Model | MAPE | RMSE (MW) | MAE (MW) | vs PSE forecast |
|---|---|---|---|---|
| Naive seasonal (load[t−168]) | — | — | — | — |
| **PSE day-ahead forecast (ENTSO-E)** | — | — | — | *(the benchmark)* |
| SARIMA | — | — | — | — |
| LightGBM (calendar only) | — | — | — | — |
| LightGBM (+ weather) | — | — | — | — |
| LightGBM (+ weather, tuned) | — | — | — | — |

A note on honesty: beating a TSO's operational forecast *overall* is genuinely hard — they have data you don't. Approaching it is already a strong result, and beating it on **specific segments** (holidays, mild-weather weekends) is a realistic, defensible win. Report where you win *and* where you lose; "interviewers will ask how you measured that it works" — point here.

**Deliverable:** `eval/backtest.py` producing the table above plus per-segment plots, all logged to MLflow.

---

## Phase 6 — FastAPI Serving

**Goal:** a real prediction service that loads the current champion from the registry and logs everything it does.

### Endpoints
```python
# src/api/main.py

POST /forecast
  Body: { "target_date": "2026-07-01", "horizon_hours": 24 }
  Response: { "predictions": [{ "timestamp": ..., "load_mw": 21380,
                                "p10": 20120, "p90": 22640 }, ...],
              "model_version": "pl_load_lgbm/7" }

GET  /health
  Response: { "status": "ok", "model_version": str, "loaded_at": ... }

POST /reload-model     # re-pull the current Production model from the registry
  Response: { "previous": str, "current": str }
```

### Serving logic
```python
import mlflow.pyfunc
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
_model = mlflow.pyfunc.load_model("models:/pl_load_lgbm@champion")

@app.post("/forecast")
def forecast(req: ForecastRequest):
    weather = fetch_forecast_weather(req.target_date)   # Open-Meteo forecast API
    features = make_features_for_inference(req.target_date, req.horizon_hours, weather)
    preds = _model.predict(features)

    log_prediction(req, preds, model_version=_model.metadata.run_id)  # persist for monitoring
    return JSONResponse(build_response(preds))
```

### Prediction logging (this is what makes monitoring possible)
Persist every prediction — timestamp, target datetime, predicted value, model version, input features — to a small store (SQLite or Postgres). When the actual load for that hour later appears in ENTSO-E, you join it back and you can compute *real* production error. Without this logging, Phase 8's drift and performance monitoring has nothing to measure.

**Deliverable:** a FastAPI service that loads the champion model, serves point + interval forecasts, exposes `/health`, and logs every prediction with its model version.

---

## Phase 7 — Orchestration & Automated Retraining

**Goal:** the live loop. A scheduled job pulls fresh data, scores recent forecasts against reality, checks for drift, and retrains behind the quality gate.

### GitHub Actions: free scheduled orchestration
```yaml
# .github/workflows/daily-update.yml
name: daily-ingest-evaluate-retrain
on:
  schedule:
    - cron: "30 5 * * *"     # 05:30 UTC daily (cron timing is best-effort)
  workflow_dispatch:          # manual trigger button

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e .

      - run: python -m pipelines.ingest        # pull yesterday's actuals + new weather
      - run: python -m pipelines.evaluate       # join logged predictions to actuals, rolling MAPE
      - run: python -m pipelines.check_drift     # Evidently; writes a retrain flag if triggered
      - run: python -m pipelines.retrain_if_needed   # retrain + gate + maybe promote
    env:
      ENTSOE_API_KEY: ${{ secrets.ENTSOE_API_KEY }}
```

### Retraining trigger logic
Don't retrain blindly every day. Retrain when **either**: (a) a scheduled cadence elapses (e.g. weekly), **or** (b) drift/performance degradation is detected (Phase 8). Then always run the Phase 4 promotion gate before replacing the champion.

```python
def retrain_if_needed(state):
    if not (state.scheduled_due or state.drift_detected):
        return "skipped"
    candidate = train_candidate(latest_dataset())
    if passes_gate(candidate.metrics, naive_mape(), current_prod_mape()):
        promote(candidate)   # set @champion alias in MLflow registry
        return "promoted"
    return "trained_not_promoted"   # logged, but production untouched
```

**Heavier alternative (mention in README, don't necessarily build):** Prefect or Airflow for orchestration if you outgrow Actions. Stating that you chose Actions deliberately for cost/simplicity, and know when you'd reach for Prefect, is itself a good architectural signal.

**Deliverable:** a scheduled workflow that runs end-to-end on GitHub's runners, plus `pipelines/` entrypoints each runnable locally for debugging.

---

## Phase 8 — Monitoring & Drift Detection (Evidently)

**Goal:** the standout MLOps component — detect when the world has changed and *act* on it. This is the piece most portfolio projects skip entirely.

### Three things to monitor
1. **Data drift** — have input feature distributions shifted? (e.g. a temperature regime change between seasons, or a structural shift in demand.)
2. **Prediction drift** — has the distribution of your model's outputs moved?
3. **Performance** — rolling MAPE of *served* predictions against actuals as they arrive (this is why Phase 6 logging matters).

```python
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset

def drift_report(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    result = report.run(reference_data=reference_df, current_data=current_df)
    result.save_html("monitoring/latest_report.html")   # a portfolio artifact in itself
    return result
```

### Wire detection to action (the part that matters)
A drift report nobody reads is theatre. Close the loop:
```python
drift = drift_report(reference_window, current_window)
if drift.data_drift_detected or rolling_mape_7d() > MAPE_THRESHOLD:
    set_retrain_flag()        # consumed by pipelines.retrain_if_needed in Phase 7
    notify("Drift/perf threshold breached — retrain triggered")
```

**Deliverable:** `monitoring/` producing dated Evidently HTML reports, a rolling-performance time series, and a threshold that feeds the retraining trigger.

---

## Phase 9 — Streamlit Dashboard

**Goal:** make the whole system visible in 5 seconds to anyone who opens it.

### Features (in priority order)
1. **Forecast vs actual** — overlay the next-24h forecast (with P10–P90 band) on recent actuals.
2. **Rolling error** — MAPE over the last 7/30 days, with the TSO forecast's error on the same axes so the comparison is immediate and honest.
3. **Drift status** — embed or link the latest Evidently report; a clear green/red indicator.
4. **Feature importance** — from the current champion.
5. **Live model card** — which model version is serving, when it was promoted, on what dataset hash.

### The detail that sells it
Plotting **your forecast and PSE's forecast against the same actuals** is the single most convincing visual in the project. A reviewer sees instantly that you're benchmarking against a real operational baseline, not a strawman. Without that side-by-side, they have no frame of reference for whether your MAPE is good.

**Deliverable:** `docker compose up` starts MLflow + API + dashboard with no manual steps. Test it on a clean machine before sending applications.

---

## Phase 10 — Containerization & Deployment

### Docker Compose (the local production-like stack)
```yaml
services:
  mlflow-db:
    image: postgres:16
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes: [ mlflow_pg:/var/lib/postgresql/data ]

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri postgresql://mlflow:mlflow@mlflow-db/mlflow
      --artifacts-destination /mlartifacts
    ports: [ "5000:5000" ]
    volumes: [ mlflow_artifacts:/mlartifacts ]
    depends_on: [ mlflow-db ]

  api:
    build: ./docker/api
    env_file: .env
    ports: [ "8000:8000" ]
    depends_on: [ mlflow ]

  dashboard:
    build: ./docker/dashboard
    ports: [ "8501:8501" ]
    depends_on: [ api ]

volumes:
  mlflow_pg:
  mlflow_artifacts:
```

### Deployment options (pick one)
- **Hugging Face Spaces (Docker SDK)** — zero cost, shareable link, high recruiter recognition. Cleanest for the *dashboard*; point it at a hosted API or bundle a lightweight inference path.
- **Render / Railway / Fly.io free tiers** — can host the FastAPI service; mind that free instances sleep when idle.
- **Azure free tier** — more setup overhead, but Azure is positive signal for Polish enterprise recruiters who use it. (As covered in our earlier discussion, watch free-tier limits so an always-on instance doesn't quietly start billing.)

Note in your README that the **scheduled retraining runs on GitHub Actions for free regardless of where the API is hosted** — the live loop doesn't depend on paid compute.

**Deliverable:** one-command local stack via Docker Compose, plus a deployed, shareable URL for at least the dashboard.

---

## Phase 11 — Documentation & Polish

### README sections (non-negotiable)
1. One-paragraph plain-language description: what it forecasts, for whom, and why day-ahead load matters (grid balancing, the day-ahead market).
2. Architecture diagram (the MLOps loop at the top of this plan — redraw in Excalidraw/Mermaid, export PNG).
3. The **benchmark table** vs the PSE forecast, with per-segment breakdown.
4. Data inventory: sources, horizon, resolution, date range.
5. The **train–serve weather skew** discussion — shows production thinking.
6. Setup: `git clone` → `docker compose up` in three commands.
7. A screenshot of forecast-vs-actual with the uncertainty band.
8. "What I'd do with more time" — 3 honest bullets (e.g. multi-zone, probabilistic calibration, Prefect orchestration).

### Blog post (400–600 words)
Structure: problem → approach → the one decision worth explaining (leakage-safe multi-horizon features, *or* benchmarking against the TSO) → one surprising finding from the backtest → link to the live demo. Publish before applications go out; link it from the README.

### Demo video (90 seconds, no narration needed)
Show: the dashboard's forecast-vs-actual with the P10–P90 band, then the rolling-MAPE chart with your line tracking PSE's, then trigger `workflow_dispatch` and show a fresh model getting evaluated against the gate. Let the live loop speak for itself.

---

## Common Pitfalls

**Target leakage via short lags — the project killer.** Any lagged-load feature newer than your forecast horizon leaks the future. Backtest metrics look incredible; production is garbage. Gate against it with a test. (The 168h "same hour last week" lag is always safe.)

**Train–serve weather skew.** Training on observed weather but serving on forecasts silently degrades live accuracy. Train on forecast-aligned weather or explicitly quantify and report the gap.

**Timezone & DST handling.** Europe/Warsaw has a **missing hour** every spring (clocks jump forward) and a **duplicated hour** every autumn (clocks fall back). ENTSO-E timestamps are UTC — keep everything UTC internally and convert only for display. Resampling naively across a DST boundary creates NaNs or duplicate-index errors that are maddening to debug later.

**ENTSO-E data is revised.** "Actual" load values can be updated after first publication, and some hours publish late or not at all. Re-pull a trailing window on each ingestion rather than assuming yesterday's data is final and immutable.

**Random train/test split on a time series.** Produces fantasy metrics by letting the model see the future. Always chronological holdout / rolling origin.

**Forecasting-horizon mismatch.** A model trained on a 1-hour-ahead target evaluated as if it forecasts 24h ahead is meaningless. Train for the exact horizon you serve.

**No data versioning.** If you change feature logic mid-project without DVC, you can't reproduce earlier results or explain what produced a given model. Tag datasets with a config hash.

**Drift detected but nothing happens.** A drift report that doesn't feed retraining or alerting is decoration. Close the loop explicitly (Phase 8 → Phase 7).

**MAPE reported alone.** It hides absolute magnitude and behaves badly on low-load troughs. Always pair it with RMSE/MAE in MW and a per-segment breakdown.

---

## Phase Summary

| Phase | Key Deliverable |
|---|---|
| 0 — Setup | Repo, ADR, locked deps, Docker skeleton, ENTSO-E token requested |
| 1 — Ingestion | UTC-hourly dataset (load + TSO forecast + weather), DVC-tracked, validated |
| 2 — Features | Leakage-safe feature builder shared by train/serve, leakage test passing |
| 3 — Training | Baselines + tuned LightGBM (+ optional probabilistic) |
| 4 — Tracking | MLflow experiments + registry + documented promotion gate |
| 5 — Evaluation | Rolling-origin backtest, benchmark table vs PSE, per-segment breakdown |
| 6 — Serving | FastAPI loading champion model, interval forecasts, prediction logging |
| 7 — Orchestration | GitHub Actions daily loop: ingest → evaluate → drift → gated retrain |
| 8 — Monitoring | Evidently drift + rolling-performance reports wired to retraining |
| 9 — Dashboard | Streamlit: forecast vs actual + uncertainty + drift + live model card |
| 10 — Deploy | One-command Compose stack + shareable hosted URL |
| 11 — Polish | README, blog post, demo video |

---

## Tech Stack Summary

| Layer | Tool | Why |
|---|---|---|
| Grid data | `entsoe-py` (EntsoePandasClient) | Returns pandas; gives both actual load and the TSO forecast |
| Weather data | Open-Meteo (archive + forecast APIs) | Free, no key, historical *and* forecast endpoints |
| Calendar | `holidays` (PL) | Offline Polish public-holiday calendar |
| Modeling | LightGBM (primary), Prophet/SARIMA (baselines) | Handles engineered features + nonlinear temperature effects, fast |
| Tuning | Optuna | Efficient search with chronological validation |
| Experiment tracking | MLflow tracking + model registry | Lineage, metrics, one-command rollback |
| Data versioning | DVC | Reproducible, hashed datasets |
| Drift / monitoring | Evidently | Data + prediction drift + performance, HTML reports |
| Serving | FastAPI + uvicorn | Async, clean Pydantic schemas, loads champion from registry |
| Orchestration / CI-CD | GitHub Actions | Free scheduled retraining + tests on PR |
| Data validation | pandera | Schema + range checks on ingested data |
| Dashboard | Streamlit + Plotly | Fast to build, good enough for the portfolio demo |
| Containerization | Docker + Docker Compose | Reproducibility, portfolio standard |
| Dependency mgmt | uv | Modern, fast, reproducible |
| Code quality | ruff + black + pre-commit | Professional repo hygiene |

---

### Suggested CV line
> *Built an end-to-end MLOps pipeline forecasting day-ahead Polish electricity demand (LightGBM), benchmarked against the TSO's official forecast, with MLflow experiment tracking and model registry, automated daily ingestion and gated retraining via GitHub Actions, Evidently-based drift monitoring, and a FastAPI service + Streamlit dashboard orchestrated with Docker Compose. Headline: MAPE of X% on rolling-origin backtest, within Y% of the operational PSE forecast.*
