# Day-Ahead Electricity Load Forecasting — Polish Bidding Zone

> **Status:** Phases 0–2 complete (scaffolding, ingestion, leakage-safe features).
> Training, tracking, serving, monitoring and the dashboard are not built yet.
> Sections below marked **TODO** are placeholders until those phases land.

## What this is

Hourly, day-ahead forecasts of total electricity demand for the Polish bidding zone
(PSE's control area). Day-ahead load is the number the grid operator dispatches
against and the number the day-ahead market clears on: generation must be scheduled
a day in advance, so an error of a few hundred MW is paid for in balancing costs.

The system is built as a production loop rather than a notebook — ingestion →
versioned dataset → leakage-safe features → tracked training → gated promotion →
serving → drift monitoring → automated retraining. The distinguishing constraint is
the benchmark: ENTSO-E publishes **PSE's own day-ahead forecast** alongside the
actuals, so every result is reported against the forecast the TSO actually operates
on, not only against a naive baseline.

## Architecture

**TODO** — Mermaid/Excalidraw diagram of the MLOps loop (see
[`docs/load_forecasting_mlops_plan.md`](docs/load_forecasting_mlops_plan.md) for the
ASCII version this will be redrawn from).

## Benchmark results

**TODO** — no model has been trained yet. This table is filled from the
rolling-origin backtest in Phase 5, and reports per-segment breakdowns (peak vs
off-peak, weekday vs weekend, holidays) including the segments where the model
*loses* to PSE.

| Model | MAPE | RMSE (MW) | MAE (MW) | vs PSE forecast |
|---|---|---|---|---|
| Naive seasonal (`load[t−168]`) | TODO | TODO | TODO | TODO |
| **PSE day-ahead forecast (ENTSO-E)** | TODO | TODO | TODO | *(the benchmark)* |
| LightGBM (calendar only) | TODO | TODO | TODO | TODO |
| LightGBM (+ weather) | TODO | TODO | TODO | TODO |
| LightGBM (+ weather, tuned) | TODO | TODO | TODO | TODO |

## Data inventory

| Data | Source | Resolution | Range | Notes |
|---|---|---|---|---|
| Actual total load (PL) | ENTSO-E Transparency Platform | native 15-min or 60-min → resampled to hourly | configured in [`config/config.yaml`](config/config.yaml) | the target |
| Day-ahead load forecast (PL) | ENTSO-E Transparency Platform | hourly | same | the benchmark (PSE) |
| Historical weather | Open-Meteo Archive API | hourly | same | 7 cities, population-weighted |
| Forecast weather | Open-Meteo Forecast API | hourly | rolling | used at serving time |
| Public holidays | `holidays` (PL), offline | daily | — | evaluated on the **Europe/Warsaw** calendar date |

Everything is stored and computed on a **UTC** hourly index. `Europe/Warsaw` is used
only where the local calendar is semantically required (holiday flags) and for display.

## The train–serve weather skew

At training time the weather features come from Open-Meteo's *archive* (observed
weather). At serving time they can only come from the *forecast* endpoint, which is
less accurate — so a model trained on perfect weather will do measurably worse live
than its backtest suggests, with no visible failure.

The ingestion layer keeps both paths available
([`src/ingestion/weather_client.py`](src/ingestion/weather_client.py) exposes an
archive fetch and a forecast fetch producing the identical column set), and the
feature builder is horizon-parametrised so the same code path serves both.

**TODO** — quantify the gap: backtest once with observed weather and once with
forecast-aligned weather, and report both numbers here rather than only the
flattering one.

Weather features are aligned to the **hour being predicted**, not to the moment the
forecast is made. That is legitimate rather than leakage: Open-Meteo publishes the
forecast for hour `T` well before `T` arrives, so the value exists at prediction time.
Nothing equivalent is true of the load series, which is exactly why this project is
interesting. See [ADR-005](docs/ADR.md) for the row-indexing convention that makes the
alignment explicit.

## Setup

```bash
git clone <repo> && cd load-forecasting
uv sync
uv run pre-commit install
cp .env.example .env   # then paste your ENTSO-E token into ENTSOE_API_KEY
```

On macOS, LightGBM needs the OpenMP runtime, which is not a Python package:

```bash
brew install libomp
```

Run the test suite (no network and no `.env` required — the suite runs against a
synthetic Polish load series):

```bash
uv run pytest
```

Ingest the full configured history, then keep it current:

```bash
uv run python -m pipelines.ingest --full
```

```bash
uv run python -m pipelines.ingest
```

The second form re-pulls only a trailing window (`ingestion.trailing_repull_days` in
config) and merges it over the existing dataset, because ENTSO-E revises "actual"
values after first publication.

**TODO** — `docker compose up` for the full stack (Phase 10).

## Screenshot

**TODO** — forecast vs actual with the P10–P90 band (Phase 9).

## What I'd do with more time

**TODO** — to be written honestly once there are results to be honest about.
Current candidates: multi-zone (CZ/SK/DE) transfer, probabilistic calibration via
conformal prediction, and Prefect instead of GitHub Actions once the DAG outgrows a
linear job.

## Repository layout

```
src/ingestion/   ENTSO-E + Open-Meteo clients, dataset assembly, validation
src/features/    leakage-safe feature builder (shared by training and serving)
src/models/      training, tuning, baselines            (Phase 3, empty)
src/api/         FastAPI service                        (Phase 6, empty)
src/monitoring/  Evidently drift + performance reports  (Phase 8, empty)
pipelines/       runnable entrypoints (ingest, …)
config/          pipeline parameters (dates, cities, horizon, bounds)
tests/           leakage, DST, validation and ingestion tests
docs/            plan + architecture decision record
```

## Decisions

Four load-bearing choices are recorded with their reasoning in
[`docs/ADR.md`](docs/ADR.md): LightGBM over Prophet/SARIMA, direct per-horizon
forecasting over recursive, MLflow + DVC over ad-hoc artifact files, and Evidently
over a hand-rolled drift check.
