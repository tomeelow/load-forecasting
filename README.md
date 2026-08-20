# Day-Ahead Electricity Load Forecasting — Polish Bidding Zone

> **Status:** Phases 0–8 complete — the loop is closed. Real ENTSO-E data is ingested
> daily, forecasts are served from the registered champion and logged, served
> predictions are scored against actuals as they arrive, drift is monitored against a
> season-matched reference, and retraining runs behind the promotion gate without
> anyone present. The Streamlit dashboard and the deployment target are what remain.

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

> ⚠️ **This table is stale: the numbers below came from synthetic data.** Real ENTSO-E
> data has since been ingested (2019-01-01 to date, 66,768 hourly rows) and the champion
> is trained on it, but the rolling-origin backtest has not yet been re-run — that is a
> ~20 minute job scheduled for the next pass. Until it is, read this table as evidence
> that the evaluation machinery works, not as a result about Polish demand. The
> synthetic "PSE forecast" in particular was constructed as *the actual load plus small
> autocorrelated noise*, making it a near-oracle nothing could beat; the real PSE
> forecast has no such advantage. Regenerate with:
>
> ```bash
> uv run python -m pipelines.backtest
> ```

Rolling-origin backtest, 24-hour horizon: 26 expanding-window origins, **8,736
out-of-sample hours covering a full year** (2023-01-09 → 2024-01-08), with a 24-hour
embargo between each training window and the block it predicts. Full report, including
the per-segment breakdown, in [`reports/benchmark_h24.md`](reports/benchmark_h24.md).

| Model | MAPE | RMSE (MW) | MAE (MW) | vs PSE forecast |
|---|---|---|---|---|
| Naive seasonal (`load[T−168]`) | 3.405% | 837 | 636 | +2.286 pp worse |
| Linear (calendar + weather) | 4.412% | 1012 | 828 | +3.293 pp worse |
| **PSE day-ahead forecast (ENTSO-E)** | 1.119% | 260 | 208 | *(the benchmark)* |
| LightGBM (calendar + lags) | 2.402% | 581 | 451 | +1.283 pp worse |
| LightGBM (+ weather) | 1.868% | 446 | 348 | +0.748 pp worse |
| LightGBM (+ weather, tuned) | 1.837% | 436 | 343 | +0.718 pp worse |

P10/P50/P90 pinball losses: 82.2 / 171.7 / 87.5 MW.

### Where the model loses

It loses everywhere, by 0.66–0.87 pp, for the reason given in the warning above. The
*shape* of the losses is still informative and matches what the literature predicts:

| Segment | Hours | Model MAPE | PSE MAPE | Gap |
|---|---|---|---|---|
| Holidays | 312 | 2.166% | 1.300% | +0.866 pp |
| Christmas–New Year | 216 | 1.908% | 1.112% | +0.796 pp |
| Weekends | 2496 | 2.041% | 1.249% | +0.792 pp |
| Off-peak hours | 3276 | 2.005% | 1.219% | +0.787 pp |
| Weekdays | 6240 | 1.756% | 1.067% | +0.688 pp |
| Peak hours | 5460 | 1.736% | 1.059% | +0.677 pp |

The worst segments are holidays and the Christmas–New Year week — the two the plan
predicted, and the two with the fewest examples to learn from. Weather is worth
0.53 pp of MAPE (2.402% → 1.868%); tuning adds a further 0.03 pp, which is small enough
to be worth knowing before anyone spends a day on hyperparameters.

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

Train the baselines and every LightGBM variant, log them to MLflow and run the
promotion gate:

```bash
uv run python -m pipelines.train
```

Produce the benchmark table from a rolling-origin backtest (takes ~20 minutes: each
origin is a full refit of every variant — use `--max-splits 3` for a quick check):

```bash
uv run python -m pipelines.backtest
```

Browse the tracked runs, the registered model and the champion alias:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Serve forecasts from the registered champion:

```bash
uv run uvicorn src.api.main:app --port 8000
```

The four pipelines of the scheduled loop, each runnable on its own for debugging:

```bash
uv run python -m pipelines.ingest
```

```bash
uv run python -m pipelines.evaluate
```

```bash
uv run python -m pipelines.check_drift
```

```bash
uv run python -m pipelines.retrain_if_needed
```

They run in that order daily via
[`.github/workflows/daily-loop.yml`](.github/workflows/daily-loop.yml), which also
carries state between runs — see [ADR-008](docs/ADR.md). Trigger one by hand with:

```bash
gh workflow run daily-loop.yml -f force_retrain=false
```

Without an `ENTSOE_API_KEY`, both pipelines fall back to the synthetic generator in
[src/synthetic.py](src/synthetic.py) so the loop runs end to end. Those runs are named
`*_synthetic`, tagged `data_source=synthetic` in MLflow, and their reports carry a
warning banner — a metric from invented data measures the plumbing, nothing else.

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
src/models/      baselines, LightGBM, tuning, promotion gate, MLflow tracking
src/evaluation/  metrics, chronological splits, rolling-origin backtest
src/api/         FastAPI service                        (Phase 6, empty)
src/monitoring/  Evidently drift + performance reports  (Phase 8, empty)
pipelines/       runnable entrypoints (ingest, train, backtest)
reports/         generated benchmark tables
config/          pipeline parameters (dates, cities, horizon, bounds)
tests/           leakage, DST, validation and ingestion tests
docs/            plan + architecture decision record
```

## Decisions

Nine load-bearing choices are recorded with their reasoning in
[`docs/ADR.md`](docs/ADR.md): LightGBM over Prophet/SARIMA, direct per-horizon
forecasting over recursive, an MLflow registry plus a dataset content hash over ad-hoc
artifact files (and over DVC, which would need a remote nobody is paying for),
Evidently over a hand-rolled drift check, indexing feature rows by target hour,
embargoing one horizon between training and test blocks, tuning once per backtest
rather than per origin, carrying pipeline state on a force-pushed orphan branch, and
measuring drift against the same weeks in previous years rather than against last
fortnight.
