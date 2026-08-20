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

Two numbers, measuring different things, and the difference matters. The **rolling-origin
backtest** is the reported result: a full year of out-of-sample hours, refitted at 26
origins. The **champion holdout** is what the promotion gate looked at when the model
currently serving was promoted: one recent 60-day block. Neither is a substitute for the
other, so both are here.

### Rolling-origin backtest — the reported figure

24-hour horizon, real ENTSO-E data (dataset `65a36efba487`): 26 expanding-window origins,
**8,709 out-of-sample hours covering a full year** (2021-01-08 → 2022-01-07), with a
24-hour embargo between each training window and the block it predicts. Baselines and
PSE are scored on exactly the same hours. Full report, including every segment, in
[`reports/benchmark_h24.md`](reports/benchmark_h24.md).

| Model | MAPE | RMSE (MW) | MAE (MW) | Bias (MW) | vs PSE forecast |
|---|---|---|---|---|---|
| Naive seasonal (`load[T−168]`) | 4.300% | 1339 | 839 | +6 | +1.641 pp worse |
| Linear (calendar + weather) | 6.195% | 1504 | 1209 | −486 | +3.537 pp worse |
| **PSE day-ahead forecast (ENTSO-E)** | 2.658% | 674 | 526 | +391 | *(the benchmark)* |
| LightGBM (calendar + lags) | 2.158% | 648 | 428 | −76 | **−0.501 pp better** |
| LightGBM (+ weather) | 1.983% | 593 | 392 | −100 | **−0.675 pp better** |
| LightGBM (+ weather, tuned) | 1.947% | 584 | 385 | −114 | **−0.711 pp better** |

P10/P50/P90 pinball losses: 110.4 / 192.4 / 118.9 MW.

The model beats PSE's own day-ahead forecast by 0.71 pp of MAPE over the year — 27%
lower error, 90 MW less RMSE. Weather is worth 0.175 pp of that (2.158% → 1.983%) and
the Optuna search a further 0.036 pp, which is worth knowing before anyone spends a day
on hyperparameters. PSE's forecast runs +391 MW high on average; ours runs 114 MW low.

### Where it loses

It loses in exactly the places with the fewest examples, and it loses badly:

| Segment | Hours | Model MAPE | PSE MAPE | Gap |
|---|---|---|---|---|
| Christmas–New Year | 215 | 7.273% | 2.954% | **+4.319 pp** |
| Holidays | 311 | 5.099% | 2.967% | **+2.132 pp** |
| Winter | 2109 | 2.715% | 2.561% | +0.153 pp |
| Weekends | 2470 | 2.053% | 2.658% | −0.605 pp |
| Peak hours | 5445 | 2.025% | 2.719% | −0.693 pp |
| Off-peak hours | 3264 | 1.816% | 2.558% | −0.742 pp |
| Weekdays | 6239 | 1.905% | 2.658% | −0.754 pp |
| Summer | 2208 | 1.577% | 2.853% | −1.277 pp |

A 7.3% MAPE across the Christmas–New Year week against PSE's 2.95% is the single worst
result in this repository, and it is not a rounding error: those 215 hours are a demand
regime the model has seen twice in its training data and a human dispatcher has seen
every year of their career. A holiday flag is not enough — the week behaves like neither
a weekday nor a weekend nor a normal holiday, and PSE's forecasters know that. This is
the first thing to fix, and it is the reason the segment table is reported worst-first
rather than best-first.

**What this year is and is not.** The backtest starts where the data allows: with
`initial_train_days: 730`, the first origin sits two years after 2019-01-01, so the
covered year is 2021 — the earliest available, not the most recent. 2021 also carries
pandemic-recovery demand. Sweeping every origin the data supports (`max_splits: null`,
~146 origins, 2021 → 2026) is the same methodology at roughly ten hours of compute
instead of forty minutes; the section below is what covers the recent period until that
is run.

### Champion holdout — what is serving now

The registered champion (`pl_load_lgbm@champion`, v4) was trained and gated on a single
chronological 60-day block, **2026-06-07 → 2026-08-06, 1,440 hours** of real data:

| Model | MAPE | vs PSE |
|---|---|---|
| Naive seasonal | 5.732% | +2.640 pp worse |
| **PSE day-ahead forecast** | 3.092% | *(the benchmark)* |
| LightGBM (calendar + lags) | 2.701% | −0.391 pp better |
| LightGBM (+ weather) | 2.170% | −0.922 pp better |
| **LightGBM (+ weather, tuned) — champion** | 2.115% | −0.977 pp better |

Read this as gate evidence, not as a published result, for two reasons. It is **one draw
from one summer**: PSE's own MAPE in that window is 3.09% against an annual range of
2.2–3.2% across 2019–2026 (July 2026 alone was 3.57%), so some of the margin is the
window rather than the model. And the block doubles as the **early-stopping set**, so the
iteration count was chosen on the hours being scored — the backtest above avoids that
with a second embargoed split inside each origin, which is precisely why it, and not
this, is the reported figure.

Regenerate either with:

```bash
uv run python -m pipelines.backtest
```

```bash
uv run python -m pipelines.train
```

## Data inventory

| Data | Source | Resolution | Range | Notes |
|---|---|---|---|---|
| Actual total load (PL) | ENTSO-E Transparency Platform | native 15-min or 60-min → resampled to hourly | 2019-01-01 → now (start date in [`config/config.yaml`](config/config.yaml)) | the target |
| Day-ahead load forecast (PL) | ENTSO-E Transparency Platform | hourly | same | the benchmark (PSE) |
| Historical weather | Open-Meteo Archive API | hourly | same, up to ~5 days behind real time | 7 cities, population-weighted |
| Forecast weather | Open-Meteo Forecast API | hourly | the days the archive has not reached, plus the future | used at serving time and for the freshest ingested hours |
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

### What the ingested data can and cannot say about it

**The two paths do disagree.** A full ingest keeps archive values wherever the archive
reaches (it lags real time by about five days) and fills the rest from the forecast
endpoint, so the freshest hours of the stored dataset are forecast-sourced and can be
compared against the archive once it catches up. For the 192 such hours in the current
dataset (2026-07-30 → 2026-08-06):

| Feature | MAE | RMSE | Bias |
|---|---|---|---|
| `temp_c` | 0.90 °C | 1.14 °C | +0.64 °C |
| `wind_ms` | 0.35 m/s | 0.49 m/s | +0.13 m/s |
| `cloud_cover` | 17.4 pp | 23.9 pp | +14.0 pp |

Substituting one for the other moves the champion's forecast by **92 MW on average
(0.54% of load), and by up to 444 MW** in the worst hour — the input difference is not
cosmetic.

**What that costs in accuracy cannot be measured from what is stored, and the numbers
above are not the skew.** Three reasons, all of them disqualifying on their own:

- **The lead time is wrong and unknown.** Those forecast-sourced hours were pulled with
  `past_days`, so for hours already in the past Open-Meteo answers from its most recent
  model run — not from the forecast that was issued 24 hours before the target, which is
  what serving actually uses. The comparison mixes lead-time error with the plain
  difference between a reanalysis (the archive) and a forecast model.
- **The sample is eight days of one summer.** Over the 182 of those hours that have a
  published actual, the champion scores 2.035% MAPE on the stored weather and 2.053% on
  the observed archive. The gap is smaller than the noise and points the wrong way; those
  hours also sit inside the champion's own early-stopping block, so they are not
  out-of-sample either.
- **Nothing records which path a row came from.** The dataset has one `data_source_version`
  per row for the ingestion run, not per column for the weather source, and a later run
  overwrites forecast weather with archive weather in place. The evidence is erased on a
  seven-day delay.

**The envelope, which is measurable.** Weather is worth 0.175 pp of MAPE over the
backtest year (2.158% without it, 1.983% with it) and 0.53 pp on the 60-day holdout.
Forecast weather is worse than observed weather but far better than no weather, so the
skew penalty lives inside that gap — it cannot quietly be half a percentage point over a
year. That is a bound, not a measurement, and it is stated as one.

**TODO — what would make the real measurement possible**, in the order it should be done:

1. Stamp the weather source per row at ingestion (`weather_source` ∈ `archive |
   forecast`, plus the lead time in hours) so the question stops being unanswerable in
   hindsight. This changes the dataset schema, so it needs the sign-off CLAUDE.md asks
   for.
2. Keep every daily forecast pull instead of overwriting it — one small parquet per run,
   indexed by (issued_at, target hour). Nine months of that produces a genuine
   day-ahead-aligned weather series with no new dependency and no new API.
3. Only then re-run the rolling-origin backtest twice, once on each weather series, and
   report both numbers here. Back-filling it sooner means Open-Meteo's Historical
   Forecast API, which archives past forecast runs at fixed lead times — a new external
   source, and therefore a decision rather than a task.

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

Produce the benchmark table from a rolling-origin backtest (~45 minutes for the
configured 26 origins: a five-minute Optuna search, then a full refit of every variant
and of the quantile band at each origin — use `--max-splits 3` for a quick check):

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

`pipelines.ingest` needs an `ENTSOE_API_KEY` and stops without one — there is nothing
honest for it to pull. Training and the backtest do not: with no dataset on disk they
fall back to the synthetic generator in [src/synthetic.py](src/synthetic.py) so the
rest of the loop still runs end to end. Those runs are named `*_synthetic`, tagged
`data_source=synthetic` in MLflow, and their reports carry a warning banner — a metric
from invented data measures the plumbing, nothing else.

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
src/api/         FastAPI service: /forecast, /health, /reload-model
src/monitoring/  Evidently drift, the seasonal reference window, rolling performance
src/             prediction log, pipeline state, holiday calendar, synthetic generator
pipelines/       ingest, train, backtest + the loop: evaluate, check_drift,
                 retrain_if_needed
config/          pipeline parameters (dates, cities, horizon, thresholds, bounds)
tests/           leakage, DST, timezone, validation, retry, gate and loop tests
docs/            plan + architecture decision record
reports/         generated benchmark tables (the .md files are versioned)
monitoring/      generated drift reports (HTML, not versioned)
state/           prediction log and last-success markers (not versioned; ADR-008)
mlruns/          MLflow tracking database and artifacts (not versioned)
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
