# CLAUDE.md

Guidance for Claude when working in this repository.

## Project context

Short-term (day-ahead) **electricity load forecasting for the Polish bidding zone**, built as a production-shaped MLOps system rather than a notebook. Data comes from the **ENTSO-E Transparency Platform** (actual total load + PSE's official day-ahead forecast) and **Open-Meteo** (historical and forecast weather), joined on a UTC-hourly index.

The point of this project is the **loop**, not the model: ingestion → versioned dataset → leakage-safe features → tracked training → gated promotion → serving → drift monitoring → automated retraining. LightGBM on engineered features is the easy part. Treat the pipeline plumbing as the deliverable.

Two things distinguish this project and must not be quietly dropped:

1. **The benchmark is PSE's own day-ahead forecast**, retrieved free from ENTSO-E alongside the actuals — not a naive baseline alone. Every evaluation reports against it.
2. **Forecast honesty.** Leakage-safe features and train–serve weather skew are treated as first-class concerns, not footnotes.

## Core principles

- **Concise over clever.** Smallest amount of code that solves the problem clearly.
- **No premature abstraction.** Don't add layers, factories, or config switches until a second concrete use case exists.
- **Best practices, not ceremony.** Follow Python idioms; don't import patterns from other ecosystems just because they're familiar.
- **Readable beats compact.** Concise is not the same as obfuscated.
- **Delete more than you add.** When refactoring, the diff should usually shrink.

## Git workflow

- **Branch per change.** Never commit features or fixes directly to `main`. Use `feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`.
- **One concern per branch.** If scope grows, split.
- **Commit messages: short, imperative, descriptive. No filler.**
  - Good: `feat: add ENTSO-E load and TSO-forecast ingestion`
  - Good: `fix: handle DST fold when resampling Europe/Warsaw to UTC`
  - Good: `chore: pin lightgbm and drop unused xgboost dependency`
  - Bad: `updates`, `WIP`, `various changes`, `fixed stuff`
- **Squash noisy histories before merging.** Keep `main` linear and readable.
- **Never commit secrets, API keys, or `.env`.** Provide `.env.example` instead. The ENTSO-E security token is a secret.
- **Never commit data or model binaries.** Datasets and artifacts go through DVC / MLflow, not into the main tree.

## Code quality

- **Python:** type hints on public functions; `ruff` for lint, `ruff format` for formatting; `pytest` for tests.
- **Dependencies:** pin them. This project uses `uv`; stick with it.
- **Configuration:** environment variables for secrets; YAML/TOML for pipeline parameters (horizons, city weights, thresholds, date ranges). No hardcoded paths, endpoints, or magic numbers.
- **Logging over print.** Use `logging` (or `loguru`) with sensible levels. Log ingestion row counts, gap locations, and model versions — not full dataframes.
- **Tests where it matters.** Required for feature engineering (especially leakage), timezone/DST handling, and evaluation metrics. Not required for exploratory notebooks.
- **Notebooks are scratch space.** If logic matures, move it into the package. Don't import from notebooks.

## Forecasting-specific practices

### The leakage rule (most important rule in this repo)

When forecasting `load[t + H]`, only information available at time `t` may be used. **Every lagged-load feature must be at least `H` hours old relative to the target.** A model that scores brilliantly offline and fails live is almost always leaking.

- Lags shorter than the horizon are a bug, not a tuning choice.
- The 168-hour lag (same hour, one week ago) is always safe for day-ahead horizons.
- Rolling statistics must be computed on a series already shifted by at least `H`.
- Any new feature touching the target series requires a corresponding leakage test before merge.

### Train–serve consistency

- **One feature builder, used by both training and serving.** Never a second copy in the API layer — divergence between them is a silent, expensive bug class.
- **Weather skew is real.** Training uses observed weather; serving uses forecast weather, which is less accurate. Either train on forecast-aligned (reforecast) weather or explicitly quantify the gap in evaluation. Do not silently train on perfect weather and serve on forecasts.
- **Train for the horizon you serve.** Direct per-horizon models, not recursive multi-step. A model trained 1h-ahead and evaluated as 24h-ahead is meaningless.

### Time handling

- **Store and compute in UTC. Always.** Convert to `Europe/Warsaw` only for display.
- **DST is not optional to handle.** Europe/Warsaw loses an hour each spring and repeats one each autumn. Naive resampling across those boundaries produces NaNs or duplicate-index errors. Cover both transitions in tests.
- **ENTSO-E load is often 15-minute resolution** — resample to hourly explicitly and document the aggregation.

### Data

- **ENTSO-E data gets revised.** "Actual" values can be updated after first publication, and some hours publish late. Re-pull a trailing window on each ingestion instead of assuming yesterday is final.
- **Pull the TSO forecast alongside the actuals, always.** `query_load_forecast` is the benchmark; a dataset without it is incomplete.
- **Validate before use.** Schema and range checks (`pandera`) on every ingestion: timestamp gaps, plausible load bounds for the Polish system, missing TSO-forecast hours.
- **Version datasets with DVC.** A model without a traceable dataset hash cannot be reproduced or explained.

### Evaluation

- **Never a random train/test split.** Chronological holdout and rolling-origin (expanding window) backtesting only. Random splits on time series produce fantasy metrics.
- **Always report against both baselines:** naive seasonal (`load[t-168]`) and the PSE day-ahead forecast. A result stated without the PSE comparison is incomplete.
- **Never MAPE alone.** Pair it with RMSE/MAE in MW, plus a per-segment breakdown: peak vs off-peak, weekday vs weekend, holidays.
- **Report failures.** Where the model loses to PSE matters as much as where it wins. Do not tune the reported segments to flatter the model.

### Experiment tracking and promotion

- **Every training run goes to MLflow.** Params, metrics, dataset version, feature-set identifier, and the model with a signature. Untracked runs don't exist.
- **Promotion is gated, never automatic.** A retrained candidate replaces the champion only if it beats the naive baseline and does not materially regress against the current production model. A scheduled retrain that produces a worse model must leave production untouched.
- **Pin versions explicitly** and set seeds for stochastic steps.

### Monitoring

- **Log every served prediction** — timestamp, target time, value, model version, input features. Without this, production performance cannot be measured when actuals arrive.
- **Drift detection must trigger something.** A report nobody reads is decoration. Drift or a rolling-performance breach sets the retrain flag consumed by the scheduled pipeline.

## Ask before acting

When unsure, stop and ask. Specifically:

- Changes to the feature schema, the horizon definition, or anything that alters what "the target" means
- Adding a lag, rolling window, or external feature that could touch the target series
- New external data sources or dependencies
- Anything that changes the promotion gate or auto-promotes a model
- Anything that re-pulls or overwrites the full historical dataset
- Anything that rewrites git history on shared branches
