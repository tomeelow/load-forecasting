# Architecture Decision Record

Decisions that are expensive to reverse, recorded with the reasoning that produced
them. Each entry states the alternatives considered and what would make us change
our mind — an ADR that only lists the winner is a press release, not a record.

---

## ADR-001 — LightGBM is the production model; Prophet and SARIMA are baselines

**Status:** accepted · **Date:** 2026-08-05

### Context

Short-term load is driven by three things that interact non-additively: the calendar
(hour, weekday, holiday), the weather (a U-shaped temperature response — demand rises
in both cold and heat), and recent demand levels. We need a model that can absorb
~25 engineered features, retrain unattended on a schedule, and be honest about a
24-hour horizon.

### Decision

LightGBM is the model that gets promoted to production. Prophet and SARIMA stay in
the repo as interpretable reference baselines, alongside the naive seasonal
(`load[t−168]`) and PSE's own forecast.

### Reasoning

Gradient-boosted trees take arbitrary tabular features without a specified functional
form, which is exactly the shape of this problem: the effect of temperature depends on
the hour, the effect of the hour depends on the day of week, and holidays behave like
Sundays but not quite. Encoding those interactions into a SARIMA specification means
hand-building exogenous regressors and still assuming additivity; LightGBM finds them
from `temp_c`, `temp_sq`, `hour`, `dow` and their splits.

Training cost matters more than it looks. Phase 7 retrains on a schedule inside a
GitHub Actions runner, and Phase 5 runs a rolling-origin backtest that refits the
model dozens of times. LightGBM fits ~40k hourly rows in seconds on CPU; Prophet's MCMC
path and SARIMA's optimisation do not, and a backtest nobody can afford to re-run is a
backtest nobody re-runs.

LightGBM also gives probabilistic forecasts cheaply — `objective="quantile"` with
`alpha=0.1/0.9` produces P10/P90 bands from the same feature matrix, which utilities
care about far more than a bare point estimate.

Prophet and SARIMA earn their place as baselines rather than contenders: if the
tuned LightGBM cannot beat a SARIMA on the same chronological split, that is a signal
the features are wrong, and we want that signal available.

### Consequences

- Extrapolation is bounded: trees cannot predict outside the range of the training
  target. A record-breaking demand peak will be under-forecast. Accepted for a
  day-ahead horizon on a mature system; it would not be acceptable for long-horizon
  capacity planning.
- Feature engineering carries the model's quality, so the feature builder (and its
  leakage tests) is the code most worth reviewing.

### What would change our mind

A neural sequence model (N-HiTS, TFT) beating tuned LightGBM by a margin that
survives the rolling-origin backtest — not just a single lucky holdout — and doing it
within the retraining time budget.

---

## ADR-002 — Direct per-horizon forecasting, not recursive multi-step

**Status:** accepted · **Date:** 2026-08-05

### Context

Producing 24 hourly values for tomorrow can be done two ways: train one 1-step model
and feed its own predictions back as inputs (recursive), or train a model per horizon
whose target is `load[t + H]` for that specific `H` (direct).

### Decision

Direct. `make_features(df, horizon)` derives every lag from the horizon it is given,
and a model trained for horizon `H` is only ever evaluated and served at horizon `H`.

### Reasoning

Recursive forecasting compounds error: step 24 is conditioned on 23 previous
predictions, each carrying its own error, and the error growth is neither measurable
per step nor correctable. Worse, it invites leakage silently. A 1-step model happily
uses `load[t−1]`, which is legitimate at 1 step and impossible at 24 — the "safe"
lag set depends on the horizon, so a single model cannot be correct at all of them.
Making the horizon an explicit parameter forces the lag set to move with it, and the
leakage test in `tests/test_no_leakage.py` is parametrised over horizons precisely so
the guarantee is not accidental at 24.

Direct models are also honestly evaluable: the backtest metric for horizon `H` is
computed on targets `H` hours after the information cutoff, which is what the service
actually does. There is no gap between how the model is scored and how it is used.

The cost is *N* models for *N* horizons. At the day-ahead scope this is cheap
(LightGBM fits in seconds), and each model can be tuned for its own horizon — the
useful features at H=1 and H=24 genuinely differ.

### Consequences

- Model artifacts and registry entries are per-horizon; the registry name carries the
  horizon.
- Serving a full 24-hour profile means loading several models, or one model per
  horizon invoked in a loop.
- Cross-horizon coherence is not guaranteed (nothing forces the 24 predictions to form
  a smooth curve). Acceptable — smoothness is not an operational requirement here.

### What would change our mind

Expanding to many horizons (say 1–72) where *N* models becomes an operational burden;
at that point a single multi-output model with horizon as a feature is worth the
re-evaluation.

---

## ADR-003 — MLflow registry + DVC, not hand-saved artifact files

**Status:** accepted · **Date:** 2026-08-05

### Context

A model in production raises three questions that must be answerable months later:
which data produced it, which features and parameters produced it, and how do we get
the previous one back in a hurry.

### Decision

Every training run is logged to MLflow with params, metrics, dataset version and a
model signature; promotion to `@champion` happens through the registry. Raw and
processed datasets are versioned with DVC, and the DVC hash is logged as a run
parameter so the model points at the exact bytes it was trained on.

### Reasoning

The alternative — `model_final_v3.pkl` next to a spreadsheet of metrics — is precisely
the practice this project exists to demonstrate we've outgrown. It fails in a specific,
predictable way: six weeks later a metric looks wrong, and there is no way to tell
whether the model changed, the features changed, or the upstream data was revised.
This dataset *is* revised — ENTSO-E updates "actual" values after first publication —
so "the data changed under me" is not a hypothetical here, it's the normal case. Only
a content hash distinguishes a genuine model regression from a re-pulled trailing
window.

MLflow's registry gives the operational property that matters at 6am: rollback is
moving an alias, not rebuilding an environment. Combined with the promotion gate
(Phase 4), automated retraining becomes safe to leave unattended — a scheduled retrain
that produces a worse model logs its run and leaves production untouched.

DVC rather than committing parquet: the processed dataset is tens of megabytes and
grows daily; git would bloat irrecoverably, and `.gitignore` already blocks `data/`.

### Consequences

- Local development needs an MLflow tracking server running (containerised in Phase 10),
  which is friction compared to writing a pickle.
- Two systems of record instead of one: MLflow for models, DVC for data. The link
  between them is a logged parameter, and it is only as reliable as the discipline of
  logging it — so the training entrypoint sets it, never the human.

### What would change our mind

Little at this scale. If the project ever needed multi-tenant model governance or
approval workflows, a heavier platform would displace the registry — but not the
principle.

---

## ADR-004 — Evidently for drift detection, not a hand-rolled check

**Status:** accepted · **Date:** 2026-08-05

### Context

The system must notice when the world has moved: a temperature regime change, a
structural shift in demand, a change in the distribution of its own predictions, or a
quiet decay in rolling accuracy against actuals as they arrive.

### Decision

Evidently produces the drift and performance reports; a threshold breach sets a
retrain flag consumed by the scheduled retraining pipeline.

### Reasoning

Hand-rolling this means implementing per-feature statistical tests, picking them
correctly per column type, choosing thresholds, and then building the reporting — a
few hundred lines that are themselves untested statistics. Evidently covers data
drift, prediction drift and regression performance in one pass, and emits a
self-contained HTML report.

The report being *shareable* is a real reason, not a cosmetic one: drift findings have
to be legible to someone who is not reading the code, and a dated HTML artifact is
also a portfolio artifact.

The decision that actually matters is not the library but the wiring: **detection is
connected to action.** Drift or a rolling-MAPE breach sets the flag that
`pipelines/retrain_if_needed` consumes, and that path still runs the promotion gate.
A report nobody reads is decoration, and this project treats that as a design failure
rather than a documentation gap.

### Consequences

- A dependency with its own release cadence and API churn sits in the monitoring path;
  it is pinned, and monitoring failures must not be able to take down serving.
- Evidently's defaults are generic. Load data is strongly seasonal, so a naive
  reference window will report "drift" every spring and autumn. The reference window
  has to be chosen with the seasonality in mind (season-matched, not merely recent),
  otherwise the retrain flag fires on a calendar rather than on a problem.

### What would change our mind

Persistent false positives from seasonality that the reference-window choice cannot
fix, which would push toward a purpose-built residual-based monitor for the
performance signal while keeping Evidently for input drift.

---

## ADR-005 — Feature rows are indexed by target hour, not by prediction moment

**Status:** accepted · **Date:** 2026-08-05 · **Supersedes:** the convention in the
plan's Phase 2 snippet

### Context

A supervised row for a forecasting problem joins two different moments: the hour whose
load is being predicted (`T`) and the moment the prediction is made (`T - H`). Only one
of them can be the index, and the choice silently determines what "a lag of 24 hours"
means.

The first implementation followed the plan's snippet and indexed by prediction moment:
row `t` carried `target = load[t + H]` and lags `load[t - L]` with `L >= H`. Reading
that back against the target revealed a factor-of-two problem — `load[t - H]` is
`load[T - 2H]`, so a 24-hour-ahead model was being fed nothing newer than 48 hours
before the hour it was predicting.

### Decision

Index rows by the target hour `T`. For the row at `T`: the target is `load[T]`, lags
are `load[T - L]` for `L >= H`, rolling statistics run over `load[T - H - W + 1 .. T - H]`,
and weather and calendar features describe `T` itself.

### Reasoning

It makes the leakage rule and the code say the same thing. CLAUDE.md states the rule
as "at least `H` hours old **relative to the target**", and under this convention the
minimum lag is `H` relative to the target — exactly, not approximately. Under the old
convention the code satisfied a stricter rule than the one written down, which is the
kind of mismatch that survives review precisely because it errs safe.

Over-conservatism is not free. Load has strong short-range autocorrelation, so the
hours between `T - 2H` and `T - H` are the most informative ones available; discarding
them buys no safety, because they are genuinely known at the prediction moment.

Three other things fall out rather than needing separate handling:

- **Weather aligns to the hour being predicted.** Demand responds to the weather it is
  in, and the forecast endpoint publishes that value in advance, so it is available.
- **Calendar features describe the target hour** — the hour whose demand profile the
  model is being asked about.
- **Serving reads naturally.** The output index *is* the set of hours being forecast,
  and the benchmark column `tso_forecast_mw[T]` lines up with the model's prediction
  for `T` without any shifting, which matters everywhere PSE is the comparison.

### Consequences

- The horizon is no longer visible in the index; a row at `T` looks the same whether it
  came from a 1-hour or a 48-hour model. The horizon must therefore be carried
  explicitly as run metadata and as part of the registered model's identity.
- Two models at different horizons produce predictions on the *same* index, so anything
  storing predictions must key on (target hour, horizon), not target hour alone.
- The rule "no negative shifts on the load series" is no longer a sufficient smell test
  for leakage, because the target is now an unshifted column. The behavioural leakage
  tests, which poison the input and watch the output, carry that weight instead.

### What would change our mind

Nothing foreseeable. The alternative is defensible but strictly less informative, and
the mismatch it creates between the stated rule and the implemented one is the more
dangerous property.
