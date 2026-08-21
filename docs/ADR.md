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

## ADR-003 — MLflow registry plus a dataset content hash, not hand-saved artifact files

**Status:** accepted · **Date:** 2026-08-05 · **Amended:** 2026-08-20 (DVC dropped in
favour of a content fingerprint; see *Why the dataset is fingerprinted rather than
DVC-tracked* below) · **Supersedes:** the plan's `dvc.yaml` in
`docs/load_forecasting_mlops_plan.md`

### Context

A model in production raises three questions that must be answerable months later:
which data produced it, which features and parameters produced it, and how do we get
the previous one back in a hurry.

### Decision

Every training run is logged to MLflow with params, metrics, dataset version and a
model signature; promotion to `@champion` happens through the registry.

The dataset is identified by a **content hash computed at training time** —
`dataset_fingerprint()` in `src/ingestion/dataset.py`, a SHA-256 over the hashed rows
and index, truncated to 12 hex characters — and logged as the `dataset_version`
parameter of every run. Each row additionally carries `data_source_version`, the id of
the ingestion run that last wrote it. No DVC.

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

### Why the dataset is fingerprinted rather than DVC-tracked

The original decision named DVC, and the plan carries a `dvc.yaml`. It was not
implemented, and this ADR is amended rather than left standing because a record that
describes a system nobody built is worse than no record.

DVC's value is a *content-addressed store*: the hash in git resolves to the exact bytes
through a remote. Without a remote it is a local cache with extra files, and a remote
means an S3 or GDrive account, credentials in repository secrets, and a bill — rejected
for the same reason as in ADR-008, that the whole loop is supposed to run for free. A
committed `.dvc` file pointing at a remote nobody configured would be ceremony: it looks
like data versioning in a screenshot and answers no question at 6am.

What the fingerprint buys, and it is the property the reasoning above actually needs, is
**detection**: two runs with the same `dataset_version` saw byte-identical data, and two
runs with different ones did not. That is what separates "the model regressed" from "the
trailing window was revised underneath it".

### Consequences

- **The old bytes are not retrievable.** The fingerprint proves the data differed; it
  cannot reproduce it. `data/processed/` is carried between scheduled runs as a single
  snapshot (ADR-008), so yesterday's copy is gone. Re-pulling from ENTSO-E reconstructs
  everything except the revisions themselves — which is exactly the part a fingerprint
  can only detect. This is the real cost of the choice and it is accepted knowingly: on
  this project the dataset is reproducible from a free API in about five minutes, and
  the fingerprint tells us when it stopped matching.
- Local development needs an MLflow store (SQLite here, containerised Postgres in
  Phase 10), which is friction compared to writing a pickle.
- The link between model and data is a logged parameter, and it is only as reliable as
  the discipline of logging it — so the training entrypoint sets it, never the human.

### What would change our mind

Two things, separately. A dataset that stops being cheaply reproducible — a paid feed, a
scraped source, or one whose revisions matter enough to audit — makes a content-addressed
store worth its remote, and DVC (or `lakeFS`, or plain versioned object storage) comes
back. Multi-tenant model governance or approval workflows would displace the registry —
but not the principle.

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

---

## ADR-006 — The backtest embargoes one horizon between training and testing

**Status:** accepted · **Date:** 2026-08-05

### Context

Rolling-origin backtesting is the standard answer to "never use a random split". The
usual implementation trains up to an origin and predicts the block immediately after it.
With a *direct H-step* model, that is still subtly optimistic.

A row is labelled with the load at its target hour, and that load only becomes known at
that hour. To predict target `T` the model is standing at `T - H`. If training ran right
up to the origin, the last training labels belong to hours that had not been published
when the prediction was made — the model was fitted on facts from its own future. The
effect is small in MAPE and completely invisible in the output.

### Decision

Every split leaves a gap of exactly `horizon` hours between the last training target and
the first test target, at both levels: the outer rolling-origin splits and the inner
train/validation split used for early stopping.

### Reasoning

It is the difference between "out-of-sample" and "out-of-sample and reproducible in
production". A number produced without the embargo cannot be delivered on live, because
live has no access to those labels — and the gap between backtest and production is
exactly what this project claims to care about.

The cost is `horizon - 1` rows per split, which at a 24-hour horizon is under a day of
data per origin. That is a rounding error against two years of training data, and the
alternative is a metric that is quietly wrong in the flattering direction.

Applying it to the inner validation split matters just as much: early stopping chooses
the iteration count, so a validation block that touches the training data picks a model
that is over-fitted in a way the outer split cannot detect.

### Consequences

- Reported metrics are slightly worse than a naive implementation would produce. That is
  the point; the comparison against PSE is only meaningful if our side is honest.
- `Split.embargo` is asserted in the tests, so removing the gap breaks the suite rather
  than silently improving the numbers.

---

## ADR-007 — Hyperparameters are tuned once per backtest, not once per origin

**Status:** accepted · **Date:** 2026-08-05

### Context

The backtest refits at every origin. Re-running Optuna at each one would be the
theoretically pure choice: the production system would presumably re-tune periodically
too.

### Decision

Search once, on the first origin's training block, then freeze the parameters for every
subsequent origin.

### Reasoning

Cost, and attribution. Tuning at all 26 origins multiplies the search cost by 26 for a
result nobody could interpret — if the parameters move between origins, a change in error
cannot be attributed to the data, the model, or the search.

It leaks nothing. The first origin's training block precedes every test block in the
backtest, so parameters chosen there are chosen from information available before any
scored hour. This is weaker than re-tuning but it is not optimistic, which is the
property that matters.

The search is also bounded by `model.tuning.timeout_s`, because an evaluation nobody can
afford to re-run stops being an evaluation and becomes a screenshot.

### Consequences

- Parameters fitted to the earliest window are used on the latest one, so a genuine
  regime change late in the backtest is handled with stale settings — which, if anything,
  understates the model.
- When Phase 7 retrains on a schedule, it should re-tune periodically rather than inherit
  this choice; that is a production cadence question, not an evaluation one.

---

## ADR-008 — Pipeline state lives on a force-pushed orphan branch

**Status:** accepted · **Date:** 2026-08-06

### Context

GitHub Actions runners are ephemeral: the filesystem is destroyed when the job ends.
The daily loop nevertheless has to remember things between runs, and they are not all
equally replaceable:

| State | Size | Replaceable? |
|---|---|---|
| Prediction log (`state/predictions.db`) | ~30 KB, +50 KB/year | **No.** What the service told someone yesterday cannot be recovered. |
| Last-success markers and retrain flag | < 1 KB | **No.** Losing them means a missed run is never backfilled and a raised flag never acted on. |
| MLflow store (`mlruns/`) | ~50 MB | Partly. Run history is lost; the champion could be retrained but not reproduced. |
| Dataset (`data/processed/`) | ~3 MB | **Yes**, from ENTSO-E in about five minutes. |

The prediction log is the one that decides this. Everything Phase 8 measures is a join
against it, so a mechanism that loses it occasionally produces monitoring that is
quietly meaningless.

### Decision

At the start of each run, restore `state/`, `mlruns/` and `data/processed/` from an
orphan branch named `pipeline-state`. At the end, commit them as a **single snapshot**
and `git push --force` back to that branch.

Runs older than `retraining.keep_runs_days` that no alias points at are deleted and
their artifacts reclaimed with `mlflow gc`, so the snapshot does not grow without bound.

### Reasoning

The alternatives were each rejected for a specific reason rather than on taste:

- **`actions/cache`** is evicted after seven days of no access and has no durability
  guarantee at all. Perfectly good for the dataset, which is reproducible; unacceptable
  for the prediction log, which is not.
- **Workflow artifacts** expire (90 days by default) and cannot be updated in place —
  each run would create a new one, and reading "the latest" means an API call to find it.
- **An external object store** (S3, GCS) is the right answer at real scale and the wrong
  answer here: it needs an account, credentials in secrets, and a bill, for a project
  whose entire premise is that the loop runs for free.
- **Committing to `main`** would bury the source history under a daily binary commit and
  make every `git log` useless.

Force-pushing a snapshot rather than appending commits is the part that makes it work.
An append-only state branch would store a fresh copy of a multi-megabyte parquet and a
50 MB artifact tree every single day — a gigabyte a year of history nobody will ever
read. A snapshot keeps exactly one version, which is all that state means.

The trade-off accepted: **the history of the state is not recoverable.** That is the
correct thing to give up. MLflow already records run history inside its own database,
and the prediction log accumulates rather than being rewritten, so what is lost is only
the ability to see yesterday's copy of today's file.

Force-pushing is normally something this repository does not do (see the git workflow in
CLAUDE.md). It is safe here for reasons that do not generalise: `pipeline-state` is an
orphan branch created by the workflow, it shares no history with `main`, nothing is ever
developed on it, and `concurrency` prevents two runs from racing on it.

### Consequences

- The workflow needs `permissions: contents: write`.
- A failed push loses that run's state. Mitigated by also uploading `state/` as a
  workflow artifact with 14-day retention on every run, including failed ones.
- Restoring is a `git checkout FETCH_HEAD -- <paths>`, which means a first run — or a run
  after someone deletes the branch — starts empty and backfills. That path is exercised
  rather than assumed: the ingest window widens to cover whatever gap the markers show.
- Someone reading the repository sees a branch full of binary state. It is named for what
  it is.

### What would change our mind

Serving moving off a laptop and onto real infrastructure. At that point the prediction
log wants to be Postgres, the artifact store wants to be object storage, and this
mechanism should be deleted rather than extended.

---

## ADR-009 — Drift is measured against the same weeks in previous years

**Status:** accepted, with a correction · **Date:** 2026-08-06 · **Amended:** 2026-08-20
(the evidence quoted below was produced by a parsing bug; see *Correction* )

### Context

Drift detection compares a current window against a reference window. On electricity
load, the choice of reference decides whether the result means anything.

The conventional choice is a trailing reference: the last fortnight against the fortnight
before it. On this target that reports drift every March and every October, because the
temperature genuinely changed and demand genuinely moved with it. The model has not
decayed; the seasons turned. An alert that fires twice a year on schedule gets muted, and
a muted alert is not a control.

### Decision

The reference window is **the same calendar weeks in each of the previous three years**,
padded by a week either side. September is judged against Septembers.

Two supporting decisions fall out of it:

- **Calendar features are excluded from drift testing entirely.** `hour`, `dow`, `month`,
  `is_weekend` and the cyclical encodings are deterministic functions of the timestamp.
  `hour` cannot drift — testing it asks whether two windows happen to contain a whole
  number of days, and a fortnight ending mid-afternoon fails that every time. `month` is
  worse: the seasonal reference pads each historical window by a week, so the month mix
  differs *by construction* and drift is reported with certainty. Including them adds
  nine columns that cannot carry information about the model and can only dilute the
  drifted share, which is what the trigger reads. (An earlier version of this bullet
  quoted a count of features that "had actually moved"; that number came from the
  inverted comparison described in the correction below and is withdrawn.)
- **Performance is the decisive signal, not input drift.** Rolling error of served
  predictions against actuals has no seasonal confound at all. Input drift is an early
  warning that may or may not matter; a model getting worse is proof that it does.

### Reasoning

Removing the seasonal signal by construction is better than tuning a threshold until the
false alarms stop, because a threshold high enough to ignore an October temperature swing
is also high enough to ignore a real regime change in October.

Measured on the ingested Polish data, the difference is real but smaller than this ADR
first claimed. Against a seasonal reference the fortnight ending 2026-08-06 flags 7 of 11
inputs; against a trailing reference over the same hours, 9 of 11. The difference is
entirely in the load-level features — three drift seasonally against five trailing — which
is the level shift the seasonal reference is meant to absorb. The weather features drift
under **both**. See the correction below.

### Consequences

- It needs years of history. In the first year of operation there is none, so
  `fallback_to_trailing` applies and the check says which reference it used. During that
  year seasonal false positives are expected and the performance signal is the one to
  trust — this is stated in the report rather than left for someone to discover.
- Three years of padded windows is a much larger reference sample than the current
  window, which makes the K-S test more sensitive. That is the right direction for an
  early-warning signal, and the share threshold rather than any single feature is what
  triggers action.
- A genuine multi-year trend — electrification, efficiency, a structural demand shift —
  registers as drift, correctly, because this year stops looking like previous years.

### Correction, 2026-08-20

The sentence this ADR originally rested on — *"against a seasonal reference, no weather
feature drifts"* — was false, and the way it became false is worth recording.

Evidently answers a drift question with one of two kinds of number, and it picks which by
sample size rather than being told: a **p-value** on small samples (drift when the value
is *below* the threshold) and a **distance** on large ones (drift when it is *above*).
`_drift_from_snapshot` read every value as a p-value, so on samples large enough to get a
Wasserstein distance — which is every real comparison this project makes — it reported the
*complement* of the drifted columns. The four "load-level features that drifted" were the
four that had not.

What is true, measured across four dates on the ingested series and on the synthetic
fixture, now that the comparison is the right way round:

- The seasonal reference always flags **no more** features than the trailing one, and the
  saving is in the load-level features. That part of the decision holds.
- **The weather drifts under either reference, at every date tried.** The seasonal window
  does not remove it.
- The share is 0.45–1.00 depending on the date, against a `drift_share_threshold` of 0.3.
  Input drift therefore fires more or less continuously, so in practice the loop attempts
  a retrain most nights and the promotion gate is what protects production.

The cause is not seasonality but the shape of the comparison: fourteen days of one
realised fortnight against a three-year mixture is a narrow distribution against a wide
one, and a normed Wasserstein distance above 0.1 is close to guaranteed. A seasonal
reference cannot fix that, because it makes the reference *wider*, not narrower.

This does not change the decision — judging September by Septembers is still right, and
ADR-004 already makes rolling performance the decisive signal precisely because input
drift is an early warning rather than proof. It does leave an open question, recorded here
rather than quietly patched: the input-drift trigger needs either a like-sized reference
sample, a threshold chosen against measured false-alarm rates, or demotion to
report-only. Picking among those is an operational judgement with evidence still to
gather, not a cleanup.

The behaviour of the trigger is unchanged by the fix: `should_retrain` reads Evidently's
own drifted-share, which was always right. What changed is that the feature names in the
report are now the ones that actually moved.

### What would change our mind

Evidence that the load-level drift this reports is dominated by year-on-year trend rather
than by anything a retrain fixes. The answer then is not a different reference window but
detrending the comparison, which is a larger change than it sounds.

---

## ADR-010 — The benchmark is measured at PSE's lead time, not at a flat 24 hours

**Status:** accepted · **Date:** 2026-08-21 · **Amends the reported figure in**
`README.md` and `reports/benchmark_h24.md`

### Context

Every result in this project is reported against PSE's published day-ahead forecast. That
comparison was being made against a model forecasting a flat 24 hours ahead of every
target hour — and PSE does not work that way.

PSE's day-ahead load forecast is published **once per day**, by 10:00 local on `D-1` at
the latest: the ENTSO-E deadline is two hours before the 12:00 day-ahead market gate
closure, and one publication covers all 24 hours of delivery day `D`. Its effective lead
time therefore runs from about 14 hours for the 00:00 hour to about 37 hours for the
23:00 hour, and averages roughly 25.5 — near 24 only by coincidence.

A flat-24h model is therefore **not uniformly easier or harder**. For every hour after
mid-morning it holds load history PSE did not have when it published; for the small hours
PSE holds fresher information than it does. Both sides of that asymmetry were present in
the reported number, and the net direction favoured us.

### Decision

The reported benchmark is produced by a **gate-closure-aligned evaluation**
(`src/evaluation/gate_closure.py`): for each target hour, the lead time a forecaster
standing at PSE's publication deadline would have faced is computed from the timestamps,
and that hour is scored by a direct model trained for exactly that horizon. Twenty-five
models — `H` from 14 to 38 — instead of one.

The flat-24h figure is not deleted. It is kept as row A of the audit table, because the
difference between the two is itself the finding.

### Reasoning

It is the only version of the comparison that is a comparison. Anything else measures a
product PSE does not sell.

The alternatives were both worse. **Arguing the asymmetry away** was tempting because the
evidence half-supports it — the correlation between the per-hour gap and PSE's lead time
is only −0.10, and the model's advantage is concentrated in the 05:00–08:00 morning ramp
where PSE is simply weak rather than where its lead is longest. But "the unfairness
probably did not matter" is not a claim that survives an interview, and it turned out to
be worth a measurable amount. **A uniform conservative horizon** — one model at H=38,
legal at gate closure for every hour of the delivery day — is cheap and bulletproof, and
it understates the model badly on the early hours where PSE has only a 14-hour lead. It
answers "can the model still win when handicapped", which is a different and lesser
question.

The cost is real: 25 fits per origin rather than one, which is why this is a separate
evaluation run rather than a change to the served model. **ADR-002 is unaffected** —
direct per-horizon forecasting is exactly what makes this possible; the evaluation simply
instantiates one model per horizon the product actually needs.

Publication is assumed at the 10:00 deadline rather than at PSE's habit, which is usually
earlier. That gives PSE the benefit of the doubt and this evaluation the harder side.

### Consequences

- **The published margin over PSE is smaller than it was.** That is the point.
- The audit re-runs the whole thing a third time with weather features taken from
  Open-Meteo's archive of past *forecasts* rather than from observed weather, so the
  horizon effect and the train–serve weather effect are separated rather than pooled.
- Coverage is anchored to the end of the data (`first_test_start`) rather than starting
  `initial_train_days` after it begins. On this series the old behaviour reported 2021 —
  the oldest year available, and one carrying pandemic-recovery demand.
- The audit is expensive enough that it is not part of CI or the nightly loop. It is run
  deliberately, and the number it produces is the one that gets quoted.
- The served model is still a flat 24-hour direct model. What it is *scored against* has
  changed; what it does has not.

### What would change our mind

PSE publishing intraday updates to the day-ahead figure, or ENTSO-E exposing per-row
publication timestamps. Either would replace the assumed 10:00 deadline with a measured
lead time per hour, which is strictly better than the assumption made here.
