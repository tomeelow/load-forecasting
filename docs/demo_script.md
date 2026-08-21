# Demo script — 90 seconds

Three shots, no narration needed. Record at 1440×900 or wider so the axis labels stay
readable when the video is scaled down in a browser.

Before recording, make sure the dashboard has data: `python -m pipelines.ingest`, then
`python -m pipelines.forecast`, then `streamlit run src/dashboard/app.py`. A dashboard
showing its empty states is honest but it is not a demo.

---

## Shot 1 — the forecast, with its uncertainty (0:00–0:30)

**Screen:** the dashboard's *Forecast vs actual* panel, full width.

**What to do:** land on the panel already scrolled into view. Let it sit for three
seconds so the shape of the week reads, then hover along the forecast line so the
tooltip walks through a few hours.

**What it shows:** black is realised demand, orange dotted is PSE's published day-ahead
forecast, blue is this model with its P10–P90 band. The band widening into the evening
peak is the point — the model is saying where it is less sure, not just what it thinks.

**Caption if you add one:** *Day-ahead hourly forecast for the Polish bidding zone, with
a P10–P90 band. The dotted line is the grid operator's own forecast.*

---

## Shot 2 — the benchmark (0:30–1:00)

**Screen:** scroll to *Model error vs PSE error*. Both panels visible at once.

**What to do:** pause on the rolling-MAPE chart with the two lines separated, then move
right so the "accumulating" served panel is clearly in frame beside it.

**What it shows:** the model's 30-day rolling error tracking below PSE's across a full
year of out-of-sample hours — and, next to it, the served-prediction panel saying plainly
how many hours it has scored and how many it needs. Two panels rather than one is the
whole point: the backtest has a year, production has days, and they are not averaged
together.

**Caption:** *Benchmarked against the operator's own forecast on identical hours and at
the operator's own lead time. Production monitoring is shown separately, and says so
until it has enough data.*

---

## Shot 3 — the gate doing its job (1:00–1:30)

**Screen:** GitHub → Actions → **daily-loop** → *Run workflow* → tick **force_retrain** →
run it. Then open the run and expand *Retrain if needed, promote only through the gate*.

**What to do:** show the dispatch, then cut to the finished log line. If a recent
scheduled run already shows `TRAINED_NOT_PROMOTED`, use that instead — a real rejection
is better footage than a forced one.

**What it shows:** a candidate is trained, scored against the naive baseline and the
current champion, and *not promoted* when it is worse. Production is untouched by an
unattended retrain that produced a weaker model.

**Caption:** *Retraining is automated. Promotion is not — a candidate replaces the
champion only if it beats the naive baseline and does not regress against the model
already serving.*

---

## If you only get one shot

Shot 2. The side-by-side against PSE is the thing a reviewer cannot get from a README
screenshot, and the two-panel split is the thing that makes them trust the rest of it.
