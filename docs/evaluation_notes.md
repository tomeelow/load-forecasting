# Evaluation audit — is the margin over PSE a fair comparison?

**Date:** 2026-08-21 · **Subject:** the headline claim that this model beats PSE's own
day-ahead forecast · **Verdict:** **the claim does not survive.** The model does
not beat PSE's day-ahead forecast over a recent year. It matches it on the flat-horizon
comparison that was previously reported, and loses to it by 0.29 pp once the comparison
is made like-for-like. The previously published margin came from measuring the wrong
year, at the wrong horizon, with weather nobody had.

The reported figure was about to go on a CV and into a README, so it was checked before
it went anywhere public. Five questions, in the order that would occur to someone trying
to knock the claim down.

Everything below is reproducible with `uv run python -m pipelines.audit`, which writes
[`reports/audit_h24.md`](../reports/audit_h24.md) and the per-hour predictions behind it.

---

## Summary

**8,736 matched out-of-sample hours, 2025-08-21 → 2026-08-19**, 13 expanding-window
origins, every model and every reference scored on identical timestamps. Full report:
[`reports/audit_h24.md`](../reports/audit_h24.md).

| Variant | Model MAPE | PSE MAPE | Gap | Model RMSE | PSE RMSE |
|---|---|---|---|---|---|
| **A.** flat 24 h horizon, observed weather | 2.364% | 2.351% | **+0.013 pp** | 603 MW | 566 MW |
| **B.** gate-closure horizons, observed weather | 2.576% | 2.351% | **+0.225 pp** | 640 MW | 566 MW |
| **C.** gate-closure horizons, day-ahead forecast weather | 2.642% | 2.351% | **+0.291 pp** | 670 MW | 566 MW |

Naive seasonal (`load[T−168]`) over the same hours: 5.763% MAPE, 1,559 MW RMSE. Both the
model and PSE beat it comfortably; that was never the interesting question.

**C is the number this project is entitled to publish: 2.64% MAPE against PSE's 2.35%,
which is 12% *more* error, not less.**

What the two corrections were worth, measured rather than argued:

| Correction | Cost |
|---|---|
| Scoring each hour at PSE's own lead time instead of a flat 24 h | **0.212 pp** |
| Using the weather that was forecast a day ahead instead of what was observed | **0.066 pp** |
| **Total** | **0.278 pp** |

That total is larger than the entire margin variant A showed. Two effects that each
sounded like a footnote were, together, the whole of the claim.

### Why the previously reported figure was so different

[`reports/benchmark_h24.md`](../reports/benchmark_h24.md) reports 1.947% against PSE's
2.658% — a 0.711 pp win. Three things separate it from the +0.291 pp above, and none of
them is a bug:

1. **It covers 2021**, the oldest year the data supports, because the backtest began
   `initial_train_days` after the series starts rather than at a chosen window. PSE's own
   forecast was materially worse then (2.658% vs 2.351% now), and 2021 carries
   pandemic-recovery demand. Roughly half the difference is that PSE got better.
2. **It is a flat-24 h comparison**, worth 0.212 pp here.
3. **It uses observed weather**, worth a further 0.066 pp.

The 2021 result is not withdrawn — it is a real measurement of a real year. It is simply
not evidence for a claim about now, and it was being used as one.

### A note on what is *not* being corrected for

The audit's variants are trained on LightGBM defaults, with no Optuna search, because
tuning 25 horizons at 13 origins is an order of magnitude more compute than the result
would justify. On the 2021 backtest, tuning was worth **0.036 pp** (1.983% → 1.947%). If
it is worth the same here, a tuned audited model would sit around 2.61% against PSE's
2.35% — still losing. The conclusion does not turn on it, and the direction of the
approximation is conservative, so it is stated rather than removed.

---

## 1. Are both errors computed over the same hours?

**Yes — structurally in the backtest, and in practice everywhere else.**

The concern is real: `point_metrics` drops pairs where either side is missing, so an
hour absent from `tso_forecast_mw` would be scored for the model and skipped for PSE,
flattering whichever side had more hours.

Two findings.

The **backtest** is safe by construction. `run_backtest` ends with
`predictions.dropna()` across every column at once, so any hour without a PSE value is
dropped from *all* models before scoring. The same is true of the gate-closure
evaluation and the audit, which additionally intersects the three variants' indices so
each row of the headline table covers the identical set of hours.

The **data** turns out not to test the guarantee. Over the whole ingested series
(67,104 hours, 2018-12-31 → 2026-08-26) there are:

| | hours |
|---|---|
| load present, PSE forecast missing | **0** |
| load missing, PSE forecast present | 10 |
| both missing | 28 |

The 28 shared gaps are the spring-forward hour, one per year, plus 23 hours at the very
start of the series before ingestion begins. PSE publishes a forecast for every hour it
later reports an actual for, so no hour was ever silently dropped from one side of the
comparison. The gate-closure audit confirms this end to end: all three variants scored
exactly the same number of hours, and PSE was scored on all of them.

**Caveat worth stating:** this is a property of the data as ingested, not a guarantee of
the metric function. If ENTSO-E ever publishes an actual without a forecast,
`point_metrics` would silently compare different hour sets. The backtest is immune; a
one-off `point_metrics` call is not.

---

## 2. Is the horizon the same on both sides?

**No — and this was the real flaw in the reported number.**

PSE's day-ahead load forecast is published **once per day**, by 10:00 local on `D−1` at
the latest (the ENTSO-E deadline is two hours before the 12:00 day-ahead gate closure).
One publication covers all 24 hours of delivery day `D`. Its effective lead time
therefore runs:

| Local hour of the delivery day | PSE's lead time |
|---|---|
| 00:00 | 14 hours |
| 12:00 | 26 hours |
| 23:00 | 37 hours |

The model was a **flat 24 hours ahead of every target hour**. So for every hour after
mid-morning it had load history PSE did not have when it published, and for the small
hours PSE had fresher information than the model did. That is not a comparison of
forecasting skill; it is a comparison of two different products.

### The fix

`src/evaluation/gate_closure.py` reproduces PSE's product. For each target hour it
computes the lead time a forecaster standing at gate closure would have faced —
`H(T) = 14 + local_hour(T)`, derived from the timestamps so the two DST days come out at
23 and 25 hours rather than off by one — and scores that hour with a **direct model
trained for exactly that horizon**. Twenty-five models instead of one. That is the
honest price of the comparison.

The evaluation is otherwise unchanged: same expanding-window origins, same
horizon-length embargo between train and test ([ADR-006](ADR.md)), same references on
the same rows.

**It was worth 0.212 pp** — 2.364% → 2.576% while PSE stayed at 2.351%, which is what
turned a tie into a loss. Making the comparison fair is most of what the audit did.

### How the two forecasts differ across the day

The hour-of-day breakdown is more informative than the aggregate, and it is where the
audit stopped being an accounting exercise and turned up something worth knowing:

| Local hour | Lead (h) | Model | PSE | Gap |
|---|---|---|---|---|
| 00:00 | 14 | 2.051% | 1.848% | +0.204 pp |
| 03:00 | 17 | 2.309% | 2.095% | +0.215 pp |
| 06:00 | 20 | 2.500% | 2.164% | +0.336 pp |
| 09:00 | 23 | 2.887% | 2.960% | **−0.073 pp** |
| 11:00 | 25 | 3.171% | 3.220% | **−0.050 pp** |
| 13:00 | 27 | 3.217% | 3.190% | +0.027 pp |
| 16:00 | 30 | 2.561% | 2.244% | +0.317 pp |
| 19:00 | 33 | 2.518% | 1.916% | +0.602 pp |
| 21:00 | 35 | 2.580% | 1.905% | +0.675 pp |
| 23:00 | 37 | 2.828% | 2.006% | **+0.821 pp** |

(Full 24 rows in [`reports/audit_h24.md`](../reports/audit_h24.md).)

Two things stand out, and the second is the more useful.

**PSE's error is driven by the hour, not by its lead time.** Its worst hours are late
morning through early afternoon (3.2% around 10:00–13:00) and its best are late evening
(1.9% at 20:00–21:00) — despite the evening being the *longest* lead it faces. Whatever
makes midday hard for PSE, it is not how far ahead it is forecasting.

**This model degrades with lead time and PSE does not.** The correlation between the gap
and the lead is **+0.65**: the further ahead both are forecasting, the worse this model
does *relative* to PSE. It is roughly level with PSE at 23–25 hours — the only place it
wins — and loses by 0.82 pp at a 37-hour lead. The model leans hard on recent load lags
(`load_lag_168` and `load_lag_24` dominate its importance), and at a 37-hour horizon the
useful ones are gone. PSE presumably has structural information — planned outages,
industrial schedules, its own load research — that does not decay with lead time.

This is the finding worth taking away from the audit, and it is not one the aggregate
number shows. It also points at the fix: a model built for the day-ahead product should
be trained per-horizon with features chosen for that horizon, rather than trained once at
24 hours and asked to stretch.

---

## 3. Is there any leakage left in the model?

**No.** Re-confirmed against the exact feature set the champion was trained on. This
question mattered more when the number looked too good; it is answered here anyway,
because "the model is not as good as we thought" and "the model is leaking" are different
findings and only one of them is true.

The champion in the registry logged `feature_set = 5832aa336d` over 20 features at a
24-hour horizon. Rebuilding the feature frame with today's `make_features` produces the
same hash, and the champion's **logged MLflow signature is column-for-column identical**
to the builder's output — so the tests are not covering a feature set that has since
drifted away from what is serving.

Those tests ([`tests/test_no_leakage.py`](../tests/test_no_leakage.py)) run at horizons
1, 6, 24 and 48 and attack it three ways:

- **Structural** — every `load_lag_*` offset is read off the generated columns and
  asserted to be at least the horizon; the column set itself is pinned, so a new feature
  cannot appear without someone editing the list and stating its leakage argument.
- **Behavioural** — the future of the load series is replaced with a sentinel no grid
  produces and every feature value on the past side is asserted unchanged. A second
  variant poisons one hour at a time across the whole `[T−H, T]` boundary. This is the
  test that would catch leakage the column names do not reveal, and it also asserts it
  is not vacuous: the sentinel *does* reach the target on the rows it should.
- **Arithmetic** — the rolling window and the target are recomputed by hand from the raw
  frame and compared.

Plus a gap test: a missing input row would turn `shift(24)` into a 23-hour reach, which
is leakage, so the builder reindexes to a complete hourly index and the test proves it.

**On the weather alignment specifically.** Weather features are aligned to the hour being
*predicted*, unshifted — `temp_c[T]`, not `temp_c[T−H]`. That looks like leakage and is
not: Open-Meteo publishes the forecast for hour `T` well before `T` arrives, so the value
genuinely exists at prediction time. There is a dedicated test for the alignment
(`test_weather_is_the_weather_of_the_hour_being_predicted`), and the audit's variant C
removes any remaining doubt by using the value that was actually forecast a day ahead —
see below. Nothing equivalent is true of the load series, which is why the lag rules
above exist at all.

---

## 4. Was the model handed weather PSE did not have?

**Yes, and it is now measured rather than acknowledged.**

Training and the backtest used Open-Meteo's *archive* — observed weather, a reanalysis
of what actually happened. PSE's forecasters were working from a genuine weather forecast
issued the day before. Some of the margin was therefore the model being handed
information nobody had at forecast time.

Open-Meteo's Historical Forecast API archives past forecast runs and exposes them as
`*_previous_dayN` variables, which gives exactly the series needed: for each past hour,
**what the forecast said about it a day earlier**. Audit variant C re-runs the whole
gate-closure backtest on that series.

**It was worth 0.066 pp** — 2.576% → 2.642%, roughly a quarter of the horizon effect and
about a fifth of the total correction. Smaller than the horizon problem, real, and no
longer a hand-wave.

For scale: weather features are worth about 0.175 pp of MAPE in total (measured on the
2021 backtest, 2.158% without them against 1.983% with). So switching from observed to
day-ahead-forecast weather gives back about **38% of everything the weather buys**. The
README used to bound the skew inside that 0.175 pp gap and say it could not be larger;
that bound was right, and the measured value sits in the upper half of it.

### What is measured and what is not

Only **temperature** is swapped. Open-Meteo's forecast archive carries `temperature_2m`
from 2021-03 but wind, cloud and humidity only from 2024-01-19, and swapping all four
would leave under two years of history before the coverage window — so the audited
variant would train on a fraction of what the others do, and the measured gap would be
partly a training-size artefact rather than a weather artefact.

Temperature is the variable that moves load (`temp_c` and `temp_sq` are what the model
leans on; wind and cloud sit far down the importance list), so this captures most of the
effect. It is a lower bound on the skew, not the whole of it, and it is stated as one.

All three variants are truncated to the same span and trained on the same history, so
the differences between the rows of the table are the variable being changed and nothing
else.

---

## 5. Where does it win, and where does it lose?

Worst first. This is where the aggregate stops being the interesting number.

| Segment | Hours | Model | PSE | Gap | |
|---|---|---|---|---|---|
| Christmas–New Year | 216 | 4.216% | 3.036% | +1.180 pp | PSE |
| Winter | 2160 | 3.132% | 2.121% | +1.010 pp | PSE |
| Holidays | 336 | 4.134% | 3.513% | +0.621 pp | PSE |
| Autumn | 2185 | 2.446% | 1.888% | +0.558 pp | PSE |
| Weekdays | 6240 | 2.635% | 2.270% | +0.366 pp | PSE |
| Off-peak hours | 3276 | 2.383% | 2.032% | +0.350 pp | PSE |
| Ordinary days | 8400 | 2.582% | 2.305% | +0.277 pp | PSE |
| Peak hours | 5460 | 2.797% | 2.542% | +0.255 pp | PSE |
| Weekends | 2496 | 2.657% | 2.555% | +0.102 pp | PSE |
| **Spring** | 2207 | 2.336% | 2.473% | **−0.137 pp** | **model** |
| **Summer** | 2184 | 2.662% | 2.918% | **−0.256 pp** | **model** |

**The model wins in spring and summer and loses everywhere else.** The seasonal split is
the whole story: mild weather is where a temperature-driven gradient booster does well,
and winter — where heating demand, daylight and holidays interact — is where PSE's
operational knowledge shows. Christmas–New Year remains the single worst segment in the
repository, as it was in the 2021 backtest.

This also explains the smoke run. A two-origin version of this audit covering
2026-07-23 → 2026-08-19 — pure summer — reported the model *beating* PSE by 0.73 pp. Four
weeks of the model's best season is not a result, and reading one would have confirmed
exactly the wrong conclusion.

---

## What changed as a result

**The headline claim is withdrawn.** The README no longer says the model beats PSE,
because over a recent year it does not. What it says instead is that a LightGBM model on
free public data lands within 0.3 pp of a national TSO's operational forecast, beats it in
two seasons of four, and loses badly on holidays — which is a defensible result and a more
informative one.

**The reported figure is now produced by `pipelines.audit`**, anchored to the most recent
year rather than the oldest, at PSE's own lead time, on forecast weather. The flat-horizon
2021 backtest stays in `reports/` as a labelled historical measurement.

**[ADR-010](ADR.md) records the evaluation change** and has been amended: it originally
anticipated that the margin would shrink, and the margin did not shrink, it inverted.

**The dashboard reads the audited predictions in preference to the flat-horizon ones**, so
the page cannot show the flattering number just because it was computed first.

## Reproducing this

```bash
uv run python -m pipelines.audit --max-splits 13 --test-days 28 --step-days 28
```

Roughly two and a half hours: three backtests, two of which refit 25 direct models at 13
origins each. `--max-splits 2` runs the same path in about twenty minutes and is enough
to check the wiring, but not enough to quote.

The day-ahead weather archive has to be fetched once first — see
`src.ingestion.weather_client.fetch_national_day_ahead`. It is free and needs no key.
