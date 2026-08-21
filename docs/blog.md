# I did not beat the grid operator, and finding that out was the project

Poland's transmission system operator, PSE, decides today how much electricity the
country will use tomorrow. Generation is scheduled a day ahead and the market clears on
that number, so a few hundred megawatts of error costs real money in balancing.

PSE publishes its forecast to the ENTSO-E Transparency Platform, free, next to the actual
load it later measures. That is why I picked this problem: most forecasting projects
benchmark against a naive baseline, which tells you nothing. Here there is a forecast a
real control room dispatches against.

So I built the loop — daily ingestion, a leakage-safe feature builder shared by training
and serving, LightGBM, MLflow tracking and a registry, drift monitoring, and a gated
nightly retrain on GitHub Actions. The model was the easy part; it took an afternoon.

My first number was a **0.71 pp win** over PSE. I nearly published it.

## What was wrong with it

Before putting that on a CV I checked whether it was a fair comparison. It was not, in
two ways that each sound like a footnote.

**PSE publishes once a day**, by 10:00 the day before, covering all 24 hours of the
delivery day. Its lead time is therefore 14 hours for the midnight hour and 37 for the
23:00 hour — not 24. My model forecast a flat 24 hours ahead of *every* hour, so for most
of the day it held load history PSE did not have. Not leakage — every feature was
legitimately available 24 hours before its target — but not the same problem either.

**And it trained on observed weather.** Open-Meteo's archive is a reanalysis of what
actually happened. PSE's forecasters had a weather forecast.

Both are fixable. Open-Meteo archives its past forecast runs, so I could re-run on what
the forecast said a day earlier. And I could reproduce PSE's product: for each target
hour, compute the lead time a forecaster at gate closure faced, and score that hour with a
direct model trained for it. Twenty-five models instead of one.

Over the most recent year — 8,736 out-of-sample hours, scored on identical timestamps:

| | Model | PSE | Gap |
|---|---|---|---|
| Flat 24 h, observed weather | 2.364% | 2.351% | +0.013 pp |
| At PSE's own lead time | 2.576% | 2.351% | +0.225 pp |
| …and on forecast weather | **2.642%** | **2.351%** | **+0.291 pp** |

The horizon was worth 0.212 pp and the weather 0.066 — together more than the entire
margin the first row shows. And the first row is already a tie, because my original 0.71
pp also came from the *oldest* year in my data, when PSE's own forecast was worse than it
is now.

## The surprising part

Not the aggregate. It is that **my model degrades with lead time and PSE's does not.**

At 23 hours the two are level — the only place mine wins. At 37 hours PSE is ahead by
0.82 pp. The correlation between the gap and the lead is +0.65.

That follows from what the model leans on: `load_lag_168` and `load_lag_24` dominate its
feature importance, and at 37 hours the useful lags are gone. PSE has something that does
not decay with lead — planned outages, industrial schedules, decades of knowing what
Poland does in the first week of January. The seasonal split agrees: I win in spring and
summer and lose by 1.01 pp in winter, where heating, daylight and holidays interact.

Both are actionable. The aggregate was not.

## The lesson

Nothing in the code was broken. Every test passed, and the leakage tests are thorough —
they poison the future of the load series and assert the past does not move. The error was
in *what I compared to what*, which no test suite catches, because it is not a property of
the code.

A LightGBM model on free public data landing within 0.3 pp of a national TSO is a result I
can defend. A 0.71 pp win I could not have defended past the first question.

---

Code, decisions and the full audit: <https://github.com/tomeelow/load-forecasting> ·
[the audit itself](https://github.com/tomeelow/load-forecasting/blob/main/docs/evaluation_notes.md)
