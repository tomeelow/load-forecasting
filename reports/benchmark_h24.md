# Benchmark — 24-hour ahead

Rolling-origin backtest: 26 origins, 8709 out-of-sample hours covering 364 days (2021-01-08 to 2022-01-07). Dataset `65a36efba487`.

| Model | MAPE | RMSE (MW) | MAE (MW) | Bias (MW) | vs PSE forecast |
|---|---|---|---|---|---|
| Naive seasonal (load[T-168]) | 4.300% | 1339 | 839 | +6 | +1.641 pp worse |
| Linear (calendar + weather) | 6.195% | 1504 | 1209 | -486 | +3.537 pp worse |
| **PSE day-ahead forecast (ENTSO-E)** | 2.658% | 674 | 526 | +391 | *(the benchmark)* |
| LightGBM (calendar + lags) | 2.158% | 648 | 428 | -76 | -0.501 pp better |
| LightGBM (+ weather) | 1.983% | 593 | 392 | -100 | -0.675 pp better |
| LightGBM (+ weather, tuned) | 1.947% | 584 | 385 | -114 | -0.711 pp better |

## Probabilistic forecast

| Quantile | Pinball loss (MW) |
|---|---|
| P10 | 110.4 |
| P50 | 192.4 |
| P90 | 118.9 |

## Where LightGBM (+ weather, tuned) wins and loses against PSE

| Segment | Hours | Model MAPE | PSE MAPE | Gap | Verdict |
|---|---|---|---|---|---|
| special: christmas–new year | 215 | 7.273% | 2.954% | +4.319 pp | PSE wins |
| holiday: holiday | 311 | 5.099% | 2.967% | +2.132 pp | PSE wins |
| season: winter | 2109 | 2.715% | 2.561% | +0.153 pp | PSE wins |
| daytype: weekend | 2470 | 2.053% | 2.658% | -0.605 pp | model wins |
| period: peak | 5445 | 2.025% | 2.719% | -0.693 pp | model wins |
| season: autumn | 2185 | 1.761% | 2.503% | -0.742 pp | model wins |
| period: off-peak | 3264 | 1.816% | 2.558% | -0.742 pp | model wins |
| daytype: weekday | 6239 | 1.905% | 2.658% | -0.754 pp | model wins |
| holiday: ordinary day | 8398 | 1.830% | 2.647% | -0.817 pp | model wins |
| special: rest of year | 8494 | 1.812% | 2.651% | -0.839 pp | model wins |
| season: spring | 2207 | 1.768% | 2.711% | -0.942 pp | model wins |
| season: summer | 2208 | 1.577% | 2.853% | -1.277 pp | model wins |

Segments are cut on the Europe/Warsaw clock and calendar. Rows are ordered worst first: everything above the first *model wins* row is a segment where PSE is better.
