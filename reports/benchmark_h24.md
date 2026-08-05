# Benchmark — 24-hour ahead

Rolling-origin backtest: 26 origins, 8736 out-of-sample hours covering 364 days (2023-01-09 to 2024-01-08). Dataset `115ff8919f41`.

> **These numbers come from SYNTHETIC data.** No ENTSO-E token was available, so the load series, the weather and the 'PSE' forecast are all generated. The table demonstrates that the evaluation runs end to end; it says nothing about Polish demand. Re-run once the token arrives before quoting any figure here.

| Model | MAPE | RMSE (MW) | MAE (MW) | Bias (MW) | vs PSE forecast |
|---|---|---|---|---|---|
| Naive seasonal (load[T-168]) | 3.405% | 837 | 636 | -2 | +2.286 pp worse |
| Linear (calendar + weather) | 4.412% | 1012 | 828 | -77 | +3.293 pp worse |
| **PSE day-ahead forecast (ENTSO-E)** | 1.119% | 260 | 208 | +1 | *(the benchmark)* |
| LightGBM (calendar + lags) | 2.402% | 581 | 451 | +1 | +1.283 pp worse |
| LightGBM (+ weather) | 1.868% | 446 | 348 | -39 | +0.748 pp worse |
| LightGBM (+ weather, tuned) | 1.837% | 436 | 343 | -38 | +0.718 pp worse |

## Probabilistic forecast

| Quantile | Pinball loss (MW) |
|---|---|
| P10 | 82.2 |
| P50 | 171.7 |
| P90 | 87.5 |

## Where LightGBM (+ weather, tuned) wins and loses against PSE

| Segment | Hours | Model MAPE | PSE MAPE | Gap | Verdict |
|---|---|---|---|---|---|
| holiday: holiday | 312 | 2.166% | 1.300% | +0.866 pp | PSE wins |
| special: christmas–new year | 216 | 1.908% | 1.112% | +0.796 pp | PSE wins |
| daytype: weekend | 2496 | 2.041% | 1.249% | +0.792 pp | PSE wins |
| period: off-peak | 3276 | 2.005% | 1.219% | +0.787 pp | PSE wins |
| season: summer | 2208 | 1.999% | 1.225% | +0.774 pp | PSE wins |
| season: spring | 2207 | 1.872% | 1.143% | +0.729 pp | PSE wins |
| special: rest of year | 8520 | 1.835% | 1.119% | +0.716 pp | PSE wins |
| holiday: ordinary day | 8424 | 1.825% | 1.112% | +0.712 pp | PSE wins |
| season: winter | 2136 | 1.706% | 0.995% | +0.711 pp | PSE wins |
| daytype: weekday | 6240 | 1.756% | 1.067% | +0.688 pp | PSE wins |
| period: peak | 5460 | 1.736% | 1.059% | +0.677 pp | PSE wins |
| season: autumn | 2185 | 1.767% | 1.110% | +0.657 pp | PSE wins |

Segments are cut on the Europe/Warsaw clock and calendar. Rows are ordered worst first: everything above the first *model wins* row is a segment where PSE is better.
