# Benchmark — 24-hour ahead

Rolling-origin backtest: 2 origins, 336 out-of-sample hours covering 14 days (2023-01-09 to 2023-01-23). Dataset `115ff8919f41`.

> **These numbers come from SYNTHETIC data.** No ENTSO-E token was available, so the load series, the weather and the 'PSE' forecast are all generated. The table demonstrates that the evaluation runs end to end; it says nothing about Polish demand. Re-run once the token arrives before quoting any figure here.

| Model | MAPE | RMSE (MW) | MAE (MW) | Bias (MW) | vs PSE forecast |
|---|---|---|---|---|---|
| Naive seasonal (load[T-168]) | 3.356% | 949 | 732 | -191 | +2.438 pp worse |
| Linear (calendar + weather) | 4.132% | 1143 | 917 | -457 | +3.214 pp worse |
| **PSE day-ahead forecast (ENTSO-E)** | 0.917% | 239 | 197 | -47 | *(the benchmark)* |
| LightGBM (calendar + lags) | 2.244% | 646 | 488 | -8 | +1.326 pp worse |
| LightGBM (+ weather) | 1.657% | 440 | 362 | -94 | +0.740 pp worse |

## Probabilistic forecast

| Quantile | Pinball loss (MW) |
|---|---|
| P10 | 91.8 |
| P50 | 189.5 |
| P90 | 121.5 |

## Where LightGBM (+ weather) wins and loses against PSE

| Segment | Hours | Model MAPE | PSE MAPE | Gap | Verdict |
|---|---|---|---|---|---|
| daytype: weekday | 240 | 1.712% | 0.893% | +0.818 pp | PSE wins |
| period: peak | 210 | 1.586% | 0.808% | +0.779 pp | PSE wins |
| holiday: ordinary day | 336 | 1.657% | 0.917% | +0.740 pp | PSE wins |
| season: winter | 336 | 1.657% | 0.917% | +0.740 pp | PSE wins |
| special: rest of year | 336 | 1.657% | 0.917% | +0.740 pp | PSE wins |
| period: off-peak | 126 | 1.775% | 1.100% | +0.675 pp | PSE wins |
| daytype: weekend | 96 | 1.520% | 0.977% | +0.544 pp | PSE wins |

Segments are cut on the Europe/Warsaw clock and calendar. Rows are ordered worst first: everything above the first *model wins* row is a segment where PSE is better.
