# Evaluation audit — is the margin over PSE a fair comparison?

8736 matched out-of-sample hours, 2025-08-21 to 2026-08-19, 8736 rows scored identically for every row of every table below. Dataset `855169c719e0`.

Each variant removes one advantage the published figure enjoyed. Every model and every reference is scored on exactly the same timestamps.

| Variant | Model MAPE | PSE MAPE | Gap | Model RMSE (MW) | PSE RMSE (MW) |
|---|---|---|---|---|---|
| A. flat 24h horizon, observed weather | 2.364% | 2.351% | +0.013 pp | 603 | 566 |
| B. gate-closure horizons, observed weather | 2.576% | 2.351% | +0.225 pp | 640 | 566 |
| C. gate-closure horizons, day-ahead forecast weather | 2.642% | 2.351% | +0.291 pp | 670 | 566 |

Naive seasonal (`load[T-168]`) over the same hours: 5.763% MAPE.

## Error by hour of day

PSE publishes once a day, so its lead time grows through the delivery day while a flat-horizon model's does not. This is where that shows.

| Local hour | PSE lead (h) | Model MAPE | PSE MAPE | Gap |
|---|---|---|---|---|
| 00:00 | 14 | 2.051% | 1.848% | +0.204 pp |
| 01:00 | 15 | 2.143% | 1.976% | +0.167 pp |
| 02:00 | 16 | 2.196% | 2.058% | +0.138 pp |
| 03:00 | 17 | 2.309% | 2.095% | +0.215 pp |
| 04:00 | 18 | 2.274% | 2.077% | +0.196 pp |
| 05:00 | 19 | 2.434% | 2.106% | +0.328 pp |
| 06:00 | 20 | 2.500% | 2.164% | +0.336 pp |
| 07:00 | 21 | 2.612% | 2.277% | +0.336 pp |
| 08:00 | 22 | 2.730% | 2.593% | +0.137 pp |
| 09:00 | 23 | 2.887% | 2.960% | -0.073 pp |
| 10:00 | 24 | 3.200% | 3.194% | +0.005 pp |
| 11:00 | 25 | 3.171% | 3.220% | -0.050 pp |
| 12:00 | 26 | 3.393% | 3.275% | +0.118 pp |
| 13:00 | 27 | 3.217% | 3.190% | +0.027 pp |
| 14:00 | 28 | 2.956% | 2.842% | +0.113 pp |
| 15:00 | 29 | 2.759% | 2.559% | +0.200 pp |
| 16:00 | 30 | 2.561% | 2.244% | +0.317 pp |
| 17:00 | 31 | 2.429% | 2.049% | +0.380 pp |
| 18:00 | 32 | 2.513% | 2.025% | +0.488 pp |
| 19:00 | 33 | 2.518% | 1.916% | +0.602 pp |
| 20:00 | 34 | 2.427% | 1.884% | +0.543 pp |
| 21:00 | 35 | 2.580% | 1.905% | +0.675 pp |
| 22:00 | 36 | 2.711% | 1.962% | +0.749 pp |
| 23:00 | 37 | 2.828% | 2.006% | +0.821 pp |

## Segment breakdown — the audited variant

| Segment | Hours | Model MAPE | PSE MAPE | Gap | Verdict |
|---|---|---|---|---|---|
| special: christmas–new year | 216 | 4.216% | 3.036% | +1.180 pp | PSE wins |
| season: winter | 2160 | 3.132% | 2.121% | +1.010 pp | PSE wins |
| holiday: holiday | 336 | 4.134% | 3.513% | +0.621 pp | PSE wins |
| season: autumn | 2185 | 2.446% | 1.888% | +0.558 pp | PSE wins |
| daytype: weekday | 6240 | 2.635% | 2.270% | +0.366 pp | PSE wins |
| period: off-peak | 3276 | 2.383% | 2.032% | +0.350 pp | PSE wins |
| holiday: ordinary day | 8400 | 2.582% | 2.305% | +0.277 pp | PSE wins |
| special: rest of year | 8520 | 2.602% | 2.334% | +0.268 pp | PSE wins |
| period: peak | 5460 | 2.797% | 2.542% | +0.255 pp | PSE wins |
| daytype: weekend | 2496 | 2.657% | 2.555% | +0.102 pp | PSE wins |
| season: spring | 2207 | 2.336% | 2.473% | -0.137 pp | model wins |
| season: summer | 2184 | 2.662% | 2.918% | -0.256 pp | model wins |

Segments are cut on the Europe/Warsaw clock and calendar, worst first.
