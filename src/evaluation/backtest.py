"""Rolling-origin backtest: the numbers this project is allowed to quote.

A single holdout is one draw from one season. This walks the origin forward across a
full year, refitting at each one and accumulating out-of-sample predictions, so the
reported error covers winter peaks, summer troughs, holidays and both DST transitions.

Every fit sees only data from before its own test block, with a `horizon`-hour embargo
(see `splits.py`). The baselines and PSE's forecast are scored on exactly the same rows,
because a comparison on different hours is not a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.evaluation.metrics import (
    DEFAULT_PEAK_HOURS,
    metrics_by_segment,
    pinball_loss,
    point_metrics,
)
from src.evaluation.splits import Split, chronological_split, rolling_origin_splits
from src.features.builder import TARGET_COLUMN, feature_columns, make_features
from src.models import lgbm
from src.models.baselines import LINEAR, NAIVE_SEASONAL, TSO_FORECAST, LinearBaseline
from src.models.baselines import naive_seasonal as naive_baseline
from src.models.baselines import tso_forecast as tso_baseline

ACTUAL = "actual"
CALENDAR_ONLY = "lgbm_calendar"
WITH_WEATHER = "lgbm_weather"
TUNED = "lgbm_weather_tuned"

# The order the benchmark table is presented in: references first, then the model.
TABLE_ORDER = [NAIVE_SEASONAL, LINEAR, TSO_FORECAST, CALENDAR_ONLY, WITH_WEATHER, TUNED]

PRETTY_NAMES = {
    NAIVE_SEASONAL: "Naive seasonal (load[T-168])",
    LINEAR: "Linear (calendar + weather)",
    TSO_FORECAST: "**PSE day-ahead forecast (ENTSO-E)**",
    CALENDAR_ONLY: "LightGBM (calendar + lags)",
    WITH_WEATHER: "LightGBM (+ weather)",
    TUNED: "LightGBM (+ weather, tuned)",
}


@dataclass(frozen=True)
class BacktestResult:
    """Accumulated out-of-sample predictions and everything computed from them."""

    horizon: int
    predictions: pd.DataFrame  # index = target hour; `actual` plus one column per model
    splits: list[Split]
    quantiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuned_params: dict[str, object] = field(default_factory=dict)

    @property
    def actual(self) -> pd.Series:
        return self.predictions[ACTUAL]

    @property
    def model_columns(self) -> list[str]:
        return [c for c in self.predictions.columns if c != ACTUAL]

    @property
    def coverage_days(self) -> int:
        return (self.predictions.index.max() - self.predictions.index.min()).days + 1

    def overall(self) -> pd.DataFrame:
        """One row per model: MAPE, RMSE, MAE, bias, and the gap to PSE."""
        rows = {c: point_metrics(self.actual, self.predictions[c]) for c in self.model_columns}
        table = pd.DataFrame(rows).T
        if TSO_FORECAST in table.index:
            table["mape_vs_tso"] = table["mape"] - table.loc[TSO_FORECAST, "mape"]
        ordered = [c for c in TABLE_ORDER if c in table.index]
        return table.loc[ordered + [c for c in table.index if c not in ordered]]

    def by_segment(self, peak_hours: tuple[int, int] = DEFAULT_PEAK_HOURS) -> pd.DataFrame:
        """The per-segment breakdown, for every model including the benchmark."""
        return metrics_by_segment(
            self.actual, self.predictions[self.model_columns], peak_hours=peak_hours
        )

    @property
    def primary_model(self) -> str:
        """The strongest variant this backtest actually produced.

        Not always the tuned one: `--no-tune` is a legitimate way to run, and the report
        should describe the model that exists rather than the one that was planned.
        """
        for name in (TUNED, WITH_WEATHER, CALENDAR_ONLY):
            if name in self.predictions.columns:
                return name
        raise ValueError("Backtest produced no LightGBM variant to report on")

    def pinball(self) -> dict[str, float]:
        """Pinball loss per quantile, empty if the band was not backtested."""
        return {
            f"pinball_p{int(float(q) * 100)}": pinball_loss(
                self.actual, self.quantiles[q], float(q)
            )
            for q in self.quantiles.columns
        }

    def versus_tso(
        self, model: str | None = None, peak_hours: tuple[int, int] = DEFAULT_PEAK_HOURS
    ) -> pd.DataFrame:
        """Where the model beats PSE and where it loses, worst first.

        Sorted so the losses are at the top. Reporting only the wins would make the
        whole evaluation decorative.
        """
        model = model or self.primary_model
        segments = self.by_segment(peak_hours)
        wide = segments.pivot_table(
            index=["segment_kind", "segment", "n"], columns="model", values="mape"
        ).reset_index()
        if model not in wide or TSO_FORECAST not in wide:
            raise ValueError(f"Backtest has no '{model}' or '{TSO_FORECAST}' predictions")
        wide["gap_vs_tso"] = wide[model] - wide[TSO_FORECAST]
        wide["verdict"] = wide["gap_vs_tso"].map(lambda g: "model wins" if g < 0 else "PSE wins")
        return wide.sort_values("gap_vs_tso", ascending=False).reset_index(drop=True)


def _fit_variants(
    X: pd.DataFrame,
    y: pd.Series,
    inner_train: pd.DatetimeIndex,
    inner_val: pd.DatetimeIndex,
    *,
    tuned_params: dict[str, object] | None,
    fit_kwargs: dict[str, object],
) -> dict[str, lgbm.TrainedModel]:
    """Fit each variant on one origin's training block.

    The inner train/validation split is passed in rather than derived here, so the point
    models and the quantile band are guaranteed to have seen exactly the same rows.
    """
    all_columns = list(X.columns)
    specs = {
        CALENDAR_ONLY: (feature_columns(all_columns, include_weather=False), None),
        WITH_WEATHER: (all_columns, None),
    }
    if tuned_params is not None:
        specs[TUNED] = (all_columns, tuned_params)

    return {
        name: lgbm.train(
            X.loc[inner_train, columns],
            y.loc[inner_train],
            X.loc[inner_val, columns],
            y.loc[inner_val],
            params=params,
            **fit_kwargs,
        )
        for name, (columns, params) in specs.items()
    }


def run_backtest(
    frame: pd.DataFrame,
    *,
    horizon: int,
    initial_train_days: int,
    test_days: int,
    step_days: int,
    max_splits: int | None = None,
    inner_validation_days: int = 45,
    tune: bool = True,
    n_trials: int = 20,
    tuning_timeout_s: int | None = None,
    quantiles: tuple[float, ...] = (),
    seed: int = 42,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    rolling_window: int = 24,
    weekly_lag: int = 168,
) -> BacktestResult:
    """Walk the origin forward, refitting at each one, and accumulate the predictions.

    Hyperparameters are searched **once**, on the first origin's training block, and then
    frozen. Re-tuning at every origin would cost an order of magnitude more compute for a
    result nobody could attribute; tuning once on the earliest data leaks nothing, because
    every test block still lies in that origin's future.
    """
    features = make_features(frame, horizon, rolling_window=rolling_window, weekly_lag=weekly_lag)
    columns = feature_columns(list(features.columns))
    X, y = features[columns], features[TARGET_COLUMN]

    splits = rolling_origin_splits(
        features.index,
        horizon=horizon,
        initial_train_days=initial_train_days,
        test_days=test_days,
        step_days=step_days,
        max_splits=max_splits,
    )

    fit_kwargs = {
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "seed": seed,
    }

    tuned_params: dict[str, object] | None = None
    if tune:
        first = splits[0].train_index(features.index)
        inner_train, inner_val = chronological_split(
            first, validation_days=inner_validation_days, horizon=horizon
        )
        logger.info("Tuning once on the first origin ({} rows)", len(first))
        tuned_params = lgbm.tune(
            X.loc[inner_train],
            y.loc[inner_train],
            X.loc[inner_val],
            y.loc[inner_val],
            n_trials=n_trials,
            seed=seed,
            timeout_s=tuning_timeout_s,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

    point_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []

    for number, split in enumerate(splits, start=1):
        train_index = split.train_index(features.index)
        test_index = split.test_index(features.index)
        if len(test_index) == 0:
            continue

        # Early stopping needs its own validation block, taken from the tail of the
        # training data with the same embargo — never from the test block. Computed once
        # per origin so every model fitted here has seen exactly the same rows.
        inner_train, inner_val = chronological_split(
            train_index, validation_days=inner_validation_days, horizon=horizon
        )
        models = _fit_variants(
            X, y, inner_train, inner_val, tuned_params=tuned_params, fit_kwargs=fit_kwargs
        )
        block = pd.DataFrame({ACTUAL: y.loc[test_index]})
        for name, model in models.items():
            block[name] = model.predict(X.loc[test_index, model.columns])
        point_frames.append(block)

        if quantiles:
            band = lgbm.train_quantiles(
                X.loc[inner_train],
                y.loc[inner_train],
                X.loc[inner_val],
                y.loc[inner_val],
                quantiles,
                params=tuned_params,
                **fit_kwargs,
            )
            quantile_frames.append(
                pd.DataFrame(
                    {q: m.predict(X.loc[test_index]) for q, m in band.items()}, index=test_index
                )
            )

        logger.info("Origin {}/{}: {} | {} test rows", number, len(splits), split, len(test_index))

    predictions = pd.concat(point_frames).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="last")]

    # The references are scored on exactly the rows the models were scored on.
    predictions[NAIVE_SEASONAL] = naive_baseline(frame, weekly_lag).reindex(predictions.index)
    predictions[TSO_FORECAST] = tso_baseline(frame).reindex(predictions.index)
    predictions[LINEAR] = _linear_baseline_predictions(X, y, splits, features.index)

    band_frame = pd.DataFrame()
    if quantile_frames:
        band_frame = pd.concat(quantile_frames).sort_index()
        band_frame = band_frame[~band_frame.index.duplicated(keep="last")]

    logger.info(
        "Backtest h={}: {} origins, {} out-of-sample hours from {} to {}",
        horizon,
        len(splits),
        len(predictions),
        predictions.index.min(),
        predictions.index.max(),
    )
    return BacktestResult(horizon, predictions.dropna(), splits, band_frame, tuned_params or {})


def _linear_baseline_predictions(
    X: pd.DataFrame, y: pd.Series, splits: list[Split], index: pd.DatetimeIndex
) -> pd.Series:
    """The linear reference, refitted at each origin like every other model."""
    pieces = []
    for split in splits:
        train_index = split.train_index(index)
        test_index = split.test_index(index)
        if len(test_index) == 0:
            continue
        model = LinearBaseline().fit(X.loc[train_index], y.loc[train_index])
        pieces.append(model.predict(X.loc[test_index]))
    combined = pd.concat(pieces).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def benchmark_markdown(result: BacktestResult, *, synthetic: bool, dataset_version: str) -> str:
    """The headline artifact: the benchmark table, ready to paste into the README."""
    table = result.overall()
    lines = [
        f"# Benchmark — {result.horizon}-hour ahead",
        "",
        f"Rolling-origin backtest: {len(result.splits)} origins, {len(result.predictions)} "
        f"out-of-sample hours covering {result.coverage_days} days ("
        f"{result.predictions.index.min():%Y-%m-%d} to "
        f"{result.predictions.index.max():%Y-%m-%d}). Dataset `{dataset_version}`.",
        "",
    ]
    if synthetic:
        lines += [
            "> **These numbers come from SYNTHETIC data.** No ENTSO-E token was available, so "
            "the load series, the weather and the 'PSE' forecast are all generated. The table "
            "demonstrates that the evaluation runs end to end; it says nothing about Polish "
            "demand. Re-run once the token arrives before quoting any figure here.",
            "",
        ]

    lines += [
        "| Model | MAPE | RMSE (MW) | MAE (MW) | Bias (MW) | vs PSE forecast |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in table.iterrows():
        gap = row.get("mape_vs_tso")
        if name == TSO_FORECAST:
            verdict = "*(the benchmark)*"
        elif pd.isna(gap):
            verdict = "—"
        else:
            verdict = f"{gap:+.3f} pp {'better' if gap < 0 else 'worse'}"
        lines.append(
            f"| {PRETTY_NAMES.get(name, name)} | {row['mape']:.3f}% | {row['rmse_mw']:.0f} | "
            f"{row['mae_mw']:.0f} | {row['bias_mw']:+.0f} | {verdict} |"
        )

    pinball = result.pinball()
    if pinball:
        lines += [
            "",
            "## Probabilistic forecast",
            "",
            "| Quantile | Pinball loss (MW) |",
            "|---|---|",
            *(f"| {k.removeprefix('pinball_').upper()} | {v:.1f} |" for k, v in pinball.items()),
        ]

    model = result.primary_model
    lines += ["", f"## Where {PRETTY_NAMES.get(model, model)} wins and loses against PSE", ""]
    comparison = result.versus_tso(model)
    lines += [
        "| Segment | Hours | Model MAPE | PSE MAPE | Gap | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['segment_kind']}: {row['segment']} | {int(row['n'])} | {row[model]:.3f}% | "
            f"{row[TSO_FORECAST]:.3f}% | {row['gap_vs_tso']:+.3f} pp | {row['verdict']} |"
        )

    lines += [
        "",
        "Segments are cut on the Europe/Warsaw clock and calendar. Rows are ordered worst "
        "first: everything above the first *model wins* row is a segment where PSE is better.",
        "",
    ]
    return "\n".join(lines)
