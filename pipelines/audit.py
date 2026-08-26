"""Audit the headline claim: is the model's margin over PSE a fair comparison?

    uv run python -m pipelines.audit                  # the full audit, hours of compute
    uv run python -m pipelines.audit --max-splits 2   # a smoke run

Three backtests over the same recent year, the same origins and the same scored hours,
each removing one advantage the reported number quietly enjoyed:

  A. **flat 24h, observed weather** — the published methodology, on the recent window.
  B. **gate-closure horizons, observed weather** — every hour forecast at the lead time
     PSE actually had for it (`src/evaluation/gate_closure.py`), which removes the
     horizon advantage.
  C. **gate-closure horizons, day-ahead forecast weather** — as B, but the weather
     features are what Open-Meteo *forecast* a day ahead rather than what was later
     observed, which removes the train-serve weather advantage too.

C is the number this project is entitled to publish. A minus C is what the two
unfairnesses were worth, and B separates them.

Writes `reports/audit_h24.md` and the underlying per-hour predictions; the narrative
lives in `docs/evaluation_notes.md`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import Config, load_config, load_project_env
from src.evaluation.backtest import ACTUAL, run_backtest
from src.evaluation.gate_closure import (
    MODEL,
    GateClosureResult,
    gate_closure_horizons,
    run_gate_closure_backtest,
)
from src.evaluation.metrics import metrics_by_segment, point_metrics
from src.ingestion.dataset import TrainingData, load_training_frame
from src.models.baselines import NAIVE_SEASONAL, TSO_FORECAST

DAY_AHEAD_WEATHER = Path("data/processed/weather_day_ahead.parquet")

FLAT = "A. flat 24h horizon, observed weather"
GATE_OBSERVED = "B. gate-closure horizons, observed weather"
GATE_FORECAST = "C. gate-closure horizons, day-ahead forecast weather"


@dataclass(frozen=True)
class Variant:
    """One backtest in the audit, reduced to what the report needs."""

    label: str
    predictions: pd.DataFrame  # `actual`, the model, and the references, on matched rows
    model_column: str

    def metrics(self) -> pd.DataFrame:
        rows = {
            column: point_metrics(self.predictions[ACTUAL], self.predictions[column])
            for column in self.predictions.columns
            if column != ACTUAL
        }
        table = pd.DataFrame(rows).T
        table["mape_vs_tso"] = table["mape"] - table.loc[TSO_FORECAST, "mape"]
        return table

    @property
    def gap(self) -> float:
        return float(self.metrics().loc[self.model_column, "mape_vs_tso"])


def day_ahead_weather_frame(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ("temp_c",),
    path: Path = DAY_AHEAD_WEATHER,
) -> pd.DataFrame:
    """The dataset with observed weather swapped for what was forecast a day earlier.

    Only `columns` are replaced, and the frame is truncated to the hours where every one
    of them is covered. Falling back to observed values outside that span would put the
    advantage being measured straight back in, so the span shrinks instead.

    **Why the default is temperature alone.** Open-Meteo's forecast archive carries
    `temperature_2m` from 2021-03 but wind, cloud and humidity only from 2024-01-19.
    Swapping all four would leave under two years of history before the coverage window,
    so the audited variant would train on a fraction of what the others do and the
    measured gap would be partly a training-size effect. Temperature is also the weather
    variable that moves load: `temp_c` and `temp_sq` are the two the model leans on.
    The remaining variables are bounded separately — see `docs/evaluation_notes.md`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No day-ahead weather at {path}. Fetch it with "
            "`src.ingestion.weather_client.fetch_national_day_ahead` first."
        )

    forecast = pd.read_parquet(path)
    unknown = [c for c in columns if c not in forecast.columns or c not in frame.columns]
    if unknown:
        raise ValueError(f"Cannot swap {unknown}: not present in both frames")

    wanted = list(columns)
    covered = forecast[wanted].dropna().index.intersection(frame.index)
    if covered.empty:
        raise ValueError("Day-ahead weather does not overlap the dataset at all")

    span = frame.index[(frame.index >= covered.min()) & (frame.index <= covered.max())]
    swapped = frame.loc[span].copy()
    swapped[wanted] = forecast.reindex(span)[wanted]
    logger.info(
        "Swapped {} for day-ahead forecast values over {} hours, {} to {}",
        ", ".join(wanted),
        len(swapped),
        swapped.index.min(),
        swapped.index.max(),
    )
    return swapped


@dataclass(frozen=True)
class Origins:
    """The rolling-origin schedule the audit runs on.

    Separate from `cfg.backtest` because the audit refits 25 models per origin instead of
    one, so it trades refit *frequency* for horizon coverage: thirteen monthly blocks
    span the same year as twenty-six fortnightly ones at half the compute, and a model
    left standing for four weeks rather than two understates itself if anything.
    """

    max_splits: int
    test_days: int
    step_days: int

    def first_test_start(self, frame: pd.DataFrame) -> pd.Timestamp:
        """Where coverage must begin for the origins to end at the freshest usable hour.

        Anchoring to the end rather than the start is the difference between reporting
        the most recent year and reporting whichever year happens to sit
        `initial_train_days` after the data begins — which on this series is 2021, the
        oldest.
        """
        last = frame["load_mw"].last_valid_index()
        return (last - pd.Timedelta(days=self.step_days * self.max_splits)).floor("D")

    @property
    def coverage_days(self) -> int:
        return self.step_days * self.max_splits


def _flat_variant(cfg: Config, frame: pd.DataFrame, horizon: int, origins: Origins) -> Variant:
    """The published methodology, re-run on the audit's window and origins."""
    result = run_backtest(
        frame,
        horizon=horizon,
        initial_train_days=cfg.backtest.initial_train_days,
        test_days=origins.test_days,
        step_days=origins.step_days,
        max_splits=origins.max_splits,
        first_test_start=origins.first_test_start(frame),
        inner_validation_days=cfg.model.validation_days,
        tune=False,
        variants=("lgbm_weather",),
        quantiles=(),
        seed=cfg.model.seed,
        num_boost_round=cfg.model.num_boost_round,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
        rolling_window=cfg.features.rolling_window_hours,
        weekly_lag=cfg.features.weekly_lag_hours,
    )
    keep = [ACTUAL, "lgbm_weather", NAIVE_SEASONAL, TSO_FORECAST]
    return Variant(FLAT, result.predictions[keep], "lgbm_weather")


def _gate_variant(cfg: Config, frame: pd.DataFrame, origins: Origins, label: str) -> Variant:
    result: GateClosureResult = run_gate_closure_backtest(
        frame,
        initial_train_days=cfg.backtest.initial_train_days,
        test_days=origins.test_days,
        step_days=origins.step_days,
        max_splits=origins.max_splits,
        first_test_start=origins.first_test_start(frame),
        inner_validation_days=cfg.model.validation_days,
        tz=cfg.data.timezone_local,
        seed=cfg.model.seed,
        num_boost_round=cfg.model.num_boost_round,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
        rolling_window=cfg.features.rolling_window_hours,
        weekly_lag=cfg.features.weekly_lag_hours,
        label=label,
    )
    return Variant(label, result.predictions, MODEL)


def matched(variants: list[Variant]) -> list[Variant]:
    """Restrict every variant to the hours all of them scored.

    Without this the three rows of the headline table would each be an average over a
    slightly different set of hours, which is exactly the flaw this audit exists to find.
    """
    common = variants[0].predictions.index
    for variant in variants[1:]:
        common = common.intersection(variant.predictions.index)
    logger.info(
        "Matched hours across {} variants: {} (from {})",
        len(variants),
        len(common),
        ", ".join(str(len(v.predictions)) for v in variants),
    )
    return [Variant(v.label, v.predictions.loc[common], v.model_column) for v in variants]


def audit_markdown(variants: list[Variant], cfg: Config, dataset_version: str) -> str:
    """The audit table, plus the hour-of-day and segment views behind it."""
    headline = variants[-1]
    index = headline.predictions.index
    lines = [
        "# Evaluation audit — is the margin over PSE a fair comparison?",
        "",
        f"{len(index)} matched out-of-sample hours, {index.min():%Y-%m-%d} to "
        f"{index.max():%Y-%m-%d}, {len(headline.predictions)} rows scored identically for "
        f"every row of every table below. Dataset `{dataset_version}`.",
        "",
        "Each variant removes one advantage the published figure enjoyed. Every model "
        "and every reference is scored on exactly the same timestamps.",
        "",
        "| Variant | Model MAPE | PSE MAPE | Gap | Model RMSE (MW) | PSE RMSE (MW) |",
        "|---|---|---|---|---|---|",
    ]
    for variant in variants:
        table = variant.metrics()
        model, pse = table.loc[variant.model_column], table.loc[TSO_FORECAST]
        lines.append(
            f"| {variant.label} | {model['mape']:.3f}% | {pse['mape']:.3f}% | "
            f"{variant.gap:+.3f} pp | {model['rmse_mw']:.0f} | {pse['rmse_mw']:.0f} |"
        )

    lines += [
        "",
        f"Naive seasonal (`load[T-168]`) over the same hours: "
        f"{headline.metrics().loc[NAIVE_SEASONAL, 'mape']:.3f}% MAPE.",
        "",
        "## Error by hour of day",
        "",
        "PSE publishes once a day, so its lead time grows through the delivery day while "
        "a flat-horizon model's does not. This is where that shows.",
        "",
        "| Local hour | PSE lead (h) | Model MAPE | PSE MAPE | Gap |",
        "|---|---|---|---|---|",
    ]
    lines += _hour_of_day_rows(headline, cfg)

    lines += [
        "",
        "## Segment breakdown — the audited variant",
        "",
        "| Segment | Hours | Model MAPE | PSE MAPE | Gap | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    lines += _segment_rows(headline, cfg)
    lines += [
        "",
        "Segments are cut on the Europe/Warsaw clock and calendar, worst first.",
        "",
    ]
    return "\n".join(lines)


def _hour_of_day_rows(variant: Variant, cfg: Config) -> list[str]:
    predictions = variant.predictions
    local_hour = predictions.index.tz_convert(cfg.data.timezone_local).hour
    lead = gate_closure_horizons(predictions.index, tz=cfg.data.timezone_local)
    ape = pd.DataFrame(
        {
            "hour": local_hour,
            "lead": lead.to_numpy(),
            "model": (predictions[variant.model_column] - predictions[ACTUAL]).abs()
            / predictions[ACTUAL]
            * 100,
            "pse": (predictions[TSO_FORECAST] - predictions[ACTUAL]).abs()
            / predictions[ACTUAL]
            * 100,
        }
    )
    grouped = ape.groupby("hour").mean()
    return [
        f"| {hour:02d}:00 | {row['lead']:.0f} | {row['model']:.3f}% | {row['pse']:.3f}% | "
        f"{row['model'] - row['pse']:+.3f} pp |"
        for hour, row in grouped.iterrows()
    ]


def _segment_rows(variant: Variant, cfg: Config) -> list[str]:
    segments = metrics_by_segment(
        variant.predictions[ACTUAL],
        variant.predictions[[variant.model_column, TSO_FORECAST]],
        tz=cfg.data.timezone_local,
        peak_hours=cfg.evaluation.peak_hours,
    )
    wide = segments.pivot_table(
        index=["segment_kind", "segment", "n"], columns="model", values="mape"
    ).reset_index()
    wide["gap"] = wide[variant.model_column] - wide[TSO_FORECAST]
    wide = wide.sort_values("gap", ascending=False)
    return [
        f"| {row['segment_kind']}: {row['segment']} | {int(row['n'])} | "
        f"{row[variant.model_column]:.3f}% | {row[TSO_FORECAST]:.3f}% | {row['gap']:+.3f} pp | "
        f"{'model wins' if row['gap'] < 0 else 'PSE wins'} |"
        for _, row in wide.iterrows()
    ]


def run(
    cfg: Config,
    *,
    origins: Origins,
    horizon: int = 24,
    swap: tuple[str, ...] = ("temp_c",),
) -> Path:
    """Run all three variants, match their hours, and write the audit report.

    All three see the same training span. The forecast-weather variant's coverage is the
    shorter one, so the other two are truncated to it rather than being handed years of
    extra history — otherwise the audited difference would be partly a training-size
    difference, which is the sort of thing an audit is supposed to catch.
    """
    data: TrainingData = load_training_frame(cfg)
    if data.synthetic:
        raise RuntimeError("The audit is meaningless on synthetic data; ingest first")

    forecast_frame = day_ahead_weather_frame(data.frame, columns=swap)
    observed_frame = data.frame.loc[forecast_frame.index]

    variants = [
        _flat_variant(cfg, observed_frame, horizon, origins),
        _gate_variant(cfg, observed_frame, origins, GATE_OBSERVED),
        _gate_variant(cfg, forecast_frame, origins, GATE_FORECAST),
    ]
    variants = matched(variants)

    reports = cfg.backtest.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"audit_h{horizon}.md"
    report_path.write_text(audit_markdown(variants, cfg, data.version))

    for variant in variants:
        name = variant.label.split(".", 1)[0].strip().lower()
        variant.predictions.to_csv(reports / f"audit_{name}_h{horizon}.csv")

    for variant in variants:
        logger.info("{}: gap vs PSE {:+.3f} pp", variant.label, variant.gap)
    logger.info("Wrote {}", report_path)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the PSE comparison for fairness.")
    parser.add_argument("--max-splits", type=int, default=13, help="number of origins")
    parser.add_argument("--test-days", type=int, default=28, help="days each origin predicts")
    parser.add_argument("--step-days", type=int, default=28, help="how far the origin moves")
    parser.add_argument("--horizon", type=int, default=24, help="the flat-horizon reference")
    parser.add_argument(
        "--swap",
        action="append",
        help="weather column to take from the day-ahead archive (repeatable)",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    load_project_env()
    cfg = load_config(args.config)
    run(
        cfg,
        origins=Origins(args.max_splits, args.test_days, args.step_days),
        horizon=args.horizon,
        swap=tuple(args.swap or ("temp_c",)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
