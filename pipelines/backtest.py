"""Backtest entrypoint: the rolling-origin evaluation and the benchmark table.

    uv run python -m pipelines.backtest                  # every horizon in config
    uv run python -m pipelines.backtest --horizon 24
    uv run python -m pipelines.backtest --max-splits 4   # a quick smoke run

Writes `reports/benchmark_h<H>.md` and `reports/segments_h<H>.csv`, and logs the whole
evaluation to MLflow as its own run so the numbers are traceable to the data and code
that produced them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
from loguru import logger

from src.config import Config, load_config, load_project_env
from src.evaluation.backtest import BacktestResult, benchmark_markdown, run_backtest
from src.ingestion.dataset import TrainingData, load_training_frame
from src.models.tracking import configure


def run(cfg: Config, horizon: int, *, max_splits: int | None = None, tune: bool = True) -> Path:
    """Backtest one horizon, write the reports, and log the run."""
    data: TrainingData = load_training_frame(cfg)

    result = run_backtest(
        data.frame,
        horizon=horizon,
        initial_train_days=cfg.backtest.initial_train_days,
        test_days=cfg.backtest.test_days,
        step_days=cfg.backtest.step_days,
        max_splits=max_splits if max_splits is not None else cfg.backtest.max_splits,
        inner_validation_days=cfg.model.validation_days,
        tune=tune,
        n_trials=cfg.model.tuning.n_trials,
        tuning_timeout_s=cfg.model.tuning.timeout_s,
        quantiles=tuple(cfg.model.quantiles) if cfg.backtest.quantiles else (),
        seed=cfg.model.seed,
        num_boost_round=cfg.model.num_boost_round,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
        rolling_window=cfg.features.rolling_window_hours,
        weekly_lag=cfg.features.weekly_lag_hours,
    )

    reports = cfg.backtest.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    markdown = benchmark_markdown(result, synthetic=data.synthetic, dataset_version=data.version)
    report_path = reports / f"benchmark_h{horizon}.md"
    report_path.write_text(markdown)

    segments = result.by_segment(cfg.evaluation.peak_hours)
    segments.to_csv(reports / f"segments_h{horizon}.csv", index=False)
    result.predictions.to_csv(reports / f"predictions_h{horizon}.csv")

    _log_to_mlflow(cfg, result, data, horizon, report_path)

    logger.info("\n{}", result.overall().round(3).to_string())
    logger.info("Wrote {} and the segment breakdown alongside it", report_path)
    return report_path


def _log_to_mlflow(
    cfg: Config, result: BacktestResult, data: TrainingData, horizon: int, report: Path
) -> None:
    suffix = "_synthetic" if data.synthetic else ""
    with mlflow.start_run(run_name=f"backtest_h{horizon}{suffix}"):
        mlflow.set_tags(
            {
                "run_type": "backtest",
                "horizon": str(horizon),
                "data_source": "synthetic" if data.synthetic else "ingested",
                "synthetic": str(data.synthetic).lower(),
            }
        )
        mlflow.log_params(
            {
                "horizon_hours": horizon,
                "dataset_version": data.version,
                "origins": len(result.splits),
                "initial_train_days": cfg.backtest.initial_train_days,
                "test_days": cfg.backtest.test_days,
                "step_days": cfg.backtest.step_days,
                "embargo_hours": horizon,
                **{f"tuned_{k}": v for k, v in result.tuned_params.items()},
            }
        )
        overall = result.overall()
        for model, row in overall.iterrows():
            for metric, value in row.items():
                mlflow.log_metric(f"{model}.{metric}", float(value))
        mlflow.log_metrics(result.pinball())

        mlflow.log_artifact(str(report))
        mlflow.log_table(result.by_segment(cfg.evaluation.peak_hours), "segments.json")
        mlflow.log_table(overall.reset_index(names="model"), "benchmark.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rolling-origin backtest producing the benchmark table against PSE."
    )
    parser.add_argument("--horizon", type=int, action="append", help="override config horizons")
    parser.add_argument("--max-splits", type=int, default=None, help="cap the number of origins")
    parser.add_argument("--no-tune", action="store_true", help="skip the Optuna search")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    load_project_env()
    cfg = load_config(args.config)
    configure(cfg.mlflow)

    for horizon in args.horizon or cfg.model.horizons:
        run(cfg, horizon, max_splits=args.max_splits, tune=not args.no_tune)
    return 0


if __name__ == "__main__":
    sys.exit(main())
