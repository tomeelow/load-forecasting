"""Run the drift checks and raise the retrain flag when something is actually wrong.

    uv run python -m pipelines.check_drift

This is the half of monitoring that makes it more than decoration. A drift report saved
to disk and never read changes nothing; what changes something is the flag this writes,
which `pipelines.retrain_if_needed` consumes on the same schedule.

Thresholds live in `config/config.yaml`, never here — the point at which drift becomes
worth retraining for is an operational judgement that will be revised, and revising it
should not require a code change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import Config, load_config, load_project_env
from src.features.builder import feature_columns, make_features
from src.ingestion.dataset import read_dataset
from src.monitoring.drift import DriftResult, check_drift, monitored_columns
from src.monitoring.history import DriftHistory
from src.pipeline_state import PipelineState
from src.prediction_log import PredictionLog

PIPELINE = "check_drift"


def run(
    cfg: Config,
    *,
    log: PredictionLog | None = None,
    state: PipelineState | None = None,
    history: DriftHistory | None = None,
    now: pd.Timestamp | None = None,
    write_report: bool = True,
) -> DriftResult:
    """Check for drift, and set the retrain flag if it is warranted."""
    log = log or PredictionLog(cfg.state.prediction_log_path)
    state = state or PipelineState(cfg.state.pipeline_state_path)
    history = history or DriftHistory(cfg.state.drift_history_path)

    dataset = read_dataset(cfg.data.dataset_path)
    result = check_drift(cfg, dataset, log, now=now, write_report=write_report)

    # Recorded before the flag is decided, so a run that finds nothing is a data point
    # too — a trend of quiet nights is what makes a noisy one mean something.
    history.append(result, monitored_features=_monitored_count(cfg, dataset))

    if result.should_retrain:
        state.raise_retrain_flag("; ".join(result.reasons))
    else:
        logger.info("No retrain warranted: {}", "; ".join(result.reasons))

    state.record_success(PIPELINE, now)
    return result


def _monitored_count(cfg: Config, dataset: pd.DataFrame) -> int | None:
    """How many inputs the share was taken over, so a share can be read back as a count."""
    try:
        features = make_features(
            dataset,
            cfg.model.horizons[0],
            rolling_window=cfg.features.rolling_window_hours,
            weekly_lag=cfg.features.weekly_lag_hours,
        )
    except ValueError:
        return None
    return len(monitored_columns(feature_columns(list(features.columns))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check data, prediction and performance drift; set the retrain flag."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-report", action="store_true", help="skip writing the HTML report")
    args = parser.parse_args(argv)

    load_project_env()
    run(load_config(args.config), write_report=not args.no_report)
    # Drift is a finding, not a failure: exiting non-zero here would fail the workflow
    # and stop the retraining step that is supposed to respond to it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
