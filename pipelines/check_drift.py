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
from dotenv import load_dotenv
from loguru import logger

from src.config import Config, load_config
from src.ingestion.dataset import read_dataset
from src.monitoring.drift import DriftResult, check_drift
from src.pipeline_state import PipelineState
from src.prediction_log import PredictionLog

PIPELINE = "check_drift"


def run(
    cfg: Config,
    *,
    log: PredictionLog | None = None,
    state: PipelineState | None = None,
    now: pd.Timestamp | None = None,
    write_report: bool = True,
) -> DriftResult:
    """Check for drift, and set the retrain flag if it is warranted."""
    log = log or PredictionLog(cfg.state.prediction_log_path)
    state = state or PipelineState(cfg.state.pipeline_state_path)

    dataset = read_dataset(cfg.data.dataset_path)
    result = check_drift(cfg, dataset, log, now=now, write_report=write_report)

    if result.should_retrain:
        state.raise_retrain_flag("; ".join(result.reasons))
    else:
        logger.info("No retrain warranted: {}", "; ".join(result.reasons))

    state.record_success(PIPELINE, now)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check data, prediction and performance drift; set the retrain flag."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-report", action="store_true", help="skip writing the HTML report")
    args = parser.parse_args(argv)

    load_dotenv()
    run(load_config(args.config), write_report=not args.no_report)
    # Drift is a finding, not a failure: exiting non-zero here would fail the workflow
    # and stop the retraining step that is supposed to respond to it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
