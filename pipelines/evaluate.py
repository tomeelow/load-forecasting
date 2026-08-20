"""Score what the service actually served against what actually happened.

    uv run python -m pipelines.evaluate

This is the step that turns a prediction log into a performance measurement. It joins
every logged prediction to the latest available actual and re-scores the lot, because
ENTSO-E revises published values and a score computed against a provisional number can
become wrong without anything else changing.

The comparison against PSE matters as much as the absolute number: both are scored on
exactly the same hours, so "our rolling MAPE was 2.1%" is always accompanied by what the
operator's own forecast managed over the same period.

Reports "insufficient data" rather than a number when too few hours have been scored —
in the first days of operation there is nothing meaningful to say, and saying it anyway
is how a monitoring system loses credibility.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import Config, load_config, load_project_env
from src.evaluation.metrics import mape, point_metrics
from src.ingestion.dataset import read_dataset
from src.pipeline_state import PipelineState
from src.prediction_log import PredictionLog

PIPELINE = "evaluate"


@dataclass(frozen=True)
class EvaluationSummary:
    """Production performance over the scored window."""

    scored: int
    window_days: int
    model_mape: float | None
    tso_mape: float | None
    model_metrics: dict[str, float]
    sufficient: bool

    @property
    def versus_tso(self) -> float | None:
        if self.model_mape is None or self.tso_mape is None:
            return None
        return self.model_mape - self.tso_mape

    def format(self) -> str:
        if not self.sufficient:
            return (
                f"Production performance: insufficient data — {self.scored} scored "
                f"prediction(s) in the last {self.window_days} days"
            )
        lines = [
            f"Production performance over {self.window_days} days ({self.scored} scored hour(s))",
            f"  served model   MAPE {self.model_mape:.3f}%  "
            f"RMSE {self.model_metrics['rmse_mw']:.0f} MW  "
            f"MAE {self.model_metrics['mae_mw']:.0f} MW  "
            f"bias {self.model_metrics['bias_mw']:+.0f} MW",
        ]
        if self.tso_mape is not None:
            gap = self.versus_tso
            lines.append(
                f"  PSE day-ahead  MAPE {self.tso_mape:.3f}%"
                f"   ({gap:+.3f} pp {'better' if gap < 0 else 'worse'} for us)"
            )
        return "\n".join(lines)


def run(
    cfg: Config,
    *,
    log: PredictionLog | None = None,
    state: PipelineState | None = None,
    now: pd.Timestamp | None = None,
) -> EvaluationSummary:
    """Re-score the prediction log against the latest actuals and summarise."""
    log = log or PredictionLog(cfg.state.prediction_log_path)
    state = state or PipelineState(cfg.state.pipeline_state_path)
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")

    dataset = read_dataset(cfg.data.dataset_path)
    rescored = log.score(dataset)
    logger.info("Re-scored {} prediction(s) against the latest published actuals", rescored)

    window_days = cfg.monitoring.current_days
    scored = log.scored(since=now - pd.Timedelta(days=window_days))
    scored = scored[scored.index <= now] if not scored.empty else scored

    sufficient = len(scored) >= cfg.monitoring.min_scored_predictions
    summary = EvaluationSummary(
        scored=len(scored),
        window_days=window_days,
        model_mape=float(scored["ape"].mean() * 100) if sufficient else None,
        tso_mape=_tso_mape(scored) if sufficient else None,
        model_metrics=point_metrics(scored["actual_mw"], scored["load_mw"]) if sufficient else {},
        sufficient=sufficient,
    )
    logger.info("\n{}", summary.format())
    state.record_success(PIPELINE, now)
    return summary


def _tso_mape(scored: pd.DataFrame) -> float | None:
    with_tso = scored.dropna(subset=["tso_mw"])
    return None if with_tso.empty else mape(with_tso["actual_mw"], with_tso["tso_mw"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join logged predictions to actuals and report production performance."
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    load_project_env()
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
