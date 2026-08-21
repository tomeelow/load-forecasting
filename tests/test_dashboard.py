"""The dashboard's data layer, especially where the data is not there.

A dashboard fails in a particular way: it renders. A panel that quietly shows the wrong
window, or the backtest's numbers under a heading about production, looks exactly like a
panel that is right. So these tests are mostly about the *empty* and *partial* paths —
no registry, no prediction log, a handful of scored hours — because those are the states
a fresh deployment is in and the states nobody looks at before publishing a link.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src import synthetic
from src.dashboard import data
from src.ingestion.dataset import write_dataset
from src.models.baselines import NAIVE_SEASONAL, TSO_FORECAST
from src.monitoring.history import DriftHistory
from src.prediction_log import PredictionLog, PredictionRecord

NOW = pd.Timestamp("2024-03-01 12:00", tz="UTC")


@pytest.fixture(scope="module")
def dataset():
    return synthetic.make_dataset(start="2024-01-01", end="2024-03-01")


@pytest.fixture
def bare_cfg(cfg, tmp_path):
    """A configuration pointing at empty directories — a first-run deployment."""
    return dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, processed_dir=tmp_path / "data", dataset_filename="dataset.parquet"
        ),
        state=dataclasses.replace(cfg.state, dir=tmp_path / "state"),
        backtest=dataclasses.replace(cfg.backtest, reports_dir=tmp_path / "reports"),
        monitoring=dataclasses.replace(cfg.monitoring, reports_dir=tmp_path / "monitoring"),
        mlflow=dataclasses.replace(
            cfg.mlflow, tracking_uri=f"sqlite:///{tmp_path}/empty/mlflow.db"
        ),
    )


@pytest.fixture
def populated_cfg(bare_cfg, dataset):
    write_dataset(dataset, bare_cfg.data.dataset_path)
    return bare_cfg


def log_served(cfg, dataset, hours: int, *, error: float = 0.02) -> PredictionLog:
    log = PredictionLog(cfg.state.prediction_log_path)
    actual = dataset["load_mw"].dropna()
    targets = actual.index[-hours:]
    log.log(
        [
            PredictionRecord(
                predicted_at=target - pd.Timedelta(hours=24),
                target_time=target,
                horizon_hours=24,
                load_mw=float(actual.loc[target]) * (1 + error),
                p10=float(actual.loc[target]) * 0.95,
                p50=float(actual.loc[target]),
                p90=float(actual.loc[target]) * 1.05,
                model_name="pl_load_lgbm",
                model_version="4",
                run_id="run-abc",
                dataset_version="hash1234",
            )
            for target in targets
        ]
    )
    return log


def backtest_csv(cfg, dataset, model_column: str = data.BACKTEST_MODEL) -> None:
    actual = dataset["load_mw"].dropna()
    frame = pd.DataFrame(
        {
            "actual": actual,
            model_column: actual * 1.01,
            TSO_FORECAST: actual * 1.03,
            NAIVE_SEASONAL: actual * 1.05,
        }
    )
    cfg.backtest.reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.backtest.reports_dir / "predictions_h24.csv")


class TestEmptyDeployment:
    """Nothing has run yet. Every loader must answer, and none may raise."""

    def test_the_model_card_reports_why_there_is_no_champion(self, bare_cfg):
        card = data.load_model_card(bare_cfg)

        assert not card.available
        assert card.error
        assert card.data_source == "unknown"

    def test_a_missing_dataset_yields_an_empty_frame_not_a_crash(self, bare_cfg):
        assert data.load_recent_actuals(bare_cfg).empty

    def test_an_empty_prediction_log_yields_no_served_forecast(self, bare_cfg):
        assert data.load_served_forecast(bare_cfg).empty

    def test_an_empty_prediction_log_reports_empty_not_zero_percent(self, bare_cfg):
        """Zero scored hours must not become a 0% MAPE anywhere on the page."""
        performance = data.load_served_performance(bare_cfg)

        assert performance.status == "empty"
        assert not performance.sufficient
        assert performance.model_mape is None
        assert performance.tso_mape is None

    def test_an_absent_drift_history_reads_as_an_empty_typed_frame(self, bare_cfg):
        history = data.load_drift_history(bare_cfg)

        assert history.empty
        assert "drift_share" in history.columns
        assert history.index.tz is not None

    def test_no_drift_report_on_disk_is_none(self, bare_cfg):
        assert data.latest_drift_report(bare_cfg) is None

    def test_no_backtest_predictions_reads_as_unavailable(self, bare_cfg):
        assert not data.load_backtest(bare_cfg).available

    def test_no_audit_report_is_none(self, bare_cfg):
        assert data.audit_summary(bare_cfg) is None


class TestServedPerformance:
    """The accumulating state is the one the deployment will sit in for a week."""

    def test_a_handful_of_scored_hours_is_not_enough_to_report(self, populated_cfg, dataset):
        log = log_served(populated_cfg, dataset, hours=20)
        log.score(dataset)

        performance = data.load_served_performance(populated_cfg)

        assert performance.status == "accumulating"
        assert not performance.sufficient
        assert 0 < len(performance.scored) < performance.required

    def test_logged_but_unscored_predictions_still_count_as_empty(self, populated_cfg, dataset):
        """A forecast whose hour has not happened yet says nothing about accuracy."""
        log_served(populated_cfg, dataset, hours=24)

        performance = data.load_served_performance(populated_cfg)

        assert performance.status == "empty"
        assert performance.logged == 24

    def test_enough_scored_hours_flips_it_to_measuring(self, populated_cfg, dataset):
        log = log_served(populated_cfg, dataset, hours=200)
        log.score(dataset)

        performance = data.load_served_performance(populated_cfg)

        assert performance.status == "measuring"
        assert performance.sufficient
        assert performance.model_mape == pytest.approx(2.0, abs=0.3)

    def test_pse_is_reported_on_the_hours_the_model_was_scored_on(self, populated_cfg, dataset):
        log = log_served(populated_cfg, dataset, hours=200)
        log.score(dataset)

        performance = data.load_served_performance(populated_cfg)

        assert performance.tso_mape is not None
        assert performance.tso_mape > 0

    def test_the_threshold_comes_from_config_not_from_the_page(self, populated_cfg):
        performance = data.load_served_performance(populated_cfg)

        assert performance.required == populated_cfg.monitoring.min_scored_predictions


class TestServedForecast:
    def test_the_band_survives_the_round_trip(self, populated_cfg, dataset):
        log_served(populated_cfg, dataset, hours=24)

        served = data.load_served_forecast(populated_cfg)

        assert (served["p10"] < served["p90"]).all()

    def test_only_the_most_recent_forecast_per_hour_is_shown(self, populated_cfg, dataset):
        """Two runs forecasting the same hour must not draw two lines."""
        log_served(populated_cfg, dataset, hours=24)
        log_served(populated_cfg, dataset, hours=24, error=0.05)

        served = data.load_served_forecast(populated_cfg)

        assert not served.index.has_duplicates


class TestBacktestEvidence:
    def test_it_reads_the_predictions_and_finds_the_model_column(self, populated_cfg, dataset):
        backtest_csv(populated_cfg, dataset)

        backtest = data.load_backtest(populated_cfg)

        assert backtest.available
        assert backtest.model_column == data.BACKTEST_MODEL

    def test_the_audited_run_is_preferred_over_the_flat_horizon_one(self, populated_cfg, dataset):
        """The flat-horizon figure compares two different products; prefer the fair one."""
        backtest_csv(populated_cfg, dataset)
        actual = dataset["load_mw"].dropna()
        pd.DataFrame(
            {
                "actual": actual,
                data.AUDIT_MODEL: actual * 1.02,
                TSO_FORECAST: actual * 1.03,
            }
        ).to_csv(populated_cfg.backtest.reports_dir / "audit_c_h24.csv")

        backtest = data.load_backtest(populated_cfg)

        assert backtest.model_column == data.AUDIT_MODEL
        assert "gate-closure" in backtest.label

    def test_the_flat_horizon_run_is_used_when_no_audit_exists(self, populated_cfg, dataset):
        backtest_csv(populated_cfg, dataset)

        backtest = data.load_backtest(populated_cfg)

        assert backtest.model_column == data.BACKTEST_MODEL

    def test_the_index_comes_back_as_utc(self, populated_cfg, dataset):
        backtest_csv(populated_cfg, dataset)

        assert data.load_backtest(populated_cfg).predictions.index.tz is not None

    def test_overall_scores_every_column_against_the_same_actuals(self, populated_cfg, dataset):
        backtest_csv(populated_cfg, dataset)

        overall = data.load_backtest(populated_cfg).overall()

        assert set(overall.index) >= {data.BACKTEST_MODEL, TSO_FORECAST, NAIVE_SEASONAL}
        assert overall.loc[data.BACKTEST_MODEL, "mape_vs_tso"] < 0

    def test_rolling_error_puts_the_model_and_pse_on_one_axis(self, populated_cfg, dataset):
        backtest_csv(populated_cfg, dataset)

        rolling = data.load_backtest(populated_cfg).rolling_mape(window_days=7)

        assert list(rolling.columns) == ["model", "PSE"]
        assert (rolling["model"] < rolling["PSE"]).all()

    def test_the_segment_table_reports_where_pse_wins_first(self, populated_cfg, dataset):
        backtest_csv(populated_cfg, dataset)

        segments = data.load_backtest(populated_cfg).by_segment((7, 22))

        assert segments["gap"].is_monotonic_decreasing
        assert {"segment_kind", "segment", "n", "gap"} <= set(segments.columns)


class TestDriftHistory:
    @staticmethod
    def result(share: float, when: str, status: str = "drift"):
        from src.monitoring.drift import DriftResult

        return DriftResult(
            status=status,
            drift_share=share,
            drifted_features=["temp_c", "wind_ms"],
            reference_strategy="seasonal",
            reference_rows=2016,
            current_rows=336,
            scored_predictions=12,
            checked_at=pd.Timestamp(when, tz="UTC").to_pydatetime(),
        )

    def test_each_check_appends_one_row(self, tmp_path):
        history = DriftHistory(tmp_path / "drift.csv")

        history.append(self.result(0.45, "2026-08-01"), monitored_features=11)
        history.append(self.result(0.55, "2026-08-02"), monitored_features=11)

        assert len(history.read()) == 2

    def test_the_share_reads_back_as_a_number_on_a_time_index(self, tmp_path):
        history = DriftHistory(tmp_path / "drift.csv")
        history.append(self.result(0.45, "2026-08-01"), monitored_features=11)

        frame = history.read()

        assert frame["drift_share"].iloc[0] == pytest.approx(0.45)
        assert frame.index[0] == pd.Timestamp("2026-08-01", tz="UTC")

    def test_it_keeps_the_names_so_a_repeating_feature_is_visible(self, tmp_path):
        history = DriftHistory(tmp_path / "drift.csv")
        history.append(self.result(0.45, "2026-08-01"), monitored_features=11)

        assert history.read()["drifted_names"].iloc[0] == "temp_c wind_ms"

    def test_quiet_checks_are_recorded_too(self, tmp_path):
        """A trend of nothing happening is what makes something happening legible."""
        history = DriftHistory(tmp_path / "drift.csv")
        history.append(self.result(0.09, "2026-08-01", status="ok"), monitored_features=11)

        assert history.read()["status"].iloc[0] == "ok"

    def test_history_is_returned_oldest_first(self, tmp_path):
        history = DriftHistory(tmp_path / "drift.csv")
        history.append(self.result(0.55, "2026-08-02"), monitored_features=11)
        history.append(self.result(0.45, "2026-08-01"), monitored_features=11)

        assert history.read().index.is_monotonic_increasing


class TestSyntheticLabelling:
    """A model trained on generated data must never be presentable as a real result."""

    def test_a_synthetic_run_is_flagged_unmistakably(self):
        assert data.ModelCard(version="1", synthetic=True).data_source == "SYNTHETIC"

    def test_an_ingested_run_says_so(self):
        assert data.ModelCard(version="1", synthetic=False).data_source == "real ENTSO-E"

    def test_an_untagged_run_is_unknown_rather_than_assumed_real(self):
        assert data.ModelCard(version="1").data_source == "unknown"
