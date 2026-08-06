"""The prediction log: write, read back, and score against actuals that move.

If this is wrong, Phase 8 measures nothing — every drift and performance number is a
join against this table. The revision case matters most: ENTSO-E publishes a
provisional actual and corrects it later, and a score computed against the provisional
value has to be replaced rather than kept.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.prediction_log import PredictionLog, PredictionRecord

MADE_AT = pd.Timestamp("2026-08-05 12:00", tz="UTC")
TARGET = pd.Timestamp("2026-08-06 12:00", tz="UTC")


@pytest.fixture
def log(tmp_path) -> PredictionLog:
    return PredictionLog(tmp_path / "predictions.db")


def record(target=TARGET, load_mw=20_000.0, version="1", made_at=MADE_AT) -> PredictionRecord:
    return PredictionRecord(
        predicted_at=made_at,
        target_time=target,
        horizon_hours=24,
        load_mw=load_mw,
        p10=load_mw - 800,
        p50=load_mw,
        p90=load_mw + 800,
        model_name="pl_load_lgbm",
        model_version=version,
        run_id="run-abc",
        dataset_version="ds-123",
        features={"hour": 14.0, "temp_c": 21.5},
    )


def actuals(load_mw: float = 20_500.0, tso: float | None = 20_300.0) -> pd.DataFrame:
    index = pd.date_range(TARGET - pd.Timedelta(hours=2), periods=5, freq="1h", tz="UTC")
    frame = pd.DataFrame({"load_mw": load_mw}, index=index)
    if tso is not None:
        frame["tso_forecast_mw"] = tso
    return frame


def test_a_prediction_survives_the_round_trip(log):
    log.log([record()])

    stored = log.read()

    assert len(stored) == 1
    row = stored.iloc[0]
    assert stored.index[0] == TARGET
    assert row["load_mw"] == pytest.approx(20_000.0)
    assert (row["p10"], row["p90"]) == pytest.approx((19_200.0, 20_800.0))
    assert row["model_version"] == "1"
    assert row["dataset_version"] == "ds-123"


def test_the_feature_vector_is_recoverable(log):
    log.log([record()])

    stored = log.read(with_features=True)

    assert stored.iloc[0]["temp_c"] == pytest.approx(21.5)
    assert stored.iloc[0]["hour"] == pytest.approx(14.0)


def test_the_schema_version_is_recorded(log):
    assert log.schema_version == 1


def test_a_log_written_by_a_future_schema_is_refused(tmp_path):
    path = tmp_path / "predictions.db"
    PredictionLog(path)
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_version SET version = 99")

    with pytest.raises(RuntimeError, match="schema v99"):
        PredictionLog(path)


def test_a_naive_timestamp_is_refused(log):
    naive = PredictionRecord(
        predicted_at=pd.Timestamp("2026-08-05 12:00"),
        target_time=TARGET,
        horizon_hours=24,
        load_mw=20_000.0,
        model_name="m",
        model_version="1",
    )

    with pytest.raises(ValueError, match="naive timestamp"):
        log.log([naive])


def test_logging_nothing_is_allowed(log):
    assert log.log([]) == 0
    assert log.read().empty


def test_reading_an_empty_log_gives_an_empty_frame_not_an_error(log):
    frame = log.read()

    assert frame.empty
    assert frame.index.name == "target_time"
    assert str(frame.index.tz) == "UTC"


def test_every_served_prediction_is_kept(log):
    """Two calls for the same hour are two facts about what was served."""
    log.log([record(load_mw=20_000.0, made_at=MADE_AT)])
    log.log([record(load_mw=20_400.0, made_at=MADE_AT + pd.Timedelta(hours=1))])

    assert len(log.read()) == 2


def test_monitoring_sees_only_the_forecast_that_stood(log):
    log.log([record(load_mw=20_000.0, made_at=MADE_AT)])
    log.log([record(load_mw=20_400.0, made_at=MADE_AT + pd.Timedelta(hours=1))])

    latest = log.latest_per_target()

    assert len(latest) == 1
    assert latest.iloc[0]["load_mw"] == pytest.approx(20_400.0)


def test_predictions_are_scored_against_actuals(log):
    log.log([record(load_mw=20_000.0)])

    assert log.score(actuals(load_mw=20_500.0, tso=20_300.0)) == 1

    scored = log.scored()
    row = scored.iloc[0]
    assert row["actual_mw"] == pytest.approx(20_500.0)
    assert row["abs_error_mw"] == pytest.approx(500.0)
    assert row["ape"] == pytest.approx(500 / 20_500)
    # The benchmark is scored on the same hour, so the comparison is like for like.
    assert row["tso_abs_error_mw"] == pytest.approx(200.0)


def test_a_revised_actual_replaces_the_earlier_score(log):
    """The case that makes this table hard: the truth itself is republished."""
    log.log([record(load_mw=20_000.0)])
    log.score(actuals(load_mw=20_500.0))
    first = log.scored().iloc[0]

    log.score(actuals(load_mw=21_000.0))  # ENTSO-E revises the published value
    revised = log.scored().iloc[0]

    assert len(log.scored()) == 1, "a revision must update the score, not add another"
    assert revised["actual_mw"] == pytest.approx(21_000.0)
    assert revised["abs_error_mw"] == pytest.approx(1_000.0)
    assert revised["scored_at"] >= first["scored_at"]


def test_scoring_twice_with_the_same_actuals_is_idempotent(log):
    log.log([record()])

    assert log.score(actuals()) == 1
    assert log.score(actuals()) == 1
    assert len(log.scored()) == 1


def test_predictions_without_an_actual_yet_are_left_unscored(log):
    log.log([record(target=pd.Timestamp("2027-01-01 00:00", tz="UTC"))])

    assert log.score(actuals()) == 0
    assert log.scored().empty


def test_scoring_an_empty_log_is_not_an_error(log):
    assert log.score(actuals()) == 0


def test_scoring_with_no_actuals_available_is_not_an_error(log):
    log.log([record()])
    empty = actuals()
    empty["load_mw"] = float("nan")

    assert log.score(empty) == 0


def test_scoring_needs_a_utc_indexed_frame_with_load(log):
    log.log([record()])

    with pytest.raises(ValueError, match="load_mw"):
        log.score(pd.DataFrame({"other": [1.0]}, index=actuals().index[:1]))

    naive = actuals()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        log.score(naive)


def test_actuals_without_a_tso_forecast_still_score(log):
    log.log([record()])

    log.score(actuals(tso=None))

    row = log.scored().iloc[0]
    assert row["abs_error_mw"] == pytest.approx(500.0)
    assert pd.isna(row["tso_mw"])


def test_reads_can_be_windowed_by_target_hour(log):
    log.log([record(target=TARGET), record(target=TARGET + pd.Timedelta(days=2))])

    assert len(log.read()) == 2
    assert len(log.read(since=TARGET + pd.Timedelta(days=1))) == 1
    assert len(log.read(until=TARGET)) == 1


def test_reads_can_be_filtered_by_model_version(log):
    log.log([record(version="1"), record(version="2")])

    assert len(log.read(model_version="2")) == 1
    assert log.count() == 2
