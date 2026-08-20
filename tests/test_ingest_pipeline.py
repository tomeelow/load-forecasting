"""The ingestion entrypoint end to end, with the two network calls stubbed out.

This is the path that runs on a schedule, so the parts worth proving are the ones a
unit test of any single module misses: that an incremental run merges onto what is
already on disk instead of replacing it, and that revised values win.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from pipelines import ingest
from src import synthetic
from src.ingestion.dataset import read_dataset, write_dataset
from src.ingestion.entsoe_client import FORECAST_COLUMN, LOAD_COLUMN
from src.pipeline_state import PipelineState

WEATHER_COLUMNS = ["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]


@pytest.fixture
def tmp_cfg(cfg, tmp_path):
    """The real config, pointed at a temporary data directory."""
    data = dataclasses.replace(cfg.data, processed_dir=tmp_path, raw_dir=tmp_path)
    state = dataclasses.replace(cfg.state, dir=tmp_path)
    return dataclasses.replace(cfg, data=data, state=state)


def stub_sources(monkeypatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(ingest, "make_client", lambda: object())
    monkeypatch.setattr(
        ingest,
        "fetch_load_frame",
        lambda *a, **k: frame[[LOAD_COLUMN, FORECAST_COLUMN]],
    )
    monkeypatch.setattr(ingest, "_weather", lambda *a, **k: frame[WEATHER_COLUMNS])


def test_a_full_run_writes_a_validated_dataset(tmp_cfg, monkeypatch):
    frame = synthetic.make_dataset(start="2024-01-01", end="2024-02-01")
    stub_sources(monkeypatch, frame)

    path, report = ingest.run(tmp_cfg, full=True)

    assert path.exists()
    assert report.ok
    written = read_dataset(path)
    assert len(written) == len(frame)
    assert str(written.index.tz) == "UTC"


def test_an_incremental_run_merges_onto_what_is_already_stored(tmp_cfg, monkeypatch):
    history = synthetic.make_dataset(start="2024-01-01", end="2024-02-01")
    stub_sources(monkeypatch, history)
    ingest.run(tmp_cfg, full=True)

    # A later window, overlapping the tail of what is stored, with revised actuals.
    revision = synthetic.make_dataset(start="2024-01-25", end="2024-02-10")
    revision = revision.assign(load_mw=revision["load_mw"] + 1_000)
    stub_sources(monkeypatch, revision)

    path, report = ingest.run(tmp_cfg, full=False)

    merged = read_dataset(path)
    assert merged.index.min() == history.index.min()
    assert merged.index.max() == revision.index.max()
    assert merged.index.equals(
        pd.date_range(merged.index.min(), merged.index.max(), freq="1h", tz="UTC")
    )
    # Revised hours took the new values; hours outside the window kept the old ones.
    assert merged.loc["2024-01-26 00:00", LOAD_COLUMN] == pytest.approx(
        revision.loc["2024-01-26 00:00", LOAD_COLUMN]
    )
    assert merged.loc["2024-01-10 00:00", LOAD_COLUMN] == pytest.approx(
        history.loc["2024-01-10 00:00", LOAD_COLUMN]
    )
    assert report.ok


def test_the_exit_code_reports_validation_failure(tmp_cfg, monkeypatch):
    broken = synthetic.make_dataset(start="2024-01-01", end="2024-02-01")
    broken.iloc[100, broken.columns.get_loc(LOAD_COLUMN)] = 91_000.0  # a whole grid too much
    stub_sources(monkeypatch, broken)
    monkeypatch.setattr(ingest, "load_config", lambda _: tmp_cfg)

    exit_code = ingest.main([])

    # The data is still written; the non-zero code is what a scheduled run reacts to.
    assert exit_code == 1
    assert tmp_cfg.data.dataset_path.exists()


def test_running_twice_for_the_same_period_changes_nothing(tmp_cfg, monkeypatch):
    """GitHub's cron can fire twice for the same day; that must be a no-op, not a mess."""
    frame = synthetic.make_dataset(start="2024-01-01", end="2024-02-01")
    stub_sources(monkeypatch, frame)

    ingest.run(tmp_cfg, full=True)
    first = read_dataset(tmp_cfg.data.dataset_path)

    ingest.run(tmp_cfg, full=False)
    second = read_dataset(tmp_cfg.data.dataset_path)

    assert not second.index.has_duplicates
    assert len(second) == len(first)
    pd.testing.assert_index_equal(first.index, second.index)


def test_a_missed_run_is_backfilled_rather_than_skipped(tmp_cfg, tmp_path):
    """The failure this prevents is silent: a hole nothing ever goes back for."""
    state = PipelineState(tmp_path / "state.json")
    now = pd.Timestamp("2026-08-06 05:00", tz="UTC")
    state.record_success(ingest.PIPELINE, now - pd.Timedelta(days=40))

    start, end = ingest._window(
        tmp_cfg, full=False, last_success=state.last_success(ingest.PIPELINE)
    )

    # The trailing window is 14 days, but the gap is 40 — the pull must cover the gap.
    assert (end - start).days >= 40
    assert start <= now - pd.Timedelta(days=40)


def test_a_recent_run_does_not_widen_the_window(tmp_cfg, tmp_path):
    state = PipelineState(tmp_path / "state.json")
    state.record_success(ingest.PIPELINE, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1))

    start, end = ingest._window(
        tmp_cfg, full=False, last_success=state.last_success(ingest.PIPELINE)
    )
    plain_start, _ = ingest._window(tmp_cfg, full=False)

    assert start == plain_start


def test_a_successful_ingest_records_its_marker(tmp_cfg, tmp_path, monkeypatch):
    frame = synthetic.make_dataset(start="2024-01-01", end="2024-02-01")
    stub_sources(monkeypatch, frame)
    state = PipelineState(tmp_path / "state.json")

    assert state.last_success(ingest.PIPELINE) is None
    ingest.run(tmp_cfg, full=True, state=state)
    assert state.last_success(ingest.PIPELINE) is not None


def test_a_lost_marker_still_backfills_from_what_the_data_says(tmp_cfg, monkeypatch):
    """State and data are restored together, but they can come apart.

    Delete the state file and keep the parquet — a hand-edited runner, a partial
    restore, or a dataset ingested before the markers existed — and the plain trailing
    window would re-pull a fortnight over the top of a month-old hole and call it done.
    """
    stale = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=40)
    frame = synthetic.make_dataset(
        start=(stale - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        end=stale.strftime("%Y-%m-%d %H:00"),
    )
    write_dataset(frame, tmp_cfg.data.dataset_path)
    state = PipelineState(tmp_cfg.state.pipeline_state_path)
    assert state.last_success(ingest.PIPELINE) is None

    windows = []
    monkeypatch.setattr(ingest, "make_client", lambda: object())
    monkeypatch.setattr(ingest, "_weather", lambda *a, **k: frame[WEATHER_COLUMNS])

    def record_window(client, country, start, end, aggregation):
        windows.append((start, end))
        return frame[[LOAD_COLUMN, FORECAST_COLUMN]]

    monkeypatch.setattr(ingest, "fetch_load_frame", record_window)

    ingest.run(tmp_cfg, full=False, state=state)

    start, _ = windows[0]
    assert start <= frame["load_mw"].last_valid_index(), (
        "the gap since the data ends must be covered"
    )
    assert start > pd.Timestamp(tmp_cfg.data.start_date, tz="UTC"), "and it is not a full rebuild"


def test_a_marker_wins_over_the_data_when_both_exist(tmp_cfg):
    """The marker says when a run completed; the data only says what it stored."""
    frame = synthetic.make_dataset(start="2024-01-01", end="2024-02-01")
    write_dataset(frame, tmp_cfg.data.dataset_path)
    state = PipelineState(tmp_cfg.state.pipeline_state_path)
    state.record_success(ingest.PIPELINE, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1))

    resumed = ingest._resume_from(state, frame)

    assert resumed == state.last_success(ingest.PIPELINE)


def test_a_fresh_dataset_without_a_marker_does_not_widen_the_window(tmp_cfg):
    recent = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=1)
    frame = synthetic.make_dataset(
        start=(recent - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=recent.strftime("%Y-%m-%d %H:00"),
    )
    state = PipelineState(tmp_cfg.state.pipeline_state_path)

    start, _ = ingest._window(tmp_cfg, full=False, last_success=ingest._resume_from(state, frame))
    plain_start, _ = ingest._window(tmp_cfg, full=False)

    assert start == plain_start


def test_a_full_window_starts_from_the_configured_date(tmp_cfg):
    start, end = ingest._window(tmp_cfg, full=True)

    assert start == pd.Timestamp(tmp_cfg.data.start_date, tz=tmp_cfg.data.timezone_local)
    assert end > start


def test_an_incremental_window_is_the_configured_trailing_repull(tmp_cfg):
    start, end = ingest._window(tmp_cfg, full=False)

    assert (end - start).days >= tmp_cfg.ingestion.trailing_repull_days
    assert end > pd.Timestamp.now(tz="UTC")


def test_a_runner_with_no_dataset_rebuilds_the_full_history(tmp_cfg, monkeypatch):
    """A fresh Actions runner must not end up with a fortnight of data forever.

    An incremental pull onto nothing writes only the trailing window, and the next run
    re-pulls the same window — so without this the dataset never grows past 14 days and
    training can never happen.
    """
    frame = synthetic.make_dataset(start="2024-01-01", end="2024-02-01")
    windows = []
    monkeypatch.setattr(ingest, "make_client", lambda: object())
    monkeypatch.setattr(ingest, "_weather", lambda *a, **k: frame[WEATHER_COLUMNS])

    def record_window(client, country, start, end, aggregation):
        windows.append((start, end))
        return frame[[LOAD_COLUMN, FORECAST_COLUMN]]

    monkeypatch.setattr(ingest, "fetch_load_frame", record_window)

    assert not tmp_cfg.data.dataset_path.exists()
    ingest.run(tmp_cfg, full=False)

    start, end = windows[0]
    assert (end - start).days > 365, "a missing dataset must trigger a full rebuild"
    assert start == pd.Timestamp(tmp_cfg.data.start_date, tz=tmp_cfg.data.timezone_local)
