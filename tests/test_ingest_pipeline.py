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
from src.ingestion.dataset import read_dataset
from src.ingestion.entsoe_client import FORECAST_COLUMN, LOAD_COLUMN

WEATHER_COLUMNS = ["temp_c", "wind_ms", "cloud_cover", "humidity_pct"]


@pytest.fixture
def tmp_cfg(cfg, tmp_path):
    """The real config, pointed at a temporary data directory."""
    data = dataclasses.replace(cfg.data, processed_dir=tmp_path, raw_dir=tmp_path)
    return dataclasses.replace(cfg, data=data)


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


def test_a_full_window_starts_from_the_configured_date(tmp_cfg):
    start, end = ingest._window(tmp_cfg, full=True)

    assert start == pd.Timestamp(tmp_cfg.data.start_date, tz=tmp_cfg.data.timezone_local)
    assert end > start


def test_an_incremental_window_is_the_configured_trailing_repull(tmp_cfg):
    start, end = ingest._window(tmp_cfg, full=False)

    assert (end - start).days >= tmp_cfg.ingestion.trailing_repull_days
    assert end > pd.Timestamp.now(tz="UTC")
