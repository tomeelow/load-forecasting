"""Validation: feed it deliberately broken data and check it says what is broken."""

from __future__ import annotations

import pandas as pd
import pytest

from src import synthetic
from src.ingestion.validate import DatasetValidationError, validate_dataset


@pytest.fixture
def clean():
    return synthetic.make_dataset(start="2024-01-01", end="2024-02-15")


def names(report) -> list[str]:
    return [issue.name for issue in report.issues]


def test_a_clean_dataset_passes(clean, cfg):
    report = validate_dataset(clean, cfg.validation)

    assert report.ok
    assert report.issues == []
    assert report.rows == len(clean)


def test_report_lists_rows_per_year_and_coverage(clean, cfg):
    report = validate_dataset(clean, cfg.validation)

    assert report.rows_per_year == {2024: len(clean)}
    assert report.coverage["load_mw"] == len(clean)
    text = report.format()
    assert "per year" in text and "2024" in text
    assert "status      OK" in text


def test_a_timestamp_gap_is_flagged_with_its_location(clean, cfg):
    gapped = clean.drop(clean.index[100:103])

    report = validate_dataset(gapped, cfg.validation)

    assert not report.ok
    assert report.has("missing hours")
    issue = next(i for i in report.issues if i.name == "missing hours")
    assert issue.count == 3
    assert str(clean.index[100]) in issue.examples[0]
    assert "(3 h)" in issue.examples[0]


def test_an_implausible_load_value_is_flagged(clean, cfg):
    broken = clean.copy()
    broken.iloc[10, broken.columns.get_loc("load_mw")] = 91_000.0  # a whole grid too much

    report = validate_dataset(broken, cfg.validation)

    assert not report.ok
    assert any("load_mw failed in_range" in name for name in names(report))


def test_a_missing_tso_forecast_hour_is_flagged(clean, cfg):
    broken = clean.copy()
    broken.iloc[20:26, broken.columns.get_loc("tso_forecast_mw")] = None

    report = validate_dataset(broken, cfg.validation)

    assert not report.ok
    assert report.has("load present but TSO forecast missing")
    issue = next(i for i in report.issues if i.name == "load present but TSO forecast missing")
    assert issue.count == 6


def test_missing_load_is_a_warning_not_an_error(clean, cfg):
    # Trailing hours with no actuals yet are normal, not a fault.
    partial = clean.copy()
    partial.iloc[-5:, partial.columns.get_loc("load_mw")] = None
    partial.iloc[-5:, partial.columns.get_loc("tso_forecast_mw")] = 20_000.0

    report = validate_dataset(partial, cfg.validation)

    assert report.ok
    assert report.has("load missing")
    assert [i.severity for i in report.issues] == ["warning"]


def test_duplicate_timestamps_are_flagged(clean, cfg):
    duplicated = pd.concat([clean, clean.iloc[:3]]).sort_index()

    report = validate_dataset(duplicated, cfg.validation)

    assert not report.ok
    assert report.has("duplicate timestamps")


def test_an_out_of_range_temperature_is_flagged(clean, cfg):
    broken = clean.copy()
    broken.iloc[0, broken.columns.get_loc("temp_c")] = 300.0

    report = validate_dataset(broken, cfg.validation)

    assert not report.ok
    assert any("temp_c failed" in name for name in names(report))


def test_several_faults_are_reported_together_not_one_at_a_time(clean, cfg):
    broken = clean.drop(clean.index[50:52]).copy()
    broken.iloc[10, broken.columns.get_loc("load_mw")] = 91_000.0
    broken.iloc[20, broken.columns.get_loc("tso_forecast_mw")] = None

    report = validate_dataset(broken, cfg.validation)

    assert len(report.errors) >= 3
    assert report.has("missing hours")
    assert report.has("load present but TSO forecast missing")


def test_raise_for_status_raises_only_on_errors(clean, cfg):
    validate_dataset(clean, cfg.validation).raise_for_status()

    broken = clean.drop(clean.index[10:12])
    with pytest.raises(DatasetValidationError, match="missing hours"):
        validate_dataset(broken, cfg.validation).raise_for_status()


def test_a_naive_index_is_rejected_outright(clean, cfg):
    naive = clean.copy()
    naive.index = naive.index.tz_localize(None)

    with pytest.raises(ValueError, match="timezone-aware"):
        validate_dataset(naive, cfg.validation)
