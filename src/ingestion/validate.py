"""Schema, range and continuity checks on the assembled dataset.

Runs on every ingestion. The output is a report a human can read — per-year row
counts, gap locations, where the TSO forecast is missing — because "the ingestion
raised" tells you nothing about what is wrong with five years of hourly data.

Raising is opt-in (`raise_for_status`): a pipeline decides whether a problem is fatal;
the validator's job is to find it and say so precisely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa
from loguru import logger

from src.config import ValidationConfig
from src.ingestion.dataset import FORECAST_COLUMN, LOAD_COLUMN, WEATHER_COLUMNS, hourly_index

MAX_EXAMPLES = 8


class DatasetValidationError(Exception):
    """Raised by `ValidationReport.raise_for_status` when the dataset has errors."""


@dataclass(frozen=True)
class Issue:
    name: str
    count: int
    severity: str = "error"
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    rows_per_year: dict[int, int]
    coverage: dict[str, int]
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def has(self, name: str) -> bool:
        return any(i.name == name for i in self.issues)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise DatasetValidationError(
                f"{len(self.errors)} validation error(s):\n{self.format()}"
            )

    def format(self) -> str:
        years = "  ".join(f"{y}: {n}" for y, n in sorted(self.rows_per_year.items()))
        coverage = "  ".join(f"{c}: {n}" for c, n in self.coverage.items())
        lines = [
            "Dataset validation report",
            f"  rows        {self.rows}   {self.start} -> {self.end}",
            f"  per year    {years or '-'}",
            f"  non-null    {coverage or '-'}",
        ]
        if not self.issues:
            lines.append("  issues      none")
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.name}: {issue.count}")
            lines.extend(f"      {e}" for e in issue.examples)
        lines.append(f"  status      {'OK' if self.ok else 'FAILED'}")
        return "\n".join(lines)


def _schema(cfg: ValidationConfig) -> pa.DataFrameSchema:
    """Column-level contract: dtypes, nullability and physically plausible ranges."""
    mw = pa.Check.in_range(cfg.load_min_mw, cfg.load_max_mw)
    return pa.DataFrameSchema(
        columns={
            LOAD_COLUMN: pa.Column(float, mw, nullable=True),
            FORECAST_COLUMN: pa.Column(float, mw, nullable=True),
            "temp_c": pa.Column(
                float, pa.Check.in_range(cfg.temp_min_c, cfg.temp_max_c), nullable=True
            ),
            "wind_ms": pa.Column(float, pa.Check.ge(0), nullable=True),
            "cloud_cover": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
            "humidity_pct": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
            "is_holiday": pa.Column(bool),
            "data_source_version": pa.Column(str, required=False),
        },
        index=pa.Index(nullable=False, unique=True),
        strict=False,
        coerce=False,
    )


def _runs(timestamps: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Group consecutive hourly timestamps into (first, last, hours) runs."""
    if len(timestamps) == 0:
        return []
    ts = timestamps.sort_values()
    breaks = ts.to_series().diff() != pd.Timedelta(hours=1)
    groups = breaks.cumsum()
    return [(g.index[0], g.index[-1], len(g)) for _, g in ts.to_series().groupby(groups)]


def _run_examples(runs: list[tuple[pd.Timestamp, pd.Timestamp, int]]) -> list[str]:
    shown = [f"{a} -> {b}  ({n} h)" for a, b, n in runs[:MAX_EXAMPLES]]
    if len(runs) > MAX_EXAMPLES:
        shown.append(f"... and {len(runs) - MAX_EXAMPLES} more")
    return shown


def _schema_issues(df: pd.DataFrame, cfg: ValidationConfig) -> list[Issue]:
    try:
        _schema(cfg).validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        issues = []
        for (column, check), group in cases.groupby(["column", "check"], dropna=False):
            examples = [
                f"{row.get('index', '-')}  {row['failure_case']}"
                for _, row in group.head(MAX_EXAMPLES).iterrows()
            ]
            if len(group) > MAX_EXAMPLES:
                examples.append(f"... and {len(group) - MAX_EXAMPLES} more")
            issues.append(
                Issue(name=f"schema: {column} failed {check}", count=len(group), examples=examples)
            )
        return issues
    return []


def validate_dataset(df: pd.DataFrame, cfg: ValidationConfig) -> ValidationReport:
    """Check the dataset and return a readable report. Never raises on data problems."""
    issues: list[Issue] = []

    if df.index.tz is None:
        raise ValueError("Dataset index must be timezone-aware UTC")

    duplicated = df.index.duplicated()
    if duplicated.any():
        issues.append(
            Issue(
                name="duplicate timestamps",
                count=int(duplicated.sum()),
                examples=[str(t) for t in df.index[duplicated][:MAX_EXAMPLES]],
            )
        )

    if not df.empty:
        expected = hourly_index(df.index.min(), df.index.max())
        missing = expected.difference(df.index)
        if len(missing):
            runs = _runs(missing)
            issues.append(
                Issue(name="missing hours", count=len(missing), examples=_run_examples(runs))
            )

    issues.extend(_schema_issues(df, cfg))

    load = df[LOAD_COLUMN]
    forecast = df[FORECAST_COLUMN]

    no_forecast = df.index[load.notna() & forecast.isna()]
    if len(no_forecast):
        issues.append(
            Issue(
                name="load present but TSO forecast missing",
                count=len(no_forecast),
                examples=_run_examples(_runs(no_forecast)),
            )
        )

    no_load = df.index[load.isna()]
    if len(no_load):
        issues.append(
            Issue(
                name="load missing",
                count=len(no_load),
                severity="warning",
                examples=_run_examples(_runs(no_load)),
            )
        )

    report = ValidationReport(
        rows=len(df),
        start=df.index.min() if not df.empty else None,
        end=df.index.max() if not df.empty else None,
        rows_per_year=df.index.year.value_counts().sort_index().to_dict() if not df.empty else {},
        coverage={
            c: int(df[c].notna().sum())
            for c in [LOAD_COLUMN, FORECAST_COLUMN, *WEATHER_COLUMNS]
            if c in df
        },
        issues=issues,
    )
    logger.log("ERROR" if not report.ok else "INFO", "\n{}", report.format())
    return report
