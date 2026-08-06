"""Choosing what "normal" means — the decision that makes drift detection useful or noise.

Drift detection asks whether the current window looks like the reference window. On a
strongly seasonal target, that question is only meaningful if the reference is chosen
with the seasonality in mind.

**Why the obvious choice fails.** The default is a trailing reference: compare the last
two weeks against the two weeks before them. On Polish electricity load this reports
drift every March and every October, because the temperature genuinely changed and the
demand profile genuinely moved with it. Nothing is wrong with the model — the seasons
turned. A monitor that cries drift twice a year on schedule gets muted by whoever is
on call, and then it is not a monitor at all.

**What this module does instead.** The seasonal strategy compares the current window
against *the same calendar weeks in previous years*. September is judged against
Septembers. A cold snap that is normal for the season produces no alarm; a September
that behaves like no September before it does. The seasonal signal is removed by
construction rather than by tuning a threshold until the alarms stop.

The cost is honest and worth stating: it needs years of history, and in the first year
of operation there is none, so it falls back to a trailing window. During that year
seasonal false positives are expected, and the performance signal — rolling error of
served predictions, which has no seasonal confound — is the one to trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

SEASONAL = "seasonal"
TRAILING = "trailing"


@dataclass(frozen=True)
class ReferenceWindow:
    """The rows drift is measured against, and how they were chosen."""

    index: pd.DatetimeIndex
    strategy: str
    detail: str

    @property
    def rows(self) -> int:
        return len(self.index)


def trailing_reference(
    available: pd.DatetimeIndex, current_start: pd.Timestamp, days: int
) -> ReferenceWindow:
    """The period immediately before the current window."""
    start = current_start - pd.Timedelta(days=days)
    index = available[(available >= start) & (available < current_start)]
    return ReferenceWindow(
        index,
        TRAILING,
        f"{days} days immediately before {current_start:%Y-%m-%d}",
    )


def seasonal_reference(
    available: pd.DatetimeIndex,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
    *,
    years_back: int = 3,
    pad_days: int = 7,
) -> ReferenceWindow:
    """The same calendar window in each of the previous `years_back` years.

    Each historical window is widened by `pad_days` either side, both to gather enough
    rows and to avoid demanding that the calendar line up exactly — the same week of the
    year falls on different weekdays each year, and weekday structure matters here.
    """
    pad = pd.Timedelta(days=pad_days)
    pieces = []
    covered = []
    for years in range(1, years_back + 1):
        start = current_start - pd.DateOffset(years=years) - pad
        end = current_end - pd.DateOffset(years=years) + pad
        window = available[(available >= start) & (available <= end)]
        if len(window):
            pieces.append(window)
            covered.append(f"{start:%Y-%m-%d}..{end:%Y-%m-%d}")

    index = (
        pd.DatetimeIndex(sorted(set().union(*(set(p) for p in pieces))), tz="UTC")
        if pieces
        else pd.DatetimeIndex([], tz="UTC")
    )
    return ReferenceWindow(
        index,
        SEASONAL,
        f"same weeks in {len(pieces)} previous year(s): {', '.join(covered) or 'none available'}",
    )


def choose_reference(
    available: pd.DatetimeIndex,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
    *,
    strategy: str,
    years_back: int,
    pad_days: int,
    min_rows: int,
    fallback_to_trailing: bool,
    current_days: int,
) -> ReferenceWindow:
    """Pick the reference window, falling back when the preferred one is too thin.

    A fallback is reported in the result rather than applied quietly: a seasonal
    comparison and a trailing one answer different questions, and a reader of the report
    needs to know which one they are looking at.
    """
    if strategy == SEASONAL:
        window = seasonal_reference(
            available, current_start, current_end, years_back=years_back, pad_days=pad_days
        )
        if window.rows >= min_rows:
            logger.info("Reference: {} ({} rows)", window.detail, window.rows)
            return window
        if not fallback_to_trailing:
            return window
        logger.warning(
            "Seasonal reference has only {} rows (need {}); falling back to a trailing "
            "window. Expect seasonal false positives until a year of history exists.",
            window.rows,
            min_rows,
        )

    window = trailing_reference(available, current_start, current_days)
    logger.info("Reference: {} ({} rows)", window.detail, window.rows)
    return window
