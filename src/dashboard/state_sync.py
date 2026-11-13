"""Mirror the pipeline's state into a deployment that has no pipeline of its own.

The hosted dashboard is a free container with no scheduler, no ENTSO-E token and no
MLflow server. Everything it needs already exists though: the daily loop pushes the
dataset, the MLflow store and the prediction log to the `pipeline-state` branch as a
single snapshot (ADR-008), and that branch is public. So the deployment reads the same
artifacts the loop writes, one clone behind, instead of showing empty panels or — worse —
a demo dataset baked into the image at build time and quietly ageing.

**What this is not.** It is not a substitute for the API. The hosted page shows what the
system recorded; it does not serve fresh forecasts. It does carry the champion's booster,
for the what-if panel, which re-runs it on weather the loop already ingested and logs
nothing. The page says so rather than implying otherwise.

Enabled by setting `STATE_REPO_URL`. Unset — which is every local run — this module does
nothing at all, and the dashboard reads the working tree as usual.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

STATE_BRANCH = "pipeline-state"
MIRRORED = ("state", "mlruns", "data/processed")
CLONE_TIMEOUT_S = 120

# What the dashboard actually opens, and nothing else. The branch carries the loop's whole
# working state, which is dominated by the LightGBM boosters under `mlruns/1/models` — a
# few hundred megabytes of them, because a 2000-round model is tens of megabytes and the
# registry keeps every run the pruner has not yet reached. The dashboard opens exactly one
# of them, the champion's, for the what-if panel; everything else it shows comes from the
# registry *database* and a sub-kilobyte importance JSON. Cloning every booster to render
# a page that ignores all but one cost the whole mirror, because the transfer could not
# finish inside CLONE_TIMEOUT_S. So the first checkout takes none, and the champion's is
# added once the database it arrived with says which one that is.
#
# `--filter=blob:none` skips the blob contents at transfer time and `sparse-checkout`
# decides which ones are worth fetching back. A server that refuses the filter — the
# default for a bare local remote, though not for GitHub — degrades to sending everything
# and checking out the same narrow set, so the mirror still works, just slower.
SPARSE_PATTERNS = (
    "/state/**",
    "/data/processed/**",
    "/mlruns/mlflow.db",
    "/mlruns/**/feature_importance.json",
)


@dataclass(frozen=True)
class MirrorResult:
    """What the deployment managed to fetch, and when."""

    enabled: bool
    ok: bool = False
    fetched_at: datetime | None = None
    commit: str | None = None
    committed_at: datetime | None = None
    paths: tuple[str, ...] = ()
    error: str | None = None

    def describe(self) -> str:
        """One line for the page. Never claims freshness it cannot demonstrate."""
        if not self.enabled:
            return "Reading the local working tree."
        if not self.ok:
            return f"Could not mirror pipeline state ({self.error}). Panels below may be empty."
        age = ""
        if self.committed_at is not None:
            hours = (datetime.now(UTC) - self.committed_at).total_seconds() / 3600
            age = f", snapshot {self.committed_at:%Y-%m-%d %H:%M} UTC ({hours:.0f}h old)"
        return f"Mirrored from `{STATE_BRANCH}`{age}."


def _run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell, URL from our own env
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=CLONE_TIMEOUT_S,
    ).stdout.strip()


def _aliased_boosters(database: Path) -> list[str]:
    """Sparse patterns for the boosters a registry alias points at — the champion's.

    Read straight from the registry database rather than through MLflow, which would be
    a heavy import to answer one join. An unreadable database yields nothing, and the
    what-if panel then says the champion is not on disk.
    """
    try:
        with closing(sqlite3.connect(database)) as connection:
            sources = connection.execute(
                """SELECT v.source FROM registered_model_aliases a
                   JOIN model_versions v ON v.name = a.name AND v.version = a.version"""
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Could not read aliases from {}: {}", database, exc)
        return []
    return [
        f"/mlruns/*/models/{source.removeprefix('models:/')}/artifacts/model.lgb"
        for (source,) in sources
        if source
    ]


def mirror_state(destination: Path, repo_url: str | None = None) -> MirrorResult:
    """Shallow-clone the state branch and copy its paths over `destination`.

    A failure is reported, never raised. A dashboard that cannot reach GitHub should say
    so at the top of the page and render whatever it already has, rather than returning a
    stack trace to whoever opened the link.
    """
    repo_url = repo_url or os.environ.get("STATE_REPO_URL")
    if not repo_url:
        return MirrorResult(enabled=False)

    with tempfile.TemporaryDirectory() as workspace:
        clone = Path(workspace) / "state"
        try:
            _run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    STATE_BRANCH,
                    "--single-branch",
                    "--filter=blob:none",
                    "--sparse",
                    "--no-checkout",
                    repo_url,
                    str(clone),
                ]
            )
            _run(["git", "-C", str(clone), "sparse-checkout", "set", "--no-cone", *SPARSE_PATTERNS])
            _run(["git", "-C", str(clone), "checkout"])
            commit = _run(["git", "-C", str(clone), "rev-parse", "--short", "HEAD"])
            committed = _run(["git", "-C", str(clone), "log", "-1", "--format=%cI"])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            logger.warning("Could not mirror {}: {}", repo_url, detail)
            return MirrorResult(enabled=True, ok=False, error=str(detail).strip()[:200])

        # Its own step, and allowed to fail: one panel going without its model is not a
        # reason to throw away a mirror that everything else on the page can use.
        boosters = _aliased_boosters(clone / "mlruns" / "mlflow.db")
        if boosters:
            try:
                _run(["git", "-C", str(clone), "sparse-checkout", "add", *boosters])
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                logger.warning("Could not fetch the champion's booster: {}", exc)

        copied = []
        for relative in MIRRORED:
            source = clone / relative
            if not source.exists():
                logger.warning("State branch has no {}", relative)
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(source, target)
            copied.append(relative)

    logger.info("Mirrored {} from {}@{}", ", ".join(copied), STATE_BRANCH, commit)
    return MirrorResult(
        enabled=True,
        ok=bool(copied),
        fetched_at=datetime.now(UTC),
        commit=commit,
        committed_at=datetime.fromisoformat(committed).astimezone(UTC),
        paths=tuple(copied),
        error=None if copied else "the branch carried none of the expected paths",
    )
