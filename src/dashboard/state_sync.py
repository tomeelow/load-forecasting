"""Mirror the pipeline's state into a deployment that has no pipeline of its own.

The hosted dashboard is a free container with no scheduler, no ENTSO-E token and no
MLflow server. Everything it needs already exists though: the daily loop pushes the
dataset, the MLflow store and the prediction log to the `pipeline-state` branch as a
single snapshot (ADR-008), and that branch is public. So the deployment reads the same
artifacts the loop writes, one clone behind, instead of showing empty panels or — worse —
a demo dataset baked into the image at build time and quietly ageing.

**What this is not.** It is not a substitute for the API. The hosted page shows what the
system recorded; it does not serve fresh forecasts, because serving needs a live weather
call and a model in memory. The page says so rather than implying otherwise.

Enabled by setting `STATE_REPO_URL`. Unset — which is every local run — this module does
nothing at all, and the dashboard reads the working tree as usual.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

STATE_BRANCH = "pipeline-state"
MIRRORED = ("state", "mlruns", "data/processed")
CLONE_TIMEOUT_S = 120


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
                    repo_url,
                    str(clone),
                ]
            )
            commit = _run(["git", "-C", str(clone), "rev-parse", "--short", "HEAD"])
            committed = _run(["git", "-C", str(clone), "log", "-1", "--format=%cI"])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            logger.warning("Could not mirror {}: {}", repo_url, detail)
            return MirrorResult(enabled=True, ok=False, error=str(detail).strip()[:200])

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
