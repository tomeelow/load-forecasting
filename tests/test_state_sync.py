"""Mirroring pipeline state into a deployment that has no pipeline.

The hosted dashboard's data comes from here, so the failure that matters is the silent
one: a clone that did not happen, or happened and copied nothing, while the page carries
on describing numbers as though they were current. Every test below is about the page
being able to tell the truth about where its data came from.

No network. The remote is a local bare repository, which exercises the real `git clone`
rather than a stub of it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.dashboard import state_sync
from src.dashboard.state_sync import STATE_BRANCH, mirror_state


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def state_remote(tmp_path) -> str:
    """A bare repository carrying a `pipeline-state` branch, like the real one."""
    work = tmp_path / "work"
    work.mkdir()
    git("init", "-q", "-b", STATE_BRANCH, cwd=work)
    git("config", "user.email", "test@example.com", cwd=work)
    git("config", "user.name", "test", cwd=work)

    for relative, content in (
        ("state/predictions.db", "not really sqlite"),
        ("state/drift_history.csv", "checked_at,status\n"),
        ("mlruns/mlflow.db", "not really sqlite either"),
        ("data/processed/dataset.parquet", "not really parquet"),
        ("mlruns/1/abc/artifacts/feature_importance.json", '{"columns": [], "data": []}'),
        # The real branch is ~99% these, at tens of megabytes each. Nothing on the page
        # opens one, and fetching them is what used to exhaust the clone timeout.
        ("mlruns/1/models/m-abc/artifacts/model.lgb", "tree\n" * 10_000),
    ):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "state snapshot", cwd=work)

    bare = tmp_path / "remote.git"
    git("clone", "-q", "--bare", str(work), str(bare))
    return str(bare)


def test_it_does_nothing_at_all_when_no_repository_is_configured(tmp_path, monkeypatch):
    """Every local run takes this path; it must not touch the working tree."""
    monkeypatch.delenv("STATE_REPO_URL", raising=False)

    result = mirror_state(tmp_path)

    assert not result.enabled
    assert not any(tmp_path.iterdir())
    assert result.describe() == "Reading the local working tree."


def test_it_copies_every_path_the_loop_persists(tmp_path, state_remote):
    destination = tmp_path / "app"

    result = mirror_state(destination, state_remote)

    assert result.ok
    assert set(result.paths) == {"state", "mlruns", "data/processed"}
    assert (destination / "state" / "predictions.db").exists()
    assert (destination / "mlruns" / "mlflow.db").exists()
    assert (destination / "data" / "processed" / "dataset.parquet").exists()


def test_it_leaves_the_boosters_on_the_branch(tmp_path, state_remote):
    """The mirror fetches what the page reads, not the models it never loads.

    The registry database and the importance JSON have to arrive — the model card and
    the importance chart are read from them. `model.lgb` must not, because it is the
    entire reason a 355MB branch could not be cloned inside the timeout.
    """
    destination = tmp_path / "app"

    result = mirror_state(destination, state_remote)

    assert result.ok
    assert (destination / "mlruns" / "mlflow.db").exists()
    assert (destination / "mlruns" / "1" / "abc" / "artifacts" / "feature_importance.json").exists()
    assert not list(destination.glob("mlruns/**/*.lgb"))


def test_it_reports_which_snapshot_it_got(tmp_path, state_remote):
    """The page prints this, so a stale mirror is visible rather than assumed fresh."""
    result = mirror_state(tmp_path / "app", state_remote)

    assert result.commit
    assert result.committed_at is not None
    assert STATE_BRANCH in result.describe()
    assert "old)" in result.describe()


def test_a_second_mirror_replaces_rather_than_merges(tmp_path, state_remote):
    """A file deleted upstream must not survive in the deployment forever."""
    destination = tmp_path / "app"
    mirror_state(destination, state_remote)
    (destination / "state" / "stale.txt").write_text("left over from a previous snapshot")

    mirror_state(destination, state_remote)

    assert not (destination / "state" / "stale.txt").exists()


def test_an_unreachable_remote_is_reported_not_raised(tmp_path):
    """Whoever opened the link gets a page that explains itself, not a stack trace."""
    result = mirror_state(tmp_path / "app", str(tmp_path / "does-not-exist.git"))

    assert result.enabled
    assert not result.ok
    assert result.error
    assert "Could not mirror" in result.describe()


def test_a_branch_without_the_expected_paths_is_not_reported_as_success(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    git("init", "-q", "-b", STATE_BRANCH, cwd=work)
    git("config", "user.email", "test@example.com", cwd=work)
    git("config", "user.name", "test", cwd=work)
    (work / "unrelated.txt").write_text("nothing the dashboard reads")
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "empty state", cwd=work)

    result = mirror_state(tmp_path / "app", str(work))

    assert not result.ok
    assert "none of the expected paths" in result.error


def test_a_hanging_clone_gives_up_rather_than_hanging_the_page(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git clone", timeout=state_sync.CLONE_TIMEOUT_S)

    monkeypatch.setattr(state_sync.subprocess, "run", timeout)

    result = mirror_state(tmp_path / "app", "https://example.invalid/repo.git")

    assert not result.ok
    assert result.error


def test_the_environment_variable_is_what_switches_it_on(tmp_path, state_remote, monkeypatch):
    monkeypatch.setenv("STATE_REPO_URL", state_remote)

    assert mirror_state(tmp_path / "app").ok
