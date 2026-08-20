"""Where the ENTSO-E token comes from, and where it must never come from.

One variable, `ENTSOE_API_KEY`, read in exactly one place
(`src/ingestion/entsoe_client.make_client`). What this file pins down is everything
around that read: which file supplies it, what beats what, and the promise the suite
makes that it is never the developer's own.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from src import config

# Deliberately not the real variable: these tests are about the loading mechanism, and
# a test that writes a plausible key name into the process environment is a test that
# can confuse the next one.
TOKEN = "PL_LOAD_FORECASTING_TEST_ONLY"


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """A .env of our own, in place of the repository's."""
    path = tmp_path / "project.env"
    monkeypatch.setattr(config, "ENV_PATH", path)
    return path


def forget_after_the_test(monkeypatch) -> None:
    """Make monkeypatch clean up a variable the *code under test* will set.

    setenv records that the variable was absent; delenv then removes the placeholder.
    Teardown unwinds both, so whatever `load_project_env` puts there does not survive
    into the next test.
    """
    monkeypatch.setenv(TOKEN, "placeholder")
    monkeypatch.delenv(TOKEN)


def test_the_env_file_is_read_from_the_configured_path(env_file, monkeypatch):
    forget_after_the_test(monkeypatch)
    env_file.write_text(f"{TOKEN}=from-the-file\n")

    assert config.load_project_env() is True
    assert os.environ[TOKEN] == "from-the-file"


def test_an_exported_variable_wins_over_the_file(env_file, monkeypatch):
    """override=False, on purpose: a CI secret must beat a file left on a disk."""
    monkeypatch.setenv(TOKEN, "from-the-shell")
    env_file.write_text(f"{TOKEN}=from-the-file\n")

    config.load_project_env()

    assert os.environ[TOKEN] == "from-the-shell"


def test_a_missing_env_file_is_not_an_error(env_file):
    """CI has no .env at all — the workflow injects the secret directly."""
    assert not env_file.exists()

    assert config.load_project_env() is False


def test_the_path_is_the_repository_root_not_the_working_directory(
    real_env_path, monkeypatch, tmp_path
):
    """`python-dotenv`'s own discovery walks up from the calling module; this does not."""
    monkeypatch.chdir(tmp_path)

    assert real_env_path.is_absolute()
    assert real_env_path.parent == config.REPO_ROOT


def test_starting_the_service_does_not_import_the_developers_token(cfg, tmp_path):
    """The leak this closes: `create_app` loads .env, seconds after the fixture cleared it.

    Fourteen tests used to finish holding a live ENTSO-E key. Nothing escaped — the
    transport is blocked — but the suite advertised a guarantee it was not keeping.
    """
    from src.api.main import create_app

    isolated = dataclasses.replace(cfg, state=dataclasses.replace(cfg.state, dir=tmp_path))

    create_app(isolated, load_model_on_startup=False)

    assert "ENTSOE_API_KEY" not in os.environ


def test_the_real_env_file_is_out_of_reach_for_every_test(real_env_path):
    """The autouse fixture redirects the path; without that the test above is luck."""
    assert real_env_path != config.ENV_PATH
    assert not config.ENV_PATH.exists()
