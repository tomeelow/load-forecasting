"""Shared fixtures. Nothing here touches the network, reads a secret, or writes to the repo.

The session-scoped frames are shared, so a test that mutates one must copy it first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src import config, synthetic
from src.config import Config, load_config

# Captured at import, before the autouse fixture below redirects it, so a test can still
# ask what the real path is without being able to read the file.
REAL_ENV_PATH = config.ENV_PATH


@pytest.fixture(scope="session")
def real_env_path() -> Path:
    return REAL_ENV_PATH


@pytest.fixture(scope="session")
def disposable_mlflow_store(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("mlflow-default") / "mlflow.db"


@pytest.fixture(autouse=True)
def mlflow_store_outside_the_repository(monkeypatch, disposable_mlflow_store):
    """Send MLflow's *default* store somewhere disposable.

    MLflow 3 defaults to `sqlite:///mlflow.db` in the current working directory, so any
    test that reaches MLflow without configuring it first — a monkeypatched `configure`,
    a bare `MlflowClient()` — silently creates a second store in the repository root.
    That is exactly how the orphan mlflow.db this project used to carry came to exist,
    and deleting it achieves nothing while the suite keeps making a new one.

    Per test rather than per session, because `mlflow.set_tracking_uri(None)` — which is
    how a test hands the global URI back — *unsets* this variable as a side effect. One
    teardown would otherwise leave every later test unprotected again.

    A test that sets its own tracking URI is unaffected: an explicit
    `mlflow.set_tracking_uri` takes precedence over the environment.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{disposable_mlflow_store}")


@pytest.fixture(autouse=True)
def offline_and_tokenless(monkeypatch, tmp_path):
    """Enforce the constraint rather than trusting it.

    The suite must pass on a machine with no network and no `.env`, so an accidental
    HTTP call or a real token in the environment fails loudly here instead of turning
    into a flaky test on someone else's laptop.

    Deleting the variable is not enough on its own. Application code loads `.env` when
    it starts — `create_app` does, `ingest.run` does — so the developer's real token
    walked straight back into `os.environ` a moment after this fixture removed it.
    Fourteen tests in `test_api.py` finished holding a live key. Nothing escaped,
    because the transport above is blocked, but the guarantee this fixture advertises
    was not true. Pointing the loader at a file that does not exist makes it true, and
    still runs the real loading code rather than stubbing it out.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("tests must not make network calls")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "absent.env")
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def dataset():
    """~21 months of synthetic hourly data, spanning both DST transitions."""
    return synthetic.make_dataset()


@pytest.fixture(scope="session")
def short_dataset():
    """Two months — enough history for a 168-hour lag, cheap enough to build often."""
    return synthetic.make_dataset(start="2024-01-01", end="2024-03-01")
