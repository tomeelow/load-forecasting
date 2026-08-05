"""Shared fixtures. Nothing here touches the network or reads a secret.

The session-scoped frames are shared, so a test that mutates one must copy it first.
"""

from __future__ import annotations

import pytest
import requests

from src import synthetic
from src.config import Config, load_config


@pytest.fixture(autouse=True)
def offline_and_tokenless(monkeypatch):
    """Enforce the constraint rather than trusting it.

    The suite must pass on a machine with no network and no `.env`, so an accidental
    HTTP call or a real token in the environment fails loudly here instead of turning
    into a flaky test on someone else's laptop.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("tests must not make network calls")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
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
