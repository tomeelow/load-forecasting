"""Shared fixtures. Nothing here touches the network or reads a secret.

The session-scoped frames are shared, so a test that mutates one must copy it first.
"""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from tests.fixtures import synthetic


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
