"""Shared test setup."""

import pytest
from langsmith import tracing_context


@pytest.fixture(autouse=True)
def no_tracing():
    """Keep tests offline even when tracing is turned on in .env."""
    with tracing_context(enabled=False):
        yield
