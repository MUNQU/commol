"""Plots draw on a non-interactive backend, and every figure is closed."""

from collections.abc import Iterator

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close the figures a test opens."""
    yield
    plt.close("all")
