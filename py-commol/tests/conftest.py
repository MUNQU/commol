"""Fixtures shared by every test package."""

import os
import sys
from collections.abc import Iterator

import pytest

from commol import Simulation
from commol.context.model import Model
from tests.support import abc_builder, build

pytest_plugins = ["tests.suite.durations"]

# Figures draw on a non-interactive backend; matplotlib reads it when a test
# first draws a figure, so no test that draws none imports it.
os.environ["MPLBACKEND"] = "Agg"


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close the figures a test opens. A test that drew no figure loaded no
    ``matplotlib.pyplot``, and this imports none."""
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


@pytest.fixture
def abc_model() -> Model:
    """The default model of ``abc_builder``: k1 0.3, k2 0.1."""
    return build(abc_builder())


@pytest.fixture
def abc_simulation(abc_model: Model) -> Simulation:
    """A simulation of ``abc_model``."""
    return Simulation(abc_model)
