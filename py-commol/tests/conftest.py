"""Fixtures shared by every test package."""

import pytest

from commol import Simulation
from commol.context.model import Model
from tests.support import abc_builder, build


@pytest.fixture
def abc_model() -> Model:
    """The default model of ``abc_builder``: k1 0.3, k2 0.1."""
    return build(abc_builder())


@pytest.fixture
def abc_simulation(abc_model: Model) -> Simulation:
    """A simulation of ``abc_model``."""
    return Simulation(abc_model)
