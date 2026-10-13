"""Fixtures shared by every test package."""

import pytest

from commol import Simulation
from commol.context.model import Model
from tests.support import build, sir_builder


@pytest.fixture
def sir_model() -> Model:
    """The default SIR model of ``sir_builder``: beta 0.3, gamma 0.1."""
    return build(sir_builder())


@pytest.fixture
def sir_simulation(sir_model: Model) -> Simulation:
    """A simulation of ``sir_model``."""
    return Simulation(sir_model)
