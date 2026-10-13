"""The model whose parameters the calibration tests recover."""

import pytest

from commol import Simulation
from commol.context.model import Model
from tests.support import build, sir_builder


@pytest.fixture
def truth() -> Model:
    """SIR with ``beta = 0.1`` and ``gamma = 0.05``, the values to recover."""
    return build(sir_builder(parameters={"beta": 0.1, "gamma": 0.05}))


@pytest.fixture
def truth_run(truth: Model) -> dict[str, list[float]]:
    """100 steps of ``truth``."""
    return Simulation(truth).run(100)
