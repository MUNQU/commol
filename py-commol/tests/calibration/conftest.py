"""The model whose parameters the calibration tests recover."""

import pytest

from commol import Simulation
from commol._model.model import Model
from tests.support import abc_builder, build


@pytest.fixture
def truth() -> Model:
    """``abc_builder`` with ``k1 = 0.1`` and ``k2 = 0.05``, the values to recover."""
    return build(abc_builder(parameters={"k1": 0.1, "k2": 0.05}))


@pytest.fixture
def truth_run(truth: Model) -> dict[str, list[float]]:
    """100 steps of ``truth``."""
    return Simulation(truth).run(100)
