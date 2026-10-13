"""Rate expressions: constants, operators and population variables."""

import math

import pytest

from commol import Simulation
from commol.api.model_builder import ModelBuilder
from commol.constants import ModelTypes
from commol.context.model import Model


def _ab_model(rate: str) -> Model:
    return (
        ModelBuilder(name="AB")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=0.1)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate=rate)
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
            ],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


class TestConstants:
    @pytest.mark.parametrize(
        ("rate", "constant"),
        [
            ("pi * k * A", math.pi),
            ("e * k * A", math.e),
            ("k * pi", math.pi),
            ("pi / 10", math.pi),
        ],
    )
    def test_pi_and_e_are_constants_in_a_rate(self, rate: str, constant: float) -> None:
        result = Simulation(_ab_model(rate)).run(1)

        assert result["B"][1] == pytest.approx(constant * 0.1 * 1000)
