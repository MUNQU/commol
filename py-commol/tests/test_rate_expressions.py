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


def _stratified_model(rate: str) -> Model:
    return (
        ModelBuilder(name="Stratified")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=0.1)
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_stratification(id="type", categories=["t1", "t2"])
        .add_transition(id="t_ab", source=["A"], target=["B"], rate=rate)
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
            ],
            stratification_fractions=[
                {
                    "stratification": "group",
                    "fractions": [
                        {"category": "g1", "fraction": 0.5},
                        {"category": "g2", "fraction": 0.5},
                    ],
                },
                {
                    "stratification": "type",
                    "fractions": [
                        {"category": "t1", "fraction": 0.5},
                        {"category": "t2", "fraction": 0.5},
                    ],
                },
            ],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


class TestSubpopulationVariables:
    def test_categories_out_of_declaration_order_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="N_t1_g1") as error:
            _stratified_model("k * N_t1_g1 / N")

        assert "N_g1_t1" in str(error.value)

    def test_categories_in_declaration_order_are_accepted(self) -> None:
        result = Simulation(_stratified_model("k * N_g1_t1 / N")).run(1)

        # 0.1 * 250 / 1000 moves from each A compartment.
        assert result["B_g1_t1"][1] == pytest.approx(0.025)


class TestOperators:
    def test_modulo_alone_is_an_operator(self) -> None:
        result = Simulation(_ab_model("step % 2")).run(2)

        # Per-capita rate 0 at step 0, 1 at step 1.
        assert result["A"][1] == pytest.approx(1000.0)
        assert result["A"][2] == pytest.approx(0.0)

    def test_modulo_with_a_parameter(self) -> None:
        result = Simulation(_ab_model("k * A % 7")).run(1)

        assert result["B"][1] == pytest.approx((0.1 * 1000) % 7)


class TestSubpopulationVariablesInFormulaParameters:
    def test_an_unknown_subpopulation_in_a_formula_parameter_is_rejected(
        self,
    ) -> None:
        builder = (
            ModelBuilder(name="Formula")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="share", value="N_t1_g1 / N")
            .add_stratification(id="group", categories=["g1", "g2"])
            .add_stratification(id="type", categories=["t1", "t2"])
            .add_transition(id="t_ab", source=["A"], target=["B"], rate="share")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": 1.0},
                    {"bin": "B", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "group",
                        "fractions": [
                            {"category": "g1", "fraction": 0.5},
                            {"category": "g2", "fraction": 0.5},
                        ],
                    },
                    {
                        "stratification": "type",
                        "fractions": [
                            {"category": "t1", "fraction": 0.5},
                            {"category": "t2", "fraction": 0.5},
                        ],
                    },
                ],
            )
        )

        with pytest.raises(ValueError, match="'N_t1_g1'.*Valid names: N_g1, "):
            builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

    def test_a_combination_a_conditional_stratification_never_creates(
        self,
    ) -> None:
        builder = (
            ModelBuilder(name="Conditional")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_stratification(id="group", categories=["g1", "g2"])
            .add_stratification(
                id="sub",
                categories=["s1", "s2"],
                conditions=[{"stratification": "group", "category": "g1"}],
            )
            .add_transition(id="t_ab", source=["A"], target=["B"], rate="k * N_g2_s1")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": 1.0},
                    {"bin": "B", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "group",
                        "fractions": [
                            {"category": "g1", "fraction": 0.5},
                            {"category": "g2", "fraction": 0.5},
                        ],
                    },
                    {
                        "stratification": "sub",
                        "fractions": [
                            {"category": "s1", "fraction": 0.5},
                            {"category": "s2", "fraction": 0.5},
                        ],
                    },
                ],
            )
        )

        with pytest.raises(ValueError, match="'N_g2_s1'"):
            builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
