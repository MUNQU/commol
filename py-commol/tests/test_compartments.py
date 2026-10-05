"""Parity between the Python compartment-name generator and the Rust engine."""

import pytest

from commol.api.model_builder import ModelBuilder
from commol.api.simulation import Simulation
from commol.constants import ModelTypes, PrintEquationsOutputFormat
from commol.context._model.helpers import ModelCompartmentHelper
from commol.context.compartments import generate_compartment_names
from commol.context.stratification import Stratification
from commol.context.stratification_condition import StratificationCondition


def _engine_compartments(model) -> list[str]:
    return Simulation(model).engine.compartments


def _base_builder() -> ModelBuilder:
    return (
        ModelBuilder(name="Compartments", version="1.0")
        .add_bin(id="S", name="S")
        .add_bin(id="I", name="I")
        .add_parameter(id="beta", value=0.1)
        .add_transition(
            id="flow",
            source=["S"],
            target=["I"],
            rate="beta * S",
            per_compartment=True,
        )
    )


class TestGeneratorParity:
    def test_without_stratifications(self) -> None:
        builder = _base_builder().set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "S", "fraction": 0.9},
                {"bin": "I", "fraction": 0.1},
            ],
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        assert generate_compartment_names(
            model.population.bins, model.population.stratifications
        ) == _engine_compartments(model)

    def test_with_plain_stratifications(self) -> None:
        builder = (
            _base_builder()
            .add_stratification(id="age", categories=["young", "old"])
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.9},
                    {"bin": "I", "fraction": 0.1},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    }
                ],
            )
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        assert generate_compartment_names(
            model.population.bins, model.population.stratifications
        ) == _engine_compartments(model)

    def test_with_conditional_stratification(self) -> None:
        builder = (
            _base_builder()
            .add_stratification(id="age", categories=["young", "old"])
            .add_stratification(
                id="care",
                categories=["home", "hosp"],
                conditions=[{"stratification": "age", "category": "old"}],
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.9},
                    {"bin": "I", "fraction": 0.1},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    },
                    {
                        "stratification": "care",
                        "fractions": [
                            {"category": "home", "fraction": 0.5},
                            {"category": "hosp", "fraction": 0.5},
                        ],
                    },
                ],
            )
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        names = generate_compartment_names(
            model.population.bins, model.population.stratifications
        )
        assert names == _engine_compartments(model)
        assert "S_old_hosp" in names
        assert "S_young_hosp" not in names


class TestCategoryLessConditionParity:
    def test_category_less_condition_expands_every_compartment(self) -> None:
        model = (
            _base_builder()
            .add_stratification(id="age", categories=["young", "old"])
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.9},
                    {"bin": "I", "fraction": 0.1},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    }
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )
        # A category-less condition is permitted by the engine but rejected by
        # the Pydantic validator, so the stratification is built without it.
        model.population.stratifications.append(
            Stratification.model_construct(
                id="care",
                categories=["home", "hosp"],
                description=None,
                conditions=[
                    StratificationCondition.model_construct(
                        stratification="age", category=None, to=None
                    )
                ],
            )
        )

        names = generate_compartment_names(
            model.population.bins, model.population.stratifications
        )
        helper = ModelCompartmentHelper(model)
        helper_names = [
            helper.compartment_to_string(compartment, PrintEquationsOutputFormat.TEXT)
            for compartment in helper.generate_compartments()
        ]

        assert names == _engine_compartments(model)
        assert helper_names == names
        assert "S_young_home" in names
        assert "S_old_hosp" in names


class TestFirstStratificationEntryParity:
    def test_builder_expansion_uses_the_first_entry_like_the_engine(self) -> None:
        bin_fractions = [
            {"bin": "S", "fraction": 1.0},
            {"bin": "I", "fraction": 0.0},
        ]
        stratification_fractions = [
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.25},
                    {"category": "g2", "fraction": 0.75},
                ],
            },
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.9},
                    {"category": "g2", "fraction": 0.1},
                ],
            },
        ]
        builder = _base_builder().add_stratification(
            id="group", categories=["g1", "g2"]
        )
        builder.set_initial_conditions(
            population_size=1000,
            bin_fractions=bin_fractions,
            stratification_fractions=stratification_fractions,
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        builder_fractions = builder._uniform_compartment_fractions(
            bin_fractions, stratification_fractions
        )
        simulation = Simulation(model)
        initial = dict(
            zip(simulation.engine.compartments, simulation.engine.population)
        )
        expected = {
            name: fraction * 1000 for name, fraction in builder_fractions.items()
        }

        assert initial == pytest.approx(expected)
        assert initial["S_g1"] == pytest.approx(250.0)
