"""Parity between the Python compartment-name generator and the Rust engine."""

from commol.api.model_builder import ModelBuilder
from commol.api.simulation import Simulation
from commol.constants import ModelTypes
from commol.context.compartments import generate_compartment_names


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
