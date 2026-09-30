"""Per-compartment initial conditions: types, builder validation, integration."""

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.api.model_builder import ModelBuilder
from commol.constants import ModelTypes
from commol.context import CompartmentFraction
from commol.context.initial_conditions import BinFraction, InitialConditions
from commol.context.model import Model


def _stratified_builder() -> ModelBuilder:
    return (
        ModelBuilder(name="PerCompartment", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=0.1)
        .add_transition(
            id="flow",
            source=["A"],
            target=["B"],
            rate="k * A",
            per_compartment=True,
        )
        .add_stratification(id="group", categories=["g1", "g2"])
    )


def _set_initial_conditions(builder: ModelBuilder, overrides):
    return builder.set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 1.0},
            {"bin": "B", "fraction": 0.0},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 1.0},
                    {"category": "g2", "fraction": 0.0},
                ],
            }
        ],
        compartment_fractions=overrides,
    )


class TestCompartmentFractionType:
    def test_is_exported_from_context(self) -> None:
        assert CompartmentFraction.__name__ == "CompartmentFraction"

    def test_rejects_fraction_above_one(self) -> None:
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            CompartmentFraction(compartment="A_g1", fraction=1.5)

    def test_rejects_negative_fraction(self) -> None:
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            CompartmentFraction(compartment="A_g1", fraction=-0.1)

    def test_rejects_duplicate_compartment_names(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate compartment fractions"):
            InitialConditions(
                population_size=1000,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
                compartment_fractions=[
                    CompartmentFraction(compartment="A_g1", fraction=0.1),
                    CompartmentFraction(compartment="A_g1", fraction=0.2),
                ],
            )

    def test_defaults_to_empty_list(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[BinFraction(bin="A", fraction=1.0)],
        )
        assert conditions.compartment_fractions == []


class TestBuilderCompartmentFractions:
    def test_valid_override_is_stored(self) -> None:
        builder = _set_initial_conditions(
            _stratified_builder(), [{"compartment": "A_g2", "fraction": 0.0}]
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        overrides = model.population.initial_conditions.compartment_fractions
        assert [(cf.compartment, cf.fraction) for cf in overrides] == [("A_g2", 0.0)]

    def test_unknown_compartment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not part of the model expansion"):
            _set_initial_conditions(
                _stratified_builder(), [{"compartment": "A_missing", "fraction": 0.0}]
            )

    def test_duplicate_compartment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate compartment fraction"):
            _set_initial_conditions(
                _stratified_builder(),
                [
                    {"compartment": "A_g2", "fraction": 0.0},
                    {"compartment": "A_g2", "fraction": 0.1},
                ],
            )

    def test_out_of_range_fraction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            _set_initial_conditions(
                _stratified_builder(), [{"compartment": "A_g2", "fraction": 1.5}]
            )

    def test_conservation_is_enforced(self) -> None:
        with pytest.raises(ValueError, match="exceed population_size"):
            _set_initial_conditions(
                _stratified_builder(), [{"compartment": "A_g2", "fraction": 0.5}]
            )

    def test_empty_overrides_reproduce_previous_behaviour(self) -> None:
        model = _set_initial_conditions(_stratified_builder(), []).build(
            typology=ModelTypes.DIFFERENCE_EQUATIONS.value
        )
        assert model.population.initial_conditions.compartment_fractions == []

    @staticmethod
    def _uncalibrated_builder() -> ModelBuilder:
        return (
            ModelBuilder(name="Uncalibrated", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="flow", source=["A"], target=["B"], rate="k * A")
        )

    def test_uncalibrated_bin_fractions_are_treated_as_unknown_base(self) -> None:
        # A is uncalibrated (None) and B carries 0.5. The override on A keeps
        # the known total at 0.9, within population_size. Treating the unknown
        # base as 0.0 is sound: it never rejects a configuration whose
        # calibrated total stays within population_size.
        builder = self._uncalibrated_builder()
        builder.set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": None},
                {"bin": "B", "fraction": 0.5},
            ],
            compartment_fractions=[{"compartment": "A", "fraction": 0.4}],
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        overrides = model.population.initial_conditions.compartment_fractions
        assert [(cf.compartment, cf.fraction) for cf in overrides] == [("A", 0.4)]

    def test_uncalibrated_override_exceeding_population_is_rejected(self) -> None:
        builder = self._uncalibrated_builder()
        with pytest.raises(ValueError, match="exceed population_size"):
            builder.set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": None},
                    {"bin": "B", "fraction": 0.5},
                ],
                compartment_fractions=[{"compartment": "A", "fraction": 1.0}],
            )


def _conditional_pathway_model() -> Model:
    """A model whose IH pathway only exists in the ``oe60`` stratum."""
    return (
        ModelBuilder(name="ConditionalPathway", version="1.0")
        .add_bin(id="S", name="Susceptible")
        .add_bin(id="IH", name="Hospitalized")
        .add_parameter(id="beta", value=0.01)
        .add_transition(
            id="hosp",
            source=["S"],
            target=["IH"],
            rate="0.0",
            stratified_rates=[
                {
                    "conditions": [{"stratification": "age", "category": "oe60"}],
                    "rate": "beta * S",
                }
            ],
        )
        .add_stratification(id="age", categories=["y60", "oe60"])
        .add_stratification(
            id="care",
            categories=["home", "hosp"],
            conditions=[{"stratification": "age", "category": "oe60"}],
        )
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "S", "fraction": 0.9},
                {"bin": "IH", "fraction": 0.1},
            ],
            stratification_fractions=[
                {
                    "stratification": "age",
                    "fractions": [
                        {"category": "y60", "fraction": 0.5},
                        {"category": "oe60", "fraction": 0.5},
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
            compartment_fractions=[{"compartment": "IH_y60", "fraction": 0.0}],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


class TestEndToEndOverride:
    def test_zero_override_keeps_the_other_stratum_empty(self) -> None:
        model = _conditional_pathway_model()
        results = Simulation(model).run(10)

        assert all(value == 0.0 for value in results["IH_y60"])
        assert results["IH_oe60_hosp"][-1] > results["IH_oe60_hosp"][0]
        assert results["S_oe60_hosp"][-1] < results["S_oe60_hosp"][0]

    def test_uniform_expansion_is_preserved_without_overrides(self) -> None:
        builder = (
            ModelBuilder(name="NoOverride", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="flow", source=["A"], target=["B"], rate="k * A")
            .add_stratification(id="group", categories=["g1", "g2"])
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": 0.5},
                    {"bin": "B", "fraction": 0.5},
                ],
                stratification_fractions=[
                    {
                        "stratification": "group",
                        "fractions": [
                            {"category": "g1", "fraction": 0.5},
                            {"category": "g2", "fraction": 0.5},
                        ],
                    }
                ],
            )
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        engine = Simulation(model).engine
        initial = dict(zip(engine.compartments, engine.population))

        assert initial == {"A_g1": 250.0, "A_g2": 250.0, "B_g1": 250.0, "B_g2": 250.0}


class TestCalibrationRoundTrip:
    def _problem(self) -> CalibrationProblem:
        return CalibrationProblem(
            observed_data=[ObservedDataPoint(step=1, compartment="A_g1", value=1.0)],
            parameters=[
                CalibrationParameter(
                    id="A_g1",
                    parameter_type="initial_condition",
                    min_bound=0.0,
                    max_bound=1.0,
                )
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )

    def test_exact_compartment_is_written_back_and_rebuilds_identically(self) -> None:
        builder = (
            ModelBuilder(name="CalibrationRoundTrip", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_transition(
                id="flow",
                source=["A"],
                target=["B"],
                rate="k * A",
                per_compartment=True,
            )
            .add_stratification(id="group", categories=["g1", "g2"])
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
                    }
                ],
            )
        )
        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        problem = self._problem()

        model.apply_calibration_parameters({"A_g1": 0.2}, problem)

        overrides = {
            fraction.compartment: fraction.fraction
            for fraction in model.population.initial_conditions.compartment_fractions
        }
        assert overrides == {"A_g1": pytest.approx(0.2)}

        simulation = Simulation(model)
        initial = dict(
            zip(simulation.engine.compartments, simulation.engine.population)
        )
        assert initial["A_g1"] == pytest.approx(200.0)
