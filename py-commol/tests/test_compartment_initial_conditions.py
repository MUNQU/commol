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

    def test_empty_compartment_fractions_use_uniform_expansion(self) -> None:
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


def _aggregate_override_model() -> Model:
    """A model with an overridden compartment inside an aggregate bin."""
    return (
        ModelBuilder(name="AggregateOverride", version="1.0")
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
            compartment_fractions=[{"compartment": "A_g1", "fraction": 0.3}],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


def _initial_condition_problem(*ids: str) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(step=1, compartment="A_g1", value=1.0)],
        parameters=[
            CalibrationParameter(
                id=entry_id,
                parameter_type="initial_condition",
                min_bound=0.0,
                max_bound=1.0,
            )
            for entry_id in ids
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )


class TestAggregateOverrideWriteBack:
    def test_bin_parameter_with_override_reproduces_the_scored_state(self) -> None:
        # Aggregate A = 0.6 over the base A_g1 = 0.3 and A_g2 = 0.5:
        # weights 0.375 / 0.625 give 0.225 / 0.375 of the population.
        model = _aggregate_override_model()

        model.apply_calibration_parameters({"A": 0.6}, _initial_condition_problem("A"))

        simulation = Simulation(model)
        initial = dict(
            zip(simulation.engine.compartments, simulation.engine.population)
        )
        assert initial["A_g1"] == pytest.approx(225.0)
        assert initial["A_g2"] == pytest.approx(375.0)

    def test_bin_parameter_without_override_still_updates_the_bin_fraction(
        self,
    ) -> None:
        model = (
            ModelBuilder(name="AggregatePlain", version="1.0")
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
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        model.apply_calibration_parameters({"A": 0.6}, _initial_condition_problem("A"))

        bin_fractions = {
            fraction.bin: fraction.fraction
            for fraction in model.population.initial_conditions.bin_fractions
        }
        assert bin_fractions["A"] == pytest.approx(0.6)
        assert model.population.initial_conditions.compartment_fractions == []

    def test_category_parameter_without_override_still_updates_the_category(
        self,
    ) -> None:
        model = (
            ModelBuilder(name="CategoryPlain", version="1.0")
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
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        model.apply_calibration_parameters(
            {"g1": 0.6}, _initial_condition_problem("g1")
        )

        initial_conditions = model.population.initial_conditions
        categories = {
            fraction.category: fraction.fraction
            for stratification in initial_conditions.stratification_fractions
            for fraction in stratification.fractions
        }
        assert categories["g1"] == pytest.approx(0.6)
        assert model.population.initial_conditions.compartment_fractions == []

    def test_category_parameter_with_override_reproduces_the_pair_split(self) -> None:
        model = _aggregate_override_model()

        model.apply_calibration_parameters(
            {"g1": 0.6}, _initial_condition_problem("g1")
        )

        simulation = Simulation(model)
        initial = dict(
            zip(simulation.engine.compartments, simulation.engine.population)
        )
        # Pair total is the overridden A_g1 = 0.3 plus A_g2 = 0.5.
        assert initial["A_g1"] == pytest.approx(0.6 * 0.8 * 1000)
        assert initial["A_g2"] == pytest.approx(0.4 * 0.8 * 1000)

    def test_aggregate_then_category_replays_the_engine_sequence(self) -> None:
        # Base A_g1 = 0.3 and A_g2 = 0.5. Aggregate A = 0.9 splits them by
        # 0.375 / 0.625, then category g1 = 0.4 splits the live pair
        # A_g1 + A_g2, which holds 0.9 of the population.
        model = _aggregate_override_model()

        model.apply_calibration_parameters(
            {"A": 0.9, "g1": 0.4}, _initial_condition_problem("A", "g1")
        )

        simulation = Simulation(model)
        initial = dict(
            zip(simulation.engine.compartments, simulation.engine.population)
        )
        assert initial["A_g1"] == pytest.approx(0.4 * 0.9 * 1000)
        assert initial["A_g2"] == pytest.approx(0.6 * 0.9 * 1000)
        assert initial["B_g1"] == pytest.approx(0.0)
        assert initial["B_g2"] == pytest.approx(0.0)

    def test_category_then_aggregate_without_override_replays_the_engine(
        self,
    ) -> None:
        # With no override, category g1 = 0.4 splits the base pairs first, then
        # aggregate A = 0.9 overwrites both A compartments.
        model = (
            ModelBuilder(name="CategoryThenAggregate", version="1.0")
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
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        model.apply_calibration_parameters(
            {"g1": 0.4, "A": 0.9}, _initial_condition_problem("g1", "A")
        )

        simulation = Simulation(model)
        initial = dict(
            zip(simulation.engine.compartments, simulation.engine.population)
        )
        assert initial["A_g1"] == pytest.approx(0.9 * 0.5 * 1000)
        assert initial["A_g2"] == pytest.approx(0.9 * 0.5 * 1000)
        assert initial["B_g1"] == pytest.approx(0.0)
        assert initial["B_g2"] == pytest.approx(0.0)

    def test_diverted_write_back_pins_the_scored_absolute_values(self) -> None:
        model = _aggregate_override_model()

        model.apply_calibration_parameters({"A": 0.6}, _initial_condition_problem("A"))

        initial_conditions = model.population.initial_conditions
        overrides = {
            fraction.compartment: fraction.fraction
            for fraction in initial_conditions.compartment_fractions
        }
        # The scored values are pinned as overrides; the bin fraction the
        # aggregate did not update stays at 1.0.
        assert overrides["A_g1"] == pytest.approx(0.225)
        assert overrides["A_g2"] == pytest.approx(0.375)
        bin_a = next(
            fraction
            for fraction in initial_conditions.bin_fractions
            if fraction.bin == "A"
        )
        assert bin_a.fraction == pytest.approx(1.0)


class TestUnknownCompartmentOverride:
    def test_building_a_simulation_raises_a_catchable_error(self) -> None:
        model = _aggregate_override_model()
        model.population.initial_conditions.compartment_fractions.append(
            CompartmentFraction(compartment="A_missing", fraction=0.0)
        )

        with pytest.raises(RuntimeError, match="A_missing") as excinfo:
            Simulation(model)

        assert isinstance(excinfo.value, Exception)


class TestCompartmentFractionValidation:
    def test_model_rejects_an_unknown_compartment_name(self) -> None:
        model = _aggregate_override_model()

        with pytest.raises(ValueError, match="Unknown compartment"):
            model.update_compartment_fractions({"A_missing": 0.1})

    def test_initial_conditions_reject_over_allocation(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[
                BinFraction(bin="A", fraction=1.0),
                BinFraction(bin="B", fraction=0.0),
            ],
        )

        with pytest.raises(ValueError, match="exceed population_size"):
            conditions.update_compartment_fractions({"A": 0.7, "B": 0.7})

    def test_initial_conditions_allow_overrides_within_the_population(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[
                BinFraction(bin="A", fraction=0.6),
                BinFraction(bin="B", fraction=0.4),
            ],
        )

        conditions.update_compartment_fractions({"A": 0.5})

        assert [
            (fraction.compartment, fraction.fraction)
            for fraction in conditions.compartment_fractions
        ] == [("A", 0.5)]

    def test_construction_accepts_overrides_exceeding_the_population(self) -> None:
        # ESC-1: the definition may represent any state the engine can hold.
        # Total guards live in authoring paths (ModelBuilder, public
        # update_* methods), not in construction/deserialization.
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[BinFraction(bin="A", fraction=1.0)],
            compartment_fractions=[
                CompartmentFraction(compartment="A_g1", fraction=0.6),
                CompartmentFraction(compartment="A_g2", fraction=0.6),
            ],
        )

        assert [
            (fraction.compartment, fraction.fraction)
            for fraction in conditions.compartment_fractions
        ] == [("A_g1", 0.6), ("A_g2", 0.6)]

    def test_model_rejects_over_allocation_at_compartment_granularity(self) -> None:
        model = _aggregate_override_model()

        # Uniform A_g1 = A_g2 = 0.5, so an override of 0.9 on A_g1 plus the
        # untouched A_g2 totals 1.4 of the population.
        with pytest.raises(ValueError, match="exceed population_size"):
            model.update_compartment_fractions({"A_g1": 0.9})

        assert [
            (fraction.compartment, fraction.fraction)
            for fraction in model.population.initial_conditions.compartment_fractions
        ] == [("A_g1", 0.3)]

    def test_model_allows_over_allocation_within_the_population(self) -> None:
        model = _aggregate_override_model()

        model.update_compartment_fractions({"A_g1": 0.4})

        overrides = {
            fraction.compartment: fraction.fraction
            for fraction in model.population.initial_conditions.compartment_fractions
        }
        assert overrides == {"A_g1": pytest.approx(0.4)}


class TestNonConservingReload:
    """ESC-1: deserialization accepts whatever the write-back persisted."""

    @staticmethod
    def _non_conserving_model() -> tuple[Model, list[float]]:
        model = _shared_category_model()
        model.apply_calibration_parameters({"y": 0.6}, _initial_condition_problem("y"))
        expected = list(Simulation(model).engine.population)
        return model, expected

    def test_overrides_may_exceed_the_population(self) -> None:
        model, _ = self._non_conserving_model()

        total = sum(
            fraction.fraction
            for fraction in model.population.initial_conditions.compartment_fractions
        )
        assert total == pytest.approx(1.1)

    def test_model_validate_json_round_trips(self) -> None:
        model, expected = self._non_conserving_model()

        reloaded = Model.model_validate_json(model.model_dump_json())

        assert list(Simulation(reloaded).engine.population) == pytest.approx(expected)

    def test_from_json_round_trips(self, tmp_path) -> None:
        model, expected = self._non_conserving_model()
        path = tmp_path / "model.json"
        model.to_json(path)

        reloaded = Model.from_json(path)

        assert list(Simulation(reloaded).engine.population) == pytest.approx(expected)


class TestBinFractionValidation:
    def _conditions(self) -> InitialConditions:
        return InitialConditions(
            population_size=1000,
            bin_fractions=[
                BinFraction(bin="A", fraction=0.5),
                BinFraction(bin="B", fraction=0.5),
            ],
        )

    def test_complete_update_must_sum_to_one(self) -> None:
        conditions = self._conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.9, "B": 0.9})

    def test_partial_update_must_not_exceed_one(self) -> None:
        conditions = self._conditions()

        with pytest.raises(ValueError, match="more than 1.0"):
            conditions.update_bin_fractions({"A": 0.9})

    def test_rejected_update_leaves_the_bins_unchanged(self) -> None:
        conditions = self._conditions()

        with pytest.raises(ValueError, match="more than 1.0"):
            conditions.update_bin_fractions({"A": 0.9})

        assert [(bf.bin, bf.fraction) for bf in conditions.bin_fractions] == [
            ("A", 0.5),
            ("B", 0.5),
        ]

    def test_partial_update_within_the_population_is_allowed(self) -> None:
        conditions = self._conditions()

        conditions.update_bin_fractions({"A": 0.4})

        assert [(bf.bin, bf.fraction) for bf in conditions.bin_fractions] == [
            ("A", 0.4),
            ("B", 0.5),
        ]


class TestBinWriteBackParity:
    """JD-G-3: a non-conserving bin value is diverted, not persisted."""

    @staticmethod
    def _two_bin_model() -> Model:
        return (
            ModelBuilder(name="TwoBin", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="flow", source=["A"], target=["B"], rate="k * A")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": 0.5},
                    {"bin": "B", "fraction": 0.5},
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

    def test_non_conserving_bin_value_is_diverted_to_overrides(self) -> None:
        model = self._two_bin_model()
        problem = _initial_condition_problem("A")

        model.apply_calibration_parameters({"A": 0.6}, problem)

        initial_conditions = model.population.initial_conditions
        # The declared mapping stays valid: the bin fractions are untouched
        # and the replayed head counts are pinned as per-compartment overrides.
        assert {
            fraction.bin: fraction.fraction
            for fraction in initial_conditions.bin_fractions
        } == {"A": 0.5, "B": 0.5}
        overrides = {
            fraction.compartment: fraction.fraction
            for fraction in initial_conditions.compartment_fractions
        }
        assert overrides == {"A": pytest.approx(0.6)}

        # The diverted definition round-trips and rebuilds the scored state.
        reloaded = Model.model_validate_json(model.model_dump_json())
        assert reloaded.model_dump_json() == model.model_dump_json()
        rebuilt = Simulation(reloaded).engine
        assert dict(zip(rebuilt.compartments, rebuilt.population)) == pytest.approx(
            {"A": 600.0, "B": 500.0}
        )
        assert sum(rebuilt.population) == pytest.approx(1100.0)

    def test_non_conserving_bin_apply_is_idempotent(self) -> None:
        model = self._two_bin_model()
        problem = _initial_condition_problem("A")

        model.apply_calibration_parameters({"A": 0.6}, problem)
        first_json = model.model_dump_json()
        first_population = list(Simulation(model).engine.population)

        model.apply_calibration_parameters({"A": 0.6}, problem)

        assert model.model_dump_json() == first_json
        assert list(Simulation(model).engine.population) == first_population

    def test_conserving_bin_value_still_updates_the_bin_fraction(self) -> None:
        model = self._two_bin_model()
        problem = _initial_condition_problem("A")

        model.apply_calibration_parameters({"A": 0.4}, problem)

        initial_conditions = model.population.initial_conditions
        assert {
            fraction.bin: fraction.fraction
            for fraction in initial_conditions.bin_fractions
        } == {"A": 0.4, "B": 0.5}
        assert initial_conditions.compartment_fractions == []

        first_json = model.model_dump_json()
        model.apply_calibration_parameters({"A": 0.4}, problem)
        assert model.model_dump_json() == first_json

    def test_public_update_bin_fractions_still_rejects_non_conserving(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[
                BinFraction(bin="A", fraction=0.5),
                BinFraction(bin="B", fraction=0.5),
            ],
        )

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.6, "B": 0.5})


class TestPopulationSizeValidation:
    def test_zero_population_size_is_a_domain_error(self) -> None:
        with pytest.raises(
            ValidationError, match="population_size must be greater than 0"
        ):
            InitialConditions(
                population_size=0,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
            )

    def test_negative_population_size_is_a_domain_error(self) -> None:
        with pytest.raises(
            ValidationError, match="population_size must be greater than 0"
        ):
            InitialConditions(
                population_size=-10,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
            )


class TestManyCompartmentFractions:
    def test_thousands_of_unique_names_still_validate(self) -> None:
        conditions = InitialConditions(
            population_size=1_000_000,
            bin_fractions=[BinFraction(bin="A", fraction=1.0)],
            compartment_fractions=[
                CompartmentFraction(compartment=f"A_c{index}", fraction=1e-6)
                for index in range(4000)
            ],
        )

        assert len(conditions.compartment_fractions) == 4000

    def test_duplicates_are_still_detected_in_large_inputs(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate compartment fractions"):
            InitialConditions(
                population_size=1_000_000,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
                compartment_fractions=[
                    CompartmentFraction(compartment="A_c0", fraction=1e-6),
                    *(
                        CompartmentFraction(compartment=f"A_c{index}", fraction=1e-6)
                        for index in range(1, 2000)
                    ),
                    CompartmentFraction(compartment="A_c0", fraction=1e-6),
                ],
            )


def _shared_category_model() -> Model:
    """Two stratifications that declare the same category names."""
    return (
        ModelBuilder(name="SharedCategory", version="1.0")
        .add_bin(id="A", name="A")
        .add_parameter(id="k", value=0.1)
        .add_transition(
            id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
        )
        .add_stratification(id="age", categories=["y", "o"])
        .add_stratification(id="sex", categories=["y", "o"])
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 1.0}],
            stratification_fractions=[
                {
                    "stratification": "age",
                    "fractions": [
                        {"category": "y", "fraction": 0.5},
                        {"category": "o", "fraction": 0.5},
                    ],
                },
                {
                    "stratification": "sex",
                    "fractions": [
                        {"category": "y", "fraction": 0.5},
                        {"category": "o", "fraction": 0.5},
                    ],
                },
            ],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


def _shared_category_override_model() -> Model:
    """Shared-category model with a declared override on one compartment."""
    return (
        ModelBuilder(name="SharedCategoryOverride", version="1.0")
        .add_bin(id="A", name="A")
        .add_parameter(id="k", value=0.1)
        .add_transition(
            id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
        )
        .add_stratification(id="age", categories=["y", "o"])
        .add_stratification(id="sex", categories=["y", "o"])
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 1.0}],
            stratification_fractions=[
                {
                    "stratification": "age",
                    "fractions": [
                        {"category": "y", "fraction": 0.5},
                        {"category": "o", "fraction": 0.5},
                    ],
                },
                {
                    "stratification": "sex",
                    "fractions": [
                        {"category": "y", "fraction": 0.5},
                        {"category": "o", "fraction": 0.5},
                    ],
                },
            ],
            compartment_fractions=[{"compartment": "A_y_y", "fraction": 0.0}],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


def _engine_category_replay(
    compartments: list[str],
    population: list[float],
    category: str,
    value: float,
) -> list[float]:
    """Mirror the engine's paired-category replay (snapshot + sorted pairs)."""
    pairs: list[tuple[int, int]] = []
    for index, name in enumerate(compartments):
        tokens = name.split("_")
        for token_index, token in enumerate(tokens):
            if token_index == 0 or token != category:
                continue
            complements = [
                candidate_index
                for candidate_index, candidate in enumerate(compartments)
                if len(candidate.split("_")) == len(tokens)
                and candidate.split("_")[token_index] != category
                and all(
                    position == token_index or candidate.split("_")[position] == other
                    for position, other in enumerate(tokens)
                )
            ]
            assert len(complements) == 1
            pairs.append((index, complements[0]))
    pairs.sort()

    result = list(population)
    snapshot = list(population)
    for category_index, complement_index in pairs:
        pair_total = snapshot[category_index] + snapshot[complement_index]
        result[category_index] = value * pair_total
        result[complement_index] = (1.0 - value) * pair_total
    return result


def _mixed_replay_model() -> Model:
    """Shared category, a unique category, and two bins for a mixed replay."""
    return (
        ModelBuilder(name="MixedReplay", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=0.1)
        .add_transition(
            id="flow", source=["A"], target=[], rate="k * A", per_compartment=True
        )
        .add_stratification(id="age", categories=["y", "o"])
        .add_stratification(id="sex", categories=["y", "o"])
        .add_stratification(id="grp", categories=["g1", "g2"])
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 0.6},
                {"bin": "B", "fraction": 0.4},
            ],
            stratification_fractions=[
                {
                    "stratification": "age",
                    "fractions": [
                        {"category": "y", "fraction": 0.5},
                        {"category": "o", "fraction": 0.5},
                    ],
                },
                {
                    "stratification": "sex",
                    "fractions": [
                        {"category": "y", "fraction": 0.5},
                        {"category": "o", "fraction": 0.5},
                    ],
                },
                {
                    "stratification": "grp",
                    "fractions": [
                        {"category": "g1", "fraction": 0.5},
                        {"category": "g2", "fraction": 0.5},
                    ],
                },
            ],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


def _engine_aggregate_weights(
    compartments: list[str], base: list[float], bin_id: str
) -> list[tuple[int, float]]:
    """Engine aggregate weights for a bin, taken from the base population."""
    covered = [
        index
        for index, name in enumerate(compartments)
        if name == bin_id or name.startswith(f"{bin_id}_")
    ]
    total = sum(base[index] for index in covered)
    if total > 0.0:
        return [(index, base[index] / total) for index in covered]
    return [(index, 1.0 / len(covered)) for index in covered]


def _engine_category_pairs(
    compartments: list[str], category: str
) -> list[tuple[int, int]]:
    """Engine complementary pairs for a category, sorted by compartment index."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, name in enumerate(compartments):
        tokens = name.split("_")
        for token_index, token in enumerate(tokens):
            if token_index == 0 or token != category:
                continue
            complements = [
                candidate_index
                for candidate_index, candidate in enumerate(compartments)
                if len(candidate.split("_")) == len(tokens)
                and candidate.split("_")[token_index] != category
                and all(
                    position == token_index or candidate.split("_")[position] == other
                    for position, other in enumerate(tokens)
                )
            ]
            assert len(complements) == 1
            pair = (index, complements[0])
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    pairs.sort()
    return pairs


def _engine_replay(
    model: Model,
    ordered_values: list[tuple[str, float]],
) -> list[float]:
    """Mirror the engine's initial-condition replay (calibration_problem.rs).

    A fresh problem rebuilt from ``model`` precomputes aggregate weights and
    category pairs from the current base engine; each parameter then reads the
    population once before writing its targets. Category pairs are sorted by
    compartment index, matching the Rust replay.
    """
    engine = Simulation(model).engine
    compartments = list(engine.compartments)
    base = list(engine.population)
    initial_conditions = model.population.initial_conditions
    population_size = float(initial_conditions.population_size)
    bin_ids = {bin_item.id for bin_item in model.population.bins}
    categories = initial_conditions.get_categories_with_fractions()

    result = list(base)
    for param_id, value in ordered_values:
        if param_id in bin_ids:
            for index, weight in _engine_aggregate_weights(
                compartments, base, param_id
            ):
                result[index] = value * weight * population_size
        elif param_id in categories:
            snapshot = list(result)
            for category_index, complement_index in _engine_category_pairs(
                compartments, param_id
            ):
                pair_total = snapshot[category_index] + snapshot[complement_index]
                result[category_index] = value * pair_total
                result[complement_index] = (1.0 - value) * pair_total
        elif param_id in compartments:
            result[compartments.index(param_id)] = value * population_size
        else:
            raise AssertionError(f"Unknown initial-condition id: {param_id}")
    return result


class TestSharedCategoryReplay:
    def test_write_back_matches_the_engine_snapshot_replay(self) -> None:
        model = _shared_category_model()
        base_simulation = Simulation(model)
        compartments = list(base_simulation.engine.compartments)
        base_population = list(base_simulation.engine.population)
        expected = _engine_category_replay(compartments, base_population, "y", 0.4)

        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(step=1, compartment="A_y_y", value=1.0)],
            parameters=[
                CalibrationParameter(
                    id="y",
                    parameter_type="initial_condition",
                    min_bound=0.0,
                    max_bound=1.0,
                )
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )

        model.apply_calibration_parameters({"y": 0.4}, problem)

        # The shared category cannot be expressed by one stratification
        # fraction, so it is persisted as per-compartment overrides that rebuild
        # the engine replay exactly.
        rebuilt = Simulation(model)
        assert list(rebuilt.engine.population) == pytest.approx(expected)
        assert model.population.initial_conditions.stratification_fractions[
            0
        ].fractions[0].fraction == pytest.approx(0.5)


class TestCalibrationIdempotence:
    """Applying the same calibration result twice must not move the model."""

    @staticmethod
    def _flat_category_model() -> Model:
        return (
            ModelBuilder(name="FlatCategory", version="1.0")
            .add_bin(id="A", name="A")
            .add_parameter(id="k", value=0.1)
            .add_transition(
                id="flow",
                source=["A"],
                target=[],
                rate="k * A",
                per_compartment=True,
            )
            .add_stratification(id="group", categories=["g1", "g2"])
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[{"bin": "A", "fraction": 1.0}],
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
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

    @staticmethod
    def _apply_twice(
        model: Model, values: dict[str, float], problem: CalibrationProblem
    ) -> tuple[str, list[float]]:
        model.apply_calibration_parameters(values, problem)
        first_json = model.model_dump_json()
        first_population = list(Simulation(model).engine.population)
        model.apply_calibration_parameters(values, problem)
        return first_json, first_population

    def test_flat_category_double_apply_is_idempotent(self) -> None:
        model = self._flat_category_model()
        problem = _initial_condition_problem("g1")

        first_json, first_population = self._apply_twice(model, {"g1": 0.4}, problem)

        assert model.model_dump_json() == first_json
        assert list(Simulation(model).engine.population) == first_population

    def test_conditional_category_double_apply_is_idempotent(self) -> None:
        model = _conditional_pathway_model()
        problem = _initial_condition_problem("y60")

        first_json, first_population = self._apply_twice(model, {"y60": 0.4}, problem)

        assert model.model_dump_json() == first_json
        assert list(Simulation(model).engine.population) == first_population

    def test_shared_category_double_apply_rescores_like_the_engine(self) -> None:
        # ESC-2: a category declared by more than one stratification is
        # intentionally NOT idempotent on re-apply. The second apply replays
        # from the reloaded (already diverted) population, exactly like a fresh
        # engine problem built on that state, so the shared split is re-scored.
        model = _shared_category_model()
        problem = _initial_condition_problem("y")

        first = _engine_replay(model, [("y", 0.4)])
        model.apply_calibration_parameters({"y": 0.4}, problem)
        assert list(Simulation(model).engine.population) == pytest.approx(first)

        second = _engine_replay(model, [("y", 0.4)])
        model.apply_calibration_parameters({"y": 0.4}, problem)
        assert list(Simulation(model).engine.population) == pytest.approx(second)
        assert second != pytest.approx(first)

    def test_shared_category_non_conserving_apply_is_accepted(self) -> None:
        model = _shared_category_model()
        problem = _initial_condition_problem("y")
        base = Simulation(model)
        expected = _engine_category_replay(
            list(base.engine.compartments), list(base.engine.population), "y", 0.6
        )

        model.apply_calibration_parameters({"y": 0.6}, problem)

        assert list(Simulation(model).engine.population) == pytest.approx(expected)
        assert sum(expected) == pytest.approx(1100.0)

    def test_multi_parameter_shared_category_replay_matches_the_engine(self) -> None:
        model = _shared_category_model()
        problem = _initial_condition_problem("y", "A_y_y")
        base = Simulation(model)
        compartments = list(base.engine.compartments)
        # The engine applies the shared category first and then pins one exact
        # compartment, so the shared diversion must compose in parameter order.
        after_category = _engine_category_replay(
            compartments, list(base.engine.population), "y", 0.4
        )
        expected = list(after_category)
        expected[compartments.index("A_y_y")] = 0.5 * 1000

        model.apply_calibration_parameters({"y": 0.4, "A_y_y": 0.5}, problem)

        assert list(Simulation(model).engine.population) == pytest.approx(expected)


class TestEngineParityProbes:
    def test_pre_existing_override_participates_in_the_live_replay(self) -> None:
        # ESC-2 probe: the declared override A_y_y = 0.0 is part of the live
        # working population the replay starts from, so the shared-category
        # replay reproduces the engine's [100, 200, 200, 300] rather than the
        # declared-base [200, 200, 200, 300].
        model = _shared_category_override_model()
        problem = _initial_condition_problem("y")
        expected = _engine_replay(model, [("y", 0.4)])
        assert list(expected) == pytest.approx([100.0, 200.0, 200.0, 300.0])

        model.apply_calibration_parameters({"y": 0.4}, problem)

        assert list(Simulation(model).engine.population) == pytest.approx(expected)
        assert list(Simulation(model).engine.population) != pytest.approx(
            [200.0, 200.0, 200.0, 300.0]
        )

    def test_sequential_shared_category_applies_match_the_engine_replay(self) -> None:
        # ESC-2 probe: applying y = 0.4 then y = 0.6 re-scores from the reloaded
        # population, matching the engine's [240, 300, 300, 200].
        model = _shared_category_model()
        problem = _initial_condition_problem("y")

        first = _engine_replay(model, [("y", 0.4)])
        model.apply_calibration_parameters({"y": 0.4}, problem)
        assert list(Simulation(model).engine.population) == pytest.approx(first)
        assert list(first) == pytest.approx([200.0, 200.0, 200.0, 300.0])

        second = _engine_replay(model, [("y", 0.6)])
        model.apply_calibration_parameters({"y": 0.6}, problem)
        assert list(Simulation(model).engine.population) == pytest.approx(second)
        assert list(second) == pytest.approx([240.0, 300.0, 300.0, 200.0])


class TestMixedReplay:
    def test_mixed_dict_applied_twice_matches_the_engine_mirror(self) -> None:
        # ESC-2: a mixed dict of shared category, unique category, and bin
        # values replays in parameter order from the live population. Each
        # apply is checked against an independent engine mirror.
        model = _mixed_replay_model()
        problem = _initial_condition_problem("y", "g1", "A")
        ordered = [("y", 0.4), ("g1", 0.6), ("A", 0.5)]
        values = {"y": 0.4, "g1": 0.6, "A": 0.5}

        first = _engine_replay(model, ordered)
        model.apply_calibration_parameters(values, problem)
        assert list(Simulation(model).engine.population) == pytest.approx(first)

        second = _engine_replay(model, ordered)
        model.apply_calibration_parameters(values, problem)
        assert list(Simulation(model).engine.population) == pytest.approx(second)


class TestApplyCalibrationAtomicity:
    def _model(self) -> Model:
        return (
            ModelBuilder(name="Atomic", version="1.0")
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
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

    def test_failing_write_back_restores_the_definition(self) -> None:
        model = self._model()
        problem = _initial_condition_problem("A", "A_g1")

        before = {
            fraction.bin: fraction.fraction
            for fraction in model.population.initial_conditions.bin_fractions
        }

        # The bin write-back lands first, then the out-of-range exact
        # compartment fraction fails, so the whole call must roll back.
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            model.apply_calibration_parameters({"A": 0.4, "A_g1": 1.5}, problem)

        after = {
            fraction.bin: fraction.fraction
            for fraction in model.population.initial_conditions.bin_fractions
        }
        assert after == before
        assert model.population.initial_conditions.compartment_fractions == []


class TestPyO3CompartmentFractions:
    def test_low_level_binding_exposes_and_applies_overrides(self) -> None:
        from commol.commol_rs import _commol_rs as commol_rs

        core = commol_rs.core
        difference = commol_rs.difference

        initial_conditions = core.InitialConditions(
            population_size=1000,
            bin_fractions=[("A", 1.0)],
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
            compartment_fractions=[("A_g1", 0.2)],
        )

        assert initial_conditions.compartment_fractions == [("A_g1", 0.2)]

        model = core.Model(
            name="LowLevel",
            population=core.Population(
                bins=[core.Bin(id="A", name="A")],
                stratifications=[
                    core.Stratification(id="group", categories=["g1", "g2"])
                ],
                initial_conditions=initial_conditions,
            ),
            dynamics=core.Dynamics(typology="DifferenceEquations", transitions=[]),
            parameters=[],
        )
        engine = difference.DifferenceEquations(model)

        assert engine.compartments == ["A_g1", "A_g2"]
        assert engine.population == [200.0, 500.0]

    def test_defaults_to_no_overrides(self) -> None:
        from commol.commol_rs import _commol_rs as commol_rs

        initial_conditions = commol_rs.core.InitialConditions(
            population_size=1000,
            bin_fractions=[("A", 1.0)],
        )

        assert initial_conditions.compartment_fractions == []

    def test_uncalibrated_bin_fraction_round_trips(self) -> None:
        from commol.commol_rs import _commol_rs as commol_rs

        initial_conditions = commol_rs.core.InitialConditions(
            population_size=1000,
            bin_fractions=[("A", 0.6), ("B", None)],
        )

        assert initial_conditions.bin_fractions == [("A", 0.6), ("B", None)]

    def test_every_declared_initial_conditions_attribute_is_readable(self) -> None:
        from commol.commol_rs import _commol_rs as commol_rs

        initial_conditions = commol_rs.core.InitialConditions(
            population_size=1000,
            bin_fractions=[("A", 0.5), ("B", 0.5)],
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
            compartment_fractions=[("A_g1", 0.2)],
        )

        assert initial_conditions.population_size == 1000
        assert initial_conditions.bin_fractions == [("A", 0.5), ("B", 0.5)]
        assert {
            name: dict(categories)
            for name, categories in initial_conditions.stratification_fractions
        } == {"group": {"g1": 0.5, "g2": 0.5}}
        assert initial_conditions.compartment_fractions == [("A_g1", 0.2)]
