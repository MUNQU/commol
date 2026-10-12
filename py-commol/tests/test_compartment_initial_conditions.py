"""Per-compartment initial conditions: types, validation, write-back."""

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.api.model_builder import ModelBuilder
from commol.constants import ModelTypes
from commol.context import CompartmentFraction
from commol.context._model.engine_state import calibrated_initial_population
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

    def test_unknown_compartment_is_rejected_by_the_engine_at_build(self) -> None:
        builder = _set_initial_conditions(
            _stratified_builder(), [{"compartment": "A_missing", "fraction": 0.0}]
        )

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

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

    def test_over_allocation_is_rejected_by_the_engine_at_build(self) -> None:
        builder = _set_initial_conditions(
            _stratified_builder(), [{"compartment": "A_g2", "fraction": 0.5}]
        )

        with pytest.raises(ValueError, match="exceed population_size"):
            builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

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

    def test_uncalibrated_bin_fractions_count_as_zero(self) -> None:
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
        builder = self._uncalibrated_builder().set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": None},
                {"bin": "B", "fraction": 0.5},
            ],
            compartment_fractions=[{"compartment": "A", "fraction": 1.0}],
        )

        with pytest.raises(ValueError, match="exceed population_size"):
            builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)


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

    def test_initial_state_matches_the_engine(self) -> None:
        model = _conditional_pathway_model()
        engine = Simulation(model).engine

        state = model.initial_state()

        assert [c.name for c in state] == engine.compartments
        assert [c.population for c in state] == pytest.approx(engine.population)
        ih_y60 = next(c for c in state if c.name == "IH_y60")
        assert ih_y60.bin == "IH"
        assert ih_y60.categories == {"age": "y60"}
        assert ih_y60.population == 0.0

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


def _initial_condition_problem(
    *ids: str, remainder: str | None = None
) -> CalibrationProblem:
    return CalibrationProblem(
        initial_condition_remainder=remainder,
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


def _rebuilt(model: Model) -> dict[str, float]:
    engine = Simulation(model).engine
    return dict(zip(engine.compartments, engine.population, strict=True))


def _scored(
    model: Model, problem: CalibrationProblem, values: dict[str, float]
) -> dict[str, float]:
    """Population the engine scores for the values, before any write-back."""
    parameters = [
        param
        for param in problem.parameters
        if param.parameter_type == "initial_condition" and param.id in values
    ]
    return calibrated_initial_population(
        model,
        parameters,
        [values[param.id] for param in parameters],
        problem.initial_condition_remainder,
    ).population


def _overrides(model: Model) -> dict[str, float]:
    return {
        fraction.compartment: fraction.fraction
        for fraction in model.population.initial_conditions.compartment_fractions
    }


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


def _flat_category_model() -> Model:
    return _set_initial_conditions(_stratified_builder(), []).build(
        typology=ModelTypes.DIFFERENCE_EQUATIONS.value
    )


WRITE_BACK_CASES = [
    pytest.param(_aggregate_override_model, {"A_g1": 0.2}, None, id="single-exact"),
    pytest.param(
        _aggregate_override_model, {"A_g1": 0.2}, "B", id="single-exact-remainder"
    ),
    pytest.param(
        _aggregate_override_model, {"A_g1": 0.2, "A_g2": 0.6}, None, id="two-exact"
    ),
    pytest.param(_aggregate_override_model, {"A": 0.6}, None, id="bin-over-override"),
    pytest.param(
        _aggregate_override_model, {"g1": 0.3}, None, id="category-over-override"
    ),
    pytest.param(
        _aggregate_override_model, {"A": 0.6, "g1": 0.3}, "B", id="bin-and-category"
    ),
    pytest.param(
        _aggregate_override_model, {"g1": 0.3, "A": 0.6}, "B", id="category-and-bin"
    ),
    pytest.param(_two_bin_model, {"A": 0.6}, "B", id="bin-with-remainder"),
    pytest.param(_two_bin_model, {"A": 0.4, "B": 0.6}, None, id="complete-bins"),
    pytest.param(_flat_category_model, {"g1": 0.3}, None, id="flat-category"),
    pytest.param(_shared_category_model, {"sex:y": 0.6}, None, id="qualified"),
    pytest.param(
        _shared_category_override_model,
        {"age:y": 0.6, "sex:o": 0.3},
        None,
        id="qualified-over-override",
    ),
    pytest.param(_mixed_replay_model, {"A": 0.5, "g1": 0.4}, "B", id="mixed"),
    pytest.param(
        _conditional_pathway_model,
        {"IH_oe60_hosp": 0.07},
        "S",
        id="composite-state",
    ),
    pytest.param(
        _conditional_pathway_model, {"home": 0.2}, None, id="conditional-category"
    ),
]


class TestWriteBackRebuildsTheScoredState:
    """The definition written back rebuilds the population the engine scored."""

    @pytest.mark.parametrize(("factory", "values", "remainder"), WRITE_BACK_CASES)
    def test_rebuild_matches_the_engine(self, factory, values, remainder) -> None:
        model = factory()
        problem = _initial_condition_problem(*values, remainder=remainder)
        expected = _scored(model, problem, values)

        model.apply_calibration_parameters(values, problem)

        assert _rebuilt(model) == pytest.approx(expected)

    @pytest.mark.parametrize(("factory", "values", "remainder"), WRITE_BACK_CASES)
    def test_definition_round_trips_through_json(
        self, factory, values, remainder
    ) -> None:
        model = factory()
        problem = _initial_condition_problem(*values, remainder=remainder)
        model.apply_calibration_parameters(values, problem)

        reloaded = Model.model_validate_json(model.model_dump_json())

        assert _rebuilt(reloaded) == pytest.approx(_rebuilt(model))


class TestWriteBackReadability:
    """Values the declared fractions express are written to those fractions."""

    def test_complete_bin_values_update_the_bin_fractions(self) -> None:
        model = _two_bin_model()

        model.apply_calibration_parameters(
            {"A": 0.4, "B": 0.6}, _initial_condition_problem("A", "B")
        )

        bins = model.population.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"A": 0.4, "B": 0.6}
        )
        assert _overrides(model) == {}

    def test_remainder_bin_takes_the_rest_of_the_population(self) -> None:
        model = _two_bin_model()

        model.apply_calibration_parameters(
            {"A": 0.7}, _initial_condition_problem("A", remainder="B")
        )

        bins = model.population.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"A": 0.7, "B": 0.3}
        )
        assert _overrides(model) == {}

    def test_category_value_updates_the_category_fraction(self) -> None:
        model = _flat_category_model()

        model.apply_calibration_parameters(
            {"g1": 0.3}, _initial_condition_problem("g1")
        )

        fractions = {
            f.category: f.fraction
            for s in model.population.initial_conditions.stratification_fractions
            for f in s.fractions
        }
        assert fractions == pytest.approx({"g1": 0.3, "g2": 0.7})
        assert _overrides(model) == {}

    def test_qualified_category_updates_its_own_stratification(self) -> None:
        model = _shared_category_model()

        model.apply_calibration_parameters(
            {"sex:y": 0.6}, _initial_condition_problem("sex:y")
        )

        fractions = {
            (s.stratification, f.category): f.fraction
            for s in model.population.initial_conditions.stratification_fractions
            for f in s.fractions
        }
        assert fractions == pytest.approx(
            {
                ("age", "y"): 0.5,
                ("age", "o"): 0.5,
                ("sex", "y"): 0.6,
                ("sex", "o"): 0.4,
            }
        )
        assert _overrides(model) == {}

    def test_single_composite_state_is_free_with_a_remainder(self) -> None:
        model = _conditional_pathway_model()

        model.apply_calibration_parameters(
            {"IH_oe60_hosp": 0.07},
            _initial_condition_problem("IH_oe60_hosp", remainder="S"),
        )

        # IH totals 0 + 25 + 70 = 95, so S takes the remaining 905 and keeps
        # its split; only the IH compartments need overrides.
        bins = model.population.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"S": 0.905, "IH": 0.095}
        )
        assert set(_overrides(model)) == {"IH_y60", "IH_oe60_home", "IH_oe60_hosp"}
        rebuilt = _rebuilt(model)
        assert rebuilt["IH_oe60_hosp"] == pytest.approx(70.0)
        assert rebuilt["IH_y60"] == 0.0
        assert sum(rebuilt.values()) == pytest.approx(1000.0)

    def test_without_remainder_values_are_applied_as_given(self) -> None:
        model = _flat_category_model()

        model.apply_calibration_parameters(
            {"A_g1": 0.2}, _initial_condition_problem("A_g1")
        )

        assert _rebuilt(model)["A_g1"] == pytest.approx(200.0)
        assert sum(_rebuilt(model).values()) == pytest.approx(200.0)

    def test_without_remainder_an_excess_is_rejected(self) -> None:
        model = _two_bin_model()
        before = model.model_dump_json()

        with pytest.raises(ValueError, match="exceed the population size"):
            model.apply_calibration_parameters(
                {"A": 0.6}, _initial_condition_problem("A")
            )

        assert model.model_dump_json() == before

    def test_applying_the_written_back_values_again_is_a_no_op(self) -> None:
        model = _two_bin_model()
        problem = _initial_condition_problem("A", remainder="B")
        model.apply_calibration_parameters({"A": 0.7}, problem)
        first = model.model_dump_json()

        model.apply_calibration_parameters({"A": 0.7}, problem)

        assert model.model_dump_json() == first


class TestSharedCategory:
    """A category name declared by two stratifications must be qualified."""

    @pytest.mark.parametrize(
        "factory", [_shared_category_model, _shared_category_override_model]
    )
    def test_bare_name_is_rejected_as_ambiguous(self, factory) -> None:
        model = factory()
        before = model.model_dump_json()

        with pytest.raises(ValueError, match="'age:y' or 'sex:y'"):
            model.apply_calibration_parameters(
                {"y": 0.6}, _initial_condition_problem("y")
            )

        assert model.model_dump_json() == before

    def test_qualified_id_conserves_the_population(self) -> None:
        model = _shared_category_model()

        model.apply_calibration_parameters(
            {"age:y": 0.6}, _initial_condition_problem("age:y")
        )

        assert sum(_rebuilt(model).values()) == pytest.approx(1000.0)

    def test_update_rejects_a_bare_ambiguous_category(self) -> None:
        model = _shared_category_model()

        with pytest.raises(ValueError, match="ambiguous"):
            model.update_stratification_fractions({"y": 0.6})


class TestCompositeStateCalibration:
    """End to end: calibrate the initial value of a composite state."""

    @staticmethod
    def _truth() -> Model:
        model = _conditional_pathway_model()
        model.update_compartment_fractions({"IH_oe60_hosp": 0.07})
        return model

    def test_calibrated_composite_state_is_written_back(self) -> None:
        observed = Simulation(self._truth()).run(10)
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(
                    step=step,
                    compartment="IH_oe60_hosp",
                    value=observed["IH_oe60_hosp"][step],
                )
                for step in range(1, 10)
            ],
            parameters=[
                CalibrationParameter(
                    id="IH_oe60_hosp",
                    parameter_type="initial_condition",
                    min_bound=0.0,
                    max_bound=0.1,
                ),
            ],
            initial_condition_remainder="S",
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=500),
        )
        model = _conditional_pathway_model()
        result = Calibrator(Simulation(model), problem).run()
        expected = _scored(model, problem, result.best_parameters)

        model.apply_calibration_parameters(result, problem)

        rebuilt = _rebuilt(model)
        assert rebuilt == pytest.approx(expected)
        assert rebuilt["IH_y60"] == 0.0
        assert rebuilt["IH_oe60_hosp"] == pytest.approx(70.0, abs=0.5)
        assert sum(rebuilt.values()) == pytest.approx(1000.0)
        assert result.best_parameters["IH_oe60_hosp"] == pytest.approx(0.07, abs=5e-4)


class TestCompartmentFractionValidation:
    def test_model_rejects_an_unknown_compartment_name(self) -> None:
        model = _aggregate_override_model()

        with pytest.raises(ValueError, match="Unknown compartment 'A_missing'"):
            model.update_compartment_fractions({"A_missing": 0.1})

    def test_initial_conditions_upsert_without_the_expansion(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[
                BinFraction(bin="A", fraction=0.6),
                BinFraction(bin="B", fraction=0.4),
            ],
            compartment_fractions=[CompartmentFraction(compartment="A", fraction=0.1)],
        )

        conditions.update_compartment_fractions({"A": 0.5, "B": 0.2})

        assert [
            (fraction.compartment, fraction.fraction)
            for fraction in conditions.compartment_fractions
        ] == [("A", 0.5), ("B", 0.2)]

    def test_initial_conditions_reject_an_out_of_range_upsert(self) -> None:
        conditions = InitialConditions(
            population_size=1000,
            bin_fractions=[BinFraction(bin="A", fraction=1.0)],
        )

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            conditions.update_compartment_fractions({"A": 1.5})

    def test_model_rejects_over_allocation(self) -> None:
        model = _aggregate_override_model()

        # Uniform A_g2 = 0.5, so an override of 0.9 on A_g1 totals 1.4.
        with pytest.raises(ValueError, match="exceed population_size"):
            model.update_compartment_fractions({"A_g1": 0.9})

        assert _overrides(model) == {"A_g1": 0.3}

    def test_model_allows_overrides_within_the_population(self) -> None:
        model = _aggregate_override_model()

        model.update_compartment_fractions({"A_g1": 0.4})

        assert _overrides(model) == pytest.approx({"A_g1": 0.4})

    def test_json_with_over_allocation_is_rejected(self) -> None:
        model = _aggregate_override_model()
        payload = model.model_dump_json().replace('"fraction":0.3}', '"fraction":0.9}')

        with pytest.raises(ValidationError, match="exceed population_size"):
            Model.model_validate_json(payload)


class TestApplyCalibrationAtomicity:
    def test_unresolvable_initial_condition_leaves_the_model_unchanged(self) -> None:
        model = _aggregate_override_model()
        problem = _initial_condition_problem("A", "missing")
        problem.parameters.append(
            CalibrationParameter(
                id="k", parameter_type="parameter", min_bound=0.0, max_bound=1.0
            )
        )
        before = model.model_dump_json()

        with pytest.raises(ValueError, match="missing"):
            model.apply_calibration_parameters(
                {"k": 0.5, "A": 0.4, "missing": 0.1}, problem
            )

        assert model.model_dump_json() == before


class TestUnknownCompartmentOverride:
    def test_building_a_simulation_raises_a_catchable_error(self) -> None:
        model = _aggregate_override_model()
        model.population.initial_conditions.compartment_fractions.append(
            CompartmentFraction(compartment="A_missing", fraction=0.0)
        )

        with pytest.raises(RuntimeError, match="A_missing") as excinfo:
            Simulation(model)

        assert isinstance(excinfo.value, Exception)


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

    def test_partial_update_must_keep_the_sum(self) -> None:
        conditions = self._conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.4})

    def test_rejected_update_leaves_the_bins_unchanged(self) -> None:
        conditions = self._conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.9})

        assert [(bf.bin, bf.fraction) for bf in conditions.bin_fractions] == [
            ("A", 0.5),
            ("B", 0.5),
        ]

    def test_updated_bins_always_reload(self) -> None:
        conditions = self._conditions()

        conditions.update_bin_fractions({"A": 0.4, "B": 0.6})

        reloaded = InitialConditions.model_validate_json(conditions.model_dump_json())
        assert [(bf.bin, bf.fraction) for bf in reloaded.bin_fractions] == [
            ("A", 0.4),
            ("B", 0.6),
        ]


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

    def test_initial_state_binding_reports_bins_and_categories(self) -> None:
        from commol.commol_rs import _commol_rs as commol_rs

        model = _conditional_pathway_model()
        rust_model = commol_rs.core.Model.from_json(model.model_dump_json())

        state = commol_rs.difference.initial_state(rust_model)

        assert state[0] == ("S_y60", "S", [("age", "y60")], 450.0)
        assert [name for name, _, _, _ in state] == Simulation(
            model
        ).engine.compartments


class TestEquationHelperParity:
    def test_category_less_condition_expands_like_the_engine(self) -> None:
        from commol.constants import PrintEquationsOutputFormat
        from commol.context._model.helpers import ModelCompartmentHelper
        from commol.context.stratification import Stratification
        from commol.context.stratification_condition import StratificationCondition

        model = _flat_category_model()
        # A category-less condition is permitted by the engine but rejected by
        # the Pydantic validator, so the stratification is built without it.
        model.population.stratifications.append(
            Stratification.model_construct(
                id="care",
                categories=["home", "hosp"],
                description=None,
                conditions=[
                    StratificationCondition.model_construct(
                        stratification="group", category=None, to=None
                    )
                ],
            )
        )
        helper = ModelCompartmentHelper(model)

        helper_names = [
            helper.compartment_to_string(compartment, PrintEquationsOutputFormat.TEXT)
            for compartment in helper.generate_compartments()
        ]

        assert helper_names == [c.name for c in model.initial_state()]
        assert "A_g2_hosp" in helper_names


class TestQualifiedCategoryReads:
    """Read methods resolve category ids like calibration does."""

    def test_bare_shared_name_is_ambiguous(self) -> None:
        model = _shared_category_model()

        with pytest.raises(ValueError, match="ambiguous"):
            model.subgroup_population(["y"])
        with pytest.raises(ValueError, match="ambiguous"):
            model.population.initial_conditions.get_category_fraction("y")

    def test_qualified_name_selects_its_stratification(self) -> None:
        model = _shared_category_model()
        model.update_stratification_fractions({"sex:y": 0.8})

        assert model.subgroup_population(["sex:y"]) == pytest.approx(800.0)
        assert model.subgroup_population(["age:y"]) == pytest.approx(500.0)

    def test_qualified_name_with_overrides_sums_the_engine_state(self) -> None:
        model = _shared_category_override_model()

        # A_y_y is pinned to 0, so only A_o_y holds sex:y.
        assert model.subgroup_population(["sex:y"]) == pytest.approx(250.0)
        assert model.subgroup_population(["age:o", "sex:y"]) == pytest.approx(250.0)

    def test_conditioning_categories_are_qualified_when_shared(self) -> None:
        model = (
            ModelBuilder(name="Conditioned", version="1.0")
            .add_bin(id="A", name="A")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="flow", source=["A"], target=[], rate="k * A")
            .add_stratification(id="age", categories=["y", "o"])
            .add_stratification(id="sex", categories=["y", "o"])
            .add_stratification(
                id="care",
                categories=["home", "hosp"],
                conditions=[{"stratification": "sex", "category": "y"}],
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[{"bin": "A", "fraction": 1.0}],
                stratification_fractions=[
                    {
                        "stratification": strat,
                        "fractions": [
                            {"category": first, "fraction": 0.5},
                            {"category": second, "fraction": 0.5},
                        ],
                    }
                    for strat, first, second in (
                        ("age", "y", "o"),
                        ("sex", "y", "o"),
                        ("care", "home", "hosp"),
                    )
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        assert model.get_conditioning_categories("care") == ("sex:y",)
        assert model.subgroup_population(("sex:y",)) == pytest.approx(500.0)


class TestBinPrefixes:
    """A bin id covers its own compartments, not those of a longer bin id."""

    @staticmethod
    def _model() -> Model:
        return (
            ModelBuilder(name="Prefixes", version="1.0")
            .add_bin(id="A", name="A")
            .add_bin(id="A_x", name="A x")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="flow", source=["A"], target=["A_x"], rate="k * A")
            .add_stratification(id="group", categories=["g1", "g2"])
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "A", "fraction": 0.5},
                    {"bin": "A_x", "fraction": 0.5},
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

    def test_remainder_bin_with_a_longer_id_is_not_an_overlap(self) -> None:
        model = self._model()

        model.apply_calibration_parameters(
            {"A": 0.3}, _initial_condition_problem("A", remainder="A_x")
        )

        bins = model.population.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"A": 0.3, "A_x": 0.7}
        )
        assert _overrides(model) == {}
        assert _rebuilt(model) == pytest.approx(
            {"A_g1": 150.0, "A_g2": 150.0, "A_x_g1": 350.0, "A_x_g2": 350.0}
        )


class TestEngineExpansion:
    def test_equation_helper_uses_the_engine_expansion(self) -> None:
        from commol.context._model.engine_state import expanded_outputs
        from commol.context._model.helpers import ModelCompartmentHelper

        model = _conditional_pathway_model()
        compartments, accumulators = expanded_outputs(model)
        helper = ModelCompartmentHelper(model)

        assert helper.generate_compartments() == [
            (output.source, output.categories) for output in compartments
        ]
        assert helper.generate_accumulator_outputs() == [
            (output.source, output.categories) for output in accumulators
        ]

    def test_difference_engine_rejects_a_differential_model(self) -> None:
        from commol.commol_rs import _commol_rs as commol_rs

        model = _conditional_pathway_model()
        payload = model.model_dump_json().replace(
            '"DifferenceEquations"', '"DifferentialEquations"'
        )
        rust_model = commol_rs.core.Model.from_json(payload)

        with pytest.raises(RuntimeError, match="cannot solve"):
            commol_rs.difference.DifferenceEquations(rust_model)
