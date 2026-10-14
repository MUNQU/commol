"""Writing calibrated values back to the model reproduces what was scored."""

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.api.model_builder import ModelBuilder
from commol.context._model.engine_state import calibrated_initial_population
from commol.context.model import Model
from tests.support import (
    a_to_b_builder,
    build,
    conditional_pathway_builder,
    initial_condition_problem,
    initial_population,
    override_builder,
    shared_category_builder,
    stratify,
)

SEED = 42


def _two_bins() -> ModelBuilder:
    return a_to_b_builder(rate="k * A", fractions={"A": 0.5, "B": 0.5})


def _flat_category() -> ModelBuilder:
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 1.0, "g2": 0.0}},
    )


def _shared_category_override() -> ModelBuilder:
    return shared_category_builder(
        compartment_fractions=[{"compartment": "A_y_y", "fraction": 0.0}]
    )


def _mixed_replay() -> ModelBuilder:
    """A shared category, a unique category and two bins."""
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        fractions={"A": 0.6, "B": 0.4},
        stratifications={
            "age": {"y": 0.5, "o": 0.5},
            "sex": {"y": 0.5, "o": 0.5},
            "grp": {"g1": 0.5, "g2": 0.5},
        },
    )


def _prefixed_bins() -> ModelBuilder:
    """Bins A and A_x: the id of one begins with the id of the other."""
    builder = (
        ModelBuilder(name="Prefixes", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="A_x", name="A x")
        .add_parameter(id="k", value=0.1)
        .add_transition(id="flow", source=["A"], target=["A_x"], rate="k * A")
    )
    return builder.set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.5},
            {"bin": "A_x", "fraction": 0.5},
        ],
        stratification_fractions=stratify(builder, {"group": {"g1": 0.5, "g2": 0.5}}),
    )


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
        for fraction in model.structure.initial_conditions.compartment_fractions
    }


WRITE_BACK_CASES = [
    pytest.param(override_builder, {"A_g1": 0.2}, None, id="single-exact"),
    pytest.param(override_builder, {"A_g1": 0.2}, "B", id="single-exact-remainder"),
    pytest.param(override_builder, {"A_g1": 0.2, "A_g2": 0.6}, None, id="two-exact"),
    pytest.param(override_builder, {"A": 0.6}, None, id="bin-over-override"),
    pytest.param(override_builder, {"g1": 0.3}, None, id="category-over-override"),
    pytest.param(override_builder, {"A": 0.6, "g1": 0.3}, "B", id="bin-and-category"),
    pytest.param(override_builder, {"g1": 0.3, "A": 0.6}, "B", id="category-and-bin"),
    pytest.param(_two_bins, {"A": 0.6}, "B", id="bin-with-remainder"),
    pytest.param(_two_bins, {"A": 0.4, "B": 0.6}, None, id="complete-bins"),
    pytest.param(_flat_category, {"g1": 0.3}, None, id="flat-category"),
    pytest.param(shared_category_builder, {"sex:y": 0.6}, None, id="qualified"),
    pytest.param(
        _shared_category_override,
        {"age:y": 0.6, "sex:o": 0.3},
        None,
        id="qualified-over-override",
    ),
    pytest.param(_mixed_replay, {"A": 0.5, "g1": 0.4}, "B", id="mixed"),
    pytest.param(
        conditional_pathway_builder,
        {"IH_oe60_hosp": 0.07},
        "S",
        id="composite-state",
    ),
    pytest.param(
        conditional_pathway_builder, {"home": 0.2}, None, id="conditional-category"
    ),
]


class TestWriteBackRebuildsTheScoredState:
    """The definition written back rebuilds the population the engine scored."""

    @pytest.mark.parametrize(("factory", "values", "remainder"), WRITE_BACK_CASES)
    def test_rebuild_matches_the_engine(self, factory, values, remainder) -> None:
        model = build(factory())
        problem = initial_condition_problem(*values, remainder=remainder)
        expected = _scored(model, problem, values)

        model.apply_calibration_parameters(values, problem)

        assert initial_population(model) == pytest.approx(expected)

    @pytest.mark.parametrize(("factory", "values", "remainder"), WRITE_BACK_CASES)
    def test_definition_round_trips_through_json(
        self, factory, values, remainder
    ) -> None:
        model = build(factory())
        problem = initial_condition_problem(*values, remainder=remainder)
        model.apply_calibration_parameters(values, problem)

        reloaded = Model.model_validate_json(model.model_dump_json())

        assert initial_population(reloaded) == pytest.approx(initial_population(model))


class TestWriteBackReadability:
    """Values the declared fractions express are written to those fractions."""

    def test_complete_bin_values_update_the_bin_fractions(self) -> None:
        model = build(_two_bins())

        model.apply_calibration_parameters(
            {"A": 0.4, "B": 0.6}, initial_condition_problem("A", "B")
        )

        bins = model.structure.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"A": 0.4, "B": 0.6}
        )
        assert _overrides(model) == {}

    def test_remainder_bin_takes_the_rest_of_the_population(self) -> None:
        model = build(_two_bins())

        model.apply_calibration_parameters(
            {"A": 0.7}, initial_condition_problem("A", remainder="B")
        )

        bins = model.structure.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"A": 0.7, "B": 0.3}
        )
        assert _overrides(model) == {}

    def test_category_value_updates_the_category_fraction(self) -> None:
        model = build(_flat_category())

        model.apply_calibration_parameters({"g1": 0.3}, initial_condition_problem("g1"))

        fractions = {
            f.category: f.fraction
            for s in model.structure.initial_conditions.stratification_fractions
            for f in s.fractions
        }
        assert fractions == pytest.approx({"g1": 0.3, "g2": 0.7})
        assert _overrides(model) == {}

    def test_qualified_category_updates_its_own_stratification(self) -> None:
        model = build(shared_category_builder())

        model.apply_calibration_parameters(
            {"sex:y": 0.6}, initial_condition_problem("sex:y")
        )

        fractions = {
            (s.stratification, f.category): f.fraction
            for s in model.structure.initial_conditions.stratification_fractions
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
        model = build(conditional_pathway_builder())

        model.apply_calibration_parameters(
            {"IH_oe60_hosp": 0.07},
            initial_condition_problem("IH_oe60_hosp", remainder="S"),
        )

        # IH totals 0 + 25 + 70 = 95, so S takes the remaining 905 and keeps
        # its split; only the IH compartments need overrides.
        bins = model.structure.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"S": 0.905, "IH": 0.095}
        )
        assert set(_overrides(model)) == {"IH_y60", "IH_oe60_home", "IH_oe60_hosp"}
        rebuilt = initial_population(model)
        assert rebuilt["IH_oe60_hosp"] == pytest.approx(70.0)
        assert rebuilt["IH_y60"] == 0.0
        assert sum(rebuilt.values()) == pytest.approx(1000.0)

    def test_without_remainder_values_are_applied_as_given(self) -> None:
        model = build(_flat_category())

        model.apply_calibration_parameters(
            {"A_g1": 0.2}, initial_condition_problem("A_g1")
        )

        assert initial_population(model)["A_g1"] == pytest.approx(200.0)
        assert sum(initial_population(model).values()) == pytest.approx(200.0)

    def test_without_remainder_an_excess_is_rejected(self) -> None:
        model = build(_two_bins())
        before = model.model_dump_json()

        with pytest.raises(ValueError, match="exceed the population size"):
            model.apply_calibration_parameters(
                {"A": 0.6}, initial_condition_problem("A")
            )

        assert model.model_dump_json() == before

    def test_applying_the_written_back_values_again_is_a_no_op(self) -> None:
        model = build(_two_bins())
        problem = initial_condition_problem("A", remainder="B")
        model.apply_calibration_parameters({"A": 0.7}, problem)
        first = model.model_dump_json()

        model.apply_calibration_parameters({"A": 0.7}, problem)

        assert model.model_dump_json() == first

    def test_a_remainder_bin_with_a_longer_id_is_not_an_overlap(self) -> None:
        model = build(_prefixed_bins())

        model.apply_calibration_parameters(
            {"A": 0.3}, initial_condition_problem("A", remainder="A_x")
        )

        bins = model.structure.initial_conditions.bin_fractions
        assert {bf.bin: bf.fraction for bf in bins} == pytest.approx(
            {"A": 0.3, "A_x": 0.7}
        )
        assert _overrides(model) == {}
        assert initial_population(model) == pytest.approx(
            {"A_g1": 150.0, "A_g2": 150.0, "A_x_g1": 350.0, "A_x_g2": 350.0}
        )


class TestSharedCategory:
    """A category name declared by two stratifications must be qualified."""

    @pytest.mark.parametrize(
        "factory", [shared_category_builder, _shared_category_override]
    )
    def test_bare_name_is_rejected_as_ambiguous(self, factory) -> None:
        model = build(factory())
        before = model.model_dump_json()

        with pytest.raises(ValueError, match="'age:y' or 'sex:y'"):
            model.apply_calibration_parameters(
                {"y": 0.6}, initial_condition_problem("y")
            )

        assert model.model_dump_json() == before

    def test_qualified_id_conserves_the_population(self) -> None:
        model = build(shared_category_builder())

        model.apply_calibration_parameters(
            {"age:y": 0.6}, initial_condition_problem("age:y")
        )

        assert sum(initial_population(model).values()) == pytest.approx(1000.0)


class TestAtomicity:
    def test_unresolvable_initial_condition_leaves_the_model_unchanged(self) -> None:
        model = build(override_builder())
        problem = initial_condition_problem("A", "missing")
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


class TestCompositeStateCalibration:
    """End to end: calibrate the initial value of a composite state."""

    def test_calibrated_composite_state_is_written_back(self) -> None:
        truth = build(conditional_pathway_builder())
        truth.structure.update_compartment_fractions({"IH_oe60_hosp": 0.07})
        observed = Simulation(truth).run(10)
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(
                    at=step,
                    output="IH_oe60_hosp",
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
        model = build(conditional_pathway_builder())
        result = Calibrator(Simulation(model), problem).run()
        expected = _scored(model, problem, result.best_parameters)

        model.apply_calibration_parameters(result, problem)

        rebuilt = initial_population(model)
        assert rebuilt == pytest.approx(expected)
        assert rebuilt["IH_y60"] == 0.0
        assert rebuilt["IH_oe60_hosp"] == pytest.approx(70.0, abs=0.5)
        assert sum(rebuilt.values()) == pytest.approx(1000.0)
        assert result.best_parameters["IH_oe60_hosp"] == pytest.approx(0.07, abs=5e-4)


def _binary_split(group1_fraction: float = 0.5) -> Model:
    """A binary stratification whose split can itself be calibrated."""
    return build(
        a_to_b_builder(
            parameters={"k1": None},
            rate="k1 * A",
            per_compartment=True,
            stratifications={
                "group": {"group1": group1_fraction, "group2": 1.0 - group1_fraction}
            },
        )
    )


def _split_problem() -> CalibrationProblem:
    truth = _binary_split(0.3)
    truth.update_parameters({"k1": 0.2})
    observed_series = Simulation(truth).run(20)
    return CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=step, output=output, value=values[step])
            for output, values in observed_series.items()
            for step in range(1, 20)
        ],
        parameters=[
            CalibrationParameter(
                id="k1", parameter_type="parameter", min_bound=0.01, max_bound=0.5
            ),
            CalibrationParameter(
                id="group1",
                parameter_type="initial_condition",
                min_bound=0.05,
                max_bound=0.95,
            ),
        ],
        loss_function="sse",
        optimization_config=ParticleSwarmConfig(num_particles=30, max_iterations=150),
        seed=SEED,
    )


class TestApplyCalibrationParameters:
    """
    Writing a fitted stratification split back onto the model.

    The engine applies a calibrated binary category by redistributing each
    affected compartment pair as ``value`` and ``1 - value`` of the pair total.
    The Python model instead stores the pair as two category fractions. These
    are two expressions of one rule, in two languages, and they have to agree:
    if they do not, the saved model does not reproduce the fit that was scored.
    """

    def test_applied_model_reproduces_the_loss_the_optimizer_scored(self) -> None:
        model = _binary_split()
        problem = _split_problem()

        result = Calibrator(Simulation(model), problem).run()
        model.apply_calibration_parameters(result, problem)

        replayed = Simulation(model).run(20)
        recomputed_loss = sum(
            (replayed[point.label][int(point.at)] - point.value) ** 2
            for point in problem.observed_data
        )

        assert recomputed_loss == pytest.approx(result.final_loss, rel=1e-9)

    def test_the_fitted_split_lands_in_the_model(self) -> None:
        model = _binary_split()
        problem = _split_problem()

        result = Calibrator(Simulation(model), problem).run()
        model.apply_calibration_parameters(result, problem)

        fractions = {
            f.category: f.fraction
            for s in model.structure.initial_conditions.stratification_fractions
            for f in s.fractions
        }
        assert fractions["group1"] == pytest.approx(result.best_parameters["group1"])
        assert fractions["group2"] == pytest.approx(
            1.0 - result.best_parameters["group1"]
        )
        assert model.parameters[0].value == pytest.approx(result.best_parameters["k1"])

    def test_a_bare_mapping_is_accepted(self) -> None:
        model = _binary_split()

        model.apply_calibration_parameters(
            {"k1": 0.15, "group1": 0.4}, _split_problem()
        )

        assert model.parameters[0].value == pytest.approx(0.15)
        group = model.structure.initial_conditions.stratification_fractions[0]
        assert group.fractions[0].fraction == pytest.approx(0.4)

    def test_scale_parameters_are_ignored(self) -> None:
        model = _binary_split()
        problem = _split_problem()
        problem.parameters.append(
            CalibrationParameter(
                id="reporting_rate",
                parameter_type="factor",
                min_bound=0.01,
                max_bound=1.0,
            )
        )

        model.apply_calibration_parameters({"k1": 0.15, "reporting_rate": 0.5}, problem)

        assert model.parameters[0].value == pytest.approx(0.15)
        assert [p.id for p in model.parameters] == ["k1"]

    def test_undeclared_id_is_rejected(self) -> None:
        model = _binary_split()

        with pytest.raises(ValueError, match="no declared parameter type"):
            model.apply_calibration_parameters({"mystery": 0.15}, _split_problem())

    def test_per_compartment_initial_condition_is_written_back(self) -> None:
        model = _binary_split()
        problem = initial_condition_problem("A_group1", "A_group2")

        model.apply_calibration_parameters({"A_group1": 0.3, "A_group2": 0.7}, problem)

        assert _overrides(model) == {
            "A_group1": pytest.approx(0.3),
            "A_group2": pytest.approx(0.7),
        }

    def test_per_compartment_initial_condition_is_upserted(self) -> None:
        model = _binary_split()
        problem = initial_condition_problem("A_group1", "A_group2")

        model.apply_calibration_parameters({"A_group1": 0.3, "A_group2": 0.7}, problem)
        model.apply_calibration_parameters({"A_group1": 0.4, "A_group2": 0.6}, problem)

        overrides = [
            (fraction.compartment, fraction.fraction)
            for fraction in model.structure.initial_conditions.compartment_fractions
        ]
        assert overrides == [
            ("A_group1", pytest.approx(0.4)),
            ("A_group2", pytest.approx(0.6)),
        ]
