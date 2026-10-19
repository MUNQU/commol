"""Writing calibrated values back to the model reproduces what was scored."""

import pytest

from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    ParticleSwarmConfig,
    SampleMethod,
    Simulation,
)
from commol.api.model_builder import ModelBuilder
from commol.context._model.engine_state import calibrated_initial_state
from commol.context.model import Model
from tests.support import (
    FactoryBuilder,
    a_to_b_builder,
    build,
    conditional_pathway_builder,
    initial_condition_problem,
    initial_state,
    override_builder,
    shared_category_builder,
    stratify,
)

SEED = 42


def _two_bins() -> ModelBuilder:
    return a_to_b_builder(rate="k * A", amounts={"A": 500, "B": 500})


def _flat_category() -> ModelBuilder:
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        stratifications={"group": {"g1": 1.0, "g2": 0.0}},
    )


def _shared_category_override() -> ModelBuilder:
    return shared_category_builder(compartment_amounts={"A_c1_c1": 0})


def _mixed_replay() -> ModelBuilder:
    """A shared category, a unique category and two bins."""
    return a_to_b_builder(
        rate="k * A",
        per_compartment=True,
        amounts={"A": 600, "B": 400},
        stratifications={
            "group": {"c1": 0.5, "c2": 0.5},
            "level": {"c1": 0.5, "c2": 0.5},
            "grp": {"g1": 0.5, "g2": 0.5},
        },
    )


def _prefixed_bins() -> ModelBuilder:
    """Bins A and A_x: the id of one begins with the id of the other."""
    builder = (
        FactoryBuilder(name="Prefixes", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="A_x", name="A x")
        .add_parameter(id="k", value=0.1)
        .add_transition(id="flow", source=["A"], target=["A_x"], rate="k * A")
    )
    return builder.set_structure(
        Aggregate(
            bin_amounts={"A": 500, "A_x": 500},
            stratification_fractions=stratify(
                builder, {"group": {"g1": 0.5, "g2": 0.5}}
            ),
        )
    )


def _scored(
    model: Model, problem: CalibrationProblem, values: dict[str, float]
) -> dict[str, float]:
    """State the engine scores for the values, before any write-back."""
    parameters = [
        param
        for param in problem.parameters
        if param.parameter_type == "initial_condition" and param.id in values
    ]
    return calibrated_initial_state(
        model,
        parameters,
        [values[param.id] for param in parameters],
        problem.initial_condition_remainder,
    ).state


def _overrides(model: Model) -> dict[str, float]:
    return dict(model.structure.compartment_amounts)


WRITE_BACK_CASES = [
    pytest.param(override_builder, {"A_g1": 200}, None, id="single-exact"),
    pytest.param(override_builder, {"A_g1": 200}, "B", id="single-exact-remainder"),
    pytest.param(override_builder, {"A_g1": 200, "A_g2": 600}, None, id="two-exact"),
    pytest.param(override_builder, {"A": 600}, None, id="bin-over-override"),
    pytest.param(override_builder, {"g1": 0.3}, None, id="category-over-override"),
    pytest.param(override_builder, {"A": 600, "g1": 0.3}, "B", id="bin-and-category"),
    pytest.param(override_builder, {"g1": 0.3, "A": 600}, "B", id="category-and-bin"),
    pytest.param(_two_bins, {"A": 600}, "B", id="bin-with-remainder"),
    pytest.param(_two_bins, {"A": 400, "B": 600}, None, id="complete-bins"),
    pytest.param(_flat_category, {"g1": 0.3}, None, id="flat-category"),
    pytest.param(_flat_category, {"g1": 0.3, "g2": 0.3}, None, id="both-categories"),
    pytest.param(shared_category_builder, {"level:c1": 0.6}, None, id="qualified"),
    pytest.param(
        _shared_category_override,
        {"group:c1": 0.6, "level:c2": 0.3},
        None,
        id="qualified-over-override",
    ),
    pytest.param(_mixed_replay, {"A": 500, "g1": 0.4}, "B", id="mixed"),
    pytest.param(
        conditional_pathway_builder,
        {"B_g2_t2": 70},
        "A",
        id="composite-state",
    ),
    pytest.param(
        conditional_pathway_builder, {"t1": 0.2}, None, id="conditional-category"
    ),
]


class TestWriteBackRebuildsTheScoredState:
    """The definition written back rebuilds the state the engine scored."""

    @pytest.mark.parametrize(("factory", "values", "remainder"), WRITE_BACK_CASES)
    def test_rebuild_matches_the_engine(self, factory, values, remainder) -> None:
        model = build(factory())
        problem = initial_condition_problem(*values, remainder=remainder)
        expected = _scored(model, problem, values)

        model.apply_calibration_parameters(values, problem)

        assert initial_state(model) == pytest.approx(expected)

    @pytest.mark.parametrize(("factory", "values", "remainder"), WRITE_BACK_CASES)
    def test_definition_round_trips_through_json(
        self, factory, values, remainder
    ) -> None:
        model = build(factory())
        problem = initial_condition_problem(*values, remainder=remainder)
        model.apply_calibration_parameters(values, problem)

        reloaded = Model.model_validate_json(model.model_dump_json())

        assert initial_state(reloaded) == pytest.approx(initial_state(model))


class TestWriteBackReadability:
    """Values the declared amounts and splits express are written to them."""

    def test_an_amount_in_the_unit_of_its_bin_is_written_back(self) -> None:
        model = (
            ModelBuilder(name="Mixed units")
            .add_bin(id="A", name="State A", unit="item")
            .add_bin(id="B", name="State B", unit="item")
            .add_bin(id="C", name="State C", unit="mg/l")
            .add_parameter(id="k", value=0.1)
            .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
            .build(
                Aggregate(bin_amounts={"A": 990, "B": 10, "C": None}),
                engine="DifferenceEquations",
            )
        )

        model.apply_calibration_parameters({"C": 2.5e5}, initial_condition_problem("C"))

        assert model.structure.bin_amounts == {"A": 990, "B": 10, "C": 2.5e5}
        assert _overrides(model) == {}
        assert initial_state(model) == {"A": 990, "B": 10, "C": 2.5e5}

    def test_complete_bin_values_update_the_bin_amounts(self) -> None:
        model = build(_two_bins())

        model.apply_calibration_parameters(
            {"A": 400, "B": 600}, initial_condition_problem("A", "B")
        )

        assert model.structure.bin_amounts == pytest.approx({"A": 400, "B": 600})
        assert _overrides(model) == {}

    def test_remainder_bin_takes_what_the_others_leave(self) -> None:
        model = build(_two_bins())

        model.apply_calibration_parameters(
            {"A": 700}, initial_condition_problem("A", remainder="B")
        )

        assert model.structure.bin_amounts == pytest.approx({"A": 700, "B": 300})
        assert _overrides(model) == {}

    def test_category_value_updates_the_category_fraction(self) -> None:
        model = build(_flat_category())

        model.apply_calibration_parameters({"g1": 0.3}, initial_condition_problem("g1"))

        fractions = {
            category: fraction
            for categories in model.structure.stratification_fractions.values()
            for category, fraction in categories.items()
        }
        assert fractions == pytest.approx({"g1": 0.3, "g2": 0.7})
        assert _overrides(model) == {}

    def test_qualified_category_updates_its_own_stratification(self) -> None:
        model = build(shared_category_builder())

        model.apply_calibration_parameters(
            {"level:c1": 0.6}, initial_condition_problem("level:c1")
        )

        fractions = {
            (stratification, category): fraction
            for stratification, categories in (
                model.structure.stratification_fractions.items()
            )
            for category, fraction in categories.items()
        }
        assert fractions == pytest.approx(
            {
                ("group", "c1"): 0.5,
                ("group", "c2"): 0.5,
                ("level", "c1"): 0.6,
                ("level", "c2"): 0.4,
            }
        )
        assert _overrides(model) == {}

    def test_single_composite_state_is_free_with_a_remainder(self) -> None:
        model = build(conditional_pathway_builder())

        model.apply_calibration_parameters(
            {"B_g2_t2": 70},
            initial_condition_problem("B_g2_t2", remainder="A"),
        )

        # A and B_g2_t2 keep their declared 900 + 25, so A takes 925 - 70 =
        # 855 and keeps its split. B keeps its amount, and only the
        # compartment the value sets joins the override of B_g1.
        assert model.structure.bin_amounts == pytest.approx({"A": 855, "B": 100})
        assert set(_overrides(model)) == {"B_g1", "B_g2_t2"}
        rebuilt = initial_state(model)
        assert rebuilt["B_g2_t2"] == pytest.approx(70.0)
        assert rebuilt["B_g1"] == 0.0
        assert sum(rebuilt.values()) == pytest.approx(950.0)

    def test_a_compartment_value_leaves_a_category_value_to_its_fraction(
        self,
    ) -> None:
        model = build(
            a_to_b_builder(
                rate="k * A",
                per_compartment=True,
                amounts={"A": 600, "B": 400},
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            )
        )
        values = {"B_g1": 100, "g1": 0.3}

        model.apply_calibration_parameters(values, initial_condition_problem(*values))

        assert model.structure.stratification_fractions == {
            "group": {"g1": 0.3, "g2": 0.7}
        }
        assert _overrides(model) == {"B_g1": 100}

    def test_without_remainder_values_are_applied_as_given(self) -> None:
        model = build(_flat_category())

        model.apply_calibration_parameters(
            {"A_g1": 200}, initial_condition_problem("A_g1")
        )

        assert initial_state(model)["A_g1"] == pytest.approx(200.0)
        assert sum(initial_state(model).values()) == pytest.approx(200.0)

    def test_without_remainder_the_other_bins_keep_their_amounts(self) -> None:
        model = build(_two_bins())

        model.apply_calibration_parameters({"A": 600}, initial_condition_problem("A"))

        assert model.structure.bin_amounts == {"A": 600, "B": 500}

    def test_a_remainder_left_below_zero_leaves_the_model_unchanged(self) -> None:
        model = build(_two_bins())
        before = model.model_dump_json()

        with pytest.raises(ValueError, match="remainder 'B' keeps by 0.166667"):
            model.apply_calibration_parameters(
                {"A": 1200}, initial_condition_problem("A", remainder="B")
            )

        assert model.model_dump_json() == before

    def test_applying_the_written_back_values_again_is_a_no_op(self) -> None:
        model = build(_two_bins())
        problem = initial_condition_problem("A", remainder="B")
        model.apply_calibration_parameters({"A": 700}, problem)
        first = model.model_dump_json()

        model.apply_calibration_parameters({"A": 700}, problem)

        assert model.model_dump_json() == first

    def test_a_remainder_bin_with_a_longer_id_is_not_an_overlap(self) -> None:
        model = build(_prefixed_bins())

        model.apply_calibration_parameters(
            {"A": 300}, initial_condition_problem("A", remainder="A_x")
        )

        assert model.structure.bin_amounts == pytest.approx({"A": 300, "A_x": 700})
        assert _overrides(model) == {}
        assert initial_state(model) == pytest.approx(
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

        with pytest.raises(ValueError, match="'group:c1' or 'level:c1'"):
            model.apply_calibration_parameters(
                {"c1": 0.6}, initial_condition_problem("c1")
            )

        assert model.model_dump_json() == before

    def test_qualified_id_keeps_the_sum_of_the_bin(self) -> None:
        model = build(shared_category_builder())

        model.apply_calibration_parameters(
            {"group:c1": 0.6}, initial_condition_problem("group:c1")
        )

        assert sum(initial_state(model).values()) == pytest.approx(1000.0)


class TestAtomicity:
    @pytest.mark.parametrize(("factory", "values", "remainder"), WRITE_BACK_CASES)
    def test_the_written_back_structure_is_a_copy(
        self, factory, values, remainder
    ) -> None:
        model = build(factory())
        before = model.structure.model_copy(deep=True)
        problem = initial_condition_problem(*values, remainder=remainder)

        written = model.structure.with_calibrated_values(model, problem, values)

        assert model.structure == before
        assert written != before

    def test_no_initial_condition_value_gives_an_independent_copy(self) -> None:
        model = build(override_builder())

        written = model.structure.with_calibrated_values(
            model, initial_condition_problem("A_g1"), {}
        )
        written.compartment_amounts["A_g1"] = 100

        assert written.bin_amounts == model.structure.bin_amounts
        assert model.structure.compartment_amounts == {"A_g1": 300}

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
                {"k": 0.5, "A": 400, "missing": 100}, problem
            )

        assert model.model_dump_json() == before


class TestCompositeStateCalibration:
    """End to end: calibrate the initial value of a composite state."""

    def test_calibrated_composite_state_is_written_back(self) -> None:
        truth = build(conditional_pathway_builder())
        # The remainder A keeps the declared 900 + 25 of A and B_g2_t2.
        truth.structure.update_compartment_amounts(truth.process, {"B_g2_t2": 70})
        truth.structure.update_bin_amounts({"A": 855})
        observed = Simulation(truth).run(10)
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(
                    at=step,
                    output="B_g2_t2",
                    value=observed["B_g2_t2"][step],
                )
                for step in range(1, 10)
            ],
            parameters=[
                CalibrationParameter(
                    id="B_g2_t2",
                    parameter_type="initial_condition",
                    min_bound=0.0,
                    max_bound=100.0,
                ),
            ],
            initial_condition_remainder="A",
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=500),
        )
        model = build(conditional_pathway_builder())
        result = Calibrator(Simulation(model), problem).run()
        expected = _scored(model, problem, result.best_parameters)

        model.apply_calibration_parameters(result, problem)

        rebuilt = initial_state(model)
        assert rebuilt == pytest.approx(expected)
        assert rebuilt["B_g1"] == 0.0
        assert rebuilt["B_g2_t2"] == pytest.approx(70.0, abs=0.5)
        assert sum(rebuilt.values()) == pytest.approx(950.0)
        assert result.best_parameters["B_g2_t2"] == pytest.approx(70, abs=0.5)


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
            category: fraction
            for categories in model.structure.stratification_fractions.values()
            for category, fraction in categories.items()
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
        group = model.structure.stratification_fractions["group"]
        assert group["group1"] == pytest.approx(0.4)

    def test_a_sample_is_rejected_for_one_of_its_draws(self) -> None:
        model = _binary_split()
        sample = ParameterSample.equally_weighted(
            ["k1"], [(0.0, 1.0)], [[0.1], [0.2]], SampleMethod(name="test")
        )

        with pytest.raises(TypeError) as error:
            model.apply_calibration_parameters(sample, _split_problem())  # ty: ignore[invalid-argument-type]

        assert str(error.value) == (
            "A ParameterSample has no single set of parameters; pass one of "
            "sample.draws() instead."
        )

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

    def test_per_compartment_initial_condition_is_upserted(self) -> None:
        model = _binary_split()
        problem = initial_condition_problem("A_group1", "A_group2")

        model.apply_calibration_parameters({"A_group1": 300, "A_group2": 700}, problem)
        model.apply_calibration_parameters({"A_group1": 400, "A_group2": 600}, problem)

        overrides = list(model.structure.compartment_amounts.items())
        assert overrides == [
            ("A_group1", pytest.approx(400)),
            ("A_group2", pytest.approx(600)),
        ]
