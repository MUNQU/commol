"""Calibrating initial conditions: which ids name them, and fitting them."""

import pytest

from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)
from commol._model.engine_state import calibrated_initial_state
from commol._model.model import Model
from commol.uncertainty import ParameterSample, Predictive, SampleMethod
from tests.support import (
    a_to_b_builder,
    abc_builder,
    assert_accepted,
    bounded,
    build,
    initial_condition_problem,
    initial_state,
    mixed_units_builder,
    observed,
)

SEED = 42


def _model() -> Model:
    """A binary and a ternary stratification."""
    return build(
        a_to_b_builder(
            name="Ids",
            amounts={"A": 900, "B": 100},
            stratifications={
                "group": {"g1": 0.5, "g2": 0.5},
                "age": {"y": 1 / 3, "m": 1 / 3, "o": 1 / 3},
            },
        )
    )


def _problem(parameter_id: str) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="B_g1_y", value=10.0)],
        parameters=[bounded(parameter_id, 0.0, 0.5, kind="initial_condition")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=5),
    )


class TestIds:
    """Initial-condition ids are resolved by the engine when a calibrator is built."""

    @pytest.mark.parametrize("parameter_id", ["y", "age:y"])
    def test_a_category_of_a_non_binary_stratification_is_rejected(
        self, parameter_id: str
    ) -> None:
        with pytest.raises(ValueError, match="needs a binary stratification"):
            Calibrator(Simulation(_model()), _problem(parameter_id))

    @pytest.mark.parametrize("parameter_id", ["B", "B_g1", "B_g1_y", "g1", "group:g1"])
    def test_every_documented_id_kind_is_accepted(self, parameter_id: str) -> None:
        assert_accepted(Calibrator, Simulation(_model()), _problem(parameter_id))

    def test_an_unknown_id_is_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="Invalid bin, compartment or category 'X'"
        ):
            Calibrator(Simulation(_model()), _problem("X"))

    def test_an_unknown_bin_of_an_unstratified_model_is_rejected(self) -> None:
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=10, output="B", value=50.0)],
            parameters=[bounded("X", 0.0, 100.0, kind="initial_condition")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=100),
        )

        with pytest.raises(ValueError, match="Invalid bin, compartment or category .X"):
            Calibrator(Simulation(build(abc_builder())), problem)

    @pytest.mark.parametrize(
        ("parameter_id", "missing"),
        [
            ("B_g1", r'the amounts of the compartments \["B_g2"\]'),
            ("g1", r'the amounts of the bins \["B"\]'),
            ("group:g1", r'the amounts of the bins \["B"\]'),
        ],
    )
    def test_a_bin_left_to_calibrate_needs_a_value_in_each_compartment(
        self, parameter_id: str, missing: str
    ) -> None:
        """A split of a category shares out what a bin holds, so it gives no value."""
        model = build(
            a_to_b_builder(
                amounts={"A": 1000, "B": None},
                stratifications={"group": {"g1": 0.5, "g2": 0.5}},
            )
        )
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=1, output="B_g1", value=10.0)],
            parameters=[bounded(parameter_id, 0.0, 1.0, kind="initial_condition")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=5),
        )

        with pytest.raises(ValueError, match=f"^The model cannot run: {missing}"):
            Calibrator(Simulation(model), problem)


def _abc(b_amount: float | None, k1: float | None = 0.3) -> Model:
    return build(
        abc_builder(
            parameters={"k1": k1, "k2": 0.1},
            amounts={"A": 980, "B": b_amount, "C": 0},
        )
    )


@pytest.fixture
def b_every_5_steps() -> list[ObservedDataPoint]:
    """B of ``abc_builder`` with 20 in B at the start, at steps 0, 5, …, 45."""
    return observed(Simulation(_abc(20)).run(50), "B", range(0, 50, 5))


class TestFitting:
    def test_an_initial_amount_is_recovered(self, b_every_5_steps) -> None:
        model = _abc(None)
        problem = CalibrationProblem(
            observed_data=b_every_5_steps,
            parameters=[
                bounded("B", 0.0, 100.0, kind="initial_condition", initial_guess=10.0)
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(
                max_iterations=5000, sd_tolerance=1e-9, verbose=False
            ),
        )

        result = Calibrator(Simulation(model), problem).run()

        assert result.converged
        assert result.best_parameters["B"] == pytest.approx(20, rel=1e-9)

    def test_a_parameter_and_an_initial_amount_together(self, b_every_5_steps) -> None:
        config = ParticleSwarmConfig(
            particles=40,
            max_iterations=1000,
            verbose=False,
            acceleration="time_varying",
            cognitive_initial=2.5,
            cognitive_final=0.5,
            social_initial=0.5,
            social_final=2.5,
            velocity_clamp_factor=0.2,
            mutation="gaussian",
            mutation_scale=0.1,
            mutation_probability=0.05,
            mutation_application="global_best",
        )
        problem = CalibrationProblem(
            observed_data=b_every_5_steps,
            parameters=[
                bounded("k1"),
                bounded("B", 0.0, 100.0, kind="initial_condition"),
            ],
            loss_function="sse",
            optimization_config=config,
            seed=SEED,
        )

        result = Calibrator(Simulation(_abc(None, k1=None)), problem).run()

        assert result.best_parameters["k1"] == pytest.approx(0.3, abs=0.001)
        assert result.best_parameters["B"] == pytest.approx(20, rel=0.05)


def _remainder_problem(parameter_id: str) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="B", value=20.0)],
        parameters=[bounded(parameter_id, 0.0, 1e6, kind="initial_condition")],
        initial_condition_remainder="A",
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=5),
    )


class TestRemainderUnits:
    def test_a_remainder_of_another_unit_is_rejected(self) -> None:
        with pytest.raises(ValueError) as error:
            Calibrator(
                Simulation(build(mixed_units_builder())), _remainder_problem("C")
            )

        assert str(error.value) == (
            "The initial-condition remainder 'A' is in 'item', but the "
            "calibrated initial condition 'C' sets 'C', in 'mg/l'. The "
            "remainder keeps a sum of amounts of one unit: name a remainder of "
            "the unit of 'C'."
        )

    def test_an_error_of_the_values_has_no_prefix(self) -> None:
        problem = _remainder_problem("B").model_copy(
            update={"initial_condition_remainder": "X"}
        )

        with pytest.raises(ValueError) as error:
            Calibrator(Simulation(build(abc_builder())), problem)

        assert str(error.value) == (
            "Initial-condition remainder 'X' is not a compartment or a bin "
            "(available: A, B, C)"
        )

    def test_a_remainder_of_the_same_unit_is_accepted(self) -> None:
        assert_accepted(
            Calibrator,
            Simulation(build(mixed_units_builder())),
            _remainder_problem("B"),
        )


def _moving_split() -> Model:
    """``k = 0.1`` of ``A_g1`` moves to ``A_g2`` each step; 100 in A, 900 in B."""
    return (
        ModelBuilder(name="Moving split")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_stratification(id="g", categories=["g1", "g2"])
        .add_parameter(id="k", value=0.1)
        .add_transition(
            id="move",
            source=["A"],
            target=["A"],
            stratified_rates=[
                StratifiedRate(
                    conditions=[
                        StratificationCondition(
                            stratification="g", category="g1", to="g2"
                        )
                    ],
                    rate="k * A_g1",
                )
            ],
        )
        .build(
            Aggregate(
                bin_amounts={"A": 100, "B": 900},
                stratification_fractions={"g": {"g1": 0.5, "g2": 0.5}},
            ),
            engine="DifferenceEquations",
        )
    )


def test_a_calibrator_on_a_simulation_that_has_run_splits_a_bin_by_the_initial_state():  # noqa: E501
    # A = 100 splits 50/50, and 0.1 of the 50 in A_g1 moves in one step: 45.
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="A_g1", value=45.0)],
        parameters=[bounded("A", 0.0, 200.0, kind="initial_condition")],
        initial_condition_remainder="B",
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=10),
    )
    sample = ParameterSample.equally_weighted(
        names=["A"],
        bounds=[(0.0, 200.0)],
        values=[[100.0]],
        method=SampleMethod(name="hand"),
    )
    simulation = Simulation(_moving_split())
    simulation.run(num_steps=30)
    calibrator = Calibrator(simulation, problem)

    start = Predictive(calibrator, sample, positions=[0]).runs()[0]
    result = calibrator.run()

    assert (start["A_g1"][0], start["A_g2"][0]) == pytest.approx((50.0, 50.0))
    assert result.best_parameters["A"] == pytest.approx(100.0, abs=1e-6)
    assert result.final_loss == pytest.approx(0.0, abs=1e-9)


def _split_by_group(a_amount: float | None) -> Model:
    """``group`` splits A 0.4 / 0.6; ``type`` splits ``g1`` 0.25 / 0.75 and
    ``g2`` 0.5 / 0.5; B starts at 0."""
    builder = a_to_b_builder()
    builder.add_stratification(id="group", categories=["g1", "g2"])
    builder.add_stratification(id="type", categories=["t1", "t2"])
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": a_amount, "B": 0},
            stratification_fractions={
                "group": {"g1": 0.4, "g2": 0.6},
                "type": {
                    "by": "group",
                    "g1": {"t1": 0.25, "t2": 0.75},
                    "g2": {"t1": 0.5, "t2": 0.5},
                },
            },
        )
    )
    return build(builder)


def test_a_calibrated_amount_keeps_the_split_by_category() -> None:
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="B_g1_t1", value=10.0)],
        parameters=[bounded("A", 0.0, 4000.0, kind="initial_condition")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=5),
    )
    sample = ParameterSample.equally_weighted(
        names=["A"],
        bounds=[(0.0, 4000.0)],
        values=[[2000.0]],
        method=SampleMethod(name="hand"),
    )
    calibrator = Calibrator(Simulation(_split_by_group(None)), problem)

    start = Predictive(calibrator, sample, positions=[0]).runs()[0]

    assert {
        name: start[name][0] for name in ("A_g1_t1", "A_g1_t2", "A_g2_t1", "A_g2_t2")
    } == pytest.approx(
        {"A_g1_t1": 200.0, "A_g1_t2": 600.0, "A_g2_t1": 600.0, "A_g2_t2": 600.0}
    )


def test_compartments_the_fractions_leave_empty_keep_their_declared_split() -> None:
    # g1 takes no share of A, so A_g1 splits as type declares, 0.25 / 0.75.
    builder = a_to_b_builder(
        amounts={"A": None, "B": 0},
        stratifications={
            "group": {"g1": 0.0, "g2": 1.0},
            "type": {"t1": 0.25, "t2": 0.75},
        },
    )
    parameter = CalibrationParameter(
        id="A_g1",
        parameter_type="initial_condition",
        min_bound=0.0,
        max_bound=200.0,
    )

    state = calibrated_initial_state(build(builder), [parameter], [100.0]).state

    assert (state["A_g1_t1"], state["A_g1_t2"]) == pytest.approx((25.0, 75.0))


def test_a_category_keeps_a_chain_of_splits_that_differ_by_it() -> None:
    # level differs by type, which differs by group.
    builder = a_to_b_builder()
    builder.add_stratification(id="group", categories=["g1", "g2"])
    builder.add_stratification(id="type", categories=["t1", "t2"])
    builder.add_stratification(id="level", categories=["l1", "l2"])
    builder.set_structure(
        Aggregate(
            bin_amounts={"A": 1000.0, "B": 0},
            stratification_fractions={
                "group": {"g1": 0.4, "g2": 0.6},
                "type": {
                    "by": "group",
                    "g1": {"t1": 0.25, "t2": 0.75},
                    "g2": {"t1": 0.5, "t2": 0.5},
                },
                "level": {
                    "by": "type",
                    "t1": {"l1": 1.0, "l2": 0.0},
                    "t2": {"l1": 0.0, "l2": 1.0},
                },
            },
        )
    )
    parameter = CalibrationParameter(
        id="g1", parameter_type="initial_condition", min_bound=0.0, max_bound=1.0
    )

    state = calibrated_initial_state(build(builder), [parameter], [0.5]).state

    names = ["A_g1_t1_l1", "A_g1_t2_l2", "A_g2_t1_l1", "A_g2_t2_l2"]
    assert [state[name] for name in names] == pytest.approx([125, 375, 250, 250])


def test_a_category_of_the_stratification_a_split_differs_by_keeps_it() -> None:
    # g1 = 0.5 of A = 1000 is 500, which type splits 0.25 / 0.75.
    parameter = CalibrationParameter(
        id="g1", parameter_type="initial_condition", min_bound=0.0, max_bound=1.0
    )

    state = calibrated_initial_state(_split_by_group(1000.0), [parameter], [0.5]).state

    names = ["A_g1_t1", "A_g1_t2", "A_g2_t1", "A_g2_t2"]
    assert [state[name] for name in names] == pytest.approx([125, 375, 250, 250])


@pytest.mark.parametrize(
    ("values", "remainder"),
    [({"A": 800.0}, "B"), ({"t1": 0.2}, None), ({"g1": 0.5}, None)],
    ids=["bin", "category", "category-it-differs-by"],
)
def test_the_write_back_of_a_split_by_category_rebuilds_the_scored_state(
    values: dict[str, float], remainder: str | None
) -> None:
    model = _split_by_group(1000.0)
    problem = initial_condition_problem(*values, remainder=remainder)
    parameters = list(problem.parameters)
    scored = calibrated_initial_state(
        model, parameters, [values[p.id] for p in parameters], remainder
    ).state

    model.apply_calibration_parameters(values, problem)

    assert initial_state(model) == pytest.approx(scored)
    assert model.structure.compartment_amounts == {}
    assert model.structure.stratification_fractions["type"]["by"] == "group"


def test_the_report_gives_the_calibrated_value_of_each_split() -> None:
    model = _split_by_group(1000.0)
    problem = initial_condition_problem("t1")

    model.apply_calibration_parameters({"t1": 0.2}, problem)
    rows = model.applied_parameters({"t1": 0.2})["category_splits"]

    assert rows["t1 of g1"] == {"group": ["g1"], "fraction": 0.2, "calibrated": True}
    assert rows["t2 of g2"]["fraction"] == pytest.approx(0.8)
