"""A Simulation runs from the initial state of the model as it is now."""

import json

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
    SimulationResults,
)
from commol.context.engine import EngineName
from commol.context.model import Model
from tests.support import (
    ENGINE_REL,
    ABC_START,
    a_to_b_builder,
    assert_close,
    assert_total_conserved,
    build,
    build_ode,
    build_stochastic,
    reference_run,
    abc_builder,
    abc_flows,
)


def _model(k: float | None = 0.1) -> Model:
    return build(a_to_b_builder(parameters={"k": k}))


def _proportional_abc() -> Model:
    return build(
        abc_builder(
            parameters={"rate_ab": 0.1, "rate_bc": 0.05},
            ab="rate_ab",
            bc="rate_bc",
        )
    )


PROPORTIONAL_ABC = reference_run(
    ABC_START,
    abc_flows(lambda s, _: 0.1 * s["A"], lambda s, _: 0.05 * s["B"]),
    10,
)
"""
The run of ``_proportional_abc``. By hand, step 1 is A = 990 - 99 = 891,
B = 10 + 99 - 0.5 = 108.5 and C = 0.5.
"""


class TestResults:
    def test_results_print_their_positions_their_seed_and_their_values(self) -> None:
        plain = Simulation(build(abc_builder())).run(1)
        drawn = Simulation(build_stochastic(abc_builder(), "StochasticEvents")).run(
            1, seed=3
        )

        assert repr(plain) == (
            "SimulationResults(positions=[0.0, 1.0], {'A': [990.0, 987.03], "
            "'B': [10.0, 11.97], 'C': [0.0, 1.0]})"
        )
        assert repr(drawn).startswith(
            "SimulationResults(positions=[0.0, 1.0], seed=3, {'A': [990.0, "
        )

    def test_results_have_one_value_per_position(self) -> None:
        with pytest.raises(ValueError, match="'A' has 1 values for 2 positions"):
            SimulationResults([0.0, 1.0], {"A": [1.0]})

    def test_results_map_each_output_to_its_values(self) -> None:
        result = Simulation(_proportional_abc()).run(10)

        assert isinstance(result, dict)
        assert list(result) == ["A", "B", "C"]
        assert_close(result, PROPORTIONAL_ABC, rel=ENGINE_REL)
        assert result["B"][1] == pytest.approx(108.5)
        assert_total_conserved(result, 1000.0)

    def test_a_run_holds_the_positions_of_its_grid(self) -> None:
        result = Simulation(_proportional_abc()).run(10)

        assert result.positions == [float(step) for step in range(11)]

    def test_rows_give_every_output_at_each_position(self) -> None:
        rows = Simulation(_proportional_abc()).run(10).rows()

        assert len(rows) == 11
        for step, row in enumerate(rows):
            expected = [PROPORTIONAL_ABC[name][step] for name in ("A", "B", "C")]
            assert_close(row, expected, rel=ENGINE_REL)

    def test_results_serialize_as_a_mapping(self) -> None:
        result = Simulation(_proportional_abc()).run(2)

        assert json.loads(json.dumps(result)) == dict(result)

    def test_a_run_with_an_uncalibrated_parameter_is_rejected(self) -> None:
        simulation = Simulation(build(abc_builder(parameters={"k1": None, "k2": 0.05})))

        with pytest.raises(ValueError, match="Cannot run Simulation") as error:
            simulation.run(100)

        assert "k1" in str(error.value)
        assert "calibration" in str(error.value).lower()


class TestRepeatableRuns:
    def test_two_runs_give_the_same_result(self) -> None:
        simulation = Simulation(_model())

        first = simulation.run(10)
        second = simulation.run(10)

        assert second == first

    def test_a_run_restarts_at_step_zero(self) -> None:
        simulation = Simulation(_model())
        simulation.run(5)

        result = simulation.run(5)

        assert result["A"][0] == 1000.0
        assert simulation.engine.current_position() == 5

    def test_the_engine_continues_a_run(self) -> None:
        simulation = Simulation(_model())
        result = simulation.run(5)

        simulation.engine.step()

        assert simulation.engine.state[0] == pytest.approx(result["A"][5] * 0.9)


class TestModelChanges:
    def test_a_parameter_set_after_creation_from_none(self) -> None:
        model = _model(k=None)
        simulation = Simulation(model)

        model.update_parameters({"k": 0.2})

        assert simulation.run(1)["A"][1] == pytest.approx(800.0)

    def test_a_parameter_changed_after_creation(self) -> None:
        model = _model(k=0.1)
        simulation = Simulation(model)
        simulation.run(1)

        model.update_parameters({"k": 0.5})

        assert simulation.run(1)["A"][1] == pytest.approx(500.0)

    def test_an_unknown_parameter_names_the_parameters_of_the_model(self) -> None:
        model = build(abc_builder())

        with pytest.raises(
            ValueError,
            match="Parameter 'x' not found in model. Available parameters: k1, k2",
        ):
            model.update_parameters({"x": 1.0})

    def test_a_direct_field_edit(self) -> None:
        model = _model(k=0.1)
        simulation = Simulation(model)

        model.parameters[0].value = 0.3

        assert simulation.run(1)["A"][1] == pytest.approx(700.0)

    def test_an_unchanged_model_keeps_the_engine(self) -> None:
        simulation = Simulation(_model())

        assert simulation.engine is simulation.engine

    def test_a_calibrator_after_a_change_uses_the_new_model(self) -> None:
        model = _model(k=None)
        simulation = Simulation(model)
        model.update_parameters({"k": 0.2})
        truth = simulation.run(10)
        model.update_parameters({"k": None})
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(at=step, output="A", value=truth["A"][step])
                for step in range(1, 11)
            ],
            parameters=[
                CalibrationParameter(
                    id="k", parameter_type="parameter", min_bound=0.0, max_bound=0.5
                )
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=500),
        )

        result = Calibrator(simulation, problem).run()

        assert result.best_parameters["k"] == pytest.approx(0.2, abs=1e-3)


@pytest.mark.parametrize("ode", [False, True], ids=["difference", "differential"])
def test_a_step_of_the_engine_continues_the_run(ode: bool) -> None:
    model = build_ode(abc_builder()) if ode else build(abc_builder())
    simulation = Simulation(model)
    two_steps = simulation.run(2)
    simulation.run(1)

    simulation.engine.step()

    assert simulation.engine.state == pytest.approx(
        [two_steps[name][2] for name in ("A", "B", "C")], rel=1e-6
    )


@pytest.mark.parametrize(
    ("ode", "position"),
    [(False, "2"), (True, r"1\.5\d*")],
    ids=["difference", "differential"],
)
def test_a_rate_that_is_not_a_number_stops_the_run(ode: bool, position: str) -> None:
    builder = a_to_b_builder(rate="k * sqrt(1.5 - t)")
    simulation = Simulation(build_ode(builder) if ode else build(builder))

    with pytest.raises(
        RuntimeError,
        match=f"The rate of flow 't_ab' at {position} is not a number",
    ):
        simulation.run(3)


@pytest.mark.parametrize(
    "engine",
    [
        "DifferenceEquations",
        "DifferentialEquations",
        "StochasticEvents",
        "StochasticSteps",
    ],
)
def test_each_engine_names_itself_when_it_rejects_a_position(
    engine: EngineName,
) -> None:
    builder = abc_builder()
    simulation = Simulation(builder.build(builder.structure, engine=engine))
    integer = engine in ("DifferenceEquations", "StochasticSteps")
    positions = [2.5, -1.0] if integer else [-1.0]

    for position in positions:
        with pytest.raises(ValueError) as error:
            simulation.run_at([position], seed=1)
        assert str(error.value).startswith(f"The {engine} engine reports")
        assert f"{position:g} is not one" in str(error.value)
    if not integer:
        assert simulation.run_at([2.5], seed=1).positions == [2.5]


def test_a_result_reads_a_position_with_the_tolerance_of_the_engine() -> None:
    simulation = Simulation(build_ode(a_to_b_builder(parameters={"k": 0.1})))
    results = simulation.run_at([0.3])

    assert simulation.sample(results, ["A"], [0.1 + 0.2]) == results["A"]
    with pytest.raises(ValueError, match=r"no value at 0\.31;"):
        simulation.sample(results, ["A"], [0.31])
