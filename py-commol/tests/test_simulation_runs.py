"""A Simulation runs from the initial state of the model as it is now."""

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.context.model import Model


def _model(k: float | None = 0.1) -> Model:
    return (
        ModelBuilder(name="AB")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=k)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
            ],
        )
        .build(typology="DifferenceEquations")
    )


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
        assert simulation.engine.current_step() == 5

    def test_the_engine_continues_a_run(self) -> None:
        simulation = Simulation(_model())
        result = simulation.run(5)

        simulation.engine.step()

        assert simulation.engine.population[0] == pytest.approx(result["A"][5] * 0.9)


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
                ObservedDataPoint(step=step, compartment="A", value=truth["A"][step])
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
