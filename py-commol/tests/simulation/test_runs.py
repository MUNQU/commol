"""A Simulation runs from the initial state of the model as it is now."""

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.context.model import Model
from tests.support import (
    ENGINE_REL,
    SIR_START,
    a_to_b_builder,
    assert_close,
    assert_population_conserved,
    build,
    reference_run,
    sir_builder,
    sir_flows,
)


def _model(k: float | None = 0.1) -> Model:
    return build(a_to_b_builder(parameters={"k": k}))


def _per_capita_sir() -> Model:
    return build(
        sir_builder(
            parameters={"infection_rate": 0.1, "recovery_rate": 0.05},
            infection="infection_rate",
            recovery="recovery_rate",
        )
    )


PER_CAPITA_SIR = reference_run(
    SIR_START,
    sir_flows(lambda s, _: 0.1 * s["S"], lambda s, _: 0.05 * s["I"]),
    10,
)
"""
The run of ``_per_capita_sir``. By hand, step 1 is S = 990 - 99 = 891,
I = 10 + 99 - 0.5 = 108.5 and R = 0.5.
"""


class TestOutputFormats:
    def test_dict_of_lists_is_the_default(self) -> None:
        result = Simulation(_per_capita_sir()).run(10)

        assert list(result) == ["S", "I", "R"]
        assert_close(result, PER_CAPITA_SIR, rel=ENGINE_REL)
        assert result["I"][1] == pytest.approx(108.5)
        assert_population_conserved(result, 1000.0)

    def test_list_of_lists_has_one_row_per_step(self) -> None:
        rows = Simulation(_per_capita_sir()).run(10, output_format="list_of_lists")

        assert len(rows) == 11
        for step, row in enumerate(rows):
            expected = [PER_CAPITA_SIR[name][step] for name in ("S", "I", "R")]
            assert_close(row, expected, rel=ENGINE_REL)

    def test_a_run_with_an_uncalibrated_parameter_is_rejected(self) -> None:
        simulation = Simulation(
            build(sir_builder(parameters={"beta": None, "gamma": 0.05}))
        )

        with pytest.raises(ValueError, match="Cannot run Simulation") as error:
            simulation.run(100)

        assert "beta" in str(error.value)
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
