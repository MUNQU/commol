"""Scale factors fit observations that see only a share of an output."""

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationConstraint,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    build_ode,
    observed,
    sir_builder,
)

SEED = 42


def _fit(
    truth: Model,
    observations,
    parameters,
    config: NelderMeadConfig | ParticleSwarmConfig,
):
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=parameters,
        loss_function="sse",
        optimization_config=config,
        seed=SEED,
    )
    return Calibrator(Simulation(truth), problem).run()


class TestScaleFactors:
    def test_a_scale_alone_is_recovered(self, truth: Model, truth_run) -> None:
        observations = observed(
            truth_run, "I", range(0, 100, 10), factor=0.5, factor_id="detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [bounded("detection_rate", 0.1, 1.0, kind="factor", initial_guess=0.7)],
            NelderMeadConfig(max_iterations=500, verbose=False),
        )

        assert result.best_parameters["detection_rate"] == pytest.approx(0.5, abs=1e-4)
        assert result.converged

    def test_a_scale_and_a_parameter_are_recovered_together(
        self, truth: Model, truth_run
    ) -> None:
        observations = observed(
            truth_run, "I", range(0, 100, 5), factor=0.3, factor_id="detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [bounded("beta"), bounded("detection_rate", 0.1, 1.0, kind="factor")],
            ParticleSwarmConfig(max_iterations=300, num_particles=25, verbose=False),
        )

        assert result.best_parameters == pytest.approx(
            {"beta": 0.1, "detection_rate": 0.3}, abs=1e-3
        )

    def test_each_output_has_its_own_scale(self, truth: Model, truth_run) -> None:
        observations = observed(
            truth_run, "I", range(0, 100, 10), factor=0.6, factor_id="i_detection_rate"
        ) + observed(
            truth_run, "R", range(0, 100, 10), factor=0.9, factor_id="r_detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [
                bounded("i_detection_rate", 0.1, 1.0, kind="factor"),
                bounded("r_detection_rate", 0.1, 1.0, kind="factor"),
            ],
            NelderMeadConfig(max_iterations=500, verbose=False),
        )

        assert result.best_parameters == pytest.approx(
            {"i_detection_rate": 0.6, "r_detection_rate": 0.9}, abs=1e-3
        )


ODE_TRUTH = {"beta": 0.3, "gamma": 0.1}
STEPS = [0.5 + 1.5 * j for j in range(40)]
"""Observation steps, none of them an integer."""


def _fit_ode(problem: CalibrationProblem, model) -> dict[str, float]:
    return Calibrator(Simulation(model), problem).run().best_parameters


class TestStepsBetweenIntegers:
    """A DifferentialEquations model observed at any step."""

    def test_observations_at_non_integer_steps_recover_the_parameters(self) -> None:
        truth = build_ode(sir_builder())
        results = Simulation(truth).run_at(STEPS)
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(at=step, output="I", value=value)
                for step, value in zip(STEPS, results["I"], strict=True)
            ],
            parameters=[
                bounded("beta", initial_guess=0.5),
                bounded("gamma", high=0.5, initial_guess=0.2),
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=2000, verbose=False),
        )

        fitted = _fit_ode(
            problem, build_ode(sir_builder(parameters={"beta": None, "gamma": None}))
        )

        assert fitted == pytest.approx(ODE_TRUTH, rel=1e-4)

    def test_a_window_over_non_integer_steps_recovers_the_rate(self) -> None:
        truth = build_ode(a_to_b_builder(parameters={"k": 0.2}, accumulator="moved"))
        results = Simulation(truth).run_at([0.5, 3.5, 6.5])
        observations = [
            ObservedDataPoint(
                at=end,
                output="moved",
                value=results["moved"][i + 1] - results["moved"][i],
                window=3.0,
            )
            for i, end in enumerate([3.5, 6.5])
        ]
        problem = CalibrationProblem(
            observed_data=observations,
            parameters=[bounded("k", initial_guess=0.5)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=500, verbose=False),
        )

        fitted = _fit_ode(
            problem,
            build_ode(a_to_b_builder(parameters={"k": None}, accumulator="moved")),
        )

        assert fitted["k"] == pytest.approx(0.2, rel=1e-5)


class TestPositionsAndTypology:
    def test_one_output_names_its_series(self) -> None:
        point = ObservedDataPoint(at=3, output="I", value=1.0)

        assert point.label == "I" and point.observed_outputs == ["I"]

    def test_a_sum_of_outputs_needs_a_label(self) -> None:
        with pytest.raises(ValidationError, match="label"):
            ObservedDataPoint(at=3, outputs=["I", "R"], value=1.0)

    def test_output_and_outputs_exclude_each_other(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            ObservedDataPoint(at=3, output="I", outputs=["R"], label="x", value=1.0)

    def test_a_difference_model_rejects_a_non_integer_observation(self) -> None:
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=2.5, output="A", value=1.0)],
            parameters=[bounded("k")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        model = build(a_to_b_builder(parameters={"k": None}))

        with pytest.raises(ValueError, match="Observation 'A': .*2.5 is not one"):
            Calibrator(Simulation(model), problem)

    def test_a_difference_model_rejects_a_non_integer_constraint_step(self) -> None:
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=2, output="A", value=1.0)],
            parameters=[bounded("k")],
            constraints=[CalibrationConstraint(id="cap", expression="1 - A", at=[1.5])],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        model = build(a_to_b_builder(parameters={"k": None}))

        with pytest.raises(ValueError, match="Constraint 'cap': .*1.5 is not one"):
            Calibrator(Simulation(model), problem)
