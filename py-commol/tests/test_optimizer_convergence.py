"""`converged` reports convergence, and Nelder-Mead searches inside the bounds."""

import pytest

from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model

TRUE_K = 0.37


def _model(k: float | None) -> Model:
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


def _problem(
    config: NelderMeadConfig | ParticleSwarmConfig,
    initial_guess: float | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> CalibrationProblem:
    truth = Simulation(_model(TRUE_K)).run(10)
    return CalibrationProblem(
        observed_data=[
            ObservedDataPoint(step=step, compartment="B", value=truth["B"][step])
            for step in range(1, 11)
        ],
        parameters=[
            CalibrationParameter(
                id="k",
                parameter_type="parameter",
                min_bound=bounds[0],
                max_bound=bounds[1],
                initial_guess=initial_guess,
            )
        ],
        loss_function="sse",
        optimization_config=config,
    )


def _calibrate(problem: CalibrationProblem):
    return Calibrator(Simulation(_model(None)), problem).run()


class TestConverged:
    def test_a_run_that_hits_max_iterations_did_not_converge(self) -> None:
        result = _calibrate(_problem(NelderMeadConfig(max_iterations=3)))

        assert result.converged is False
        assert "MaxItersReached" in result.termination_reason

    def test_a_run_that_meets_its_tolerance_converged(self) -> None:
        result = _calibrate(_problem(NelderMeadConfig(max_iterations=2000)))

        assert result.converged is True
        assert "SolverConverged" in result.termination_reason

    def test_a_particle_swarm_that_hits_max_iterations_did_not_converge(
        self,
    ) -> None:
        config = ParticleSwarmConfig(num_particles=5, max_iterations=3, seed=1)

        result = _calibrate(_problem(config))

        assert result.converged is False


class TestNelderMeadInsideBounds:
    def test_the_optimum_is_found_from_the_midpoint(self) -> None:
        result = _calibrate(_problem(NelderMeadConfig(max_iterations=2000)))

        assert result.best_parameters["k"] == pytest.approx(TRUE_K, abs=1e-4)

    def test_a_start_value_of_zero_is_not_degenerate(self) -> None:
        config = NelderMeadConfig(max_iterations=2000)

        result = _calibrate(_problem(config, initial_guess=0.0))

        assert result.converged is True
        assert result.best_parameters["k"] == pytest.approx(TRUE_K, abs=1e-4)

    def test_the_result_is_inside_the_bounds(self) -> None:
        config = NelderMeadConfig(max_iterations=2000)

        result = _calibrate(_problem(config, bounds=(0.5, 0.9)))

        # The optimum is below the bounds, so the best value is the lower bound.
        assert 0.5 <= result.best_parameters["k"] <= 0.9
        assert result.best_parameters["k"] == pytest.approx(0.5, abs=1e-3)

    def test_initial_step_must_be_a_fraction_of_the_width(self) -> None:
        with pytest.raises(ValueError, match="initial_step"):
            NelderMeadConfig(initial_step=1.5)
