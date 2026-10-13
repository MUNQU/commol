"""Nelder-Mead and particle swarm recover known parameters and report convergence."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.model import Model
from tests.support import a_to_b_builder, bounded, build, observed

SEED = 42
TRUTH = {"beta": 0.1, "gamma": 0.05}
"""The parameters of the ``truth`` fixture."""
TRUE_K = 0.37


def _fit_beta_and_gamma(
    truth: Model,
    truth_run: dict[str, list[float]],
    config: NelderMeadConfig | ParticleSwarmConfig,
) -> dict[str, float]:
    problem = CalibrationProblem(
        observed_data=observed(truth_run, "I", range(100)),
        parameters=[bounded("beta"), bounded("gamma")],
        loss_function="sse",
        optimization_config=config,
        seed=SEED,
    )
    return Calibrator(Simulation(truth), problem).run().best_parameters


class TestRecovery:
    def test_nelder_mead(self, truth: Model, truth_run) -> None:
        fitted = _fit_beta_and_gamma(
            truth, truth_run, NelderMeadConfig(max_iterations=1000, verbose=False)
        )

        assert fitted == pytest.approx(TRUTH, abs=1e-5)

    def test_particle_swarm(self, truth: Model, truth_run) -> None:
        fitted = _fit_beta_and_gamma(
            truth, truth_run, ParticleSwarmConfig(max_iterations=200, verbose=False)
        )

        assert fitted == pytest.approx(TRUTH, abs=1e-5)

    def test_particle_swarm_with_time_varying_acceleration_and_mutation(
        self, truth: Model, truth_run
    ) -> None:
        config = (
            ParticleSwarmConfig(num_particles=30, max_iterations=200, verbose=False)
            .acceleration(
                "time_varying",
                c1_initial=2.5,
                c1_final=0.5,
                c2_initial=0.5,
                c2_final=2.5,
            )
            .velocity(clamp_factor=0.2)
            .mutation(
                "gaussian", scale=0.1, probability=0.05, application="global_best"
            )
        )

        fitted = _fit_beta_and_gamma(truth, truth_run, config)

        assert fitted == pytest.approx(TRUTH, abs=1e-4)

    def test_particle_swarm_with_chaotic_inertia(self, truth: Model, truth_run) -> None:
        config = ParticleSwarmConfig(
            num_particles=25,
            max_iterations=200,
            verbose=False,
            initialization="opposition_based",
        ).inertia("chaotic", w_min=0.4, w_max=0.9)

        fitted = _fit_beta_and_gamma(truth, truth_run, config)

        assert fitted == pytest.approx(TRUTH, abs=2e-4)


def _ab(k: float | None) -> Model:
    return build(a_to_b_builder(parameters={"k": k}))


def _k_problem(
    config: NelderMeadConfig | ParticleSwarmConfig,
    initial_guess: float | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> CalibrationProblem:
    truth = Simulation(_ab(TRUE_K)).run(10)
    return CalibrationProblem(
        observed_data=observed(truth, "B", range(1, 11)),
        parameters=[bounded("k", *bounds, initial_guess=initial_guess)],
        loss_function="sse",
        optimization_config=config,
    )


def _calibrate(problem: CalibrationProblem):
    return Calibrator(Simulation(_ab(None)), problem).run()


class TestConverged:
    def test_a_run_that_hits_max_iterations_did_not_converge(self) -> None:
        result = _calibrate(_k_problem(NelderMeadConfig(max_iterations=3)))

        assert result.converged is False
        assert "MaxItersReached" in result.termination_reason

    def test_a_run_that_meets_its_tolerance_converged(self) -> None:
        result = _calibrate(_k_problem(NelderMeadConfig(max_iterations=2000)))

        assert result.converged is True
        assert "SolverConverged" in result.termination_reason

    def test_a_particle_swarm_that_hits_max_iterations_did_not_converge(
        self,
    ) -> None:
        config = ParticleSwarmConfig(num_particles=5, max_iterations=3, seed=1)

        result = _calibrate(_k_problem(config))

        assert result.converged is False


class TestNelderMeadInsideBounds:
    def test_the_optimum_is_found_from_the_midpoint(self) -> None:
        result = _calibrate(_k_problem(NelderMeadConfig(max_iterations=2000)))

        assert result.best_parameters["k"] == pytest.approx(TRUE_K, abs=1e-4)

    def test_a_start_value_of_zero_is_not_degenerate(self) -> None:
        config = NelderMeadConfig(max_iterations=2000)

        result = _calibrate(_k_problem(config, initial_guess=0.0))

        assert result.converged is True
        assert result.best_parameters["k"] == pytest.approx(TRUE_K, abs=1e-4)

    def test_the_result_is_inside_the_bounds(self) -> None:
        config = NelderMeadConfig(max_iterations=2000)

        result = _calibrate(_k_problem(config, bounds=(0.5, 0.9)))

        # The optimum is below the bounds, so the best value is the lower bound.
        assert 0.5 <= result.best_parameters["k"] <= 0.9
        assert result.best_parameters["k"] == pytest.approx(0.5, abs=1e-3)

    def test_initial_step_must_be_a_fraction_of_the_width(self) -> None:
        with pytest.raises(ValueError, match="initial_step"):
            NelderMeadConfig(initial_step=1.5)
