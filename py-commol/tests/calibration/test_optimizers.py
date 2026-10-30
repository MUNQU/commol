"""Nelder-Mead and particle swarm recover known parameters and report convergence."""

import math
import re

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol._fit.result import CalibrationResult
from commol._model.model import Model
from tests.support import a_to_b_builder, bounded, build, observed

SEED = 7
TRUTH = {"k1": 0.1, "k2": 0.05}
"""The parameters of the ``truth`` fixture."""
TRUE_K = 0.37


def _fit_k1_and_k2(
    truth: Model,
    truth_run: dict[str, list[float]],
    config: NelderMeadConfig | ParticleSwarmConfig,
) -> dict[str, float]:
    problem = CalibrationProblem(
        observed_data=observed(truth_run, "B", range(100)),
        parameters=[bounded("k1"), bounded("k2")],
        loss_function="sse",
        optimization_config=config,
        seed=SEED,
    )
    return Calibrator(Simulation(truth), problem).run().best_parameters


class TestRecovery:
    def test_nelder_mead(self, truth: Model, truth_run) -> None:
        fitted = _fit_k1_and_k2(
            truth, truth_run, NelderMeadConfig(max_iterations=1000, verbose=False)
        )

        assert fitted == pytest.approx(TRUTH, abs=1e-5)

    def test_particle_swarm(self, truth: Model, truth_run) -> None:
        fitted = _fit_k1_and_k2(
            truth, truth_run, ParticleSwarmConfig(max_iterations=200, verbose=False)
        )

        assert fitted == pytest.approx(TRUTH, abs=1e-5)

    def test_particle_swarm_with_time_varying_acceleration_and_mutation(
        self, truth: Model, truth_run
    ) -> None:
        config = ParticleSwarmConfig(
            particles=30,
            max_iterations=200,
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

        fitted = _fit_k1_and_k2(truth, truth_run, config)

        assert fitted == pytest.approx(TRUTH, abs=1e-4)

    def test_particle_swarm_with_chaotic_inertia(self, truth: Model, truth_run) -> None:
        config = ParticleSwarmConfig(
            particles=25,
            max_iterations=200,
            verbose=False,
            initialization="opposition_based",
            inertia="chaotic",
            inertia_min=0.4,
            inertia_max=0.9,
        )

        fitted = _fit_k1_and_k2(truth, truth_run, config)

        assert fitted == pytest.approx(TRUTH, abs=2e-4)


def _ab(k: float | None) -> Model:
    return build(a_to_b_builder(parameters={"k": k}))


def _k_problem(
    config: NelderMeadConfig | ParticleSwarmConfig,
    initial_guess: float | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
) -> CalibrationProblem:
    truth = Simulation(_ab(TRUE_K)).run(10)
    return CalibrationProblem(
        observed_data=observed(truth, "B", range(1, 11)),
        parameters=[bounded("k", *bounds, initial_guess=initial_guess)],
        loss_function="sse",
        optimization_config=config,
        seed=seed,
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
        config = ParticleSwarmConfig(particles=5, max_iterations=3)

        result = _calibrate(_k_problem(config))

        assert result.converged is False

    def test_a_swarm_that_stagnates_stops_and_converges(self) -> None:
        def swarm(**stop: float) -> ParticleSwarmConfig:
            return ParticleSwarmConfig.model_validate(
                {"particles": 10, "max_iterations": 500, **stop}
            )

        without = _calibrate(_k_problem(swarm(), seed=SEED))
        stopping = [
            swarm(
                improvement_window=20,
                improvement_tol_abs=1e-12,
                improvement_tol_rel=1e-10,
            ),
            swarm(radius_tol=1e-6),
        ]

        assert without.converged is False
        assert without.iterations == 500
        for config in stopping:
            result = _calibrate(_k_problem(config, seed=SEED))
            assert result.converged is True
            assert result.termination_reason == "Terminated(SolverConverged)"
            assert result.iterations < 500
            assert result.best_parameters["k"] == pytest.approx(TRUE_K, abs=1e-4)


class TestSeed:
    def test_the_seed_of_a_result_repeats_a_particle_swarm(self) -> None:
        problem = _k_problem(ParticleSwarmConfig(particles=5, max_iterations=3))

        result = _calibrate(problem)
        again = _calibrate(problem.model_copy(update={"seed": result.seed}))

        assert again == result

    def test_a_history_records_the_seed_of_its_calibration(self) -> None:
        problem = _k_problem(NelderMeadConfig(max_iterations=10), seed=SEED)

        history = Calibrator(Simulation(_ab(None)), problem).run_with_history()

        assert history.result.seed == SEED


class TestHistory:
    def test_the_history_starts_at_the_initial_guess_and_holds_the_best_point(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = _k_problem(NelderMeadConfig(max_iterations=2000))
        calibrator = Calibrator(Simulation(_ab(None)), problem)

        caplog.clear()
        with caplog.at_level("INFO", logger="commol"):
            history = calibrator.run_with_history()

        # Without a guess, the start is the midpoint of the bounds (0, 1).
        assert history.evaluations[0].parameters == pytest.approx({"k": 0.5})
        best = min(history.evaluations, key=lambda evaluation: evaluation.loss)
        assert best.loss == history.result.final_loss
        assert best.parameters == pytest.approx(history.result.best_parameters)
        assert caplog.messages[0] == "Starting calibration with history tracking"
        assert caplog.messages[-1] == (
            f"Calibration finished after {history.result.iterations} iterations. "
            f"Final loss: {history.result.final_loss:.6f}. "
            f"Collected {len(history.evaluations)} evaluations."
        )


class TestParticleSwarmSettings:
    """Each setting reaches the optimizer: it changes the points one seed
    evaluates."""

    @staticmethod
    def _base() -> ParticleSwarmConfig:
        return ParticleSwarmConfig(particles=5, max_iterations=5)

    @staticmethod
    def _search(config: ParticleSwarmConfig) -> list[float]:
        calibrator = Calibrator(Simulation(_ab(None)), _k_problem(config, seed=SEED))
        history = calibrator.run_with_history()
        return [evaluation.parameters["k"] for evaluation in history.evaluations]

    @pytest.mark.parametrize(
        "settings",
        [
            pytest.param(
                {
                    "mutation": "gaussian",
                    "mutation_scale": 0.5,
                    "mutation_probability": 1.0,
                    "mutation_application": "all_particles",
                },
                id="mutation",
            ),
            pytest.param({"inertia": "constant", "inertia_factor": 0.1}, id="inertia"),
            pytest.param(
                {"inertia": "chaotic", "inertia_min": 0.1, "inertia_max": 0.2},
                id="chaotic-inertia",
            ),
            pytest.param(
                {"acceleration": "constant", "cognitive": 2.5, "social": 0.1},
                id="acceleration",
            ),
            pytest.param(
                {
                    "acceleration": "time_varying",
                    "cognitive_initial": 2.5,
                    "cognitive_final": 0.1,
                    "social_initial": 0.1,
                    "social_final": 2.5,
                },
                id="time-varying-acceleration",
            ),
            pytest.param({"velocity_clamp_factor": 0.01}, id="velocity"),
            pytest.param({"velocity_mutation_threshold": 1.0}, id="velocity-threshold"),
            pytest.param({"particles": 6}, id="particles"),
            pytest.param({"initialization": "latin_hypercube"}, id="initialization"),
        ],
    )
    def test_a_setting_changes_the_search_of_one_seed(
        self, settings: dict[str, object]
    ) -> None:
        configured = ParticleSwarmConfig.model_validate(
            self._base().model_dump() | settings
        )

        assert self._search(self._base()) == self._search(self._base())
        assert self._search(configured) != self._search(self._base())

    def test_a_verbose_swarm_reports_its_iterations(self, capfd) -> None:
        _calibrate(_k_problem(self._base(), seed=SEED))
        quiet = capfd.readouterr().out
        _calibrate(_k_problem(self._base().model_copy(update={"verbose": True})))

        assert quiet == ""
        assert capfd.readouterr().out != ""


class TestVerbose:
    def test_a_verbose_nelder_mead_prints_its_header(self, capfd) -> None:
        _calibrate(_k_problem(NelderMeadConfig(max_iterations=3, verbose=True)))

        assert (
            "=== Nelder-Mead Optimization (Verbose Mode) ===" in capfd.readouterr().out
        )


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

    def test_initial_simplex_size_must_be_a_fraction_of_the_width(self) -> None:
        with pytest.raises(ValueError, match="initial_simplex_size"):
            NelderMeadConfig(initial_simplex_size=1.5)


class TestParticleSwarmConfig:
    @pytest.mark.parametrize(
        ("setting", "other"),
        [
            ("inertia", {"inertia_factor": 0.5}),
            ("acceleration", {"cognitive": 2.0, "social": 0.5}),
        ],
    )
    def test_a_constant_setting_has_the_default_of_the_engine(
        self, setting: str, other: dict[str, float]
    ) -> None:
        def fit(**values: float) -> CalibrationResult:
            config = ParticleSwarmConfig.model_validate(
                {"particles": 4, "max_iterations": 3, setting: "constant", **values}
            )
            return _calibrate(_k_problem(config, seed=SEED))

        unset = _calibrate(
            _k_problem(ParticleSwarmConfig(particles=4, max_iterations=3), seed=SEED)
        )

        assert fit() == unset
        assert fit(**other).best_parameters != unset.best_parameters

    def test_a_chaotic_inertia_needs_a_range(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"inertia_max \(0.4\) must be greater than inertia_min \(0.9\)",
        ):
            ParticleSwarmConfig(inertia="chaotic", inertia_min=0.9, inertia_max=0.4)

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            (
                {"inertia_factor": 0.5},
                "inertia_factor is read only with another inertia",
            ),
            (
                {"inertia": "constant", "inertia_min": 0.4},
                "inertia_min is read only with another inertia",
            ),
            (
                {"inertia": "chaotic", "inertia_factor": 0.5},
                "inertia_factor is read only with another inertia",
            ),
            (
                {"inertia": "chaotic", "inertia_max": 0.9},
                "inertia='chaotic' needs inertia_min",
            ),
            ({"cognitive": 2.0}, "cognitive is read only with another acceleration"),
            (
                {"social_initial": 2.0},
                "social_initial is read only with another acceleration",
            ),
            (
                {"acceleration": "constant", "social_final": 1.0},
                "social_final is read only with another acceleration",
            ),
            (
                {"acceleration": "time_varying", "social": 1.0},
                "social is read only with another acceleration",
            ),
            (
                {
                    "acceleration": "time_varying",
                    "cognitive_initial": 2.0,
                    "cognitive_final": 1.0,
                    "social_initial": 1.0,
                },
                "acceleration='time_varying' needs social_final",
            ),
            (
                {"mutation_application": "global_best"},
                "mutation_application is read only with another mutation",
            ),
            (
                {
                    "mutation": "cauchy",
                    "mutation_probability": 0.1,
                    "mutation_application": "global_best",
                },
                "mutation='cauchy' needs mutation_scale",
            ),
            (
                {
                    "mutation": "gaussian",
                    "mutation_scale": 0.1,
                    "mutation_application": "global_best",
                },
                "mutation='gaussian' needs mutation_probability",
            ),
            (
                {
                    "mutation": "gaussian",
                    "mutation_scale": 0.1,
                    "mutation_probability": 0.1,
                },
                "mutation='gaussian' needs mutation_application",
            ),
            (
                {"improvement_window": 5, "improvement_tol_abs": 0.0},
                "The improvement stop rule needs improvement_window, "
                "improvement_tol_abs and improvement_tol_rel, all three",
            ),
        ],
    )
    def test_a_number_is_set_with_its_strategy_and_a_strategy_with_its_numbers(
        self, settings: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=re.escape(message)):
            ParticleSwarmConfig.model_validate(settings)

    def test_an_initialization_is_one_of_the_known_ones(self) -> None:
        with pytest.raises(
            ValidationError,
            match="'uniform', 'latin_hypercube' or 'opposition_based'",
        ):
            ParticleSwarmConfig(initialization="sobol")  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        ("settings", "field"),
        [
            (
                {
                    "improvement_window": 0,
                    "improvement_tol_abs": 0.0,
                    "improvement_tol_rel": 0.0,
                },
                "improvement_window",
            ),
            (
                {
                    "improvement_window": 5,
                    "improvement_tol_abs": -1.0,
                    "improvement_tol_rel": 0.0,
                },
                "improvement_tol_abs",
            ),
            (
                {
                    "improvement_window": 5,
                    "improvement_tol_abs": 0.0,
                    "improvement_tol_rel": -1.0,
                },
                "improvement_tol_rel",
            ),
            ({"radius_tol": -1.0}, "radius_tol"),
        ],
    )
    def test_a_stop_rule_out_of_its_range_names_its_setting(
        self, settings: dict[str, object], field: str
    ) -> None:
        with pytest.raises(
            ValidationError, match=rf"Value error, {field} is a (whole )?number"
        ):
            ParticleSwarmConfig.model_validate(settings)

    @pytest.mark.parametrize(
        ("settings", "names"),
        [
            ({"inertia": "linear"}, "'constant' or 'chaotic'"),
            ({"acceleration": "linear"}, "'constant' or 'time_varying'"),
            ({"mutation": "uniform"}, "'gaussian' or 'cauchy'"),
            (
                {"mutation_application": "worst"},
                "'global_best', 'all_particles' or 'below_average'",
            ),
        ],
    )
    def test_an_unknown_strategy_lists_the_known_ones(
        self, settings: dict[str, object], names: str
    ) -> None:
        with pytest.raises(ValidationError, match=re.escape(names)):
            ParticleSwarmConfig.model_validate(settings)


def test_a_particle_swarm_takes_particles(capfd) -> None:
    config = ParticleSwarmConfig(particles=4, max_iterations=1, verbose=True)

    _calibrate(_k_problem(config))

    assert "Particles: 4\n" in capfd.readouterr().out


def test_an_optimizer_refuses_an_infinite_bound() -> None:
    # A problem keeps an infinite bound, for a prior on a half-line: it
    # serves a sampler that never fits. Each start of a fit refuses it.
    message = r"An optimizer searches inside the bounds.*'k' has the bounds \[0, inf\]"
    for config in [NelderMeadConfig(max_iterations=2), ParticleSwarmConfig()]:
        problem = _k_problem(config, bounds=(0.0, math.inf), initial_guess=0.3)
        with pytest.raises(RuntimeError, match=message):
            _calibrate(problem)
    with pytest.raises(ValueError, match=r"bounds of parameter 'k' must be numbers"):
        bounded("k", math.nan, 1.0)
    with pytest.raises(ValueError, match=r"max_bound \(0\) must be greater"):
        bounded("k", 0.0, 0.0)
    with pytest.raises(ValueError, match=r"initial_guess \(2\) must be a finite"):
        bounded("k", 0.0, 1.0, initial_guess=2.0)
