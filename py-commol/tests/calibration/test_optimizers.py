"""Nelder-Mead and particle swarm recover known parameters and report convergence."""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ParticleSwarmConfig,
    Simulation,
)
from commol.context.calibration import (
    PSOChaoticInertia,
    PSOConstantAcceleration,
    PSOConstantInertia,
    PSOMutationConfig,
    PSOTimeVaryingAcceleration,
    PSOVelocityConfig,
)
from commol.context.model import Model
from tests.support import a_to_b_builder, bounded, build, observed

SEED = 42
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

        fitted = _fit_k1_and_k2(truth, truth_run, config)

        assert fitted == pytest.approx(TRUTH, abs=1e-4)

    def test_particle_swarm_with_chaotic_inertia(self, truth: Model, truth_run) -> None:
        config = ParticleSwarmConfig(
            num_particles=25,
            max_iterations=200,
            verbose=False,
            initialization="opposition_based",
        ).inertia("chaotic", w_min=0.4, w_max=0.9)

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
        config = ParticleSwarmConfig(num_particles=5, max_iterations=3)

        result = _calibrate(_k_problem(config))

        assert result.converged is False


class TestSeed:
    def test_the_seed_of_a_result_repeats_a_particle_swarm(self) -> None:
        problem = _k_problem(ParticleSwarmConfig(num_particles=5, max_iterations=3))

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
    """Each setting reaches the optimizer: it changes the search of one seed."""

    @staticmethod
    def _base() -> ParticleSwarmConfig:
        return ParticleSwarmConfig(num_particles=5, max_iterations=5)

    @staticmethod
    def _search(config: ParticleSwarmConfig) -> float:
        return _calibrate(_k_problem(config, seed=SEED)).best_parameters["k"]

    @pytest.mark.parametrize(
        "configure",
        [
            pytest.param(
                lambda c: c.mutation(
                    "gaussian", scale=0.5, probability=1.0, application="all_particles"
                ),
                id="mutation",
            ),
            pytest.param(lambda c: c.inertia("constant", factor=0.1), id="inertia"),
            pytest.param(
                lambda c: c.inertia("chaotic", w_min=0.1, w_max=0.2),
                id="chaotic-inertia",
            ),
            pytest.param(
                lambda c: c.acceleration("constant", cognitive=2.5, social=0.1),
                id="acceleration",
            ),
            pytest.param(
                lambda c: c.acceleration(
                    "time_varying",
                    c1_initial=2.5,
                    c1_final=0.1,
                    c2_initial=0.1,
                    c2_final=2.5,
                ),
                id="time-varying-acceleration",
            ),
            pytest.param(lambda c: c.velocity(clamp_factor=0.01), id="velocity"),
            pytest.param(
                lambda c: c.velocity(mutation_threshold=1.0), id="velocity-threshold"
            ),
            pytest.param(
                lambda c: c.model_copy(update={"num_particles": 6}), id="particles"
            ),
            pytest.param(
                lambda c: c.model_copy(update={"initialization": "latin_hypercube"}),
                id="initialization",
            ),
        ],
    )
    def test_a_setting_changes_the_search_of_one_seed(self, configure) -> None:
        assert self._search(self._base()) == self._search(self._base())
        assert self._search(configure(self._base())) != self._search(self._base())

    def test_a_verbose_swarm_reports_its_iterations(self, capfd) -> None:
        self._search(self._base())
        quiet = capfd.readouterr().out
        self._search(self._base().model_copy(update={"verbose": True}))

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
    def test_a_constant_inertia_has_the_canonical_factor_by_default(self) -> None:
        assert ParticleSwarmConfig().inertia(
            "constant"
        ).inertia_config == PSOConstantInertia(factor=0.721)
        assert ParticleSwarmConfig().inertia(
            "constant", factor=0.5
        ).inertia_config == PSOConstantInertia(factor=0.5)

    def test_a_chaotic_inertia_takes_its_range(self) -> None:
        config = ParticleSwarmConfig().inertia("chaotic", w_min=0.4, w_max=0.9)

        assert config.inertia_config == PSOChaoticInertia(w_min=0.4, w_max=0.9)

    def test_a_chaotic_inertia_needs_a_range(self) -> None:
        with pytest.raises(ValueError, match=r"w_max \(0.4\) must be greater"):
            ParticleSwarmConfig().inertia("chaotic", w_min=0.9, w_max=0.4)

    def test_an_initialization_is_one_of_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="Invalid initialization: sobol"):
            ParticleSwarmConfig(initialization="sobol")

    def test_a_constant_acceleration_has_the_canonical_coefficients_by_default(
        self,
    ) -> None:
        assert ParticleSwarmConfig().acceleration(
            "constant"
        ).acceleration_config == PSOConstantAcceleration(cognitive=1.193, social=1.193)
        assert ParticleSwarmConfig().acceleration(
            "constant", cognitive=2.0, social=0.5
        ).acceleration_config == PSOConstantAcceleration(cognitive=2.0, social=0.5)

    def test_a_time_varying_acceleration_takes_its_four_ends(self) -> None:
        config = ParticleSwarmConfig().acceleration(
            "time_varying", c1_initial=2.5, c1_final=0.5, c2_initial=0.5, c2_final=2.5
        )

        assert config.acceleration_config == PSOTimeVaryingAcceleration(
            c1_initial=2.5, c1_final=0.5, c2_initial=0.5, c2_final=2.5
        )

    def test_a_mutation_and_a_velocity_keep_their_settings(self) -> None:
        config = (
            ParticleSwarmConfig()
            .mutation("cauchy", scale=0.2, probability=0.1, application="below_average")
            .velocity(clamp_factor=0.1, mutation_threshold=0.01)
        )

        assert config.mutation_config == PSOMutationConfig(
            strategy="cauchy", scale=0.2, probability=0.1, application="below_average"
        )
        assert config.velocity_config == PSOVelocityConfig(
            clamp_factor=0.1, mutation_threshold=0.01
        )

    @pytest.mark.parametrize(
        ("configure", "message"),
        [
            (
                lambda c: c.inertia("linear"),
                "Invalid inertia type: linear. Must be one of "
                "[<PSOInertiaType.CONSTANT: 'constant'>, "
                "<PSOInertiaType.CHAOTIC: 'chaotic'>]",
            ),
            (
                lambda c: c.acceleration("linear"),
                "Invalid acceleration type: linear. Must be one of "
                "[<PSOAccelerationType.CONSTANT: 'constant'>, "
                "<PSOAccelerationType.TIME_VARYING: 'time_varying'>]",
            ),
            (
                lambda c: c.mutation(
                    "uniform", scale=0.1, probability=0.1, application="global_best"
                ),
                "Invalid mutation strategy: uniform. Must be one of "
                "[<PSOMutationStrategy.GAUSSIAN: 'gaussian'>, "
                "<PSOMutationStrategy.CAUCHY: 'cauchy'>]",
            ),
        ],
    )
    def test_an_unknown_strategy_lists_the_known_ones(
        self, configure, message: str
    ) -> None:
        with pytest.raises(ValueError) as error:
            configure(ParticleSwarmConfig())

        assert str(error.value) == message

    def test_an_unknown_mutation_application_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError) as error:
            ParticleSwarmConfig().mutation(
                "gaussian", scale=0.1, probability=0.1, application="worst"
            )

        assert str(error.value).startswith(
            "Invalid mutation application: worst. Must be one of "
            "[<PSOMutationApplication.GLOBAL_BEST: 'global_best'>"
        )
