"""Observation models: their checks, maximum likelihood, the Bayesian
posterior, prediction with their noise and the parametric bootstrap.

Most tests fit a constant flow ``c`` (``flow_calibrator`` with windowed
observations), so each observation's prediction is ``c`` and the closed forms
are those of a sample of one distribution: with a normal model of known
``sd`` and a normal prior, the posterior of ``c`` is normal, and the variance
of the fit is ``sd²/n``.
"""

import logging
import math

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    AbcSmc,
    BayesianPosterior,
    BinomialObservations,
    CalibrationProblem,
    Calibrator,
    DistributionNoise,
    GibbsPosterior,
    LogNormalObservations,
    LossDistance,
    NegativeBinomialObservations,
    NelderMeadConfig,
    Normal,
    NormalObservations,
    ObservationModel,
    ObservedDataPoint,
    ParameterSample,
    ParametricBootstrap,
    PoissonObservations,
    Predictive,
    Priors,
    SampleMethod,
    Simulation,
    StudentTObservations,
    applied_parameters_report,
    scores,
)
from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    build_stochastic,
    flow_calibrator,
    flow_observations,
)

FLOW = 10.0


def _posterior(**settings: object) -> BayesianPosterior:
    defaults: dict[str, object] = {
        "priors": Priors.uniform_on_bounds(),
        "particles": 400,
        "target_cess": 0.5,
        "resample_threshold": 0.5,
        "stay_probability": 0.05,
    }
    return BayesianPosterior.model_validate(defaults | settings)


def _problem(
    observed_data: list[ObservedDataPoint],
    models: dict[str, ObservationModel],
    noise: tuple[str, ...] = (),
    **settings: object,
) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=observed_data,
        parameters=[bounded("c", 0.0, 50.0)]
        + [bounded(name, 0.0, 5.0, kind="noise") for name in noise],
        observation_models=models,
        optimization_config=NelderMeadConfig(max_iterations=500),
        **settings,  # type: ignore[arg-type]
    )


def _sd(sample: ParameterSample) -> float:
    """The weighted standard deviation of the first parameter."""
    return math.sqrt(sample.covariance()[0][0])


def _flow_simulation() -> Simulation:
    return Simulation(
        build(
            a_to_b_builder(
                parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
            )
        )
    )


def _counts(seed: int, count: int) -> list[float]:
    return np.random.default_rng(seed).poisson(FLOW, count).astype(float).tolist()


class TestModels:
    def test_every_series_needs_a_model(self) -> None:
        observed = flow_observations(_counts(1, 3), windowed=True) + [
            ObservedDataPoint(at=2, output="A", value=999_980.0)
        ]
        problem = _problem(observed, {"B": PoissonObservations()})

        with pytest.raises(ValueError, match="the series 'A' has no observation model"):
            Calibrator(_flow_simulation(), problem)

    def test_a_noise_parameter_needs_a_model(self) -> None:
        problem = _problem(
            flow_observations(_counts(1, 3), windowed=True),
            {"B": PoissonObservations()},
            noise=("s",),
        )

        with pytest.raises(ValueError, match="Noise parameter 's' is named by no"):
            Calibrator(_flow_simulation(), problem)

    def test_discrete_models_need_integer_values(self) -> None:
        problem = _problem(
            flow_observations([10.0, 10.5], windowed=True),
            {"B": NegativeBinomialObservations(overdispersion=0.1)},
        )

        with pytest.raises(
            ValueError, match=r"Observation 'B' at 2 has the value 10.5; .*whole number"
        ):
            Calibrator(_flow_simulation(), problem)

    @pytest.mark.parametrize(
        ("model_trials", "observation_trials", "message"),
        [(20, 20, "give the trials in one place"), (None, None, "needs trials")],
    )
    def test_binomial_trials_have_one_source(
        self, model_trials: int | None, observation_trials: int | None, message: str
    ) -> None:
        observed = [
            ObservedDataPoint(
                at=1, output="B", value=8.0, window=1, trials=observation_trials
            )
        ]
        problem = _problem(observed, {"B": BinomialObservations(trials=model_trials)})

        with pytest.raises(ValueError, match=message):
            Calibrator(_flow_simulation(), problem)

    def test_a_stochastic_engine_needs_a_loss(self) -> None:
        model = build_stochastic(
            a_to_b_builder(parameters={"k": None}, amounts={"A": 100, "B": 0})
        )
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=1, output="B", value=10.0)],
            parameters=[bounded("k", 0.0, 1.0)],
            observation_models={"B": PoissonObservations()},
            optimization_config=NelderMeadConfig(max_iterations=10),
            seed=1,
        )

        with pytest.raises(ValueError, match="A stochastic engine needs a loss"):
            Calibrator(Simulation(model), problem)

    def test_normalization_needs_a_loss(self) -> None:
        with pytest.raises(ValidationError, match="normalize_observations needs a"):
            _problem(
                flow_observations([1.0], windowed=True),
                {"B": PoissonObservations()},
                normalize_observations=True,
            )

    def test_the_center_of_a_log_normal_has_no_default(self) -> None:
        with pytest.raises(ValidationError, match="center"):
            LogNormalObservations(sigma=0.5)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "model",
        [
            NormalObservations(sd="s"),
            LogNormalObservations(sigma="s", center="mean"),
            LogNormalObservations(sigma="s", center="median"),
            StudentTObservations(df=4.0, scale="s"),
            PoissonObservations(),
            NegativeBinomialObservations(overdispersion="s"),
            BinomialObservations(trials=40),
            BinomialObservations(),
        ],
        ids=lambda model: f"{type(model).__name__}-{getattr(model, 'center', '')}",
    )
    def test_the_loss_of_each_model_is_its_negative_log_likelihood(
        self, model: ObservationModel
    ) -> None:
        names = [name for name in model.model_dump().values() if name == "s"]
        noise = [bounded("s", 0.01, 5.0, kind="noise")] if names else []
        trials = 40 if isinstance(model, BinomialObservations) else None
        observed = [
            ObservedDataPoint(
                at=point.at,
                output="B",
                value=point.value,
                window=1,
                trials=None if model == BinomialObservations(trials=40) else trials,
            )
            for point in flow_observations(_counts(2, 20), windowed=True)
        ]
        calibrator = Calibrator(
            _flow_simulation(),
            CalibrationProblem(
                observed_data=observed,
                parameters=[bounded("c", 1.0, 30.0), *noise],
                observation_models={"B": model},
                optimization_config=NelderMeadConfig(max_iterations=2000),
            ),
        )

        fit = calibrator.run()

        values = [fit.best_parameters[name] for name in calibrator.parameter_names]
        pointwise = calibrator.rust_problem.pointwise_log_likelihood(values)
        assert len(pointwise) == len(observed)
        assert math.fsum(pointwise) == pytest.approx(-fit.final_loss, rel=1e-12)
        assert fit.best_parameters["c"] == pytest.approx(FLOW, rel=0.2)

    def test_the_pointwise_values_of_a_poisson_model_are_its_log_probabilities(
        self,
    ) -> None:
        values = _counts(3, 5)
        calibrator = flow_calibrator(
            values, windowed=True, observation_models={"B": PoissonObservations()}
        )

        pointwise = calibrator.rust_problem.pointwise_log_likelihood([9.0])

        expected = [y * math.log(9.0) - 9.0 - math.lgamma(y + 1) for y in values]
        assert pointwise == pytest.approx(expected, rel=1e-12)

    def test_noise_parameters_are_reported_and_not_written_to_the_model(
        self,
    ) -> None:
        calibrator = flow_calibrator(
            _counts(12, 10),
            windowed=True,
            observation_models={"B": NegativeBinomialObservations(overdispersion="p")},
            noise=[bounded("p", 0.0, 2.0, kind="noise")],
        )
        fit = calibrator.run()
        model = calibrator.simulation.model_definition

        model.apply_calibration_parameters(fit, calibrator.problem)
        report = applied_parameters_report(model, fit, calibrator.problem)

        assert {p.id for p in model.parameters} == {"c"}
        assert report["parameters"]["p"]["value"] == fit.best_parameters["p"]

    @pytest.mark.parametrize("given", ["neither", "both"])
    def test_the_rust_problem_takes_a_loss_or_models(self, given: str) -> None:
        engine = _flow_simulation().engine
        observed = engine_state.rust_observed_data(
            flow_observations([1.0], windowed=True)
        )
        models = [("B", commol_rs.calibration.ObservationModel.poisson())]

        with pytest.raises(ValueError, match="exactly one of them"):
            commol_rs.inference.CalibrationProblem(
                engine,
                observed,
                engine_state.rust_parameters([bounded("c", 0.0, 5.0)]),
                [],
                loss_config=(
                    commol_rs.calibration.LossConfig.sse() if given == "both" else None
                ),
                observation_models=models if given == "both" else None,
            )

    def test_observation_draws_take_one_noise(self) -> None:
        calibrator = flow_calibrator(
            [1.0], windowed=True, observation_models={"B": PoissonObservations()}
        )
        sample = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[1.0]], SampleMethod(name="point")
        )

        observed = engine_state.rust_observed_data(
            flow_observations([1.0], windowed=True)
        )

        with pytest.raises(ValueError, match="take one noise"):
            calibrator.rust_problem.observation_draws(
                sample.rust_sample,
                observed,
                1,
                residual_noise=([1.0], 1, "additive"),
                distribution_noise=True,
            )

    def test_a_loss_has_no_pointwise_log_likelihood(self) -> None:
        calibrator = flow_calibrator([1.0], windowed=True)

        with pytest.raises(ValueError, match="needs observation models"):
            calibrator.rust_problem.pointwise_log_likelihood([1.0])


class TestMaximumLikelihood:
    def test_mle_recovers_the_dispersion(self) -> None:
        phi, count = 0.2, 400
        rng = np.random.default_rng(4)
        size = 1 / phi
        values = rng.negative_binomial(size, size / (size + FLOW), count).astype(float)
        calibrator = flow_calibrator(
            values.tolist(),
            windowed=True,
            observation_models={
                "B": NegativeBinomialObservations(overdispersion="phi")
            },
            noise=[bounded("phi", 0.0, 2.0, kind="noise")],
        )

        fit = calibrator.run().best_parameters

        # The standard error of φ from the curvature of the log-likelihood.
        def log_likelihood(phi: float) -> float:
            return math.fsum(
                calibrator.rust_problem.pointwise_log_likelihood([fit["c"], phi])
            )

        h = 1e-4
        curvature = (
            log_likelihood(fit["phi"] + h)
            - 2 * log_likelihood(fit["phi"])
            + log_likelihood(fit["phi"] - h)
        ) / h**2
        standard_error = 1 / math.sqrt(-curvature)
        assert abs(fit["phi"] - phi) < 3 * standard_error, (fit, standard_error)
        assert fit["c"] == pytest.approx(values.mean(), rel=1e-4)

    def test_normal_mle_is_least_squares(self) -> None:
        values = (FLOW + np.random.default_rng(5).normal(0.0, 2.0, 30)).tolist()
        least_squares = flow_calibrator(values, windowed=True).run()
        likelihood = flow_calibrator(
            values,
            windowed=True,
            observation_models={"B": NormalObservations(sd="sd")},
            noise=[bounded("sd", 0.1, 10.0, kind="noise")],
        ).run()

        assert likelihood.best_parameters["c"] == pytest.approx(
            least_squares.best_parameters["c"], rel=1e-6
        )
        residual_sd = math.sqrt(least_squares.final_loss / len(values))
        assert likelihood.best_parameters["sd"] == pytest.approx(residual_sd, rel=1e-4)

    def test_loss_and_models_are_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="not both"):
            _problem(
                flow_observations([1.0], windowed=True),
                {"B": PoissonObservations()},
                loss_function="sse",
            )

    def test_a_problem_needs_a_loss_or_models(self) -> None:
        with pytest.raises(ValidationError, match="needs a loss_function"):
            CalibrationProblem(
                observed_data=flow_observations([1.0], windowed=True),
                parameters=[bounded("c")],
                optimization_config=NelderMeadConfig(),
            )

    def test_the_log_names_the_likelihood(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        calibrator = flow_calibrator(
            _counts(6, 5),
            windowed=True,
            observation_models={"B": PoissonObservations()},
        )

        with caplog.at_level(logging.INFO, logger="commol.api.calibrator"):
            calibrator.run()

        assert "negative log-likelihood" in caplog.text


class TestBayesianPosterior:
    def test_conjugate_posterior_and_evidence(self) -> None:
        sd, prior_mean, prior_sd = 2.0, 9.0, 3.0
        values = (FLOW + np.random.default_rng(7).normal(0.0, sd, 12)).tolist()
        calibrator = flow_calibrator(
            values, windowed=True, observation_models={"B": NormalObservations(sd=sd)}
        )
        method = _posterior(
            priors=Priors(distributions={"c": Normal(mean=prior_mean, sd=prior_sd)}),
            particles=2000,
        )

        result = calibrator.uncertainty(method)

        y = np.array(values)
        n = y.size
        precision = n / sd**2 + 1 / prior_sd**2
        mean = (y.sum() / sd**2 + prior_mean / prior_sd**2) / precision
        assert result.sample.mean()["c"] == pytest.approx(mean, abs=0.05)
        assert _sd(result.sample) == pytest.approx(1 / math.sqrt(precision), rel=0.1)
        # y ~ N(prior_mean·1, sd²·I + prior_sd²·11ᵀ).
        covariance = sd**2 * np.eye(n) + prior_sd**2 * np.ones((n, n))
        residual = y - prior_mean
        log_evidence = -0.5 * (
            n * math.log(2 * math.pi)
            + np.linalg.slogdet(covariance)[1]
            + residual @ np.linalg.solve(covariance, residual)
        )
        assert result.log_evidence == pytest.approx(log_evidence, abs=0.1)
        assert result.sample.method.name == "bayesian_posterior"

    def test_gibbs_posterior_with_eta_one_is_the_posterior(self) -> None:
        values = _counts(8, 20)
        calibrator = flow_calibrator(
            values,
            windowed=True,
            observation_models={"B": NegativeBinomialObservations(overdispersion=0.05)},
        )

        bayes = calibrator.uncertainty(_posterior(particles=1000))
        gibbs = calibrator.uncertainty(
            GibbsPosterior(
                priors=Priors.uniform_on_bounds(),
                eta=1.0,
                particles=1000,
                target_cess=0.5,
                resample_threshold=0.5,
                stay_probability=0.05,
            )
        )

        spread = _sd(bayes.sample)
        assert gibbs.sample.mean()["c"] == pytest.approx(
            bayes.sample.mean()["c"], abs=0.2 * spread
        )
        assert _sd(gibbs.sample) == pytest.approx(spread, rel=0.15)

    def test_methods_of_a_loss_refuse_a_likelihood(self) -> None:
        calibrator = flow_calibrator(
            _counts(9, 5),
            windowed=True,
            observation_models={"B": PoissonObservations()},
        )
        abc = AbcSmc(
            priors=Priors.uniform_on_bounds(),
            distance=LossDistance(),
            particles=10,
            target_cess=0.5,
            min_acceptance=0.01,
            stay_probability=0.05,
        )

        with pytest.raises(ValueError, match="LossDistance\\(\\) needs a loss"):
            calibrator.uncertainty(abc)

    def test_a_bayesian_posterior_needs_observation_models(self) -> None:
        calibrator = flow_calibrator([1.0, 2.0], windowed=True)

        with pytest.raises(ValueError, match="needs observation models"):
            calibrator.uncertainty(_posterior(particles=10))


class TestPrediction:
    def test_randomized_pit_of_counts_is_uniform(self) -> None:
        values = _counts(10, 400)
        calibrator = flow_calibrator(
            values, windowed=True, observation_models={"B": PoissonObservations()}
        )
        sample = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[FLOW]] * 300, SampleMethod(name="truth")
        )
        predictive = Predictive(calibrator, sample, [0, 1])
        observations = flow_observations(values, windowed=True)

        pit = scores.pit(predictive, observations, DistributionNoise(), seed=1)

        # Ten bins of 40 expected values each: a chi-square of 9 degrees of
        # freedom, whose 0.999 quantile is 27.9.
        counts, _ = np.histogram(pit.per_observation, bins=10, range=(0.0, 1.0))
        chi_square = float(((counts - 40.0) ** 2 / 40.0).sum())
        assert chi_square < 27.9, counts

    def test_distribution_noise_draws_around_each_draw(self) -> None:
        calibrator = flow_calibrator(
            [10.0], windowed=True, observation_models={"B": PoissonObservations()}
        )
        sample = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[4.0]] * 4000, SampleMethod(name="point")
        )
        observation = flow_observations([0.0], windowed=True)

        draws = Predictive(calibrator, sample, [0, 1]).at_observations(
            observation, DistributionNoise()
        )

        shown = np.array([row[0] for row in draws.values])
        assert shown.mean() == pytest.approx(4.0, abs=0.15)
        assert shown.var() == pytest.approx(4.0, rel=0.1)

    def test_distribution_noise_needs_observation_models(self) -> None:
        calibrator = flow_calibrator([1.0], windowed=True)
        sample = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[1.0]], SampleMethod(name="point")
        )

        with pytest.raises(ValueError, match="DistributionNoise\\(\\) needs"):
            Predictive(calibrator, sample, [0, 1]).at_observations(
                flow_observations([1.0], windowed=True), DistributionNoise()
            )


class TestParametricBootstrap:
    def test_parametric_bootstrap_matches_the_analytic_variance(self) -> None:
        sd, count, replicates = 2.0, 25, 300
        values = (FLOW + np.random.default_rng(11).normal(0.0, sd, count)).tolist()
        calibrator = flow_calibrator(
            values, windowed=False, observation_models={"B": NormalObservations(sd=sd)}
        )

        result = calibrator.uncertainty(ParametricBootstrap(replicates=replicates))

        # The prediction is c·t at t = 1, …, n: Var(ĉ) = sd² / Σ t².
        positions = np.arange(1, count + 1)
        variance = sd**2 / float(positions @ positions)
        sampled = _sd(result.sample) ** 2
        error = variance * math.sqrt(2 / (replicates - 1))
        assert abs(sampled - variance) < 3 * error, (sampled, variance)
        assert result.diagnostics.block_length is None
        assert result.sample.method.name == "parametric_bootstrap"

    def test_a_parametric_bootstrap_needs_observation_models(self) -> None:
        calibrator = flow_calibrator([1.0, 2.0, 3.0], windowed=True)

        with pytest.raises(
            ValueError, match="ParametricBootstrap\\(\\.\\.\\.\\) needs"
        ):
            calibrator.uncertainty(ParametricBootstrap(replicates=2))
