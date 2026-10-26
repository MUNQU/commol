"""Local approximations through ``Calibrator.uncertainty``: the Laplace
approximation and the delta method, the sandwich, the Laplace mixture, and
information matching from the curvature.

Most tests fit a constant flow ``c`` (``flow_calibrator``): with windowed
observations each prediction is ``c``, so with a normal model of known ``sd``
the information of ``c`` is ``n / sd²``; without windows the prediction at
``t`` is ``c·t``, a line through the origin.
"""

import logging
import math

import numpy as np
import pytest
from pydantic import BaseModel

from commol import (
    CalibrationConstraint,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    Simulation,
)
from commol._calibration.method import UncertaintyMethod
from commol.commol_rs import _commol_rs as commol_rs
from commol.distributions import Priors
from commol.engines.engine import ODESolver
from commol.local import (
    GibbsTarget,
    Laplace,
    LaplaceMixture,
    LikelihoodTarget,
    Sandwich,
)
from commol.local.results import (
    DeltaBand,
    ImportanceResult,
    LaplaceResult,
    SandwichResult,
)
from commol.observations import NormalObservations
from commol.smc import GibbsPosterior, InformationMatchingEta
from commol.uncertainty import Predictive
from commol.uncertainty.predictive import by_output
from tests.support import (
    TWO_WELLS_FIT,
    a_to_b_builder,
    bounded,
    build,
    build_ode,
    flow_calibrator,
    flow_observations,
    two_wells_calibrator,
)


def _normal_data(count: int, seed: int, sd: float, level: float = 10.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return (level + rng.normal(0.0, sd, count)).tolist()


def _square_calibrator(values: list[float], low: float, high: float) -> Calibrator:
    """A flow ``c²`` from A to B, with a normal model of ``sd = 1``: two
    optima, ``±√ȳ``, when the bounds of ``c`` hold both."""
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
    problem = CalibrationProblem(
        observed_data=flow_observations(values, windowed=True),
        parameters=[bounded("c", low, high, initial_guess=1.0)],
        observation_models={"B": NormalObservations(sd=1.0)},
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-12),
        seed=2,
    )
    return Calibrator(Simulation(model), problem)


class TestCurvature:
    def test_a_fit_on_its_bound_points_to_a_profile(self) -> None:
        # Values below 5 put the fit of c on its lower bound.
        calibrator = flow_calibrator(
            [1.0, 2.0, 1.0],
            windowed=True,
            bounds=(5.0, 50.0),
            observation_models={"B": NormalObservations(sd=1.0)},
        )

        with pytest.raises(
            ValueError, match=r"'c' is on its bound 5.*profile likelihood"
        ) as error:
            calibrator.uncertainty(Laplace(draws=10))

        message = str(error.value)
        assert "ProfileLikelihood" in message
        assert "#" not in message

    def test_the_hessian_of_a_correct_normal_model_is_its_information(self) -> None:
        # With a known sd, −log L is quadratic in c: its Hessian is n / sd².
        calibrator = flow_calibrator(
            _normal_data(8, 7, sd=1.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )

        expected = calibrator.uncertainty(Laplace(draws=10))
        observed = calibrator.uncertainty(Laplace(draws=10, curvature="hessian"))

        assert observed.covariance[0][0] == pytest.approx(
            expected.covariance[0][0], rel=1e-5
        )
        assert observed.notes == []

    def test_a_hessian_that_steps_past_a_constraint_fails(self) -> None:
        # The flows ask for c = 12; the constraint holds the fit at c = 10,
        # and a difference of the Hessian steps past it.
        model = build(
            a_to_b_builder(
                parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
            )
        )
        problem = CalibrationProblem(
            observed_data=flow_observations([12.0, 13.0, 11.0, 12.5], windowed=True),
            parameters=[bounded("c", 0.0, 50.0, initial_guess=5.0)],
            observation_models={"B": NormalObservations(sd=1.0)},
            constraints=[CalibrationConstraint(id="at_most_10", expression="10 - c")],
            optimization_config=NelderMeadConfig(max_iterations=1000),
            seed=1,
        )
        calibrator = Calibrator(Simulation(model), problem)

        assert calibrator.run().best_parameters["c"] == pytest.approx(10.0)
        with pytest.raises(
            RuntimeError,
            match="The Hessian of the objective at the fit failed: Infeasible point",
        ):
            calibrator.uncertainty(Laplace(draws=10, curvature="hessian"))

    def test_an_uncertain_curvature_is_noted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # An ODE solved to rtol = 1e-2 makes the differences of its runs
        # coarse: the error of the curvature is a large share of it.
        model = build_ode(
            a_to_b_builder(
                parameters={"c": None}, rate="c * A / 1000", amounts={"A": 1000, "B": 0}
            ),
            ODESolver(rtol=1e-2, atol=1e-2),
        )
        values = [1000 * (1 - math.exp(-0.1 * t)) for t in range(1, 11)]
        problem = CalibrationProblem(
            observed_data=flow_observations(values, windowed=False),
            parameters=[bounded("c", 0.0, 500.0, initial_guess=50.0)],
            observation_models={"B": NormalObservations(sd=1.0)},
            optimization_config=NelderMeadConfig(max_iterations=500),
            seed=1,
        )

        with caplog.at_level(logging.WARNING):
            result = Calibrator(Simulation(model), problem).uncertainty(
                Laplace(draws=10)
            )

        assert len(result.notes) == 1
        assert "The curvature of ('c', 'c')" in result.notes[0]
        assert result.notes[0] in caplog.text


class TestLaplace:
    def test_delta_band_of_a_linear_model_equals_the_draws(self) -> None:
        # B(t) = c·t and A(t) = 10⁶ − c·t: the band of the draws is the
        # linearized band, up to the Monte Carlo error of its quantiles.
        draws, level = 4000, 0.9
        values = [10.0 * t + noise for t, noise in enumerate([0.3, -0.5, 0.2] * 4, 1)]
        calibrator = flow_calibrator(
            values,
            windowed=False,
            observation_models={"B": NormalObservations(sd=1.0)},
        )
        result = calibrator.uncertainty(Laplace(draws=draws))
        positions = [float(t) for t in range(13)]

        band = result.delta_band(calibrator, positions, level)
        drawn = Predictive(calibrator, result.sample, positions).band(level)

        # The error of a quantile at 0.05 of n normal draws, in standard
        # deviations: √(p(1 − p)/n) / φ(z_p).
        quantile_error = math.sqrt(0.05 * 0.95 / draws) / 0.10313
        sd_c = 1.0 / math.sqrt(sum(t * t for t in range(1, 13)))
        for output in ("A", "B"):
            for t, (lower, upper, low, high) in enumerate(
                zip(
                    band.lower[output],
                    band.upper[output],
                    drawn.lower[output],
                    drawn.upper[output],
                    strict=True,
                )
            ):
                tolerance = 4 * quantile_error * sd_c * t + 1e-9
                assert lower == pytest.approx(low, abs=tolerance), (output, t)
                assert upper == pytest.approx(high, abs=tolerance), (output, t)
        half = 1.6449 * sd_c * 12
        assert band.upper["B"][12] - band.center["B"][12] == pytest.approx(
            half, rel=1e-3
        )

    def test_a_result_read_from_json_gives_the_same_delta_band(self) -> None:
        calibrator = flow_calibrator(
            [9.0, 11.0, 10.0],
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )
        result = calibrator.uncertainty(Laplace(draws=10))
        read = LaplaceResult.model_validate_json(result.model_dump_json())

        band = read.delta_band(calibrator, [1.0, 2.0], 0.9)

        assert band == result.delta_band(calibrator, [1.0, 2.0], 0.9)
        assert band.lower["B"][1] < band.center["B"][1] < band.upper["B"][1]

    def test_a_band_needs_a_result_of_its_problem(self) -> None:
        models = {"B": NormalObservations(sd="sd")}
        noise = [bounded("sd", 0.1, 10.0, kind="noise")]
        other = flow_calibrator(
            [9.0, 11.0, 10.0], windowed=True, observation_models=models, noise=noise
        )
        result = other.uncertainty(Laplace(draws=10))
        calibrator = flow_calibrator(
            [9.0, 11.0, 10.0],
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )

        with pytest.raises(ValueError, match=r"parameters \['c', 'sd'\]"):
            result.delta_band(calibrator, [1.0], 0.9)

    def test_a_band_takes_the_positions_of_the_engine(self) -> None:
        calibrator = flow_calibrator(
            [9.0, 11.0, 10.0],
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )
        result = calibrator.uncertainty(Laplace(draws=10))

        with pytest.raises(ValueError, match="only at integer steps"):
            result.delta_band(calibrator, [0.5], 0.9)
        with pytest.raises(ValueError, match="1 rows of 1 values"):
            commol_rs.inference.delta_band(
                calibrator._engine_problem, [10.0], [[1.0, 0.0]], None, [1.0], 0.9
            )

    def test_laplace_refuses_a_loss(self) -> None:
        calibrator = flow_calibrator([9.0, 11.0, 10.0], windowed=True)

        with pytest.raises(
            ValueError,
            match=r"NormalObservations\(sd=\"sd\"\) with a noise parameter.*Sandwich",
        ):
            calibrator.uncertainty(Laplace(draws=10))


def test_a_delta_band_comes_from_the_local_module() -> None:
    calibrator = flow_calibrator(
        [9.0, 11.0, 10.0],
        windowed=True,
        observation_models={"B": NormalObservations(sd=1.0)},
    )
    result = calibrator.uncertainty(Sandwich(draws=10, block_length=1))

    band = result.delta_band(calibrator, [1.0, 2.0], 0.9)

    outputs, center, lower, upper = commol_rs.inference.delta_band(
        calibrator._engine_problem,
        [result.fit.best_parameters["c"]],
        result.covariance,
        result.degrees_of_freedom(),
        [1.0, 2.0],
        0.9,
    )
    assert not hasattr(calibrator, "delta_band")
    assert band == DeltaBand(
        confidence_level=0.9,
        positions=[1.0, 2.0],
        center=by_output(outputs, center),
        lower=by_output(outputs, lower),
        upper=by_output(outputs, upper),
    )


class TestSandwich:
    def test_sandwich_equals_laplace_under_gaussian_noise(self) -> None:
        count = 200
        calibrator = flow_calibrator(
            _normal_data(count, 3, sd=1.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )

        laplace = calibrator.uncertainty(Laplace(draws=10))
        sandwich = calibrator.uncertainty(Sandwich(block_length=1, draws=10))

        # B/H is the mean of the squared residuals over sd²: its relative
        # error is about √(2/n).
        assert sandwich.covariance[0][0] == pytest.approx(
            laplace.covariance[0][0], rel=3 * math.sqrt(2 / count)
        )
        assert sandwich.blocks == count

    def test_a_sandwich_band_reads_a_t_of_its_blocks(self) -> None:
        # Four blocks: the band is the prediction ± t₃(0.95) times its
        # standard deviation, J Σ Jᵀ with J = t · dc/du at position t.
        calibrator = flow_calibrator(_normal_data(8, 10, sd=1.0), windowed=True)
        result = calibrator.uncertainty(Sandwich(block_length=2, draws=10))
        c = result.fit.best_parameters["c"]
        derivative = c * (50.0 - c) / 50.0

        band = result.delta_band(calibrator, [3.0], 0.9)

        t_quantile = 2.353363434801823
        half = t_quantile * 3.0 * derivative * math.sqrt(result.covariance[0][0])
        assert result.blocks == 4
        assert band.upper["B"][0] - band.center["B"][0] == pytest.approx(half, rel=1e-6)

    def test_sandwich_refuses_too_few_blocks(self) -> None:
        calibrator = flow_calibrator(_normal_data(6, 4, sd=1.0), windowed=True)

        with pytest.raises(
            ValueError, match=r"block_length 6 makes 1 block for 1 parameter.*shorter"
        ):
            calibrator.uncertainty(Sandwich(block_length=6, draws=10))

    def test_sandwich_refuses_the_absolute_error(self) -> None:
        model = build(
            a_to_b_builder(
                parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
            )
        )
        problem = CalibrationProblem(
            observed_data=flow_observations(_normal_data(10, 5, sd=1.0), windowed=True),
            parameters=[bounded("c", 0.0, 50.0, initial_guess=25.0)],
            loss_function="mae",
            optimization_config=NelderMeadConfig(max_iterations=500),
            seed=1,
        )
        calibrator = Calibrator(Simulation(model), problem)

        with pytest.raises(ValueError, match=r"density of the residuals.*bootstrap"):
            calibrator.uncertainty(Sandwich(block_length=1, draws=10))


class TestResults:
    @pytest.mark.parametrize(
        ("method", "result_type"),
        [
            (Laplace(draws=10), LaplaceResult),
            (Sandwich(block_length=2, draws=10), SandwichResult),
            (
                LaplaceMixture(
                    starts=4,
                    draws=100,
                    target=LikelihoodTarget(priors=Priors.uniform_on_bounds()),
                ),
                ImportanceResult,
            ),
            (
                LaplaceMixture(
                    starts=4,
                    draws=100,
                    target=GibbsTarget(priors=Priors.uniform_on_bounds(), eta=0.5),
                ),
                ImportanceResult,
            ),
        ],
        ids=["laplace", "sandwich", "mixture_of_a_likelihood", "mixture_of_a_loss"],
    )
    def test_each_method_returns_its_result(
        self, method: UncertaintyMethod[BaseModel], result_type: type[BaseModel]
    ) -> None:
        data = _normal_data(12, 6, sd=1.0)
        models = (
            None
            if isinstance(method, LaplaceMixture)
            and isinstance(method.target, GibbsTarget)
            else {"B": NormalObservations(sd=1.0)}
        )
        calibrator = flow_calibrator(data, windowed=True, observation_models=models)

        result = calibrator.uncertainty(method)

        assert type(result) is result_type
        assert result.sample.names == ["c"]
        assert result.sample.method.settings["seed"] == calibrator.seed
        mean = float(np.mean(data))
        assert result.sample.mean()["c"] == pytest.approx(mean, abs=1.0)

    def test_an_end_on_a_bound_gives_no_component(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # c² near 4 has the optima ±2; the bound 1.5 holds the positive one.
        calibrator = _square_calibrator([4.2, 3.8, 4.1, 3.9], -4.0, 1.5)
        method = LaplaceMixture(
            starts=8,
            draws=200,
            target=LikelihoodTarget(priors=Priors.uniform_on_bounds()),
        )

        with caplog.at_level(logging.WARNING):
            result = calibrator.uncertainty(method)

        assert len(result.components) == 1
        assert result.components[0].center["c"] == pytest.approx(-2.0, abs=0.1)
        assert any("'c' is on its bound" in note for note in result.notes)
        assert "'c' is on its bound" in caplog.text
        # The end on the bound fits worse than the kept component: only a note.
        assert result.reliable


class TestStarts:
    @pytest.mark.parametrize(
        "method",
        [Laplace(starts=3, draws=10), Sandwich(starts=3, block_length=1, draws=10)],
        ids=["laplace", "sandwich"],
    )
    def test_laplace_and_sandwich_fit_from_their_starts(
        self, method: Laplace | Sandwich
    ) -> None:
        calibrator = two_wells_calibrator()
        # One start stays at the local minimum of the initial guess.
        assert calibrator.run().best_parameters["c"] == pytest.approx(-1.0, abs=1e-6)

        result = calibrator.uncertainty(method)

        assert len(result.fit.starts) == 3
        assert result.fit == calibrator.run(starts=3)
        assert result.fit.best_parameters["c"] == pytest.approx(TWO_WELLS_FIT, abs=1e-5)

    def test_a_laplace_mixture_fits_from_one_start_by_default(self) -> None:
        calibrator = flow_calibrator(
            _normal_data(12, 6, sd=1.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd=1.0)},
        )
        method = LaplaceMixture(
            draws=25, target=LikelihoodTarget(priors=Priors.uniform_on_bounds())
        )

        result = calibrator.uncertainty(method)

        assert len(result.fit.starts) == 1
        assert result.fit == calibrator.run()


def _matching(block_length: int) -> GibbsPosterior:
    return GibbsPosterior(
        priors=Priors.uniform_on_bounds(),
        eta=InformationMatchingEta(block_length=block_length),
        particles=50,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.05,
    )


class TestInformationMatching:
    def test_information_matching_of_a_correct_likelihood_is_one(self) -> None:
        count = 200
        calibrator = flow_calibrator(
            _normal_data(count, 8, sd=2.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd=2.0)},
        )

        result = calibrator.uncertainty(_matching(block_length=1))

        # η = n sd² / Σ r²: its relative error is about √(2/n).
        assert result.eta == pytest.approx(1.0, rel=3 * math.sqrt(2 / count))

    def test_information_matching_takes_a_likelihood(self) -> None:
        # The noise parameter sd adds its own information, 2n / sd², and the
        # variance of its scores.
        count = 200
        calibrator = flow_calibrator(
            _normal_data(count, 9, sd=2.0),
            windowed=True,
            observation_models={"B": NormalObservations(sd="sd")},
            noise=[bounded("sd", 0.1, 10.0, kind="noise")],
        )

        result = calibrator.uncertainty(_matching(block_length=4))

        assert result.fit is not None
        assert result.eta == pytest.approx(1.0, rel=6 * math.sqrt(2 / count))
