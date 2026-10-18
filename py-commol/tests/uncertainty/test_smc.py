"""Sequential Monte Carlo through ``Calibrator.uncertainty``: priors, the
Gibbs posterior, ABC and their results.

Most tests fit a constant flow ``c`` (``flow_calibrator`` with windowed
observations): each observation is ``c`` plus noise, so with the sum of
squares and ``η = 1/(2σ²)`` the Gibbs posterior is the normal posterior of a
mean, ``N(ȳ, σ²/n)``.
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    AbcRejection,
    AbcSmc,
    Beta,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    EtaCalibration,
    GibbsPosterior,
    LogUniform,
    LossDistance,
    NelderMeadConfig,
    Normal,
    ObservedDataPoint,
    Priors,
    SeriesDistance,
    Simulation,
    SmcResult,
    Uniform,
)
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    build_stochastic,
    flow_calibrator,
    flow_observations,
)

NOISE = 2.0
"""The standard deviation of the noise of the flow data."""


def _flow_data(count: int, seed: int, level: float = 10.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return (level + rng.normal(0.0, NOISE, count)).tolist()


def _gibbs(**settings: object) -> GibbsPosterior:
    defaults: dict[str, object] = {
        "priors": Priors.uniform_on_bounds(),
        "eta": 1 / (2 * NOISE**2),
        "particles": 200,
        "target_cess": 0.5,
        "resample_threshold": 0.5,
        "stay_probability": 0.05,
    }
    return GibbsPosterior.model_validate(defaults | settings)


def _unbounded_calibrator(low: float, high: float) -> Calibrator:
    """The flow model with the bounds ``[low, high]`` for ``c``."""
    model = build(a_to_b_builder(parameters={"c": None}, rate="c * A / A", total=1e6))
    problem = CalibrationProblem(
        observed_data=flow_observations(_flow_data(5, 1), windowed=True),
        parameters=[
            CalibrationParameter(
                id="c",
                parameter_type="parameter",
                min_bound=low,
                max_bound=high,
                initial_guess=10.0,
            )
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=200),
        seed=1,
    )
    return Calibrator(Simulation(model), problem)


class TestPriors:
    def test_priors_are_required_for_every_parameter(self) -> None:
        settings = _gibbs().model_dump(exclude={"priors"})
        with pytest.raises(ValueError, match=r"priors.*Priors\.uniform_on_bounds\(\)"):
            GibbsPosterior.model_validate(settings)
        with pytest.raises(ValueError, match=r"priors.*Priors\.uniform_on_bounds\(\)"):
            AbcSmc.model_validate(
                {
                    "distance": LossDistance(),
                    "particles": 100,
                    "target_cess": 0.5,
                    "min_acceptance": 0.05,
                    "stay_probability": 0.05,
                }
            )

        calibrator = flow_calibrator(_flow_data(5, 1), windowed=True)
        with pytest.raises(
            ValueError, match=r"'c' has no prior.*Priors\.uniform_on_bounds\(\)"
        ):
            calibrator.uncertainty(_gibbs(priors={}))
        with pytest.raises(ValueError, match=r"'k' is not a calibrated parameter"):
            calibrator.uncertainty(_gibbs(priors={"c": Uniform(), "k": Uniform()}))

    def test_a_prior_on_the_bounds_needs_finite_bounds(self) -> None:
        open_above = _unbounded_calibrator(0.0, math.inf)
        for prior in (Uniform(), Beta(a=2.0, b=2.0), LogUniform()):
            with pytest.raises(
                ValueError,
                match=(
                    rf"{type(prior).__name__}\(\) of 'c' needs finite bounds.*"
                    r"\[0\.0, inf\].*Normal, LogNormal, Gamma"
                ),
            ):
                open_above.uncertainty(_gibbs(priors={"c": prior}))

        from_zero = _unbounded_calibrator(0.0, 50.0)
        with pytest.raises(
            ValueError,
            match=(
                r"LogUniform\(\) of 'c' needs a lower bound above 0.*"
                r"\[0\.0, 50\.0\].*Uniform, Beta, Normal, LogNormal, Gamma"
            ),
        ):
            from_zero.uncertainty(_gibbs(priors={"c": LogUniform()}))

    def test_a_prior_takes_the_bounds_of_its_parameter(self) -> None:
        bounds = (0.0, 50.0)
        half = Normal(mean=0.0, sd=1.0)
        assert half.cdf(1.0, (0.0, math.inf)) == pytest.approx(
            2 * 0.8413447460685429 - 1
        )
        assert Uniform().quantile(0.25, bounds) == pytest.approx(12.5)
        assert Uniform().log_density(60.0, bounds) == -math.inf
        draws = Beta(a=2.0, b=5.0).sample(500, bounds, seed=3)
        assert len(draws) == 500
        assert all(0.0 <= draw <= 50.0 for draw in draws)
        assert draws == Beta(a=2.0, b=5.0).sample(500, bounds, seed=3)
        assert np.mean(draws) == pytest.approx(50.0 * 2 / 7, rel=0.05)

    def test_a_distribution_checks_its_parameters(self) -> None:
        with pytest.raises(ValidationError, match="sd"):
            Normal(mean=0.0, sd=0.0)
        with pytest.raises(ValidationError, match="a"):
            Beta(a=-1.0, b=1.0)


def _flow(values: list[float], loss_function: str = "sse") -> Calibrator:
    """``flow_calibrator`` with windowed observations and `loss_function`."""
    model = build(a_to_b_builder(parameters={"c": None}, rate="c * A / A", total=1e6))
    problem = CalibrationProblem(
        observed_data=flow_observations(values, windowed=True),
        parameters=[bounded("c", 0.0, 50.0, initial_guess=25.0)],
        loss_function=loss_function,
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-14),
        seed=1,
    )
    return Calibrator(Simulation(model), problem)


UNITS = 1000
"""Units of the binomial model: each moves from A to B with probability
``k`` in one step."""


def _binomial(
    observed: int, seed: int = 3, replicates: int = 1, loss_function: str = "mae"
) -> Calibrator:
    """
    ``k`` of a chain binomial from ``observed`` units in B after one step.

    With a uniform prior on ``[0, 1]`` the posterior of ``k`` is
    ``Beta(observed + 1, UNITS − observed + 1)``.
    """
    model = build_stochastic(
        a_to_b_builder(parameters={"k": None}, total=UNITS), "StochasticSteps"
    )
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="B", value=float(observed))],
        parameters=[bounded("k", 0.0, 1.0, initial_guess=0.5)],
        loss_function=loss_function,
        replicates=replicates,
        replicate_summary="mean",
        optimization_config=NelderMeadConfig(max_iterations=100),
        seed=seed,
    )
    return Calibrator(Simulation(model), problem)


def _abc(**settings: object) -> AbcSmc:
    defaults: dict[str, object] = {
        "priors": Priors.uniform_on_bounds(),
        "distance": LossDistance(),
        "particles": 300,
        "target_cess": 0.5,
        "min_acceptance": 0.02,
        "stay_probability": 0.05,
    }
    return AbcSmc.model_validate(defaults | settings)


def _mean_and_sd(result: SmcResult, name: str) -> tuple[float, float]:
    mean = result.sample.mean()[name]
    index = result.sample.names.index(name)
    return mean, math.sqrt(result.sample.covariance()[index][index])


class TestSampler:
    def test_stay_probability_is_required(self) -> None:
        settings = _gibbs().model_dump(exclude={"stay_probability"})
        with pytest.raises(ValidationError, match="stay_probability"):
            GibbsPosterior.model_validate(settings)
        abc_settings = _abc().model_dump(exclude={"stay_probability"})
        with pytest.raises(ValidationError, match="stay_probability"):
            AbcSmc.model_validate(abc_settings)
        for value in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValidationError, match="stay_probability"):
                _gibbs(stay_probability=value)


class TestGibbsPosterior:
    def test_gibbs_posterior_with_the_right_eta_is_the_posterior(self) -> None:
        data = _flow_data(20, 4)
        result = flow_calibrator(data, windowed=True).uncertainty(
            _gibbs(particles=1000)
        )

        mean, sd = _mean_and_sd(result, "c")
        posterior_sd = NOISE / math.sqrt(len(data))
        assert mean == pytest.approx(np.mean(data), abs=0.1 * posterior_sd)
        assert sd == pytest.approx(posterior_sd, rel=0.1)
        assert result.eta == 1 / (2 * NOISE**2)
        assert result.fit is None

    def test_information_matching_recovers_eta(self) -> None:
        count = 200
        result = _flow(_flow_data(count, 5)).uncertainty(
            _gibbs(eta=EtaCalibration.information_matching())
        )

        # η̂ = 1/(2σ̂²): its relative error is about √(2/n).
        assert result.eta == pytest.approx(
            1 / (2 * NOISE**2), rel=3 * math.sqrt(2 / count)
        )
        assert result.fit is not None
        assert result.coverage is None
        assert result.sample.method.settings["eta"] == {"kind": "information_matching"}

    def test_eta_is_required_and_positive(self) -> None:
        settings = _gibbs().model_dump(exclude={"eta"})
        hint = r"eta.*EtaCalibration\.information_matching\(\).*bootstrap_coverage"
        with pytest.raises(ValueError, match=hint):
            GibbsPosterior.model_validate(settings)
        for eta in (0.0, -1.0):
            with pytest.raises(ValueError, match=hint):
                _gibbs(eta=eta)

    def test_information_matching_needs_a_smooth_sum_loss(self) -> None:
        method = _gibbs(eta=EtaCalibration.information_matching())
        for loss in ("rmse", "mae"):
            with pytest.raises(
                ValueError,
                match=r'loss_function="sse".*EtaCalibration\.bootstrap_coverage',
            ):
                _flow(_flow_data(10, 6), loss).uncertainty(method)

        with pytest.raises(
            ValueError,
            match=r"deterministic engine.*EtaCalibration\.bootstrap_coverage",
        ):
            _binomial(300, loss_function="sse").uncertainty(method)

    def test_bootstrap_coverage_finds_an_eta_whose_intervals_cover(self) -> None:
        result = _flow(_flow_data(20, 7)).uncertainty(
            _gibbs(
                eta=EtaCalibration.bootstrap_coverage(
                    replicates=8, block_length=1, confidence_level=0.8
                ),
                particles=200,
            )
        )

        # η sits between two thresholds of the eight replicates: their
        # coverage is within one replicate of the level.
        assert result.coverage is not None
        assert abs(result.coverage - 0.8) <= 1 / 8
        assert result.fit is not None
        # The bootstrap spread of a mean is that of its posterior: η is near
        # 1/(2σ²).
        assert 0.25 < result.eta * 2 * NOISE**2 < 4.0


class TestAbc:
    def test_abc_approaches_the_posterior_as_tolerance_falls(self) -> None:
        observed = 300
        calibrator = _binomial(observed)
        early = calibrator.uncertainty(_abc(min_acceptance=0.3))
        late = calibrator.uncertainty(_abc(min_acceptance=0.01))

        a, b = observed + 1, UNITS - observed + 1
        posterior_mean = a / (a + b)
        posterior_sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        assert late.levels[-1] < early.levels[-1]
        late_mean, late_sd = _mean_and_sd(late, "k")
        _, early_sd = _mean_and_sd(early, "k")
        assert late_mean == pytest.approx(posterior_mean, abs=0.3 * posterior_sd)
        assert late_sd == pytest.approx(posterior_sd, rel=0.25)
        assert abs(late_sd - posterior_sd) < abs(early_sd - posterior_sd)

    def test_abc_with_one_seed_gives_one_result(self) -> None:
        first = _binomial(300, seed=9).uncertainty(_abc(particles=100))
        second = _binomial(300, seed=9).uncertainty(_abc(particles=100))
        other = _binomial(300, seed=10).uncertainty(_abc(particles=100))

        assert first.model_dump() == second.model_dump()
        assert first.sample.values != other.sample.values

    def test_loss_distance_reads_one_replicate(self) -> None:
        # Fifty replicates summarized by their mean would give distances
        # that are not whole numbers; one replicate of whole units gives
        # |B − observed|, a whole number.
        calibrator = _binomial(300, replicates=50)
        result = calibrator.uncertainty(
            AbcRejection(
                priors=Priors.uniform_on_bounds(),
                distance=LossDistance(),
                draws=200,
                quantile=0.5,
            )
        )

        distances = [d for d in result.kept if d is not None]
        assert len(distances) == 200
        assert all(d == round(d) for d in distances)
        assert any(
            "replicates and replicate_summary do not apply" in note
            for note in result.diagnostics.notes
        )

    def test_series_distance_compares_series_on_one_scale(self) -> None:
        def distance(factor: float, kind: SeriesDistance | LossDistance) -> float:
            flows = _flow_data(5, 8)
            amounts = [10.0 * t for t in range(1, 6)]
            model = build(
                a_to_b_builder(parameters={"c": None}, rate="c * A / A", total=1e6)
            )
            observed = flow_observations(flows, windowed=True) + [
                ObservedDataPoint(
                    at=t,
                    output="B",
                    label="stock",
                    value=factor * amount,
                    factor_id="f",
                )
                for t, amount in zip(range(1, 6), amounts, strict=True)
            ]
            problem = CalibrationProblem(
                observed_data=observed,
                parameters=[
                    bounded("c", 9.0, 9.0 + 1e-9),
                    bounded("f", factor, factor * (1 + 1e-12), kind="factor"),
                ],
                loss_function="sse",
                optimization_config=NelderMeadConfig(max_iterations=10),
                seed=1,
            )
            result = Calibrator(Simulation(model), problem).uncertainty(
                AbcRejection(
                    priors=Priors.uniform_on_bounds(),
                    distance=kind,
                    draws=4,
                    quantile=0.5,
                )
            )
            return next(d for d in result.kept if d is not None)

        series = SeriesDistance(metric="rmse")
        assert distance(1000.0, series) == pytest.approx(
            distance(1.0, series), rel=1e-6
        )
        # The raw loss grows with the unit of the scaled series.
        assert distance(1000.0, LossDistance()) > 1e5 * distance(1.0, LossDistance())

    def test_abc_rejection_keeps_the_closest_draws(self) -> None:
        result = _binomial(300).uncertainty(
            AbcRejection(
                priors=Priors.uniform_on_bounds(),
                distance=LossDistance(),
                draws=400,
                quantile=0.1,
            )
        )

        kept = [w > -math.inf for w in result.sample.log_weights]
        assert sum(kept) >= 40
        assert result.levels == [
            max(
                d for d, k in zip(result.kept, kept, strict=True) if k and d is not None
            )
        ]
        assert result.diagnostics.move_counts == [0]


class TestResult:
    def test_result_reports_each_step(self) -> None:
        gibbs = flow_calibrator(_flow_data(10, 2), windowed=True).uncertainty(_gibbs())
        abc = _binomial(300).uncertainty(_abc(particles=100))

        for result, falling in ((gibbs, False), (abc, True)):
            steps = len(result.levels)
            diagnostics = result.diagnostics
            assert steps >= 1
            assert len(diagnostics.effective_sizes) == steps
            assert len(diagnostics.acceptance_rates) == steps
            assert len(diagnostics.move_counts) == steps
            assert len(diagnostics.resampled) == steps
            assert diagnostics.evaluations >= 100
            pairs = zip(result.levels, result.levels[1:])
            assert all((b < a) if falling else (b > a) for a, b in pairs)
        assert gibbs.levels[-1] == 1.0
        assert gibbs.diagnostics.stop_reason is None
        assert gibbs.log_evidence < 0.0
