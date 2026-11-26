"""The dependence analyses read the series of their values and of the
residuals of a predictive.

`series_dependence` reads any values, one per observation; its statistics
are checked against their closed forms. `residual_dependence` reads the
residuals of a predictive: at a fit of a normal model of ``sd = 1`` to a
constant flow, the quantile residuals and the additive residuals are both
``y − ȳ``, so the statistics of both are those of `series_dependence` of
``y − ȳ``.
"""

import logging
import math
from collections.abc import Callable

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.dependence import (
    AtParameters,
    Refitted,
    SeriesDependence,
    residual_dependence,
    series_dependence,
)
from commol.observations import (
    DistributionNoise,
    LogNormalObservations,
    NoNoise,
    NormalObservations,
    PoissonObservations,
    ResidualNoise,
)
from commol.sensitivity import InputSpace, SensitivityAnalysis
from commol.resampling import BayesianBootstrap
from commol.uncertainty import ParameterSample, Predictive
from tests.support import (
    INFEASIBLE_FITS,
    a_to_b_builder,
    bounded,
    build_stochastic,
    flow_calibrator,
    flow_observations,
)

FLOWS = [9.0, 11.0, 12.0, 10.0, 8.0, 9.0, 13.0, 11.0, 10.0, 9.0, 10.0, 12.0]
"""Flows of one step at positions 1 to 12, of mean 10.333…."""


def _points(label: str, positions: list[float]) -> list[ObservedDataPoint]:
    return [ObservedDataPoint(at=at, output=label, value=0.0) for at in positions]


def _values(values: list[float], observations: list[ObservedDataPoint], **settings):
    arguments = {
        "max_lag": 2,
        "replicates": 19,
        "confidence_level": 0.9,
        "seed": 3,
    } | settings
    return series_dependence(values, observations, **arguments)


def test_series_that_have_no_statistics_say_why() -> None:
    observations = [
        *_points("twice", [1, 2, 2, 3]),
        *_points("off", [0, 2, 3, 4.5]),
        *_points("short", [1, 2, 3]),
        *_points("constant", [1, 2, 3, 4, 5]),
        *_points("gap", [1, 2, 5, 6]),
    ]
    values = (
        [1.0, 2.0, 3.0, 4.0] * 2 + [1.0, 2.0, 0.5] + [2.0] * 5 + [1.0, 3.0, 2.0, 5.0]
    )

    result = _values(values, observations)

    twice, off, short, constant, gap = result.series
    assert twice.reason is not None
    assert "two observations at 2" in twice.reason
    assert off.reason is not None
    assert "off the grid" in off.reason
    for series in (twice, off):
        assert series.portmanteau is None
        assert series.tau is None
        assert series.correlations == []
    assert short.reason is None
    assert short.tau is None
    assert short.tau_reason is not None
    assert "4 values" in short.tau_reason
    assert constant.tau is None
    assert constant.tau_reason is not None
    assert "constant" in constant.tau_reason
    # No replicate of a constant series has a τ: it has no band.
    assert constant.tau_band is None
    # A lag with no pair has no correlation.
    assert gap.pairs == [2, 0]
    assert gap.correlations[0] is not None
    assert gap.correlations[1] is None
    # Every lag reads fewer than 4 pairs: no lag enters Q̃.
    assert gap.few_pairs == [(1, 2), (2, 0)]
    assert gap.degrees_of_freedom == 0
    assert gap.portmanteau is None
    assert gap.p_value is None
    assert gap.portmanteau_reason == "every lag reads fewer than 4 pairs"
    assert twice.portmanteau_reason == twice.reason
    # Only the series with statistics pair up.
    assert [(c.first, c.second) for c in result.correlations] == [
        ("short", "constant"),
        ("short", "gap"),
        ("constant", "gap"),
    ]
    assert result.correlations[0].correlation is None


def test_any_values_of_one_set() -> None:
    # Centered: −2.5, −1.5, 0.5, −0.5, 1.5, 2.5.
    observations = _points("x", [0, 1, 2, 3, 4, 5])
    values = [1.0, 2.0, 4.0, 3.0, 5.0, 6.0]

    result = _values(values, observations)

    (x,) = result.series
    assert x.values == 6
    assert x.max_lag == 2
    variance = 17.5 / 6
    assert x.correlations == pytest.approx([5.75 / 5 / variance, -1 / 4 / variance])
    assert x.pairs == [5, 4]
    assert x.few_pairs == []
    assert x.portmanteau_reason is None
    q = 5.75**2 / 29.3125 + 1 / 4.25
    assert x.portmanteau == pytest.approx(q)
    assert x.degrees_of_freedom == 2
    assert x.p_value == pytest.approx(math.exp(-q / 2))
    assert x.tau == pytest.approx(1 + 2 * 5.75 / 5 / variance)
    assert x.cut_lag == 1
    assert x.tau_band is not None
    assert x.tau_band[0] <= x.tau_band[1]
    assert result.draws == 1
    assert result.settings.seed == 3
    assert result.settings.replicates == 19
    assert result.settings.reference is None
    # The same seed gives the same band; the band reads the seed.
    assert _values(values, observations) == result
    other = _values(values, observations, seed=4).series[0].tau_band
    assert other != x.tau_band


def test_values_are_one_set() -> None:
    observations = _points("x", [1, 2, 3, 4])
    with pytest.raises(TypeError, match="residual_dependence"):
        series_dependence(
            [[1.0, 2.0]] * 4,  # type: ignore[list-item]
            observations,
            max_lag=1,
            replicates=3,
            confidence_level=0.9,
            seed=1,
        )
    with pytest.raises(ValueError, match="residual_dependence"):
        _values([1.0, 2.0, 3.0], observations)
    with pytest.raises(ValueError, match="finite"):
        _values([1.0, 2.0, math.nan, 4.0], observations)


def test_one_largest_lag_serves_series_of_different_lengths() -> None:
    long = _points("long", list(range(35)))
    short = _points("short", list(range(6)))
    values = [math.cos(t * t) for t in range(35 + 6)]

    both = _values(values, long + short, max_lag=10)

    of_long, of_short = both.series
    assert of_long.max_lag == of_short.max_lag == both.settings.max_lag == 10
    # The long series reads all of its lags.
    assert of_long.pairs == list(range(34, 24, -1))
    assert of_long.few_pairs == []
    assert of_long.degrees_of_freedom == 10
    # The short series reads lags 1 and 2; its last lags leave Q̃.
    assert of_short.pairs == [5, 4, 3, 2, 1, 0, 0, 0, 0, 0]
    assert of_short.few_pairs == list(enumerate([3, 2, 1, 0, 0, 0, 0, 0], start=3))
    assert of_short.degrees_of_freedom == 2
    # Each one has the statistics of the lags it reads.
    for series, alone in (
        (of_long, _values(values[:35], long, max_lag=10).series[0]),
        (of_short, _values(values[35:], short, max_lag=2).series[0]),
    ):
        assert series.correlations[: alone.max_lag] == alone.correlations
        assert series.portmanteau == alone.portmanteau
        assert series.p_value == alone.p_value
    # τ, its band and the series together do not read max_lag.
    other = _values(values, long + short, max_lag=2)
    assert other.problem == both.problem
    for series, same in zip(both.series, other.series, strict=True):
        assert (series.tau, series.tau_band) == (same.tau, same.tau_band)
        assert series.cut_lag == same.cut_lag


def _normal(**models):
    return flow_calibrator(
        FLOWS,
        windowed=True,
        observation_models=models or {"B": NormalObservations(sd=1.0)},
    )


def _residuals(
    predictive: Predictive,
    observations: list[ObservedDataPoint],
    noise,
    reference=None,
    randomizations: int = 1,
    max_lag: int = 3,
) -> SeriesDependence:
    return residual_dependence(
        predictive,
        observations,
        noise,
        max_lag=max_lag,
        randomizations=randomizations,
        reference=reference or AtParameters(replicates=9),
        confidence_level=0.9,
    )


def _at_fit(calibrator) -> Predictive:
    return Predictive(calibrator, calibrator.point_sample(calibrator.run()), [1.0])


def _of_values(**settings) -> SeriesDependence:
    return _values(FLOWS, _points("x", list(range(12))), **settings)


def _of_residuals(**settings) -> SeriesDependence:
    calibrator = flow_calibrator(FLOWS, windowed=True)
    observations = calibrator.problem.observed_data
    return _residuals(_at_fit(calibrator), observations, NoNoise(), **settings)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("auto", TypeError),
        ("7", TypeError),
        (7.0, TypeError),
        (7.5, TypeError),
        (None, TypeError),
        (0, ValueError),
    ],
)
@pytest.mark.parametrize(
    ("analysis", "count"),
    [(_of_values, "replicates"), (_of_residuals, "randomizations")],
)
def test_the_largest_lag_is_a_whole_number(
    analysis: Callable[..., SeriesDependence],
    count: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="max_lag") as of_lag:
        analysis(max_lag=value)

    # It is read as the other count of the function: the same error, which
    # names the argument in its message or in its note.
    with pytest.raises(error, match=count) as of_count:
        analysis(**{count: value})
    assert str(of_lag.value).replace("max_lag", count) == str(of_count.value)


class TestResiduals:
    def test_each_noise_reads_its_residuals(self) -> None:
        calibrator = _normal()
        predictive = _at_fit(calibrator)
        observations = calibrator.problem.observed_data
        mean = sum(FLOWS) / len(FLOWS)
        expected = _values([y - mean for y in FLOWS], observations, max_lag=3)

        for noise in (DistributionNoise(), NoNoise()):
            result = _residuals(predictive, observations, noise)
            (series,) = result.series
            (reference,) = expected.series
            assert series.correlations == pytest.approx(
                reference.correlations, abs=1e-6
            )
            assert series.portmanteau == pytest.approx(reference.portmanteau, abs=1e-6)
            assert series.tau == pytest.approx(reference.tau, abs=1e-6)
            assert series.p_value is not None
            assert series.tau_band is not None
            assert result.settings.noise == noise

        with pytest.raises(TypeError, match="ResidualNoise"):
            _residuals(
                predictive,
                observations,
                ResidualNoise(
                    fit=calibrator.run(), block_length=1, residual_scale="additive"
                ),
            )
        with pytest.raises(ValueError, match="randomizations"):
            _residuals(predictive, observations, NoNoise(), randomizations=2)

    def test_counts_average_their_randomizations(self) -> None:
        calibrator = _normal(B=PoissonObservations())
        predictive = _at_fit(calibrator)
        observations = calibrator.problem.observed_data

        one = _residuals(predictive, observations, DistributionNoise())
        many = _residuals(
            predictive, observations, DistributionNoise(), randomizations=8
        )

        assert one.series[0].correlations != many.series[0].correlations
        assert many.series[0].p_value is not None
        assert many.settings.randomizations == 8

    def test_the_seed_of_the_source_repeats_them(self) -> None:
        def at_seed(seed: int) -> SeriesDependence:
            calibrator = flow_calibrator(
                FLOWS,
                windowed=True,
                observation_models={"B": PoissonObservations()},
                seed=seed,
            )
            return _residuals(
                _at_fit(calibrator),
                calibrator.problem.observed_data,
                DistributionNoise(),
                Refitted(replicates=5),
                randomizations=2,
            )

        first = at_seed(1)
        assert at_seed(1) == first
        assert at_seed(2) != first
        assert first.settings.seed == 1
        assert first.failed == []

    def test_the_randomizations_are_common_to_every_draw(self) -> None:
        calibrator = _normal(B=PoissonObservations())
        point = calibrator.point_sample(calibrator.run())
        twice = ParameterSample(
            names=point.names,
            bounds=point.bounds,
            values=point.values * 2,
            log_weights=[math.log(0.3), math.log(0.7)],
            method=point.method,
        )
        observations = calibrator.problem.observed_data

        one, two = (
            _residuals(
                Predictive(calibrator, sample, [1.0]),
                observations,
                DistributionNoise(),
                randomizations=3,
            ).series[0]
            for sample in (point, twice)
        )

        # Two equal draws with the same places have the statistics of one.
        assert two.correlations == pytest.approx(one.correlations)
        assert two.portmanteau == pytest.approx(one.portmanteau)
        assert two.tau == pytest.approx(one.tau)
        assert two.draw_p_values is not None

    def test_a_stochastic_engine_reads_the_summary_of_its_runs(self) -> None:
        def at(summary: str) -> SeriesDependence:
            problem = CalibrationProblem(
                observed_data=flow_observations(FLOWS, windowed=True),
                parameters=[bounded("c", 0.0, 50.0, initial_guess=10.0)],
                loss_function="sse",
                optimization_config=NelderMeadConfig(max_iterations=1),
                replicates=5,
                replicate_summary=summary,  # ty: ignore[invalid-argument-type]
                seed=3,
            )
            builder = a_to_b_builder(
                parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
            )
            simulation = Simulation(build_stochastic(builder, "StochasticSteps"))
            calibrator = Calibrator(simulation, problem)
            return _residuals(_at_fit(calibrator), problem.observed_data, NoNoise())

        mean, median = at("mean"), at("median")
        assert mean.series[0].tau is not None
        assert median.series[0].tau is not None
        assert mean.series[0].correlations != median.series[0].correlations

    def test_many_draws_give_statistics_and_no_test(self) -> None:
        calibrator = flow_calibrator(FLOWS, windowed=True)
        sample = calibrator.uncertainty(
            BayesianBootstrap(replicates=4, block_length=1)
        ).sample
        observations = calibrator.problem.observed_data

        result = _residuals(
            Predictive(calibrator, sample, [1.0]), observations, NoNoise()
        )

        (series,) = result.series
        assert result.draws == 4
        assert series.tau is not None
        assert series.p_value is None
        assert series.tau_band is None
        assert series.draw_p_values is not None
        assert 0 <= series.draw_p_values.share_below <= 1
        assert result.problem.p_value is None
        assert result.problem.draw_p_values is not None

    def test_draws_with_no_lag_of_enough_pairs_have_no_p_values(self) -> None:
        calibrator = flow_calibrator(FLOWS[:4], windowed=True)
        sample = calibrator.uncertainty(
            BayesianBootstrap(replicates=4, block_length=1)
        ).sample

        result = _residuals(
            Predictive(calibrator, sample, [1.0]),
            calibrator.problem.observed_data,
            NoNoise(),
        )

        (series,) = result.series
        assert series.few_pairs == [(1, 3), (2, 2), (3, 1)]
        assert series.portmanteau is None
        assert series.draw_p_values is None
        assert series.portmanteau_reason == "every lag reads fewer than 4 pairs"


class TestReferences:
    def test_refits_need_a_point_and_the_problems_data(self) -> None:
        calibrator = flow_calibrator(FLOWS, windowed=True)
        sample = calibrator.uncertainty(
            BayesianBootstrap(replicates=3, block_length=1)
        ).sample
        observations = calibrator.problem.observed_data
        refitted = Refitted(replicates=3)

        with pytest.raises(ValueError, match="point_sample"):
            _residuals(
                Predictive(calibrator, sample, [1.0]), observations, NoNoise(), refitted
            )
        held_out = [
            ObservedDataPoint(at=t, output="B", value=10.0, window=1) for t in (1, 2)
        ]
        with pytest.raises(ValueError, match="held-out"):
            _residuals(_at_fit(calibrator), held_out, NoNoise(), refitted)

        result = _residuals(_at_fit(calibrator), observations, NoNoise(), refitted)
        assert result.series[0].p_value is not None
        assert result.problem.p_value is not None
        assert result.settings.reference == refitted

    def test_a_refit_that_fails_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The fit plus the residual of the last value with its sign flipped
        # is below 0, which a log-normal model refuses: half the refits fail.
        calibrator = flow_calibrator(
            [3.0] * 11 + [9.0],
            windowed=True,
            observation_models={"B": LogNormalObservations(sigma=1.0, center="median")},
        )

        with caplog.at_level(logging.WARNING, logger="commol.dependence"):
            result = _residuals(
                _at_fit(calibrator),
                calibrator.problem.observed_data,
                NoNoise(),
                Refitted(replicates=8),
            )

        assert result.failed
        logged = [
            r.getMessage() for r in caplog.records if r.name == "commol.dependence"
        ]
        assert logged == result.failed

    def test_every_refit_that_fails_is_an_error(self) -> None:
        # Each refit flips the sign of a high value or two below 0.
        calibrator = flow_calibrator(
            [1.0, 9.0] * 6,
            windowed=True,
            observation_models={"B": LogNormalObservations(sigma=1.0, center="median")},
        )
        with pytest.raises(RuntimeError, match="Every replicate failed"):
            _residuals(
                _at_fit(calibrator),
                calibrator.problem.observed_data,
                NoNoise(),
                Refitted(replicates=3),
            )

    @pytest.mark.parametrize("kind", INFEASIBLE_FITS)
    def test_refits_refuse_a_fit_that_is_not_feasible(self, kind: str) -> None:
        build, message = INFEASIBLE_FITS[kind]
        calibrator = build()

        with pytest.raises(ValueError, match=message):
            _residuals(
                _at_fit(calibrator),
                calibrator.problem.observed_data,
                NoNoise(),
                Refitted(replicates=5),
            )

    def test_a_sensitivity_analysis_has_no_residuals(self) -> None:
        calibrator = flow_calibrator(FLOWS, windowed=True)
        sample = calibrator.point_sample(calibrator.run())
        analysis = SensitivityAnalysis(
            calibrator.simulation,
            InputSpace.from_sample(sample, calibrator.problem.parameters),
            seed=1,
        )

        with pytest.raises(ValueError, match="series_dependence"):
            _residuals(
                Predictive(analysis, sample, [1.0]),
                calibrator.problem.observed_data,
                NoNoise(),
            )
