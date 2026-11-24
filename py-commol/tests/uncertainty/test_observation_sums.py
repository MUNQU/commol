"""Observation models of sums of parts: how each model reads a sum, which
models refuse one, the draws of a sum, and an exact sum that does not end.

The problems fit a constant flow ``c`` from A to B (``a_to_b_builder`` with
the rate ``c * A / A`` and 10⁶ in A), observed over windows of ``K`` steps of
``K`` parts each: every part has the mean ``c``, and the sum ``K · c``. The
sum of ``K`` negative binomials of mean ``c`` and dispersion ``φ`` is a
negative binomial of mean ``Kc`` and size ``K/φ``, so both readings of a
negative binomial sum have a closed form here.
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
    scores,
)
from commol.local import Laplace
from commol.observations import (
    BinomialObservations,
    DistributionNoise,
    LogNormalObservations,
    NegativeBinomialObservations,
    NormalObservations,
    PoissonObservations,
    StudentTObservations,
)
from commol.observations.models import _ObservationModel
from commol.resampling import ParametricBootstrap
from commol.uncertainty import ParameterSample, Predictive, SampleMethod
from tests.support import a_to_b_builder, bounded, build, flow_observations

FLOW = 10.0


def _flow_simulation() -> Simulation:
    return Simulation(
        build(
            a_to_b_builder(
                parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
            )
        )
    )


def _problem(
    observed_data: list[ObservedDataPoint],
    models: dict[str, _ObservationModel],
    seed: int = 1,
) -> CalibrationProblem:
    return CalibrationProblem(
        observed_data=observed_data,
        parameters=[bounded("c", 0.0, 50.0)],
        observation_models=models,
        optimization_config=NelderMeadConfig(max_iterations=500),
        seed=seed,
    )


def _sums(values: list[float], parts: int) -> list[ObservedDataPoint]:
    """Observations of the flow into B over consecutive windows of `parts`
    steps, each window the sum of `parts` parts of one step."""
    return [
        ObservedDataPoint(
            at=parts * (index + 1),
            output="B",
            value=value,
            window=parts,
            parts=parts,
        )
        for index, value in enumerate(values)
    ]


class TestSums:
    def test_a_negative_binomial_sum_says_how_it_is_read(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                "The NegativeBinomial model of 'B' needs sum, \"exact\" or "
                "\"moments\": observation 'B' at 3 has parts=3"
            ),
        ):
            Calibrator(
                _flow_simulation(),
                _problem(
                    _sums([30.0], 3),
                    {"B": NegativeBinomialObservations(overdispersion=0.5)},
                ),
            )
        with pytest.raises(
            ValueError,
            match="has sum, but no observation of the series has more than one part",
        ):
            Calibrator(
                _flow_simulation(),
                _problem(
                    flow_observations([10.0], windowed=True),
                    {
                        "B": NegativeBinomialObservations(
                            overdispersion=0.5, sum="exact"
                        )
                    },
                ),
            )
        with pytest.raises(ValidationError, match="'exact' or 'moments'"):
            NegativeBinomialObservations(overdispersion=0.5, sum="normal")  # type: ignore[arg-type]

        # Each reading has its density at the fit: the exact one is that of
        # a negative binomial of three times the dispersion size, as the sum
        # of three equal parts is; the moments one matches the same mean and
        # variance, which for equal parts is the same distribution.
        phi, flow, value = 0.5, 10.0, 25.0
        size = 3 / phi
        mean = 3 * flow
        exact = (
            math.lgamma(value + size)
            - math.lgamma(size)
            - math.lgamma(value + 1)
            + size * math.log(size / (size + mean))
            + value * math.log(mean / (size + mean))
        )
        for reading in ("exact", "moments"):
            calibrator = Calibrator(
                _flow_simulation(),
                _problem(
                    _sums([value], 3),
                    {
                        "B": NegativeBinomialObservations(
                            overdispersion=phi, sum=reading
                        )
                    },
                ),
            )
            (density,) = calibrator._engine_problem.pointwise_log_likelihood([flow])
            assert density == pytest.approx(exact, rel=1e-12), reading

    def test_a_held_out_observation_reads_the_sum_of_its_model(self) -> None:
        # A model with a sum reads a held-out count of one part as NB(μ, φ);
        # a model without one refuses a held-out sum of parts.
        phi, flow, value = 0.5, 10.0, 12.0
        point = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[flow]], SampleMethod(name="point")
        )
        with_sum = Calibrator(
            _flow_simulation(),
            _problem(
                _sums([30.0, 28.0], 3),
                {"B": NegativeBinomialObservations(overdispersion=phi, sum="exact")},
            ),
        )
        score = scores.log_score(
            Predictive(with_sum, point, [0, 7]),
            flow_observations([value], windowed=True, start=7),
            DistributionNoise(),
        )
        size = 1 / phi
        assert score.per_observation == [
            pytest.approx(
                math.lgamma(value + size)
                - math.lgamma(size)
                - math.lgamma(value + 1)
                + size * math.log(size / (size + flow))
                + value * math.log(flow / (size + flow)),
                rel=1e-12,
            )
        ]

        without_sum = Calibrator(
            _flow_simulation(),
            _problem(
                flow_observations([10.0], windowed=True),
                {"B": NegativeBinomialObservations(overdispersion=phi)},
            ),
        )
        with pytest.raises(
            ValueError,
            match=(
                "Observation 'B' at 9 has parts=3, and the NegativeBinomial model "
                "of its series has no sum"
            ),
        ):
            scores.log_score(
                Predictive(without_sum, point, [0, 9]),
                _sums([0.0, 0.0, 30.0], 3)[2:],
                DistributionNoise(),
            )

    @pytest.mark.parametrize(
        "model",
        [
            LogNormalObservations(sigma=0.5, center="median"),
            StudentTObservations(df=4.0, scale=1.0),
            BinomialObservations(trials=100),
        ],
        ids=["log_normal", "student_t", "binomial"],
    )
    def test_a_model_without_a_sum_refuses_parts(
        self, model: _ObservationModel
    ) -> None:
        with pytest.raises(
            ValueError,
            match=(
                "Observation 'B' at 3 has parts=3, but its series 'B' has a "
                ".* model, which reads no sum of parts; only Poisson, Normal and "
                "NegativeBinomial models read one"
            ),
        ):
            Calibrator(_flow_simulation(), _problem(_sums([30.0], 3), {"B": model}))

    def test_an_exact_sum_that_does_not_end_says_why(self) -> None:
        # Weekly totals in 7 parts at φ = 10⁶: the mass of the sum spreads
        # far above its mean, so its information does not reach a negligible
        # term within the counts of the exact reading.
        exact = NegativeBinomialObservations(overdispersion=1e6, sum="exact")
        why = (
            r"Observation 'B' at 7 has the total 60 in 7 parts; its "
            r"NegativeBinomial model with sum=\"exact\", the mean .* of the "
            r"total and the overdispersion 1000000(\.0)?, does not end within "
            r"1000000 counts\. Read the sum with sum=\"moments\""
        )
        weekly = Calibrator(
            _flow_simulation(), _problem(_sums([60.0, 80.0], 7), {"B": exact})
        )
        with pytest.raises(RuntimeError, match=why):
            weekly.uncertainty(Laplace(draws=10))
        # The scores read the same sum: its upper tail, and the density of a
        # total at the counts.
        point = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[FLOW]], SampleMethod(name="point")
        )
        predictive = Predictive(weekly, point, [0, 14])
        with pytest.raises(RuntimeError, match="'B' at 7 has the total 0 in 7 parts"):
            scores.quantile_residuals(predictive, _sums([0.0], 7), DistributionNoise())
        with pytest.raises(RuntimeError, match="'B' at 7 has the total 1000000 in 7"):
            scores.log_score(predictive, _sums([1e6], 7), DistributionNoise())

        # A total at the counts stops the fit, with the same reason.
        far = Calibrator(
            _flow_simulation(), _problem(_sums([60.0, 1e6], 7), {"B": exact})
        )
        with pytest.raises(RuntimeError, match="'B' at 14 has the total 1000000"):
            far.run()

        # The moments read the same totals.
        moments = NegativeBinomialObservations(overdispersion=1e6, sum="moments")
        fit = Calibrator(
            _flow_simulation(), _problem(_sums([60.0, 1e6], 7), {"B": moments})
        ).run()
        assert math.isfinite(fit.final_loss)

    def test_an_outlier_of_an_exact_sum_has_a_pit_near_one(self) -> None:
        # Weekly totals of mean 70 in 7 parts: one at the mean, and others up
        # to 1400 times it. At φ = 0 the exact sum is Poisson(70), and its
        # PIT and its quantile residuals are those of that model, at the same
        # place within the mass of each total.
        totals = [70.0, 200.0, 1500.0, 99_999.0]
        point = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[FLOW]], SampleMethod(name="point")
        )
        observations = _sums(totals, 7)
        pits, residuals = [], []
        for model in (
            NegativeBinomialObservations(overdispersion=0.0, sum="exact"),
            PoissonObservations(),
        ):
            calibrator = Calibrator(
                _flow_simulation(), _problem(observations, {"B": model})
            )
            predictive = Predictive(calibrator, point, [0, 28])
            pits.append(
                scores.pit(
                    predictive, observations, DistributionNoise()
                ).per_observation
            )
            residuals.append(
                scores.quantile_residuals(
                    predictive, observations, DistributionNoise()
                ).per_observation
            )

        exact, poisson = pits
        assert exact == pytest.approx(poisson, rel=1e-12)
        assert 0.3 < exact[0] < 0.7
        assert exact[1:] == [1.0, 1.0, 1.0]
        exact, poisson = residuals
        assert exact == pytest.approx(poisson, rel=1e-9)
        assert exact[1] > 10.0
        assert exact[2:] == [math.inf, math.inf]

    def test_poisson_and_normal_sums_read_the_sum_of_their_means(self) -> None:
        flow, value = 10.0, 27.0
        poisson = Calibrator(
            _flow_simulation(),
            _problem(_sums([value], 3), {"B": PoissonObservations()}),
        )
        normal = Calibrator(
            _flow_simulation(),
            _problem(_sums([value], 3), {"B": NormalObservations(sd=2.0)}),
        )

        (density,) = poisson._engine_problem.pointwise_log_likelihood([flow])
        assert density == pytest.approx(
            value * math.log(3 * flow) - 3 * flow - math.lgamma(value + 1), rel=1e-12
        )
        (density,) = normal._engine_problem.pointwise_log_likelihood([flow])
        variance = 3 * 2.0**2
        assert density == pytest.approx(
            -0.5 * math.log(2 * math.pi * variance)
            - (value - 3 * flow) ** 2 / (2 * variance),
            rel=1e-12,
        )

    def test_a_sum_is_drawn_as_its_model_reads_it(self) -> None:
        # Windows of five steps of a constant flow c, each the sum of five
        # parts of a negative binomial of dispersion φ: the sum of equal
        # parts is a negative binomial of mean 5c and size 5/φ, so the fit
        # is the mean of the sums over five, of variance c(1 + φc)/(5n).
        phi, parts, count, replicates = 0.2, 5, 40, 300
        generator = np.random.default_rng(12)
        size = 1 / phi
        values = (
            generator.negative_binomial(size, size / (size + FLOW), (count, parts))
            .sum(axis=1)
            .astype(float)
            .tolist()
        )
        variance = FLOW * (1 + phi * FLOW) / (parts * count)
        for reading in ("exact", "moments"):
            calibrator = Calibrator(
                _flow_simulation(),
                _problem(
                    _sums(values, parts),
                    {
                        "B": NegativeBinomialObservations(
                            overdispersion=phi, sum=reading
                        )
                    },
                    seed=4,
                ),
            )

            result = calibrator.uncertainty(ParametricBootstrap(replicates=replicates))

            fit = result.fit.best_parameters["c"]
            expected = fit * (1 + phi * fit) / (parts * count)
            sampled = result.sample.covariance()[0][0]
            error = expected * math.sqrt(2 / (replicates - 1))
            assert abs(sampled - expected) < 3 * error, (reading, sampled, expected)
            assert expected == pytest.approx(variance, rel=0.2)

            # The draws of the model of an observation: the variance of a
            # sum of five parts, 5c(1 + φc).
            sample = ParameterSample.equally_weighted(
                ["c"], [(0.0, 50.0)], [[FLOW]] * 4000, SampleMethod(name="point")
            )
            draws = Predictive(calibrator, sample, [0, parts]).at_observations(
                _sums([0.0], parts), DistributionNoise()
            )
            shown = np.array([row[0] for row in draws.values])
            assert shown.mean() == pytest.approx(parts * FLOW, rel=0.03)
            assert shown.var() == pytest.approx(
                parts * FLOW * (1 + phi * FLOW), rel=0.1
            )

            # A sum of mean 0 is 0, also when read by its moments.
            empty = ParameterSample.equally_weighted(
                ["c"], [(0.0, 50.0)], [[0.0]] * 10, SampleMethod(name="point")
            )
            zeros = Predictive(calibrator, empty, [0, parts]).at_observations(
                _sums([0.0], parts), DistributionNoise()
            )
            assert zeros.values == [[0.0]] * 10
