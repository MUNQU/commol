"""Propagation: the predictive of the inputs through the model, from independent
inputs or from a sample, and the methods that need independent inputs."""

import math

import pytest
from pydantic import BaseModel

from commol import (
    Calibrator,
    DistributionNoise,
    Gain,
    InputSpace,
    Morris,
    NoNoise,
    ParameterSample,
    Peak,
    PeakPosition,
    PolynomialChaos,
    Predictive,
    Propagation,
    ResidualNoise,
    SampleMethod,
    SensitivityAnalysis,
    Simulation,
    Sobol,
    ValueAt,
    scores,
)
from commol.context.inference.method import (
    SensitivityMethod,
)
from tests.support import abc_builder, abc_rates, bounded
from tests.support.kinds import decay, decay_model
from tests.support.models import build_stochastic


def _weighted_sample() -> ParameterSample:
    """Three draws of ``k1`` and ``k2`` of the calibrator, of unequal weights."""
    return ParameterSample(
        names=["k1", "k2"],
        bounds=[(0.0, 0.5), (0.0, 0.3)],
        values=[[0.1, 0.05], [0.2, 0.1], [0.3, 0.02]],
        log_weights=[0.0, -1.0, -2.0],
        method=SampleMethod(name="test"),
    )


def _from_sample(calibrator: Calibrator) -> SensitivityAnalysis:
    return SensitivityAnalysis(
        calibrator.simulation,
        InputSpace.from_sample(_weighted_sample(), calibrator.problem.parameters),
        seed=1,
    )


class TestPropagation:
    def test_propagation_of_a_sample_is_its_predictive(
        self, calibrator: Calibrator
    ) -> None:
        analysis = _from_sample(calibrator)
        result = analysis.run(Propagation(quantities=[ValueAt(series="B", at=10)]))

        positions = [0.0, 10.0, 20.0]
        of_calibration = Predictive(calibrator, _weighted_sample(), positions)
        of_analysis = Predictive(analysis, result.sample, positions)
        assert result.sample == _weighted_sample()
        assert of_analysis.band(0.9) == of_calibration.band(0.9)
        assert result.values == [[run["B"][1]] for run in of_calibration.runs()]
        assert result.weights == pytest.approx(_weighted_sample().weights)

    def test_residual_noise_needs_a_calibration(self, calibrator: Calibrator) -> None:
        predictive = Predictive(_from_sample(calibrator), _weighted_sample(), [10.0])
        observed = calibrator.problem.observed_data
        noise = ResidualNoise(
            fit=calibrator.run(), block_length=1, residual_scale="additive"
        )

        draws = predictive.at_observations(observed, NoNoise())
        assert len(draws.values) == 3
        with pytest.raises(ValueError, match=r"ResidualNoise needs a calibration"):
            predictive.at_observations(observed, noise)
        with pytest.raises(ValueError, match=r"DistributionNoise needs a calibration"):
            predictive.at_observations(observed, DistributionNoise())
        with pytest.raises(ValueError, match=r"log score needs the observation models"):
            scores.log_score(predictive, observed, DistributionNoise())

    def test_independent_inputs_give_a_latin_hypercube(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=3)
        positions = [float(t) for t in range(51)]
        result = analysis.run(
            Propagation(draws=8, quantities=[Peak(series="B", positions=positions)])
        )

        assert result.sample.method.name == "propagation"
        assert result.weights == pytest.approx([1 / 8] * 8)
        for column, (low, high) in enumerate([(0.1, 0.5), (0.0, 1.0)]):
            strata = sorted(
                int((draw[column] - low) / (high - low) * 8)
                for draw in result.sample.values
            )
            assert strata == list(range(8))
        runs = Predictive(analysis, result.sample, positions).runs()
        assert result.values == [[max(run["B"])] for run in runs]
        (summary,) = result.summary(0.8)
        assert summary.label == "peak of B"
        assert summary.lower <= summary.median <= summary.upper
        assert summary.mean == pytest.approx(
            sum(
                row[0] * w for row, w in zip(result.values, result.weights, strict=True)
            )
        )

    def test_a_gain_and_a_peak_position_read_the_run_of_each_draw(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=3)
        positions = [float(t) for t in range(31)]
        result = analysis.run(
            Propagation(
                draws=4,
                quantities=[
                    Gain(outputs=["B", "C"], start=5, end=20),
                    PeakPosition(series="B", positions=positions),
                ],
            )
        )

        runs = Predictive(analysis, result.sample, positions).runs()
        for (gain, peak_position), run in zip(result.values, runs, strict=True):
            # B + C gains what A loses.
            assert gain == pytest.approx(run["A"][5] - run["A"][20])
            assert peak_position == positions[run["B"].index(max(run["B"]))]

    def test_a_gain_reads_its_start_exactly(self) -> None:
        # x = 10 exp(-k t) changes by several units of the last place between
        # 0.1 and 0.4 - (0.4 - 0.1), the start read from the window.
        analysis = SensitivityAnalysis(
            Simulation(decay_model()),
            InputSpace.uniform_on_bounds([bounded("k", 40, 60)]),
            seed=3,
        )
        result = analysis.run(
            Propagation(
                draws=4,
                quantities=[
                    ValueAt(series="x", at=0.1),
                    ValueAt(series="x", at=0.4),
                    Gain(outputs="x", start=0.1, end=0.4),
                ],
            )
        )

        for (at_start, at_end, gain), (k,) in zip(
            result.values, result.sample.values, strict=True
        ):
            assert at_start == decay(10, k, 0.1)
            assert gain == at_end - at_start

    def test_quantity_outside_the_model_is_rejected(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=3)
        with pytest.raises(ValueError, match=r"'Z'"):
            analysis.run(Propagation(draws=2, quantities=[ValueAt(series="Z", at=1)]))
        with pytest.raises(ValueError, match=r"1\.5"):
            analysis.run(Propagation(draws=2, quantities=[ValueAt(series="B", at=1.5)]))

    def test_the_draws_follow_the_inputs(
        self, simulation: Simulation, calibrator: Calibrator
    ) -> None:
        independent = SensitivityAnalysis(simulation, abc_rates(), seed=3)
        with pytest.raises(ValueError, match=r"needs draws"):
            independent.run(Propagation())
        with pytest.raises(ValueError, match=r"leave draws unset"):
            _from_sample(calibrator).run(Propagation(draws=4))
        only_sample = independent.run(Propagation(draws=3))
        assert only_sample.labels == []
        assert only_sample.values == []

    def test_a_stochastic_propagation_keeps_every_replicate(self) -> None:
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        analysis = SensitivityAnalysis(simulation, abc_rates(), replicates=3, seed=2)

        result = analysis.run(
            Propagation(draws=2, quantities=[ValueAt(series="C", at=20)])
        )

        assert len(result.values) == 6
        assert result.weights == pytest.approx([1 / 6] * 6)
        assert len({row[0] for row in result.values[:3]}) > 1
        # Each replicate has a seed of its own, the one of the predictive of
        # the draws: the propagation is that predictive.
        runs = Predictive(analysis, result.sample, [20.0]).runs()
        assert result.values == [[run["C"][0]] for run in runs]

    def test_propagation_values_have_one_row_per_replicate_of_each_draw(
        self,
    ) -> None:
        draws = [[0.3, 0.1], [0.4, 0.2]]
        sample = ParameterSample(
            names=["k1", "k2"],
            bounds=[(0.1, 0.5), (0.0, 1.0)],
            values=draws,
            log_weights=[math.log(0.25), math.log(0.75)],
            method=SampleMethod(name="test"),
        )
        inputs = InputSpace.from_sample(
            sample, [bounded("k1", 0.1, 0.5), bounded("k2")]
        )
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        analysis = SensitivityAnalysis(simulation, inputs, replicates=2, seed=3)

        result = analysis.run(Propagation(quantities=[ValueAt(series="C", at=20)]))

        assert result.weights == pytest.approx([0.125, 0.125, 0.375, 0.375])
        # Replicate i of draw j is replicate i of the seed of draw j.
        expected = []
        for draw, (k1, k2) in enumerate(draws):
            model = build_stochastic(
                abc_builder(parameters={"k1": k1, "k2": k2}), "StochasticSteps"
            )
            seed = result.draw_seeds[draw]
            runs = Simulation(model).replicates(2, seed=seed).run_at([20])
            expected += [[run["C"][0]] for run in runs]
        assert result.values == expected

    def test_two_draws_of_one_value_draw_their_own_noise(self) -> None:
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        sample = ParameterSample.equally_weighted(
            ["k1", "k2"],
            [(0.1, 0.5), (0.0, 1.0)],
            [[0.3, 0.1], [0.3, 0.1]],
            SampleMethod(name="test"),
        )
        inputs = InputSpace.from_sample(
            sample, [bounded("k1", 0.1, 0.5), bounded("k2")]
        )
        analysis = SensitivityAnalysis(simulation, inputs, replicates=2, seed=2)

        result = analysis.run(Propagation(quantities=[ValueAt(series="C", at=20)]))

        assert result.values[:2] != result.values[2:]


class TestIndependentInputs:
    @pytest.mark.parametrize(
        "method",
        [
            Morris(trajectories=2, levels=4, quantities=[ValueAt(series="B", at=5)]),
            Sobol(
                draws=8,
                replicates=10,
                confidence_level=0.9,
                quantities=[ValueAt(series="B", at=5)],
            ),
            PolynomialChaos(degree=1, draws=8, quantities=[ValueAt(series="B", at=5)]),
        ],
        ids=["Morris", "Sobol", "PolynomialChaos"],
    )
    def test_methods_of_independent_inputs_refuse_a_sample(
        self, calibrator: Calibrator, method: SensitivityMethod[BaseModel]
    ) -> None:
        with pytest.raises(
            ValueError,
            match=rf"{type(method).__name__} needs independent inputs.*PRCC and "
            r"propagation",
        ):
            _from_sample(calibrator).run(method)
