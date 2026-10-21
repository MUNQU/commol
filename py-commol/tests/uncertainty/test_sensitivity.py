"""Sensitivity analysis: what varies, what each run is reduced to, the
analysis that holds them, and the methods that run on it."""

import logging
import math
from typing import assert_type

import pytest
from pydantic import BaseModel

from commol import (
    Calibrator,
    ChaosResult,
    DistributionNoise,
    Gain,
    InputSpace,
    Morris,
    MorrisResult,
    NoNoise,
    Normal,
    ParameterSample,
    Peak,
    PeakPosition,
    PolynomialChaos,
    PRCC,
    PRCCResult,
    Predictive,
    Propagation,
    PropagationResult,
    ResidualNoise,
    SampleMethod,
    SensitivityAnalysis,
    SensitivityMethod,
    Simulation,
    Sobol,
    SobolResult,
    Trajectory,
    Uniform,
    ValueAt,
    scores,
)
from commol.context.inference.global_sensitivity import Quantity, expand_quantities
from tests.support import abc_builder, bounded, build
from tests.support.models import build_stochastic


@pytest.fixture
def simulation() -> Simulation:
    """``abc_builder`` with ``k1 = 0.3`` and ``k2 = 0.1``."""
    return Simulation(build(abc_builder()))


def _rates() -> InputSpace:
    return InputSpace.uniform_on_bounds([bounded("k1", 0.1, 0.5), bounded("k2")])


class TestInputs:
    def test_unknown_input_is_rejected(self, simulation: Simulation) -> None:
        inputs = InputSpace.uniform_on_bounds([bounded("k1"), bounded("k9")])
        with pytest.raises(ValueError, match=r"'k9' not found in model parameters"):
            SensitivityAnalysis(simulation, inputs)

    def test_an_unknown_initial_condition_is_rejected(
        self, simulation: Simulation
    ) -> None:
        inputs = InputSpace.uniform_on_bounds(
            [bounded("Z", 0.0, 10.0, kind="initial_condition")]
        )
        with pytest.raises(ValueError, match=r"Z"):
            SensitivityAnalysis(simulation, inputs)

    def test_inputs_may_be_initial_conditions(self, simulation: Simulation) -> None:
        inputs = InputSpace.independent(
            [bounded("k1"), bounded("A", 900.0, 1000.0, kind="initial_condition")],
            {"k1": Uniform(), "A": Normal(mean=950.0, sd=20.0)},
            initial_condition_remainder="C",
        )
        analysis = SensitivityAnalysis(simulation, inputs, seed=1)
        result = analysis.run(
            Propagation(draws=4, quantities=[ValueAt(outputs="A", at=0)])
        )
        assert result.sample.names == ["k1", "A"]
        # The initial amount of A is the input, with the remainder in C.
        assert [row[0] for row in result.values] == pytest.approx(
            [draw[1] for draw in result.sample.values]
        )

    def test_a_factor_or_a_noise_parameter_is_not_an_input(self) -> None:
        with pytest.raises(ValueError, match=r"'f' is a factor"):
            InputSpace.uniform_on_bounds([bounded("f", 0.5, 2.0, kind="factor")])
        with pytest.raises(ValueError, match=r"'s' is a noise parameter"):
            InputSpace.uniform_on_bounds([bounded("s", 0.1, 2.0, kind="noise")])

    def test_every_input_has_one_distribution(self) -> None:
        with pytest.raises(ValueError, match=r"\['k2'\] have no distribution.*"):
            InputSpace.independent([bounded("k1"), bounded("k2")], {"k1": Uniform()})
        with pytest.raises(ValueError, match=r"distributions of \['k3'\] name no"):
            InputSpace.independent([bounded("k1")], {"k1": Uniform(), "k3": Uniform()})
        with pytest.raises(ValueError, match=r"repeat the ids \['k1'\]"):
            InputSpace.uniform_on_bounds([bounded("k1"), bounded("k1")])

    def test_distributions_or_a_sample_not_both(self) -> None:
        with pytest.raises(ValueError, match=r"distributions or a sample"):
            InputSpace(parameters=[bounded("k1")])

    def test_a_sample_has_the_inputs_of_its_parameters(self) -> None:
        sample = ParameterSample.equally_weighted(
            ["k1"], [(0.0, 1.0)], [[0.2], [0.4]], SampleMethod(name="test")
        )
        inputs = InputSpace.from_sample(sample, [bounded("k1")])
        assert inputs.sample == sample
        assert inputs.rust_distributions() is None
        with pytest.raises(ValueError, match=r"the parameters are \['k2'\]"):
            InputSpace.from_sample(sample, [bounded("k2")])
        with pytest.raises(ValueError, match=r"bounds \[\(0\.0, 2\.0\)\]"):
            InputSpace.from_sample(sample, [bounded("k1", 0.0, 2.0)])

    def test_a_uniform_needs_finite_bounds(self, simulation: Simulation) -> None:
        inputs = InputSpace.uniform_on_bounds([bounded("k1", 0.0, math.inf)])
        with pytest.raises(ValueError, match=r"Uniform\(\) of 'k1' needs finite"):
            SensitivityAnalysis(simulation, inputs)


class TestQuantities:
    def test_each_quantity_has_a_label_and_a_rust_form(self) -> None:
        labels, rust = expand_quantities(
            [
                ValueAt(outputs="B", at=10),
                Gain(outputs=["B", "C"], start=5, end=10),
                Peak(outputs="B", positions=[1, 2, 3], label="peak"),
                PeakPosition(outputs="B", positions=[1, 2]),
                Trajectory(outputs="C", positions=[1, 2]),
            ]
        )
        assert labels == [
            "B at 10",
            "B + C gain over (5, 10]",
            "peak",
            "position of the peak of B",
            "C at 1",
            "C at 2",
        ]
        assert rust[1] == ("gain", ["B", "C"], [5.0, 10.0])
        assert rust[4] == ("value_at", ["C"], [1.0])

    def test_the_labels_of_close_positions_differ(self) -> None:
        labels, _ = expand_quantities(
            [
                Trajectory(outputs="B", positions=[1000000, 1000001]),
                ValueAt(outputs="B", at=0.1234567),
                ValueAt(outputs="B", at=0.1234568),
                Gain(outputs="C", start=1e-7, end=2.5e16),
            ]
        )
        assert labels == [
            "B at 1000000",
            "B at 1000001",
            "B at 0.1234567",
            "B at 0.1234568",
            "C gain over (1e-07, 2.5e+16]",
        ]

    def test_a_quantity_reads_a_valid_window_of_positions(self) -> None:
        with pytest.raises(ValueError, match=r"ends after it starts"):
            Gain(outputs="C", start=10, end=10)
        with pytest.raises(ValueError, match=r"ascend"):
            Peak(outputs="B", positions=[2, 1])
        with pytest.raises(ValueError, match=r"must not be empty"):
            ValueAt(outputs=[], at=1)
        with pytest.raises(ValueError):
            ValueAt(outputs="B", at=-1)

    def test_quantities_have_labels_of_their_own(self) -> None:
        with pytest.raises(ValueError, match=r"\['B at 1'\]"):
            expand_quantities(
                [ValueAt(outputs="B", at=1), Trajectory(outputs="B", positions=[1])]
            )
        with pytest.raises(ValueError, match=r"at least one quantity"):
            expand_quantities([])


class TestAnalysis:
    def test_a_stochastic_engine_needs_replicates_and_a_seed(self) -> None:
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        with pytest.raises(ValueError, match=r"set replicates"):
            SensitivityAnalysis(simulation, _rates(), seed=1)
        with pytest.raises(ValueError, match=r"set seed"):
            SensitivityAnalysis(simulation, _rates(), replicates=2)
        analysis = SensitivityAnalysis(simulation, _rates(), replicates=2, seed=1)
        assert analysis.replicates == 2

    def test_a_deterministic_engine_does_not_use_replicates(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), replicates=3, seed=4)
        assert analysis.seed == 4

    def test_with_no_seed_one_is_drawn_and_logged(
        self, simulation: Simulation, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="commol.api.sensitivity"):
            analysis = SensitivityAnalysis(simulation, _rates())
        assert f"drew the seed {analysis.seed}" in caplog.text
        assert analysis.seed == analysis.seed

    def test_a_stochastic_engine_runs_at_least_one_replicate(self) -> None:
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        with pytest.raises(ValueError, match=r"replicates is at least 1, got 0"):
            SensitivityAnalysis(simulation, _rates(), replicates=0, seed=1)

    def test_a_seed_is_a_whole_number_of_64_bits(self, simulation: Simulation) -> None:
        with pytest.raises(ValueError):
            SensitivityAnalysis(simulation, _rates(), seed=-1)


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
        result = analysis.run(Propagation(quantities=[ValueAt(outputs="B", at=10)]))

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
        analysis = SensitivityAnalysis(simulation, _rates(), seed=3)
        positions = [float(t) for t in range(51)]
        result = analysis.run(
            Propagation(draws=8, quantities=[Peak(outputs="B", positions=positions)])
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
        with pytest.raises(ValueError, match=r"strictly between 0 and 1"):
            result.summary(1.0)

    def test_a_gain_and_a_peak_position_read_the_run_of_each_draw(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), seed=3)
        positions = [float(t) for t in range(31)]
        result = analysis.run(
            Propagation(
                draws=4,
                quantities=[
                    Gain(outputs=["B", "C"], start=5, end=20),
                    PeakPosition(outputs="B", positions=positions),
                ],
            )
        )

        runs = Predictive(analysis, result.sample, positions).runs()
        for (gain, peak_position), run in zip(result.values, runs, strict=True):
            # B + C gains what A loses.
            assert gain == pytest.approx(run["A"][5] - run["A"][20])
            assert peak_position == positions[run["B"].index(max(run["B"]))]

    def test_quantity_outside_the_model_is_rejected(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), seed=3)
        with pytest.raises(ValueError, match=r"'Z'"):
            analysis.run(Propagation(draws=2, quantities=[ValueAt(outputs="Z", at=1)]))
        with pytest.raises(ValueError, match=r"1\.5"):
            analysis.run(
                Propagation(draws=2, quantities=[ValueAt(outputs="B", at=1.5)])
            )

    def test_the_draws_follow_the_inputs(
        self, simulation: Simulation, calibrator: Calibrator
    ) -> None:
        independent = SensitivityAnalysis(simulation, _rates(), seed=3)
        with pytest.raises(ValueError, match=r"needs draws"):
            independent.run(Propagation())
        with pytest.raises(ValueError, match=r"leave draws unset"):
            _from_sample(calibrator).run(Propagation(draws=4))
        only_sample = independent.run(Propagation(draws=3))
        assert only_sample.labels == []
        assert only_sample.values == []

    def test_a_stochastic_propagation_keeps_every_replicate(self) -> None:
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        analysis = SensitivityAnalysis(simulation, _rates(), replicates=3, seed=2)

        result = analysis.run(
            Propagation(draws=2, quantities=[ValueAt(outputs="C", at=20)])
        )

        assert len(result.values) == 6
        assert result.weights == pytest.approx([1 / 6] * 6)
        assert len({row[0] for row in result.values[:3]}) > 1
        # Each replicate has a seed of its own, the one of the predictive of
        # the draws: the propagation is that predictive.
        runs = Predictive(analysis, result.sample, [20.0]).runs()
        assert result.values == [[run["C"][0]] for run in runs]

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

        result = analysis.run(Propagation(quantities=[ValueAt(outputs="C", at=20)]))

        assert result.values[:2] != result.values[2:]


class TestPRCC:
    def test_prcc_signs_the_inputs_of_a_model_and_ignores_a_dummy(self) -> None:
        simulation = Simulation(
            build(abc_builder(parameters={"k1": 0.3, "k2": 0.1, "unused": 1.0}))
        )
        inputs = InputSpace.uniform_on_bounds(
            [bounded("k1", 0.2, 0.6), bounded("k2", 0.05, 0.2), bounded("unused")]
        )
        analysis = SensitivityAnalysis(simulation, inputs, seed=7)
        positions = [float(t) for t in range(101)]

        result = analysis.run(
            PRCC(draws=200, quantities=[Peak(outputs="B", positions=positions)])
        )

        (k1, k2, unused) = result.coefficients[0]
        assert k1 > 0.8
        assert k2 < -0.8
        assert abs(unused) < 0.2
        assert result.p_values[0][0] < 1e-6
        assert result.inputs == ["k1", "k2", "unused"]
        assert result.series == ["peak of B"]
        assert result.positions == [None]
        assert result.effective_size == pytest.approx(200)

    def test_a_trajectory_gives_coefficients_over_the_axis(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), seed=7)

        result = analysis.run(
            PRCC(draws=30, quantities=[Trajectory(outputs="C", positions=[0, 10, 20])])
        )

        assert result.labels == ["C at 0", "C at 10", "C at 20"]
        assert result.series == ["C"] * 3
        assert result.positions == [0.0, 10.0, 20.0]
        # Every run starts from the same state, so C at 0 has no coefficient.
        assert all(math.isnan(c) for c in result.coefficients[0])
        # k2 up to 1 stops the outbreak, so more recovery leaves fewer in C.
        (k1, k2) = result.coefficients[2]
        assert k1 > 0.5
        assert k2 < -0.5

    def test_prcc_of_a_sample_reads_its_weights(self, calibrator: Calibrator) -> None:
        sample = ParameterSample(
            names=["k1", "k2"],
            bounds=[(0.0, 0.5), (0.0, 0.3)],
            values=[[0.05 * i, 0.02 * ((7 * i) % 13)] for i in range(1, 11)],
            log_weights=[-0.1 * i for i in range(10)],
            method=SampleMethod(name="test"),
        )
        analysis = SensitivityAnalysis(
            calibrator.simulation,
            InputSpace.from_sample(sample, calibrator.problem.parameters),
            seed=1,
        )

        result = analysis.run(PRCC(quantities=[ValueAt(outputs="B", at=20)]))

        weights = sample.weights
        assert result.sample == sample
        assert result.effective_size == pytest.approx(
            sum(weights) ** 2 / sum(w * w for w in weights)
        )
        with pytest.raises(ValueError, match=r"leave draws unset"):
            analysis.run(PRCC(draws=5, quantities=[ValueAt(outputs="B", at=20)]))

    def test_prcc_needs_more_draws_than_inputs(self, simulation: Simulation) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), seed=7)
        with pytest.raises(ValueError, match=r"effective size above 3"):
            analysis.run(PRCC(draws=3, quantities=[ValueAt(outputs="B", at=5)]))
        with pytest.raises(ValueError, match=r"PRCC of independent inputs needs draws"):
            analysis.run(PRCC(quantities=[ValueAt(outputs="B", at=5)]))


class TestMorris:
    def test_morris_screens_out_an_input_the_model_does_not_read(self) -> None:
        simulation = Simulation(
            build(abc_builder(parameters={"k1": 0.3, "k2": 0.1, "unused": 1.0}))
        )
        inputs = InputSpace.uniform_on_bounds(
            [bounded("k1", 0.2, 0.6), bounded("k2", 0.05, 0.2), bounded("unused")]
        )
        analysis = SensitivityAnalysis(simulation, inputs, seed=7)
        positions = [float(t) for t in range(101)]

        result = analysis.run(
            Morris(
                trajectories=10,
                levels=4,
                quantities=[Peak(outputs="B", positions=positions)],
            )
        )

        (k1, k2, unused) = result.mu_star[0]
        assert unused == 0.0
        assert k1 > 0.0 and k2 > 0.0
        assert result.mu[0][0] > 0.0 > result.mu[0][1]
        assert result.runs == 10 * 4
        assert result.inputs == ["k1", "k2", "unused"]
        assert len(result.sigma[0]) == 3

    def test_morris_takes_an_even_number_of_levels(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), seed=7)
        with pytest.raises(ValueError, match=r"even number of levels"):
            analysis.run(
                Morris(
                    trajectories=2, levels=3, quantities=[ValueAt(outputs="B", at=5)]
                )
            )


class TestSobol:
    def test_an_input_the_model_does_not_read_has_no_index(self) -> None:
        simulation = Simulation(
            build(abc_builder(parameters={"k1": 0.3, "k2": 0.1, "unused": 1.0}))
        )
        inputs = InputSpace.uniform_on_bounds(
            [bounded("k1", 0.2, 0.6), bounded("k2", 0.05, 0.2), bounded("unused")]
        )
        analysis = SensitivityAnalysis(simulation, inputs, seed=7)
        positions = [float(t) for t in range(101)]

        result = analysis.run(
            Sobol(
                base_draws=64,
                bootstrap=50,
                confidence_level=0.9,
                quantities=[Peak(outputs="B", positions=positions)],
            )
        )

        assert result.first[0][2] == 0.0
        assert result.total[0][2] == 0.0
        assert result.total[0][0] > 0.1 and result.total[0][1] > 0.1
        for i in range(3):
            assert result.first_lower[0][i] <= result.first_upper[0][i]
            assert result.total_lower[0][i] <= result.total_upper[0][i]
        assert result.runs == 64 * 5
        assert result.confidence_level == 0.9

    def test_the_level_of_the_intervals_is_strictly_between_0_and_1(self) -> None:
        with pytest.raises(ValueError):
            Sobol(
                base_draws=8,
                bootstrap=10,
                confidence_level=1.0,
                quantities=[ValueAt(outputs="B", at=5)],
            )


class TestPolynomialChaos:
    def test_the_expansion_is_a_surrogate_of_the_model(
        self, simulation: Simulation
    ) -> None:
        inputs = InputSpace.independent(
            [bounded("k1", 0.2, 0.5), bounded("k2", 0.05, 0.15)],
            {"k1": Uniform(), "k2": Normal(mean=0.1, sd=0.02)},
        )
        analysis = SensitivityAnalysis(simulation, inputs, seed=5)

        result = analysis.run(
            PolynomialChaos(
                degree=4, draws=80, quantities=[ValueAt(outputs="B", at=20)]
            )
        )

        assert result.loo_error[0] < 0.01
        assert result.runs == 80
        assert len(result.indices) == 15
        assert sum(result.first[0]) <= sum(result.total[0]) + 1e-9
        truth = Simulation(build(abc_builder(parameters={"k1": 0.35, "k2": 0.12})))
        expected = truth.run(20)["B"][20]
        (predicted,) = result.predict([[0.35, 0.12]])[0]
        assert predicted == pytest.approx(expected, rel=0.02)
        assert result.inputs == ["k1", "k2"]
        assert result.input_space == inputs
        with pytest.raises(ValueError, match=r"'k1' is 0.9, outside its bounds"):
            result.predict([[0.9, 0.12]])
        assert result.mean[0] > 0.0 and result.variance[0] > 0.0

    def test_an_expansion_needs_more_draws_than_terms(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), seed=5)
        with pytest.raises(ValueError, match=r"needs more points than terms"):
            analysis.run(
                PolynomialChaos(
                    degree=3, draws=10, quantities=[ValueAt(outputs="B", at=5)]
                )
            )


class TestIndependentInputs:
    @pytest.mark.parametrize(
        "method",
        [
            Morris(trajectories=2, levels=4, quantities=[ValueAt(outputs="B", at=5)]),
            Sobol(
                base_draws=8,
                bootstrap=10,
                confidence_level=0.9,
                quantities=[ValueAt(outputs="B", at=5)],
            ),
            PolynomialChaos(degree=1, draws=8, quantities=[ValueAt(outputs="B", at=5)]),
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


class TestEntryPoint:
    def test_each_method_returns_its_result(self, simulation: Simulation) -> None:
        analysis = SensitivityAnalysis(simulation, _rates(), seed=3)
        quantities: list[Quantity] = [ValueAt(outputs="B", at=10)]

        propagation = analysis.run(Propagation(draws=4, quantities=quantities))
        prcc = analysis.run(PRCC(draws=8, quantities=quantities))
        morris = analysis.run(Morris(trajectories=2, levels=4, quantities=quantities))
        sobol = analysis.run(
            Sobol(
                base_draws=8, bootstrap=10, confidence_level=0.9, quantities=quantities
            )
        )
        chaos = analysis.run(PolynomialChaos(degree=1, draws=8, quantities=quantities))

        assert_type(propagation, PropagationResult)
        assert_type(prcc, PRCCResult)
        assert_type(morris, MorrisResult)
        assert_type(sobol, SobolResult)
        assert_type(chaos, ChaosResult)
        for result, kind in [
            (propagation, PropagationResult),
            (prcc, PRCCResult),
            (morris, MorrisResult),
            (sobol, SobolResult),
            (chaos, ChaosResult),
        ]:
            assert type(result) is kind
