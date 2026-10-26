"""Sensitivity analysis: what varies, what each run is reduced to, the
analysis that holds them, and the methods that run on it."""

import logging
import math
from typing import assert_type

import pytest

from commol import (
    Calibrator,
    Gain,
    InputSpace,
    Morris,
    Normal,
    ParameterSample,
    Peak,
    PolynomialChaos,
    PRCC,
    Propagation,
    SampleMethod,
    SensitivityAnalysis,
    Simulation,
    Sobol,
    Trajectory,
    Uniform,
    ValueAt,
)
from commol.context.inference.global_sensitivity import (
    ChaosResult,
    MorrisResult,
    PRCCResult,
    PropagationResult,
    SobolResult,
)
from commol.context.inference.quantities import _Quantity
from tests.support import abc_builder, abc_rates, bounded, build
from tests.support.models import a_to_b_builder, build_stochastic


class TestInputs:
    def test_unknown_input_is_rejected(self, simulation: Simulation) -> None:
        inputs = InputSpace.uniform_on_bounds([bounded("k1"), bounded("k9")])
        with pytest.raises(ValueError, match=r"Parameter 'k9' is not in the model"):
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
            Propagation(draws=4, quantities=[ValueAt(series="A", at=0)])
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
        assert inputs.distributions is None
        with pytest.raises(ValueError, match=r"the parameters are \['k2'\]"):
            InputSpace.from_sample(sample, [bounded("k2")])
        with pytest.raises(ValueError, match=r"bounds \[\(0\.0, 2\.0\)\]"):
            InputSpace.from_sample(sample, [bounded("k1", 0.0, 2.0)])

    def test_a_uniform_needs_finite_bounds(self, simulation: Simulation) -> None:
        inputs = InputSpace.uniform_on_bounds([bounded("k1", 0.0, math.inf)])
        with pytest.raises(ValueError, match=r"Uniform\(\) of 'k1' needs finite"):
            SensitivityAnalysis(simulation, inputs)


class TestAnalysis:
    def test_a_stochastic_engine_needs_replicates_and_a_seed(self) -> None:
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        with pytest.raises(ValueError, match=r"set replicates"):
            SensitivityAnalysis(simulation, abc_rates(), seed=1)
        with pytest.raises(ValueError, match=r"set seed"):
            SensitivityAnalysis(simulation, abc_rates(), replicates=2)
        analysis = SensitivityAnalysis(simulation, abc_rates(), replicates=2, seed=1)
        assert analysis.replicates == 2

    def test_a_deterministic_engine_does_not_use_replicates(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, abc_rates(), replicates=3, seed=4)
        assert analysis.seed == 4

    def test_with_no_seed_one_is_drawn_and_logged(
        self, simulation: Simulation, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="commol.api.sensitivity"):
            analysis = SensitivityAnalysis(simulation, abc_rates())
        method = PRCC(draws=10, quantities=[ValueAt(series="B", at=5)])

        again = SensitivityAnalysis(simulation, abc_rates(), seed=analysis.seed)

        assert f"drew the seed {analysis.seed}" in caplog.text
        assert again.run(method) == analysis.run(method)

    def test_a_stochastic_engine_runs_at_least_one_replicate(self) -> None:
        simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
        with pytest.raises(
            ValueError, match=r"replicates is a whole number of at least 1, got 0"
        ):
            SensitivityAnalysis(simulation, abc_rates(), replicates=0, seed=1)

    def test_a_negative_count_names_the_setting(self, simulation: Simulation) -> None:
        with pytest.raises(ValueError) as error:
            SensitivityAnalysis(simulation, abc_rates(), replicates=-1, seed=1)
        with pytest.raises(TypeError):
            SensitivityAnalysis(simulation, abc_rates(), replicates=1.5, seed=1)  # ty: ignore[invalid-argument-type]

        assert str(error.value) == "replicates is a whole number of at least 1, got -1"

    def test_a_seed_is_a_whole_number_of_64_bits(self, simulation: Simulation) -> None:
        with pytest.raises(ValueError):
            SensitivityAnalysis(simulation, abc_rates(), seed=-1)


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
            PRCC(draws=200, quantities=[Peak(series="B", positions=positions)])
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
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=7)

        result = analysis.run(
            PRCC(draws=30, quantities=[Trajectory(series="C", positions=[0, 10, 20])])
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

        result = analysis.run(PRCC(quantities=[ValueAt(series="B", at=20)]))

        weights = sample.weights
        assert result.sample == sample
        assert result.effective_size == pytest.approx(
            sum(weights) ** 2 / sum(w * w for w in weights)
        )
        with pytest.raises(ValueError, match=r"leave draws unset"):
            analysis.run(PRCC(draws=5, quantities=[ValueAt(series="B", at=20)]))

    def test_prcc_needs_more_draws_than_inputs(self, simulation: Simulation) -> None:
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=7)
        with pytest.raises(ValueError, match=r"effective size above 3"):
            analysis.run(PRCC(draws=3, quantities=[ValueAt(series="B", at=5)]))
        with pytest.raises(ValueError, match=r"PRCC of independent inputs needs draws"):
            analysis.run(PRCC(quantities=[ValueAt(series="B", at=5)]))

    def test_a_quantity_that_is_nan_is_an_error(self) -> None:
        # The sum of A, B and acc is above the largest float from position 1,
        # so the gain is inf − inf, NaN, at every draw.
        simulation = Simulation(
            build(
                a_to_b_builder(amounts={"A": 0.9e308, "B": 0.85e308}, accumulator="acc")
            )
        )
        inputs = InputSpace.uniform_on_bounds([bounded("k", 0.1, 0.5)])
        analysis = SensitivityAnalysis(simulation, inputs, seed=1)
        gain = Gain(outputs=["A", "B", "acc"], start=1, end=5)

        values = analysis.run(Propagation(draws=3, quantities=[gain])).values
        assert all(math.isnan(row[0]) for row in values)
        with pytest.raises(
            ValueError, match=r"quantity 1 .* NaN at the points \[0, 1, 2, 3, 4\]"
        ):
            analysis.run(PRCC(draws=5, quantities=[ValueAt(series="B", at=5), gain]))


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
                quantities=[Peak(series="B", positions=positions)],
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
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=7)
        with pytest.raises(ValueError, match=r"even number of levels"):
            analysis.run(
                Morris(trajectories=2, levels=3, quantities=[ValueAt(series="B", at=5)])
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
                draws=64,
                replicates=50,
                confidence_level=0.9,
                quantities=[Peak(series="B", positions=positions)],
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
            PolynomialChaos(degree=4, draws=80, quantities=[ValueAt(series="B", at=20)])
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
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=5)
        with pytest.raises(ValueError, match=r"needs more points than terms"):
            analysis.run(
                PolynomialChaos(
                    degree=3, draws=10, quantities=[ValueAt(series="B", at=5)]
                )
            )

    def test_a_truncation_keeps_the_terms_of_few_inputs(
        self, simulation: Simulation
    ) -> None:
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=5)
        quantities: list[_Quantity] = [ValueAt(series="B", at=20)]

        result = analysis.run(
            PolynomialChaos(
                degree=4,
                q_norm=0.5,
                max_interaction=1,
                draws=30,
                quantities=quantities,
            )
        )

        # The terms of one input only, to degree 4: 1 + 2 · 4.
        assert result.indices == [
            [0, 0],
            [1, 0],
            [0, 1],
            [2, 0],
            [0, 2],
            [3, 0],
            [0, 3],
            [4, 0],
            [0, 4],
        ]
        assert (result.q_norm, result.max_interaction) == (0.5, 1)
        # No term holds both inputs, so each total index is a first-order one.
        assert result.total[0] == pytest.approx(result.first[0])
        assert len(result.predict([[0.3, 0.5]])[0]) == 1
        with pytest.raises(ValueError, match=r"has 9 terms and needs more points"):
            analysis.run(
                PolynomialChaos(
                    degree=4, max_interaction=1, draws=9, quantities=quantities
                )
            )

    @pytest.mark.parametrize(
        "truncation",
        [{"q_norm": 0.0}, {"q_norm": 1.5}, {"max_interaction": 0}],
        ids=["q-norm zero", "q-norm above one", "no input"],
    )
    def test_a_truncation_out_of_range_is_rejected(
        self, truncation: dict[str, float]
    ) -> None:
        with pytest.raises(ValueError, match=r"(q_norm|max_interaction) is a"):
            PolynomialChaos.model_validate(
                {
                    "degree": 2,
                    "draws": 8,
                    "quantities": [ValueAt(series="B", at=5)],
                    **truncation,
                }
            )

    def test_a_result_predicts_on_the_terms_of_its_truncation(self) -> None:
        inputs = abc_rates()
        result = ChaosResult(
            inputs=["k1", "k2"],
            input_space=inputs,
            labels=["B at 5"],
            degree=2,
            q_norm=1.0,
            max_interaction=1,
            indices=[[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]],
            coefficients=[[1.0, 0.0, 0.0, 0.0, 0.0]],
            loo_error=[0.0],
            mean=[1.0],
            variance=[0.0],
            first=[[math.nan, math.nan]],
            total=[[math.nan, math.nan]],
            runs=10,
        )

        assert result.predict([[0.3, 0.5]]) == [[1.0]]
        with pytest.raises(
            ValueError, match=r"q_norm is a number greater than 0 and at most 1, got 2"
        ):
            result.model_copy(update={"q_norm": 2.0}).predict([[0.3, 0.5]])
        with pytest.raises(ValueError, match=r"has 6 terms; a row of coefficients"):
            result.model_copy(update={"max_interaction": None}).predict([[0.3, 0.5]])


class TestEntryPoint:
    def test_each_method_returns_its_result(self, simulation: Simulation) -> None:
        analysis = SensitivityAnalysis(simulation, abc_rates(), seed=3)
        quantities: list[_Quantity] = [ValueAt(series="B", at=10)]

        propagation = analysis.run(Propagation(draws=4, quantities=quantities))
        prcc = analysis.run(PRCC(draws=8, quantities=quantities))
        morris = analysis.run(Morris(trajectories=2, levels=4, quantities=quantities))
        sobol = analysis.run(
            Sobol(draws=8, replicates=10, confidence_level=0.9, quantities=quantities)
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


def test_sobol_takes_draws_and_replicates(simulation: Simulation) -> None:
    analysis = SensitivityAnalysis(simulation, abc_rates(), seed=3)

    result = analysis.run(
        Sobol(
            draws=8,
            replicates=2,
            confidence_level=0.9,
            quantities=[ValueAt(series="B", at=5)],
        )
    )

    assert result.runs == 8 * (2 + 2)
    assert result.confidence_level == 0.9
