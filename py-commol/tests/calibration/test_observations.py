"""Factors and known scales read observations that see only a share of an
output."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from commol import (
    CalibrationConstraint,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol._model.model import Model
from commol.local import Laplace, Sandwich
from commol.observations import DistributionNoise, NegativeBinomialObservations
from commol.uncertainty import ParameterSample, Predictive, SampleMethod
from tests.support import (
    ENGINE_REL,
    a_to_b_builder,
    abc_builder,
    bounded,
    build,
    build_ode,
    build_stochastic,
    observed,
    sum_calibrator,
)

SEED = 42


def _fit(
    truth: Model,
    observations,
    parameters,
    config: NelderMeadConfig | ParticleSwarmConfig,
):
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=parameters,
        loss_function="sse",
        optimization_config=config,
        seed=SEED,
    )
    return Calibrator(Simulation(truth), problem).run()


class TestScaleFactors:
    def test_a_scale_alone_is_recovered(self, truth: Model, truth_run) -> None:
        observations = observed(
            truth_run, "B", range(0, 100, 10), factor=0.5, factor_id="detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [bounded("detection_rate", 0.1, 1.0, kind="factor", initial_guess=0.7)],
            NelderMeadConfig(max_iterations=500, verbose=False),
        )

        assert result.best_parameters["detection_rate"] == pytest.approx(0.5, abs=1e-4)
        assert result.converged

    def test_a_scale_and_a_parameter_are_recovered_together(
        self, truth: Model, truth_run
    ) -> None:
        observations = observed(
            truth_run, "B", range(0, 100, 5), factor=0.3, factor_id="detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [bounded("k1"), bounded("detection_rate", 0.1, 1.0, kind="factor")],
            ParticleSwarmConfig(max_iterations=300, particles=25, verbose=False),
        )

        assert result.best_parameters == pytest.approx(
            {"k1": 0.1, "detection_rate": 0.3}, abs=1e-3
        )

    def test_each_output_has_its_own_scale(self, truth: Model, truth_run) -> None:
        observations = observed(
            truth_run, "B", range(0, 100, 10), factor=0.6, factor_id="i_detection_rate"
        ) + observed(
            truth_run, "C", range(0, 100, 10), factor=0.9, factor_id="r_detection_rate"
        )

        result = _fit(
            truth,
            observations,
            [
                bounded("i_detection_rate", 0.1, 1.0, kind="factor"),
                bounded("r_detection_rate", 0.1, 1.0, kind="factor"),
            ],
            NelderMeadConfig(max_iterations=500, verbose=False),
        )

        assert result.best_parameters == pytest.approx(
            {"i_detection_rate": 0.6, "r_detection_rate": 0.9}, abs=1e-3
        )


ODE_TRUTH = {"k1": 0.3, "k2": 0.1}
STEPS = [0.5 + 1.5 * j for j in range(40)]
"""Observation steps, none of them an integer."""


def _fit_ode(problem: CalibrationProblem, model) -> dict[str, float]:
    return Calibrator(Simulation(model), problem).run().best_parameters


class TestStepsBetweenIntegers:
    """A DifferentialEquations model observed at any step."""

    def test_observations_at_non_integer_steps_recover_the_parameters(self) -> None:
        truth = build_ode(abc_builder())
        results = Simulation(truth).run_at(STEPS)
        problem = CalibrationProblem(
            observed_data=[
                ObservedDataPoint(at=step, output="B", value=value)
                for step, value in zip(STEPS, results["B"], strict=True)
            ],
            parameters=[
                bounded("k1", initial_guess=0.5),
                bounded("k2", high=0.5, initial_guess=0.2),
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=2000, verbose=False),
        )

        fitted = _fit_ode(
            problem, build_ode(abc_builder(parameters={"k1": None, "k2": None}))
        )

        assert fitted == pytest.approx(ODE_TRUTH, rel=1e-4)

    def test_a_window_over_non_integer_steps_recovers_the_rate(self) -> None:
        truth = build_ode(a_to_b_builder(parameters={"k": 0.2}, accumulator="moved"))
        results = Simulation(truth).run_at([0.5, 3.5, 6.5])
        observations = [
            ObservedDataPoint(
                at=end,
                output="moved",
                value=results["moved"][i + 1] - results["moved"][i],
                window=3.0,
            )
            for i, end in enumerate([3.5, 6.5])
        ]
        problem = CalibrationProblem(
            observed_data=observations,
            parameters=[bounded("k", initial_guess=0.5)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=500, verbose=False),
        )

        fitted = _fit_ode(
            problem,
            build_ode(a_to_b_builder(parameters={"k": None}, accumulator="moved")),
        )

        assert fitted["k"] == pytest.approx(0.2, rel=1e-5)


class TestPositionsAndEngines:
    def test_one_output_names_its_series(self) -> None:
        point = ObservedDataPoint(at=3, output="B", value=1.0)

        assert point.label == "B" and point.observed_outputs == ["B"]

    def test_a_sum_of_outputs_needs_a_label(self) -> None:
        with pytest.raises(ValidationError, match="label"):
            ObservedDataPoint(at=3, outputs=["B", "C"], value=1.0)

    def test_output_and_outputs_exclude_each_other(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            ObservedDataPoint(at=3, output="B", outputs=["C"], label="x", value=1.0)

    def test_a_sum_needs_an_output(self) -> None:
        with pytest.raises(ValidationError, match="outputs must be non-empty"):
            ObservedDataPoint(at=3, outputs=[], label="x", value=1.0)

    def test_a_window_does_not_start_before_zero(self) -> None:
        with pytest.raises(ValidationError, match="window 4 that ends at 3 starts"):
            ObservedDataPoint(at=3, output="B", window=4, value=1.0)

    def test_a_window_that_starts_at_zero_within_its_rounding_is_calibrated(
        self,
    ) -> None:
        observation = ObservedDataPoint(
            at=1000, output="A", value=1.0, window=1000 + 1e-7
        )
        problem = CalibrationProblem(
            observed_data=[observation],
            parameters=[bounded("k")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        model = build(a_to_b_builder(parameters={"k": None}))

        calibrator = Calibrator(Simulation(model), problem)

        [[[prediction]]] = calibrator._engine_problem.predict_observations([[0.0]])
        assert prediction == 0.0

    def test_a_difference_model_rejects_a_non_integer_observation(self) -> None:
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=2.5, output="A", value=1.0)],
            parameters=[bounded("k")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        model = build(a_to_b_builder(parameters={"k": None}))

        with pytest.raises(ValueError, match="Observation 'A': .*2.5 is not one"):
            Calibrator(Simulation(model), problem)

    def test_a_difference_model_rejects_a_non_integer_constraint_step(self) -> None:
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=2, output="A", value=1.0)],
            parameters=[bounded("k")],
            constraints=[CalibrationConstraint(id="cap", expression="1 - A", at=[1.5])],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        model = build(a_to_b_builder(parameters={"k": None}))

        with pytest.raises(ValueError, match="Constraint 'cap': .*1.5 is not one"):
            Calibrator(Simulation(model), problem)


def _count_calibrator(scale: float) -> Calibrator:
    """
    A negative binomial fit of a flow ``c`` from A to B (``c`` per step, 10⁶
    in A), observed as 7 units of B over the first step with `scale`.
    """
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=1, output="B", value=7.0, window=1, scale=scale)
        ],
        parameters=[bounded("c", 1.0, 1000.0)],
        observation_models={"B": NegativeBinomialObservations(overdispersion=0.1)},
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    return Calibrator(Simulation(model), problem)


def test_an_observation_reads_the_model_times_its_scale() -> None:
    scaled = _count_calibrator(0.025)._engine_problem
    unscaled = _count_calibrator(1.0)._engine_problem

    [[[prediction]]] = scaled.predict_observations([[400.0]])

    # The negative binomial reads the mean 10 = 0.025 · 400.
    assert prediction == pytest.approx(10.0, rel=ENGINE_REL)
    assert scaled.pointwise_log_likelihood([400.0]) == pytest.approx(
        unscaled.pointwise_log_likelihood([10.0]), rel=ENGINE_REL
    )


@pytest.mark.parametrize("scale", [0.0, -0.5, math.inf, math.nan])
def test_a_scale_that_is_not_positive_is_refused(scale: float) -> None:
    with pytest.raises(ValidationError, match="scale is a finite number"):
        ObservedDataPoint(at=1, output="B", value=1.0, scale=scale)


def _windows(
    values: list[float], parts: list[int], *, length: int = 6, label: str = "B"
) -> list[ObservedDataPoint]:
    """Observations of B over consecutive windows of `length` ending at
    `length`, `2 · length`, …, with the `parts` of each."""
    return [
        ObservedDataPoint(
            at=length * (index + 1),
            output="B",
            label=label,
            value=value,
            window=length,
            parts=count,
        )
        for index, (value, count) in enumerate(zip(values, parts, strict=True))
    ]


class TestParts:
    def test_parts_need_a_window(self) -> None:
        with pytest.raises(
            ValidationError,
            match=(
                "Observation 'B' at 10 has parts=2 and no window; parts split "
                "the window"
            ),
        ):
            ObservedDataPoint(at=10, output="B", value=1.0, parts=2)
        with pytest.raises(
            ValidationError, match="parts is a whole number of at least 1"
        ):
            ObservedDataPoint(at=10, output="B", value=1.0, window=10, parts=0)

    def test_one_part_is_the_window(self) -> None:
        # Reference values of a window of one part:
        # the predictions, log densities, draws and the Laplace fit and
        # covariance of a negative binomial series.
        observations = _windows([93, 71, 80, 49, 62, 41, 30, 44], [1] * 8, length=10)
        model = build(
            a_to_b_builder(parameters={"k": 0.1}, amounts={"A": 1000, "B": 0})
        )
        problem = CalibrationProblem(
            observed_data=observations,
            parameters=[
                bounded("k", 0.001, 0.5),
                bounded("phi", 0.01, 3.0, kind="noise"),
            ],
            observation_models={
                "B": NegativeBinomialObservations(overdispersion="phi")
            },
            optimization_config=NelderMeadConfig(max_iterations=400),
            seed=3,
        )
        calibrator = Calibrator(Simulation(model), problem)
        engine = calibrator._engine_problem
        point = [0.01, 0.4]

        assert engine.predict_observations([point]) == [
            [
                [
                    95.61792499119552,
                    86.47513741157364,
                    78.20656420895048,
                    70.72861481859985,
                    63.96569143214384,
                    57.84942474677513,
                    52.317982790554254,
                    47.31544583639652,
                ]
            ]
        ]
        assert engine.pointwise_log_likelihood(point) == [
            -5.040772302400001,
            -4.818167222686185,
            -4.892027627621587,
            -4.554502303962181,
            -4.642067352360101,
            -4.36359595393138,
            -4.241396970789552,
            -4.311988152637724,
        ]
        sample = ParameterSample.equally_weighted(
            ["k", "phi"],
            [(0.001, 0.5), (0.01, 3.0)],
            [point, [0.012, 0.3]],
            SampleMethod(name="point"),
        )
        draws = Predictive(calibrator, sample, [0, 10]).at_observations(
            observations, DistributionNoise()
        )
        assert draws.values == [
            [97.0, 144.0, 139.0, 76.0, 50.0, 174.0, 33.0, 73.0],
            [57.0, 65.0, 70.0, 15.0, 140.0, 29.0, 104.0, 12.0],
        ]
        result = calibrator.uncertainty(Laplace(draws=10))
        assert result.fit.best_parameters == {
            "k": 0.008110949594956205,
            "phi": 0.023657880499419526,
        }
        assert result.covariance == [
            [0.013258180575577419, 0.0],
            [0.0, 2.2397584820553185],
        ]

    def test_a_label_mixes_any_parts(self) -> None:
        values = [250.0, 130.0, 150.0, 70.0, 95.0]
        parts = [1, 2, 3, 2, 1]
        phi = [bounded("phi", 0.01, 3.0, kind="noise")]
        mixed = sum_calibrator(
            _windows(values, parts, length=6),
            {"B": NegativeBinomialObservations(overdispersion="phi", sum="exact")},
            noise=phi,
        )
        # The same observations in two labels: the windows of one part, and
        # those of several, which share the noise parameter.
        split = [
            ObservedDataPoint(
                **(point.model_dump() | {"label": "B" if point.parts == 1 else "sums"})
            )
            for point in _windows(values, parts, length=6)
        ]
        two_labels = sum_calibrator(
            split,
            {
                "B": NegativeBinomialObservations(overdispersion="phi"),
                "sums": NegativeBinomialObservations(overdispersion="phi", sum="exact"),
            },
            noise=phi,
        )
        point = [0.04, 0.2]

        assert mixed._engine_problem.pointwise_log_likelihood(
            point
        ) == two_labels._engine_problem.pointwise_log_likelihood(point)
        for method in (Laplace(draws=10), Sandwich(block_length=1, draws=10)):
            one, other = (
                calibrator.uncertainty(method) for calibrator in (mixed, two_labels)
            )
            assert one.fit.best_parameters == other.fit.best_parameters
            assert np.allclose(one.covariance, other.covariance, rtol=1e-12, atol=0.0)

    def test_a_loss_refuses_parts_and_trials(self) -> None:
        sums = _windows([10.0], [2], length=4)
        trials = [ObservedDataPoint(at=4, output="B", value=10.0, trials=20)]
        for observations, field in ((sums, "parts=2"), (trials, "trials=20")):
            problem = CalibrationProblem(
                observed_data=observations,
                parameters=[bounded("k")],
                loss_function="sse",
                optimization_config=NelderMeadConfig(max_iterations=1),
            )
            model = build(a_to_b_builder(parameters={"k": None}))

            with pytest.raises(
                ValueError,
                match=(
                    f"Observation 'B' at 4 has {field}, a field of an observation "
                    "model, which a problem with a loss does not read: it needs "
                    "observation_models instead of loss_function$"
                ),
            ):
                Calibrator(Simulation(model), problem)

        stochastic = build_stochastic(
            a_to_b_builder(parameters={"k": None}, amounts={"A": 100, "B": 0})
        )
        problem = CalibrationProblem(
            observed_data=sums,
            parameters=[bounded("k")],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
            replicates=2,
            replicate_summary="mean",
            seed=1,
        )
        with pytest.raises(
            ValueError,
            match=(
                "it needs observation_models instead of loss_function; observation "
                "models need a deterministic engine, and this engine is stochastic"
            ),
        ):
            Calibrator(Simulation(stochastic), problem)
