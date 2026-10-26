"""Calibration of a stochastic engine against data of its own runs.

Each evaluation runs the same replicates (common random numbers, seeded by
`CalibrationProblem.seed`), so the loss is a deterministic function of the
parameters and Nelder-Mead converges.
"""

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Quantile,
    Simulation,
    weighted_quantiles,
)
from commol.context.calibration import CalibrationResult, ReplicateSummary
from commol.context.engine import EngineName
from tests.support import STOCHASTIC_ENGINES, a_to_b_builder, bounded, build_stochastic

RATE = 0.1
POSITIONS = [2.0, 5.0, 10.0]


def _data(engine: EngineName) -> list[ObservedDataPoint]:
    """The mean of A over 200 runs at the rate `RATE`."""
    simulation = Simulation(build_stochastic(a_to_b_builder(), engine))
    mean = simulation.replicates(200, seed=21).run_at(POSITIONS).mean()
    return [
        ObservedDataPoint(at=at, output="A", value=value)
        for at, value in zip(POSITIONS, mean["A"], strict=True)
    ]


def _calibrator(
    engine: EngineName, summary: ReplicateSummary, max_iterations: int = 200
) -> Calibrator:
    builder = a_to_b_builder(parameters={"k": None})
    problem = CalibrationProblem(
        observed_data=_data(engine),
        parameters=[bounded("k", 0.01, 0.5, initial_guess=0.3)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(
            max_iterations=max_iterations, sd_tolerance=1e-10
        ),
        replicates=40,
        replicate_summary=summary,
        seed=7,
    )
    simulation = Simulation(build_stochastic(builder, engine))
    return Calibrator(simulation, problem)


def _calibrate(engine: EngineName, summary: ReplicateSummary) -> CalibrationResult:
    return _calibrator(engine, summary).run()


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
class TestCalibration:
    @pytest.mark.parametrize("summary", ["mean", "crps"])
    def test_a_rate_is_recovered_from_the_runs(
        self, engine: EngineName, summary: ReplicateSummary
    ) -> None:
        result = _calibrate(engine, summary)

        # 200 runs of 1000 units against 40 replicates: within a few percent.
        assert result.best_parameters["k"] == pytest.approx(RATE, rel=0.03)

    def test_one_seed_gives_the_same_calibration(self, engine: EngineName) -> None:
        first = _calibrate(engine, "mean")
        second = _calibrate(engine, "mean")

        assert first.best_parameters == second.best_parameters
        assert first.final_loss == second.final_loss


def test_a_quantile_summary_agrees_with_the_quantiles_of_the_replicates() -> None:
    calibrator = _calibrator("StochasticEvents", Quantile(0.25), max_iterations=3)
    observed = [point.value for point in calibrator.problem.observed_data]

    history = calibrator.run_with_history()

    names = calibrator.parameter_names
    assert history.evaluations
    for evaluation in history.evaluations:
        values = [evaluation.parameters[name] for name in names]
        (replicates,) = calibrator._rust_problem.predict_observations([values])
        summaries = [
            weighted_quantiles([replicate[i] for replicate in replicates], [0.25])[0]
            for i in range(len(observed))
        ]
        expected = sum(
            (value - summary) ** 2
            for value, summary in zip(observed, summaries, strict=True)
        )
        assert evaluation.loss == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("verbose", [False, True])
def test_a_start_whose_run_fails_is_named_in_the_error(verbose: bool) -> None:
    builder = a_to_b_builder(parameters={"k": None}, rate="k * sqrt(0.5 - k)")
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=1, output="A", value=900.0)],
        parameters=[bounded("k", 0.6, 0.9)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=3, verbose=verbose),
        replicates=2,
        replicate_summary="mean",
        seed=1,
    )
    calibrator = Calibrator(
        Simulation(build_stochastic(builder, "StochasticEvents")), problem
    )

    with pytest.raises(RuntimeError, match="Start 0 failed: .*is not a number"):
        calibrator.run()


def test_the_engine_asks_for_replicates() -> None:
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=2.0, output="A", value=800.0)],
        parameters=[bounded("k", 0.01, 0.5, initial_guess=0.3)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
        replicate_summary="mean",
        seed=7,
    )
    simulation = Simulation(
        build_stochastic(a_to_b_builder(parameters={"k": None}), "StochasticEvents")
    )

    with pytest.raises(ValueError) as error:
        Calibrator(simulation, problem)

    assert str(error.value) == (
        "The StochasticEvents engine is stochastic: set replicates, how many "
        "replicates each evaluation runs."
    )
