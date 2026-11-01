"""Calibration of a stochastic engine against data of its own runs.

Each evaluation runs the same replicates (common random numbers, seeded by
`CalibrationProblem.seed`), so the loss is a deterministic function of the
parameters and Nelder-Mead converges.
"""

from typing import Literal

import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Quantile,
    Simulation,
)
from commol._fit.result import CalibrationResult
from commol.engines import StochasticEventsSettings, StochasticStepsSettings
from commol.engines.engine import EngineName
from commol.quantities import ValueAt
from commol.sensitivity import InputSpace, Propagation, SensitivityAnalysis
from commol.uncertainty import ParameterSample, SampleMethod, weighted_quantiles
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
    engine: EngineName,
    summary: Literal["mean", "median", "crps"] | Quantile,
    max_iterations: int = 200,
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


def _calibrate(
    engine: EngineName, summary: Literal["mean", "median", "crps"] | Quantile
) -> CalibrationResult:
    return _calibrator(engine, summary).run()


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
class TestCalibration:
    @pytest.mark.parametrize("summary", ["mean", "crps"])
    def test_a_rate_is_recovered_from_the_runs(
        self, engine: EngineName, summary: Literal["mean", "median", "crps"] | Quantile
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
        (replicates,) = calibrator._engine_problem.predict_observations([values])
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


def _amount_problem(low: float) -> CalibrationProblem:
    """A calibration of the initial amount of bin B in ``[low, 50]``."""
    return CalibrationProblem(
        observed_data=[ObservedDataPoint(at=2.0, output="A", value=800.0)],
        parameters=[bounded("B", low, 50.0, kind="initial_condition")],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=3),
        replicates=2,
        replicate_summary="mean",
        seed=7,
    )


def _needs_a_rounding(engine: EngineName, entry_id: str = "B") -> str:
    return (
        f"The {engine} engine moves whole units, and a method that varies the "
        f"initial condition '{entry_id}' can write amounts that are not whole. Name "
        'a rounding in the settings of the engine: "largest_remainder" or "random"'
    )


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
@pytest.mark.parametrize("low", [0.0, 0.5])
def test_a_calibrated_amount_needs_a_rounding(engine: EngineName, low: float) -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), engine))

    with pytest.raises(ValueError) as error:
        Calibrator(simulation, _amount_problem(low))

    assert str(error.value) == _needs_a_rounding(engine)


@pytest.mark.parametrize(
    ("entry_id", "high", "remainder"),
    [("B_g1", 50.0, None), ("g1", 1.0, None), ("B", 50.0, "A")],
    ids=["compartment", "category", "with a remainder"],
)
def test_every_initial_condition_that_varies_needs_a_rounding(
    entry_id: str, high: float, remainder: str | None
) -> None:
    builder = a_to_b_builder(stratifications={"group": {"g1": 0.5, "g2": 0.5}})
    simulation = Simulation(build_stochastic(builder, "StochasticSteps"))
    problem = _amount_problem(0.0).model_copy(
        update={
            "parameters": [bounded(entry_id, 0.0, high, kind="initial_condition")],
            "initial_condition_remainder": remainder,
        }
    )

    with pytest.raises(ValueError) as error:
        Calibrator(simulation, problem)

    assert str(error.value) == _needs_a_rounding("StochasticSteps", entry_id)


def test_an_analysis_that_varies_an_amount_needs_a_rounding() -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
    inputs = InputSpace.uniform_on_bounds(
        [bounded("B", 0.0, 50.0, kind="initial_condition")]
    )

    with pytest.raises(ValueError) as error:
        SensitivityAnalysis(simulation, inputs, replicates=2, seed=1)

    assert str(error.value) == _needs_a_rounding("StochasticSteps")


def _sample_of_b(draws: list[float]) -> InputSpace:
    """The initial amount of bin B, from a sample of ``draws``."""
    sample = ParameterSample.equally_weighted(
        ["B"], [(0.0, 50.0)], [[draw] for draw in draws], SampleMethod(name="grid")
    )
    return InputSpace.from_sample(
        sample, [bounded("B", 0.0, 50.0, kind="initial_condition")]
    )


def test_an_analysis_of_a_sample_writes_its_draws_as_known_amounts() -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
    analysis = SensitivityAnalysis(
        simulation, _sample_of_b([10.0, 20.0, 30.0]), replicates=2, seed=1
    )

    result = analysis.run(Propagation(quantities=[ValueAt(series="B", at=0)]))

    assert [run[0] for run in result.values] == [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
    with pytest.raises(ValueError, match=r"12\.3 is not whole"):
        SensitivityAnalysis(simulation, _sample_of_b([12.3]), replicates=2, seed=1)


def test_a_known_amount_is_written_back_when_it_is_whole() -> None:
    model = build_stochastic(a_to_b_builder(), "StochasticSteps")
    problem = _amount_problem(0.0)

    with pytest.raises(ValueError, match=r"12\.3 is not whole"):
        model.apply_calibration_parameters({"B": 12.3}, problem)
    model.apply_calibration_parameters({"B": 12.0}, problem)
    start = Simulation(model).run_at([0.0], seed=1)

    assert start["B"] == [12.0]


@pytest.mark.parametrize(
    ("engine", "settings"),
    [
        ("StochasticEvents", StochasticEventsSettings(rounding="largest_remainder")),
        ("StochasticSteps", StochasticStepsSettings(rounding="largest_remainder")),
    ],
)
def test_a_calibrated_amount_with_a_rounding_is_calibrated(
    engine: EngineName, settings: StochasticEventsSettings | StochasticStepsSettings
) -> None:
    model = build_stochastic(a_to_b_builder(), engine, settings)
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=0.0, output="B", value=12.3)],
        parameters=[
            bounded("B", 0.0, 50.0, kind="initial_condition", initial_guess=25.0)
        ],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=100),
        replicates=2,
        replicate_summary="mean",
        seed=7,
    )

    result = Calibrator(Simulation(model), problem).run()

    # Every run starts from 12 whole units of B, the nearest to 12.3: with no
    # rounding the loss would reach 0.
    assert result.final_loss == pytest.approx((12.0 - 12.3) ** 2, rel=1e-9)
