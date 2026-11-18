"""Calibration of a stochastic engine against data of its own runs.

Each evaluation runs the same replicates (common random numbers, seeded by
`CalibrationProblem.seed`), so the loss is a deterministic function of the
parameters and Nelder-Mead converges.
"""

from pathlib import Path
from typing import Literal

import pytest

from commol import (
    Aggregate,
    CalibrationProblem,
    Calibrator,
    Model,
    NelderMeadConfig,
    ObservedDataPoint,
    Quantile,
    Simulation,
)
from commol._fit.result import CalibrationResult
from commol._model.engine_state import calibrated_initial_state
from commol.engines import StochasticEventsSettings, StochasticStepsSettings
from commol.engines.engine import EngineName, Rounding
from commol.uncertainty import weighted_quantiles
from tests.support import (
    STOCHASTIC_ENGINES,
    a_to_b_builder,
    bounded,
    build_stochastic,
    needs_a_rounding,
)

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


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
@pytest.mark.parametrize("low", [0.0, 0.5])
def test_a_calibrated_amount_needs_a_rounding(engine: EngineName, low: float) -> None:
    simulation = Simulation(build_stochastic(a_to_b_builder(), engine))

    with pytest.raises(ValueError) as error:
        Calibrator(simulation, _amount_problem(low))

    assert str(error.value) == needs_a_rounding(engine)


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

    assert str(error.value) == needs_a_rounding("StochasticSteps", entry_id)


def _with_rounding(
    engine: EngineName, rounding: Rounding
) -> StochasticEventsSettings | StochasticStepsSettings:
    """The settings of `engine` that name `rounding`."""
    if engine == "StochasticEvents":
        return StochasticEventsSettings(rounding=rounding)
    return StochasticStepsSettings(rounding=rounding)


def _calibrated_amount(
    engine: EngineName, rounding: Rounding
) -> tuple[Model[Aggregate], CalibrationProblem, Calibrator]:
    """A calibration of the initial amount of bin B in ``[0, 50]`` against
    12.3 units of B at 0, on `engine` with `rounding`, and its model."""
    model = build_stochastic(a_to_b_builder(), engine, _with_rounding(engine, rounding))
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
    return model, problem, Calibrator(Simulation(model), problem)


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
def test_a_calibrated_amount_with_a_rounding_is_calibrated(engine: EngineName) -> None:
    _, _, calibrator = _calibrated_amount(engine, "largest_remainder")

    result = calibrator.run()

    # Every run starts from 12 whole units of B, the nearest to 12.3: with no
    # rounding the loss would reach 0.
    assert result.final_loss == pytest.approx((12.0 - 12.3) ** 2, rel=1e-9)


@pytest.mark.parametrize("rounding", ["largest_remainder", "random"])
@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
def test_a_calibrated_amount_is_written_back_onto_a_model_that_builds(
    engine: EngineName, rounding: Rounding
) -> None:
    model, problem, calibrator = _calibrated_amount(engine, rounding)
    result = calibrator.run()
    amount = result.best_parameters["B"]
    (scored,) = calibrator._engine_problem.predict_observations([[amount]])

    model.apply_calibration_parameters(result, problem)
    starts = Simulation(model).replicates(2, seed=7).run_at([0.0])

    # The model holds the amount the optimizer found, which is not whole, and
    # every run starts from the units the calibration scored for it.
    assert amount != round(amount)
    assert model.structure.bin_amounts["B"] == amount
    assert [run["B"] for run in starts] == scored == [[12.0], [12.0]]


def test_a_stratified_amount_is_written_back_with_its_split() -> None:
    builder = a_to_b_builder(stratifications={"group": {"g1": 0.3, "g2": 0.7}})
    settings = StochasticStepsSettings(rounding="largest_remainder")
    model = build_stochastic(builder, "StochasticSteps", settings)
    targets = {"A_g1": 296.3, "A_g2": 691.4, "B_g1": 3.7, "B_g2": 8.6}
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=0.0, output=output, value=value)
            for output, value in targets.items()
        ],
        parameters=[
            bounded("B", 0.0, 50.0, kind="initial_condition", initial_guess=25.0)
        ],
        initial_condition_remainder="A",
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=100),
        replicates=2,
        replicate_summary="mean",
        seed=7,
    )
    calibrator = Calibrator(Simulation(model), problem)
    result = calibrator.run()
    amount = result.best_parameters["B"]
    (scored,) = calibrator._engine_problem.predict_observations([[amount]])
    engine_state = calibrated_initial_state(model, problem.parameters, [amount], "A")

    model.apply_calibration_parameters(result, problem)
    starts = Simulation(model).replicates(2, seed=7).run_at([0.0])

    # Each bin holds the amount the engine scored for it and keeps its split.
    # The rounding can turn on the last bit of an amount, so each compartment
    # that the bins and the split rebuild only to their last bits gets the
    # amount the engine scored as a scored amount, never an override, and
    # every run starts from the units the calibration scored.
    scored_amounts = model.structure.scored_amounts
    declared = model.structure.model_copy(update={"scored_amounts": {}})
    rebuilt = {c.name: c.amount for c in declared.initial_state(model.process)}
    written = {c.name: c.amount for c in model.structure.initial_state(model.process)}
    assert model.structure.bin_amounts == pytest.approx(
        {"A": 1000.0 - amount, "B": amount}, rel=1e-12
    )
    assert written == engine_state.state
    assert model.structure.compartment_amounts == {}
    assert {
        name: (entry.built, entry.scored) for name, entry in scored_amounts.items()
    } == {
        name: (rebuilt[name], written[name])
        for name in rebuilt
        if rebuilt[name] != written[name]
    }
    assert [[run[output][0] for output in targets] for run in starts] == scored


@pytest.mark.parametrize("rounding", ["largest_remainder", "random"])
@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
def test_a_written_back_model_runs_as_the_calibration_scored_it(
    engine: EngineName, rounding: Rounding, tmp_path: Path
) -> None:
    split = {"g1": 0.7, "g2": 0.2, "g3": 0.1}
    builder = a_to_b_builder(stratifications={"group": split})
    model = build_stochastic(builder, engine, _with_rounding(engine, rounding))
    outputs = [f"{bin_id}_{group}" for bin_id in ("A", "B") for group in split]
    positions = [0.0, 3.0, 8.0]
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(at=at, output=output, value=1.0)
            for at in positions
            for output in outputs
        ],
        parameters=[bounded("B", 0.0, 50.0, kind="initial_condition")],
        initial_condition_remainder="A",
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
        replicates=2,
        replicate_summary="mean",
        seed=7,
    )
    calibrator = Calibrator(Simulation(model), problem)
    # B = 12 splits into 8.4, 2.4 and 1.2: two remainders tie. For the other
    # amounts, the bins and the split rebuild the scored shares only to their
    # last bits, which shifts the draws of "random".
    for amount in [12.0, *(0.5 + 3.7 * i for i in range(13))]:
        (scored,) = calibrator._engine_problem.predict_observations([[amount]])
        written = model.model_copy(deep=True)

        written.apply_calibration_parameters({"B": amount}, problem)
        written.to_json(tmp_path / "model.json")
        loaded = Model.from_json(tmp_path / "model.json")

        # Each replicate of the model, and of the model saved and loaded,
        # starts from the units the calibration scored and follows the same
        # path.
        for runnable in (written, loaded):
            runs = Simulation(runnable).replicates(2, seed=7).run_at(positions)
            assert [
                [
                    run[output][index]
                    for index in range(len(positions))
                    for output in outputs
                ]
                for run in runs
            ] == scored, amount


def test_successive_write_backs_end_with_the_scored_amounts_of_a_fresh_one() -> None:
    builder = a_to_b_builder(
        stratifications={"group": {"g1": 0.7, "g2": 0.2, "g3": 0.1}}
    )
    settings = StochasticStepsSettings(rounding="random")
    model = build_stochastic(builder, "StochasticSteps", settings)
    fresh = model.model_copy(deep=True)
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=0.0, output="B", value=1.0)],
        parameters=[bounded("B", 0.0, 50.0, kind="initial_condition")],
        initial_condition_remainder="A",
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
        replicates=2,
        replicate_summary="mean",
        seed=7,
    )

    model.apply_calibration_parameters({"B": 0.37}, problem)
    first = dict(model.structure.scored_amounts)
    model.apply_calibration_parameters({"B": 1.5}, problem)
    fresh.apply_calibration_parameters({"B": 1.5}, problem)

    # A write-back replaces the scored amounts of an earlier one: 1.5 split
    # 0.7, 0.2, 0.1 rebuilds only B_g1 other than in its last bit, where 0.37
    # rebuilt B_g2 and B_g3 so.
    assert set(first) == {"B_g2", "B_g3"}
    assert model.structure.scored_amounts == fresh.structure.scored_amounts
    assert set(model.structure.scored_amounts) == {"B_g1"}
    assert model.structure.compartment_amounts == {}
    assert model.structure.initial_state(
        model.process
    ) == fresh.structure.initial_state(fresh.process)
