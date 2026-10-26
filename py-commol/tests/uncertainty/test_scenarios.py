"""A sample run under scenarios: overrides, common seeds, paired comparisons
of quantities and the point estimate of a fit as a sample."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from commol import (
    CalibrationProblem,
    Calibrator,
    InputSpace,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    Predictive,
    SampleMethod,
    Scenarios,
    SensitivityAnalysis,
    Simulation,
    Trajectory,
    ValueAt,
    weighted_quantiles,
)
from commol.context.calibration import (
    CalibrationResult,
)
from commol.context.inference.quantities import quantity_labels
from tests.support import (
    a_to_b_builder,
    abc_builder,
    bounded,
    build,
    build_stochastic,
    observed,
)

POSITIONS = list(range(31))
SCENARIOS = {"base": {}, "slow": {"k2": 0.02}}


def _calibrator() -> Calibrator:
    """``k1`` of ``abc_builder`` with ``k2 = 0.1``, observed in B."""
    model = build(abc_builder(parameters={"k1": None, "k2": 0.1}))
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=10, output="B", value=50.0)],
        parameters=[bounded("k1", 0.0, 0.5)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
        seed=3,
    )
    return Calibrator(Simulation(model), problem)


def _sample(values: list[float], weights: list[float] | None = None):
    return ParameterSample(
        names=["k1"],
        bounds=[(0.0, 0.5)],
        values=[[value] for value in values],
        log_weights=[math.log(w) for w in weights or [1.0] * len(values)],
        method=SampleMethod(name="test"),
    )


def _run(k1: float, k2: float) -> dict[str, list[float]]:
    model = build(abc_builder(parameters={"k1": k1, "k2": k2}))
    return Simulation(model).run_at(POSITIONS)


def test_each_draw_runs_with_the_overrides_of_its_scenario() -> None:
    calibrator = _calibrator()

    scenarios = Scenarios(calibrator, _sample([0.2, 0.3]), SCENARIOS, POSITIONS)

    assert list(scenarios) == ["base", "slow"]
    assert len(scenarios) == 2
    for name, k2 in (("base", 0.1), ("slow", 0.02)):
        for run, k1 in zip(scenarios[name].runs(), [0.2, 0.3], strict=True):
            np.testing.assert_allclose(run["C"], _run(k1, k2)["C"], rtol=1e-12)
    model = calibrator.simulation.model_definition
    assert {p.id: p.value for p in model.parameters} == {"k1": None, "k2": 0.1}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"k1": 0.2}, r"Scenario 'bad' sets \['k1'\], which the sample holds"),
        ({"k9": 1.0}, "Scenario 'bad': Parameter 'k9' not found in model"),
    ],
    ids=["held by the sample", "not in the model"],
)
def test_a_scenario_cannot_override_a_parameter_of_the_sample(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Scenarios(_calibrator(), _sample([0.2]), {"bad": overrides}, POSITIONS)


def test_the_draws_of_every_scenario_share_their_seeds() -> None:
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=5, output="A", value=600.0)],
        parameters=[bounded("k", 0.01, 0.5)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
        replicates=3,
        replicate_summary="mean",
        seed=7,
    )
    model = build_stochastic(a_to_b_builder(parameters={"k": None}), "StochasticSteps")
    calibrator = Calibrator(Simulation(model), problem)
    sample = ParameterSample.equally_weighted(
        ["k"], [(0.01, 0.5)], [[0.1], [0.2]], SampleMethod(name="t")
    )

    scenarios = Scenarios(calibrator, sample, {"a": {}, "b": {}}, [0, 5, 10])
    compared = scenarios.compare("a", "b", [ValueAt(series="B", at=10)], 0.9)

    seeds = Predictive(calibrator, sample, [0, 5, 10]).draw_seeds
    assert scenarios["a"].draw_seeds == scenarios["b"].draw_seeds == seeds
    values = [run["B"][2] for run in scenarios["a"].runs()]
    assert len(set(values)) > 1
    assert (compared[0].lower, compared[0].upper) == (0.0, 0.0)


def test_the_interval_of_a_difference_is_taken_draw_by_draw() -> None:
    weights = [0.1, 0.4, 0.2, 0.2, 0.1]
    sample = _sample([0.15, 0.2, 0.25, 0.3, 0.35], weights)
    scenarios = Scenarios(_calibrator(), sample, SCENARIOS, POSITIONS)

    [compared] = scenarios.compare("slow", "base", [ValueAt(series="C", at=30)], 0.8)
    [summary] = scenarios.summary("slow", [ValueAt(series="C", at=30)], 0.8)

    differences = [
        slow["C"][30] - base["C"][30]
        for slow, base in zip(
            scenarios["slow"].runs(), scenarios["base"].runs(), strict=True
        )
    ]
    median, lower, upper = weighted_quantiles(
        differences, [0.5, 0.1, 0.9], weights=weights
    )
    assert compared.label == "C at 30"
    assert compared.mean == pytest.approx(np.average(differences, weights=weights))
    assert (compared.median, compared.lower, compared.upper) == pytest.approx(
        (median, lower, upper)
    )
    slow_values = [run["C"][30] for run in scenarios["slow"].runs()]
    assert summary.median == pytest.approx(
        weighted_quantiles(slow_values, [0.5], weights=weights)[0]
    )


def test_a_single_draw_gives_a_value_and_no_interval() -> None:
    scenarios = Scenarios(_calibrator(), _sample([0.3]), SCENARIOS, POSITIONS)
    quantity = [ValueAt(series="C", at=30)]

    [compared] = scenarios.compare("slow", "base", quantity, 0.9)
    [summary] = scenarios.summary("slow", quantity, 0.9)

    difference = _run(0.3, 0.02)["C"][30] - _run(0.3, 0.1)["C"][30]
    assert (compared.lower, compared.upper) == (None, None)
    assert compared.mean == compared.median == pytest.approx(difference)
    assert summary.median == pytest.approx(_run(0.3, 0.02)["C"][30])
    assert (summary.lower, summary.upper) == (None, None)


def test_a_trajectory_is_compared_position_by_position() -> None:
    sample = _sample([0.2, 0.25, 0.3])
    scenarios = Scenarios(_calibrator(), sample, SCENARIOS, POSITIONS)
    trajectory = Trajectory(series="C", positions=[10, 20, 30])

    compared = scenarios.compare("slow", "base", [trajectory], 0.5)

    assert [s.label for s in compared] == quantity_labels([trajectory])
    for summary, at in zip(compared, [10, 20, 30], strict=True):
        differences = [
            _run(k1, 0.02)["C"][at] - _run(k1, 0.1)["C"][at] for k1 in [0.2, 0.25, 0.3]
        ]
        expected = weighted_quantiles(differences, [0.5, 0.25, 0.75])
        assert (summary.median, summary.lower, summary.upper) == pytest.approx(expected)


def test_an_analysis_runs_its_sample_under_scenarios() -> None:
    model = build(abc_builder(parameters={"k1": 0.3, "k2": 0.1}))
    analysis = SensitivityAnalysis(
        Simulation(model), InputSpace.uniform_on_bounds([bounded("k1", 0.0, 0.5)])
    )

    scenarios = Scenarios(analysis, _sample([0.2]), SCENARIOS, POSITIONS)

    assert scenarios["slow"].runs()[0]["C"] == pytest.approx(_run(0.2, 0.02)["C"])
    assert scenarios.weights == [1.0]


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda s: s.compare("fast", "base", [ValueAt(series="C", at=30)], 0.9),
            KeyError,
            r"No scenario 'fast'. Scenarios: \['base', 'slow'\]",
        ),
        (
            lambda s: s.summary("base", [ValueAt(series="C", at=40)], 0.9),
            ValueError,
            "give those positions to the Predictive or the Scenarios",
        ),
        (
            lambda s: s.summary("base", [ValueAt(series="C", at=30, label="nan")], 0.9),
            ValueError,
            "Quantity 'nan', in the runs counted from 0",
        ),
    ],
    ids=["unknown scenario", "missing position", "NaN"],
)
def test_a_comparison_that_cannot_be_read_raises(
    call: Callable[[Scenarios], object],
    error: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = Scenarios(_calibrator(), _sample([0.2, 0.3]), SCENARIOS, POSITIONS)
    if message.startswith("Quantity"):
        predictive = scenarios["base"]
        monkeypatch.setattr(predictive, "quantities", lambda _: [{"nan": math.nan}] * 2)

    with pytest.raises(error, match=message):
        call(scenarios)


def test_scenarios_need_at_least_one() -> None:
    with pytest.raises(ValueError, match="Give at least one scenario"):
        Scenarios(_calibrator(), _sample([0.2]), {}, POSITIONS)


def test_a_fit_is_compared_as_a_sample_of_one_draw() -> None:
    truth = Simulation(build(abc_builder(parameters={"k1": 0.3, "k2": 0.1}))).run(30)
    problem = CalibrationProblem(
        observed_data=observed(truth, "B", range(5, 31, 5)),
        parameters=[bounded("k1", 0.0, 0.5)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=200, sd_tolerance=1e-12),
        seed=1,
    )
    model = build(abc_builder(parameters={"k1": None, "k2": 0.1}))
    calibrator = Calibrator(Simulation(model), problem)
    result = calibrator.run()

    sample = calibrator.point_sample(result)
    [compared] = Scenarios(calibrator, sample, SCENARIOS, POSITIONS).compare(
        "slow", "base", [ValueAt(series="C", at=30)], 0.9
    )

    fit = result.best_parameters["k1"]
    assert (sample.names, sample.bounds, sample.values) == (
        ["k1"],
        [(0.0, 0.5)],
        [[fit]],
    )
    assert sample.method == SampleMethod(name="point", settings={"seed": result.seed})
    difference = _run(fit, 0.02)["C"][30] - _run(fit, 0.1)["C"][30]
    assert compared.median == pytest.approx(difference)
    assert (compared.lower, compared.upper) == (None, None)


def test_a_point_sample_needs_the_names_of_the_problem() -> None:
    result = CalibrationResult(
        best_parameters={"k2": 0.1},
        final_loss=0.0,
        iterations=0,
        converged=True,
        termination_reason="test",
        seed=0,
    )

    with pytest.raises(ValueError, match=r"it lacks \['k1'\] and has \['k2'\]"):
        _calibrator().point_sample(result)
