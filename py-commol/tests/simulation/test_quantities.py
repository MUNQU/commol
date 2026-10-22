"""The quantities of interest of one run and of replicates, read on the
results with the code of a propagation."""

import pytest

from commol import (
    Gain,
    Increments,
    InputSpace,
    ParameterSample,
    Peak,
    PeakPosition,
    Predictive,
    Propagation,
    SampleMethod,
    SensitivityAnalysis,
    Simulation,
    SimulationResults,
    SimulationReplicates,
    Trajectory,
    ValueAt,
    quantity_schedule,
)
from commol.context.inference.quantities import Quantity
from tests.support import abc_builder, bounded, build, build_ode
from tests.support.models import build_stochastic

QUANTITIES: list[Quantity] = [
    ValueAt(series="B", at=3),
    Gain(outputs="C", start=2, end=9),
    Peak(series=Increments("C", window=3), positions=[3, 6, 9]),
    PeakPosition(series=Increments("C", window=3), positions=[3, 6, 9]),
    Trajectory(series=["A", "B"], positions=[4, 8]),
]


def _one_draw(simulation: Simulation, **analysis: int) -> SensitivityAnalysis:
    """An analysis of a sample of one draw: ``k1 = 0.3``, ``k2 = 0.1``."""
    sample = ParameterSample.equally_weighted(
        ["k1", "k2"], [(0.1, 0.5), (0.0, 1.0)], [[0.3, 0.1]], SampleMethod(name="test")
    )
    inputs = InputSpace.from_sample(sample, [bounded("k1", 0.1, 0.5), bounded("k2")])
    return SensitivityAnalysis(simulation, inputs, **analysis)


@pytest.mark.parametrize("engine", ["DifferenceEquations", "DifferentialEquations"])
def test_the_quantities_of_one_run_equal_those_of_a_propagation(engine: str) -> None:
    builder = abc_builder()
    model = build(builder) if engine == "DifferenceEquations" else build_ode(builder)
    simulation = Simulation(model)

    quantities = simulation.run_at(quantity_schedule(QUANTITIES)).quantities(QUANTITIES)
    propagation = _one_draw(simulation, seed=1).run(Propagation(quantities=QUANTITIES))

    assert list(quantities) == propagation.labels
    assert list(quantities.values()) == propagation.values[0]
    # A run on its default grid holds the schedule too. The ODE engine
    # interpolates the positions of `run` that are not breakpoints, within
    # its relative tolerance.
    on_grid = simulation.run(10).quantities(QUANTITIES)
    assert on_grid == pytest.approx(quantities, rel=1e-8)


def test_positions_are_found_within_the_tolerance_of_the_axis() -> None:
    simulation = Simulation(build_ode(abc_builder()))
    late = ValueAt(series=Increments("C", window=0.1), at=0.3)

    schedule = quantity_schedule([late])
    results = simulation.run_at([0.0, 0.2, 0.3])

    # The window starts at 0.3 - 0.1, which is not 0.2 in floating point.
    assert schedule == [0.3 - 0.1, 0.3]
    assert results.quantities([late]) == {
        late.labels()[0]: results["C"][2] - results["C"][1]
    }


def test_missing_positions_are_listed() -> None:
    results = Simulation(build(abc_builder())).run_at([0, 3, 5, 9])

    with pytest.raises(ValueError) as error:
        results.quantities(QUANTITIES)

    message = str(error.value)
    assert "'C gain over (2, 9]' reads 2;" in message
    assert "'peak of C gain over windows of 3' reads 6;" in message
    assert "'A + B' reads 4, 8." in message
    assert "'B at 3'" not in message
    assert "quantity_schedule(quantities) with Simulation.run_at" in message


def test_an_unknown_output_is_rejected() -> None:
    results = Simulation(build(abc_builder())).run(10)

    with pytest.raises(ValueError, match=r"'D'.*available: A, B, C"):
        results.quantities([ValueAt(series="D", at=3)])
    with pytest.raises(ValueError, match="at least one quantity"):
        results.quantities([])
    with pytest.raises(ValueError, match="at least one quantity"):
        quantity_schedule([])
    hand_made = SimulationResults([0.0, 1.0], {"A": [1.0, 2.0]})
    assert hand_made.quantities([Gain(outputs="A", start=0, end=1)]) == {
        "A gain over (0, 1]": 1.0
    }


def test_the_quantities_of_replicates_are_one_dict_per_replicate() -> None:
    simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
    analysis = _one_draw(simulation, replicates=3, seed=5)
    propagation = analysis.run(Propagation(quantities=QUANTITIES))

    # A propagation runs the replicates of the seed of each draw, not of its
    # own seed.
    replicates = simulation.replicates(3, seed=propagation.draw_seeds[0]).run(10)
    quantities = replicates.quantities(QUANTITIES)

    assert [list(row.values()) for row in quantities] == propagation.values
    assert quantities[0] == replicates[0].quantities(QUANTITIES)
    assert len({row["C gain over (2, 9]"] for row in quantities}) > 1
    with pytest.raises(ValueError, match="There are no replicates"):
        SimulationReplicates([], seed=1, positions=[1.0]).quantities(QUANTITIES)


def test_the_replicates_of_a_draw_are_those_of_its_seed() -> None:
    values = [[0.3, 0.1], [0.45, 0.2]]
    sample = ParameterSample.equally_weighted(
        ["k1", "k2"], [(0.1, 0.5), (0.0, 1.0)], values, SampleMethod(name="test")
    )
    inputs = InputSpace.from_sample(sample, [bounded("k1", 0.1, 0.5), bounded("k2")])
    simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
    analysis = SensitivityAnalysis(simulation, inputs, replicates=3, seed=7)

    result = analysis.run(Propagation(quantities=QUANTITIES))
    predictive = Predictive(analysis, result.sample, [4.0, 8.0])

    assert len(result.draw_seeds) == 2
    assert predictive.draw_seeds == result.draw_seeds
    runs = predictive.runs()
    for draw, (k1, k2) in enumerate(values):
        model = build_stochastic(
            abc_builder(parameters={"k1": k1, "k2": k2}), "StochasticSteps"
        )
        replicates = Simulation(model).replicates(3, seed=result.draw_seeds[draw])
        quantities = replicates.run(10).quantities(QUANTITIES)
        assert [list(row.values()) for row in quantities] == result.values[
            3 * draw : 3 * draw + 3
        ]
        at = replicates.run_at([4.0, 8.0])
        for replicate in range(3):
            assert runs[3 * draw + replicate]["A"] == at[replicate]["A"]
