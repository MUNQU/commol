"""The results of a run and of replicates compare by their values, positions,
seed and model, a NaN equal to a NaN; a plain dict or list is not a result."""

import pytest

from commol import (
    Simulation,
)
from commol.api.results import (
    SimulationReplicates,
    SimulationResults,
)
from tests.support import abc_builder, build
from tests.support.models import build_stochastic


def _run(
    values: list[float], positions: list[float], seed: int | None = None
) -> SimulationResults:
    return SimulationResults(positions, {"A": values}, seed=seed)


def _replicates(seed: int = 1) -> SimulationReplicates:
    simulation = Simulation(build_stochastic(abc_builder(), "StochasticSteps"))
    return simulation.replicates(3, seed=seed).run(5)


def test_two_runs_with_one_nan_are_equal() -> None:
    first = _run([1.0, float("nan")], [0.0, 1.0], seed=3)
    again = _run([1.0, float("nan")], [0.0, 1.0], seed=3)

    assert first == again
    assert not first != again


def test_runs_at_other_positions_are_not_equal() -> None:
    assert _run([1.0, 2.0], [0.0, 1.0]) != _run([1.0, 2.0], [0.0, 2.0])


def test_runs_with_other_seeds_are_not_equal() -> None:
    assert _run([1.0, 2.0], [0.0, 1.0], seed=1) != _run([1.0, 2.0], [0.0, 1.0], seed=2)


def test_runs_of_other_models_are_not_equal() -> None:
    simulation = Simulation(build(abc_builder()))
    other = build(abc_builder(parameters={"k1": 0.5, "k2": 0.1}))
    run = simulation.run(3)

    assert run == simulation.run(3)
    assert run != SimulationResults(run.positions, run, model=other)


def test_a_plain_dict_with_the_same_values_is_not_a_result() -> None:
    run = _run([1.0, 2.0], [0.0, 1.0])

    assert run != {"A": [1.0, 2.0]}
    assert {"A": [1.0, 2.0]} != run
    assert not run == {"A": [1.0, 2.0]}
    assert dict(run) == {"A": [1.0, 2.0]}


def test_two_replicate_runs_with_one_seed_are_equal() -> None:
    first = _replicates()

    assert first == _replicates()
    assert first != _replicates(seed=2)
    assert first != list(first)
    assert list(first) != first


def test_replicates_with_one_nan_are_equal() -> None:
    def replicates(seed: int) -> SimulationReplicates:
        runs = [_run([float("nan")], [0.0], seed=4), _run([1.0], [0.0], seed=5)]
        return SimulationReplicates(runs, seed=seed, positions=[0.0])

    assert replicates(1) == replicates(1)
    assert replicates(1) != replicates(2)


def test_replicates_compare_their_runs_and_their_positions() -> None:
    def replicates(value: float, positions: list[float]) -> SimulationReplicates:
        runs = [_run([value] * len(positions), positions, seed=4)]
        return SimulationReplicates(runs, seed=1, positions=positions)

    assert replicates(1.0, [0.0]) != replicates(2.0, [0.0])
    assert replicates(1.0, [0.0]) != replicates(1.0, [1.0])


def test_runs_and_replicates_have_no_hash() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        hash(_run([1.0], [0.0]))
    with pytest.raises(TypeError, match="unhashable"):
        hash(SimulationReplicates([], seed=1, positions=[0.0]))
