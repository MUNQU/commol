"""The threads that run a method, and a calibration of the test kind to read
them on."""

from collections.abc import Callable

from commol import (
    CalibrationProblem,
    Calibrator,
    Model,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.engine import ModelEngine
from commol.context.parameter import Parameter

from .kinds import TEST_ENGINE, DecayOutput, DecayStructure, decay, require_test_kind
from .problems import bounded

RunThread = tuple[str, int | None]
"""A thread that ran a model of the test kind: its name, and the number of
threads of its rayon pool, or None when it is no thread of a rayon pool."""


def threads_of_runs(job: Callable[[], object]) -> list[RunThread]:
    """The threads that run a model of the test kind while `job` runs, each
    once."""
    require_test_kind()
    commol_rs.test_kind.threads_of_runs()
    job()
    return commol_rs.test_kind.threads_of_runs()


def assert_on_threads_of_setting(threads: list[RunThread], limit: int) -> None:
    """
    Some thread of `threads` is a thread of a commol pool; every thread of a
    rayon pool in `threads` is a thread of a commol pool of `limit` threads,
    and they are at most `limit`. A thread of no pool, the caller, runs one
    model at a time.
    """
    pooled = [(name, pool) for name, pool in threads if pool is not None]
    assert pooled, threads
    assert all(name.startswith("commol-") and pool == limit for name, pool in pooled), (
        threads
    )
    assert len(pooled) <= limit, threads


def two_decays(
    k1: float | None = None, k2: float | None = None, engine: str = TEST_ENGINE
) -> Model[DecayStructure]:
    """A test model whose outputs ``x`` and ``y`` decay at ``k1`` and ``k2``."""
    require_test_kind()
    return Model[DecayStructure](
        name="two decays",
        parameters=[Parameter(id="k1", value=k1), Parameter(id="k2", value=k2)],
        engine=ModelEngine(type=engine),
        structure=DecayStructure(
            outputs=[
                DecayOutput(id="x", initial=10.0, rate="k1"),
                DecayOutput(id="y", initial=5.0, rate="k2"),
            ]
        ),
    )


def decay_swarm_calibrator(config: ParticleSwarmConfig) -> Calibrator:
    """
    Fit ``k1`` and ``k2`` of `two_decays` (in ``[0, 1]`` and ``[0, 0.5]``) to
    ``x`` and ``y`` at ``k1 = 0.3``, ``k2 = 0.1``, at the positions 1 to 10,
    with the swarm `config` and the seed 7.
    """
    observed = [
        ObservedDataPoint(at=at, output=output, value=decay(initial, rate, at))
        for output, initial, rate in (("x", 10.0, 0.3), ("y", 5.0, 0.1))
        for at in range(1, 11)
    ]
    problem = CalibrationProblem(
        observed_data=observed,
        parameters=[bounded("k1", 0.0, 1.0), bounded("k2", 0.0, 0.5)],
        loss_function="sse",
        optimization_config=config,
        seed=7,
    )
    return Calibrator(Simulation(two_decays()), problem)
