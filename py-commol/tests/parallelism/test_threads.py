"""The threads of the parallel methods: the setting and its levels, the threads
every parallel method runs on, the particles of a swarm in parallel, and one
result for one seed whatever the threads and the level."""

from collections.abc import Callable

import pytest

from commol import parallelism
from commol.resampling import BayesianBootstrap, BlockJackknife
from commol import Calibrator, Simulation
from commol.smc import GibbsPosterior
from commol.sensitivity import InputSpace, SensitivityAnalysis, Sobol
from commol.uncertainty import Predictive
from commol.distributions import Priors
from commol.profiles import ProfileLikelihood, Threshold
from commol.quantities import ValueAt
from commol.parallelism.threads import Levels
from tests.support import (
    abc_swarm_calibrator,
    assert_on_threads_of_setting,
    bounded,
    decay_swarm_calibrator,
    swarm,
    threads_of_runs,
    two_decays,
)
from tests.support.kinds import TEST_STOCHASTIC_ENGINE

LEVELS: tuple[Levels, ...] = ("all", "outer")


def test_a_thread_limit_below_one_is_rejected() -> None:
    for value in (0, -2):
        with pytest.raises(ValueError, match=f"at least 1, got {value}"):
            parallelism.set_threads(value)
        with pytest.raises(ValueError, match=f"at least 1, got {value}"):
            with parallelism.threads(value):
                pass
    with pytest.raises(
        ValueError, match="unknown variant `inner`, expected `all` or `outer`"
    ):
        parallelism.set_threads(2, levels="inner")  # type: ignore[arg-type]


def test_a_setting_holds_in_its_block_and_its_context() -> None:
    cores_of_machine = parallelism.get_threads()
    with parallelism.threads(3, levels="outer"):
        assert (parallelism.get_threads(), parallelism.get_levels()) == (3, "outer")
        with parallelism.threads(levels="all"):
            assert (parallelism.get_threads(), parallelism.get_levels()) == (3, "all")
        parallelism.set_threads(5)
        assert (parallelism.get_threads(), parallelism.get_levels()) == (5, "outer")
    assert (parallelism.get_threads(), parallelism.get_levels()) == (
        cores_of_machine,
        "all",
    )


def _bootstrap_predictive(calibrator: Calibrator) -> object:
    result = calibrator.uncertainty(BayesianBootstrap(replicates=4, block_length=1))
    return Predictive(calibrator, result.sample, list(range(21)))


def _sobol(_: Calibrator) -> object:
    simulation = Simulation(two_decays(k1=0.3, k2=0.1))
    inputs = InputSpace.uniform_on_bounds(
        [bounded("k1", 0.2, 0.6), bounded("k2", 0.05, 0.2)]
    )
    analysis = SensitivityAnalysis(simulation, inputs, seed=7)
    return analysis.run(
        Sobol(
            draws=64,
            replicates=10,
            confidence_level=0.9,
            quantities=[ValueAt(series=["x", "y"], at=5.0)],
        )
    )


def _replicates(_: Calibrator) -> object:
    simulation = Simulation(two_decays(k1=0.3, k2=0.1, engine=TEST_STOCHASTIC_ENGINE))
    return simulation.replicates(32, seed=3).run(10)


METHODS: dict[str, Callable[[Calibrator], object]] = {
    "starts": lambda calibrator: calibrator.run(starts=8),
    "bootstrap and predictive": _bootstrap_predictive,
    "jackknife": lambda calibrator: calibrator.uncertainty(
        BlockJackknife(block_length=2)
    ),
    "gibbs": lambda calibrator: calibrator.uncertainty(
        GibbsPosterior(
            priors=Priors.uniform_on_bounds(),
            eta=1.0,
            particles=32,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )
    ),
    "profile": lambda calibrator: calibrator.uncertainty(
        ProfileLikelihood(
            parameters=["k1", "k2"], threshold=Threshold.beale(0.95), max_steps=4
        )
    ),
    "sobol": _sobol,
    "replicates": _replicates,
}
"""Each parallel method of commol, run on the calibrator of `two_decays`."""


@pytest.mark.parametrize("method", METHODS.values(), ids=METHODS.keys())
@pytest.mark.parametrize("levels", LEVELS)
@pytest.mark.parametrize("limit", [1, 2])
def test_every_parallel_method_runs_on_the_threads_of_the_setting(
    method: Callable[[Calibrator], object], levels: Levels, limit: int
) -> None:
    calibrator = decay_swarm_calibrator(swarm(8, 10))

    with parallelism.threads(limit, levels=levels):
        threads = threads_of_runs(lambda: method(calibrator))

    # A parallel loop outside the pool of the setting runs on the threads
    # of the global rayon pool, which are no "commol-" threads.
    assert_on_threads_of_setting(threads, limit)


def test_a_parallel_swarm_keeps_every_evaluation() -> None:
    calibrator = abc_swarm_calibrator(swarm(10, 8, mutation=False))

    with parallelism.threads(4, levels="all"):
        history = calibrator.run_with_history()

    # Every point is feasible: 10 particles at the start and in each of 8
    # iterations.
    assert len(history.evaluations) == 10 * 9
    assert min(e.loss for e in history.evaluations) == history.result.final_loss


def test_a_parallel_swarm_records_in_the_order_of_the_particles() -> None:
    calibrator = abc_swarm_calibrator(swarm())

    with parallelism.threads(4, levels="outer"):
        sequential = calibrator.run_with_history()
    with parallelism.threads(4, levels="all"):
        parallel = calibrator.run_with_history()

    assert parallel.evaluations == sequential.evaluations
    assert parallel.result == sequential.result


def test_a_seed_gives_one_result_whatever_the_threads_and_the_level() -> None:
    calibrator = abc_swarm_calibrator(swarm(8, 6))
    bootstrap = BayesianBootstrap(replicates=4, block_length=1)
    gibbs = GibbsPosterior(
        priors=Priors.uniform_on_bounds(),
        eta=1.0,
        particles=16,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.05,
    )

    def results() -> tuple[object, ...]:
        return (
            calibrator.run(),
            calibrator.run(starts=8),
            calibrator.run_with_history(),
            calibrator.uncertainty(bootstrap),
            calibrator.uncertainty(gibbs),
        )

    with parallelism.threads(1, levels="outer"):
        first = results()
    for threads in (1, 2, 3, 8):
        for levels in LEVELS:
            with parallelism.threads(threads, levels=levels):
                assert results() == first, (threads, levels)
