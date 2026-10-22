"""The threads of the parallel methods: their limit, the particles of a swarm
in parallel, and one result for one seed whatever the threads and the level."""

import pytest

import commol
from commol import BayesianBootstrap, GibbsPosterior, Priors
from commol.api.parallelism import Levels
from tests.support import abc_swarm_calibrator, cores, swarm

LEVELS: tuple[Levels, ...] = ("all", "outer")


def test_a_thread_limit_bounds_every_parallel_method() -> None:
    calibrator = abc_swarm_calibrator(swarm(20, 40))
    bootstrap = BayesianBootstrap(replicates=8, block_length=1)

    with commol.threads(2):
        starts = cores(lambda: calibrator.run(starts=8))
        replicates = cores(lambda: calibrator.uncertainty(bootstrap))

    # The threads of a pool spin a little after each job while Python goes
    # on, hence the margin; without the limit, the runs keep every core busy.
    assert starts <= 2.5
    assert replicates <= 2.5


def test_a_thread_limit_below_one_is_rejected() -> None:
    for value in (0, -2):
        with pytest.raises(ValueError, match=f"at least 1, got {value}"):
            commol.set_threads(value)
        with pytest.raises(ValueError, match=f"at least 1, got {value}"):
            with commol.threads(value):
                pass
    with pytest.raises(ValueError, match="'inner'; the levels are 'all' and 'outer'"):
        commol.set_threads(2, levels="inner")  # type: ignore[arg-type]


def test_a_setting_holds_in_its_block_and_its_context() -> None:
    cores_of_machine = commol.get_threads()
    with commol.threads(3, levels="outer"):
        assert (commol.get_threads(), commol.get_levels()) == (3, "outer")
        with commol.threads(levels="all"):
            assert (commol.get_threads(), commol.get_levels()) == (3, "all")
        commol.set_threads(5)
        assert (commol.get_threads(), commol.get_levels()) == (5, "outer")
    assert (commol.get_threads(), commol.get_levels()) == (cores_of_machine, "all")


def test_a_parallel_swarm_keeps_every_evaluation() -> None:
    calibrator = abc_swarm_calibrator(swarm(10, 8, mutation=False))

    with commol.threads(4, levels="all"):
        history = calibrator.run_with_history()

    # Every point is feasible: 10 particles at the start and in each of 8
    # iterations.
    assert len(history.evaluations) == 10 * 9
    assert min(e.loss for e in history.evaluations) == history.result.final_loss


def test_a_parallel_swarm_records_in_the_order_of_the_particles() -> None:
    calibrator = abc_swarm_calibrator(swarm())

    with commol.threads(4, levels="outer"):
        sequential = calibrator.run_with_history()
    with commol.threads(4, levels="all"):
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

    with commol.threads(1, levels="outer"):
        first = results()
    for threads in (1, 2, 3, 8):
        for levels in LEVELS:
            with commol.threads(threads, levels=levels):
                assert results() == first, (threads, levels)
