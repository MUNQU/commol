"""The level of parallelism of the uncertainty methods, and the threads of
the setting that every parallel method runs on."""

from collections.abc import Callable

import pytest

import commol
from commol import (
    BayesianBootstrap,
    BlockJackknife,
    Calibrator,
    GibbsPosterior,
    InputSpace,
    Predictive,
    Priors,
    ProfileLikelihood,
    SensitivityAnalysis,
    Simulation,
    Sobol,
    Threshold,
    ValueAt,
)
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


def test_the_level_of_parallelism_does_not_change_a_bootstrap() -> None:
    calibrator = abc_swarm_calibrator(swarm(10, 10))
    bootstrap = BayesianBootstrap(replicates=6, block_length=1)

    with commol.threads(4, levels="outer"):
        replicates_in_parallel = calibrator.uncertainty(bootstrap)
    with commol.threads(4, levels="all"):
        particles_in_parallel = calibrator.uncertainty(bootstrap)

    assert particles_in_parallel == replicates_in_parallel


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


@pytest.mark.parametrize("method", METHODS.values(), ids=METHODS.keys())
@pytest.mark.parametrize("limit", [1, 2])
def test_every_parallel_method_runs_on_the_threads_of_the_setting(
    method: Callable[[Calibrator], object], limit: int
) -> None:
    calibrator = decay_swarm_calibrator(swarm(8, 10))

    with commol.threads(limit):
        threads = threads_of_runs(lambda: method(calibrator))

    # A parallel loop outside the pool of the setting runs on the threads
    # of the global rayon pool, which are no "commol-" threads.
    assert_on_threads_of_setting(threads, limit)
