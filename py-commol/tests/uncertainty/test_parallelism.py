"""The level of parallelism of the uncertainty methods, and the one thread
that every parallel method keeps to when the setting asks for one."""

from collections.abc import Callable

import pytest

import commol
from commol import (
    BayesianBootstrap,
    BlockJackknife,
    Calibrator,
    GibbsPosterior,
    InputSpace,
    Peak,
    Predictive,
    Priors,
    ProfileLikelihood,
    SensitivityAnalysis,
    Simulation,
    Sobol,
    Threshold,
)
from tests.support import (
    abc_builder,
    abc_swarm_calibrator,
    bounded,
    build,
    build_stochastic,
    cores,
    swarm,
)


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
    return Predictive(calibrator, result.sample, list(range(101)))


def _sobol(_: Calibrator) -> object:
    simulation = Simulation(build(abc_builder()))
    inputs = InputSpace.uniform_on_bounds(
        [bounded("k1", 0.2, 0.6), bounded("k2", 0.05, 0.2)]
    )
    analysis = SensitivityAnalysis(simulation, inputs, seed=7)
    positions = [float(t) for t in range(101)]
    return analysis.run(
        Sobol(
            base_draws=256,
            bootstrap=20,
            confidence_level=0.9,
            quantities=[Peak(series="B", positions=positions)],
        )
    )


def _replicates(_: Calibrator) -> object:
    simulation = Simulation(build_stochastic(abc_builder()))
    return simulation.replicates(64, seed=3).run(100)


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
            particles=64,
            target_cess=0.5,
            resample_threshold=0.5,
            stay_probability=0.05,
        )
    ),
    "profile": lambda calibrator: calibrator.uncertainty(
        ProfileLikelihood(
            parameters=["k1", "k2"], threshold=Threshold.beale(0.95), max_steps=6
        )
    ),
    "sobol": _sobol,
    "replicates": _replicates,
}


@pytest.mark.parametrize("method", METHODS.values(), ids=METHODS.keys())
def test_one_thread_keeps_every_parallel_method_on_one_core(
    method: Callable[[Calibrator], object],
) -> None:
    calibrator = abc_swarm_calibrator(swarm(10, 20))

    with commol.threads(1):
        used = cores(lambda: method(calibrator))

    # A parallel loop outside the pool of the setting runs on every core.
    # The thread of the pool spins a little after each job while Python
    # goes on, hence the margin.
    assert used <= 1.25
