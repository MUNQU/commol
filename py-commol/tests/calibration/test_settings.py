"""A setting that the engine rejects gets the message of the engine, after
the path of its field, through a union tagged by ``kind`` too."""

import pytest

from commol import (
    BootstrapCoverageEta,
    Calibrator,
    GibbsPosterior,
    ParticleSwarmConfig,
    Priors,
)
from tests.support import line_calibrator

VALUES = [7.4, 8.6, 11.3, 12.8]


def test_an_invalid_setting_names_its_path() -> None:
    calibrator = line_calibrator(VALUES)
    swarm = ParticleSwarmConfig(inertia="constant", inertia_factor=0.5)
    swarm.inertia_factor = -1.0
    problem = calibrator.problem.model_copy(update={"optimization_config": swarm})

    with pytest.raises(ValueError) as error:
        Calibrator(calibrator.simulation, problem)

    assert str(error.value) == (
        "optimization_config.inertia_factor: inertia_factor is a number greater "
        "than 0, got -1"
    )


def test_an_invalid_setting_of_a_method_names_its_path() -> None:
    eta = BootstrapCoverageEta(replicates=3, block_length=1, confidence_level=0.9)
    method = GibbsPosterior(
        priors=Priors.uniform_on_bounds(),
        eta=eta,
        particles=10,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.5,
    ).model_copy(update={"eta": eta.model_copy(update={"replicates": -1})})

    with pytest.raises(ValueError) as error:
        line_calibrator(VALUES).uncertainty(method)

    assert str(error.value) == (
        "eta.replicates: replicates is a whole number of at least 1, got -1"
    )
