"""One entry point: ``Calibrator.uncertainty`` runs the runner of a
configuration's type, and its result type follows that type."""

from collections.abc import Callable
from typing import assert_type

import pytest
from pydantic import ValidationError

from commol import (
    Calibrator,
    InputSpace,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Ratio,
    SensitivityAnalysis,
    SensitivityMethod,
    Sobol,
    SplitConformal,
    StructuralRank,
    Threshold,
    UncertaintyMethod,
    ValueAt,
)
from commol.api.inference.registry import register, registered
from tests.support.methods import StartsResult, StartsSample, run_starts_sample


class Unregistered(UncertaintyMethod[StartsResult]):
    """A method with no runner."""


class UnregisteredSensitivity(SensitivityMethod[StartsResult]):
    """A sensitivity method with no runner."""


class TestEntryPoint:
    def test_the_runner_of_the_configuration_type_runs(
        self, calibrator: Calibrator
    ) -> None:
        result = calibrator.uncertainty(StartsSample(starts=3))

        assert_type(result, StartsResult)
        assert isinstance(result, StartsResult)
        assert len(result.sample) == 3
        assert result.sample.method.settings == {"starts": 3}
        assert result.sample.names == ["k1", "k2"]

    def test_an_unregistered_configuration_names_the_registered_ones(
        self, calibrator: Calibrator
    ) -> None:
        with pytest.raises(ValueError, match=r"Unregistered.*StartsSample"):
            calibrator.uncertainty(Unregistered())

    def test_a_type_has_one_runner(self) -> None:
        with pytest.raises(ValueError, match="already has a runner"):
            register(StartsSample, run_starts_sample)

    def test_the_registry_lists_its_types(self) -> None:
        assert "StartsSample" in registered()

    def test_each_entry_point_refuses_the_other_kinds(
        self, calibrator: Calibrator
    ) -> None:
        analysis = SensitivityAnalysis(
            calibrator.simulation,
            InputSpace.uniform_on_bounds(calibrator.problem.parameters),
            seed=1,
        )
        with pytest.raises(TypeError, match=r"Calibrator\.uncertainty"):
            calibrator.identifiability(StartsSample(starts=2))  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match=r"Calibrator\.identifiability"):
            calibrator.uncertainty(
                StructuralRank(points=2, design="observations")  # ty: ignore[invalid-argument-type]
            )
        with pytest.raises(TypeError, match=r"SensitivityAnalysis\.run"):
            calibrator.uncertainty(UnregisteredSensitivity())  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match=r"Calibrator\.uncertainty"):
            analysis.run(StartsSample(starts=2))  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match=r"No method is registered"):
            analysis.run(UnregisteredSensitivity())


VALUE = ValueAt(series="B", at=5)


@pytest.mark.parametrize(
    ("build", "old"),
    [
        (
            lambda: SplitConformal(
                calibration_data=[ObservedDataPoint(at=1, output="B", value=1.0)],
                positions=[2.0],
                score="absolute",
                by="all",
                level=0.9,
            ),
            "level",
        ),
        (lambda: Threshold(kind="beale", level=0.9), "level"),
        (lambda: ParticleSwarmConfig(num_particles=5), "num_particles"),
        (
            lambda: Sobol(
                base_draws=8, replicates=2, confidence_level=0.9, quantities=[VALUE]
            ),
            "base_draws",
        ),
        (
            lambda: Sobol(
                draws=8, bootstrap=2, confidence_level=0.9, quantities=[VALUE]
            ),
            "bootstrap",
        ),
        (
            lambda: Ratio(numerator=VALUE, denominator=VALUE, replicates="within"),
            "replicates",
        ),
    ],
    ids=["conformal", "threshold", "swarm", "sobol draws", "sobol replicates", "ratio"],
)
def test_an_old_setting_name_is_rejected(build: Callable[[], object], old: str) -> None:
    with pytest.raises(
        ValidationError, match=rf"\n{old}\n  Extra inputs are not permitted"
    ):
        build()
