"""One entry point: ``Calibrator.uncertainty`` runs the runner of a
configuration's type, and its result type follows that type."""

from typing import assert_type

import pytest

from commol import Calibrator, StructuralRank, UncertaintyMethod
from commol.api.inference.registry import register, registered
from tests.support.methods import StartsResult, StartsSample, run_starts_sample


class Unregistered(UncertaintyMethod[StartsResult]):
    """A method with no runner."""


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

    def test_each_entry_point_refuses_the_other_kind(
        self, calibrator: Calibrator
    ) -> None:
        with pytest.raises(TypeError, match=r"Calibrator\.uncertainty"):
            calibrator.identifiability(StartsSample(starts=2))  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match=r"Calibrator\.identifiability"):
            calibrator.uncertainty(
                StructuralRank(points=2, design="observations")  # ty: ignore[invalid-argument-type]
            )
