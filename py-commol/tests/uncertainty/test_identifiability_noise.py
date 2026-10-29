"""Identifiability with a noise parameter of an observation model:
`StructuralRank` and `Collinearity` leave it out. The report that merges
them is in ``test_identifiability_report.py``."""

import pytest

from commol._fit.result import CalibrationResult
from commol.identifiability import Collinearity, StructuralRank
from tests.support import overdispersed_calibrator


def test_a_noise_parameter_has_no_structural_verdict() -> None:
    report = overdispersed_calibrator().identifiability(
        StructuralRank(points=5, design="observations")
    )

    section = report.structural[0]
    assert section.generic_rank == 2
    assert section.verdicts == {"k1": "identifiable", "k2": "identifiable"}
    assert list(section.projections) == ["k1", "k2"]
    assert all(len(point.singular_values) == 2 for point in section.points)
    assert list(section.points[0].values) == ["k1", "k2", "phi"]
    assert section.left_out == {
        "phi": "a noise parameter of an observation model, which no prediction reads"
    }
    k1 = report.verdicts["k1"].structural
    assert k1 is not None
    assert k1.verdict == "identifiable"
    assert "(generic rank 2 of 2)" in k1.reasons[0]
    phi = report.verdicts["phi"].structural
    assert phi is not None
    assert (phi.verdict, phi.reasons) == (
        None,
        [
            "Structural rank on the observations: left out as a noise parameter "
            "of an observation model, which no prediction reads."
        ],
    )


def test_a_rank_with_only_noise_parameters_is_an_error() -> None:
    calibrator = overdispersed_calibrator(rates=())

    with pytest.raises(
        ValueError, match="StructuralRank has no parameter to differentiate"
    ):
        calibrator.identifiability(StructuralRank(points=2, design="observations"))


def test_a_noise_parameter_is_not_in_the_collinearity() -> None:
    calibrator = overdispersed_calibrator()
    result = calibrator.run(starts=6)

    report = calibrator.identifiability(
        Collinearity(at=result, max_subset_size=2, threshold=10)
    )

    section = report.collinearity[0]
    assert section.pair_gamma["k1"]["k2"] == pytest.approx(1.11, abs=0.005)
    assert list(section.pair_gamma) == ["k1", "k2"]
    assert list(section.pair_gamma["k1"]) == ["k1", "k2"]
    assert list(section.sensitivity) == ["k1", "k2"]
    assert list(section.left_out) == ["phi"]
    assert [subset.parameters for subset in section.admissible] == [["k1", "k2"]]
    verdicts = {name: v.practical for name, v in report.verdicts.items()}
    assert verdicts["k1"] is not None
    assert verdicts["k1"].verdict == "determined"
    assert verdicts["k2"] is not None
    assert verdicts["k2"].verdict == "determined"
    assert verdicts["phi"] is None


def test_a_subset_larger_than_the_parameters_that_are_not_noise_is_an_error() -> None:
    calibrator = overdispersed_calibrator()
    at = CalibrationResult(
        best_parameters={"k1": 0.4, "k2": 0.1, "phi": 0.1},
        final_loss=0.0,
        iterations=0,
        converged=True,
        termination_reason="given",
        seed=0,
    )

    with pytest.raises(ValueError) as error:
        calibrator.identifiability(Collinearity(at=at, max_subset_size=3, threshold=10))

    assert str(error.value) == (
        "max_subset_size is a whole number from 2 to 2, got 3: 2 of the 3 "
        "parameters are not noise parameters"
    )


def test_collinearity_needs_two_parameters_that_are_not_noise_parameters() -> None:
    calibrator = overdispersed_calibrator(rates=("k1",))
    at = CalibrationResult(
        best_parameters={"k1": 0.4, "phi": 0.1},
        final_loss=0.0,
        iterations=0,
        converged=True,
        termination_reason="given",
        seed=0,
    )

    with pytest.raises(ValueError) as error:
        calibrator.identifiability(Collinearity(at=at, max_subset_size=2, threshold=10))

    assert str(error.value) == (
        "Collinearity needs at least 2 parameters that are not noise parameters, got 1"
    )
