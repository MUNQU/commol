"""The structural rank of an engine that integrates with a tolerance: a rank
lost to the tolerance gives no structural verdict."""

import pytest

from commol.engines import ODESolver
from commol.identifiability import StructuralRank
from commol.identifiability.results import Accuracy
from tests.support import flow_calibrator, rare_flow_calibrator


class TestSolverTolerance:
    def test_a_rank_lost_to_the_solver_tolerance_gives_no_structural_verdict(
        self,
    ) -> None:
        loose = rare_flow_calibrator(ODESolver(rtol=1e-6, atol=1e-6))

        report = loose.identifiability(StructuralRank(points=3, design="observations"))

        section = report.structural[0]
        assert [point.rank for point in section.points] == [1, 3, 3]
        lost = section.points[0].tighter
        assert lost is not None
        assert lost.rank == 3
        assert all(point.tighter is None for point in section.points[1:])
        assert list(section.verdicts.values()) == ["identifiable"] * 3

    def test_a_rank_lost_at_most_points_leaves_the_verdicts_out(self) -> None:
        loose = rare_flow_calibrator(ODESolver(rtol=1e-6, atol=1e-6))

        report = loose.identifiability(StructuralRank(points=1, design="observations"))

        section = report.structural[0]
        assert section.generic_rank == 2
        assert section.combinations == []
        for name in ("k1", "k2", "p"):
            structural = report.verdicts[name].structural
            assert structural is not None
            assert structural.verdict is None
            assert structural.reasons == [
                "Structural rank on the observations: the accuracy of the engine "
                "(ODESolver.rtol = 1e-06, ODESolver.atol = 1e-06), not the "
                "structure, may set its verdict. Evaluated again at "
                "ODESolver.rtol = 1e-08, ODESolver.atol = 1e-08, the rank of 1 "
                "of 1 points changes (2 to 3). Tighten ODESolver.rtol and "
                "ODESolver.atol until the rank stops changing."
            ]

    def test_an_unread_parameter_is_non_identifiable_at_any_tolerance(self) -> None:
        calibrator = rare_flow_calibrator(ODESolver(), unread=True)

        report = calibrator.identifiability(
            StructuralRank(points=3, design="observations")
        )

        section = report.structural[0]
        assert all(point.tighter is not None for point in section.points)
        structural = report.verdicts["u"].structural
        assert structural is not None
        assert structural.verdict == "non_identifiable"

    def test_a_verdict_that_stays_names_the_generic_rank_at_both_tolerances(
        self,
    ) -> None:
        calibrator = rare_flow_calibrator(ODESolver(), unread=True)

        report = calibrator.identifiability(
            StructuralRank(points=1, design="observations")
        )

        section = report.structural[0]
        assert (section.generic_rank, section.tighter_generic_rank) == (2, 3)
        structural = report.verdicts["u"].structural
        assert structural is not None
        assert structural.verdict == "non_identifiable"
        assert structural.reasons == [
            "Structural rank on the observations (generic rank 2 of 4; 3 at "
            "ODESolver.rtol = 1e-08, ODESolver.atol = 1e-08): its projection on "
            "the null space is 1.00."
        ]

    def test_a_point_evaluated_again_holds_both_ranks(self) -> None:
        loose = rare_flow_calibrator(ODESolver(rtol=1e-6, atol=1e-5))

        report = loose.identifiability(StructuralRank(points=1, design="observations"))

        section = report.structural[0]
        point = section.points[0]
        assert point.accuracy == Accuracy(relative=1e-6, absolute=1e-5)
        assert point.tighter is not None
        tighter = point.tighter.accuracy
        assert (tighter.relative, tighter.absolute) == pytest.approx((1e-8, 1e-7))
        assert (point.rank, point.tighter.rank) == (2, 3)
        assert point.tighter.threshold < point.threshold
        assert section.accuracy == point.accuracy
        assert section.tighter == point.tighter.accuracy
        assert section.unchecked == []

    def test_an_exact_engine_has_no_tighter_accuracy(self) -> None:
        calibrator = flow_calibrator([2.0, 2.0, 2.0], windowed=True)

        report = calibrator.identifiability(
            StructuralRank(points=2, design="observations")
        )

        section = report.structural[0]
        assert section.accuracy == Accuracy(relative=2.0**-52, absolute=0.0)
        assert section.tighter is None
        assert [point.tighter for point in section.points] == [None, None]
