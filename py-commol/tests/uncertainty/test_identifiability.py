"""Identifiability: the structural rank, collinearity, a sample, the starts
and the report that joins them."""

from typing import assert_type

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    CalibrationResult,
    Calibrator,
    Collinearity,
    IdentifiabilityReport,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    SampleIdentifiability,
    SampleMethod,
    Simulation,
    StartsIdentifiability,
    StartSummary,
    StructuralRank,
)
from commol.context.inference.identifiability import (
    ParameterVerdicts,
    PracticalVerdict,
    PracticalVerdictName,
    StructuralVerdict,
    StructuralVerdictName,
)
from tests.support import (
    abc_builder,
    bounded,
    build,
    build_stochastic,
    observed,
)

TRUTH = {"k1": 0.3, "k2": 0.1}


def _calibrator(
    observations: list[ObservedDataPoint],
    parameters: dict[str, tuple[float, float]],
    *,
    ab: str = "k1 * A * B / total",
    factors: tuple[str, ...] = (),
) -> Calibrator:
    """``abc_builder`` with `parameters` left for calibration in their bounds."""
    model_parameters: dict[str, float | None] = {**TRUTH, "s": 1.0}
    for parameter_id in parameters:
        if parameter_id not in factors:
            model_parameters[parameter_id] = None
    model = build(abc_builder(ab=ab, parameters=model_parameters))
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[
            bounded(
                parameter_id,
                low,
                high,
                kind="factor" if parameter_id in factors else "parameter",
            )
            for parameter_id, (low, high) in parameters.items()
        ],
        optimization_config=NelderMeadConfig(max_iterations=200),
        seed=3,
    )
    return Calibrator(Simulation(model), problem)


@pytest.fixture
def truth() -> dict[str, list[float]]:
    """50 steps of ``abc_builder`` at the true parameters."""
    return Simulation(build(abc_builder(parameters={**TRUTH, "s": 1.0}))).run(50)


@pytest.fixture
def product(truth: dict[str, list[float]]) -> Calibrator:
    """``k1`` and ``s`` read only as ``k1 * s``, with ``k2``."""
    return _calibrator(
        observed(truth, "B", range(5, 51, 5)),
        {"k1": (0.1, 0.6), "s": (0.5, 2.0), "k2": (0.02, 0.3)},
        ab="k1 * s * A * B / total",
    )


class TestStructuralRank:
    def test_a_product_of_rate_and_factor_is_identifiable_in_combination(
        self, product: Calibrator
    ) -> None:
        report = product.identifiability(
            StructuralRank(points=5, design="observations")
        )

        assert report.verdicts["k1"].structural is not None
        assert report.verdicts["k1"].structural.verdict == "in_combination"
        assert report.verdicts["s"].structural is not None
        assert report.verdicts["s"].structural.verdict == "in_combination"
        assert report.verdicts["s"].structural.combinations == ["k1 · s"]
        assert report.verdicts["k2"].structural is not None
        assert report.verdicts["k2"].structural.verdict == "identifiable"
        assert report.structural[0].generic_rank == 2

    def test_sir_observed_in_every_compartment_is_identifiable(
        self, truth: dict[str, list[float]]
    ) -> None:
        calibrator = _calibrator(
            [
                point
                for output in "ABC"
                for point in observed(truth, output, range(10, 51, 10))
            ],
            {"k1": (0.1, 0.6), "k2": (0.02, 0.3)},
        )
        for design in ("observations", "dense"):
            report = calibrator.identifiability(StructuralRank(points=4, design=design))

            section = report.structural[0]
            assert section.design == design
            assert section.generic_rank == 2
            assert all(v == "identifiable" for v in section.verdicts.values())

    def test_a_sparse_design_has_a_lower_rank_than_the_dense_one(
        self, truth: dict[str, list[float]]
    ) -> None:
        calibrator = _calibrator(
            observed(truth, "B", [10, 20], factor_id="f"),
            {"k1": (0.1, 0.6), "k2": (0.02, 0.3), "f": (0.5, 2.0)},
            factors=("f",),
        )
        sparse = calibrator.identifiability(
            StructuralRank(points=4, design="observations")
        )
        dense = calibrator.identifiability(StructuralRank(points=4, design="dense"))

        assert sparse.structural[0].generic_rank == 2
        assert dense.structural[0].generic_rank == 3
        # Only the design of the observations speaks for these data.
        assert dense.verdicts["k1"].structural is None

    def test_structural_rank_rejects_invalid_settings(self) -> None:
        with pytest.raises(ValidationError, match="points"):
            StructuralRank(points=0, design="observations")
        with pytest.raises(ValidationError, match="design"):
            StructuralRank(points=3, design="other")  # ty: ignore[invalid-argument-type]


class TestCollinearity:
    def test_the_largest_admissible_subsets_avoid_the_collinear_pair(
        self, product: Calibrator
    ) -> None:
        at = CalibrationResult(
            best_parameters={"k1": 0.3, "s": 1.0, "k2": 0.1},
            final_loss=0.0,
            iterations=0,
            converged=True,
            termination_reason="given",
        )
        report = product.identifiability(
            Collinearity(at=at, max_subset_size=3, threshold=20)
        )

        section = report.collinearity[0]
        assert section.pair_gamma["k1"]["s"] == float("inf")
        assert sorted(sorted(s.parameters) for s in section.admissible) == [
            ["k1", "k2"],
            ["k2", "s"],
        ]
        assert report.verdicts["k1"].practical is not None
        assert report.verdicts["k1"].practical.verdict == "determined"

    def test_a_sample_summarizes_the_indices_of_its_draws(
        self, product: Calibrator
    ) -> None:
        sample = ParameterSample.equally_weighted(
            ["k1", "s", "k2"],
            [(0.1, 0.6), (0.5, 2.0), (0.02, 0.3)],
            [[0.3, 1.0, 0.1], [0.25, 1.2, 0.12], [0.35, 0.9, 0.08]],
            SampleMethod(name="given"),
        )
        report = product.identifiability(
            Collinearity(at=sample, max_subset_size=2, threshold=20)
        )

        section = report.collinearity[0]
        assert section.points == 3
        assert section.pair_gamma["k1"]["s"] == float("inf")

    def test_collinearity_rejects_invalid_settings(self, product: Calibrator) -> None:
        at = CalibrationResult(
            best_parameters={"k1": 0.3, "s": 1.0, "k2": 0.1},
            final_loss=0.0,
            iterations=0,
            converged=True,
            termination_reason="given",
        )
        with pytest.raises(ValidationError, match="threshold"):
            Collinearity(at=at, max_subset_size=2)  # ty: ignore[missing-argument]
        with pytest.raises(ValueError, match="max_subset_size"):
            product.identifiability(
                Collinearity(at=at, max_subset_size=4, threshold=20)
            )


def _result(ends: list[dict[str, float]], losses: list[float]) -> CalibrationResult:
    starts = [
        StartSummary(start=end, end=end, loss=loss, converged=True)
        for end, loss in zip(ends, losses, strict=True)
    ]
    best = min(starts, key=lambda start: start.loss)
    return CalibrationResult(
        best_parameters=best.end,
        final_loss=best.loss,
        iterations=1,
        converged=True,
        termination_reason="given",
        starts=starts,
    )


class TestPractical:
    def test_a_sample_reports_contraction_correlation_and_modes(
        self, product: Calibrator
    ) -> None:
        # k1 spreads over its bounds; s and k2 follow it on lines.
        k1 = [0.1 + 0.5 * (i + 0.5) / 100 for i in range(100)]
        sample = ParameterSample.equally_weighted(
            ["k1", "s", "k2"],
            [(0.1, 0.6), (0.5, 2.0), (0.02, 0.3)],
            [[k, 2.1 - 2.0 * k, 0.05 + 0.1 * k] for k in k1],
            SampleMethod(name="given"),
        )
        report = product.identifiability(
            SampleIdentifiability(
                sample=sample, reference="bounds", correlation_threshold=0.9
            )
        )

        section = report.sample[0]
        assert section.contraction["k1"] == pytest.approx(0.0, abs=0.05)
        assert section.correlation["k1"]["s"] == pytest.approx(-1.0)
        assert [(p.first, p.second) for p in section.correlated_pairs] == [
            ("k1", "s"),
            ("k1", "k2"),
            ("s", "k2"),
        ]
        verdict = report.verdicts["k1"].practical
        assert verdict is not None
        assert verdict.verdict == "poorly_determined"

    def test_starts_with_equal_losses_flag_the_parameters_that_differ(
        self, product: Calibrator
    ) -> None:
        apart = _result(
            [{"k1": 0.2, "s": 1.5, "k2": 0.1}, {"k1": 0.5, "s": 0.6, "k2": 0.1}],
            [1.0, 1.0],
        )
        together = _result(
            [{"k1": 0.3, "s": 1.0, "k2": 0.1}, {"k1": 0.3, "s": 1.0, "k2": 0.1}],
            [1.0, 1.0],
        )
        method = {"parameter_tolerance": 0.05, "loss_tolerance": 1e-3}

        flagged = product.identifiability(StartsIdentifiability(result=apart, **method))
        still = product.identifiability(
            StartsIdentifiability(result=together, **method)
        )

        assert flagged.starts[0].flagged == ["k1", "s"]
        assert still.starts[0].flagged == []

    def test_starts_identifiability_needs_two_starts(self, product: Calibrator) -> None:
        one = _result([{"k1": 0.3, "s": 1.0, "k2": 0.1}], [1.0])
        method = StartsIdentifiability(
            result=one, parameter_tolerance=0.05, loss_tolerance=1e-3
        )

        with pytest.raises(ValueError, match=r"Calibrator\.run\(starts=n\)"):
            product.identifiability(method)


def _report(
    structural: StructuralVerdictName | None = None,
    practical: PracticalVerdictName | None = None,
) -> IdentifiabilityReport:
    return IdentifiabilityReport(
        parameters=["k"],
        verdicts={
            "k": ParameterVerdicts(
                structural=None
                if structural is None
                else StructuralVerdict(verdict=structural, reasons=[structural]),
                practical=None
                if practical is None
                else PracticalVerdict(verdict=practical, reasons=[practical]),
            )
        },
    )


class TestReport:
    def test_merge_keeps_structural_and_practical_verdicts_apart(self) -> None:
        merged = _report(structural="in_combination").merge(
            _report(practical="poorly_determined")
        )

        verdicts = merged.verdicts["k"]
        assert verdicts.structural is not None
        assert verdicts.structural.verdict == "in_combination"
        assert verdicts.practical is not None
        assert verdicts.practical.verdict == "poorly_determined"

    def test_merge_of_disagreeing_verdicts_is_conflicting(self) -> None:
        merged = _report(structural="identifiable").merge(
            _report(structural="non_identifiable")
        )

        verdict = merged.verdicts["k"].structural
        assert verdict is not None
        assert verdict.verdict == "conflicting"
        assert verdict.reasons == ["identifiable", "non_identifiable"]


class TestEntryPoint:
    def test_the_entry_point_infers_the_result_type(self, product: Calibrator) -> None:
        report = product.identifiability(
            StructuralRank(points=2, design="observations")
        )

        assert_type(report, IdentifiabilityReport)
        assert isinstance(report, IdentifiabilityReport)

    def test_derivative_methods_refuse_a_stochastic_engine(self) -> None:
        model = build_stochastic(
            abc_builder(parameters={"k1": None, "k2": 0.1}), "StochasticSteps"
        )
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=10, output="B", value=50.0)],
            parameters=[bounded("k1", 0.1, 0.6)],
            optimization_config=NelderMeadConfig(),
            replicates=4,
            replicate_summary="mean",
            seed=2,
        )
        calibrator = Calibrator(Simulation(model), problem)

        with pytest.raises(
            ValueError, match="SampleIdentifiability.*StartsIdentifiability"
        ):
            calibrator.identifiability(StructuralRank(points=2, design="observations"))
