"""Identifiability: the structural rank, collinearity, a sample, the starts
and the report that joins them."""

import pytest
from pydantic import ValidationError

from commol import (
    Beta,
    CalibrationProblem,
    CalibrationResult,
    Calibrator,
    Collinearity,
    IdentifiabilityReport,
    Gamma,
    NelderMeadConfig,
    Normal,
    ObservedDataPoint,
    ParameterSample,
    Priors,
    SampleIdentifiability,
    SampleMethod,
    Simulation,
    StartsIdentifiability,
    StartSummary,
    StructuralRank,
)
from commol.context.inference.identifiability import (
    CollinearitySection,
    ParameterVerdicts,
    PracticalVerdict,
    PracticalVerdictName,
    SampleSection,
    StartsSection,
    StructuralSection,
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
        loss_function="sse",
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
    def test_a_product_of_two_parameters_is_identifiable_in_combination(
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

    def test_a_parameter_no_observation_reads_is_non_identifiable(
        self, truth: dict[str, list[float]]
    ) -> None:
        calibrator = _calibrator(
            observed(truth, "B", range(5, 51, 5)), {"k1": (0.1, 0.6), "s": (0.5, 2.0)}
        )

        report = calibrator.identifiability(
            StructuralRank(points=3, design="observations")
        )

        structural = report.verdicts["s"].structural
        assert structural is not None
        assert structural.verdict == "non_identifiable"

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

    def test_a_run_that_fails_stops_the_structural_rank(
        self, truth: dict[str, list[float]]
    ) -> None:
        # The rate is not a number after 7.5, before the observation at 10.
        calibrator = _calibrator(
            observed(truth, "B", [5, 10]),
            {"k1": (0.1, 0.6)},
            ab="k1 * A * B / total + 0 * sqrt(7.5 - t)",
        )

        with pytest.raises(
            RuntimeError, match="The rate of flow 'ab' at 8 is not a number"
        ):
            calibrator.identifiability(StructuralRank(points=2, design="observations"))

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
        verdicts = {name: v.practical for name, v in report.verdicts.items()}
        assert {name: v.verdict for name, v in verdicts.items() if v} == {
            "k1": "poorly_determined",
            "s": "poorly_determined",
            "k2": "determined",
        }

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

    def test_a_parameter_no_observation_reads_is_poorly_determined(
        self, truth: dict[str, list[float]]
    ) -> None:
        calibrator = _calibrator(
            observed(truth, "B", range(5, 51, 5)), {"k1": (0.1, 0.6), "s": (0.5, 2.0)}
        )
        at = CalibrationResult(
            best_parameters={"k1": 0.3, "s": 1.0},
            final_loss=0.0,
            iterations=0,
            converged=True,
            termination_reason="given",
        )

        report = calibrator.identifiability(
            Collinearity(at=at, max_subset_size=2, threshold=20)
        )

        practical = report.verdicts["s"].practical
        assert practical is not None
        assert (practical.verdict, practical.reasons) == (
            "poorly_determined",
            ["Collinearity: no observation is sensitive to it."],
        )

    def test_a_result_holds_every_parameter_of_the_problem(
        self, product: Calibrator
    ) -> None:
        at = CalibrationResult(
            best_parameters={"k1": 0.3, "k2": 0.1},
            final_loss=0.0,
            iterations=0,
            converged=True,
            termination_reason="given",
        )

        with pytest.raises(ValueError, match=r"The result has no value of \['s'\]"):
            product.identifiability(
                Collinearity(at=at, max_subset_size=2, threshold=20)
            )

    def test_a_sample_holds_the_parameters_of_the_problem(
        self, product: Calibrator
    ) -> None:
        sample = ParameterSample.equally_weighted(
            ["k1", "s"], [(0.1, 0.6), (0.5, 2.0)], [[0.3, 1.0]], SampleMethod(name="g")
        )

        with pytest.raises(ValueError, match=r"parameters \['k1', 's'\] are not"):
            product.identifiability(
                Collinearity(at=sample, max_subset_size=2, threshold=20)
            )

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

    def test_a_sample_gives_the_reason_of_each_verdict(
        self, product: Calibrator
    ) -> None:
        # k1 and s lie on a line; k2 has two clusters, apart from both.
        k1 = [0.1 + 0.5 * (i + 0.5) / 100 for i in range(100)]
        k2 = [0.05 + 0.2 * (i % 2) + 0.01 * ((i * 37) % 11) / 11 for i in range(100)]
        lines = ParameterSample.equally_weighted(
            ["k1", "s", "k2"],
            [(0.1, 0.6), (0.5, 2.0), (0.02, 0.3)],
            [[k, 2.1 - 2.0 * k, c] for k, c in zip(k1, k2, strict=True)],
            SampleMethod(name="given"),
        )
        near = ParameterSample.equally_weighted(
            ["k1", "s", "k2"],
            [(0.1, 0.6), (0.5, 2.0), (0.02, 0.3)],
            [[0.3 + 0.001 * ((i * 13) % 7), 1.0, 0.1] for i in range(50)],
            SampleMethod(name="given"),
        )

        def reasons(sample: ParameterSample) -> dict[str, tuple[str, list[str]]]:
            report = product.identifiability(
                SampleIdentifiability(
                    sample=sample, reference="bounds", correlation_threshold=0.9
                )
            )
            return {
                name: (verdict.practical.verdict, verdict.practical.reasons)
                for name, verdict in report.verdicts.items()
                if verdict.practical is not None
            }

        assert reasons(lines) == {
            "k1": ("poorly_determined", ["Sample: correlation -1.000 with s."]),
            "s": ("poorly_determined", ["Sample: correlation -1.000 with k1."]),
            "k2": ("poorly_determined", ["Sample: its marginal has 2 modes."]),
        }
        assert reasons(near)["k1"] == (
            "determined",
            ["Sample: contraction 1.00, no strong pair."],
        )

    def test_contraction_against_the_prior_of_a_prior_sample_is_zero(
        self, product: Calibrator
    ) -> None:
        bounds = [(0.1, 0.6), (0.5, 2.0), (0.02, 0.3)]
        priors = Priors(
            distributions={
                "k1": Beta(a=2.0, b=5.0),
                "s": Normal(mean=1.0, sd=0.2),
                "k2": Gamma(shape=4.0, rate=40.0),
            }
        )
        columns = [
            priors.distributions[name].sample(4000, bound, seed=index)
            for index, (name, bound) in enumerate(zip(["k1", "s", "k2"], bounds))
        ]
        sample = ParameterSample.equally_weighted(
            ["k1", "s", "k2"],
            bounds,
            [list(row) for row in zip(*columns, strict=True)],
            SampleMethod(name="prior"),
        )

        against_prior = product.identifiability(
            SampleIdentifiability(
                sample=sample,
                reference="prior",
                priors=priors,
                correlation_threshold=0.9,
            )
        ).sample[0]
        against_bounds = product.identifiability(
            SampleIdentifiability(
                sample=sample, reference="bounds", correlation_threshold=0.9
            )
        ).sample[0]

        assert against_prior.reference == "prior"
        for name in ["k1", "s", "k2"]:
            assert against_prior.contraction[name] == pytest.approx(0.0, abs=0.1)
            # The priors are narrower than their bounds.
            assert against_bounds.contraction[name] > 0.3

    def test_a_reference_of_priors_needs_the_priors(self) -> None:
        sample = ParameterSample.equally_weighted(
            ["k1"], [(0.0, 1.0)], [[0.5]], SampleMethod(name="g")
        )
        with pytest.raises(ValidationError, match="reference='prior' needs priors"):
            SampleIdentifiability(
                sample=sample, reference="prior", correlation_threshold=0.9
            )
        with pytest.raises(ValidationError, match="priors are read only with"):
            SampleIdentifiability(
                sample=sample,
                reference="bounds",
                priors=Priors.uniform_on_bounds(),
                correlation_threshold=0.9,
            )

    def test_a_sample_of_other_parameters_is_rejected(
        self, product: Calibrator
    ) -> None:
        sample = ParameterSample.equally_weighted(
            ["k1", "s"], [(0.1, 0.6), (0.5, 2.0)], [[0.3, 1.0]], SampleMethod(name="g")
        )

        with pytest.raises(ValueError) as error:
            product.identifiability(
                SampleIdentifiability(
                    sample=sample, reference="bounds", correlation_threshold=0.9
                )
            )

        assert str(error.value) == (
            "The sample's parameters ['k1', 's'] are not the problem's "
            "['k1', 's', 'k2']."
        )

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
        assert {n: v.practical for n, v in flagged.verdicts.items()} == {
            "k1": PracticalVerdict(
                verdict="poorly_determined",
                reasons=[
                    "Starts: the 2 converged starts of equal loss end 60% of its "
                    "bounds apart."
                ],
            ),
            "s": PracticalVerdict(
                verdict="poorly_determined",
                reasons=[
                    "Starts: the 2 converged starts of equal loss end 60% of its "
                    "bounds apart."
                ],
            ),
            "k2": PracticalVerdict(
                verdict="determined",
                reasons=["Starts: the 2 converged starts of equal loss agree on it."],
            ),
        }

    def test_a_start_without_a_parameter_is_rejected(self, product: Calibrator) -> None:
        result = _result(
            [{"k1": 0.2, "s": 1.5}, {"k1": 0.5, "s": 0.6, "k2": 0.1}], [1.0, 1.0]
        )

        with pytest.raises(ValueError) as error:
            product.identifiability(
                StartsIdentifiability(
                    result=result, parameter_tolerance=0.05, loss_tolerance=1e-3
                )
            )

        assert str(error.value) == "A start has no value of ['k2']."

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
    def test_the_verdicts_hold_the_parameters_in_order(self) -> None:
        with pytest.raises(ValidationError, match=r"parameters \['k', 'j'\] in order"):
            IdentifiabilityReport(
                parameters=["k", "j"],
                verdicts={"j": ParameterVerdicts(), "k": ParameterVerdicts()},
            )

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

    def test_merge_of_disagreeing_practical_verdicts_is_conflicting(self) -> None:
        merged = _report(practical="determined").merge(
            _report(practical="poorly_determined")
        )

        assert merged.verdicts["k"].practical == PracticalVerdict(
            verdict="conflicting", reasons=["determined", "poorly_determined"]
        )

    def test_merge_keeps_each_combination_once_in_order(self) -> None:
        def report(*combinations: str) -> IdentifiabilityReport:
            verdict = StructuralVerdict(
                verdict="in_combination", combinations=list(combinations)
            )
            return IdentifiabilityReport(
                parameters=["k"], verdicts={"k": ParameterVerdicts(structural=verdict)}
            )

        merged = report("k · s", "k / s").merge(report("k / s", "k · t"))

        assert merged.verdicts["k"].structural == StructuralVerdict(
            verdict="in_combination", combinations=["k · s", "k / s", "k · t"]
        )

    def test_merge_keeps_the_sections_of_every_report_in_order(self) -> None:
        sections = {
            "structural": StructuralSection,
            "collinearity": CollinearitySection,
            "sample": SampleSection,
            "starts": StartsSection,
        }

        def report() -> tuple[IdentifiabilityReport, dict[str, object]]:
            held = {name: kind.model_construct() for name, kind in sections.items()}
            return _report().model_copy(
                update={name: [section] for name, section in held.items()}
            ), held

        first, first_sections = report()
        second, second_sections = report()
        merged = first.merge(second)

        for name in sections:
            assert getattr(merged, name) == [
                first_sections[name],
                second_sections[name],
            ]
            assert getattr(merged, name)[0] is first_sections[name]

    def test_reports_of_other_parameters_do_not_merge(self) -> None:
        other = IdentifiabilityReport(
            parameters=["s"], verdicts={"s": ParameterVerdicts()}
        )

        with pytest.raises(ValueError) as error:
            _report().merge(other)

        assert str(error.value) == (
            "Reports of different parameters cannot merge: ['k'] and ['s']."
        )


class TestEntryPoint:
    def test_derivative_methods_refuse_a_stochastic_engine(self) -> None:
        model = build_stochastic(
            abc_builder(parameters={"k1": None, "k2": 0.1}), "StochasticSteps"
        )
        problem = CalibrationProblem(
            observed_data=[ObservedDataPoint(at=10, output="B", value=50.0)],
            parameters=[bounded("k1", 0.1, 0.6)],
            loss_function="sse",
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
