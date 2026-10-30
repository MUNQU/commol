"""Identifiability: the structural rank, collinearity, a sample and the
starts. The report that joins them is in ``test_identifiability_report.py``."""

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    ParticleSwarmConfig,
    Simulation,
)
from commol._fit.result import CalibrationResult, StartSummary
from commol.distributions import Beta, Gamma, Normal, Priors
from commol.identifiability import (
    Collinearity,
    SampleIdentifiability,
    StartsIdentifiability,
    StructuralRank,
)
from commol.identifiability.results import PracticalVerdict
from commol.uncertainty import ParameterSample, SampleMethod
from tests.support import (
    abc_builder,
    bounded,
    build,
    observed,
)

TRUTH = {"k1": 0.3, "k2": 0.1}


def _calibrator(
    observations: list[ObservedDataPoint],
    parameters: dict[str, tuple[float, float]],
    *,
    ab: str = "k1 * A * B / total",
    factors: tuple[str, ...] = (),
    seed: int = 3,
    config: NelderMeadConfig | ParticleSwarmConfig | None = None,
) -> Calibrator:
    """``abc_builder`` with `parameters` to calibrate in their bounds by `config`."""
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
        optimization_config=config or NelderMeadConfig(max_iterations=200),
        seed=seed,
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

    def test_structural_rank_takes_the_seed_of_the_calibrator(
        self, truth: dict[str, list[float]]
    ) -> None:
        def points(seed: int) -> list[dict[str, float]]:
            calibrator = _calibrator(
                observed(truth, "B", range(5, 51, 5)),
                {"k1": (0.1, 0.6), "k2": (0.02, 0.3)},
                seed=seed,
            )
            report = calibrator.identifiability(
                StructuralRank(points=3, design="observations")
            )
            return [point.values for point in report.structural[0].points]

        assert points(3) == points(3)
        assert points(3) != points(4)

    def test_structural_rank_has_no_seed(self) -> None:
        with pytest.raises(
            ValidationError, match=r"\nseed\n  Extra inputs are not permitted"
        ):
            StructuralRank(points=3, design="observations", seed=1)

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
            seed=0,
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
            seed=0,
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
            seed=0,
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

    def test_collinearity_rejects_invalid_settings(self) -> None:
        with pytest.raises(ValidationError, match="threshold"):
            Collinearity(at=_best(), max_subset_size=2)  # ty: ignore[missing-argument]

    @pytest.mark.parametrize("size", [1, -1])
    def test_max_subset_size_below_2_gets_the_message_of_the_engine(
        self, size: int
    ) -> None:
        with pytest.raises(ValueError) as error:
            Collinearity(at=_best(), max_subset_size=size, threshold=20)

        assert f"max_subset_size is a whole number of at least 2, got {size}" in str(
            error.value
        )

    def test_max_subset_size_above_the_parameters_gets_the_message_of_the_engine(
        self, product: Calibrator
    ) -> None:
        with pytest.raises(ValueError) as error:
            product.identifiability(
                Collinearity(at=_best(), max_subset_size=9, threshold=20)
            )

        assert (
            str(error.value) == "max_subset_size is a whole number from 2 to 3, got 9"
        )


def _best() -> CalibrationResult:
    """A fit of the three parameters of `product`."""
    return CalibrationResult(
        best_parameters={"k1": 0.3, "s": 1.0, "k2": 0.1},
        final_loss=0.0,
        iterations=0,
        converged=True,
        termination_reason="given",
        seed=0,
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
        seed=0,
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


def test_the_starts_of_a_polished_fit_are_its_polished_ends(
    truth: dict[str, list[float]],
) -> None:
    # Values 2% off the run: the best loss is not 0, so equal losses agree.
    data = [
        point.model_copy(update={"value": point.value * (1.02 if i % 2 else 0.98)})
        for i, point in enumerate(observed(truth, "B", range(5, 51, 5)))
    ]
    bounds = {"k1": (0.1, 0.6), "s": (0.5, 2.0), "k2": (0.02, 0.3)}
    polish = NelderMeadConfig(max_iterations=2000, sd_tolerance=1e-12)
    calibrators = [
        _calibrator(data, bounds, ab="k1 * s * A * B / total", config=swarm)
        for swarm in (
            ParticleSwarmConfig(particles=8, max_iterations=3, polish=polish),
            ParticleSwarmConfig(particles=8, max_iterations=3),
        )
    ]
    polished, alone = (calibrator.run(starts=4) for calibrator in calibrators)
    method = {"parameter_tolerance": 0.05, "loss_tolerance": 1e-6}
    report = calibrators[0].identifiability(
        StartsIdentifiability(result=polished, **method)
    )

    # The polished ends converged at one loss, spread along k1 · s.
    assert all(start.converged for start in polished.starts)
    assert report.starts[0].flagged == ["k1", "s"]
    assert report.verdicts["k2"].practical.reasons == [
        "Starts: the 4 converged starts of equal loss agree on it."
    ]
    # The swarm alone converges at no start, so the method has none to read.
    with pytest.raises(ValueError, match="at least two that converged, got 0"):
        calibrators[1].identifiability(StartsIdentifiability(result=alone, **method))


def test_the_reference_matches_the_priors() -> None:
    # The engine reads the JSON of the method: no priors with "prior", and
    # priors only with "prior".
    given = SampleMethod(name="g")
    sample = ParameterSample.equally_weighted(["k1"], [(0, 1)], [[0.5]], given)
    method = {"sample": sample, "correlation_threshold": 0.9}
    with pytest.raises(ValidationError, match="reference='prior' needs priors"):
        SampleIdentifiability(**method, reference="prior")
    with pytest.raises(ValidationError, match="priors are read only with"):
        SampleIdentifiability(
            **method, reference="bounds", priors=Priors.uniform_on_bounds()
        )
