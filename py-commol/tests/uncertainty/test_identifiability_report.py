"""The identifiability report: its verdicts and how two reports merge; and the
entry point, which refuses a method that needs derivatives for a stochastic
engine."""

import pytest
from pydantic import ValidationError

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.engines import ODESolver
from commol.identifiability import (
    Collinearity,
    StartsIdentifiability,
    StructuralRank,
)
from commol.identifiability.results import (
    CollinearitySection,
    IdentifiabilityReport,
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
    build_stochastic,
    overdispersed_calibrator,
    rare_flow_calibrator,
)


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

    def test_a_parameter_with_no_structural_verdict_keeps_the_practical_one(
        self,
    ) -> None:
        # At this point the rank is 2 at the default tolerances and 3 at a
        # hundredth of them: no parameter has a structural verdict.
        calibrator = rare_flow_calibrator(ODESolver())
        rank = calibrator.identifiability(
            StructuralRank(points=1, design="observations")
        )
        other = IdentifiabilityReport(
            parameters=["k1", "k2", "p"],
            verdicts={
                "k1": ParameterVerdicts(
                    structural=StructuralVerdict(
                        verdict="identifiable", reasons=["profile"]
                    )
                ),
                "k2": ParameterVerdicts(
                    practical=PracticalVerdict(verdict="determined", reasons=["starts"])
                ),
                "p": ParameterVerdicts(),
            },
        )

        merged = rank.merge(other)
        data = rank.report()["parameters"]["k2"]["structural"]

        structural = rank.verdicts["k2"].structural
        assert structural is not None
        (reason,) = structural.reasons
        assert "ODESolver.rtol = 1e-06, ODESolver.atol = 1e-06" in reason
        assert data == {"verdict": None, "reasons": [reason], "combinations": []}
        assert merged.verdicts["k1"].structural == StructuralVerdict(
            verdict="identifiable", reasons=[reason, "profile"]
        )
        assert merged.verdicts["k2"].structural == StructuralVerdict(
            verdict=None, reasons=[reason]
        )
        assert merged.verdicts["k2"].practical == PracticalVerdict(
            verdict="determined", reasons=["starts"]
        )

    def test_a_noise_parameter_leaves_the_verdicts_of_the_others_alone(
        self,
    ) -> None:
        calibrator = overdispersed_calibrator()
        result = calibrator.run(starts=6)
        reports = [
            calibrator.identifiability(method)
            for method in (
                StructuralRank(points=5, design="observations"),
                Collinearity(at=result, max_subset_size=2, threshold=10),
                StartsIdentifiability(
                    result=result, parameter_tolerance=0.05, loss_tolerance=0.01
                ),
            )
        ]

        merged = reports[0].merge(*reports[1:])

        verdicts = {
            name: (
                None if v.structural is None else v.structural.verdict,
                None if v.practical is None else v.practical.verdict,
            )
            for name, v in merged.verdicts.items()
        }
        assert verdicts == {
            "k1": ("identifiable", "determined"),
            "k2": ("identifiable", "determined"),
            "phi": (None, "determined"),
        }
        sections = merged.report()["sections"]
        left_out = {
            "phi": "a noise parameter of an observation model, which no "
            "prediction reads"
        }
        assert sections["structural"][0]["left_out"] == left_out
        assert sections["collinearity"][0]["left_out"] == left_out

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
