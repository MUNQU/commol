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
from commol.identifiability import StructuralRank
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
