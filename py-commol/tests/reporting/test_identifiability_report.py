"""The identifiability report: one entry per parameter with its verdicts, and
the evidence of each section."""

from commol import (
    IdentifiabilityReport,
    identifiability_report,
)
from commol.context.inference.identifiability import (
    ParameterVerdicts,
    PracticalVerdict,
    StartsSection,
    StructuralVerdict,
)


def test_the_report_lists_each_parameter_and_section() -> None:
    report = IdentifiabilityReport(
        parameters=["k", "f"],
        verdicts={
            "k": ParameterVerdicts(
                structural=StructuralVerdict(
                    verdict="in_combination",
                    reasons=["k and f enter as k · f"],
                    combinations=["k · f"],
                ),
                practical=PracticalVerdict(
                    verdict="poorly_determined", reasons=["starts differ"]
                ),
            ),
            "f": ParameterVerdicts(),
        },
        starts=[
            StartsSection(
                parameter_tolerance=0.05,
                loss_tolerance=1e-3,
                near_best=[0, 1],
                spread={"k": 0.6, "f": 0.0},
                flagged=["k"],
            )
        ],
    )

    data = identifiability_report(report)

    assert list(data["parameters"]) == ["k", "f"]
    assert data["parameters"]["k"] == {
        "structural": {
            "verdict": "in_combination",
            "reasons": ["k and f enter as k · f"],
            "combinations": ["k · f"],
        },
        "practical": {"verdict": "poorly_determined", "reasons": ["starts differ"]},
    }
    assert data["parameters"]["f"] == {"structural": None, "practical": None}
    assert data["sections"]["starts"][0]["flagged"] == ["k"]
    assert data["sections"]["structural"] == []
