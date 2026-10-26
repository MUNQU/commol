"""The report of an identifiability analysis (`IdentifiabilityReport.report`):
plain data, whose key order is part of the contract."""

from typing import TYPE_CHECKING, TypedDict, cast

from commol.profiles.results import ProfileVerdictName

if TYPE_CHECKING:
    from commol.identifiability.results import (
        IdentifiabilityReport,
        ProfileThresholdName,
        SampleReference,
        StructuralDesign,
        StructuralVerdictName,
    )


class StructuralVerdictEntry(TypedDict):
    """The structural verdict of a parameter, its reasons and combinations."""

    verdict: str
    reasons: list[str]
    combinations: list[str]


class PracticalVerdictEntry(TypedDict):
    """The practical verdict of a parameter and its reasons."""

    verdict: str
    reasons: list[str]


class VerdictsEntry(TypedDict):
    """The two verdicts of a parameter; None when no evidence gives one."""

    structural: StructuralVerdictEntry | None
    practical: PracticalVerdictEntry | None


class PointRankData(TypedDict):
    """`PointRank` as plain data."""

    values: dict[str, float]
    singular_values: list[float]
    threshold: float
    rank: int


class CombinationData(TypedDict):
    """`Combination` as plain data."""

    exponents: dict[str, float]
    text: str


class StructuralSectionData(TypedDict):
    """`StructuralSection` as plain data."""

    design: "StructuralDesign"
    points: list[PointRankData]
    skipped: list[str]
    generic_rank: int
    projections: dict[str, float]
    verdicts: "dict[str, StructuralVerdictName]"
    combinations: list[CombinationData]


class SubsetIndexData(TypedDict):
    """`SubsetIndex` as plain data."""

    parameters: list[str]
    gamma: float
    share_below: float


class CollinearitySectionData(TypedDict):
    """`CollinearitySection` as plain data."""

    threshold: float
    points: int
    skipped: list[str]
    sensitivity: dict[str, float]
    pair_gamma: dict[str, dict[str, float]]
    best_by_size: list[SubsetIndexData]
    admissible: list[SubsetIndexData]


class CorrelatedPairData(TypedDict):
    """`CorrelatedPair` as plain data."""

    first: str
    second: str
    correlation: float


class SampleSectionData(TypedDict):
    """`SampleSection` as plain data."""

    reference: "SampleReference"
    correlation_threshold: float
    contraction: dict[str, float]
    correlation: dict[str, dict[str, float]]
    correlated_pairs: list[CorrelatedPairData]
    modes: dict[str, list[float]]


class StartsSectionData(TypedDict):
    """`StartsSection` as plain data."""

    parameter_tolerance: float
    loss_tolerance: float
    near_best: list[int]
    spread: dict[str, float]
    flagged: list[str]


class ProfileSectionData(TypedDict):
    """`ProfileSection` as plain data: each interval is a pair of ends."""

    threshold: "ProfileThresholdName"
    confidence_level: float
    tolerance: float
    verdicts: dict[str, ProfileVerdictName | None]
    intervals: dict[str, tuple[float | None, float | None]]


class IdentifiabilitySections(TypedDict):
    """The sections of each method of an identifiability report."""

    structural: list[StructuralSectionData]
    collinearity: list[CollinearitySectionData]
    sample: list[SampleSectionData]
    starts: list[StartsSectionData]
    profile: list[ProfileSectionData]


class IdentifiabilityReportData(TypedDict):
    """
    The identifiability of each parameter, and the evidence of each method.

    ``sections`` holds the sections of each method, as plain data
    (`IdentifiabilitySections`).
    """

    parameters: dict[str, VerdictsEntry]
    sections: IdentifiabilitySections


def identifiability_report(
    report: "IdentifiabilityReport",
) -> IdentifiabilityReportData:
    """The block of `IdentifiabilityReport.report`, which documents it."""
    parameters: dict[str, VerdictsEntry] = {}
    for name in report.parameters:
        verdicts = report.verdicts[name]
        structural = verdicts.structural
        practical = verdicts.practical
        parameters[name] = {
            "structural": None
            if structural is None
            else {
                "verdict": structural.verdict,
                "reasons": list(structural.reasons),
                "combinations": list(structural.combinations),
            },
            "practical": None
            if practical is None
            else {"verdict": practical.verdict, "reasons": list(practical.reasons)},
        }
    sections: IdentifiabilitySections = {
        "structural": [
            cast(StructuralSectionData, section.model_dump())
            for section in report.structural
        ],
        "collinearity": [
            cast(CollinearitySectionData, section.model_dump())
            for section in report.collinearity
        ],
        "sample": [
            cast(SampleSectionData, section.model_dump()) for section in report.sample
        ],
        "starts": [
            cast(StartsSectionData, section.model_dump()) for section in report.starts
        ],
        "profile": [
            cast(ProfileSectionData, section.model_dump()) for section in report.profile
        ],
    }
    return {"parameters": parameters, "sections": sections}
