"""The runners of the identifiability methods.

Each runner sends its method whole to the engine, with the handle of the
result the method reads, and builds the `IdentifiabilityReport` from the
analysis. The engine checks the engine type and the parameter names; no
runner computes a derivative, a rank or a statistic.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from commol._calibration.method import CalibrationPort
from commol._data.engine_json import engine_handles, engine_json
from commol.commol_rs import _commol_rs as commol_rs
from commol.identifiability.results import (
    CollinearitySection,
    Combination,
    CorrelatedPair,
    IdentifiabilityReport,
    ParameterVerdicts,
    PointRank,
    PracticalVerdict,
    ProfileSection,
    SampleSection,
    StartsSection,
    StructuralSection,
    StructuralVerdict,
    SubsetIndex,
)
from commol.profiles.results import RESET_REASON

if TYPE_CHECKING:
    from commol.identifiability.methods import (
        Collinearity,
        ProfileIdentifiability,
        SampleIdentifiability,
        StartsIdentifiability,
        StructuralRank,
    )


def _power(name: str, exponent: float) -> str:
    size = abs(exponent)
    text = f"{size:g}"
    return name if size == 1 else f"{name}^{text}"


def _combination_text(exponents: dict[str, float], negative: set[str]) -> str:
    names = {name: f"|{name}|" if name in negative else name for name in exponents}
    above = [_power(names[n], e) for n, e in exponents.items() if e > 0]
    below = [_power(names[n], e) for n, e in exponents.items() if e < 0]
    numerator = " · ".join(above) if above else "1"
    if not below:
        return numerator
    denominator = below[0] if len(below) == 1 else f"({' · '.join(below)})"
    return f"{numerator} / {denominator}"


def _report(
    calibrator: CalibrationPort,
    *,
    structural_verdicts: dict[str, StructuralVerdict] | None = None,
    practical_verdicts: dict[str, PracticalVerdict] | None = None,
    **sections: list,
) -> IdentifiabilityReport:
    names = calibrator.parameter_names
    return IdentifiabilityReport(
        parameters=names,
        verdicts={
            name: ParameterVerdicts(
                structural=(structural_verdicts or {}).get(name),
                practical=(practical_verdicts or {}).get(name),
            )
            for name in names
        },
        **sections,
    )


def run_structural_rank(
    calibrator: CalibrationPort, method: "StructuralRank"
) -> IdentifiabilityReport:
    """Run `StructuralRank` on `calibrator`."""
    names = calibrator.parameter_names
    analysis = commol_rs.inference.structural_rank(
        calibrator._engine_problem, engine_json(method)
    )
    negative = {p.id for p in calibrator.problem.parameters if p.min_bound < 0}
    combinations = []
    for pairs in analysis.combinations:
        exponents = {names[index]: exponent for index, exponent in pairs}
        combinations.append(
            Combination(
                exponents=exponents, text=_combination_text(exponents, negative)
            )
        )
    verdicts = dict(zip(names, analysis.verdicts, strict=True))
    section = StructuralSection(
        design=method.design,
        points=[
            PointRank(
                values=dict(zip(names, point.values, strict=True)),
                singular_values=point.singular_values,
                threshold=point.threshold,
                rank=point.rank,
            )
            for point in analysis.points
        ],
        skipped=analysis.skipped,
        generic_rank=analysis.generic_rank,
        projections=dict(zip(names, analysis.projections, strict=True)),
        verdicts=verdicts,
        combinations=combinations,
    )
    if method.design == "dense":
        return _report(calibrator, structural=[section])
    rank = f"generic rank {analysis.generic_rank} of {len(names)}"
    structural: dict[str, StructuralVerdict] = {}
    for name in names:
        projection = section.projections[name]
        reason = (
            f"Structural rank on the observations ({rank}): its projection on "
            f"the null space is {projection:.2f}."
        )
        structural[name] = StructuralVerdict(
            verdict=verdicts[name],
            reasons=[reason],
            combinations=[c.text for c in combinations if name in c.exponents],
        )
    return _report(calibrator, structural_verdicts=structural, structural=[section])


def run_collinearity(
    calibrator: CalibrationPort, method: "Collinearity"
) -> IdentifiabilityReport:
    """Run `Collinearity` on `calibrator`."""
    names = calibrator.parameter_names
    analysis = commol_rs.inference.collinearity(
        calibrator._engine_problem, engine_json(method), **engine_handles(method)
    )

    def subsets(entries: Sequence[tuple[list[int], float, float]]) -> list[SubsetIndex]:
        return [
            SubsetIndex(
                parameters=[names[i] for i in indices],
                gamma=gamma,
                share_below=share,
            )
            for indices, gamma, share in entries
        ]

    pair_gamma = {
        name: dict(zip(names, row, strict=True))
        for name, row in zip(names, analysis.pair_gamma, strict=True)
    }
    section = CollinearitySection(
        threshold=method.threshold,
        points=analysis.points,
        skipped=analysis.skipped,
        sensitivity=dict(zip(names, analysis.sensitivity, strict=True)),
        pair_gamma=pair_gamma,
        best_by_size=subsets(analysis.best_by_size),
        admissible=subsets(analysis.admissible),
    )
    practical: dict[str, PracticalVerdict] = {}
    for name, verdict in zip(names, analysis.verdicts, strict=True):
        if verdict.insensitive:
            reason = "Collinearity: no observation is sensitive to it."
        elif verdict.collinear_with:
            collinear = ", ".join(names[k] for k in verdict.collinear_with)
            reason = (
                f"Collinearity: it cannot be fitted together with {collinear} "
                f"(index at or above {method.threshold:g})."
            )
        else:
            reason = (
                f"Collinearity: every pair that holds it has an index below "
                f"{method.threshold:g}."
            )
        practical[name] = PracticalVerdict(verdict=verdict.verdict, reasons=[reason])
    return _report(calibrator, practical_verdicts=practical, collinearity=[section])


def run_sample_identifiability(
    calibrator: CalibrationPort, method: "SampleIdentifiability"
) -> IdentifiabilityReport:
    """Run `SampleIdentifiability` on `calibrator`."""
    names = calibrator.parameter_names
    analysis = commol_rs.inference.sample_identifiability(
        calibrator._engine_problem, engine_json(method), **engine_handles(method)
    )
    pairs = [
        CorrelatedPair(first=names[j], second=names[k], correlation=r)
        for j, k, r in analysis.correlated_pairs
    ]
    section = SampleSection(
        reference=method.reference,
        correlation_threshold=method.correlation_threshold,
        contraction=dict(zip(names, analysis.contraction, strict=True)),
        correlation={
            name: dict(zip(names, row, strict=True))
            for name, row in zip(names, analysis.correlation, strict=True)
        },
        correlated_pairs=pairs,
        modes=dict(zip(names, analysis.modes, strict=True)),
    )
    practical: dict[str, PracticalVerdict] = {}
    for name, verdict in zip(names, analysis.verdicts, strict=True):
        reasons = [
            f"Sample: correlation {r:.3f} with {names[k]}."
            for k, r in verdict.correlated_with
        ]
        if verdict.modes >= 2:
            reasons.append(f"Sample: its marginal has {verdict.modes} modes.")
        if verdict.verdict == "determined":
            contraction = section.contraction[name]
            reasons = [f"Sample: contraction {contraction:.2f}, no strong pair."]
        practical[name] = PracticalVerdict(verdict=verdict.verdict, reasons=reasons)
    return _report(calibrator, practical_verdicts=practical, sample=[section])


def run_starts_identifiability(
    calibrator: CalibrationPort, method: "StartsIdentifiability"
) -> IdentifiabilityReport:
    """Run `StartsIdentifiability` on `calibrator`."""
    names = calibrator.parameter_names
    analysis = commol_rs.inference.starts_identifiability(
        calibrator._engine_problem, engine_json(method), **engine_handles(method)
    )
    flagged = [name for name, flag in zip(names, analysis.flagged, strict=True) if flag]
    spread = dict(zip(names, analysis.spread, strict=True))
    section = StartsSection(
        parameter_tolerance=method.parameter_tolerance,
        loss_tolerance=method.loss_tolerance,
        near_best=analysis.near_best,
        spread=spread,
        flagged=flagged,
    )
    count = len(analysis.near_best)
    practical = {
        name: PracticalVerdict(
            verdict="poorly_determined",
            reasons=[
                f"Starts: the {count} converged starts of equal loss end "
                f"{spread[name]:.0%} of its bounds apart."
            ],
        )
        if name in flagged
        else PracticalVerdict(
            verdict="determined",
            reasons=[
                f"Starts: the {count} converged starts of equal loss agree on it."
            ],
        )
        for name in names
    }
    return _report(calibrator, practical_verdicts=practical, starts=[section])


def run_profile_identifiability(
    calibrator: CalibrationPort, method: "ProfileIdentifiability"
) -> IdentifiabilityReport:
    """Run `ProfileIdentifiability` on `calibrator`: the verdicts of the
    profiles, which the engine computed when it profiled."""
    result = method.result
    names = calibrator.parameter_names
    if list(result.fit.best_parameters) != names:
        raise ValueError(
            f"The profiles are of the parameters {list(result.fit.best_parameters)}; "
            f"the problem has {names}."
        )
    structural: dict[str, StructuralVerdict] = {}
    practical: dict[str, PracticalVerdict] = {}
    for name in names:
        verdict = result.verdicts.get(name)
        if verdict is None:
            continue
        reason = f"profile likelihood: {verdict}. {RESET_REASON}"
        if verdict == "structurally_non_identifiable":
            structure = result.structure[name]
            reason = structure.reason(
                name, result.profiles[name].threshold, result.evaluation_floor
            )
            if structure.verdict is None:
                practical[name] = PracticalVerdict(
                    verdict="poorly_determined", reasons=[reason]
                )
            else:
                structural[name] = StructuralVerdict(
                    verdict=structure.verdict, reasons=[reason]
                )
        else:
            practical[name] = PracticalVerdict(
                verdict=(
                    "determined" if verdict == "identifiable" else "poorly_determined"
                ),
                reasons=[reason],
            )
    section = ProfileSection(
        threshold=result.threshold.kind,
        confidence_level=result.threshold.confidence_level,
        tolerance=result.tolerance,
        verdicts=result.verdicts,
        intervals={
            name: (interval.lower, interval.upper)
            for name, interval in result.intervals.items()
        },
    )
    return _report(
        calibrator,
        structural_verdicts=structural,
        practical_verdicts=practical,
        profile=[section],
    )
