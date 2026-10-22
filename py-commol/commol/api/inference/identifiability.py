"""The runners of the identifiability methods.

Each runner checks what only Python knows (the engine, the parameter names),
calls the analysis in Rust and builds the `IdentifiabilityReport`. No runner
computes a derivative, a rank or a statistic.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.calibration import CalibrationResult
from commol.context.inference.identifiability import (
    Collinearity,
    CollinearitySection,
    Combination,
    CorrelatedPair,
    IdentifiabilityReport,
    ParameterVerdicts,
    PointRank,
    PracticalVerdict,
    SampleIdentifiability,
    SampleSection,
    StartsIdentifiability,
    StartsSection,
    StructuralRank,
    StructuralSection,
    StructuralVerdict,
    SubsetIndex,
)
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator

rust = commol_rs.inference

_PRACTICAL_METHODS = "SampleIdentifiability and StartsIdentifiability"


def _deterministic(calibrator: "Calibrator", method: str) -> None:
    if calibrator.simulation.is_stochastic:
        raise ValueError(
            f"{method} differentiates the model, which needs a deterministic "
            "engine: the runs of a stochastic engine change in jumps with its "
            f"parameters. {_PRACTICAL_METHODS} accept a stochastic engine."
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
    calibrator: "Calibrator",
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
    calibrator: "Calibrator", method: StructuralRank
) -> IdentifiabilityReport:
    """Run `StructuralRank` on `calibrator`."""
    _deterministic(calibrator, "StructuralRank")
    names = calibrator.parameter_names
    analysis = rust.structural_rank(
        calibrator.rust_problem,
        method.points,
        method.design == "dense",
        calibrator.seed,
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


def _points_of(
    calibrator: "Calibrator", at: CalibrationResult | ParameterSample
) -> tuple[list[list[float]], list[float]]:
    names = calibrator.parameter_names
    if isinstance(at, CalibrationResult):
        missing = [name for name in names if name not in at.best_parameters]
        if missing:
            raise ValueError(f"The result has no value of {missing}.")
        return [[at.best_parameters[name] for name in names]], [1.0]
    if at.names != names:
        raise ValueError(
            f"The sample's parameters {at.names} are not the problem's {names}."
        )
    return at.values, at.weights


def run_collinearity(
    calibrator: "Calibrator", method: Collinearity
) -> IdentifiabilityReport:
    """Run `Collinearity` on `calibrator`."""
    _deterministic(calibrator, "Collinearity")
    names = calibrator.parameter_names
    if method.max_subset_size > len(names):
        raise ValueError(
            f"max_subset_size must be at most the number of parameters "
            f"({len(names)}), got {method.max_subset_size}."
        )
    points, weights = _points_of(calibrator, method.at)
    analysis = rust.collinearity(
        calibrator.rust_problem,
        points,
        weights,
        method.max_subset_size,
        method.threshold,
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
    for name in names:
        others = {other: g for other, g in pair_gamma[name].items() if other != name}
        collinear = [
            other for other, gamma in others.items() if gamma >= method.threshold
        ]
        if section.sensitivity[name] == 0.0:
            verdict, reason = (
                "poorly_determined",
                "Collinearity: no observation is sensitive to it.",
            )
        elif collinear:
            verdict, reason = (
                "poorly_determined",
                "Collinearity: it cannot be fitted together with "
                f"{', '.join(collinear)} (index at or above {method.threshold:g}).",
            )
        else:
            verdict, reason = (
                "determined",
                f"Collinearity: every pair that holds it has an index below "
                f"{method.threshold:g}.",
            )
        practical[name] = PracticalVerdict(verdict=verdict, reasons=[reason])
    return _report(calibrator, practical_verdicts=practical, collinearity=[section])


def run_sample_identifiability(
    calibrator: "Calibrator", method: SampleIdentifiability
) -> IdentifiabilityReport:
    """Run `SampleIdentifiability` on `calibrator`."""
    names = calibrator.parameter_names
    sample = method.sample
    if sample.names != names:
        raise ValueError(
            f"The sample's parameters {sample.names} are not the problem's {names}."
        )
    priors = method.priors.rust_priors() if method.priors is not None else None
    analysis = rust.sample_identifiability(
        sample.rust_sample, method.correlation_threshold, priors
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
    for name, multimodal in zip(names, analysis.multimodal, strict=True):
        reasons = [
            f"Sample: correlation {pair.correlation:.3f} with "
            f"{pair.second if pair.first == name else pair.first}."
            for pair in pairs
            if name in (pair.first, pair.second)
        ]
        if multimodal:
            reasons.append(
                f"Sample: its marginal has {len(section.modes[name])} modes."
            )
        contraction = section.contraction[name]
        practical[name] = (
            PracticalVerdict(verdict="poorly_determined", reasons=reasons)
            if reasons
            else PracticalVerdict(
                verdict="determined",
                reasons=[f"Sample: contraction {contraction:.2f}, no strong pair."],
            )
        )
    return _report(calibrator, practical_verdicts=practical, sample=[section])


def run_starts_identifiability(
    calibrator: "Calibrator", method: StartsIdentifiability
) -> IdentifiabilityReport:
    """Run `StartsIdentifiability` on `calibrator`."""
    names = calibrator.parameter_names
    starts = method.result.starts
    for start in starts:
        missing = [name for name in names if name not in start.end]
        if missing:
            raise ValueError(f"A start has no value of {missing}.")
    analysis = rust.starts_identifiability(
        [[start.end[name] for name in names] for start in starts],
        [start.loss for start in starts],
        [start.converged for start in starts],
        [(p.min_bound, p.max_bound) for p in calibrator.problem.parameters],
        method.parameter_tolerance,
        method.loss_tolerance,
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


register(StructuralRank, run_structural_rank)
register(Collinearity, run_collinearity)
register(SampleIdentifiability, run_sample_identifiability)
register(StartsIdentifiability, run_starts_identifiability)
