"""The runners of the identifiability methods, which
``Calibrator.identifiability`` calls through the ``_runner`` of each method.

Each runner sends its method whole to the engine, with the handle of the
result the method reads, and builds the `IdentifiabilityReport` from the
analysis. The engine checks the engine type and the parameter names; no
runner computes a derivative, a rank or a statistic.
`run_profile_identifiability` calls no engine: it reads the verdicts of a
`ProfileResult` that the profile likelihood already computed.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from commol._calibration.method import CalibrationPort
from commol._data.engine_json import engine_handles, engine_json
from commol.commol_rs import _commol_rs as commol_rs
from commol.identifiability.results import (
    Accuracy,
    CollinearitySection,
    Combination,
    BoundSideName,
    CorrelatedPair,
    Direction,
    IdentifiabilityReport,
    ParameterScale,
    ParameterVerdicts,
    PointRank,
    PracticalVerdict,
    ProfileSection,
    SampleSection,
    StartsSection,
    StructuralSection,
    StructuralVerdict,
    SubsetIndex,
    TighterRank,
)
from commol.profiles.results import RESET_REASON

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import (
        AccuracyProtocol,
        SampleAnalysisProtocol,
        TighteningProtocol,
    )
    from commol.identifiability.methods import (
        Collinearity,
        ProfileIdentifiability,
        SampleIdentifiability,
        StartsIdentifiability,
        StructuralRank,
    )


_NOISE_PARAMETER = (
    "a noise parameter of an observation model, which no prediction reads"
)
"""Why `StructuralRank` and `Collinearity` leave a parameter out."""


def _left_out(names: list[str], differentiated: list[int]) -> dict[str, str]:
    """The parameters of `names` that are not `differentiated`, by index,
    each with the reason."""
    # The engine leaves out only the noise parameters: `differentiated_columns`
    # in `crates/commol-inference/src/identifiability/mod.rs`.
    return {
        name: _NOISE_PARAMETER
        for index, name in enumerate(names)
        if index not in differentiated
    }


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


def _accuracy(accuracy: "AccuracyProtocol") -> Accuracy:
    return Accuracy(relative=accuracy.relative, absolute=accuracy.absolute)


def _at(accuracy: Accuracy, settings: tuple[str, str]) -> str:
    relative, absolute = settings
    return f"{relative} = {accuracy.relative:g}, {absolute} = {accuracy.absolute:g}"


def _unresolved_reason(
    section: StructuralSection, tightening: "TighteningProtocol"
) -> str:
    """Why the structural rank gives no verdict: the points whose rank
    changes at the tighter accuracy, and why a point could not run there."""
    tighter = _accuracy(tightening.accuracy)
    settings = tightening.settings
    changes = [
        f"{point.rank} to {point.tighter.rank}"
        for point in section.points
        if point.tighter is not None and point.tighter.rank != point.rank
    ]
    count = f"{len(changes)} of {len(section.points)} points"
    facts = [f"the rank of {count} changes ({', '.join(changes)})"] if changes else []
    facts += [f"a point could not run: {why}" for why in section.unchecked]
    found = "; ".join(facts) or "no rank changes, but the null space does"
    relative, absolute = settings
    return (
        f"Structural rank on the observations: the accuracy of the engine "
        f"({_at(section.accuracy, settings)}), not the structure, may set its "
        f"verdict. Evaluated again at {_at(tighter, settings)}, {found}. "
        f"Tighten {relative} and {absolute} until the rank stops changing."
    )


def run_structural_rank(
    calibrator: CalibrationPort, method: "StructuralRank"
) -> IdentifiabilityReport:
    """Run `StructuralRank` on `calibrator`."""
    names = calibrator.parameter_names
    analysis = commol_rs.inference.structural_rank(
        calibrator._engine_problem, engine_json(method)
    )
    differentiated = [names[index] for index in analysis.parameters]
    negative = {p.id for p in calibrator.problem.parameters if p.min_bound < 0}
    combinations = []
    for pairs in analysis.combinations:
        exponents = {names[index]: exponent for index, exponent in pairs}
        combinations.append(
            Combination(
                exponents=exponents, text=_combination_text(exponents, negative)
            )
        )
    verdicts = dict(zip(differentiated, analysis.verdicts, strict=True))
    tightening = analysis.tighter
    section = StructuralSection(
        design=method.design,
        points=[
            PointRank(
                values=dict(zip(names, point.values, strict=True)),
                accuracy=_accuracy(point.accuracy),
                singular_values=point.singular_values,
                threshold=point.threshold,
                rank=point.rank,
                tighter=None
                if point.tighter is None
                else TighterRank(
                    accuracy=_accuracy(point.tighter.accuracy),
                    singular_values=point.tighter.singular_values,
                    threshold=point.tighter.threshold,
                    rank=point.tighter.rank,
                ),
            )
            for point in analysis.points
        ],
        skipped=analysis.skipped,
        left_out=_left_out(names, analysis.parameters),
        generic_rank=analysis.generic_rank,
        projections=dict(zip(differentiated, analysis.projections, strict=True)),
        verdicts=verdicts,
        combinations=combinations,
        accuracy=_accuracy(analysis.accuracy),
        tighter=None if tightening is None else _accuracy(tightening.accuracy),
        tighter_generic_rank=analysis.tighter_generic_rank,
        unchecked=analysis.unchecked,
    )
    if method.design == "dense":
        return _report(calibrator, structural=[section])
    rank = f"generic rank {analysis.generic_rank} of {len(differentiated)}"
    unresolved: list[str] = []
    if tightening is not None:
        unresolved.append(_unresolved_reason(section, tightening))
        tighter_rank = section.tighter_generic_rank
        if tighter_rank is not None and tighter_rank != section.generic_rank:
            at = _at(_accuracy(tightening.accuracy), tightening.settings)
            rank += f"; {tighter_rank} at {at}"
    structural: dict[str, StructuralVerdict] = {
        name: StructuralVerdict(
            verdict=None,
            reasons=[f"Structural rank on the observations: left out as {why}."],
        )
        for name, why in section.left_out.items()
    }
    for name in differentiated:
        verdict = verdicts[name]
        if verdict is None:
            # The engine leaves a verdict out only at a tighter accuracy.
            reasons = unresolved
        else:
            projection = section.projections[name]
            reasons = [
                f"Structural rank on the observations ({rank}): its projection "
                f"on the null space is {projection:.2f}."
            ]
        structural[name] = StructuralVerdict(
            verdict=verdict,
            reasons=reasons,
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
    differentiated = [names[index] for index in analysis.parameters]

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
        name: dict(zip(differentiated, row, strict=True))
        for name, row in zip(differentiated, analysis.pair_gamma, strict=True)
    }
    section = CollinearitySection(
        threshold=method.threshold,
        points=analysis.points,
        skipped=analysis.skipped,
        left_out=_left_out(names, analysis.parameters),
        sensitivity=dict(zip(differentiated, analysis.sensitivity, strict=True)),
        pair_gamma=pair_gamma,
        best_by_size=subsets(analysis.best_by_size),
        admissible=subsets(analysis.admissible),
    )
    practical: dict[str, PracticalVerdict] = {}
    for name, verdict in zip(differentiated, analysis.verdicts, strict=True):
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


def _sample_sides(
    name: str, sides: list[BoundSideName], section: SampleSection
) -> list[str]:
    """The reasons of the bounds that set a side of the marginal of `name`."""
    share = f"{100 * section.bound_share:g} %"
    reasons = []
    for side in sides:
        mass = section.mass_below[name] if side == "lower" else section.mass_above[name]
        end = "bottom" if side == "lower" else "top"
        reasons.append(
            f"Sample: the {side} bound, not the data, sets one side; "
            f"{100 * mass:.0f} % of its mass is in the {end} {share} of its "
            "reference."
        )
    return reasons


def _plural(count: int, noun: str) -> str:
    """`count` and `noun`, plural unless `count` is 1."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _sample_warnings(
    analysis: "SampleAnalysisProtocol",
    names: list[str],
    method: "SampleIdentifiability",
) -> list[str]:
    """The warnings of a `SampleSection`: an effective size below ``p + 1``
    or below ``FEW_DRAWS_PER_PARAMETER · p``, no directions, or hidden
    ridges."""
    p = len(names)
    size = f"{analysis.effective_size:.3g}"
    parameters = _plural(p, "parameter")
    warnings = []
    if analysis.size == "too_few":
        warnings.append(
            f"The effective size {size} is below p + 1 = {p + 1} for "
            f"{parameters}: the weights are worth fewer draws than a covariance "
            "of the parameters needs, so no parameter is determined."
        )
    elif analysis.size == "few":
        per_parameter = commol_rs.inference.FEW_DRAWS_PER_PARAMETER
        warnings.append(
            f"The effective size {size} is below {per_parameter:g} · p = "
            f"{per_parameter * p:g} for {parameters}: the count of hidden "
            "ridges over-counts."
        )
    if analysis.hidden_ridges is None:
        warnings.append(
            f"The sample has {analysis.distinct_rows} distinct draws for "
            f"{p} parameters, and no directions: a direction needs more "
            "distinct draws than parameters."
        )
    elif analysis.hidden_ridges > 0:
        count = analysis.hidden_ridges
        plural = "direction" if count == 1 else "directions"
        warnings.append(
            f"{count} {plural} among the determined parameters "
            f"contract{'s' if count == 1 else ''} less than "
            f"{method.contraction_threshold:g}: a ridge that no pair shows "
            f"(n_eff / p = {analysis.draws_per_parameter:.1f})."
        )
    return warnings


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
    scales: dict[str, ParameterScale] = {
        name: "log" if name in method.log_parameters else "natural" for name in names
    }
    section = SampleSection(
        reference=method.reference,
        correlation_threshold=method.correlation_threshold,
        contraction_threshold=method.contraction_threshold,
        bound_share=method.bound_share,
        log_parameters=list(method.log_parameters),
        effective_size=analysis.effective_size,
        contraction=dict(zip(names, analysis.contraction, strict=True)),
        correlation={
            name: dict(zip(names, row, strict=True))
            for name, row in zip(names, analysis.correlation, strict=True)
        },
        correlated_pairs=pairs,
        modes=dict(zip(names, analysis.modes, strict=True)),
        mass_below=dict(zip(names, analysis.mass_below, strict=True)),
        mass_above=dict(zip(names, analysis.mass_above, strict=True)),
        at_bound={
            name: list(verdict.at_bound)
            for name, verdict in zip(names, analysis.verdicts, strict=True)
        },
        overlap=dict(zip(names, analysis.overlap, strict=True)),
        bins=analysis.bins,
        directions=[
            Direction(
                contraction=direction.contraction,
                loadings=dict(zip(names, direction.loadings, strict=True)),
                scales=scales,
            )
            for direction in analysis.directions
        ],
        hidden_ridges=analysis.hidden_ridges,
        draws_per_parameter=analysis.draws_per_parameter,
        warnings=_sample_warnings(analysis, names, method),
    )
    practical: dict[str, PracticalVerdict] = {}
    for name, verdict in zip(names, analysis.verdicts, strict=True):
        contraction = section.contraction[name]
        reasons = []
        if analysis.size == "too_few":
            reasons.append(
                f"Sample: effective size {analysis.effective_size:.3g}, below "
                f"p + 1 = {len(names) + 1} for "
                f"{_plural(len(names), 'parameter')}."
            )
        if verdict.low_contraction:
            reasons.append(
                f"Sample: contraction {contraction:.2f}, below "
                f"{method.contraction_threshold:g}."
            )
        reasons.extend(
            f"Sample: correlation {r:.3f} with {names[k]}."
            for k, r in verdict.correlated_with
        )
        if verdict.modes >= 2:
            reasons.append(f"Sample: its marginal has {verdict.modes} modes.")
        if verdict.verdict == "determined":
            reasons = [f"Sample: contraction {contraction:.2f}, no strong pair."]
        reasons.extend(_sample_sides(name, verdict.at_bound, section))
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
