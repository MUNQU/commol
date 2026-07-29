"""Structured reports of a calibrated model."""

from collections.abc import Mapping
from typing import NotRequired, TypedDict

from commol.context.calibration import (
    CalibrationProblem,
    CalibrationResult,
    calibrated_values,
)
from commol.context.constants import CalibrationParameterType
from commol.context.model import Model
from commol.context.probabilistic_calibration import (
    ParameterSetStatistics,
    ProbabilisticCalibrationResult,
)

WHOLE_POPULATION_GROUP = "population"


class Interval(TypedDict):
    """Ensemble spread of one reported entry."""

    mean: float
    median: float
    ci_lower: float
    ci_upper: float
    min: float
    max: float
    std: float


class ParameterEntry(TypedDict):
    """Reported value of one parameter."""

    value: float
    calibrated: bool
    interval: NotRequired[Interval]


class InitialConditionEntry(TypedDict):
    """Reported value of one initial condition."""

    group: str
    fraction: float | None
    value: float | None
    calibrated: bool
    interval: NotRequired[Interval]


class AppliedParametersReport(TypedDict):
    """Every parameter and initial condition of a calibrated model."""

    confidence_level: NotRequired[float]
    parameters: dict[str, ParameterEntry]
    initial_conditions: dict[str, InitialConditionEntry]


def _group_label(group: tuple[str, ...]) -> str:
    """Name of the subgroup a fraction is taken of."""
    return "+".join(group) if group else WHOLE_POPULATION_GROUP


def _interval(
    entry_id: str,
    statistics: Mapping[str, ParameterSetStatistics] | None,
) -> Interval | None:
    """Ensemble spread of one entry, or None when there is no ensemble."""
    stats = (statistics or {}).get(entry_id)
    if stats is None:
        return None
    return {
        "mean": stats.mean,
        "median": stats.median,
        "ci_lower": stats.percentile_lower,
        "ci_upper": stats.percentile_upper,
        "min": stats.min,
        "max": stats.max,
        "std": stats.std,
    }


def _parameter_entry(
    entry_id: str,
    value: float,
    calibrated_ids: set[str],
    statistics: Mapping[str, ParameterSetStatistics] | None,
) -> ParameterEntry:
    """One reported parameter, with its interval when there is one."""
    entry: ParameterEntry = {
        "value": value,
        "calibrated": entry_id in calibrated_ids,
    }
    interval = _interval(entry_id, statistics)
    if interval is not None:
        entry["interval"] = interval
    return entry


def _initial_condition_entry(
    entry_id: str,
    group: str,
    fraction: float | None,
    group_population: float,
    calibrated_ids: set[str],
    statistics: Mapping[str, ParameterSetStatistics] | None,
) -> InitialConditionEntry:
    """One reported initial condition, with its interval when there is one."""
    entry: InitialConditionEntry = {
        "group": group,
        "fraction": fraction,
        "value": None if fraction is None else fraction * group_population,
        "calibrated": entry_id in calibrated_ids,
    }
    interval = _interval(entry_id, statistics)
    if interval is not None:
        entry["interval"] = interval
    return entry


def _ensemble_details(
    result: CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float],
    statistics: Mapping[str, ParameterSetStatistics] | None,
    confidence_level: float | None,
) -> tuple[Mapping[str, ParameterSetStatistics] | None, float | None]:
    """Fill statistics and confidence level from a probabilistic result."""
    if not isinstance(result, ProbabilisticCalibrationResult):
        return statistics, confidence_level
    if statistics is None:
        statistics = result.selected_ensemble.parameter_statistics
    if confidence_level is None:
        confidence_level = result.confidence_level
    return statistics, confidence_level


def applied_parameters_report(
    model: Model,
    result: CalibrationResult | ProbabilisticCalibrationResult | Mapping[str, float],
    problem: CalibrationProblem | None = None,
    parameter_statistics: Mapping[str, ParameterSetStatistics] | None = None,
    confidence_level: float | None = None,
) -> AppliedParametersReport:
    """
    Report every parameter and initial condition of a calibrated model.

    Each entry carries the value the model holds and whether calibration set
    it, so entries the calibration left alone appear alongside the fitted ones.

    Parameters that are not a single number, such as a formula or a
    time-series, are omitted: they are inputs rather than fitted values.
    ``scale`` parameters have no place in the model, so they are reported from
    `problem` when it is given.

    Initial conditions carry the ``fraction`` the model was built from, the
    head ``value`` it works out to, and the ``group`` that fraction is taken
    of: ``population`` for bins and unconditional stratifications, and the
    conditioning categories for a conditional one. ``N`` is the population the
    fractions of the whole-population group apply to.

    For a probabilistic result each entry gains an ``interval`` holding the
    ensemble spread, expressed in the unit the entry was calibrated in, which
    for an initial condition is its fraction rather than its head count.

    Parameters
    ----------
    model : Model
        The model carrying the applied values.
    result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
        The calibration outcome, or a mapping of parameter id to value such as
        one ensemble member's parameters.
    problem : CalibrationProblem | None, optional
        The problem the result came from, used to report scale parameters.
    parameter_statistics : Mapping[str, ParameterSetStatistics] | None, optional
        Ensemble spread per entry. Taken from `result` when it is a
        probabilistic result and this is not given.
    confidence_level : float | None, optional
        Confidence level the intervals describe. Taken from `result` when it is
        a probabilistic result and this is not given.

    Returns
    -------
    AppliedParametersReport
        Keys ``parameters`` and ``initial_conditions``, preceded by
        ``confidence_level`` when there is one.
    """
    calibrated = calibrated_values(result)
    parameter_statistics, confidence_level = _ensemble_details(
        result, parameter_statistics, confidence_level
    )
    calibrated_ids = set(calibrated)

    parameters: dict[str, ParameterEntry] = {
        parameter.id: _parameter_entry(
            parameter.id, parameter.value, calibrated_ids, parameter_statistics
        )
        for parameter in model.parameters
        if isinstance(parameter.value, (int, float))
    }
    for scale_id in _scale_ids(problem):
        if scale_id in calibrated:
            parameters[scale_id] = _parameter_entry(
                scale_id, calibrated[scale_id], calibrated_ids, parameter_statistics
            )

    initial_conditions = _initial_conditions_report(
        model, calibrated_ids, parameter_statistics
    )
    if confidence_level is None:
        return {
            "parameters": parameters,
            "initial_conditions": initial_conditions,
        }
    # `confidence_level` leads the report, so it is written first.
    return {
        "confidence_level": confidence_level,
        "parameters": parameters,
        "initial_conditions": initial_conditions,
    }


def _scale_ids(problem: CalibrationProblem | None) -> list[str]:
    """Ids of the problem's scale parameters, in declaration order."""
    if problem is None:
        return []
    return [
        parameter.id
        for parameter in problem.parameters
        if parameter.parameter_type == CalibrationParameterType.SCALE
    ]


def _initial_conditions_report(
    model: Model,
    calibrated_ids: set[str],
    parameter_statistics: Mapping[str, ParameterSetStatistics] | None,
) -> dict[str, InitialConditionEntry]:
    """Every bin fraction and stratification category, with its head count."""
    initial_conditions_block = model.population.initial_conditions
    population_size = float(initial_conditions_block.population_size)

    report: dict[str, InitialConditionEntry] = {
        "N": {
            "group": WHOLE_POPULATION_GROUP,
            "fraction": 1.0,
            "value": population_size,
            "calibrated": False,
        }
    }
    for bin_fraction in initial_conditions_block.bin_fractions:
        report[bin_fraction.bin] = _initial_condition_entry(
            bin_fraction.bin,
            WHOLE_POPULATION_GROUP,
            bin_fraction.fraction,
            population_size,
            calibrated_ids,
            parameter_statistics,
        )
    for stratification in initial_conditions_block.stratification_fractions:
        group = model.get_conditioning_categories(stratification.stratification)
        group_population = model.subgroup_population(group)
        for fraction in stratification.fractions:
            report[fraction.category] = _initial_condition_entry(
                fraction.category,
                _group_label(group),
                fraction.fraction,
                group_population,
                calibrated_ids,
                parameter_statistics,
            )
    return report
