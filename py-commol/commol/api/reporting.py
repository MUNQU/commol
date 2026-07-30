"""Structured reports of a calibrated model."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NotRequired, TypedDict

import numpy as np

from commol.api.probabilistic.intervals import member_statistics
from commol.api.simulation import Simulation
from commol.api.windows import windowed_totals
from commol.context.calibration import (
    CalibrationProblem,
    CalibrationResult,
    ObservedDataPoint,
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


type ReportLeaf = float | list[float] | dict[str, float | list[float]]


class ObservedSeries(TypedDict):
    """One observed series on the reported window axis."""

    values: list[float | None]
    weights: NotRequired[list[float | None]]
    compartments: NotRequired[list[str]]
    scale_id: NotRequired[str]


class WindowedAccumulator(TypedDict):
    """One reading of an accumulator, as a total and per output."""

    total: ReportLeaf
    by_output: dict[str, ReportLeaf]


class AccumulatorReport(TypedDict):
    """Both readings of one accumulator."""

    total: ReportLeaf
    windowed: WindowedAccumulator
    cumulative: WindowedAccumulator


class SimulationReport(TypedDict):
    """Observed data, compartment series and accumulators of a run."""

    observed: dict[str, ObservedSeries]
    series: dict[str, ReportLeaf]
    accumulators: dict[str, AccumulatorReport]


def _reduce(
    reduce_fn: Callable[[Mapping[str, Sequence[float]]], float | list[float]],
    results: Mapping[str, Sequence[float]],
    ensemble_runs: Sequence[Mapping[str, Sequence[float]]] | None,
    confidence_level: float | None,
) -> ReportLeaf:
    """
    One reported value, as a quantity or as the spread of that quantity.

    A single run yields the quantity itself. An ensemble yields the same
    quantity computed per member and reduced to statistics, so both layouts
    differ only in what sits at the leaves.
    """
    if not ensemble_runs or confidence_level is None:
        return reduce_fn(results)
    members = np.asarray([reduce_fn(run) for run in ensemble_runs], dtype=float)
    return member_statistics(members, confidence_level)


def sample(series: Sequence[float], steps: Iterable[int]) -> list[float]:
    """
    Read a series at the given steps.

    Compartment counts are instantaneous, so they are read at the step rather
    than averaged over the interval leading to it.

    Parameters
    ----------
    series : Sequence[float]
        A series indexed by simulation step.
    steps : Iterable[int]
        Steps to read.

    Returns
    -------
    list[float]
        One value per step.
    """
    return [series[step] for step in steps]


def observed_report(
    observed_data: Iterable[ObservedDataPoint],
    window_steps: int,
    num_windows: int,
) -> dict[str, ObservedSeries]:
    """
    Report the calibration targets on a window axis.

    Each series becomes one array of `num_windows` entries, padded with null
    where there is no observation, so it lines up index for index with the
    windowed accumulator series. ``weights``, ``compartments`` and ``scale_id``
    appear only on series that use them; a series whose weights are all 1.0
    omits them.

    Parameters
    ----------
    observed_data : Iterable[ObservedDataPoint]
        The observations to report.
    window_steps : int
        Length of one window, in steps.
    num_windows : int
        Number of windows on the reported axis.

    Returns
    -------
    dict[str, ObservedSeries]
        One entry per observed series.
    """
    series: dict[str, ObservedSeries] = {}
    for point in observed_data:
        entry: ObservedSeries = series.setdefault(
            point.compartment,
            {
                "values": [None] * num_windows,
                "weights": [None] * num_windows,
            },
        )
        window = max(point.step - 1, 0) // window_steps
        if not 0 <= window < num_windows:
            continue
        entry["values"][window] = point.value
        entry["weights"][window] = point.weight
        if point.compartments:
            entry["compartments"] = list(point.compartments)
        if point.scale_id:
            entry["scale_id"] = point.scale_id

    for entry in series.values():
        if all(weight in (None, 1.0) for weight in entry["weights"]):
            del entry["weights"]
    return series


def series_report(
    simulation: Simulation,
    results: Mapping[str, Sequence[float]],
    steps: Sequence[int],
    accumulators: Iterable[str] = (),
    ensemble_runs: Sequence[Mapping[str, Sequence[float]]] | None = None,
    confidence_level: float | None = None,
) -> dict[str, ReportLeaf]:
    """
    Report every compartment series, sampled at the given steps.

    Outputs belonging to the named accumulators are excluded, since they are
    reported separately by :func:`accumulators_report`.

    Parameters
    ----------
    simulation : Simulation
        The simulation the results came from.
    results : Mapping[str, Sequence[float]]
        Results in `dict_of_lists` form.
    steps : Sequence[int]
        Steps the series are sampled at.
    accumulators : Iterable[str], optional
        Accumulator ids whose outputs are left out.
    ensemble_runs : Sequence[Mapping[str, Sequence[float]]] | None, optional
        One results mapping per ensemble member.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    dict[str, ReportLeaf]
        One entry per compartment output, sorted by name.
    """
    excluded = {
        name
        for accumulator_id in accumulators
        for name in simulation.outputs_for(accumulator_id)
    }
    return {
        name: _reduce(
            lambda run, output=name: sample(run[output], steps),
            results,
            ensemble_runs,
            confidence_level,
        )
        for name in sorted(results)
        if name not in excluded
    }


def accumulators_report(
    simulation: Simulation,
    results: Mapping[str, Sequence[float]],
    accumulators: Iterable[str],
    steps: Sequence[int],
    window_steps: int,
    num_windows: int,
    ensemble_runs: Sequence[Mapping[str, Sequence[float]]] | None = None,
    confidence_level: float | None = None,
) -> dict[str, AccumulatorReport]:
    """
    Report both readings of each accumulator.

    ``windowed`` is the per-window increment the loss compares observations
    against, so it stays on the window axis whatever `steps` are sampled and
    lines up with :func:`observed_report`. ``cumulative`` follows `steps` like
    the compartment series do. Both are given as a total and per output.

    Parameters
    ----------
    simulation : Simulation
        The simulation the results came from.
    results : Mapping[str, Sequence[float]]
        Results in `dict_of_lists` form.
    accumulators : Iterable[str]
        Accumulator ids to report.
    steps : Sequence[int]
        Steps the cumulative series are sampled at.
    window_steps : int
        Length of one window, in steps.
    num_windows : int
        Number of windows on the reported axis.
    ensemble_runs : Sequence[Mapping[str, Sequence[float]]] | None, optional
        One results mapping per ensemble member.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    dict[str, AccumulatorReport]
        One entry per accumulator that has outputs.
    """
    anchors = [(window + 1) * window_steps for window in range(num_windows)]

    def windowed(series: Sequence[float]) -> list[float]:
        return windowed_totals(
            series, window_steps, [step for step in anchors if step < len(series)]
        )

    block: dict[str, AccumulatorReport] = {}
    for accumulator_id in accumulators:
        outputs = simulation.outputs_for(accumulator_id)
        if not outputs:
            continue

        def leaf(
            reduce_fn: Callable[[Mapping[str, Sequence[float]]], float | list[float]],
        ) -> ReportLeaf:
            return _reduce(reduce_fn, results, ensemble_runs, confidence_level)

        entry: AccumulatorReport = {
            "total": leaf(lambda run, ids=outputs: sum(run[name][-1] for name in ids)),
            "windowed": {
                "total": leaf(
                    lambda run, acc=accumulator_id: windowed(
                        simulation.total_series(run, [acc])
                    )
                ),
                "by_output": {
                    name: leaf(lambda run, output=name: windowed(run[output]))
                    for name in outputs
                },
            },
            "cumulative": {
                "total": leaf(
                    lambda run, acc=accumulator_id: sample(
                        simulation.total_series(run, [acc]), steps
                    )
                ),
                "by_output": {
                    name: leaf(lambda run, output=name: sample(run[output], steps))
                    for name in outputs
                },
            },
        }
        block[accumulator_id] = entry
    return block


def simulation_report(
    simulation: Simulation,
    results: Mapping[str, Sequence[float]],
    steps: Sequence[int],
    window_steps: int,
    num_windows: int,
    observed_data: Iterable[ObservedDataPoint] = (),
    accumulators: Iterable[str] = (),
    ensemble_runs: Sequence[Mapping[str, Sequence[float]]] | None = None,
    confidence_level: float | None = None,
) -> SimulationReport:
    """
    Report the observed data, compartment series and accumulators of a run.

    The three blocks share one window axis, so the observed values, the
    windowed accumulator increments and the sampled series can be read against
    each other. With `ensemble_runs` every leaf becomes a
    ``{mean, median, ci_lower, ci_upper, min, max}`` block computed across
    members, and the layout is otherwise unchanged.

    Callers add whatever envelope they need around these blocks, such as a run
    label, axis labels or domain-specific summary figures.

    Parameters
    ----------
    simulation : Simulation
        The simulation the results came from.
    results : Mapping[str, Sequence[float]]
        Results in `dict_of_lists` form.
    steps : Sequence[int]
        Steps the series and cumulative accumulators are sampled at.
    window_steps : int
        Length of one window, in steps.
    num_windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the run.
    accumulators : Iterable[str], optional
        Accumulator ids to report.
    ensemble_runs : Sequence[Mapping[str, Sequence[float]]] | None, optional
        One results mapping per ensemble member.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    SimulationReport
        Keys ``observed``, ``series`` and ``accumulators``.
    """
    accumulator_ids = list(accumulators)
    return {
        "observed": observed_report(observed_data, window_steps, num_windows),
        "series": series_report(
            simulation,
            results,
            steps,
            accumulator_ids,
            ensemble_runs,
            confidence_level,
        ),
        "accumulators": accumulators_report(
            simulation,
            results,
            accumulator_ids,
            steps,
            window_steps,
            num_windows,
            ensemble_runs,
            confidence_level,
        ),
    }
