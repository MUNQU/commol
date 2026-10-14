"""Structured reports of a calibrated model."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NotRequired, TypedDict, cast

import numpy as np

from commol.api.probabilistic.intervals import member_statistics
from commol.api.results import SimulationResults, sample
from commol.api.simulation import Simulation
from commol.api.windows import window_end, window_index, windowed_totals
from commol.context.calibration import (
    CalibrationProblem,
    CalibrationResult,
    ObservedDataPoint,
    calibrated_values,
)
from commol.context._model.report import InitialConditionEntry
from commol.context.constants import CalibrationParameterType
from commol.context.model import Model
from commol.context.probabilistic_calibration import (
    ParameterSetStatistics,
    ProbabilisticCalibrationResult,
)
from commol.context.report_entry import Interval, entry_interval


class ParameterEntry(TypedDict):
    """Reported value of one parameter."""

    value: float
    calibrated: bool
    interval: NotRequired[Interval]


class AppliedParametersReport(TypedDict):
    """
    Every parameter of a calibrated model, and the sections of its kind.

    A compartmental model adds ``initial_conditions``; another kind adds the
    sections its structure gives (`ModelStructure.report_sections`).
    """

    confidence_level: NotRequired[float]
    parameters: dict[str, ParameterEntry]
    initial_conditions: NotRequired[dict[str, InitialConditionEntry]]


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
    interval = entry_interval(entry_id, statistics)
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
    Report every parameter of a calibrated model, and the values of its kind.

    Each entry carries the value the model holds and whether calibration set
    it, so entries the calibration left alone appear alongside the fitted ones.
    The structure of the model adds the sections of its kind
    (`ModelStructure.report_sections`); a compartmental model adds
    ``initial_conditions``.

    Parameters that are not a single number, such as a formula or a
    series, are omitted: they are inputs rather than fitted values.
    ``factor`` parameters have no place in the model, so they are reported from
    `problem` when it is given.

    Initial conditions of a compartmental model carry the effective head
    ``value`` the engine builds
    from the uniform expansion plus the declared compartment overrides, its
    ``fraction`` as that head count over the declared population size, and the
    ``group`` it belongs to: ``population`` for bins and unconditional
    stratifications, and the conditioning categories for a conditional one.
    ``N`` is the declared population size, with fraction 1.0. A bin left
    uncalibrated keeps a ``None`` fraction and ``None`` value as its "needs
    calibration" marker even when overrides touch some of its compartments.

    For a probabilistic result each entry gains an ``interval`` holding the
    ensemble spread. A parameter is in the unit it was calibrated in. An
    initial-condition entry is in the fraction the row reports: a conditional
    stratification's category is calibrated as a group fraction but reported
    over the whole population, so its interval is rescaled by the group's share
    of the population and brackets the reported fraction.

    Parameters
    ----------
    model : Model
        The model carrying the applied values.
    result : CalibrationResult | ProbabilisticCalibrationResult | Mapping
        The calibration outcome, or a mapping of parameter id to value such as
        one ensemble member's parameters.
    problem : CalibrationProblem | None, optional
        The problem the result came from, used to report factor parameters.
    parameter_statistics : Mapping[str, ParameterSetStatistics] | None, optional
        Ensemble spread per entry. Taken from `result` when it is a
        probabilistic result and this is not given.
    confidence_level : float | None, optional
        Confidence level the intervals describe. Taken from `result` when it is
        a probabilistic result and this is not given.

    Returns
    -------
    AppliedParametersReport
        Key ``parameters`` and the sections of the kind (``initial_conditions``
        for a compartmental model), preceded by ``confidence_level`` when there
        is one.
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
    for factor_id in _factor_ids(problem):
        if factor_id in calibrated:
            parameters[factor_id] = _parameter_entry(
                factor_id, calibrated[factor_id], calibrated_ids, parameter_statistics
            )

    report: dict[str, object] = {}
    # `confidence_level` leads the report, so it is written first.
    if confidence_level is not None:
        report["confidence_level"] = confidence_level
    report["parameters"] = parameters
    report.update(
        model.structure.report_sections(model, calibrated_ids, parameter_statistics)
    )
    return cast(AppliedParametersReport, report)


def _factor_ids(problem: CalibrationProblem | None) -> list[str]:
    """Ids of the problem's factor parameters, in declaration order."""
    if problem is None:
        return []
    return [
        parameter.id
        for parameter in problem.parameters
        if parameter.parameter_type == CalibrationParameterType.FACTOR
    ]


type ReportLeaf = float | list[float] | dict[str, float | list[float]]


class ObservedSeries(TypedDict):
    """One observed series on the reported window axis."""

    values: list[float | None]
    weights: NotRequired[list[float | None]]
    outputs: NotRequired[list[str]]
    factor_id: NotRequired[str]


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
    reduce_fn: Callable[[SimulationResults], float | list[float]],
    results: SimulationResults,
    ensemble_runs: Sequence[SimulationResults] | None,
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


def observed_report(
    observed_data: Iterable[ObservedDataPoint],
    window: float,
    num_windows: int,
) -> dict[str, ObservedSeries]:
    """
    Report the calibration targets on a window axis.

    Each series becomes one array of `num_windows` entries, padded with null
    where there is no observation, so it lines up index for index with the
    windowed accumulator series. An observation goes to the window that holds
    its position (:func:`commol.window_index`). ``weights``, ``outputs``
    and ``factor_id`` appear only on series that use them; a series whose
    weights are all 1.0 omits them.

    Parameters
    ----------
    observed_data : Iterable[ObservedDataPoint]
        The observations to report.
    window : float
        Length of one window on the axis.
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
            point.label,
            {
                "values": [None] * num_windows,
                "weights": [None] * num_windows,
            },
        )
        index = window_index(point.at, window)
        if not 0 <= index < num_windows:
            continue
        entry["values"][index] = point.value
        entry["weights"][index] = point.weight
        if point.outputs:
            entry["outputs"] = list(point.outputs)
        if point.factor_id:
            entry["factor_id"] = point.factor_id

    for entry in series.values():
        if all(weight in (None, 1.0) for weight in entry["weights"]):
            del entry["weights"]
    return series


def series_report(
    simulation: Simulation,
    results: SimulationResults,
    at: Sequence[float],
    accumulators: Iterable[str] = (),
    ensemble_runs: Sequence[SimulationResults] | None = None,
    confidence_level: float | None = None,
) -> dict[str, ReportLeaf]:
    """
    Report every compartment series, sampled at the given positions.

    Outputs belonging to the named accumulators are excluded, since they are
    reported separately by :func:`accumulators_report`.

    Parameters
    ----------
    simulation : Simulation
        The simulation the results came from.
    results : SimulationResults
        Results of a run, with their positions.
    at : Sequence[float]
        Positions the series are sampled at; each must be a position of the
        results.
    accumulators : Iterable[str], optional
        Accumulator ids whose outputs are left out.
    ensemble_runs : Sequence[SimulationResults] | None, optional
        One results mapping per ensemble member.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    dict[str, ReportLeaf]
        One entry per compartment output, sorted by name.

    Raises
    ------
    ValueError
        If the results have no value at a position of `at`.
    """
    excluded = {
        name
        for accumulator_id in accumulators
        for name in simulation.outputs_for(accumulator_id)
    }
    return {
        name: _reduce(
            lambda run, output=name: sample(run[output], run.positions, at),
            results,
            ensemble_runs,
            confidence_level,
        )
        for name in sorted(results)
        if name not in excluded
    }


def accumulators_report(
    simulation: Simulation,
    results: SimulationResults,
    accumulators: Iterable[str],
    at: Sequence[float],
    window: float,
    num_windows: int,
    ensemble_runs: Sequence[SimulationResults] | None = None,
    confidence_level: float | None = None,
) -> dict[str, AccumulatorReport]:
    """
    Report both readings of each accumulator.

    ``windowed`` is the per-window increment the loss compares observations
    against, so it stays on the window axis whatever positions `at` samples
    and lines up with :func:`observed_report`. It holds the windows that end
    within the results. ``cumulative`` follows `at` like the compartment
    series do. Both are given as a total and per output.

    Parameters
    ----------
    simulation : Simulation
        The simulation the results came from.
    results : SimulationResults
        Results of a run, with their positions.
    accumulators : Iterable[str]
        Accumulator ids to report.
    at : Sequence[float]
        Positions the cumulative series are sampled at.
    window : float
        Length of one window on the axis.
    num_windows : int
        Number of windows on the reported axis.
    ensemble_runs : Sequence[SimulationResults] | None, optional
        One results mapping per ensemble member.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    dict[str, AccumulatorReport]
        One entry per accumulator that has outputs.

    Raises
    ------
    ValueError
        If the results have no value at a position of `at`, or at the start
        or the end of a window that ends within them.
    """
    ends = [window_end(index, window) for index in range(num_windows)]

    def windowed(series: Sequence[float], positions: Sequence[float]) -> list[float]:
        last = positions[-1] if positions else 0.0
        return windowed_totals(
            series, positions, window, [end for end in ends if end <= last]
        )

    block: dict[str, AccumulatorReport] = {}
    for accumulator_id in accumulators:
        outputs = simulation.outputs_for(accumulator_id)
        if not outputs:
            continue

        def leaf(
            reduce_fn: Callable[[SimulationResults], float | list[float]],
        ) -> ReportLeaf:
            return _reduce(reduce_fn, results, ensemble_runs, confidence_level)

        entry: AccumulatorReport = {
            "total": leaf(lambda run, ids=outputs: sum(run[name][-1] for name in ids)),
            "windowed": {
                "total": leaf(
                    lambda run, acc=accumulator_id: windowed(
                        simulation.total_series(run, [acc]), run.positions
                    )
                ),
                "by_output": {
                    name: leaf(
                        lambda run, output=name: windowed(run[output], run.positions)
                    )
                    for name in outputs
                },
            },
            "cumulative": {
                "total": leaf(
                    lambda run, acc=accumulator_id: sample(
                        simulation.total_series(run, [acc]), run.positions, at
                    )
                ),
                "by_output": {
                    name: leaf(
                        lambda run, output=name: sample(run[output], run.positions, at)
                    )
                    for name in outputs
                },
            },
        }
        block[accumulator_id] = entry
    return block


def simulation_report(
    simulation: Simulation,
    results: SimulationResults,
    at: Sequence[float],
    window: float,
    num_windows: int,
    observed_data: Iterable[ObservedDataPoint] = (),
    accumulators: Iterable[str] = (),
    ensemble_runs: Sequence[SimulationResults] | None = None,
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
    results : SimulationResults
        Results of a run, with their positions.
    at : Sequence[float]
        Positions the series and cumulative accumulators are sampled at.
    window : float
        Length of one window on the axis.
    num_windows : int
        Number of windows on the observed and windowed axes.
    observed_data : Iterable[ObservedDataPoint], optional
        The calibration targets to report alongside the run.
    accumulators : Iterable[str], optional
        Accumulator ids to report.
    ensemble_runs : Sequence[SimulationResults] | None, optional
        One results mapping per ensemble member.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    SimulationReport
        Keys ``observed``, ``series`` and ``accumulators``.

    Raises
    ------
    ValueError
        If the results have no value at a position the report reads.
    """
    accumulator_ids = list(accumulators)
    return {
        "observed": observed_report(observed_data, window, num_windows),
        "series": series_report(
            simulation,
            results,
            at,
            accumulator_ids,
            ensemble_runs,
            confidence_level,
        ),
        "accumulators": accumulators_report(
            simulation,
            results,
            accumulator_ids,
            at,
            window,
            num_windows,
            ensemble_runs,
            confidence_level,
        ),
    }
