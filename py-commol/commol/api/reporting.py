"""Structured reports of a calibrated model."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NotRequired, TypedDict, cast

import numpy as np

from commol.api.inference.intervals import member_statistics
from commol.api.inference.predictive import Predictive
from commol.api.results import SimulationResults, sample
from commol.api.simulation import Simulation
from commol.api.windows import windowed_totals
from commol.context._windows import window_end, window_index
from commol.context.calibration import (
    CalibrationProblem,
    CalibrationResult,
    ObservedDataPoint,
    calibrated_values,
)
from commol.context._model.report import CategorySplitEntry, InitialAmountEntry
from commol.context.constants import CalibrationParameterType
from commol.context.model import Model
from commol.context.inference.conformal import PredictionIntervals
from commol.context.inference.identifiability import IdentifiabilityReport
from commol.context.inference.sample import ParameterSample, ParameterSummary
from commol.context.report_entry import Interval, entry_interval


class ParameterEntry(TypedDict):
    """Reported value of one parameter."""

    value: float
    calibrated: bool
    interval: NotRequired[Interval]


class AppliedParametersReport(TypedDict):
    """
    Every parameter of a calibrated model, and the sections of its kind.

    An aggregate model adds ``initial_amounts`` and ``category_splits``;
    another kind adds the sections its structure gives
    (`ModelStructure.report_sections`).
    """

    confidence_level: NotRequired[float]
    parameters: dict[str, ParameterEntry]
    initial_amounts: NotRequired[dict[str, InitialAmountEntry]]
    category_splits: NotRequired[dict[str, CategorySplitEntry]]


def _parameter_entry(
    entry_id: str,
    value: float,
    calibrated_ids: set[str],
    statistics: Mapping[str, ParameterSummary] | None,
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


def _sample_statistics(
    sample: ParameterSample | None,
    confidence_level: float | None,
) -> Mapping[str, ParameterSummary] | None:
    """The statistics of each parameter of a sample, at a confidence level."""
    if sample is None:
        return None
    if confidence_level is None:
        raise ValueError("A report of a sample needs a confidence_level.")
    return sample.summary(confidence_level)


def applied_parameters_report(
    model: Model,
    result: CalibrationResult | Mapping[str, float],
    problem: CalibrationProblem | None = None,
    sample: ParameterSample | None = None,
    confidence_level: float | None = None,
) -> AppliedParametersReport:
    """
    Report every parameter of a calibrated model, and the values of its kind.

    Each entry carries the value the model holds and whether calibration set
    it, so entries the calibration left alone appear alongside the fitted ones.
    The structure of the model adds the sections of its kind
    (`ModelStructure.report_sections`); an aggregate model adds
    ``initial_amounts`` and ``category_splits``.

    Parameters that are not a single number, such as a formula or a
    series, are omitted: they are inputs rather than fitted values.
    ``factor`` and ``noise`` parameters have no place in the model, so they are
    reported from `problem` when it is given.

    ``initial_amounts`` gives each bin and each overridden compartment of an
    aggregate model its ``amount`` the engine builds and the ``unit`` of its
    bin, so bins of different units are reported side by side; no row is a
    share of a total. A bin left uncalibrated keeps a ``None`` amount as its
    "needs calibration" marker even when overrides touch some of its
    compartments. ``category_splits`` gives each category its ``fraction`` of
    the group it splits, and that ``group``: the categories a conditional
    stratification is conditioned on, empty when it splits every bin.

    With a `sample` each entry gains an ``interval`` holding the weighted
    statistics of its draws (`ParameterSample.summary`). Every entry is in
    the unit it was calibrated in: a parameter in its own, an amount in the
    unit of its bin, and a category split as a fraction of its group.

    Parameters
    ----------
    model : Model
        The model carrying the applied values.
    result : CalibrationResult | Mapping
        The calibration result, or a mapping of parameter id to value such as
        one draw of a sample.
    problem : CalibrationProblem | None, optional
        The problem the result came from, used to report factor and noise
        parameters.
    sample : ParameterSample | None, optional
        Draws of the parameters, such as the sample of an uncertainty method.
    confidence_level : float | None, optional
        Confidence level of the intervals; required with a sample.

    Returns
    -------
    AppliedParametersReport
        Key ``parameters`` and the sections of the kind (``initial_amounts``
        and ``category_splits`` for an aggregate model), preceded by
        ``confidence_level`` when there is one.

    Raises
    ------
    ValueError
        If a sample is given without a confidence level.
    """
    calibrated = calibrated_values(result)
    parameter_statistics = _sample_statistics(sample, confidence_level)
    calibrated_ids = set(calibrated)

    parameters: dict[str, ParameterEntry] = {
        parameter.id: _parameter_entry(
            parameter.id, parameter.value, calibrated_ids, parameter_statistics
        )
        for parameter in model.parameters
        if isinstance(parameter.value, (int, float))
    }
    for observation_id in _observation_parameter_ids(problem):
        if observation_id in calibrated:
            parameters[observation_id] = _parameter_entry(
                observation_id,
                calibrated[observation_id],
                calibrated_ids,
                parameter_statistics,
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


def _observation_parameter_ids(problem: CalibrationProblem | None) -> list[str]:
    """Ids of the problem's factor and noise parameters, which describe the
    observations, in declaration order."""
    if problem is None:
        return []
    return [
        parameter.id
        for parameter in problem.parameters
        if parameter.parameter_type
        in (CalibrationParameterType.FACTOR, CalibrationParameterType.NOISE)
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


class SeriesIntervalsReport(TypedDict):
    """The prediction intervals of one series."""

    positions: list[float]
    prediction: list[float]
    lower: list[float]
    upper: list[float]


class PredictionIntervalsReport(TypedDict):
    """Prediction intervals at a level, by series, and the unbounded ones."""

    level: float
    series: dict[str, SeriesIntervalsReport]
    unbounded: dict[str, list[float]]


class SimulationReport(TypedDict):
    """
    Observed data, compartment series and accumulators of a run, and the
    prediction intervals when the report has them.
    """

    observed: dict[str, ObservedSeries]
    series: dict[str, ReportLeaf]
    accumulators: dict[str, AccumulatorReport]
    prediction_intervals: NotRequired[PredictionIntervalsReport]


def _reduce(
    reduce_fn: Callable[[SimulationResults], float | list[float]],
    results: SimulationResults,
    ensemble_runs: "Sequence[SimulationResults] | Predictive | None",
    confidence_level: float | None,
) -> ReportLeaf:
    """
    One reported value, as a quantity or as the spread of that quantity.

    A single run yields the quantity itself. An ensemble yields the same
    quantity computed per member and reduced to statistics, so both layouts
    differ only in what sits at the leaves.
    """
    if ensemble_runs is None or confidence_level is None:
        return reduce_fn(results)
    if isinstance(ensemble_runs, Predictive):
        runs, weights = ensemble_runs.runs(), ensemble_runs.weights
    else:
        runs, weights = list(ensemble_runs), None
    if not runs:
        return reduce_fn(results)
    members = np.asarray([reduce_fn(run) for run in runs], dtype=float)
    return member_statistics(members, confidence_level, weights)


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
    its position (`AxisUnits.window_index`). ``weights``, ``outputs``
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
    ensemble_runs: "Sequence[SimulationResults] | Predictive | None" = None,
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
    ensemble_runs : Sequence[SimulationResults] | Predictive | None, optional
        One results mapping per member, such as the replicates of a stochastic
        engine, or a `Predictive`, whose draws carry their weights.
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
    ensemble_runs: "Sequence[SimulationResults] | Predictive | None" = None,
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
    ensemble_runs : Sequence[SimulationResults] | Predictive | None, optional
        One results mapping per member, such as the replicates of a stochastic
        engine, or a `Predictive`, whose draws carry their weights.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.

    Returns
    -------
    dict[str, AccumulatorReport]
        One entry per accumulator.

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
    ensemble_runs: "Sequence[SimulationResults] | Predictive | None" = None,
    confidence_level: float | None = None,
    prediction_intervals: PredictionIntervals | None = None,
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
    ensemble_runs : Sequence[SimulationResults] | Predictive | None, optional
        One results mapping per member, such as the replicates of a stochastic
        engine, or a `Predictive`, whose draws carry their weights.
    confidence_level : float | None, optional
        Confidence level of the reported intervals.
    prediction_intervals : PredictionIntervals | None, optional
        Prediction intervals of a conformal method
        (``ConformalResult.intervals``), reported as they are.

    Returns
    -------
    SimulationReport
        Keys ``observed``, ``series`` and ``accumulators``, then
        ``prediction_intervals`` when they are given.

    Raises
    ------
    ValueError
        If the results have no value at a position the report reads.
    """
    accumulator_ids = list(accumulators)
    report: SimulationReport = {
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
    if prediction_intervals is not None:
        report["prediction_intervals"] = cast(
            "PredictionIntervalsReport", prediction_intervals.model_dump()
        )
    return report


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


class IdentifiabilityReportData(TypedDict):
    """
    The identifiability of each parameter, and the evidence of each method.

    ``sections`` maps ``structural``, ``collinearity``, ``sample`` and
    ``starts`` to the list of their sections, as plain data.
    """

    parameters: dict[str, VerdictsEntry]
    sections: dict[str, list[dict[str, object]]]


def identifiability_report(report: IdentifiabilityReport) -> IdentifiabilityReportData:
    """
    Report the identifiability of each parameter as plain data.

    Parameters
    ----------
    report : IdentifiabilityReport
        The result of ``Calibrator.identifiability``, or of
        `IdentifiabilityReport.merge`.

    Returns
    -------
    IdentifiabilityReportData
        ``parameters``: each parameter, in problem order, with its structural
        and practical verdicts; ``sections``: the evidence of each method that
        ran.
    """
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
    sections: dict[str, list[dict[str, object]]] = {
        "structural": [section.model_dump() for section in report.structural],
        "collinearity": [section.model_dump() for section in report.collinearity],
        "sample": [section.model_dump() for section in report.sample],
        "starts": [section.model_dump() for section in report.starts],
    }
    return {"parameters": parameters, "sections": sections}
