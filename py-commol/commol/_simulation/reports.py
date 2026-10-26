"""
The blocks of the reports: plain data, built for the methods that report.

`SimulationResults.report`, `SimulationReplicates.report`,
`Predictive.report`, `Model.applied_parameters` and
`IdentifiabilityReport.report` call the private functions of this module; the
key order of each block is part of the contract.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import NotRequired, TypedDict

import numpy as np

from commol._model.model import Model
from commol._problem.problem import (
    ObservedDataPoint,
)
from commol._model.outputs import outputs_for
from commol._simulation.results import SimulationResults
from commol._simulation.series import _sample, total_series
from commol._simulation.windows import _windowed_totals
from commol.commol_rs import _commol_rs as commol_rs
from commol._data.intervals import PredictionIntervalsData
from commol._statistics.members import spread


type _ReportLeaf = float | list[float] | dict[str, float | list[float]]


class ObservedSeries(TypedDict):
    """One observed series on the reported window axis."""

    values: list[float | None]
    weights: NotRequired[list[float | None]]
    outputs: NotRequired[list[str]]
    factor_id: NotRequired[str]


class WindowedAccumulator(TypedDict):
    """One reading of an accumulator, as a total and per output."""

    total: _ReportLeaf
    by_output: dict[str, _ReportLeaf]


class AccumulatorReport(TypedDict):
    """Both readings of one accumulator."""

    total: _ReportLeaf
    windowed: WindowedAccumulator
    cumulative: WindowedAccumulator


class SeriesIntervalsReport(TypedDict):
    """The prediction intervals of one series."""

    positions: list[float]
    prediction: list[float]
    lower: list[float]
    upper: list[float]


class PredictionIntervalsReport(TypedDict):
    """Prediction intervals at a confidence level, by series, and the unbounded
    ones."""

    confidence_level: float
    series: dict[str, SeriesIntervalsReport]
    unbounded: dict[str, list[float]]


class SimulationReport(TypedDict):
    """
    Observed data, compartment series and accumulators of a run, and the
    prediction intervals when the report has them.
    """

    observed: dict[str, ObservedSeries]
    series: dict[str, _ReportLeaf]
    accumulators: dict[str, AccumulatorReport]
    prediction_intervals: NotRequired[PredictionIntervalsReport]


@dataclass(frozen=True)
class _ReportRuns:
    """
    The runs a report reads, and the model whose outputs they hold.

    `runs` holds at least one run, and the outputs of the first name the
    series. Without a confidence level the report reads the one run in
    `runs`; with one, each leaf is computed per member of `runs` and reduced
    to statistics with `weights`, so both layouts differ only at the leaves.
    """

    model: Model | None
    runs: Sequence[SimulationResults]
    confidence_level: float | None = None
    weights: Sequence[float] | None = None

    def leaf(
        self, reduce_fn: Callable[[SimulationResults], float | list[float]]
    ) -> _ReportLeaf:
        """One reported value, as a quantity or as the spread of that quantity."""
        if self.confidence_level is None:
            return reduce_fn(self.runs[0])
        members = np.asarray([reduce_fn(run) for run in self.runs], dtype=float)
        return spread(members, self.confidence_level, self.weights)

    def outputs_for(self, id: str) -> list[str]:
        """The outputs of a bin or accumulator, from the structure of the model."""
        model = self.model
        if model is None:
            raise ValueError(
                f"The results know no model, so they cannot report '{id}': "
                "report the results of Simulation.run, run_at or replicates."
            )
        return outputs_for(model, id)


def _observed_report(
    observed_data: Iterable[ObservedDataPoint],
    window: float,
    num_windows: int,
) -> dict[str, ObservedSeries]:
    """
    Report the calibration targets on a window axis.

    Each series becomes one array of `num_windows` entries, padded with null
    where there is no observation, so it lines up index for index with the
    windowed accumulator series. An observation goes to the window that holds
    its position (`window_index` of the engine, as `AxisUnits.window_index`
    gives it). ``weights``, ``outputs``
    and ``factor_id`` appear only on series that use them; a series whose
    weights are all 1.0 omits them.
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
        index = commol_rs.calibration.window_index(point.at, window)
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


def _series_report(
    runs: _ReportRuns,
    at: Sequence[float],
    accumulators: Iterable[str],
) -> dict[str, _ReportLeaf]:
    """
    Report every compartment series, sampled at the given positions.

    Outputs of the named accumulators are left out, since
    `_accumulators_report` reports them. Each entry is one output, sorted by
    name.
    """
    excluded = {
        name
        for accumulator_id in accumulators
        for name in runs.outputs_for(accumulator_id)
    }
    return {
        name: runs.leaf(
            lambda run, output=name: _sample(run[output], run.positions, at)
        )
        for name in sorted(runs.runs[0])
        if name not in excluded
    }


def _accumulators_report(
    runs: _ReportRuns,
    accumulators: Iterable[str],
    at: Sequence[float],
    window: float,
    num_windows: int,
) -> dict[str, AccumulatorReport]:
    """
    Report both readings of each accumulator.

    ``windowed`` is the per-window increment the loss compares observations
    against, so it stays on the window axis whatever positions `at` samples
    and lines up with `_observed_report`. It holds the windows that end
    within each run. ``cumulative`` follows `at` like the compartment series
    do. Both are given as a total and per output.
    """
    ends = [
        commol_rs.calibration.window_bounds(index, window)[1]
        for index in range(num_windows)
    ]

    def windowed(series: Sequence[float], positions: Sequence[float]) -> list[float]:
        last = positions[-1] if positions else 0.0
        return _windowed_totals(
            series, positions, window, [end for end in ends if end <= last]
        )

    block: dict[str, AccumulatorReport] = {}
    for accumulator_id in accumulators:
        outputs = runs.outputs_for(accumulator_id)
        entry: AccumulatorReport = {
            "total": runs.leaf(
                lambda run, ids=outputs: sum(run[name][-1] for name in ids)
            ),
            "windowed": {
                "total": runs.leaf(
                    lambda run, ids=outputs: windowed(
                        total_series(run, ids), run.positions
                    )
                ),
                "by_output": {
                    name: runs.leaf(
                        lambda run, output=name: windowed(run[output], run.positions)
                    )
                    for name in outputs
                },
            },
            "cumulative": {
                "total": runs.leaf(
                    lambda run, ids=outputs: _sample(
                        total_series(run, ids), run.positions, at
                    )
                ),
                "by_output": {
                    name: runs.leaf(
                        lambda run, output=name: _sample(run[output], run.positions, at)
                    )
                    for name in outputs
                },
            },
        }
        block[accumulator_id] = entry
    return block


def _simulation_report(
    runs: _ReportRuns,
    at: Sequence[float],
    window: float,
    num_windows: int,
    observed_data: Iterable[ObservedDataPoint],
    accumulators: Iterable[str],
    prediction_intervals: PredictionIntervalsData | None,
) -> SimulationReport:
    """
    The observed data, compartment series and accumulators of some runs.

    `SimulationResults.report`, `SimulationReplicates.report` and
    `Predictive.report` build their block here. The three blocks share one
    window axis; with a confidence level every leaf becomes a ``{mean,
    median, ci_lower, ci_upper, min, max}`` block computed across the runs,
    and the layout is otherwise unchanged. The key order is part of the
    contract.
    """
    accumulator_ids = list(accumulators)
    report: SimulationReport = {
        "observed": _observed_report(observed_data, window, num_windows),
        "series": _series_report(runs, at, accumulator_ids),
        "accumulators": _accumulators_report(
            runs, accumulator_ids, at, window, num_windows
        ),
    }
    if prediction_intervals is not None:
        report["prediction_intervals"] = {
            "confidence_level": prediction_intervals.confidence_level,
            "series": {
                label: {
                    "positions": list(series.positions),
                    "prediction": list(series.prediction),
                    "lower": list(series.lower),
                    "upper": list(series.upper),
                }
                for label, series in prediction_intervals.series.items()
            },
            "unbounded": {
                label: list(positions)
                for label, positions in prediction_intervals.unbounded.items()
            },
        }
    return report
