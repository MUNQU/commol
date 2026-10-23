"""The data of the panels of a figure, computed without matplotlib.

A panel is one series: its name and unit, the line of the model, the band of
replicates or of draws, the observed values and the prediction intervals.
`commol.api.plotting.figures` draws the panels; nothing here imports
matplotlib.
"""

import itertools
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from commol.api.inference.intervals import member_statistics
from commol.api.results import (
    SimulationReplicates,
    SimulationResults,
    outputs_for,
    sample,
    total_series,
)
from commol.api.windows import windowed_totals
from commol.context._windows import check_window, window_ends
from commol.context.calibration import CalibrationResult, ObservedDataPoint
from commol.context.inference.noise import NoNoise

if TYPE_CHECKING:
    from commol.api.inference.predictive import Predictive, SeriesBand
    from commol.context.inference.conformal import PredictionIntervals
    from commol.context.model import Model

logger = logging.getLogger(__name__)

_MODEL_BAND = "model band\n(no observation noise),\nnot a prediction interval"
"""What the band of a panel of observed series is, in its legend."""


@dataclass(frozen=True)
class Trace:
    """
    Values at positions, drawn as a line or as points.

    Attributes
    ----------
    positions : list[float]
        Positions on the axis.
    values : list[float]
        One value per position.
    label : str
        The label of the legend.
    """

    positions: list[float]
    values: list[float]
    label: str


@dataclass(frozen=True)
class Band:
    """
    A band between two values at each position.

    Attributes
    ----------
    positions : list[float]
        Positions on the axis.
    lower, upper : list[float]
        The ends of the band at each position.
    label : str
        The label of the legend.
    """

    positions: list[float]
    lower: list[float]
    upper: list[float]
    label: str


@dataclass(frozen=True)
class Panel:
    """
    One series of a figure.

    Attributes
    ----------
    series : str
        The id of the series: an output name, or a bin or accumulator id. The
        `Axes` of the panel has it as its label.
    name : str
        The title of the panel.
    unit : str
        The unit of the values, or ``""`` when they have none.
    line : Trace
        The model: one run, the mean of the replicates or the median of the
        draws.
    band : Band | None
        The band of the replicates or of the draws.
    observed : Trace | None
        The observed values, drawn as points.
    intervals : Band | None
        The bounded prediction intervals of a conformal method.
    prediction : Trace | None
        The point prediction of those intervals.
    """

    series: str
    name: str
    unit: str
    line: Trace
    band: Band | None = None
    observed: Trace | None = None
    intervals: Band | None = None
    prediction: Trace | None = None


@dataclass(frozen=True)
class ComparedPanel:
    """
    One series of a figure that compares results.

    Attributes
    ----------
    series : str
        The id of the series. The `Axes` of the panel has it as its label.
    name : str
        The title of the panel.
    unit : str
        The unit of the values, or ``""`` when they have none.
    lines : list[Trace]
        The line of each result, labelled with its key, in the order of the
        results.
    bands : list[Band | None]
        The band of each result, in the same order: of its replicates or of
        its draws, or None for a run.
    observed : Trace | None
        The observed values, drawn once.
    """

    series: str
    name: str
    unit: str
    lines: list[Trace]
    bands: list[Band | None]
    observed: Trace | None = None


type Compared = SimulationResults | SimulationReplicates | Predictive


@dataclass(frozen=True)
class ObservationPanel:
    """
    One observed series, with the model read only at its observations.

    Attributes
    ----------
    series : str
        The label of the observations. The `Axes` of the panel has it as its
        label.
    name : str
        The title of the panel.
    unit : str
        The unit of the values, or ``""`` when they have none.
    windows : list[float | None]
        The window of each observation, or None for a value read at its
        position.
    model : Trace
        The model value of each observation, read through its own operator,
        at its position.
    observed : Trace
        The observed values.
    band : Band | None
        The band of the replicates or of the draws at each observation.
    """

    series: str
    name: str
    unit: str
    windows: list[float | None]
    model: Trace
    observed: Trace
    band: Band | None = None


@dataclass(frozen=True)
class ComparedObservationPanel:
    """
    One observed series of a figure that compares results, with each result
    read only at its observations.

    Attributes
    ----------
    series : str
        The label of the observations. The `Axes` of the panel has it as its
        label.
    name : str
        The title of the panel.
    unit : str
        The unit of the values, or ``""`` when they have none.
    windows : list[float | None]
        The window of each observation, or None for a value read at its
        position.
    models : list[Trace]
        The model value of each observation in each result, labelled with
        its key, in the order of the results.
    bands : list[Band | None]
        The band of each result at each observation, in the same order: of
        its replicates or of its draws, or None for a run.
    observed : Trace
        The observed values, drawn once.
    """

    series: str
    name: str
    unit: str
    windows: list[float | None]
    models: list[Trace]
    bands: list[Band | None]
    observed: Trace


class _Outputs:
    """The outputs of results, and the bins and accumulators of their model."""

    def __init__(self, outputs: Iterable[str], model: "Model | None"):
        self.outputs = list(outputs)
        self.model = model

    def select(self, series: Sequence[str] | None) -> list[str]:
        """The series to draw, each checked to resolve to outputs."""
        if series is None:
            return list(self.outputs)
        for series_id in series:
            self.components(series_id)
        return list(series)

    def components(self, series_id: str) -> list[str]:
        """The outputs whose sum is the series: itself, or those of an id."""
        if series_id in self.outputs:
            return [series_id]
        if self.model is None:
            raise ValueError(
                f"'{series_id}' is not an output of the results, and the results "
                "have no model to read the outputs of a bin or accumulator from. "
                f"Outputs: {self.outputs}"
            )
        return outputs_for(self.model, series_id)

    def label(self, series_id: str) -> tuple[str, str]:
        """The name and unit of a series, as the kind of the model names it."""
        if self.model is None:
            return series_id, ""
        name, unit = self.model.structure.series_label(self.model, series_id)
        return name, unit or ""

    def panel(
        self,
        series_id: str,
        line: Trace,
        band: Band | None = None,
        observed: Trace | None = None,
    ) -> Panel:
        """The panel of a series, named as the kind of the model names it."""
        name, unit = self.label(series_id)
        return Panel(
            series=series_id,
            name=name,
            unit=unit,
            line=line,
            band=band,
            observed=observed,
        )


def run_panels(
    results: SimulationResults,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    calibration_result: CalibrationResult | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of one run: the model of each series and its observations.

    Parameters
    ----------
    results : SimulationResults
        The run.
    observed_data : Iterable[ObservedDataPoint] | None
        Observations, each on the panel of its label.
    calibration_result : CalibrationResult | None
        The values of the factors of the observations.
    series : Sequence[str] | None
        The series, or every output when None.

    Returns
    -------
    list[Panel]
        One panel per series, in order.
    """
    outputs = _Outputs(results, results.model)
    factors = calibration_result.best_parameters if calibration_result else {}
    by_series = _group_observed(observed_data)
    panels = []
    for series_id in outputs.select(series):
        observed = by_series.get(series_id, [])
        positions, values = _model_line(results, outputs, series_id, observed, factors)
        panels.append(
            outputs.panel(
                series_id,
                Trace(positions, values, "Simulation"),
                observed=_observed(observed),
            )
        )
    return panels


def replicate_panels(
    replicates: SimulationReplicates,
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    calibration_result: CalibrationResult | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of replicates: the mean of each series and the band of
    `confidence_level` of the replicates, each reduced to the series first
    (`member_statistics`).
    """
    mean = replicates.mean()
    outputs = _Outputs(mean, replicates.model)
    factors = calibration_result.best_parameters if calibration_result else {}
    by_series = _group_observed(observed_data)
    panels = []
    for series_id in outputs.select(series):
        observed = by_series.get(series_id, [])
        positions, values = _model_line(mean, outputs, series_id, observed, factors)
        statistics = member_statistics(
            [
                _model_line(run, outputs, series_id, observed, factors)[1]
                for run in replicates
            ],
            confidence_level,
        )
        panels.append(
            outputs.panel(
                series_id,
                Trace(positions, values, f"Mean of {len(replicates)} replicates"),
                band=Band(
                    positions,
                    statistics["ci_lower"],
                    statistics["ci_upper"],
                    f"{confidence_level:.0%} of the replicates",
                ),
                observed=_observed(observed),
            )
        )
    return panels


def predictive_panels(
    predictive: "Predictive",
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of a predictive: the weighted median of the draws of each
    series and their band of `confidence_level`.

    Each draw is read through the operator of the observations of the series,
    as the loss reads it (`_predictive_band`).
    """
    outputs = _Outputs(
        predictive.output_names, predictive.source.simulation.model_definition
    )
    by_series = _group_observed(observed_data)
    panels = []
    for series_id in outputs.select(series):
        observed = by_series.get(series_id, [])
        band = _predictive_band(
            predictive, outputs, series_id, observed, confidence_level
        )
        panels.append(
            outputs.panel(
                series_id,
                Trace(band.positions, band.median, "Median of the draws"),
                band=_draws_band(band),
                observed=_observed(observed),
            )
        )
    return panels


def cumulative_panels(
    results: SimulationResults,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    calibration_result: CalibrationResult | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of the running total of each series of a run, and of its
    observations, each divided by the fitted value of its factor.
    """
    outputs = _Outputs(results, results.model)
    factors = calibration_result.best_parameters if calibration_result else {}
    by_series = _group_observed(observed_data)
    return [
        outputs.panel(
            series_id,
            Trace(
                results.positions,
                list(
                    itertools.accumulate(
                        total_series(results, outputs.components(series_id))
                    )
                ),
                "Simulation (Cumulative)",
            ),
            observed=_cumulative_observed(by_series.get(series_id, []), factors),
        )
        for series_id in outputs.select(series)
    ]


def predictive_cumulative_panels(
    predictive: "Predictive",
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of the band of the running total of each series of a
    predictive, and of its observations, each divided by the weighted median
    of its factor.
    """
    outputs = _Outputs(
        predictive.output_names, predictive.source.simulation.model_definition
    )
    factors = {
        name: values[0] for name, values in predictive.sample.quantiles([0.5]).items()
    }
    by_series = _group_observed(observed_data)
    panels = []
    for series_id in outputs.select(series):
        band = predictive.series_band(
            outputs.components(series_id), confidence_level, cumulative=True
        )
        panels.append(
            outputs.panel(
                series_id,
                Trace(band.positions, band.median, "Median of the draws (cumulative)"),
                band=_draws_band(band),
                observed=_cumulative_observed(by_series.get(series_id, []), factors),
            )
        )
    return panels


def run_observation_panels(
    results: SimulationResults,
    observed_data: Iterable[ObservedDataPoint],
    calibration_result: CalibrationResult | None = None,
) -> list[ObservationPanel]:
    """
    The panels of the observed series of one run, one per label, in the order
    of their first observation.

    The model value of each observation is read through its own operator, as
    the loss reads it: the sum of its outputs at its position, or its
    increment over its window, times the value of its factor in
    `calibration_result`.

    Raises
    ------
    ValueError
        If there are no observations, or the results have no value at the
        position of an observation or at the start of its window.
    KeyError
        If an observation names an output the results do not have.
    """
    outputs = _outputs_of(results)
    factors = calibration_result.best_parameters if calibration_result else {}
    return [
        _observation_panel(
            outputs,
            label,
            observed,
            Trace(
                _positions(observed),
                _at_observations(results, observed, factors),
                "Simulation",
            ),
        )
        for label, observed in _observed_series(observed_data).items()
    ]


def replicate_observation_panels(
    replicates: SimulationReplicates,
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint],
    calibration_result: CalibrationResult | None = None,
) -> list[ObservationPanel]:
    """
    The panels of the observed series of replicates: the mean of the
    replicates and the band of `confidence_level` across them, each
    replicate read through the operator of each observation first
    (`member_statistics`).
    """
    mean = replicates.mean()
    outputs = _Outputs(mean, replicates.model)
    factors = calibration_result.best_parameters if calibration_result else {}
    panels = []
    for label, observed in _observed_series(observed_data).items():
        statistics = member_statistics(
            [_at_observations(run, observed, factors) for run in replicates],
            confidence_level,
        )
        panels.append(
            _observation_panel(
                outputs,
                label,
                observed,
                Trace(
                    _positions(observed),
                    _at_observations(mean, observed, factors),
                    f"Mean of {len(replicates)} replicates",
                ),
                Band(
                    _positions(observed),
                    statistics["ci_lower"],
                    statistics["ci_upper"],
                    f"{confidence_level:.0%} of the replicates:\n{_MODEL_BAND}",
                ),
            )
        )
    return panels


def predictive_observation_panels(
    predictive: "Predictive",
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint],
) -> list[ObservationPanel]:
    """
    The panels of the observed series of a predictive: the weighted median of
    the draws and their band of `confidence_level`, read through
    ``Predictive.at_observations(..., NoNoise())``, the values the loss
    compares.
    """
    by_label = _observed_series(observed_data)
    band = predictive.at_observations(
        [point for observed in by_label.values() for point in observed], NoNoise()
    ).band(confidence_level)
    outputs = _outputs_of(predictive)
    panels = []
    start = 0
    for label, observed in by_label.items():
        end = start + len(observed)
        panels.append(
            _observation_panel(
                outputs,
                label,
                observed,
                Trace(
                    _positions(observed), band.median[start:end], "Median of the draws"
                ),
                Band(
                    _positions(observed),
                    band.lower[start:end],
                    band.upper[start:end],
                    f"{confidence_level:.0%} of the draws:\n{_MODEL_BAND}",
                ),
            )
        )
        start = end
    return panels


def with_intervals(
    panels: list[Panel], intervals: "PredictionIntervals | None"
) -> list[Panel]:
    """
    The panels with the prediction intervals of their series, by label: the
    band of the bounded intervals and the point prediction.
    """
    if intervals is None:
        return panels
    with_series = []
    for panel in panels:
        series = intervals.series.get(panel.series)
        if series is None:
            with_series.append(panel)
            continue
        bounded = [
            (at, lower, upper)
            for at, lower, upper in zip(
                series.positions, series.lower, series.upper, strict=True
            )
            if math.isfinite(lower) and math.isfinite(upper)
        ]
        with_series.append(
            replace(
                panel,
                intervals=Band(
                    [at for at, _, _ in bounded],
                    [lower for _, lower, _ in bounded],
                    [upper for _, _, upper in bounded],
                    f"{intervals.confidence_level:.0%} prediction interval",
                ),
                prediction=Trace(series.positions, series.prediction, "Prediction"),
            )
        )
    return with_series


def compared_panels(
    results: Mapping[str, Compared],
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    series: Sequence[str] | None = None,
    window: float | None = None,
) -> list[ComparedPanel]:
    """
    The panels of results drawn together: the line and the band of each
    result in each series, and the observations once.

    Without `window`, the line and the band of a result are those of its own
    figure (`run_panels`, `replicate_panels` or `predictive_panels`) with
    `observed_data`. With `window`, they are the gain of the series over each
    complete window (`_windowed`). The first result names the series and
    gives the title and the unit of each panel.

    Parameters
    ----------
    results : Mapping[str, SimulationResults | SimulationReplicates | Predictive]
        The results, by the label of their lines.
    confidence_level : float
        The level of the band of replicates and of draws.
    observed_data : Iterable[ObservedDataPoint] | None
        Observations, each on the panel of its label.
    series : Sequence[str] | None
        The series, or every output of the first result when None.
    window : float | None
        The length of a window, or None to read each position.

    Returns
    -------
    list[ComparedPanel]
        One panel per series, in order.

    Raises
    ------
    ValueError
        If there are no results, a result does not have the outputs of the
        first, or `window` is not positive.
    """
    outputs = _compared_outputs(results)
    if window is not None:
        check_window(window)
    first = next(iter(results))
    selected = outputs[first].select(series)
    observed = list(observed_data or ())
    by_label = {
        label: _result_panels(
            result, outputs[label], confidence_level, observed, selected, window
        )
        for label, result in results.items()
    }
    return [
        ComparedPanel(
            series=panel.series,
            name=panel.name,
            unit=panel.unit,
            lines=[
                replace(by_label[label][index].line, label=label) for label in results
            ],
            bands=[_labelled(by_label[label][index].band, label) for label in results],
            observed=panel.observed,
        )
        for index, panel in enumerate(by_label[first])
    ]


def compared_observation_panels(
    results: Mapping[str, Compared],
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint],
) -> list[ComparedObservationPanel]:
    """
    The panels of the observed series of results drawn together: each result
    read through the operator of each observation, and the observations once.

    The model and the band of a result are those of its own figure of
    observed series (`run_observation_panels`,
    `replicate_observation_panels` or `predictive_observation_panels`). The
    first result gives the title and the unit of each panel.

    Parameters
    ----------
    results : Mapping[str, SimulationResults | SimulationReplicates | Predictive]
        The results, by the label of their lines.
    confidence_level : float
        The level of the band of replicates and of draws.
    observed_data : Iterable[ObservedDataPoint]
        The observations: one panel per label, in the order of its first
        observation.

    Returns
    -------
    list[ComparedObservationPanel]
        One panel per observed series, in order.

    Raises
    ------
    ValueError
        If there are no results, a result does not have the outputs of the
        first, there are no observations, or a result has no value at the
        position of an observation or at the start of its window.
    """
    _compared_outputs(results)
    observed = list(observed_data)
    by_label = {
        label: _result_observation_panels(result, confidence_level, observed)
        for label, result in results.items()
    }
    first = next(iter(results))
    return [
        ComparedObservationPanel(
            series=panel.series,
            name=panel.name,
            unit=panel.unit,
            windows=panel.windows,
            models=[
                replace(by_label[label][index].model, label=label) for label in results
            ],
            bands=[_labelled(by_label[label][index].band, label) for label in results],
            observed=panel.observed,
        )
        for index, panel in enumerate(by_label[first])
    ]


def model_of(result: Compared) -> "Model | None":
    """The model of a run, of replicates or of the source of a predictive."""
    if isinstance(result, SimulationResults | SimulationReplicates):
        return result.model
    return result.source.simulation.model_definition


def _outputs_of(result: Compared) -> _Outputs:
    """The outputs of a run, of replicates or of a predictive."""
    if isinstance(result, SimulationReplicates):
        return _Outputs(result.mean(), result.model)
    if isinstance(result, SimulationResults):
        return _Outputs(result, result.model)
    return _Outputs(result.output_names, model_of(result))


def _compared_outputs(results: Mapping[str, Compared]) -> dict[str, _Outputs]:
    """
    The outputs of each result of a comparison, checked against the first.

    Raises
    ------
    ValueError
        If there are no results, or a result does not have the outputs of the
        first.
    """
    if not results:
        raise ValueError("Give at least one result to compare.")
    outputs = {label: _outputs_of(result) for label, result in results.items()}
    (first, reference), *others = outputs.items()
    for label, other in others:
        missing = [name for name in reference.outputs if name not in other.outputs]
        extra = [name for name in other.outputs if name not in reference.outputs]
        if missing or extra:
            differences = [f"it lacks {missing}"] if missing else []
            differences += [f"it also has {extra}"] if extra else []
            raise ValueError(
                f"'{label}' does not have the outputs of '{first}': "
                f"{' and '.join(differences)}."
            )
    return outputs


def _result_observation_panels(
    result: Compared,
    confidence_level: float,
    observed_data: list[ObservedDataPoint],
) -> list[ObservationPanel]:
    """The panels of the observed series of one result of a comparison."""
    if isinstance(result, SimulationReplicates):
        return replicate_observation_panels(result, confidence_level, observed_data)
    if isinstance(result, SimulationResults):
        return run_observation_panels(result, observed_data)
    return predictive_observation_panels(result, confidence_level, observed_data)


def _result_panels(
    result: Compared,
    outputs: _Outputs,
    confidence_level: float,
    observed_data: list[ObservedDataPoint],
    series: list[str],
    window: float | None,
) -> list[Panel]:
    """The panels of one result of a comparison, with or without windows."""
    if window is None:
        if isinstance(result, SimulationReplicates):
            return replicate_panels(
                result, confidence_level, observed_data, series=series
            )
        if isinstance(result, SimulationResults):
            return run_panels(result, observed_data, series=series)
        return predictive_panels(result, confidence_level, observed_data, series)
    by_series = _group_observed(observed_data)
    return [
        outputs.panel(
            series_id,
            *_windowed(result, outputs, series_id, confidence_level, window),
            observed=_observed(by_series.get(series_id, [])),
        )
        for series_id in series
    ]


def _windowed(
    result: Compared,
    outputs: _Outputs,
    series_id: str,
    confidence_level: float,
    window: float,
) -> tuple[Trace, Band | None]:
    """
    The gain of a series over each complete window, as `windowed_totals`
    reads it: of a run; of the mean of replicates, with the band of the gains
    of the replicates; or the weighted median and the band of the gains of
    the draws of a predictive, read through the observation operator.
    """
    components = outputs.components(series_id)
    ends = window_ends(window, result.positions[-1])
    if isinstance(result, SimulationReplicates):
        gains = [
            windowed_totals(total_series(run, components), run.positions, window, ends)
            for run in result
        ]
        statistics = member_statistics(gains, confidence_level)
        return Trace(ends, statistics["mean"], "Mean of the replicates"), Band(
            ends,
            statistics["ci_lower"],
            statistics["ci_upper"],
            f"{confidence_level:.0%} of the replicates",
        )
    if isinstance(result, SimulationResults):
        gains = windowed_totals(
            total_series(result, components), result.positions, window, ends
        )
        return Trace(ends, gains, "Simulation"), None
    points = [
        ObservedDataPoint(
            at=end, outputs=components, label=series_id, value=0.0, window=window
        )
        for end in ends
    ]
    band = result.at_observations(points, NoNoise()).band(confidence_level)
    return Trace(band.positions, band.median, "Median of the draws"), _draws_band(band)


def _labelled(band: Band | None, label: str) -> Band | None:
    """A band whose legend starts with the label of its result."""
    if band is None:
        return None
    return replace(band, label=f"{label}: {band.label}")


def _group_observed(
    observed_data: Iterable[ObservedDataPoint] | None,
) -> dict[str, list[ObservedDataPoint]]:
    """The observations of each series, by label, in order of position."""
    grouped: dict[str, list[ObservedDataPoint]] = defaultdict(list)
    for point in observed_data or ():
        grouped[point.label].append(point)
    return {
        label: sorted(points, key=lambda point: point.at)
        for label, points in grouped.items()
    }


def _observed_series(
    observed_data: Iterable[ObservedDataPoint],
) -> dict[str, list[ObservedDataPoint]]:
    """
    The observations of each label, in order of position.

    Raises
    ------
    ValueError
        If there are no observations.
    """
    by_label = _group_observed(observed_data)
    if not by_label:
        raise ValueError(
            "A figure of the observed series needs at least one observation."
        )
    return by_label


def _positions(observed: list[ObservedDataPoint]) -> list[float]:
    """The position of each observation."""
    return [point.at for point in observed]


def _at_observations(
    results: SimulationResults,
    observed: list[ObservedDataPoint],
    factors: Mapping[str, float],
) -> list[float]:
    """
    The model value of each observation in one run, through its own
    operator: the sum of its outputs, its window and its factor.
    """
    return [
        _read(results, point.observed_outputs, point)
        * _factor(point.factor_id, factors)
        for point in observed
    ]


def _observation_panel(
    outputs: _Outputs,
    label: str,
    observed: list[ObservedDataPoint],
    model: Trace,
    band: Band | None = None,
) -> ObservationPanel:
    """The panel of an observed series, named as the kind of the model names it."""
    name, unit = outputs.label(label)
    return ObservationPanel(
        series=label,
        name=name,
        unit=unit,
        windows=[point.window for point in observed],
        model=model,
        observed=Trace(
            _positions(observed), [point.value for point in observed], "Observed"
        ),
        band=band,
    )


def _observed(observed: list[ObservedDataPoint]) -> Trace | None:
    """The observed values of a series, or None when it has none."""
    if not observed:
        return None
    return Trace(
        [point.at for point in observed],
        [point.value for point in observed],
        "Observed",
    )


def _cumulative_observed(
    observed: list[ObservedDataPoint], factors: Mapping[str, float]
) -> Trace | None:
    """
    The running total of the observations of a series, each divided by the
    value of its factor when `factors` has it.
    """
    if not observed:
        return None
    return Trace(
        [point.at for point in observed],
        list(
            itertools.accumulate(
                point.value / factors[point.factor_id]
                if point.factor_id is not None and point.factor_id in factors
                else point.value
                for point in observed
            )
        ),
        "Observed (Cumulative)",
    )


def _draws_band(band: "SeriesBand") -> Band:
    """The central interval of the draws of a series band."""
    return Band(
        band.positions,
        band.lower,
        band.upper,
        f"{band.confidence_level:.0%} of the draws",
    )


def _factor(factor_id: str | None, factors: Mapping[str, float]) -> float:
    """The value of a factor, or 1 for none or one the factors lack."""
    if factor_id is None:
        return 1.0
    if factor_id not in factors:
        logger.warning(
            f"Factor parameter '{factor_id}' not found in calibration result; "
            "leaving model predictions unscaled."
        )
        return 1.0
    return factors[factor_id]


def _reads_each_observation(observed: list[ObservedDataPoint]) -> bool:
    """
    Whether the model of a series is read at each observation: when one has
    a window, or the observations differ in outputs or factor.

    Otherwise one operator reads every observation, and the model line covers
    every position of the results.
    """
    operators = {(tuple(point.outputs or ()), point.factor_id) for point in observed}
    return any(point.window for point in observed) or len(operators) > 1


def _read(
    results: SimulationResults, outputs: list[str], point: ObservedDataPoint
) -> float:
    """
    The sum of `outputs` of one run as `point` reads it: at its position, or
    its increment over its window (`windowed_totals`).
    """
    series = total_series(results, outputs)
    if point.window:
        return windowed_totals(series, results.positions, point.window, [point.at])[0]
    return sample(series, results.positions, [point.at])[0]


def _model_line(
    results: SimulationResults,
    outputs: _Outputs,
    series_id: str,
    observed: list[ObservedDataPoint],
    factors: Mapping[str, float],
) -> tuple[list[float], list[float]]:
    """
    The positions and values of the model of one series in one run.

    When the observations of the series do not share one operator with no
    window, each observation is read through its own operator, as the loss
    reads it: the sum of its outputs at its position, or its increment over
    its window (`windowed_totals`), times its factor. Otherwise the sum of the
    outputs of their operator at every position of the results, times its
    factor.

    Raises
    ------
    ValueError
        If the results have no value at the position of an observation read
        on its own or at the start of its window.
    """
    if _reads_each_observation(observed):
        return [point.at for point in observed], [
            _read(results, point.outputs or outputs.components(series_id), point)
            * _factor(point.factor_id, factors)
            for point in observed
        ]
    components = (
        observed[0].outputs
        if observed and observed[0].outputs
        else outputs.components(series_id)
    )
    factor = _factor(observed[0].factor_id if observed else None, factors)
    return list(results.positions), [
        value * factor for value in total_series(results, components)
    ]


def _predictive_band(
    predictive: "Predictive",
    outputs: _Outputs,
    series_id: str,
    observed: list[ObservedDataPoint],
    confidence_level: float,
) -> "SeriesBand":
    """
    The band of one series of a predictive.

    When the observations of the series do not share one operator with no
    window, each draw is read through the operator of each observation at its
    position, as the loss reads it. With one operator and a factor, each draw
    is read through it at every position of the predictive. Otherwise, the
    band of the sum of the outputs.
    """
    if _reads_each_observation(observed):
        points = [
            ObservedDataPoint(
                at=point.at,
                outputs=point.outputs or outputs.components(series_id),
                label=series_id,
                value=0.0,
                window=point.window,
                factor_id=point.factor_id,
            )
            for point in observed
        ]
        return predictive.at_observations(points, NoNoise()).band(confidence_level)
    components = (
        observed[0].outputs
        if observed and observed[0].outputs
        else outputs.components(series_id)
    )
    factor_id = observed[0].factor_id if observed else None
    if factor_id is None:
        return predictive.series_band(components, confidence_level)
    points = [
        ObservedDataPoint(
            at=at, outputs=components, label=series_id, value=0.0, factor_id=factor_id
        )
        for at in predictive.positions
    ]
    return predictive.at_observations(points, NoNoise()).band(confidence_level)
