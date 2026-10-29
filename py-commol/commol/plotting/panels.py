"""The data of the panels of a figure, computed without matplotlib.

A panel is one series: its name and unit, the line of the model, the band of
replicates or of draws, the observed values and the prediction intervals.
Each result builds the panels of its figures with the helpers here
(`commol._simulation.panels`, `commol.uncertainty.panels`), and a comparison
reads each result through `PanelSource`. `commol.plotting.figures` draws the
panels; nothing here imports matplotlib or a result.
"""

import itertools
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from commol._model.outputs import outputs_for
from commol._problem.problem import ObservedDataPoint
from commol.commol_rs import _commol_rs as commol_rs

if TYPE_CHECKING:
    from commol._data.intervals import PredictionIntervalsData
    from commol._model.model import Model

logger = logging.getLogger("commol.plotting")

MODEL_BAND = "model band\n(no observation noise),\nnot a prediction interval"
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


class PanelOutputs:
    """The outputs of results, and the bins and accumulators of their model."""

    def __init__(self, outputs: Iterable[str], model: "Model | None") -> None:
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


class PanelSource(Protocol):
    """
    What a comparison reads of one result: the `plot` accessor of a run, of
    replicates or of a predictive.

    Each result builds its own panels; `compared_panels` and
    `compared_observation_panels` set them side by side.
    """

    def _panel_model(self) -> "Model | None":
        """The model that names the series."""
        ...

    def _panel_outputs(self) -> PanelOutputs:
        """The outputs of the result, and the bins and accumulators of its
        model."""
        ...

    def _series_panels(
        self,
        confidence_level: float,
        observed_data: list[ObservedDataPoint],
        series: list[str],
    ) -> list[Panel]:
        """The panels of `series`, as the figure of the result draws them."""
        ...

    def _window_trace(
        self,
        outputs: PanelOutputs,
        series_id: str,
        confidence_level: float,
        window: float,
    ) -> tuple[Trace, Band | None]:
        """The gain of a series over each complete window, and its band."""
        ...

    def _observation_panels(
        self, confidence_level: float, observed_data: list[ObservedDataPoint]
    ) -> list[ObservationPanel]:
        """The panels of the observed series, as the figure of the result
        draws them."""
        ...


class Plotted(Protocol):
    """A result with a `plot` accessor that a comparison reads."""

    @property
    def plot(self) -> PanelSource:
        """The figures of the result."""
        ...


def with_intervals(
    panels: list[Panel], intervals: "PredictionIntervalsData | None"
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
                prediction=Trace(
                    list(series.positions), list(series.prediction), "Prediction"
                ),
            )
        )
    return with_series


def compared_panels(
    results: Mapping[str, "Plotted"],
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    series: Sequence[str] | None = None,
    window: float | None = None,
) -> list[ComparedPanel]:
    """
    The panels of results drawn together: the line and the band of each
    result in each series, and the observations once.

    Without `window`, the line and the band of a result are those of its own
    figure with `observed_data` (`PanelSource._series_panels`). With
    `window`, they are the gain of the series over each complete window
    (`PanelSource._window_trace`). The first result names the series and
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
    sources = {label: result.plot for label, result in results.items()}
    outputs = _compared_outputs(sources)
    if window is not None:
        commol_rs.calibration.check_window(window)
    first = next(iter(results))
    selected = outputs[first].select(series)
    observed = list(observed_data or ())
    by_label = {
        label: _result_panels(
            source, outputs[label], confidence_level, observed, selected, window
        )
        for label, source in sources.items()
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
    results: Mapping[str, "Plotted"],
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint],
) -> list[ComparedObservationPanel]:
    """
    The panels of the observed series of results drawn together: each result
    read through the operator of each observation, and the observations once.

    The model and the band of a result are those of its own figure of
    observed series (`PanelSource._observation_panels`). The first result
    gives the title and the unit of each panel.

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
    sources = {label: result.plot for label, result in results.items()}
    _compared_outputs(sources)
    observed = list(observed_data)
    by_label = {
        label: source._observation_panels(confidence_level, observed)
        for label, source in sources.items()
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


def _compared_outputs(
    sources: Mapping[str, "PanelSource"],
) -> dict[str, PanelOutputs]:
    """
    The outputs of each result of a comparison, checked against the first.

    Raises
    ------
    ValueError
        If there are no results, or a result does not have the outputs of the
        first.
    """
    if not sources:
        raise ValueError("Give at least one result to compare.")
    outputs = {label: source._panel_outputs() for label, source in sources.items()}
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


def _result_panels(
    source: "PanelSource",
    outputs: PanelOutputs,
    confidence_level: float,
    observed_data: list[ObservedDataPoint],
    series: list[str],
    window: float | None,
) -> list[Panel]:
    """The panels of one result of a comparison, with or without windows."""
    if window is None:
        return source._series_panels(confidence_level, observed_data, series)
    by_series = group_observed(observed_data)
    return [
        outputs.panel(
            series_id,
            *source._window_trace(outputs, series_id, confidence_level, window),
            observed=observed_trace(by_series.get(series_id, [])),
        )
        for series_id in series
    ]


def _labelled(band: Band | None, label: str) -> Band | None:
    """A band whose legend starts with the label of its result."""
    if band is None:
        return None
    return replace(band, label=f"{label}: {band.label}")


def group_observed(
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


def observed_series(
    observed_data: Iterable[ObservedDataPoint],
) -> dict[str, list[ObservedDataPoint]]:
    """
    The observations of each label, in order of position.

    Raises
    ------
    ValueError
        If there are no observations.
    """
    by_label = group_observed(observed_data)
    if not by_label:
        raise ValueError(
            "A figure of the observed series needs at least one observation."
        )
    return by_label


def observation_positions(observed: list[ObservedDataPoint]) -> list[float]:
    """The position of each observation."""
    return [point.at for point in observed]


def observation_panel(
    outputs: PanelOutputs,
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
            observation_positions(observed),
            [point.value for point in observed],
            "Observed",
        ),
        band=band,
    )


def observed_trace(observed: list[ObservedDataPoint]) -> Trace | None:
    """The observed values of a series, or None when it has none."""
    if not observed:
        return None
    return Trace(
        [point.at for point in observed],
        [point.value for point in observed],
        "Observed",
    )


def cumulative_observed(
    observed: list[ObservedDataPoint], factors: Mapping[str, float]
) -> Trace | None:
    """
    The running total of the observations of a series, each divided by its
    scale and by the value of its factor when `factors` has it
    (`model_multiplier`).
    """
    if not observed:
        return None
    return Trace(
        [point.at for point in observed],
        list(
            itertools.accumulate(
                point.value / model_multiplier(point, factors) for point in observed
            )
        ),
        "Observed (Cumulative)",
    )


def model_multiplier(
    point: ObservedDataPoint | None,
    factors: Mapping[str, float],
) -> float:
    """
    The number an observation multiplies the model by, as its operator in the
    engine reads it: its scale times the value of its factor. A factor that
    `factors` lacks counts as 1 (`warn_missing_factors` names it); no
    observation is 1.
    """
    if point is None:
        return 1.0
    factor_id = point.factor_id
    if factor_id is None or factor_id not in factors:
        return point.scale
    return point.scale * factors[factor_id]


def warn_missing_factors(
    observed: Iterable[ObservedDataPoint], factors: Mapping[str, float]
) -> None:
    """
    Log one warning for each factor of `observed` that `factors` lacks, in
    order of first appearance: `model_multiplier` counts it as 1.
    """
    missing = dict.fromkeys(
        point.factor_id
        for point in observed
        if point.factor_id is not None and point.factor_id not in factors
    )
    for factor_id in missing:
        logger.warning(
            f"Factor parameter '{factor_id}' has no value in calibration_result; "
            "the figure counts it as 1."
        )


def reads_each_observation(observed: list[ObservedDataPoint]) -> bool:
    """
    Whether the model of a series is read at each observation: when one has
    a window, or the observations differ in outputs, factor or scale.

    Otherwise one operator reads every observation, and the model line covers
    every position of the results.
    """
    operators = {
        (tuple(point.outputs or ()), point.factor_id, point.scale) for point in observed
    }
    return any(point.window for point in observed) or len(operators) > 1
