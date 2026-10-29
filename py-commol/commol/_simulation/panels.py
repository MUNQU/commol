"""The data of the panels of a run and of replicates: each series, its band
and its observations, as `commol.plotting` draws them."""

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import itertools

from commol._fit.result import CalibrationResult
from commol._problem.problem import ObservedDataPoint
from commol._simulation.series import _sample, total_series
from commol._simulation.windows import _windowed_totals
from commol._statistics.members import spread
from commol.commol_rs import _commol_rs as commol_rs
from commol.plotting.panels import (
    MODEL_BAND,
    Band,
    ObservationPanel,
    Panel,
    PanelOutputs,
    Trace,
    cumulative_observed,
    model_multiplier,
    group_observed,
    observation_panel,
    observation_positions,
    observed_series,
    observed_trace,
    reads_each_observation,
    warn_missing_factors,
)

if TYPE_CHECKING:
    from commol._simulation.results import SimulationReplicates, SimulationResults


def run_panels(
    results: "SimulationResults",
    observed_data: Iterable[ObservedDataPoint] | None = None,
    calibration_result: CalibrationResult | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of one run: the model of each series and its observations.

    Parameters
    ----------
    results : "SimulationResults"
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
    outputs = PanelOutputs(results, results.model)
    selected = outputs.select(series)
    by_series = group_observed(observed_data)
    factors = _factors(calibration_result, by_series, selected)
    panels = []
    for series_id in selected:
        observed = by_series.get(series_id, [])
        positions, values = _model_line(results, outputs, series_id, observed, factors)
        panels.append(
            outputs.panel(
                series_id,
                Trace(positions, values, "Simulation"),
                observed=observed_trace(observed),
            )
        )
    return panels


def replicate_panels(
    replicates: "SimulationReplicates",
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
    outputs = PanelOutputs(mean, replicates.model)
    selected = outputs.select(series)
    by_series = group_observed(observed_data)
    factors = _factors(calibration_result, by_series, selected)
    panels = []
    for series_id in selected:
        observed = by_series.get(series_id, [])
        positions, values = _model_line(mean, outputs, series_id, observed, factors)
        statistics = spread(
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
                observed=observed_trace(observed),
            )
        )
    return panels


def cumulative_panels(
    results: "SimulationResults",
    observed_data: Iterable[ObservedDataPoint] | None = None,
    calibration_result: CalibrationResult | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of the running total of each series of a run, and of its
    observations, each divided by its scale and by the fitted value of its
    factor.
    """
    outputs = PanelOutputs(results, results.model)
    selected = outputs.select(series)
    by_series = group_observed(observed_data)
    factors = _factors(calibration_result, by_series, selected)
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
            observed=cumulative_observed(by_series.get(series_id, []), factors),
        )
        for series_id in selected
    ]


def run_observation_panels(
    results: "SimulationResults",
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
    outputs = PanelOutputs(results, results.model)
    by_label = observed_series(observed_data)
    factors = _factors(calibration_result, by_label, list(by_label))
    return [
        observation_panel(
            outputs,
            label,
            observed,
            Trace(
                observation_positions(observed),
                _at_observations(results, observed, factors),
                "Simulation",
            ),
        )
        for label, observed in by_label.items()
    ]


def replicate_observation_panels(
    replicates: "SimulationReplicates",
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
    outputs = PanelOutputs(mean, replicates.model)
    by_label = observed_series(observed_data)
    factors = _factors(calibration_result, by_label, list(by_label))
    panels = []
    for label, observed in by_label.items():
        statistics = spread(
            [_at_observations(run, observed, factors) for run in replicates],
            confidence_level,
        )
        panels.append(
            observation_panel(
                outputs,
                label,
                observed,
                Trace(
                    observation_positions(observed),
                    _at_observations(mean, observed, factors),
                    f"Mean of {len(replicates)} replicates",
                ),
                Band(
                    observation_positions(observed),
                    statistics["ci_lower"],
                    statistics["ci_upper"],
                    f"{confidence_level:.0%} of the replicates:\n{MODEL_BAND}",
                ),
            )
        )
    return panels


def _factors(
    calibration_result: CalibrationResult | None,
    by_series: Mapping[str, list[ObservedDataPoint]],
    selected: Iterable[str],
) -> Mapping[str, float]:
    """
    The fitted values of the factors, with one warning for each factor of the
    observations of the `selected` series that `calibration_result` lacks
    (`warn_missing_factors`).
    """
    factors = calibration_result.best_parameters if calibration_result else {}
    warn_missing_factors(
        (point for series_id in selected for point in by_series.get(series_id, [])),
        factors,
    )
    return factors


def _at_observations(
    results: "SimulationResults",
    observed: list[ObservedDataPoint],
    factors: Mapping[str, float],
) -> list[float]:
    """
    The model value of each observation in one run, through its own
    operator: the sum of its outputs, its window, its scale and its factor.
    """
    return [
        _read(results, point.observed_outputs, point) * model_multiplier(point, factors)
        for point in observed
    ]


def _read(
    results: "SimulationResults", outputs: list[str], point: ObservedDataPoint
) -> float:
    """
    The sum of `outputs` of one run as `point` reads it: at its position, or
    its increment over its window (`windowed_totals`).
    """
    series = total_series(results, outputs)
    if point.window:
        return _windowed_totals(series, results.positions, point.window, [point.at])[0]
    return _sample(series, results.positions, [point.at])[0]


def _model_line(
    results: "SimulationResults",
    outputs: PanelOutputs,
    series_id: str,
    observed: list[ObservedDataPoint],
    factors: Mapping[str, float],
) -> tuple[list[float], list[float]]:
    """
    The positions and values of the model of one series in one run.

    When the observations of the series do not share one operator with no
    window, each observation is read through its own operator, as the loss
    reads it: the sum of its outputs at its position, or its increment over
    its window (`windowed_totals`), times its scale and its factor. Otherwise
    the sum of the outputs of their operator at every position of the results,
    times its scale and its factor.

    Raises
    ------
    ValueError
        If the results have no value at the position of an observation read
        on its own or at the start of its window.
    """
    if reads_each_observation(observed):
        return [point.at for point in observed], [
            _read(results, point.outputs or outputs.components(series_id), point)
            * model_multiplier(point, factors)
            for point in observed
        ]
    components = (
        observed[0].outputs
        if observed and observed[0].outputs
        else outputs.components(series_id)
    )
    multiplier = model_multiplier(observed[0] if observed else None, factors)
    return list(results.positions), [
        value * multiplier for value in total_series(results, components)
    ]


def run_window(
    results: "SimulationResults",
    outputs: PanelOutputs,
    series_id: str,
    window: float,
) -> tuple[Trace, Band | None]:
    """The gain of a series of a run over each complete window, as
    `windowed_totals` reads it."""
    ends = commol_rs.calibration.window_ends(window, results.positions[-1])
    gains = _windowed_totals(
        total_series(results, outputs.components(series_id)),
        results.positions,
        window,
        ends,
    )
    return Trace(ends, gains, "Simulation"), None


def replicates_window(
    replicates: "SimulationReplicates",
    outputs: PanelOutputs,
    series_id: str,
    confidence_level: float,
    window: float,
) -> tuple[Trace, Band | None]:
    """The mean gain of a series of replicates over each complete window, and
    the band of the gains of the replicates."""
    components = outputs.components(series_id)
    ends = commol_rs.calibration.window_ends(window, replicates.positions[-1])
    gains = [
        _windowed_totals(total_series(run, components), run.positions, window, ends)
        for run in replicates
    ]
    statistics = spread(gains, confidence_level)
    return Trace(ends, statistics["mean"], "Mean of the replicates"), Band(
        ends,
        statistics["ci_lower"],
        statistics["ci_upper"],
        f"{confidence_level:.0%} of the replicates",
    )
