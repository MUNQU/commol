"""The data of the panels of a predictive: the median of the draws of each
series, their band and the observations, as `commol.plotting` draws them."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from commol._problem.problem import ObservedDataPoint
from commol.commol_rs import _commol_rs as commol_rs
from commol.observations.noise import NoNoise
from commol.plotting.panels import (
    MODEL_BAND,
    Band,
    ObservationPanel,
    Panel,
    PanelOutputs,
    Trace,
    cumulative_observed,
    group_observed,
    observation_panel,
    observation_positions,
    observed_series,
    observed_trace,
    reads_each_observation,
)

if TYPE_CHECKING:
    from commol.uncertainty.predictive import Predictive, SeriesBand


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
    outputs = PanelOutputs(
        predictive.output_names, predictive.source.simulation.model_definition
    )
    by_series = group_observed(observed_data)
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
                observed=observed_trace(observed),
            )
        )
    return panels


def predictive_cumulative_panels(
    predictive: "Predictive",
    confidence_level: float,
    observed_data: Iterable[ObservedDataPoint] | None = None,
    series: Sequence[str] | None = None,
) -> list[Panel]:
    """
    The panels of the band of the running total of each series of a
    predictive, and of its observations, each divided by its scale and by the
    weighted median of its factor.

    Raises
    ------
    ValueError
        If an observation names a factor the problem does not have
        (`_check_factors`).
    """
    outputs = PanelOutputs(
        predictive.output_names, predictive.source.simulation.model_definition
    )
    factors = {
        name: values[0] for name, values in predictive.sample.quantiles([0.5]).items()
    }
    selected = outputs.select(series)
    by_series = group_observed(observed_data)
    _check_factors(
        predictive,
        [point for series_id in selected for point in by_series.get(series_id, [])],
    )
    panels = []
    for series_id in selected:
        band = predictive.series_band(
            outputs.components(series_id), confidence_level, cumulative=True
        )
        panels.append(
            outputs.panel(
                series_id,
                Trace(band.positions, band.median, "Median of the draws (cumulative)"),
                band=_draws_band(band),
                observed=cumulative_observed(by_series.get(series_id, []), factors),
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
    by_label = observed_series(observed_data)
    band = predictive.at_observations(
        [point for observed in by_label.values() for point in observed], NoNoise()
    ).band(confidence_level)
    outputs = PanelOutputs(
        predictive.output_names, predictive.source.simulation.model_definition
    )
    panels = []
    start = 0
    for label, observed in by_label.items():
        end = start + len(observed)
        panels.append(
            observation_panel(
                outputs,
                label,
                observed,
                Trace(
                    observation_positions(observed),
                    band.median[start:end],
                    "Median of the draws",
                ),
                Band(
                    observation_positions(observed),
                    band.lower[start:end],
                    band.upper[start:end],
                    f"{confidence_level:.0%} of the draws:\n{MODEL_BAND}",
                ),
            )
        )
        start = end
    return panels


def _check_factors(predictive: "Predictive", observed: list[ObservedDataPoint]) -> None:
    """
    Check the observations with a factor against the source of the predictive,
    with no run: the engine refuses an unknown factor, or a parameter that is
    not a factor, as the band of a series with that factor does.

    Raises
    ------
    ValueError
        With the message of the engine.
    """
    with_factor = [point for point in observed if point.factor_id is not None]
    if with_factor:
        predictive.source._check_observations(with_factor)


def _draws_band(band: "SeriesBand") -> Band:
    """The central interval of the draws of a series band."""
    return Band(
        band.positions,
        band.lower,
        band.upper,
        f"{band.confidence_level:.0%} of the draws",
    )


def _predictive_band(
    predictive: "Predictive",
    outputs: PanelOutputs,
    series_id: str,
    observed: list[ObservedDataPoint],
    confidence_level: float,
) -> "SeriesBand":
    """
    The band of one series of a predictive.

    When the observations of the series do not share one operator with no
    window, each draw is read through the operator of each observation at its
    position, as the loss reads it. With one operator and a factor or a
    scale, each draw is read through it at every position of the predictive.
    Otherwise, the band of the sum of the outputs.
    """
    if reads_each_observation(observed):
        points = [
            ObservedDataPoint(
                at=point.at,
                outputs=point.outputs or outputs.components(series_id),
                label=series_id,
                value=0.0,
                window=point.window,
                factor_id=point.factor_id,
                scale=point.scale,
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
    scale = observed[0].scale if observed else 1.0
    if factor_id is None and scale == 1.0:
        return predictive.series_band(components, confidence_level)
    points = [
        ObservedDataPoint(
            at=at,
            outputs=components,
            label=series_id,
            value=0.0,
            factor_id=factor_id,
            scale=scale,
        )
        for at in predictive.positions
    ]
    return predictive.at_observations(points, NoNoise()).band(confidence_level)


def predictive_window(
    predictive: "Predictive",
    outputs: PanelOutputs,
    series_id: str,
    confidence_level: float,
    window: float,
) -> tuple[Trace, Band | None]:
    """The weighted median gain of a series of a predictive over each complete
    window, and the band of the draws, read through the observation
    operator."""
    components = outputs.components(series_id)
    ends = commol_rs.calibration.window_ends(window, predictive.positions[-1])
    points = [
        ObservedDataPoint(
            at=end, outputs=components, label=series_id, value=0.0, window=window
        )
        for end in ends
    ]
    band = predictive.at_observations(points, NoNoise()).band(confidence_level)
    return Trace(band.positions, band.median, "Median of the draws"), _draws_band(band)
