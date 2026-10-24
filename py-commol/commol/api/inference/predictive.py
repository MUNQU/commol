"""The predictive of a parameter sample: the run of every draw and its bands."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from commol.api.plotting.accessors import PredictivePlot
from commol.api.results import SimulationResults, quantities_of_runs
from commol.context.calibration import ObservedDataPoint
from commol.context.inference.noise import ObservationNoise
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from commol.api.reporting import SimulationReport
    from commol.api.simulation import Simulation
    from commol.context.inference.conformal import PredictionIntervals
    from commol.context.inference.quantities import QuantityBase
    from commol.context.model import Model
    from commol.commol_rs._commol_rs import (
        ObservationDrawsProtocol,
        PredictiveProtocol,
    )


class PredictiveSource(Protocol):
    """
    What a `Predictive` runs its draws through: a `Calibrator`, or a
    `SensitivityAnalysis` of the inputs of the sample.

    A source runs the draws and reads observations in Rust; a `Predictive`
    knows no source.
    """

    @property
    def simulation(self) -> "Simulation":
        """The simulation of the source."""
        ...

    def rust_predictive(
        self, sample: ParameterSample, positions: list[float]
    ) -> "PredictiveProtocol":
        """The run of every draw of `sample` at `positions`."""
        ...

    def rust_observation_draws(
        self,
        sample: ParameterSample,
        observed_data: list[ObservedDataPoint],
        noise: ObservationNoise,
    ) -> "ObservationDrawsProtocol":
        """Every draw's value of `observed_data` with `noise`."""
        ...

    def with_model(self, model: "Model") -> "PredictiveSource":
        """The same source on `model`, with the same seed."""
        ...

    def rust_log_score(
        self, sample: ParameterSample, observed_data: list[ObservedDataPoint]
    ) -> list[float]:
        """The log score of each observation under the predictive."""
        ...


class SeriesBand(BaseModel):
    """
    The band of one series: four values at each of its positions.

    Attributes
    ----------
    confidence_level : float
        The level of the central interval.
    positions : list[float]
        Where the values are.
    mean : list[float]
        The weighted mean across the draws.
    median : list[float]
        The weighted median.
    lower : list[float]
        The quantile ``(1 − c)/2``.
    upper : list[float]
        The quantile ``(1 + c)/2``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_level: float
    positions: list[float]
    mean: list[float]
    median: list[float]
    lower: list[float]
    upper: list[float]


class PredictiveBand(BaseModel):
    """
    The band of every output of a predictive.

    Attributes
    ----------
    confidence_level : float
        The level of the central interval.
    positions : list[float]
        Where the values are.
    mean, median, lower, upper : dict[str, list[float]]
        For each output, one value per position, reduced across the draws
        with their weights.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_level: float
    positions: list[float]
    mean: dict[str, list[float]] = Field(description="Weighted mean")
    median: dict[str, list[float]] = Field(description="Weighted median")
    lower: dict[str, list[float]] = Field(description="Lower quantile")
    upper: dict[str, list[float]] = Field(description="Upper quantile")


class ObservationDraws:
    """
    Every draw's value of a set of observations.

    The values are read through the observation operator of the calibration
    problem (sums of outputs, windows, factors), the one the loss uses. Without
    observation noise their interval is a band of the model, not a prediction
    interval.
    """

    def __init__(
        self,
        observations: Sequence[ObservedDataPoint],
        draws: "ObservationDrawsProtocol",
    ) -> None:
        """
        Wrap the draws of `observations`.

        Parameters
        ----------
        observations : Sequence[ObservedDataPoint]
            The observations, in the order of each draw's values.
        draws : ObservationDrawsProtocol
            The draws, computed in Rust.
        """
        self.observations: list[ObservedDataPoint] = list(observations)
        self.rust_draws: "ObservationDrawsProtocol" = draws

    @property
    def values(self) -> list[list[float]]:
        """Each draw's value of each observation, ``[draw][observation]``."""
        return self.rust_draws.values

    @property
    def weights(self) -> list[float]:
        """The weight of each draw; they sum to 1."""
        return self.rust_draws.weights

    @property
    def observed(self) -> list[float]:
        """The observed value of each observation."""
        return [point.value for point in self.observations]

    def band(self, confidence_level: float) -> SeriesBand:
        """
        The band of each observation across the draws.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        SeriesBand
            One value per observation, at its position.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        mean, median, lower, upper = self.rust_draws.band(confidence_level)
        return SeriesBand(
            confidence_level=confidence_level,
            positions=[point.at for point in self.observations],
            mean=mean,
            median=median,
            lower=lower,
            upper=upper,
        )


def by_output(
    names: Sequence[str], table: Sequence[Sequence[float]]
) -> dict[str, list[float]]:
    """
    The columns of a table of a band, ``[position][output]``, by output name.

    Every band of the engine has this layout: the band of a predictive and the
    band of the delta method.

    Parameters
    ----------
    names : Sequence[str]
        The name of each output, one per column.
    table : Sequence[Sequence[float]]
        One row per position, one value per output.

    Returns
    -------
    dict[str, list[float]]
        For each output, its value at each position.
    """
    return {name: [row[index] for row in table] for index, name in enumerate(names)}


class Predictive:
    """
    The run of every draw of a parameter sample.

    The draws run in parallel in Rust, through the forward model of a
    calibration problem or of a sensitivity analysis: with a stochastic engine
    every replicate of a draw is a draw of the predictive, with an equal share
    of its weight. A band
    reduces the draws position by position with their weights; without
    observation noise it is a band of the model, not a prediction interval.
    """

    def __init__(
        self,
        source: PredictiveSource,
        sample: ParameterSample,
        positions: Sequence[float],
    ) -> None:
        """
        Run every draw of `sample` at `positions`.

        Parameters
        ----------
        source : PredictiveSource
            The calibration problem the sample is a sample of (`Calibrator`),
            or the sensitivity analysis of its inputs (`SensitivityAnalysis`).
        sample : ParameterSample
            Its parameters, in the order of the problem or the analysis.
        positions : Sequence[float]
            Positions to report at, sorted and unique.

        Raises
        ------
        ValueError
            Before any draw runs: if the sample's parameters are not the
            source's, or the positions are not sorted and unique or the engine
            does not report at one of them (a difference engine reports only
            at integer steps).
        InfeasibleStateError
            If a draw leaves the values its model allows. The message names
            the parameter set, then says what the run says (the compartment,
            the step and the flows), or what the initial amounts exceed.
        """
        self.source: PredictiveSource = source
        self.sample: ParameterSample = sample
        self.positions: list[float] = [float(position) for position in positions]
        self.output_names: list[str] = list(source.simulation.engine.output_names)
        self._rust = source.rust_predictive(sample, self.positions)

    @property
    def weights(self) -> list[float]:
        """The weight of each draw of the predictive; they sum to 1."""
        return self._rust.weights

    @property
    def draw_seeds(self) -> list[int]:
        """
        The seed of each draw of `sample`, in its order.

        With a stochastic engine, replicate ``i`` of draw ``j`` runs with the
        seed replicate ``i`` of ``Simulation.replicates(count, seed=draw_seeds[j])``
        takes, so those replicates of the model at that draw give its runs.
        """
        return self._rust.draw_seeds

    @property
    def plot(self) -> PredictivePlot:
        """
        The figures of the draws: ``plot.series()`` and ``plot.cumulative()``,
        the weighted median and the band of each series.
        """
        return PredictivePlot(self)

    def runs(self) -> list[SimulationResults]:
        """
        The trajectory of every draw, in the order of `weights`.

        Returns
        -------
        list[SimulationResults]
            One run per draw, at `positions`, with the model of the source.
        """
        model = self.source.simulation.model_definition
        return [
            SimulationResults.from_rows(
                self.positions, self.output_names, rows, model=model
            )
            for rows in self._rust.trajectories
        ]

    def quantities(
        self, quantities: "Sequence[QuantityBase]"
    ) -> list[dict[str, float]]:
        """
        The quantities of interest of every draw, in the order of `weights`.

        Each run is reduced on its own, as `SimulationResults.quantities`
        reduces one, with the Rust code of `Propagation`. The predictive runs
        at `positions`, so they must hold ``quantity_schedule(quantities)``.

        Parameters
        ----------
        quantities : Sequence[Quantity]
            The quantities; a `Trajectory` gives one scalar per position.

        Returns
        -------
        list[dict[str, float]]
            One dict per draw of the predictive, with one value per label of
            the quantities.

        Raises
        ------
        ValueError
            If there is no quantity, two scalars have one label, a quantity
            names an output the model does not have, or `positions` lack a
            position a quantity reads; the message lists every such position
            with its quantity.
        """
        return quantities_of_runs(self.runs(), self.positions, quantities)

    def report(
        self,
        confidence_level: float,
        positions: Sequence[float],
        window: float,
        windows: int,
        observed_data: Iterable[ObservedDataPoint] = (),
        accumulators: Iterable[str] = (),
        prediction_intervals: "PredictionIntervals | None" = None,
    ) -> "SimulationReport":
        """
        Report the runs of the draws as `SimulationResults.report` reports one.

        Every number becomes a ``{mean, median, ci_lower, ci_upper, min,
        max}`` block, reduced draw by draw with the weights of the draws; the
        layout is the one of the report of a run. Each leaf comes from the
        draws only, so the report takes no point estimate. The outputs of an
        accumulator come from the model of the source.

        Parameters
        ----------
        confidence_level : float
            Confidence level of the reported intervals, strictly between 0
            and 1.
        positions : Sequence[float]
            Positions the series and cumulative accumulators are sampled at;
            each must be one of `positions` of the predictive.
        window : float
            Length of one window on the axis.
        windows : int
            Number of windows on the observed and windowed axes.
        observed_data : Iterable[ObservedDataPoint], optional
            The calibration targets to report alongside the draws.
        accumulators : Iterable[str], optional
            Accumulator ids to report.
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
            If the confidence level is out of range, or the predictive has no
            value at a position the report reads.
        KeyError
            If an accumulator is not a bin or accumulator of the model.
        """
        from commol.api.reporting import _ReportRuns, _simulation_report

        return _simulation_report(
            _ReportRuns(
                model=self.source.simulation.model_definition,
                runs=self.runs(),
                confidence_level=confidence_level,
                weights=self.weights,
            ),
            positions,
            window,
            windows,
            observed_data,
            accumulators,
            prediction_intervals,
        )

    def band(self, confidence_level: float) -> PredictiveBand:
        """
        The band of every output.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        PredictiveBand
            The weighted mean, median and central interval of each output at
            each position.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        mean, median, lower, upper = self._rust.band(confidence_level)
        names = self.output_names
        return PredictiveBand(
            confidence_level=confidence_level,
            positions=self.positions,
            mean=by_output(names, mean),
            median=by_output(names, median),
            lower=by_output(names, lower),
            upper=by_output(names, upper),
        )

    def series_band(
        self,
        outputs: Sequence[str],
        confidence_level: float,
        cumulative: bool = False,
    ) -> SeriesBand:
        """
        The band of the sum of some outputs, or of its running total.

        Each draw is reduced to the series first, then the series are reduced
        across the draws.

        Parameters
        ----------
        outputs : Sequence[str]
            The outputs summed.
        confidence_level : float
            Strictly between 0 and 1.
        cumulative : bool, optional
            Whether the series is the running total along the positions.

        Returns
        -------
        SeriesBand
            One value per position.

        Raises
        ------
        ValueError
            If an output is unknown or the confidence level is out of range.
        """
        unknown = [name for name in outputs if name not in self.output_names]
        if unknown:
            raise ValueError(
                f"Unknown outputs {unknown}. Outputs: {self.output_names}."
            )
        indices = [self.output_names.index(name) for name in outputs]
        mean, median, lower, upper = self._rust.series_band(
            indices, cumulative, confidence_level
        )
        return SeriesBand(
            confidence_level=confidence_level,
            positions=self.positions,
            mean=mean,
            median=median,
            lower=lower,
            upper=upper,
        )

    def at_observations(
        self,
        observed_data: Sequence[ObservedDataPoint],
        noise: ObservationNoise,
    ) -> ObservationDraws:
        """
        Every draw's value of each observation, with observation noise.

        The observations need not be the ones the problem was fitted to: held
        out data are read the same way, through the observation operator,
        and so are the observations of the model of an analysis. The noise
        is drawn from the calibrator's seed.

        Parameters
        ----------
        observed_data : Sequence[ObservedDataPoint]
            The observations.
        noise : NoNoise | ResidualNoise | DistributionNoise
            What an observation shows around a draw's prediction, with no
            default: ``NoNoise()`` gives a band of the model,
            ``ResidualNoise(...)`` and ``DistributionNoise()`` a prediction
            interval. Both need a calibrator: residuals come from a fit, and
            observation models from a problem.

        Returns
        -------
        ObservationDraws
            Each draw's value of each observation.

        Raises
        ------
        ValueError
            If an observation names an unknown output or factor, a series the
            residual noise has no residuals of, or a series with no
            observation model; `DistributionNoise` comes with a problem that
            has a loss; or the noise is not `NoNoise()` and the source is an
            analysis.
        InfeasibleStateError
            If a draw leaves the values its model allows. The message names
            the parameter set, then says what the run says (the compartment,
            the step and the flows), or what the initial amounts exceed.
        """
        draws = self.source.rust_observation_draws(
            self.sample, list(observed_data), noise
        )
        return ObservationDraws(observed_data, draws)
