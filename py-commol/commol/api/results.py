"""Simulation results that carry the positions on the axis they were read at.

`SimulationResults` holds one run; `SimulationReplicates` holds the replicates
of a stochastic engine, one run per seed, with their mean, quantiles and
member statistics. Both know the `Model` they ran, report themselves
(`report`) and draw their figures (`plot`).
"""

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from commol.api.plotting.accessors import ReplicatesPlot, ResultsPlot

if TYPE_CHECKING:
    from commol.api.reporting import SimulationReport
    from commol.context.calibration import ObservedDataPoint
    from commol.context.inference.conformal import PredictionIntervals
    from commol.context.inference.quantities import QuantityBase
    from commol.context.model import Model

# Relative tolerance with which two positions on the axis are the same, as in
# the ODE engine (`crates/commol-differential/src/solve.rs`).
POSITION_TOLERANCE = 1e-9


class SimulationResults(dict[str, list[float]]):
    """
    Values of every output at each position on the axis.

    A ``dict`` from output name to one value per position, so
    ``results["A"]``, ``json.dumps(results)`` and
    ``pandas.DataFrame(results)`` work as for any mapping. The positions the
    values were read at are in :attr:`positions`.

    Attributes
    ----------
    positions : list[float]
        Positions on the axis, ascending. ``results[name][i]`` is the value of
        ``name`` at ``positions[i]``.
    seed : int | None
        The seed of the run of a stochastic engine, which repeats it; None for
        a deterministic engine, whose runs depend on no seed.
    model : Model | None
        The model of the run, which `Simulation.run` and `Simulation.run_at`
        set; None for results built by hand. A report reads the outputs of
        each bin and accumulator from its structure, never the values of its
        parameters.
    """

    positions: list[float]
    seed: int | None
    model: "Model | None"

    def __init__(
        self,
        positions: Iterable[float],
        values: Mapping[str, Sequence[float]] | None = None,
        seed: int | None = None,
        model: "Model | None" = None,
    ):
        """
        Results at the given positions.

        Parameters
        ----------
        positions : Iterable[float]
            Positions on the axis, ascending.
        values : Mapping[str, Sequence[float]] | None, optional
            One value per position for each output.
        seed : int | None, optional
            The seed of the run, or None when it depends on no seed.
        model : Model | None, optional
            The model of the run, or None when no model is known.

        Raises
        ------
        ValueError
            If an output does not have one value per position.
        """
        super().__init__(
            {name: list(series) for name, series in (values or {}).items()}
        )
        self.positions = [float(position) for position in positions]
        self.seed = seed
        self.model = model
        for name, series in self.items():
            if len(series) != len(self.positions):
                raise ValueError(
                    f"Output '{name}' has {len(series)} values for "
                    f"{len(self.positions)} positions."
                )

    @classmethod
    def from_rows(
        cls,
        positions: Sequence[float],
        names: Sequence[str],
        rows: Sequence[Sequence[float]],
        seed: int | None = None,
        model: "Model | None" = None,
    ) -> "SimulationResults":
        """
        Results from one row of output values per position.

        Parameters
        ----------
        positions : Sequence[float]
            Positions on the axis, ascending.
        names : Sequence[str]
            Output names, in the order of each row.
        rows : Sequence[Sequence[float]]
            One row per position.
        seed : int | None, optional
            The seed of the run, or None when it depends on no seed.
        model : Model | None, optional
            The model of the run, or None when no model is known.

        Returns
        -------
        SimulationResults
            One series per output name.
        """
        columns = list(zip(*rows)) if rows else [() for _ in names]
        return cls(
            positions, dict(zip(names, columns, strict=True)), seed=seed, model=model
        )

    def rows(self) -> list[list[float]]:
        """
        Every output at each position, one row per position.

        Returns
        -------
        list[list[float]]
            ``rows()[i][j]`` is the value of the ``j``-th output, in key order
            (the order of ``Simulation.simulation_outputs``), at
            ``positions[i]``.
        """
        return [list(row) for row in zip(*self.values())] if self else []

    def quantities(self, quantities: "Sequence[QuantityBase]") -> dict[str, float]:
        """
        The quantities of interest of this run, as a `Propagation` reduces
        each of its runs.

        The run must hold every position the quantities read: the results of
        ``Simulation.run_at(quantity_schedule(quantities))`` hold them all.
        Each position is found within ``1e-9``, as `index_at` finds it, and
        Rust reduces the rows with the code of `Propagation`, so the values
        equal those of a `Propagation` of a sample of one draw with the
        parameters of this run.

        On the ODE engine, `Simulation.run` reports a position that is not a
        breakpoint by interpolation, so a value read from it can differ from
        that of a run that ends there by about the relative tolerance of the
        solver, and `PeakPosition` can change on a near-tie.

        Parameters
        ----------
        quantities : Sequence[Quantity]
            The quantities; a `Trajectory` gives one scalar per position.

        Returns
        -------
        dict[str, float]
            One value per label of the quantities, in their order.

        Raises
        ------
        ValueError
            If there is no quantity, two scalars have one label, a quantity
            names an output the results do not have, or the run lacks a
            position a quantity reads; the message lists every such position
            with its quantity and says to call `Simulation.run_at`.
        """
        return quantities_of_runs([self], self.positions, quantities)[0]

    @property
    def plot(self) -> ResultsPlot:
        """
        The figures of the run: ``plot.series()`` and ``plot.cumulative()``.

        They name each series from `model`; results with no model draw only
        their outputs.
        """
        return ResultsPlot(self)

    def report(
        self,
        positions: Sequence[float],
        window: float,
        windows: int,
        observed_data: "Iterable[ObservedDataPoint]" = (),
        accumulators: Iterable[str] = (),
        prediction_intervals: "PredictionIntervals | None" = None,
    ) -> "SimulationReport":
        """
        Report the observed data, compartment series and accumulators of the run.

        The blocks share one window axis, so the observed values, the windowed
        accumulator increments and the sampled series can be read against
        each other. Add whatever envelope you need around them, such as a run
        label, axis labels or summary figures in the terms of your domain.

        Parameters
        ----------
        positions : Sequence[float]
            Positions the series and cumulative accumulators are sampled at;
            each must be a position of the results.
        window : float
            Length of one window on the axis.
        windows : int
            Number of windows on the observed and windowed axes.
        observed_data : Iterable[ObservedDataPoint], optional
            The calibration targets to report alongside the run.
        accumulators : Iterable[str], optional
            Accumulator ids to report; their outputs come from `model`.
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
            If the results have no value at a position the report reads, or
            they have no `model` and an accumulator is reported.
        KeyError
            If an accumulator is not a bin or accumulator of the model.
        """
        from commol.api.reporting import _ReportRuns, _simulation_report

        return _simulation_report(
            _ReportRuns(model=self.model, runs=[self]),
            positions,
            window,
            windows,
            observed_data,
            accumulators,
            prediction_intervals,
        )

    def __repr__(self) -> str:
        """The positions, the seed and the values."""
        seed = "" if self.seed is None else f"seed={self.seed!r}, "
        return (
            f"SimulationResults(positions={self.positions!r}, {seed}"
            f"{dict.__repr__(self)})"
        )


def index_at(positions: Sequence[float], at: float) -> int:
    """
    Index of a position in an ascending list of positions.

    Two positions are the same within a relative tolerance of ``1e-9``, so a
    position computed as ``end - window`` finds its entry.

    Parameters
    ----------
    positions : Sequence[float]
        Positions on the axis, ascending.
    at : float
        The position to find.

    Returns
    -------
    int
        Index of `at` in `positions`.

    Raises
    ------
    ValueError
        If `at` is not one of `positions`.
    """
    index = bisect.bisect_left(positions, at)
    for candidate in (index - 1, index):
        if 0 <= candidate < len(positions) and same_position(positions[candidate], at):
            return candidate
    raise ValueError(
        f"The results have no value at {at}; run the simulation at that position "
        f"with Simulation.run_at."
    )


def same_position(a: float, b: float) -> bool:
    """Whether two positions on the axis are the same, within ``1e-9``."""
    return math.isclose(a, b, rel_tol=POSITION_TOLERANCE, abs_tol=POSITION_TOLERANCE)


def quantities_of_runs(
    runs: Sequence[SimulationResults],
    positions: Sequence[float],
    quantities: "Sequence[QuantityBase]",
) -> list[dict[str, float]]:
    """
    The quantities of each run, reduced in Rust as a `Propagation` reduces
    its runs.

    Parameters
    ----------
    runs : Sequence[SimulationResults]
        The runs, at `positions`, with the same outputs.
    positions : Sequence[float]
        Positions of the runs, ascending.
    quantities : Sequence[Quantity]
        The quantities.

    Returns
    -------
    list[dict[str, float]]
        One value per label of the quantities, for each run.

    Raises
    ------
    ValueError
        If there is no quantity, two scalars have one label, a quantity names
        an output the runs do not have, or the runs lack a position a
        quantity reads.
    """
    from commol.commol_rs import _commol_rs as commol_rs
    from commol.context.inference.quantities import expand_quantities

    labels, rust = expand_quantities(quantities)
    names = [quantity.axes()[0][0] for quantity in quantities]
    values = commol_rs.inference.quantities_of_runs(
        rust,
        names,
        list(runs[0]),
        [float(position) for position in positions],
        [run.rows() for run in runs],
    )
    return [dict(zip(labels, row, strict=True)) for row in values]


def sample(
    series: Sequence[float], positions: Sequence[float], at: Iterable[float]
) -> list[float]:
    """
    Read a series at the given positions.

    Compartment counts are instantaneous, so they are read at the position
    rather than averaged over the interval leading to it.

    Parameters
    ----------
    series : Sequence[float]
        One value per position.
    positions : Sequence[float]
        Positions of the series, ascending.
    at : Iterable[float]
        Positions to read.

    Returns
    -------
    list[float]
        One value per position in `at`.

    Raises
    ------
    ValueError
        If a position in `at` is not one of `positions`.
    """
    return [series[index_at(positions, position)] for position in at]


class SimulationReplicates(list[SimulationResults]):
    """
    Replicates of a run of a stochastic engine, one `SimulationResults` each.

    A ``list`` of runs at the same positions, each with its own seed:
    ``replicates[i].seed`` repeats replicate ``i``. The statistics reduce the
    replicates position by position, and `member_statistics` reduces them as
    ``ci_percentiles`` requires.

    Attributes
    ----------
    seed : int
        The seed the replicate seeds derive from.
    positions : list[float]
        Positions on the axis, shared by every replicate.
    model : Model | None
        The model of the replicates, which `Simulation.replicates` sets; None
        for replicates built by hand.
    """

    seed: int
    positions: list[float]
    model: "Model | None"

    def __init__(
        self,
        runs: Iterable[SimulationResults],
        seed: int,
        positions: Sequence[float],
        model: "Model | None" = None,
    ):
        """
        Replicates of one run.

        Parameters
        ----------
        runs : Iterable[SimulationResults]
            One run per replicate, at `positions`.
        seed : int
            The seed the replicate seeds derive from.
        positions : Sequence[float]
            Positions on the axis, ascending.
        model : Model | None, optional
            The model of the replicates, or None when no model is known.

        Raises
        ------
        ValueError
            If a run is not at `positions`.
        """
        super().__init__(runs)
        self.seed = seed
        self.positions = [float(position) for position in positions]
        self.model = model
        for index, run in enumerate(self):
            if run.positions != self.positions:
                raise ValueError(
                    f"Replicate {index} is at other positions than the replicates."
                )

    @property
    def seeds(self) -> list[int]:
        """The seed of each replicate, in order."""
        return [run.seed for run in self if run.seed is not None]

    @property
    def plot(self) -> ReplicatesPlot:
        """
        The figures of the replicates: ``plot.series()``, their mean and band,
        and ``plot.cumulative()``.
        """
        return ReplicatesPlot(self)

    def _values(self, name: str) -> np.ndarray:
        return np.asarray([run[name] for run in self], dtype=float)

    def mean(self) -> SimulationResults:
        """
        The mean of the replicates at each position, for each output.

        Returns
        -------
        SimulationResults
            One series of means per output, with no seed.

        Raises
        ------
        ValueError
            If there are no replicates.
        """
        names = list(self[0]) if self else []
        if not names:
            raise ValueError("There are no replicates.")
        return SimulationResults(
            self.positions,
            {name: self._values(name).mean(axis=0).tolist() for name in names},
            model=self.model,
        )

    def quantiles(self, levels: Sequence[float]) -> dict[float, SimulationResults]:
        """
        Quantiles of the replicates at each position, for each output.

        The quantile is the one convention of commol (`weighted_quantiles`
        with equal weights, NumPy's ``method="hazen"``).

        Parameters
        ----------
        levels : Sequence[float]
            Quantile levels, each between 0 and 1, such as ``[0.05, 0.5,
            0.95]``.

        Returns
        -------
        dict[float, SimulationResults]
            The results of each level.

        Raises
        ------
        ValueError
            If a level is outside ``[0, 1]`` or there are no replicates.
        """
        for level in levels:
            if not 0.0 <= level <= 1.0:
                raise ValueError(f"A quantile level is between 0 and 1, got {level}.")
        if not self:
            raise ValueError("There are no replicates.")
        from commol.api.inference.intervals import weighted_quantiles

        names = list(self[0])
        per_name = {
            name: [
                weighted_quantiles(column.tolist(), levels)
                for column in self._values(name).T
            ]
            for name in names
        }
        return {
            level: SimulationResults(
                self.positions,
                {
                    name: [at_position[index] for at_position in per_name[name]]
                    for name in names
                },
                model=self.model,
            )
            for index, level in enumerate(levels)
        }

    def quantities(
        self, quantities: "Sequence[QuantityBase]"
    ) -> list[dict[str, float]]:
        """
        The quantities of interest of each replicate, as
        `SimulationResults.quantities` reads one run.

        Each replicate is reduced on its own, with no mean: a summary across
        the replicates is a reduction of these values, such as
        `member_statistics`.

        A stochastic `Propagation` seeds replicate ``i`` of draw ``j`` with
        ``replicate_seed(stream_seed(seed, 4), j * replicates + i)``, a stream
        of the seed of the analysis (`crates/commol-inference/src/predictive.rs`,
        `run_sets`). So the replicates of ``Simulation.replicates(n,
        seed=s)`` repeat the runs of the first draw only when ``s`` is that
        stream seed, and ``Simulation.run(seed=s)`` repeats none of them.

        Parameters
        ----------
        quantities : Sequence[Quantity]
            The quantities; a `Trajectory` gives one scalar per position.

        Returns
        -------
        list[dict[str, float]]
            One dict per replicate, in seed order, with one value per label of
            the quantities.

        Raises
        ------
        ValueError
            If there are no replicates, there is no quantity, two scalars have
            one label, a quantity names an output the replicates do not have,
            or they lack a position a quantity reads.
        """
        if not self:
            raise ValueError("There are no replicates.")
        return quantities_of_runs(self, self.positions, quantities)

    def member_statistics(
        self, output: str, confidence_level: float
    ) -> dict[str, list[float]]:
        """
        The spread of one output across the replicates, at each position.

        Parameters
        ----------
        output : str
            An output name.
        confidence_level : float
            Confidence level of the interval, strictly between 0 and 1.

        Returns
        -------
        dict[str, list[float]]
            Keys ``mean``, ``median``, ``ci_lower``, ``ci_upper``, ``min`` and
            ``max``, one value per position (`member_statistics`).

        Raises
        ------
        KeyError
            If no replicate has the output.
        ValueError
            If the confidence level is out of range or there are no
            replicates.
        """
        from commol.api.inference.intervals import member_statistics

        if self and output not in self[0]:
            raise KeyError(
                f"'{output}' is not an output of the replicates. "
                f"Outputs: {sorted(self[0])}"
            )
        return member_statistics([run[output] for run in self], confidence_level)

    def report(
        self,
        confidence_level: float,
        positions: Sequence[float],
        window: float,
        windows: int,
        observed_data: "Iterable[ObservedDataPoint]" = (),
        accumulators: Iterable[str] = (),
        prediction_intervals: "PredictionIntervals | None" = None,
    ) -> "SimulationReport":
        """
        Report the replicates as `SimulationResults.report` reports one run.

        Every number becomes a ``{mean, median, ci_lower, ci_upper, min,
        max}`` block, reduced replicate by replicate; the layout is the one of
        the report of a run.

        Parameters
        ----------
        confidence_level : float
            Confidence level of the reported intervals, strictly between 0
            and 1.
        positions : Sequence[float]
            Positions the series and cumulative accumulators are sampled at;
            each must be a position of the replicates.
        window : float
            Length of one window on the axis.
        windows : int
            Number of windows on the observed and windowed axes.
        observed_data : Iterable[ObservedDataPoint], optional
            The calibration targets to report alongside the replicates.
        accumulators : Iterable[str], optional
            Accumulator ids to report; their outputs come from `model`.
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
            If there are no replicates, the confidence level is out of range,
            the replicates have no value at a position the report reads, or
            they have no `model` and an accumulator is reported.
        KeyError
            If an accumulator is not a bin or accumulator of the model.
        """
        from commol.api.reporting import _ReportRuns, _simulation_report

        if not self:
            raise ValueError("There are no replicates.")
        return _simulation_report(
            _ReportRuns(
                model=self.model,
                runs=list(self),
                confidence_level=confidence_level,
            ),
            positions,
            window,
            windows,
            observed_data,
            accumulators,
            prediction_intervals,
        )

    def __repr__(self) -> str:
        """The number of replicates, their seed and their positions."""
        return (
            f"SimulationReplicates({len(self)} replicates, seed={self.seed!r}, "
            f"positions={self.positions!r})"
        )
