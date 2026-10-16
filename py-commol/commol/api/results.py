"""Simulation results that carry the positions on the axis they were read at.

`SimulationResults` holds one run; `SimulationReplicates` holds the replicates
of a stochastic engine, one run per seed, with their mean, quantiles and
member statistics.
"""

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

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
    """

    positions: list[float]
    seed: int | None

    def __init__(
        self,
        positions: Iterable[float],
        values: Mapping[str, Sequence[float]] | None = None,
        seed: int | None = None,
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

        Returns
        -------
        SimulationResults
            One series per output name.
        """
        columns = list(zip(*rows)) if rows else [() for _ in names]
        return cls(positions, dict(zip(names, columns, strict=True)), seed=seed)

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
    """

    seed: int
    positions: list[float]

    def __init__(
        self, runs: Iterable[SimulationResults], seed: int, positions: Sequence[float]
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

        Raises
        ------
        ValueError
            If a run is not at `positions`.
        """
        super().__init__(runs)
        self.seed = seed
        self.positions = [float(position) for position in positions]
        for index, run in enumerate(self):
            if run.positions != self.positions:
                raise ValueError(
                    f"Replicate {index} is at other positions than the replicates."
                )

    @property
    def seeds(self) -> list[int]:
        """The seed of each replicate, in order."""
        return [run.seed for run in self if run.seed is not None]

    def _values(self, name: str) -> np.ndarray:
        if not self:
            raise ValueError("There are no replicates.")
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
        )

    def quantiles(self, levels: Sequence[float]) -> dict[float, SimulationResults]:
        """
        Quantiles of the replicates at each position, for each output.

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
        names = list(self[0])
        return {
            level: SimulationResults(
                self.positions,
                {
                    name: np.quantile(self._values(name), level, axis=0).tolist()
                    for name in names
                },
            )
            for level in levels
        }

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
        from commol.api.probabilistic.intervals import member_statistics

        if self and output not in self[0]:
            raise KeyError(
                f"'{output}' is not an output of the replicates. "
                f"Outputs: {sorted(self[0])}"
            )
        return member_statistics([run[output] for run in self], confidence_level)

    def __repr__(self) -> str:
        """The number of replicates, their seed and their positions."""
        return (
            f"SimulationReplicates({len(self)} replicates, seed={self.seed!r}, "
            f"positions={self.positions!r})"
        )
