"""Simulation results that carry the positions on the axis they were read at."""

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence

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
    """

    positions: list[float]

    def __init__(
        self,
        positions: Iterable[float],
        values: Mapping[str, Sequence[float]] | None = None,
    ):
        """
        Results at the given positions.

        Parameters
        ----------
        positions : Iterable[float]
            Positions on the axis, ascending.
        values : Mapping[str, Sequence[float]] | None, optional
            One value per position for each output.

        Raises
        ------
        ValueError
            If an output does not have one value per position.
        """
        super().__init__(
            {name: list(series) for name, series in (values or {}).items()}
        )
        self.positions = [float(position) for position in positions]
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

        Returns
        -------
        SimulationResults
            One series per output name.
        """
        columns = list(zip(*rows)) if rows else [() for _ in names]
        return cls(positions, dict(zip(names, columns, strict=True)))

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
        """The positions and the values."""
        return f"SimulationResults(positions={self.positions!r}, {dict.__repr__(self)})"


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
