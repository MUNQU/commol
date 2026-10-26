"""Read the series of a run: the sum of some outputs, and the values at some
positions."""

from collections.abc import Iterable, Mapping, Sequence

from commol.commol_rs import _commol_rs as commol_rs


def total_series(
    results: Mapping[str, Sequence[float]], outputs: Sequence[str]
) -> list[float]:
    """
    The sum of some outputs at each position.

    Parameters
    ----------
    results : Mapping[str, Sequence[float]]
        Simulation results, such as those of `Simulation.run`.
    outputs : Sequence[str]
        Output names.

    Returns
    -------
    list[float]
        One value per position, or no value without outputs.

    Raises
    ------
    KeyError
        If an output is missing from `results`.
    """
    missing = [name for name in outputs if name not in results]
    if missing:
        raise KeyError(f"Missing output series in results: {sorted(missing)}")
    return [
        sum(values) for values in zip(*(results[name] for name in outputs), strict=True)
    ]


def _sample(
    series: Sequence[float], positions: Sequence[float], at: Iterable[float]
) -> list[float]:
    """
    Read a series at the given positions.

    Compartment counts are instantaneous, so they are read at the position
    rather than averaged over the interval leading to it. Each position is
    found with the tolerance of the axis, as an observation finds its row
    (`index_at` of `crates/commol-calibration/src/observation.rs`), so ``0.1
    + 0.2`` reads the value at ``0.3``.

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
    rows = commol_rs.calibration.index_at(
        [float(position) for position in positions],
        [float(position) for position in at],
    )
    return [series[row] for row in rows]
