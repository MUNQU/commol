"""The threads of the parallel methods and the levels that run in parallel."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, cast

from commol.commol_rs import _commol_rs as commol_rs
from commol.context.engine_json import engine_json

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import ParallelismProtocol

Levels = Literal["all", "outer"]
"""Which levels run in parallel when a method nests them: `"all"`, or
`"outer"` to run the particles of a swarm one after the other."""


def _choice(n: int | None, levels: Levels | None) -> "ParallelismProtocol":
    """The current choice with `n` and `levels` in place of the given ones."""
    current = commol_rs.parallelism.current()
    return commol_rs.parallelism.Parallelism(
        current.threads if n is None else n,
        engine_json(current.levels if levels is None else levels),
    )


def set_threads(n: int, levels: Levels | None = None) -> None:
    """
    Set the number of threads of every parallel method, in this context.

    The setting holds in the current Python thread (a `contextvars`
    context), not in other threads. Without it, commol uses one thread per
    core and every level in parallel.

    Parameters
    ----------
    n : int
        The number of threads, at least 1.
    levels : {"all", "outer"} | None
        Which levels run in parallel when a method nests them; `None` keeps
        the current levels.

    Raises
    ------
    ValueError
        If `n` is below 1 or `levels` is neither name.
    """
    commol_rs.parallelism.setting().set(_choice(n, levels))


@contextmanager
def threads(n: int | None = None, levels: Levels | None = None) -> Generator[None]:
    """
    Limit the threads of every parallel method within a block.

    The previous setting comes back when the block ends, also on an error.

    Parameters
    ----------
    n : int | None
        The number of threads, at least 1; `None` keeps the current number.
    levels : {"all", "outer"} | None
        Which levels run in parallel when a method nests them; `None` keeps
        the current levels.

    Yields
    ------
    None
        The block runs with the setting.

    Raises
    ------
    ValueError
        If `n` is below 1 or `levels` is neither name.
    """
    variable = commol_rs.parallelism.setting()
    token = variable.set(_choice(n, levels))
    try:
        yield
    finally:
        variable.reset(token)


def get_threads() -> int:
    """
    The number of threads of the parallel methods in this context.

    Returns
    -------
    int
        The number set, or the number of cores when none is set.
    """
    return commol_rs.parallelism.current().threads


def get_levels() -> Levels:
    """
    The levels that run in parallel in this context.

    Returns
    -------
    {"all", "outer"}
        The levels set, or `"all"` when none is set.
    """
    return cast(Levels, commol_rs.parallelism.current().levels)
