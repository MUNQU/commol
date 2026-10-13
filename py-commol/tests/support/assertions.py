"""Assertions on simulation results that state their tolerance."""

import math
from collections.abc import Iterable, Mapping, Sequence

Series = Mapping[str, Sequence[float]]
Values = Mapping[str, float]
"""One value per output, such as the state at one position."""

ENGINE_REL = 1e-9
"""
Relative tolerance between the engine and a hand computation of the same
arithmetic. The two only differ in the order of floating-point operations.
"""

ODE_REL = 1e-6
"""
Relative tolerance between a differential model, solved with the ``TIGHT``
settings, and a stated result: a closed form, a final size or an equilibrium.
"""


def assert_close(
    actual: float | Sequence[float] | Series | Values,
    expected: float | Sequence[float] | Series | Values,
    *,
    rel: float = 0.0,
    abs: float = 0.0,
) -> None:
    """
    Assert that two numbers, sequences or series match within a tolerance.

    Parameters
    ----------
    actual, expected : float | Sequence[float] | Series | Values
        Both of the same shape. Series and values must have the same keys.
    rel, abs : float
        The relative and the absolute tolerance, as in ``math.isclose``. At
        least one must be given, so every comparison states its tolerance.

    Raises
    ------
    AssertionError
        At the first value outside the tolerance, with its key and index.
    TypeError
        If no tolerance is given.
    """
    if rel <= 0.0 and abs <= 0.0:
        raise TypeError("assert_close needs a tolerance: pass rel, abs or both")
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"expected series, got {type(actual)}"
        assert set(actual) == set(expected), (
            f"series differ: {sorted(actual)} != {sorted(expected)}"
        )
        for name in expected:
            got, want = actual[name], expected[name]
            if isinstance(want, Sequence):
                _assert_sequence_close(got, want, rel, abs, name)
            else:
                assert math.isclose(got, want, rel_tol=rel, abs_tol=abs), (
                    f"{name}: {got!r} != {want!r} (rel={rel}, abs={abs})"
                )
        return
    if isinstance(expected, Sequence):
        assert isinstance(actual, Sequence), f"expected a sequence, got {actual!r}"
        _assert_sequence_close(actual, expected, rel, abs, "value")
        return
    assert isinstance(actual, (int, float)), f"expected a number, got {actual!r}"
    assert math.isclose(actual, expected, rel_tol=rel, abs_tol=abs), (
        f"{actual!r} != {expected!r} (rel={rel}, abs={abs})"
    )


def _assert_sequence_close(
    actual: Sequence[float],
    expected: Sequence[float],
    rel: float,
    abs: float,
    name: str,
) -> None:
    assert len(actual) == len(expected), (
        f"{name}: {len(actual)} values, expected {len(expected)}"
    )
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert math.isclose(got, want, rel_tol=rel, abs_tol=abs), (
            f"{name}[{index}]: {got!r} != {want!r} (rel={rel}, abs={abs})"
        )


def assert_population_conserved(
    results: Series,
    total: float,
    *,
    outputs: Iterable[str] | None = None,
    rel: float = ENGINE_REL,
) -> None:
    """
    Assert that the outputs sum to ``total`` at every step.

    Parameters
    ----------
    results : Series
        A simulation result in the ``dict_of_lists`` format.
    total : float
        The population every step must hold.
    outputs : Iterable[str] | None
        The outputs to sum. Defaults to every output, so a model with
        accumulators must name its compartments.
    rel : float
        Relative tolerance of each step's sum.
    """
    names = list(results if outputs is None else outputs)
    steps = len(results[names[0]])
    for step in range(steps):
        held = sum(results[name][step] for name in names)
        assert math.isclose(held, total, rel_tol=rel), (
            f"step {step}: the outputs hold {held!r}, expected {total!r}"
        )
