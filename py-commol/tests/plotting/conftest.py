"""Figures draw on a non-interactive backend, and every figure is closed.

The results of the sensitivity methods and of a profile likelihood that the
figures draw are built here, so each test file can draw them.
"""

from collections.abc import Iterator

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from commol import (  # noqa: E402
    CalibrationResult,
    MorrisResult,
    ParameterSample,
    PRCCResult,
    Profile,
    ProfileInterval,
    ProfileResult,
    SampleMethod,
    SobolResult,
    Threshold,
)

INPUTS = ["k1", "k2", "k3"]


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close the figures a test opens."""
    yield
    plt.close("all")


@pytest.fixture
def default_style() -> Iterator[dict[str, object]]:
    """
    The defaults of ``matplotlib.rcParams`` for one test, restored after it,
    so a figure that changes the global style changes no other test.
    """
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        yield dict(matplotlib.rcParams)


@pytest.fixture
def prcc_result() -> PRCCResult:
    """Coefficients of three inputs: a series over three positions, a scalar."""
    return PRCCResult(
        sample=ParameterSample.equally_weighted(
            INPUTS, [(0.0, 1.0)] * 3, [[0.1, 0.2, 0.3]], SampleMethod(name="test")
        ),
        inputs=INPUTS,
        labels=["C at 0", "C at 10", "C at 20", "peak of B"],
        series=["C", "C", "C", "peak of B"],
        positions=[0.0, 10.0, 20.0, None],
        coefficients=[
            [0.1, 0.2, 0.3],
            [0.5, -0.4, 0.0],
            [0.9, -0.8, 0.1],
            [0.7, -0.6, 0.0],
        ],
        p_values=[[0.5] * 3] * 4,
        effective_size=100.0,
    )


@pytest.fixture
def morris_result() -> MorrisResult:
    """The Morris indices of three inputs for one quantity."""
    return MorrisResult(
        inputs=INPUTS,
        labels=["peak of B"],
        mu=[[1.0, -2.0, 0.0]],
        mu_star=[[1.0, 2.0, 0.0]],
        sigma=[[0.5, 0.1, 0.0]],
        runs=40,
    )


@pytest.fixture
def sobol_result() -> SobolResult:
    """The Sobol indices of three inputs for one quantity, with intervals."""
    return SobolResult(
        inputs=INPUTS,
        labels=["peak of B"],
        confidence_level=0.9,
        first=[[0.3, 0.4, 0.0]],
        total=[[0.5, 0.45, 0.2]],
        first_lower=[[0.25, 0.35, 0.0]],
        first_upper=[[0.35, 0.45, 0.05]],
        total_lower=[[0.45, 0.4, 0.15]],
        total_upper=[[0.55, 0.5, 0.25]],
        runs=100,
    )


def _profile(center: float, threshold: float) -> Profile:
    values = [center - 2.0, center - 1.0, center, center + 1.0, center + 2.0]
    return Profile(
        center=center,
        threshold=threshold,
        values=values,
        statistics=[(v - center) ** 2 for v in values],
        parameters=[{"k1": v, "k2": 0.5} for v in values],
        multipliers=[None] * 5,
        converged=[True] * 5,
        ends=("crossed", "crossed"),
        evaluations=40,
    )


@pytest.fixture
def profile_result() -> ProfileResult:
    """Two parabolic profiles; the interval of ``k2`` is open above."""
    return ProfileResult(
        fit=CalibrationResult(
            best_parameters={"k1": 1.0, "k2": 3.0},
            final_loss=0.5,
            iterations=10,
            converged=True,
            termination_reason="converged",
            seed=0,
        ),
        quantity="parameter",
        threshold=Threshold.chi_square(0.95),
        profiles={"k1": _profile(1.0, 1.92), "k2": _profile(3.0, 1.5)},
        intervals={
            "k1": ProfileInterval(lower=1.0 - 1.92**0.5, upper=1.0 + 1.92**0.5),
            "k2": ProfileInterval(lower=3.0 - 1.5**0.5, upper=None),
        },
        verdicts={"k1": "identifiable", "k2": "practically_non_identifiable"},
        tolerance=1e-9,
    )
