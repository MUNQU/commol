"""SimulationPlotter draws profiles: one panel per profile, with its trace,
its threshold and its interval."""

import numpy as np

from commol import (
    CalibrationResult,
    Profile,
    ProfileInterval,
    ProfileResult,
    Threshold,
)
from commol.api.plotter import SimulationPlotter


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


def test_plotter_draws_each_profile_and_threshold() -> None:
    result = ProfileResult(
        fit=CalibrationResult(
            best_parameters={"k1": 1.0, "k2": 3.0},
            final_loss=0.5,
            iterations=10,
            converged=True,
            termination_reason="converged",
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

    figure = SimulationPlotter.plot_profiles(result)

    axes = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == ["k1", "k2"]
    for ax, name in zip(axes, ["k1", "k2"], strict=True):
        profile = result.profiles[name]
        trace, threshold, *interval = ax.get_lines()
        np.testing.assert_allclose(
            np.asarray(trace.get_xdata(), dtype=float), profile.values
        )
        np.testing.assert_allclose(
            np.asarray(trace.get_ydata(), dtype=float), profile.statistics
        )
        np.testing.assert_allclose(
            np.asarray(threshold.get_ydata(), dtype=float), [profile.threshold] * 2
        )
        ends = [
            value
            for value in (result.intervals[name].lower, result.intervals[name].upper)
            if value is not None
        ]
        assert [line.get_xdata()[0] for line in interval] == ends
        assert ax.get_xlabel() == name
