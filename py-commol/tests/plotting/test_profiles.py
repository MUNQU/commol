"""A profile result draws its figure: one panel per profile, with its trace,
its threshold and its interval."""

import numpy as np

from commol import ProfileResult


def test_each_profile_is_drawn_with_its_threshold(
    profile_result: ProfileResult,
) -> None:
    result = profile_result

    figure = result.plot()

    axes = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in axes] == ["k1", "k2"]
    assert [ax.get_label() for ax in axes] == ["k1", "k2"]
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
