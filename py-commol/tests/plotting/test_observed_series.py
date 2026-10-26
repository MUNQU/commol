"""``results.plot.observations()``: one panel per observed series, the model
read through the operator of each observation."""

import numpy as np
import pytest

from commol import (
    NoNoise,
    ObservedDataPoint,
    Simulation,
)
from commol.api.inference.intervals import member_statistics
from tests.support import (
    a_to_b_builder,
    build,
    build_stochastic,
)
from tests.support.plotted import (
    MODEL_BAND,
    band_bars,
    fit,
    legend_texts,
    predictive_of,
    scatter,
    visible,
)


def _model_points(ax) -> np.ndarray:
    """The positions and values of the markers of the model of a panel."""
    (markers,) = [line for line in ax.lines if line.get_marker() == "o"]
    assert markers.get_linestyle() == "None"
    return np.asarray([markers.get_xdata(), markers.get_ydata()], dtype=float)


def _events_model(**stratifications: dict[str, float]) -> Simulation:
    """A model from A to B, stratified as given, that counts its events."""
    return Simulation(
        build(
            a_to_b_builder(
                accumulator="events", stratifications=stratifications or None
            )
        )
    )


def test_a_panel_per_observed_series_reads_the_observation_operator() -> None:
    """
    One panel per label, in the order of its first observation: a sum of two
    outputs over a window is the windowed sum of the run at each position,
    and one output with no window is its value there.
    """
    simulation = _events_model(group={"g1": 0.2, "g2": 0.3, "g3": 0.5})
    run = simulation.run(30)
    observed = [
        ObservedDataPoint(at=at, output="B_g1", value=1.0) for at in (10, 20)
    ] + [
        ObservedDataPoint(
            at=at,
            outputs=["events_g2", "events_g3"],
            label="events_g2_g3",
            value=2.0,
            window=5,
        )
        for at in (30, 10, 20)
    ]

    fig = run.plot.observations(observed)

    one, summed = visible(fig)
    assert [ax.get_label() for ax in (one, summed)] == ["B_g1", "events_g2_g3"]
    assert [ax.get_title() for ax in (one, summed)] == ["B_g1", "events_g2_g3"]
    np.testing.assert_allclose(
        _model_points(summed),
        [
            [10, 20, 30],
            run.windowed_totals(["events_g2", "events_g3"], 5, [10, 20, 30]),
        ],
    )
    np.testing.assert_allclose(
        _model_points(one), [[10, 20], run.sample(["B_g1"], [10, 20])]
    )
    np.testing.assert_allclose(scatter(summed), [[10, 2.0], [20, 2.0], [30, 2.0]])
    assert legend_texts(summed) == {"Simulation", "Observed"}


def test_the_panel_of_an_observed_series_applies_its_factor() -> None:
    simulation = _events_model()
    run = simulation.run(20)
    observed = [
        ObservedDataPoint(at=at, output="B", value=1.0, factor_id="share")
        for at in (10, 20)
    ]

    fig = run.plot.observations(observed, calibration_result=fit(share=0.5))

    (ax,) = fig.axes
    unscaled = run.sample(["B"], [10, 20])
    np.testing.assert_allclose(_model_points(ax)[1], 0.5 * np.asarray(unscaled))
    np.testing.assert_allclose(scatter(ax)[:, 1], [1.0, 1.0])


def test_the_panel_of_an_observed_series_shows_the_band_of_a_predictive() -> None:
    """
    The median and the band of each panel are those of the `ObservationDraws`
    of its series: a rectangle over each window, and a band filled between
    the values with no window.
    """
    predictive = predictive_of([0.02, 0.05, 0.1, 0.2])
    windowed = [
        ObservedDataPoint(at=at, output="events", value=1.0, window=5)
        for at in (10, 20)
    ]
    single = [ObservedDataPoint(at=at, output="B", value=1.0) for at in (15, 25, 35)]

    fig = predictive.plot.observations(windowed + single, confidence_level=0.9)

    events, b = visible(fig)
    expected = predictive.at_observations(windowed, NoNoise()).band(0.9)
    np.testing.assert_allclose(_model_points(events), [[10, 20], expected.median])
    np.testing.assert_allclose(
        band_bars(events),
        [
            (at - 5, 5, lower, upper - lower)
            for at, lower, upper in zip(
                (10, 20), expected.lower, expected.upper, strict=True
            )
        ],
    )
    assert legend_texts(events) == {
        "Median of the draws",
        f"90% of the draws:\n{MODEL_BAND}",
        "Observed",
    }
    expected = predictive.at_observations(single, NoNoise()).band(0.9)
    np.testing.assert_allclose(_model_points(b), [[15, 25, 35], expected.median])
    (filled,) = [c for c in b.collections if c.get_label() != "Observed"]
    vertices = {tuple(v) for v in filled.get_paths()[0].vertices.tolist()}
    for at, lower, upper in zip((15, 25, 35), expected.lower, expected.upper):
        assert any(x == at and y == pytest.approx(lower) for x, y in vertices)
        assert any(x == at and y == pytest.approx(upper) for x, y in vertices)
    assert legend_texts(b) == {
        "Median of the draws",
        f"90% of the draws:\n{MODEL_BAND}",
        "Observed",
    }


def test_each_observed_point_is_read_through_its_own_window() -> None:
    simulation = _events_model()
    run = simulation.run(30)
    observed = [
        ObservedDataPoint(at=10, output="events", value=1.0, window=7),
        ObservedDataPoint(at=28, output="events", value=1.0, window=14),
    ]

    fig = run.plot.observations(observed)

    steps = fig.axes[0].lines[0]
    first = run.windowed_totals(["events"], 7, [10])[0]
    second = run.windowed_totals(["events"], 14, [28])[0]
    np.testing.assert_allclose(
        steps.get_xdata(), [3, 10, np.nan, 14, 28, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        steps.get_ydata(),
        [first, first, np.nan, second, second, np.nan],
        equal_nan=True,
    )


def test_a_windowed_value_is_drawn_as_a_step_over_its_window() -> None:
    """
    Each value is a step over ``(at - window, at]`` with a marker at ``at``,
    of the colour of the step; no line joins two windows.
    """
    run = _events_model().run(20)
    observed = [
        ObservedDataPoint(at=at, output="events", value=1.0, window=5)
        for at in (5, 10, 15, 20)
    ]

    fig = run.plot.observations(observed, linewidth=3)

    (ax,) = fig.axes
    steps, markers = ax.lines
    x = np.asarray(steps.get_xdata(), dtype=float).reshape(-1, 3)
    y = np.asarray(steps.get_ydata(), dtype=float).reshape(-1, 3)
    np.testing.assert_allclose(x[:, :2], [[0, 5], [5, 10], [10, 15], [15, 20]])
    assert np.isnan(x[:, 2]).all() and np.isnan(y[:, 2]).all()
    np.testing.assert_allclose(y[:, 0], y[:, 1])
    np.testing.assert_allclose(_model_points(ax), [[5, 10, 15, 20], y[:, 1]])
    assert steps.get_linewidth() == 3
    assert markers.get_color() == steps.get_color()


def test_a_value_with_no_window_is_a_marker_joined_as_a_guide() -> None:
    simulation = _events_model()
    run = simulation.run(20)
    observed = [ObservedDataPoint(at=at, output="B", value=1.0) for at in (5, 12)]

    fig = run.plot.observations(observed)

    (ax,) = fig.axes
    guide, _ = ax.lines
    assert guide.get_linestyle() == ":"
    assert guide.get_label() == "Simulation"
    np.testing.assert_allclose(
        [guide.get_xdata(), guide.get_ydata()],
        [[5, 12], run.sample(["B"], [5, 12])],
    )
    np.testing.assert_allclose(
        _model_points(ax), [guide.get_xdata(), guide.get_ydata()]
    )


def test_the_panel_of_replicates_shows_their_mean_and_band() -> None:
    """
    A label that mixes a window and a value with no window: the mean of the
    replicates and the band across them, each replicate read through the
    operator of each observation.
    """
    simulation = Simulation(
        build_stochastic(a_to_b_builder(accumulator="events"), "StochasticSteps")
    )
    replicates = simulation.replicates(5, seed=2).run(20)
    observed = [
        ObservedDataPoint(at=10, output="events", value=1.0, window=5),
        ObservedDataPoint(at=20, output="events", value=1.0),
    ]

    fig = replicates.plot.observations(observed, confidence_level=0.8)

    (ax,) = fig.axes
    per_run = [
        [
            run.windowed_totals(["events"], 5, [10])[0],
            run.sample(["events"], [20])[0],
        ]
        for run in replicates
    ]
    statistics = member_statistics(per_run, 0.8)
    np.testing.assert_allclose(_model_points(ax)[1], np.mean(per_run, axis=0))
    ((left, width, bottom, height),) = band_bars(ax)
    assert (left, width) == (5, 5)
    assert bottom == pytest.approx(statistics["ci_lower"][0])
    assert bottom + height == pytest.approx(statistics["ci_upper"][0])
    assert legend_texts(ax) == {
        "Mean of 5 replicates",
        f"80% of the replicates:\n{MODEL_BAND}",
        "Observed",
    }


def test_observed_series_draw_on_the_panels_of_a_figure() -> None:
    """
    The run along the whole axis and its observed series share the panels
    of the labels that name a series.
    """
    run = _events_model().run(20)
    fig = run.plot.series(series=["B"])
    observed = [ObservedDataPoint(at=at, output="B", value=1.0) for at in (5, 12)]

    again = run.plot.observations(observed, axes=fig)

    assert again is fig
    (ax,) = visible(fig)
    assert len(ax.lines) == 3


def test_a_figure_of_observed_series_needs_observations() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        _events_model().run(5).plot.observations([])
