"""``commol.plot_compared``: runs, replicates and predictives on shared panels,
along the axis or one panel per observed series."""

import numpy as np
import pytest
from matplotlib.colors import to_rgba

import commol
from commol import (
    NoNoise,
    ObservedDataPoint,
    Simulation,
    SimulationResults,
)
from commol.api.inference.intervals import member_statistics
from tests.support import (
    a_to_b_builder,
    abc_builder,
    build,
    build_stochastic,
)
from tests.support.plotted import (
    MODEL_BAND,
    TITLES,
    band_bars,
    dated_run,
    legend_texts,
    predictive_of,
    scatter,
    tick_labels,
    visible,
)


def _abc_run(k1: float, accumulator: str | None = None) -> SimulationResults:
    """20 steps of ``abc_builder`` at ``k1``."""
    builder = abc_builder(parameters={"k1": k1, "k2": 0.1}, accumulator=accumulator)
    return Simulation(build(builder)).run(20)


def _colors(ax) -> list[tuple[float, float, float]]:
    """The color of each line of a panel, with no alpha."""
    return [to_rgba(line.get_color())[:3] for line in ax.lines]


def test_compared_runs_share_the_panels() -> None:
    """
    Each panel has one line per run, labelled with its key, and a run has
    the same color in every panel.
    """
    slow, fast = _abc_run(0.3), _abc_run(0.6)

    fig = commol.plot_compared({"slow": slow, "fast": fast}, series=["A", "B"])

    axes = visible(fig)
    assert [ax.get_label() for ax in axes] == ["A", "B"]
    for ax, name in zip(axes, ["A", "B"], strict=True):
        assert ax.get_title() == TITLES[name]
        assert [line.get_label() for line in ax.lines] == ["slow", "fast"]
        np.testing.assert_allclose(
            [line.get_ydata() for line in ax.lines], [slow[name], fast[name]]
        )
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "slow",
            "fast",
        ]
    first, second = (_colors(ax) for ax in axes)
    assert first == second
    assert first[0] != first[1]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (("abc", "ab"), r"'ab' does not have the outputs of 'abc': it lacks \['C'\]"),
        (
            ("ab", "abc"),
            r"'abc' does not have the outputs of 'ab': it also has \['C'\]",
        ),
    ],
    ids=["missing", "extra"],
)
def test_compared_runs_need_the_same_outputs(
    labels: tuple[str, str], message: str
) -> None:
    runs = {
        "abc": _abc_run(0.3),
        "ab": Simulation(build(a_to_b_builder())).run(20),
    }

    with pytest.raises(ValueError, match=message):
        commol.plot_compared({label: runs[label] for label in labels})


def test_a_comparison_needs_a_result() -> None:
    with pytest.raises(ValueError, match="at least one result"):
        commol.plot_compared({})


def test_a_compared_predictive_shows_its_band() -> None:
    """
    The predictive shows the median and the band of its draws in the color
    of its key; the run shows its line; the observations are drawn once.
    """
    predictive = predictive_of([0.02, 0.1])
    model = build(
        a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
    )
    run = Simulation(model).run(40)
    observed = [ObservedDataPoint(at=at, output="B", value=50.0) for at in (10, 20)]

    fig = commol.plot_compared(
        {"draws": predictive, "run": run},
        series=["B"],
        observed_data=observed,
        confidence_level=0.9,
    )

    (ax,) = fig.axes
    band = predictive.series_band(["B"], 0.9)
    median, line = ax.lines
    np.testing.assert_allclose(np.asarray(median.get_ydata()), band.median)
    np.testing.assert_allclose(np.asarray(line.get_ydata()), run["B"])
    filled, points = ax.collections
    heights = filled.get_paths()[0].vertices[:, 1]
    assert heights.min() == pytest.approx(min(band.lower))
    assert heights.max() == pytest.approx(max(band.upper))
    assert to_rgba(filled.get_facecolor()[0])[:3] == to_rgba(median.get_color())[:3]
    np.testing.assert_allclose(points.get_offsets(), [[10.0, 50.0], [20.0, 50.0]])
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == sorted(
        ["draws", "draws: 90% of the draws", "run", "Observed"]
    )


def test_compared_replicates_show_their_band() -> None:
    """Replicates show their mean and the band of the replicates."""
    model = build_stochastic(a_to_b_builder(), "StochasticSteps")
    replicates = Simulation(model).replicates(5, seed=2).run(20)
    run = Simulation(build(a_to_b_builder())).run(20)

    fig = commol.plot_compared({"replicates": replicates, "run": run}, series=["B"])

    (ax,) = fig.axes
    mean, line = ax.lines
    np.testing.assert_allclose(np.asarray(mean.get_ydata()), replicates.mean()["B"])
    np.testing.assert_allclose(np.asarray(line.get_ydata()), run["B"])
    (band,) = ax.collections
    heights = band.get_paths()[0].vertices[:, 1]
    values = [replicate["B"] for replicate in replicates]
    assert heights.min() >= np.min(values) - 1e-9
    assert heights.max() <= np.max(values) + 1e-9
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "replicates",
        "replicates: 95% of the replicates",
        "run",
    ]


def test_compared_runs_can_be_windowed() -> None:
    """Each line is the gain of the accumulator over each window of 7."""
    slow, fast = _abc_run(0.3, "moved"), _abc_run(0.6, "moved")
    simulation = Simulation(build(abc_builder(accumulator="moved")))

    fig = commol.plot_compared({"slow": slow, "fast": fast}, series=["moved"], window=7)

    (ax,) = fig.axes
    for line, run in zip(ax.lines, (slow, fast), strict=True):
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == [7.0, 14.0]
        np.testing.assert_allclose(
            np.asarray(line.get_ydata()),
            simulation.windowed_totals(run, ["moved"], window=7),
        )


def test_compared_replicates_and_draws_can_be_windowed() -> None:
    """
    Replicates show the mean gain over each window and the band of the gains
    of the replicates; a predictive the median and the band of the gains of
    its draws, read through the observation operator.
    """
    model = build_stochastic(
        a_to_b_builder(parameters={"k1": 0.1}, rate="k1", accumulator="events"),
        "StochasticSteps",
    )
    replicates = Simulation(model).replicates(5, seed=2).run(40)
    predictive = predictive_of([0.02, 0.1])

    fig = commol.plot_compared(
        {"replicates": replicates, "draws": predictive},
        series=["events"],
        window=10,
        confidence_level=0.9,
    )

    (ax,) = fig.axes
    ends = [10.0, 20.0, 30.0, 40.0]
    gains = [
        [run["events"][int(end)] - run["events"][int(end) - 10] for end in ends]
        for run in replicates
    ]
    mean, median = ax.lines
    np.testing.assert_allclose(np.asarray(mean.get_ydata()), np.mean(gains, axis=0))
    points = [
        ObservedDataPoint(at=end, output="events", value=0.0, window=10) for end in ends
    ]
    draws = predictive.at_observations(points, commol.NoNoise()).band(0.9)
    assert np.asarray(median.get_xdata(), dtype=float).tolist() == ends
    np.testing.assert_allclose(np.asarray(median.get_ydata()), draws.median)
    replicate_band, draws_band = ax.collections
    heights = replicate_band.get_paths()[0].vertices[:, 1]
    assert heights.min() >= np.min(gains) - 1e-9
    assert heights.max() <= np.max(gains) + 1e-9
    assert draws_band.get_paths()[0].vertices[:, 1].max() == pytest.approx(
        max(draws.upper)
    )
    assert [t.get_text() for t in ax.get_legend().get_texts()][1::2] == [
        "replicates: 90% of the replicates",
        "draws: 90% of the draws",
    ]


def test_compared_results_draw_on_given_axes_with_dates() -> None:
    """
    With ``axes=``, each series is drawn on its panel; a model with an
    origin labels the ticks with dates.
    """
    first = dated_run("day", 20)
    fig = first.plot.series(series=["B"])

    again = commol.plot_compared(
        {"first": first, "again": first},
        series=["B"],
        axes=fig,
        tick_every=10,
        show_legend=False,
    )

    assert again is fig
    (ax,) = fig.axes
    assert [line.get_label() for line in ax.lines] == ["Simulation", "first", "again"]
    assert tick_labels(fig) == {
        0.0: "2020-01-01",
        10.0: "2020-01-11",
        20.0: "2020-01-21",
    }


def _events_run(k: float) -> tuple[Simulation, SimulationResults]:
    """30 steps of a stratified model from A to B at ``k`` that counts its events."""
    simulation = Simulation(
        build(
            a_to_b_builder(
                parameters={"k": k},
                accumulator="events",
                stratifications={"group": {"g1": 0.2, "g2": 0.3, "g3": 0.5}},
            )
        )
    )
    return simulation, simulation.run(30)


def test_compared_results_draw_a_panel_per_observed_series() -> None:
    """
    One panel per label of the observations, with each result read through
    the operator of each observation in its own color, and the observations
    drawn once.
    """
    simulation, slow = _events_run(0.05)
    _, fast = _events_run(0.2)
    observed = [
        ObservedDataPoint(at=at, output="B_g1", value=1.0) for at in (10, 20)
    ] + [
        ObservedDataPoint(
            at=at,
            outputs=["events_g2", "events_g3"],
            label="events_late",
            value=2.0,
            window=5,
        )
        for at in (10, 20, 30)
    ]

    fig = commol.plot_compared(
        {"slow": slow, "fast": fast}, observed_data=observed, panels="observations"
    )

    one, summed = visible(fig)
    assert [ax.get_label() for ax in (one, summed)] == ["B_g1", "events_late"]
    for run, markers in zip((slow, fast), _markers(summed), strict=True):
        np.testing.assert_allclose(
            markers,
            [
                [10, 20, 30],
                simulation.windowed_totals(
                    run, ["events_g2", "events_g3"], 5, [10, 20, 30]
                ),
            ],
        )
    for run, markers in zip((slow, fast), _markers(one), strict=True):
        np.testing.assert_allclose(
            markers, [[10, 20], simulation.sample(run, ["B_g1"], [10, 20])]
        )
    for ax in (one, summed):
        (points,) = ax.collections
        assert points.get_label() == "Observed"
        assert [to_rgba(line.get_color())[:3] for line in ax.lines[::2]] == [
            to_rgba("C0")[:3],
            to_rgba("C1")[:3],
        ]
        assert legend_texts(ax) == {"slow", "fast", "Observed"}
    np.testing.assert_allclose(scatter(summed), [[10, 2.0], [20, 2.0], [30, 2.0]])


def _markers(ax) -> list[np.ndarray]:
    """The positions and values of the markers of each model of a panel."""
    return [
        np.asarray([line.get_xdata(), line.get_ydata()], dtype=float)
        for line in ax.lines
        if line.get_marker() == "o"
    ]


def test_compared_observed_series_show_the_band_of_each_result() -> None:
    """
    Replicates show their mean and the band across them, and a predictive
    the median and the band of its draws, read through the observation
    operator; each band in the color of its key, its legend led by the key.
    """
    model = build_stochastic(
        a_to_b_builder(parameters={"k1": 0.1}, rate="k1", accumulator="events"),
        "StochasticSteps",
    )
    simulation = Simulation(model)
    replicates = simulation.replicates(5, seed=2).run(40)
    predictive = predictive_of([0.02, 0.1])
    observed = [
        ObservedDataPoint(at=at, output="events", value=1.0, window=10)
        for at in (20, 30)
    ]

    fig = commol.plot_compared(
        {"replicates": replicates, "draws": predictive},
        observed_data=observed,
        panels="observations",
        confidence_level=0.9,
    )

    (ax,) = fig.axes
    per_run = [
        simulation.windowed_totals(run, ["events"], 10, [20, 30]) for run in replicates
    ]
    statistics = member_statistics(per_run, 0.9)
    draws = predictive.at_observations(observed, NoNoise()).band(0.9)
    mean, median = _markers(ax)
    np.testing.assert_allclose(mean[1], np.mean(per_run, axis=0))
    np.testing.assert_allclose(median[1], draws.median)
    bars = band_bars(ax)
    expected = [
        (at - 10, 10, lower, upper - lower)
        for lower_upper in (
            zip(statistics["ci_lower"], statistics["ci_upper"], strict=True),
            zip(draws.lower, draws.upper, strict=True),
        )
        for at, (lower, upper) in zip((20, 30), lower_upper, strict=True)
    ]
    np.testing.assert_allclose(bars, expected)
    colors = [to_rgba(patch.get_facecolor())[:3] for patch in ax.patches]
    assert colors == [to_rgba(c)[:3] for c in ("C0", "C0", "C1", "C1")]
    assert legend_texts(ax) == {
        "replicates",
        f"replicates: 90% of the replicates:\n{MODEL_BAND}",
        "draws",
        f"draws: 90% of the draws:\n{MODEL_BAND}",
        "Observed",
    }


def test_compared_observed_series_draw_on_given_axes() -> None:
    """The panel of a label that names a series is the panel of that series."""
    _, slow = _events_run(0.05)
    _, fast = _events_run(0.2)
    fig = slow.plot.series(series=["B_g1"], show_legend=False)
    observed = [ObservedDataPoint(at=at, output="B_g1", value=1.0) for at in (5, 12)]

    again = commol.plot_compared(
        {"slow": slow, "fast": fast},
        observed_data=observed,
        panels="observations",
        axes=fig,
        show_legend=False,
    )

    assert again is fig
    (ax,) = visible(fig)
    assert len(_markers(ax)) == 2
    assert ax.get_legend() is None


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"series": ["B_g1"]}, "give no `series` and no `window`"),
        ({"window": 5.0}, "give no `series` and no `window`"),
        ({"observed_data": []}, "at least one observation"),
        ({"panels": "bins"}, "`panels` is 'series' or 'observations', not 'bins'"),
    ],
    ids=["series", "window", "no observations", "unknown panels"],
)
def test_compared_observed_series_reject_what_they_cannot_draw(
    arguments: dict, message: str
) -> None:
    _, run = _events_run(0.1)
    arguments = {
        "observed_data": [ObservedDataPoint(at=10, output="B_g1", value=1.0)],
        "panels": "observations",
        **arguments,
    }

    with pytest.raises(ValueError, match=message):
        commol.plot_compared({"run": run}, **arguments)
