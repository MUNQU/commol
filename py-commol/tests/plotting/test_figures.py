"""The figures of a run, of replicates and of a predictive are read from them:
``results.plot.series()`` and ``results.plot.cumulative()``. Each draws the
series, observations and bands of its results, and returns the `Figure`."""

import inspect
import subprocess
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

import commol
from commol import (
    CalibrationProblem,
    Calibrator,
    ChaosResult,
    MorrisResult,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    PlotConfig,
    PRCCResult,
    Predictive,
    ProfileResult,
    SampleMethod,
    Simulation,
    SimulationResults,
    SobolResult,
)
from commol.api.plotting import figures
from commol.api.plotting.accessors import PredictivePlot, ReplicatesPlot, ResultsPlot
from commol.api.plotting.panels import _group_observed
from commol.context.calibration import CalibrationResult
from tests.support import (
    a_to_b_builder,
    abc_builder,
    bounded,
    build,
    build_ode,
    build_stochastic,
)

BINS = ["A", "B", "C"]
TITLES = {"A": "State A", "B": "State B", "C": "State C"}


@pytest.fixture
def run(abc_simulation: Simulation) -> SimulationResults:
    """100 steps of the default model of ``abc_builder``."""
    return abc_simulation.run(100)


def _visible(fig: Figure):
    return [ax for ax in fig.axes if ax.get_visible()]


def _assert_draws(fig: Figure, expected: dict[str, list[float]]) -> None:
    """One visible panel per series, titled by its bin, with its values."""
    axes = _visible(fig)
    assert [ax.get_title() for ax in axes] == [TITLES[name] for name in expected]
    for ax, values in zip(axes, expected.values(), strict=True):
        np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata()), values)


def _scatter(ax) -> np.ndarray:
    """The offsets of the observation markers of a panel."""
    offsets = [np.asarray(c.get_offsets(), dtype=float) for c in ax.collections]
    return offsets[int(np.argmax([len(o) for o in offsets]))]


def _observed_b(results: dict[str, list[float]]) -> list[ObservedDataPoint]:
    return [
        ObservedDataPoint(at=step, output="B", value=results["B"][step])
        for step in range(0, 100, 10)
    ]


def _fit(**factors: float) -> CalibrationResult:
    return CalibrationResult(
        best_parameters=factors,
        final_loss=0.0,
        iterations=1,
        converged=True,
        termination_reason="Test",
    )


def _calibrator() -> Calibrator:
    """A calibrator of ``k1`` of a model from A to B that counts its events."""
    model = build(
        a_to_b_builder(parameters={"k1": None}, rate="k1", accumulator="events")
    )
    problem = CalibrationProblem(
        observed_data=[ObservedDataPoint(at=10, output="events", value=1.0, window=5)],
        parameters=[bounded("k1", 0.0, 0.5)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    return Calibrator(Simulation(model), problem)


def _predictive(values: list[float]) -> Predictive:
    """The predictive of one draw of ``k1`` per value, at positions 0 to 40."""
    sample = ParameterSample.equally_weighted(
        ["k1"], [(0.0, 0.5)], [[value] for value in values], SampleMethod(name="t")
    )
    return Predictive(_calibrator(), sample, list(range(41)))


def test_the_figure_of_a_run_is_read_from_the_run() -> None:
    """
    One panel per output, labelled by the id of its series, titled and with
    the unit the kind gives it, with the line of the run.
    """
    run = Simulation(build(abc_builder(bin_unit="item"))).run(10)

    fig = run.plot.series()

    axes = _visible(fig)
    assert [ax.get_label() for ax in axes] == BINS
    assert [ax.get_title() for ax in axes] == [TITLES[name] for name in BINS]
    assert {(ax.get_xlabel(), ax.get_ylabel()) for ax in axes} == {("Step", "item")}
    for ax, name in zip(axes, BINS, strict=True):
        (line,) = ax.lines
        assert line.get_label() == "Simulation"
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == run.positions
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == run[name]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Simulation"]
        grid = ax.xaxis.get_gridlines()[0]
        assert (grid.get_visible(), grid.get_alpha()) == (True, 0.3)


def test_the_figure_of_a_predictive_is_read_from_the_predictive() -> None:
    predictive = _predictive([0.02, 0.1])

    fig = predictive.plot.series(series=["B"], confidence_level=0.9)

    (ax,) = fig.axes
    band = predictive.series_band(["B"], 0.9)
    (median,) = ax.lines
    np.testing.assert_allclose(np.asarray(median.get_ydata()), band.median)
    (filled,) = ax.collections
    heights = filled.get_paths()[0].vertices[:, 1]
    assert heights.min() == pytest.approx(min(band.lower))
    assert heights.max() == pytest.approx(max(band.upper))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Median of the draws",
        "90% of the draws",
    ]
    assert ax.get_title() == "B"


def test_a_figure_draws_on_given_axes(abc_simulation: Simulation) -> None:
    first = abc_simulation.run(20)
    second = Simulation(build(abc_builder(parameters={"k1": 0.6, "k2": 0.1}))).run(20)
    fig = first.plot.series(series=["A", "B"])

    again = second.plot.series(series=["B"], axes=fig)

    assert again is fig
    panel_a, panel_b = _visible(fig)
    assert len(panel_a.lines) == 1
    np.testing.assert_allclose(
        [line.get_ydata() for line in panel_b.lines], [first["B"], second["B"]]
    )


def test_a_cumulative_figure_draws_on_given_axes(run: SimulationResults) -> None:
    fig = run.plot.series(series=["C"])

    run.plot.cumulative(series=["C"], axes=fig)

    (ax,) = fig.axes
    np.testing.assert_allclose(ax.lines[1].get_ydata(), np.cumsum(run["C"]))


def test_a_series_with_no_panel_is_named(run: SimulationResults) -> None:
    fig = run.plot.series(series=["A", "B"])

    with pytest.raises(ValueError, match=r"no panel of \['C'\]"):
        run.plot.series(series=["B", "C"], axes=fig)


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

    axes = _visible(fig)
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
    predictive = _predictive([0.02, 0.1])
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
    predictive = _predictive([0.02, 0.1])

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
    first = _dated_run("day", 20)
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
    assert _tick_labels(fig) == {
        0.0: "2020-01-01",
        10.0: "2020-01-11",
        20.0: "2020-01-21",
    }


def test_results_built_by_hand_plot_only_outputs() -> None:
    results = SimulationResults(
        [0.0, 1.0, 2.0], {"B_g1": [1.0, 2.0, 3.0], "B_g2": [0.0, 1.0, 1.0]}
    )

    fig = results.plot.series()

    assert [ax.get_title() for ax in _visible(fig)] == ["B_g1", "B_g2"]
    assert {ax.get_ylabel() for ax in _visible(fig)} == {""}
    with pytest.raises(ValueError, match="'B' is not an output of the results"):
        results.plot.series(series=["B"])


def test_a_sensitivity_result_plots_itself(prcc_result: PRCCResult) -> None:
    """
    The PRCC of a series over positions is one line per input with markers;
    that of a scalar quantity, one bar per input; each panel lies in
    ``[-1.05, 1.05]`` with a line at zero.
    """
    fig = prcc_result.plot(x_label="Day")

    over_axis, bars = _visible(fig)
    assert [ax.get_label() for ax in (over_axis, bars)] == ["C", "peak of B"]
    *inputs, zero = over_axis.get_lines()
    for j, line in enumerate(inputs):
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == [0, 10, 20]
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == [
            prcc_result.coefficients[i][j] for i in range(3)
        ]
        assert line.get_marker() == "o"
    assert np.asarray(zero.get_ydata(), dtype=float).tolist() == [0.0, 0.0]
    assert [patch.get_height() for patch in bars.patches] == [0.7, -0.6, 0.0]
    for ax in (over_axis, bars):
        assert ax.get_ylim() == (-1.05, 1.05)
        assert ax.get_ylabel() == "PRCC"
    assert over_axis.get_xlabel() == "Day"


def test_figures_are_only_methods() -> None:
    assert not hasattr(commol, "SimulationPlotter")
    assert not hasattr(commol.api, "SimulationPlotter")
    assert not [name for name in commol.__all__ if name.startswith("_")]


ACCESSORS = [
    (ResultsPlot.series, figures._run_series),
    (ResultsPlot.cumulative, figures._run_cumulative),
    (ReplicatesPlot.series, figures._replicates_series),
    (ReplicatesPlot.cumulative, figures._replicates_cumulative),
    (PredictivePlot.series, figures._predictive_series),
    (PredictivePlot.cumulative, figures._predictive_cumulative),
    (PRCCResult.plot, figures._prcc),
    (MorrisResult.plot, figures._morris),
    (SobolResult.plot, figures._sobol),
    (ChaosResult.plot, figures._sobol),
    (ProfileResult.plot, figures._profiles),
    (commol.plot_compared, figures._compared_series),
]


@pytest.mark.parametrize(
    ("method", "draw"), ACCESSORS, ids=[m.__qualname__ for m, _ in ACCESSORS]
)
def test_each_accessor_passes_on_the_arguments_of_its_function(
    method: Callable, draw: Callable
) -> None:
    """
    After ``self`` and the result, a method and its drawing function have
    the same parameters, in order, with the same kinds and defaults.
    """

    def parameters(function: Callable) -> list[tuple[str, object, object]]:
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in list(inspect.signature(function).parameters.values())[1:]
        ]

    assert parameters(method) == parameters(draw)


def test_import_commol_loads_no_plotting_library() -> None:
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, commol; "
            "print(sorted({m.split('.')[0] for m in sys.modules} "
            "& {'matplotlib', 'seaborn'}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded.stdout.strip() == "[]"


def test_a_figure_without_matplotlib_says_how_to_install_it(
    run: SimulationResults, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "commol.api.plotting.figures")

    with pytest.raises(ImportError, match=r"pip install 'commol\[plot\]'"):
        run.plot.series()


STYLE = PlotConfig(style=["seaborn-v0_8-darkgrid", "seaborn-v0_8-poster"])


@pytest.mark.parametrize(
    "draw",
    [
        lambda run, _: run.plot.series(config=STYLE),
        lambda run, _: run.plot.cumulative(config=STYLE),
        lambda _, results: results["prcc"].plot(config=STYLE),
        lambda _, results: results["morris"].plot(config=STYLE),
        lambda _, results: results["sobol"].plot(config=STYLE),
        lambda _, results: results["profile"].plot(config=STYLE),
    ],
    ids=["series", "cumulative", "prcc", "morris", "sobol", "profile"],
)
def test_a_figure_leaves_the_global_style(
    run: SimulationResults,
    prcc_result: PRCCResult,
    morris_result: MorrisResult,
    sobol_result: SobolResult,
    profile_result: ProfileResult,
    default_style: dict[str, object],
    draw: Callable[[SimulationResults, dict[str, object]], Figure],
) -> None:
    """
    The styles of the configuration are in the figure, and
    ``matplotlib.rcParams`` is as it was before the call.
    """
    fig = draw(
        run,
        {
            "prcc": prcc_result,
            "morris": morris_result,
            "sobol": sobol_result,
            "profile": profile_result,
        },
    )

    assert {k for k, v in default_style.items() if mpl.rcParams[k] != v} == set()
    ax = _visible(fig)[0]
    darkgrid = mpl.style.library["seaborn-v0_8-darkgrid"]
    assert ax.get_facecolor() == to_rgba(darkgrid["axes.facecolor"])
    poster = mpl.style.library["seaborn-v0_8-poster"]["axes.titlesize"]
    assert ax.title.get_fontsize() == poster


def test_a_figure_draws_the_first_color_of_its_style(run: SimulationResults) -> None:
    fig = run.plot.series(config=PlotConfig(style="seaborn-v0_8-colorblind"))

    colors = mpl.style.library["seaborn-v0_8-colorblind"]["axes.prop_cycle"]
    first = colors.by_key()["color"][0]
    assert to_rgba(_visible(fig)[0].lines[0].get_color()) == to_rgba(first)


class TestRunFigures:
    def test_plot_series_draws_every_bin(self, run: SimulationResults) -> None:
        fig = run.plot.series()

        _assert_draws(fig, {name: run[name] for name in BINS})

    def test_plot_series_saves_to_file(
        self, run: SimulationResults, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "series_plot.png"

        run.plot.series(output_file=str(output_path))

        assert output_path.stat().st_size > 0

    def test_plot_series_draws_the_observations(self, run: SimulationResults) -> None:
        observed = _observed_b(run)

        fig = run.plot.series(observed_data=observed)

        ax = _visible(fig)[1]
        markers = _scatter(ax)
        np.testing.assert_allclose(markers[:, 0], [p.at for p in observed])
        np.testing.assert_allclose(markers[:, 1], [p.value for p in observed])
        (points,) = ax.collections
        np.testing.assert_allclose(points.get_facecolor(), [[1.0, 0.0, 0.0, 0.7]])
        assert (list(points.get_sizes()), points.get_zorder()) == ([30.0], 5)

    def test_plot_series_with_calibration_result(self, run: SimulationResults) -> None:
        scale_factor = 0.5
        observed = [
            ObservedDataPoint(
                at=i,
                output="B",
                value=run["B"][i] * scale_factor,
                factor_id="detection_rate",
            )
            for i in range(0, 100, 10)
        ]

        fig = run.plot.series(
            observed_data=observed,
            calibration_result=_fit(detection_rate=scale_factor),
            series=["B"],
        )

        ax = fig.axes[0]
        plotted_y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert plotted_y[10] == pytest.approx(run["B"][10] * scale_factor)
        assert _scatter(ax)[1, 1] == pytest.approx(run["B"][10] * scale_factor)

    def test_plot_series_subset_of_bins(self, run: SimulationResults) -> None:
        fig = run.plot.series(series=["B", "C"])

        _assert_draws(fig, {name: run[name] for name in ("B", "C")})

    def test_plot_series_with_custom_config(self, run: SimulationResults) -> None:
        config = PlotConfig(
            figsize=(12, 8),
            dpi=150,
            layout=(1, 3),
            style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-notebook"],
        )

        fig = run.plot.series(config=config)

        assert len(fig.axes) == 3
        assert fig.dpi == 150
        _assert_draws(fig, {name: run[name] for name in BINS})

    def test_plot_cumulative_draws_the_running_sums(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.cumulative()

        _assert_draws(fig, {name: list(np.cumsum(run[name])) for name in BINS})

    def test_plot_cumulative_saves_to_file(
        self, run: SimulationResults, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "cumulative_plot.png"

        run.plot.cumulative(output_file=str(output_path))

        assert output_path.stat().st_size > 0

    def test_plot_cumulative_draws_the_running_sum_of_observations(
        self, run: SimulationResults
    ) -> None:
        observed = _observed_b(run)

        fig = run.plot.cumulative(observed_data=observed)

        markers = _scatter(_visible(fig)[1])
        np.testing.assert_allclose(
            markers[:, 1], np.cumsum([p.value for p in observed])
        )

    def test_a_cumulative_plot_has_the_size_of_its_configuration(
        self, run: SimulationResults
    ) -> None:
        config = PlotConfig(figsize=(10, 6), dpi=80, layout=(1, 4))

        fig = run.plot.cumulative(config=config)

        assert tuple(fig.get_size_inches()) == (10, 6)
        assert fig.dpi == 80
        assert (len(fig.axes), len(_visible(fig))) == (4, 3)

    def test_a_series_plot_has_the_size_of_its_panels(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.series(config=PlotConfig(layout=(2, 2)))

        assert tuple(fig.get_size_inches()) == (9.0, 8.0)
        assert (len(fig.axes), len(_visible(fig))) == (4, 3)

    @pytest.mark.parametrize("method", ["series", "cumulative"])
    def test_a_plot_of_one_series_has_one_panel(
        self, run: SimulationResults, method: str
    ) -> None:
        fig = getattr(run.plot, method)(series=["B"])

        assert [ax.get_title() for ax in fig.axes] == ["State B"]

    @pytest.mark.parametrize(
        ("method", "name"),
        [("series", "Series"), ("cumulative", "Cumulative")],
    )
    def test_a_saved_plot_is_cropped_and_logged(
        self,
        run: SimulationResults,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        method: str,
        name: str,
    ) -> None:
        path = tmp_path / "plot.png"
        uncropped = tmp_path / "uncropped.png"

        with caplog.at_level("INFO", logger="commol"):
            fig = getattr(run.plot, method)(output_file=str(path))
        fig.savefig(uncropped, dpi=fig.dpi)

        assert f"{name} figure saved to {path}" in caplog.messages
        assert plt.imread(path).shape != plt.imread(uncropped).shape

    def test_a_cumulative_plot_labels_its_line_and_its_observations(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.cumulative(observed_data=_observed_b(run))

        ax = _visible(fig)[1]
        (markers,) = [c for c in ax.collections if len(c.get_offsets()) == 10]
        np.testing.assert_allclose(_scatter(ax)[:, 0], range(0, 100, 10))
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == (
            "Step",
            "",
            "State B",
        )
        grid = ax.xaxis.get_gridlines()[0]
        assert (grid.get_visible(), grid.get_alpha()) == (True, 0.3)
        assert ax.lines[0].get_label() == "Simulation (Cumulative)"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "Simulation (Cumulative)",
            "Observed (Cumulative)",
        ]
        np.testing.assert_allclose(markers.get_facecolor(), [[1.0, 0.0, 0.0, 0.7]])
        assert (list(markers.get_sizes()), markers.get_zorder()) == ([30.0], 5)

    def test_a_cumulative_plot_labels_the_unit_of_its_series(self) -> None:
        run = Simulation(build(abc_builder(bin_unit="item"))).run(5)

        fig = run.plot.cumulative()

        assert [ax.get_ylabel() for ax in _visible(fig)] == ["item"] * 3

    def test_cumulative_observations_of_a_factor_are_divided_by_it(
        self, run: SimulationResults
    ) -> None:
        observed = [
            ObservedDataPoint(at=step, output="B", value=10.0, factor_id="rate")
            for step in (10, 20)
        ] + [
            ObservedDataPoint(at=30, output="B", value=10.0),
            ObservedDataPoint(at=40, output="B", value=10.0, factor_id="other"),
        ]

        fig = run.plot.cumulative(
            observed_data=observed, calibration_result=_fit(rate=0.5), series=["B"]
        )

        np.testing.assert_allclose(
            _scatter(fig.axes[0])[:, 1], [20.0, 40.0, 50.0, 60.0]
        )

    def test_layout(self) -> None:
        assert figures._layout(1) == (1, 1)
        assert figures._layout(2) == (1, 2)
        assert figures._layout(3) == (2, 2)
        assert figures._layout(4) == (2, 2)
        assert figures._layout(5) == (2, 3)
        assert figures._layout(9) == (3, 3)

    def test_group_observed_data(self) -> None:
        observed = [
            ObservedDataPoint(at=20, output="B", value=60.0),
            ObservedDataPoint(at=10, output="B", value=50.0),
            ObservedDataPoint(at=10, output="C", value=10.0),
        ]

        grouped = _group_observed(observed)

        assert {name: [p.at for p in points] for name, points in grouped.items()} == {
            "B": [10, 20],
            "C": [10],
        }

    def test_apply_x_labels_anchors_ticks_to_first_plotted_step(self) -> None:
        """Ticks follow the series' own grid, not the absolute step 0."""
        _fig, ax = plt.subplots()

        # A series read at its observations: the plotted steps are their
        # positions, which start at 7 rather than at 0.
        window = 7
        positions = list(range(window, 8 * window + 1, window))
        figures._apply_x_labels(ax, positions, str, tick_every=2 * window)

        ticks = list(ax.get_xticks())
        assert ticks == [window, 3 * window, 5 * window, 7 * window]

    def test_apply_x_labels_without_tick_every_labels_all_steps(self) -> None:
        _fig, ax = plt.subplots()

        figures._apply_x_labels(ax, [5, 10, 15], lambda p: f"day {p}", tick_every=None)

        assert list(ax.get_xticks()) == [5, 10, 15]
        labels = ax.get_xticklabels()
        assert [label.get_text() for label in labels] == ["day 5", "day 10", "day 15"]
        assert {(label.get_rotation(), label.get_ha()) for label in labels} == {
            (45.0, "right")
        }

    def test_apply_x_labels_with_empty_series(self) -> None:
        """An empty series produces no ticks instead of an index error."""
        _fig, ax = plt.subplots()

        figures._apply_x_labels(ax, [], str, tick_every=4)

        assert list(ax.get_xticks()) == []

    def test_plot_series_passes_keywords_to_the_lines(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.series(linewidth=3, alpha=0.7, linestyle="--")

        for ax in _visible(fig):
            line = ax.lines[0]
            assert (line.get_linewidth(), line.get_alpha(), line.get_linestyle()) == (
                3.0,
                0.7,
                "--",
            )

    def test_plot_cumulative_passes_keywords_to_the_lines(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.cumulative(linewidth=2, alpha=0.8)

        for ax in _visible(fig):
            assert (ax.lines[0].get_linewidth(), ax.lines[0].get_alpha()) == (2.0, 0.8)

    def test_a_series_figure_without_legend_has_none(
        self, run: SimulationResults
    ) -> None:
        fig = run.plot.series(show_legend=False)

        assert {ax.get_legend() for ax in _visible(fig)} == {None}

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            (
                PlotConfig.presentation,
                PlotConfig(
                    figsize=(16, 10),
                    dpi=150,
                    style=[
                        "seaborn-v0_8-whitegrid",
                        "seaborn-v0_8-talk",
                        "seaborn-v0_8-colorblind",
                    ],
                ),
            ),
            (
                PlotConfig.publication,
                PlotConfig(
                    figsize=(8, 6),
                    dpi=300,
                    style=["seaborn-v0_8-ticks", "seaborn-v0_8-paper"],
                ),
            ),
            (
                PlotConfig.notebook,
                PlotConfig(
                    figsize=(12, 8),
                    dpi=100,
                    style=["seaborn-v0_8-whitegrid", "seaborn-v0_8-notebook"],
                ),
            ),
        ],
    )
    def test_plot_config_presets(
        self, run: SimulationResults, preset, expected: PlotConfig
    ) -> None:
        config = preset()

        fig = run.plot.series(config=config)

        assert config == expected
        assert fig.dpi == expected.dpi


def _dated_run(step_unit: str, steps: int) -> SimulationResults:
    """`steps` steps of a model whose axis starts on 2020-01-01."""
    model = build(abc_builder(step_unit=step_unit, origin=date(2020, 1, 1)))
    return Simulation(model).run(steps)


def _tick_labels(fig: Figure) -> dict[float, str]:
    """The labelled ticks of the first panel, after the figure is drawn."""
    fig.canvas.draw()
    ax = fig.axes[0]
    return {
        float(tick): label.get_text()
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=True)
        if label.get_text()
    }


def test_an_axis_with_origin_labels_ticks_with_dates() -> None:
    fig = _dated_run("day", 40).plot.series(series=["B"])

    labels = _tick_labels(fig)
    assert labels
    for tick, label in labels.items():
        assert label == (date(2020, 1, 1) + timedelta(days=tick)).isoformat()


def test_dates_label_the_plotted_positions_tick_every_apart() -> None:
    fig = _dated_run("day", 20).plot.series(series=["B"], tick_every=7)

    assert _tick_labels(fig) == {
        0.0: "2020-01-01",
        7.0: "2020-01-08",
        14.0: "2020-01-15",
    }


def test_a_date_label_shows_the_time_away_from_midnight() -> None:
    fig = _dated_run("hour", 30).plot.series(series=["B"], tick_every=6)

    assert _tick_labels(fig) == {
        0.0: "2020-01-01",
        6.0: "2020-01-01 06:00:00",
        12.0: "2020-01-01 12:00:00",
        18.0: "2020-01-01 18:00:00",
        24.0: "2020-01-02",
        30.0: "2020-01-02 06:00:00",
    }


def test_a_position_label_wins_over_the_dates() -> None:
    fig = _dated_run("day", 20).plot.series(
        series=["B"], position_label=lambda at: f"p{at:g}", tick_every=10
    )

    assert _tick_labels(fig) == {0.0: "p0", 10.0: "p10", 20.0: "p20"}


class TestPredictiveSeries:
    """
    A predictive draws the weighted median and band of its draws, each read
    through the observation operator of the problem.
    """

    window = 5

    def test_a_windowed_band_of_one_draw_is_its_windowed_run(self) -> None:
        predictive = _predictive([0.05])
        run = predictive.runs()[0]
        observed = [
            ObservedDataPoint(at=at, output="events", value=1.0, window=self.window)
            for at in (10, 25, 40)
        ]

        figure = predictive.plot.series(observed_data=observed, series=["events"])

        ends = [10, 25, 40]
        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == ends
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [run["events"][end] - run["events"][end - 5] for end in ends]
        )

    def test_a_band_reads_each_point_through_its_own_window(self) -> None:
        predictive = _predictive([0.05])
        run = predictive.runs()[0]
        observed = [
            ObservedDataPoint(at=at, output="events", value=1.0, window=window)
            for at, window in ((10, 5), (30, 20))
        ]

        figure = predictive.plot.series(observed_data=observed, series=["events"])

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == [10, 30]
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [
                run["events"][10] - run["events"][5],
                run["events"][30] - run["events"][10],
            ]
        )

    def test_the_band_of_two_draws_spans_their_runs(self) -> None:
        predictive = _predictive([0.02, 0.1])
        low, high = predictive.runs()

        figure = predictive.plot.series(series=["B"])

        filled = [c for c in figure.axes[0].collections if c.get_paths()]
        band = filled[0].get_paths()[0].vertices[:, 1]
        assert band.min() == pytest.approx(min(low["B"]))
        assert band.max() == pytest.approx(max(high["B"]))

    @staticmethod
    def _factor_predictive(observed: list[ObservedDataPoint]) -> Predictive:
        """A predictive of one draw, ``k1 = 0.05`` and the factor ``f1 = 0.5``."""
        model = build(a_to_b_builder(parameters={"k1": None}, rate="k1"))
        problem = CalibrationProblem(
            observed_data=observed,
            parameters=[
                bounded("k1", 0.0, 0.5),
                bounded("f1", 0.1, 1.0, kind="factor"),
            ],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        sample = ParameterSample.equally_weighted(
            ["k1", "f1"],
            [(0.0, 0.5), (0.1, 1.0)],
            [[0.05, 0.5]],
            SampleMethod(name="t"),
        )
        return Predictive(
            Calibrator(Simulation(model), problem), sample, list(range(41))
        )

    def test_a_band_reads_each_point_through_its_own_factor(self) -> None:
        observed = [
            ObservedDataPoint(at=10, output="B", value=1.0, factor_id="f1"),
            ObservedDataPoint(at=20, output="B", value=1.0),
        ]
        predictive = self._factor_predictive(observed)
        run = predictive.runs()[0]

        figure = predictive.plot.series(observed_data=observed, series=["B"])

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == [10, 20]
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [0.5 * run["B"][10], run["B"][20]]
        )

    def test_a_band_of_one_factor_covers_every_position(self) -> None:
        observed = [
            ObservedDataPoint(at=at, output="B", value=1.0, factor_id="f1")
            for at in (10, 20)
        ]
        predictive = self._factor_predictive(observed)
        run = predictive.runs()[0]

        figure = predictive.plot.series(observed_data=observed, series=["B"])

        line = figure.axes[0].lines[0]
        assert np.asarray(line.get_xdata(), dtype=float).tolist() == run.positions
        assert np.asarray(line.get_ydata(), dtype=float).tolist() == pytest.approx(
            [0.5 * value for value in run["B"]]
        )

    def test_a_band_labels_its_positions(self) -> None:
        predictive = _predictive([0.05])

        figure = predictive.plot.series(
            series=["B"],
            position_label=lambda at: f"day {at:g}",
            tick_every=20,
        )

        labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
        assert labels == ["day 0", "day 20", "day 40"]

    def test_a_cumulative_band_spans_the_running_totals_of_its_draws(self) -> None:
        predictive = _predictive([0.02, 0.1])
        low, high = predictive.runs()
        observed = [ObservedDataPoint(at=at, output="B", value=1.0) for at in (5, 9)]

        figure = predictive.plot.cumulative(observed_data=observed, series=["B"])

        ax = figure.axes[0]
        band = [c for c in ax.collections if c.get_paths()][0]
        vertices = band.get_paths()[0].vertices[:, 1]
        assert vertices.max() == pytest.approx(float(np.cumsum(high["B"])[-1]))
        edges = band.get_paths()[0].vertices
        # The closed path repeats a vertex.
        at_the_end = sorted({round(float(y), 6) for x, y in edges if x == 40})
        assert at_the_end == pytest.approx(
            [float(np.cumsum(low["B"])[-1]), float(np.cumsum(high["B"])[-1])]
        )
        assert band.get_alpha() == 0.3
        assert "95% of the draws" in [t.get_text() for t in ax.get_legend().get_texts()]
        assert ax.lines[0].get_label() == "Median of the draws (cumulative)"
        np.testing.assert_allclose(_scatter(ax), [[5.0, 1.0], [9.0, 2.0]])
        assert ax.get_title() == "B"


class TestReplicateFigures:
    @pytest.fixture
    def replicates(self):
        simulation = Simulation(build_stochastic(a_to_b_builder(), "StochasticSteps"))
        return simulation.replicates(5, seed=2).run(20)

    def test_the_band_of_windowed_replicates_spans_their_window_totals(
        self, replicates
    ) -> None:
        observed = [
            ObservedDataPoint(at=at, output="B", value=1.0, window=5) for at in (10, 20)
        ]

        fig = replicates.plot.series(observed_data=observed, series=["B"])

        totals = [
            [run["B"][end] - run["B"][end - 5] for end in (10, 20)]
            for run in replicates
        ]
        band = [c for c in fig.axes[0].collections if c.get_paths()][0]
        vertices = band.get_paths()[0].vertices
        assert sorted({float(x) for x, _ in vertices}) == [10.0, 20.0]
        assert vertices[:, 1].min() >= np.min(totals) - 1e-9
        assert vertices[:, 1].max() <= np.max(totals) + 1e-9

    def test_the_cumulative_figure_of_replicates_is_that_of_their_mean(
        self, replicates
    ) -> None:
        fig = replicates.plot.cumulative(series=["B"])

        np.testing.assert_allclose(
            fig.axes[0].lines[0].get_ydata(), np.cumsum(replicates.mean()["B"])
        )


class TestObservationOperator:
    """The plotted line is the series as the observations read it."""

    def test_a_sum_of_outputs_plots_their_sum(self, run: SimulationResults) -> None:
        observed = [ObservedDataPoint(at=10, outputs=["B", "C"], label="B", value=1.0)]

        fig = run.plot.series(observed_data=observed, series=["B"])

        expected = np.add(run["B"], run["C"])
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), expected)

    def test_a_sum_of_an_unknown_output_is_an_error(
        self, run: SimulationResults
    ) -> None:
        observed = [ObservedDataPoint(at=10, outputs=["B", "Z"], label="B", value=1.0)]

        with pytest.raises(KeyError, match=r"aggregate components: \['Z'\]"):
            run.plot.series(observed_data=observed, series=["B"])

    def test_a_factor_it_cannot_read_leaves_the_line_unscaled(
        self,
        run: SimulationResults,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        observed = [ObservedDataPoint(at=10, output="B", value=1.0, factor_id="f3")]

        fig = run.plot.series(
            observed_data=observed, calibration_result=_fit(f1=0.5), series=["B"]
        )

        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), run["B"])
        assert "Factor parameter 'f3' not found" in caplog.text

    def test_points_of_different_factors_are_each_read_through_their_own(
        self, run: SimulationResults
    ) -> None:
        observed = [
            ObservedDataPoint(at=10, output="B", value=1.0, factor_id="f1"),
            ObservedDataPoint(at=20, output="B", value=1.0, factor_id="f2"),
            ObservedDataPoint(at=30, outputs=["B", "C"], label="B", value=1.0),
        ]

        fig = run.plot.series(
            observed_data=observed,
            calibration_result=_fit(f1=0.5, f2=2.0),
            series=["B"],
        )

        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(line.get_xdata(), [10, 20, 30])
        np.testing.assert_allclose(
            line.get_ydata(),
            [0.5 * run["B"][10], 2.0 * run["B"][20], run["B"][30] + run["C"][30]],
        )


class TestSeriesSelection:
    @pytest.fixture
    def stratified(self) -> SimulationResults:
        model = build(a_to_b_builder(stratifications={"group": {"g1": 0.3, "g2": 0.7}}))
        return Simulation(model).run(10)

    def test_a_bin_id_plots_the_sum_of_its_outputs(
        self, stratified: SimulationResults
    ) -> None:
        fig = stratified.plot.series(series=["B"])

        plotted = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        expected = np.add(stratified["B_g1"], stratified["B_g2"])
        np.testing.assert_allclose(plotted, expected)

    def test_an_output_name_plots_that_output(
        self, stratified: SimulationResults
    ) -> None:
        fig = stratified.plot.series(series=["B_g1"])

        plotted = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        np.testing.assert_allclose(plotted, stratified["B_g1"])

    def test_a_bin_id_in_a_cumulative_plot(self, stratified: SimulationResults) -> None:
        fig = stratified.plot.cumulative(series=["B"])

        plotted = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        expected = np.cumsum(np.add(stratified["B_g1"], stratified["B_g2"]))
        np.testing.assert_allclose(plotted, expected)

    @pytest.mark.parametrize("method", ["series", "cumulative"])
    def test_an_unknown_id_is_a_clear_error(
        self, stratified: SimulationResults, method: str
    ) -> None:
        with pytest.raises(ValueError, match="'X' is neither an output"):
            getattr(stratified.plot, method)(series=["X"])


class TestDifferentialModel:
    def test_observations_at_non_integer_steps_are_drawn_where_they_are(
        self,
    ) -> None:
        simulation = Simulation(build_ode(abc_builder()))
        steps = [0.5, 10.25, 40.75]
        at = simulation.run_at(steps)
        observed = [
            ObservedDataPoint(at=step, output="B", value=value)
            for step, value in zip(steps, at["B"], strict=True)
        ]

        fig = simulation.run(50).plot.series(observed_data=observed, series=["B"])

        np.testing.assert_allclose(_scatter(fig.axes[0])[:, 0], steps)

    def _events(self) -> Simulation:
        return Simulation(
            build_ode(
                a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
            )
        )

    def test_a_window_of_non_whole_steps_draws_its_increments(self) -> None:
        simulation = self._events()
        ends = [5.0, 12.5, 20.0]
        observed = [
            ObservedDataPoint(at=end, output="events", value=1.0, window=2.5)
            for end in ends
        ]
        results = simulation.run_at([2.5 * j for j in range(9)])

        fig = results.plot.series(observed_data=observed, series=["events"])

        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(line.get_xdata(), ends)
        np.testing.assert_allclose(
            line.get_ydata(),
            simulation.windowed_totals(results, ["events"], 2.5, ends),
        )

    def test_a_window_the_results_do_not_hold_is_an_error(self) -> None:
        simulation = self._events()
        observed = [ObservedDataPoint(at=5, output="events", value=1.0, window=2.5)]

        with pytest.raises(ValueError, match="no value at 2.5"):
            simulation.run(20).plot.series(observed_data=observed)


def test_each_observed_point_is_read_through_its_own_window_and_position() -> None:
    """
    The model line of a series sits on each observation: its value there is
    the observation read through its own window, as the loss reads it.
    """
    simulation = Simulation(
        build_ode(
            a_to_b_builder(parameters={"k1": 0.05}, rate="k1", accumulator="events")
        )
    )
    run_at_half_steps = simulation.run_at([0.5 * j for j in range(41)])
    observed = [
        ObservedDataPoint(at=7.5, output="events", value=10.0, window=7),
        ObservedDataPoint(at=17.5, output="events", value=12.0, window=14),
    ]

    fig = run_at_half_steps.plot.series(observed_data=observed, series=["events"])

    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [7.5, 17.5])
    np.testing.assert_allclose(
        line.get_ydata(),
        [
            simulation.windowed_totals(run_at_half_steps, ["events"], 7, [7.5])[0],
            simulation.windowed_totals(run_at_half_steps, ["events"], 14, [17.5])[0],
        ],
    )
