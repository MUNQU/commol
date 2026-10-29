"""What every figure of results does: it is read from its results and returns
the `Figure`, draws on given axes, reads a series through the observation
operator and a bin id as the sum of its outputs, keeps the global style, and
needs matplotlib only when it draws. The figures of each kind of result are in
the other modules of this package."""

import inspect
import subprocess
import sys
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from commol import ObservedDataPoint, Simulation
from commol._simulation.results import SimulationResults
from commol.plotting import PlotConfig, figures, plot_compared
from commol._simulation import plots as simulation_plots
from commol._simulation.plots import ReplicatesPlot, ResultsPlot
from commol.profiles import plots as profile_plots
from commol.profiles.results import ProfileResult
from commol.sensitivity.results import (
    ChaosResult,
    MorrisResult,
    PRCCResult,
    SobolResult,
)
from commol.sensitivity import plots as sensitivity_plots
from commol.uncertainty import plots as predictive_plots
from commol.uncertainty.plots import PredictivePlot
from tests.suite.public_surface import EXPORTS
from tests.support import (
    a_to_b_builder,
    abc_builder,
    build,
    build_ode,
)
from tests.support.plotted import (
    BINS,
    TITLES,
    fit,
    predictive_of,
    scatter,
    visible,
)


def test_the_figure_of_a_run_is_read_from_the_run() -> None:
    """
    One panel per output, labelled by the id of its series, titled and with
    the unit the kind gives it, with the line of the run.
    """
    run = Simulation(build(abc_builder(bin_unit="item"))).run(10)

    fig = run.plot.series()

    axes = visible(fig)
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
    predictive = predictive_of([0.02, 0.1])

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
    panel_a, panel_b = visible(fig)
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


def test_results_built_by_hand_plot_only_outputs() -> None:
    results = SimulationResults(
        [0.0, 1.0, 2.0], {"B_g1": [1.0, 2.0, 3.0], "B_g2": [0.0, 1.0, 1.0]}
    )

    fig = results.plot.series()

    assert [ax.get_title() for ax in visible(fig)] == ["B_g1", "B_g2"]
    assert {ax.get_ylabel() for ax in visible(fig)} == {""}
    with pytest.raises(ValueError, match="'B' is not an output of the results"):
        results.plot.series(series=["B"])


def test_a_sensitivity_result_plots_itself(prcc_result: PRCCResult) -> None:
    """
    The PRCC of a series over positions is one line per input with markers;
    that of a scalar quantity, one bar per input; each panel lies in
    ``[-1.05, 1.05]`` with a line at zero.
    """
    fig = prcc_result.plot(x_label="Day")

    over_axis, bars = visible(fig)
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


ACCESSORS = [
    (ResultsPlot.series, simulation_plots.run_series),
    (ResultsPlot.cumulative, simulation_plots.run_cumulative),
    (ReplicatesPlot.series, simulation_plots.replicates_series),
    (ReplicatesPlot.cumulative, simulation_plots.replicates_cumulative),
    (PredictivePlot.series, predictive_plots.predictive_series),
    (PredictivePlot.cumulative, predictive_plots.predictive_cumulative),
    (ResultsPlot.observations, simulation_plots.run_observations),
    (ReplicatesPlot.observations, simulation_plots.replicates_observations),
    (PredictivePlot.observations, predictive_plots.predictive_observations),
    (PRCCResult.plot, sensitivity_plots.prcc_figure),
    (MorrisResult.plot, sensitivity_plots.morris_figure),
    (SobolResult.plot, sensitivity_plots.sobol_figure),
    (ChaosResult.plot, sensitivity_plots.sobol_figure),
    (ProfileResult.plot, profile_plots.profiles_figure),
    (plot_compared, figures._compared_series),
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
            "import importlib, sys; "
            f"[importlib.import_module(m) for m in {sorted(EXPORTS)}]; "
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
    monkeypatch.delitem(sys.modules, "commol.plotting.figures")

    with pytest.raises(ImportError, match=r"pip install 'commol\[plot\]'"):
        run.plot.series()


STYLE = PlotConfig(style=["seaborn-v0_8-darkgrid", "seaborn-v0_8-poster"])


@pytest.mark.parametrize(
    "draw",
    [
        lambda run, _: run.plot.series(config=STYLE),
        lambda run, _: run.plot.cumulative(config=STYLE),
        lambda run, _: run.plot.observations(
            [ObservedDataPoint(at=10, output="B", value=1.0)], config=STYLE
        ),
        lambda _, results: results["prcc"].plot(config=STYLE),
        lambda _, results: results["morris"].plot(config=STYLE),
        lambda _, results: results["sobol"].plot(config=STYLE),
        lambda _, results: results["profile"].plot(config=STYLE),
    ],
    ids=["series", "cumulative", "observations", "prcc", "morris", "sobol", "profile"],
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
    ax = visible(fig)[0]
    darkgrid = mpl.style.library["seaborn-v0_8-darkgrid"]
    assert ax.get_facecolor() == to_rgba(darkgrid["axes.facecolor"])
    poster = mpl.style.library["seaborn-v0_8-poster"]["axes.titlesize"]
    assert ax.title.get_fontsize() == poster


def test_a_figure_draws_the_first_color_of_its_style(run: SimulationResults) -> None:
    fig = run.plot.series(config=PlotConfig(style="seaborn-v0_8-colorblind"))

    colors = mpl.style.library["seaborn-v0_8-colorblind"]["axes.prop_cycle"]
    first = colors.by_key()["color"][0]
    assert to_rgba(visible(fig)[0].lines[0].get_color()) == to_rgba(first)


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

        with pytest.raises(KeyError, match=r"in results: \['Z'\]"):
            run.plot.series(observed_data=observed, series=["B"])

    def test_a_factor_it_cannot_read_leaves_the_line_unscaled(
        self,
        run: SimulationResults,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        observed = [ObservedDataPoint(at=10, output="B", value=1.0, factor_id="f3")]

        fig = run.plot.series(
            observed_data=observed, calibration_result=fit(f1=0.5), series=["B"]
        )

        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), run["B"])
        assert "Factor parameter 'f3' not found" in caplog.text

    @pytest.mark.parametrize("figure", ["series", "cumulative", "observations"])
    def test_a_missing_factor_warns_once_per_factor(
        self,
        run: SimulationResults,
        caplog: pytest.LogCaptureFixture,
        figure: str,
    ) -> None:
        observed = [
            ObservedDataPoint(at=step, output="B", value=1.0, factor_id="f")
            for step in (10, 20, 30)
        ] + [ObservedDataPoint(at=40, output="B", value=1.0, factor_id="g")]

        getattr(run.plot, figure)(observed_data=observed)

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelname == "WARNING"
        ]
        assert [message.split(";")[0] for message in warnings] == [
            "Factor parameter 'f' not found in calibration result",
            "Factor parameter 'g' not found in calibration result",
        ]

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
            calibration_result=fit(f1=0.5, f2=2.0),
            series=["B"],
        )

        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(line.get_xdata(), [10, 20, 30])
        np.testing.assert_allclose(
            line.get_ydata(),
            [0.5 * run["B"][10], 2.0 * run["B"][20], run["B"][30] + run["C"][30]],
        )

    def test_a_scale_multiplies_the_line_and_each_point_of_its_own(
        self, run: SimulationResults
    ) -> None:
        one = [ObservedDataPoint(at=10, output="B", value=1.0, scale=0.5)]
        each = [
            ObservedDataPoint(at=10, output="B", value=1.0, scale=0.5),
            ObservedDataPoint(at=20, output="B", value=1.0, scale=0.25),
        ]

        line = run.plot.series(observed_data=one, series=["B"]).axes[0].lines[0]
        points = run.plot.series(observed_data=each, series=["B"]).axes[0].lines[0]

        np.testing.assert_allclose(line.get_ydata(), 0.5 * np.asarray(run["B"]))
        np.testing.assert_allclose(points.get_xdata(), [10, 20])
        np.testing.assert_allclose(
            points.get_ydata(), [0.5 * run["B"][10], 0.25 * run["B"][20]]
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
        with pytest.raises(KeyError, match="'X' is not a bin or accumulator"):
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

        np.testing.assert_allclose(scatter(fig.axes[0])[:, 0], steps)

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
            results.windowed_totals(["events"], 2.5, ends),
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
            run_at_half_steps.windowed_totals(["events"], 7, [7.5])[0],
            run_at_half_steps.windowed_totals(["events"], 14, [17.5])[0],
        ],
    )
