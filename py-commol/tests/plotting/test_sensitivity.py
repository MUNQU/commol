"""SimulationPlotter draws the results of the sensitivity methods: one series
or bar per input."""

import matplotlib as mpl
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from seaborn import axes_style

from commol import (
    ChaosResult,
    InputSpace,
    MorrisResult,
    ParameterSample,
    PlotConfig,
    PRCCResult,
    SampleMethod,
    SobolResult,
)
from commol.api.plotter import SimulationPlotter
from tests.support import bounded

INPUTS = ["k1", "k2", "k3"]


def _prcc() -> PRCCResult:
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


def _morris() -> MorrisResult:
    return MorrisResult(
        inputs=INPUTS,
        labels=["peak of B"],
        mu=[[1.0, -2.0, 0.0]],
        mu_star=[[1.0, 2.0, 0.0]],
        sigma=[[0.5, 0.1, 0.0]],
        runs=40,
    )


def _sobol() -> SobolResult:
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


def test_plotter_draws_each_sensitivity_result() -> None:
    prcc = SimulationPlotter.plot_prcc(_prcc(), x_label="Day")
    over_axis, bars = prcc.axes[:2]
    assert len(over_axis.get_lines()) == 3 + 1  # one per input, and zero
    assert [line.get_label() for line in over_axis.get_lines()[:3]] == INPUTS
    assert over_axis.get_xlabel() == "Day"
    assert len(bars.patches) == 3
    assert over_axis.get_title() == "C" and bars.get_title() == "peak of B"

    morris = SimulationPlotter.plot_morris(_morris())
    (ax,) = [ax for ax in morris.axes if ax.get_visible()]
    assert len(ax.collections) == 3
    assert [text.get_text() for text in ax.texts] == INPUTS

    sobol = SimulationPlotter.plot_sobol(_sobol())
    (ax,) = [ax for ax in sobol.axes if ax.get_visible()]
    assert len(ax.patches) == 2 * 3
    assert [tick.get_text() for tick in ax.get_xticklabels()] == INPUTS

    chaos = SimulationPlotter.plot_sobol(
        ChaosResult(
            inputs=INPUTS,
            input_space=InputSpace.uniform_on_bounds(
                [bounded(name) for name in INPUTS]
            ),
            labels=["B at 20"],
            degree=1,
            indices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            coefficients=[[1.0, 0.5, 0.5, 0.0]],
            loo_error=[0.0],
            mean=[1.0],
            variance=[0.5],
            first=[[0.5, 0.5, 0.0]],
            total=[[0.5, 0.5, 0.0]],
            runs=10,
        )
    )
    (ax,) = [ax for ax in chaos.axes if ax.get_visible()]
    assert len(ax.patches) == 2 * 3


def test_a_plot_is_saved_to_its_file(tmp_path) -> None:
    path = tmp_path / "prcc.png"
    SimulationPlotter.plot_prcc(_prcc(), output_file=str(path))
    assert path.exists()


@pytest.mark.parametrize(
    "draw",
    [
        lambda config: SimulationPlotter.plot_prcc(_prcc(), config=config),
        lambda config: SimulationPlotter.plot_morris(_morris(), config=config),
        lambda config: SimulationPlotter.plot_sobol(_sobol(), config=config),
    ],
    ids=["prcc", "morris", "sobol"],
)
def test_a_sensitivity_figure_leaves_the_global_style(
    default_style: dict[str, object], draw
) -> None:
    """The style is in the figure, and ``matplotlib.rcParams`` is unchanged."""
    figure: Figure = draw(PlotConfig(style="darkgrid", context="poster"))

    assert {k for k, v in default_style.items() if mpl.rcParams[k] != v} == set()
    assert figure.axes[0].get_facecolor() == to_rgba(
        axes_style("darkgrid")["axes.facecolor"]
    )
