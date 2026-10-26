"""The results of the sensitivity methods draw their figures: one series or
bar per input."""

from commol.sensitivity import InputSpace
from commol.sensitivity.methods import (
    ChaosResult,
    MorrisResult,
    PRCCResult,
    SobolResult,
)

from tests.support import bounded

INPUTS = ["k1", "k2", "k3"]


def test_each_sensitivity_result_draws_its_figure(
    prcc_result: PRCCResult, morris_result: MorrisResult, sobol_result: SobolResult
) -> None:
    prcc = prcc_result.plot(x_label="Day")
    over_axis, bars = prcc.axes[:2]
    assert len(over_axis.get_lines()) == 3 + 1  # one per input, and zero
    assert [line.get_label() for line in over_axis.get_lines()[:3]] == INPUTS
    assert over_axis.get_xlabel() == "Day"
    assert len(bars.patches) == 3
    assert over_axis.get_title() == "C" and bars.get_title() == "peak of B"

    morris = morris_result.plot()
    (ax,) = [ax for ax in morris.axes if ax.get_visible()]
    assert len(ax.collections) == 3
    assert [text.get_text() for text in ax.texts] == INPUTS

    sobol = sobol_result.plot()
    (ax,) = [ax for ax in sobol.axes if ax.get_visible()]
    assert len(ax.patches) == 2 * 3
    assert [tick.get_text() for tick in ax.get_xticklabels()] == INPUTS

    chaos = ChaosResult(
        inputs=INPUTS,
        input_space=InputSpace.uniform_on_bounds([bounded(name) for name in INPUTS]),
        labels=["B at 20"],
        degree=1,
        q_norm=1.0,
        max_interaction=None,
        indices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        coefficients=[[1.0, 0.5, 0.5, 0.0]],
        loo_error=[0.0],
        mean=[1.0],
        variance=[0.5],
        first=[[0.5, 0.5, 0.0]],
        total=[[0.5, 0.5, 0.0]],
        runs=10,
    ).plot()
    (ax,) = [ax for ax in chaos.axes if ax.get_visible()]
    assert len(ax.patches) == 2 * 3


def test_a_figure_is_saved_to_its_file(prcc_result: PRCCResult, tmp_path) -> None:
    path = tmp_path / "prcc.png"

    prcc_result.plot(output_file=str(path))

    assert path.exists()


def test_a_sensitivity_figure_draws_on_the_panels_of_its_quantities(
    sobol_result: SobolResult,
) -> None:
    figure = sobol_result.plot()

    again = sobol_result.plot(axes=figure)

    (ax,) = [ax for ax in again.axes if ax.get_visible()]
    assert again is figure
    assert ax.get_label() == "peak of B"
    assert len(ax.patches) == 2 * 2 * 3
