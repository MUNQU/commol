"""What the tests of figures read from a figure, and the results they draw.

It imports matplotlib, so ``tests.support`` does not import it: a test module
of ``tests/plotting/`` imports it by name.
"""

from datetime import date

import numpy as np
from matplotlib.figure import Figure

from commol import (
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol._fit.result import CalibrationResult
from commol._simulation.results import SimulationResults
from commol.uncertainty import ParameterSample, Predictive, SampleMethod

from .models import a_to_b_builder, abc_builder, build
from .problems import bounded

BINS = ["A", "B", "C"]
"""The bins of ``abc_builder``."""

TITLES = {"A": "State A", "B": "State B", "C": "State C"}
"""The title of the panel of each bin of ``abc_builder``: the name of the bin."""

MODEL_BAND = "model band\n(no observation noise),\nnot a prediction interval"
"""The legend of a band of the model in a panel of an observed series."""


def visible(fig: Figure):
    """The panels of ``fig`` that are drawn."""
    return [ax for ax in fig.axes if ax.get_visible()]


def scatter(ax) -> np.ndarray:
    """The offsets of the observation markers of a panel."""
    offsets = [np.asarray(c.get_offsets(), dtype=float) for c in ax.collections]
    return offsets[int(np.argmax([len(o) for o in offsets]))]


def band_bars(ax) -> list[tuple[float, float, float, float]]:
    """The left end, width, bottom and height of each band rectangle."""
    return [
        (patch.get_x(), patch.get_width(), patch.get_y(), patch.get_height())
        for patch in ax.patches
    ]


def legend_texts(ax) -> set[str]:
    """The texts of the legend of a panel."""
    return {text.get_text() for text in ax.get_legend().get_texts()}


def tick_labels(fig: Figure) -> dict[float, str]:
    """The labelled ticks of the first panel, after the figure is drawn."""
    fig.canvas.draw()
    ax = fig.axes[0]
    return {
        float(tick): label.get_text()
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=True)
        if label.get_text()
    }


def fit(**factors: float) -> CalibrationResult:
    """A calibration result whose best parameters are ``factors``."""
    return CalibrationResult(
        best_parameters=factors,
        final_loss=0.0,
        iterations=1,
        converged=True,
        termination_reason="Test",
        seed=0,
    )


def dated_run(step_unit: str, steps: int) -> SimulationResults:
    """`steps` steps of a model whose axis starts on 2020-01-01."""
    model = build(abc_builder(step_unit=step_unit, origin=date(2020, 1, 1)))
    return Simulation(model).run(steps)


def _events_calibrator(factor: bool = False) -> Calibrator:
    """A calibrator of ``k1`` of a model from A to B that counts its events,
    and of the factor ``f`` of its observation when `factor`."""
    model = build(
        a_to_b_builder(parameters={"k1": None}, rate="k1", accumulator="events")
    )
    problem = CalibrationProblem(
        observed_data=[
            ObservedDataPoint(
                at=10,
                output="events",
                value=1.0,
                window=5,
                factor_id="f" if factor else None,
            )
        ],
        parameters=[bounded("k1", 0.0, 0.5)]
        + ([bounded("f", 0.1, 2.0, kind="factor")] if factor else []),
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1),
    )
    return Calibrator(Simulation(model), problem)


def predictive_of(values: list[float]) -> Predictive:
    """The predictive of one draw of ``k1`` per value, at positions 0 to 40, of
    a model from A to B that counts its events."""
    sample = ParameterSample.equally_weighted(
        ["k1"], [(0.0, 0.5)], [[value] for value in values], SampleMethod(name="t")
    )
    return Predictive(_events_calibrator(), sample, list(range(41)))


def factor_predictive_of(values: list[float]) -> Predictive:
    """As `predictive_of`, with the factor ``f`` of the observation at 0.5 in
    each draw."""
    sample = ParameterSample.equally_weighted(
        ["k1", "f"],
        [(0.0, 0.5), (0.1, 2.0)],
        [[value, 0.5] for value in values],
        SampleMethod(name="t"),
    )
    return Predictive(_events_calibrator(factor=True), sample, list(range(41)))
