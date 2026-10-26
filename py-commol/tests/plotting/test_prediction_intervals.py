"""A figure of series draws the prediction intervals it is given."""

import numpy as np
import pytest

from commol import (
    Simulation,
)
from commol.conformal.methods import PredictionIntervals, SeriesIntervals


def test_a_figure_draws_prediction_intervals(abc_simulation: Simulation) -> None:
    intervals = PredictionIntervals(
        confidence_level=0.9,
        series={
            "B": SeriesIntervals(
                positions=[10.0, 20.0, 30.0],
                prediction=[5.0, 6.0, 7.0],
                lower=[4.0, 4.5, 5.0],
                upper=[6.0, 7.5, 9.0],
            )
        },
        unbounded={},
    )

    figure = abc_simulation.run(40).plot.series(
        prediction_intervals=intervals, series=["A", "B"]
    )

    no_band, band_axes = figure.axes[:2]
    assert not [c for c in no_band.collections if c.get_paths()]
    (band,) = [c for c in band_axes.collections if c.get_paths()]
    vertices = band.get_paths()[0].vertices
    for at, lower, upper in [(10.0, 4.0, 6.0), (20.0, 4.5, 7.5), (30.0, 5.0, 9.0)]:
        heights = vertices[np.isclose(vertices[:, 0], at), 1]
        assert heights.min() == pytest.approx(lower)
        assert heights.max() == pytest.approx(upper)
    assert "90% prediction interval" in [
        t.get_text() for t in band_axes.get_legend().texts
    ]
    assert band_axes.lines[1].get_linestyle() == "--"
