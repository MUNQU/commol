"""Conformal prediction intervals through ``Calibrator.uncertainty``: the
scores, the quantile per series, and each variant.

The model is linear in its one parameter (``flow_calibrator``): each
observation of B is ``c·t``, or ``c`` with a window of one step.
"""

import math

import numpy as np
import pytest

from commol import (
    AdaptiveConformal,
    EnbPI,
    JackknifePlus,
    ObservedDataPoint,
    ParameterSample,
    RollingConformal,
    SampleMethod,
    SplitConformal,
)
from tests.support import flow_calibrator

C = 10.0


def _amounts(positions: range, seed: int, noise: float = 1.0) -> list[float]:
    """Observed amounts of B, ``c·t`` plus normal noise, at `positions`."""
    rng = np.random.default_rng(seed)
    return [C * t + rng.normal(0.0, noise) for t in positions]


def _held_out(
    positions: range, seed: int, output: str = "B"
) -> list[ObservedDataPoint]:
    values = _amounts(positions, seed)
    if output == "A":
        values = [1e6 - value for value in values]
    return [
        ObservedDataPoint(at=t, output=output, value=value)
        for t, value in zip(positions, values, strict=True)
    ]


def _variant_settings() -> dict[type, dict[str, object]]:
    """Valid settings of each variant."""
    return {
        SplitConformal: {
            "calibration_data": _held_out(range(21, 41), 2),
            "level": 0.9,
            "positions": [41.0],
        },
        RollingConformal: {"origins": [10.0, 12.0], "level": 0.9, "positions": [41.0]},
        JackknifePlus: {"block_length": 1, "level": 0.9, "positions": [41.0]},
        EnbPI: {"replicates": 20, "block_length": 1, "level": 0.9, "positions": [41.0]},
        AdaptiveConformal: {
            "level": 0.9,
            "step": 0.01,
            "stream": _held_out(range(21, 41), 2),
        },
    }


VARIANTS = [SplitConformal, RollingConformal, JackknifePlus, EnbPI, AdaptiveConformal]


class TestScores:
    def test_spread_score_gives_adaptive_intervals(self) -> None:
        calibrator = flow_calibrator(_amounts(range(1, 21), 1), windowed=False)
        sample = ParameterSample.equally_weighted(
            ["c"], [(0.0, 50.0)], [[9.9], [10.0], [10.1]], SampleMethod(name="test")
        )
        positions = [45.0, 60.0, 80.0]
        held_out = _held_out(range(21, 41), 2)

        spread = calibrator.uncertainty(
            SplitConformal(
                calibration_data=held_out,
                score="spread",
                by="series",
                level=0.8,
                positions=positions,
                sample=sample,
            )
        )
        absolute = calibrator.uncertainty(
            SplitConformal(
                calibration_data=held_out,
                score="absolute",
                by="series",
                level=0.8,
                positions=positions,
            )
        )

        def widths(result: object) -> list[float]:
            series = result.intervals.series["B"]  # type: ignore[attr-defined]
            return [hi - lo for lo, hi in zip(series.lower, series.upper, strict=True)]

        # The spread of c·t over the sample is proportional to t.
        assert widths(spread) == pytest.approx(
            [widths(spread)[0] * t / positions[0] for t in positions]
        )
        assert widths(absolute) == pytest.approx([widths(absolute)[0]] * 3)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize(
        ("argument", "values"),
        [("score", ["absolute", "normalized", "spread"]), ("by", ["series", "all"])],
    )
    def test_score_and_by_are_required(
        self, variant: type, argument: str, values: list[str]
    ) -> None:
        settings = {**_variant_settings()[variant], "score": "absolute", "by": "all"}

        for given in (None, "other"):
            if given is None:
                settings.pop(argument)
            else:
                settings[argument] = given
            with pytest.raises(ValueError) as error:
                variant(**settings)
            message = str(error.value)
            assert argument in message
            for value in values:
                assert f'"{value}"' in message, message

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_normalized_score_needs_pooled_quantiles(self, variant: type) -> None:
        settings = _variant_settings()[variant]

        with pytest.raises(ValueError) as error:
            variant(**settings, score="normalized", by="series")

        message = str(error.value)
        assert '"absolute"' in message and "equal" in message
        assert 'by="all"' in message and 'score="absolute"' in message
        assert variant(**settings, score="normalized", by="all").by == "all"

    def test_a_series_with_few_points_gets_an_unbounded_interval(self) -> None:
        calibrator = flow_calibrator(_amounts(range(1, 21), 1), windowed=False)
        # At level 0.9 a series needs 1/0.1 − 1 = 9 held-out points.
        held_out = _held_out(range(21, 41), 2) + _held_out(range(21, 29), 3, "A")

        result = calibrator.uncertainty(
            SplitConformal(
                calibration_data=held_out,
                score="absolute",
                by="series",
                level=0.9,
                positions=[45.0, 50.0],
            )
        )

        intervals = result.intervals
        assert intervals.unbounded == {"A": [45.0, 50.0]}
        assert intervals.series["A"].lower == [-math.inf, -math.inf]
        assert intervals.series["A"].upper == [math.inf, math.inf]
        assert all(math.isfinite(v) for v in intervals.series["B"].upper)
