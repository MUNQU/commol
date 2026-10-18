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
    CalibrationProblem,
    Calibrator,
    EnbPI,
    JackknifePlus,
    NelderMeadConfig,
    ObservedDataPoint,
    ParameterSample,
    RollingConformal,
    SampleMethod,
    Simulation,
    SplitConformal,
)
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    flow_calibrator,
    flow_observations,
)

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


def _flow_calibrator(observations: list[ObservedDataPoint]) -> Calibrator:
    """The calibrator of ``flow_calibrator`` with other observations."""
    model = build(a_to_b_builder(parameters={"c": None}, rate="c * A / A", total=1e6))
    problem = CalibrationProblem(
        observed_data=observations,
        parameters=[bounded("c", 0.0, 50.0, initial_guess=25.0)],
        loss_function="sse",
        optimization_config=NelderMeadConfig(max_iterations=1000, sd_tolerance=1e-14),
    )
    return Calibrator(Simulation(model), problem)


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


class TestRollingConformal:
    def test_horizon_intervals_widen_on_a_random_walk(self) -> None:
        rng = np.random.default_rng(4)
        walk = np.cumsum(rng.normal(0.0, 1.0, 60))
        values = [C * t + walk[t - 1] for t in range(1, 61)]
        calibrator = flow_calibrator(values, windowed=False)

        result = calibrator.uncertainty(
            RollingConformal(
                origins=[float(t) for t in range(20, 51)],
                score="absolute",
                by="series",
                level=0.8,
                positions=[61.0, 62.0, 63.0, 64.0, 65.0, 66.0],
            )
        )

        series = result.intervals.series["B"]
        widths = [hi - lo for lo, hi in zip(series.lower, series.upper, strict=True)]
        # The error of a random walk k steps ahead grows as √k.
        assert widths[0] < widths[2] < widths[5], widths

    def test_a_horizon_counts_the_observations_after_the_origin(self) -> None:
        # Flows of 10 up to the origin at 5, then 11, 12 and 13 at irregular
        # positions: the fit at the origin is 10, so horizon k scores k.
        observations = flow_observations([10.0] * 5, windowed=True) + [
            ObservedDataPoint(at=at, output="B", value=value, window=1)
            for at, value in [(7, 11.0), (10, 12.0), (20, 13.0)]
        ]
        calibrator = _flow_calibrator(observations)

        result = calibrator.uncertainty(
            RollingConformal(
                origins=[5.0],
                score="absolute",
                by="series",
                level=0.5,
                positions=[21.0, 22.0, 23.0],
            )
        )

        assert result.scores["B"] == pytest.approx([1.0, 2.0, 3.0])
        series = result.intervals.series["B"]
        half_widths = [
            (hi - lo) / 2 for lo, hi in zip(series.lower, series.upper, strict=True)
        ]
        assert half_widths == pytest.approx([1.0, 2.0, 3.0])


class TestEnbPI:
    def test_enbpi_needs_each_block_left_out(self) -> None:
        calibrator = flow_calibrator(_amounts(range(1, 41), 1), windowed=False)

        # Two replicates of 40 blocks: a block is left out by neither with
        # probability (1 − 1/e)² ≈ 0.4.
        with pytest.raises(ValueError, match=r"[Bb]lock \[.*replicates"):
            calibrator.uncertainty(
                EnbPI(
                    replicates=2,
                    block_length=1,
                    score="absolute",
                    by="series",
                    level=0.9,
                    positions=[41.0],
                )
            )
