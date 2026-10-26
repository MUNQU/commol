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
    BayesianBootstrap,
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
from commol.context.inference.conformal import (
    ConformalResult,
)
from tests.support import (
    a_to_b_builder,
    bounded,
    build,
    flow_calibrator,
    flow_observations,
    rate_calibrator,
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
    model = build(
        a_to_b_builder(
            parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
        )
    )
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
            "confidence_level": 0.9,
            "positions": [41.0],
        },
        RollingConformal: {
            "origins": [10.0, 12.0],
            "confidence_level": 0.9,
            "positions": [41.0],
        },
        JackknifePlus: {
            "block_length": 1,
            "confidence_level": 0.9,
            "positions": [41.0],
        },
        EnbPI: {
            "replicates": 20,
            "block_length": 1,
            "confidence_level": 0.9,
            "positions": [41.0],
        },
        AdaptiveConformal: {
            "confidence_level": 0.9,
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
                confidence_level=0.8,
                positions=positions,
                sample=sample,
            )
        )
        absolute = calibrator.uncertainty(
            SplitConformal(
                calibration_data=held_out,
                score="absolute",
                by="series",
                confidence_level=0.8,
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
            if given is None:
                assert f"missing field `{argument}`" in message, message
            else:
                assert f"{argument}: unknown variant `other`" in message, message
                for value in values:
                    assert f"`{value}`" in message, message

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_normalized_score_needs_pooled_quantiles(self, variant: type) -> None:
        settings = _variant_settings()[variant]

        with pytest.raises(ValueError) as error:
            variant(**settings, score="normalized", by="series")

        message = str(error.value)
        assert '"absolute"' in message and "equal" in message
        assert 'by="all"' in message and 'score="absolute"' in message
        assert variant(**settings, score="normalized", by="all").by == "all"

    @pytest.mark.parametrize("variant", [SplitConformal, AdaptiveConformal])
    def test_the_spread_score_needs_a_sample(self, variant: type) -> None:
        settings = _variant_settings()[variant]

        with pytest.raises(ValueError, match="give a sample"):
            variant(**settings, score="spread", by="series")

    def test_rolling_origins_have_no_spread(self) -> None:
        settings = _variant_settings()[RollingConformal]

        with pytest.raises(ValueError, match='once per origin.*"absolute"'):
            RollingConformal(**settings, score="spread", by="series")

    def test_a_series_with_few_points_gets_an_unbounded_interval(self) -> None:
        calibrator = flow_calibrator(_amounts(range(1, 21), 1), windowed=False)
        # At level 0.9 a series needs 1/0.1 − 1 = 9 held-out points.
        held_out = _held_out(range(21, 41), 2) + _held_out(range(21, 29), 3, "A")

        result = calibrator.uncertainty(
            SplitConformal(
                calibration_data=held_out,
                score="absolute",
                by="series",
                confidence_level=0.9,
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
                confidence_level=0.8,
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
                confidence_level=0.5,
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
                    confidence_level=0.9,
                    positions=[41.0],
                )
            )


class TestAdaptiveConformal:
    def test_adaptive_conformal_recovers_after_a_change_of_regime(self) -> None:
        rng = np.random.default_rng(8)
        calibrator = flow_calibrator(
            (C + rng.normal(0.0, 1.0, 30)).tolist(), windowed=True
        )
        # The noise of the flows grows from 1 to 3 after 100 points.
        noise = np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(0.0, 3.0, 500)])
        stream = flow_observations((C + noise).tolist(), windowed=True, start=31)
        settings = {"score": "absolute", "by": "series", "confidence_level": 0.9}

        adaptive = calibrator.uncertainty(
            AdaptiveConformal(**settings, step=0.02, stream=stream)
        )
        split = calibrator.uncertainty(
            SplitConformal(
                **settings,
                calibration_data=stream[:100],
                positions=[point.at for point in stream[100:]],
            )
        )

        def late_miss_rate(series: object, observed: list[ObservedDataPoint]) -> float:
            lower, upper = series.lower[-300:], series.upper[-300:]  # type: ignore[attr-defined]
            values = [point.value for point in observed[-300:]]
            misses = [
                not lo <= y <= hi
                for lo, hi, y in zip(lower, upper, values, strict=True)
            ]
            return sum(misses) / len(misses)

        assert late_miss_rate(adaptive.intervals.series["B"], stream) == pytest.approx(
            0.1, abs=0.04
        )
        assert late_miss_rate(split.intervals.series["B"], stream) > 0.4


class TestResults:
    @pytest.mark.parametrize(
        ("variant", "assumption"),
        [
            (SplitConformal, "exchangeable"),
            (RollingConformal, "exchangeable"),
            (JackknifePlus, "exchangeable"),
            (EnbPI, "mixing"),
            (AdaptiveConformal, "no assumption"),
        ],
    )
    def test_each_result_states_its_guarantee(
        self, variant: type, assumption: str
    ) -> None:
        values = (C + np.random.default_rng(5).normal(0.0, 1.0, 50)).tolist()
        calibrator = flow_calibrator(values[:30], windowed=True)
        later = flow_observations(values[30:], windowed=True, start=31)
        settings = {
            SplitConformal: {"calibration_data": later, "positions": [51.0]},
            RollingConformal: {"origins": [10.0, 15.0, 20.0], "positions": [31.0]},
            JackknifePlus: {"block_length": 1, "positions": [31.0]},
            EnbPI: {"replicates": 20, "block_length": 3, "positions": [31.0]},
            AdaptiveConformal: {"step": 0.01, "stream": later},
        }[variant]

        result = calibrator.uncertainty(
            variant(**settings, score="absolute", by="series", confidence_level=0.5)
        )

        assert assumption in result.guarantee, result.guarantee
        assert list(result.scores) == ["B"]


class TestFailures:
    def test_a_conformal_result_counts_the_refits_that_did_not_converge(
        self,
    ) -> None:
        model = build(
            a_to_b_builder(
                parameters={"c": None}, rate="c * A / A", amounts={"A": 1e6, "B": 0}
            )
        )
        problem = CalibrationProblem(
            observed_data=flow_observations(_amounts(range(1, 21), 3), windowed=False),
            parameters=[bounded("c", 0.0, 50.0, initial_guess=25.0)],
            loss_function="sse",
            optimization_config=NelderMeadConfig(max_iterations=1),
        )
        calibrator = Calibrator(Simulation(model), problem)

        result = calibrator.uncertainty(
            RollingConformal(
                origins=[10.0, 15.0],
                score="absolute",
                by="series",
                confidence_level=0.5,
                positions=[21.0],
            )
        )

        # One iteration stops both refits.
        assert result.not_converged == 2
        assert result.failed == []

    def test_a_prediction_that_fails_names_its_parameter_set(self) -> None:
        # The rate is not a number after 25.5, inside the calibration data.
        calibrator = rate_calibrator(
            "c * A / A + 0 * sqrt(25.5 - t)", [10.0, 11.0, 9.0, 10.5]
        )

        with pytest.raises(
            RuntimeError,
            match="Parameter set 0: Simulation failed: The rate of flow 't_ab' at 26",
        ):
            calibrator.uncertainty(
                SplitConformal(
                    calibration_data=flow_observations(
                        [C] * 30, windowed=True, start=5
                    ),
                    score="absolute",
                    by="series",
                    confidence_level=0.5,
                    positions=[40.0],
                )
            )

    def test_every_refit_failing_gives_the_reasons(self) -> None:
        # The fit of every flow is 11.5; the refit at the origin 2 heads for
        # 40, past 20, where the rate is not a number.
        calibrator = rate_calibrator(
            "c * A / A + 0 * sqrt(20 - c)", [40.0] * 2 + [C] * 38
        )

        assert calibrator.run().best_parameters["c"] == pytest.approx(11.5)
        with pytest.raises(
            RuntimeError,
            match=r"Every origin failed: 0 of 1 origins fitted, and the method needs "
            r"1; the first reasons: origin 0: Every start failed: Start 0 failed: "
            r"Optimization failed: Simulation failed",
        ):
            calibrator.uncertainty(
                RollingConformal(
                    origins=[2.0],
                    score="absolute",
                    by="series",
                    confidence_level=0.5,
                    positions=[41.0],
                )
            )


def test_a_conformal_method_takes_a_confidence_level() -> None:
    values = (C + np.random.default_rng(5).normal(0.0, 1.0, 50)).tolist()
    calibrator = flow_calibrator(values[:30], windowed=True)
    later = flow_observations(values[30:], windowed=True, start=31)

    result = calibrator.uncertainty(
        SplitConformal(
            calibration_data=later,
            positions=[51.0],
            score="absolute",
            by="series",
            confidence_level=0.5,
        )
    )

    assert result.intervals.confidence_level == 0.5
    assert "probability at least 0.5 " in result.guarantee
    # Split conformal draws no blocks.
    assert result.block_length is None
    assert result.block_length_estimates is None


@pytest.mark.parametrize(
    ("variant", "settings"),
    [(JackknifePlus, {}), (EnbPI, {"replicates": 20})],
    ids=["jackknife_plus", "enbpi"],
)
def test_a_conformal_result_gives_its_block_length_and_estimates(
    variant: type, settings: dict[str, object]
) -> None:
    values = (C + np.random.default_rng(6).normal(0.0, 1.0, 30)).tolist()
    calibrator = flow_calibrator(values, windowed=True)

    def run(block_length: int | str) -> ConformalResult:
        return calibrator.uncertainty(
            variant(
                **settings,
                block_length=block_length,
                score="absolute",
                by="series",
                confidence_level=0.5,
                positions=[31.0],
            )
        )

    automatic, given = run("auto"), run(3)

    # The blocks are those of the Bayesian bootstrap: non-overlapping, on the
    # residuals of the fit.
    blocks = calibrator.uncertainty(
        BayesianBootstrap(replicates=1, block_length="auto")
    ).diagnostics
    estimates = automatic.block_length_estimates
    assert estimates is not None and list(estimates) == ["B"]
    assert estimates == blocks.block_length_estimates
    assert automatic.block_length == blocks.block_length
    assert given.block_length == 3
    assert given.block_length_estimates is None


@pytest.mark.parametrize(
    ("variant", "settings"),
    [(JackknifePlus, {}), (EnbPI, {"replicates": 20})],
    ids=["jackknife_plus", "enbpi"],
)
def test_a_block_longer_than_the_data_is_an_error(
    variant: type, settings: dict[str, object]
) -> None:
    values = (C + np.random.default_rng(6).normal(0.0, 1.0, 30)).tolist()
    method = variant(
        **settings,
        block_length=1000,
        score="absolute",
        by="series",
        confidence_level=0.5,
        positions=[31.0],
    )

    with pytest.raises(ValueError, match="block_length is 1000; it must be between 1"):
        flow_calibrator(values, windowed=True).uncertainty(method)
