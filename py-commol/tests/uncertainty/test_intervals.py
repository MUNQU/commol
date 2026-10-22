"""The one confidence-interval convention: weighted quantiles in Rust, and
one check of every confidence level."""

import re
from collections.abc import Callable

import numpy as np
import pytest

from commol import (
    BayesianBootstrap,
    BootstrapCoverageEta,
    BootstrapResult,
    InputSpace,
    ObservedDataPoint,
    ParameterSample,
    PropagationResult,
    SampleMethod,
    Scenarios,
    SensitivityAnalysis,
    Simulation,
    Sobol,
    SplitConformal,
    Threshold,
    ValueAt,
    ci_percentiles,
    member_statistics,
    weighted_quantiles,
)
from tests.support import abc_builder, bounded, build


class TestCiPercentiles:
    def test_returns_two_sided_points_on_the_numpy_scale(self) -> None:
        assert ci_percentiles(0.95) == pytest.approx((2.5, 97.5))

    def test_points_are_symmetric_about_the_median(self) -> None:
        for confidence_level in (0.5, 0.8, 0.9, 0.99):
            lower, upper = ci_percentiles(confidence_level)
            assert lower + upper == pytest.approx(100.0)

    def test_width_matches_the_confidence_level(self) -> None:
        lower, upper = ci_percentiles(0.8)
        assert upper - lower == pytest.approx(80.0)


class TestMemberStatistics:
    def test_scalar_members_reduce_to_scalars(self) -> None:
        stats = member_statistics([1.0, 2.0, 3.0, 4.0, 5.0], 0.95)

        assert stats["mean"] == pytest.approx(3.0)
        assert stats["median"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert isinstance(stats["mean"], float)

    def test_series_members_reduce_positionwise(self) -> None:
        stats = member_statistics([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], 0.95)

        assert stats["mean"] == pytest.approx([2.0, 20.0])
        assert stats["median"] == pytest.approx([2.0, 20.0])
        assert stats["min"] == pytest.approx([1.0, 10.0])
        assert stats["max"] == pytest.approx([3.0, 30.0])

    def test_interval_uses_the_percentile_points(self) -> None:
        values = list(np.linspace(0.0, 100.0, 401) ** 2)
        lower_point, upper_point = ci_percentiles(0.9)
        stats = member_statistics(values, 0.9)

        assert stats["ci_lower"] == pytest.approx(
            float(np.percentile(values, lower_point, method="hazen"))
        )
        assert stats["ci_upper"] == pytest.approx(
            float(np.percentile(values, upper_point, method="hazen"))
        )

    def test_weights_move_the_statistics_towards_heavy_members(self) -> None:
        stats = member_statistics([0.0, 1.0], 0.5, weights=[1.0, 3.0])

        # Mean 0.75; midpoints 1/8 and 5/8, so the median is 3/4 of the way.
        assert stats["mean"] == pytest.approx(0.75)
        assert stats["median"] == pytest.approx(0.75)

    def test_a_member_of_weight_zero_is_left_out(self) -> None:
        with_zero = member_statistics([1.0, 100.0, 3.0], 0.9, weights=[1, 0, 1])
        without = member_statistics([1.0, 3.0], 0.9)

        assert with_zero == pytest.approx(without)

    def test_weights_must_match_the_members(self) -> None:
        with pytest.raises(ValueError, match="One weight per member"):
            member_statistics([1.0, 2.0], 0.9, weights=[1.0])

    def test_interval_stays_within_the_range_of_member_values(self) -> None:
        """Statistics of a reduced quantity are bounded by the member values."""
        members = [[0.0, float(scale), float(2 * scale)] for scale in (1, 2, 3, 4, 5)]
        final_values = [member[-1] for member in members]

        reduced_first = member_statistics(final_values, 0.95)
        band_per_step = member_statistics(members, 0.95)

        assert reduced_first["ci_upper"] == pytest.approx(band_per_step["ci_upper"][-1])
        assert reduced_first["ci_upper"] <= max(final_values)
        assert reduced_first["ci_lower"] >= min(final_values)

    def test_single_member_has_a_degenerate_interval(self) -> None:
        stats = member_statistics([[7.0, 8.0]], 0.95)

        assert stats["ci_lower"] == pytest.approx(stats["ci_upper"])
        assert stats["ci_lower"] == pytest.approx([7.0, 8.0])

    def test_empty_ensemble_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one ensemble member"):
            member_statistics([], 0.95)


class TestWeightedQuantiles:
    """The quantile of commol, by its definition."""

    @pytest.mark.parametrize("levels", [[0.1, 0.5, 0.8], [0.0, 0.33, 1.0]])
    def test_equal_weights_give_numpy_hazen(self, levels: list[float]) -> None:
        values = [4.0, 1.0, 8.0, 2.0, 16.0]

        assert weighted_quantiles(values, levels) == pytest.approx(
            np.percentile(values, [100 * level for level in levels], method="hazen")
        )

    def test_a_weight_of_zero_removes_a_value(self) -> None:
        levels = [0.2, 0.5, 0.8]

        assert weighted_quantiles([1.0, 100.0, 3.0], levels, [0.5, 0.0, 0.5]) == (
            weighted_quantiles([1.0, 3.0], levels)
        )

    def test_the_scale_of_the_weights_does_not_matter(self) -> None:
        values = [1.0, 2.0, 5.0]

        assert weighted_quantiles(values, [0.3, 0.6], [1, 2, 1]) == pytest.approx(
            weighted_quantiles(values, [0.3, 0.6], [0.25, 0.5, 0.25])
        )

    def test_a_weighted_quantile_interpolates_between_midpoints(self) -> None:
        # Weights 1/4 and 3/4 put the values at 1/8 and 5/8.
        assert weighted_quantiles([0.0, 1.0], [0.5], [1, 3]) == pytest.approx([0.75])

    def test_a_weight_per_value(self) -> None:
        with pytest.raises(
            ValueError, match="One weight per value: 2 values, 1 weights"
        ):
            weighted_quantiles([1.0, 2.0], [0.5], [1.0])

    def test_member_statistics_need_a_positive_weight(self) -> None:
        with pytest.raises(ValueError, match="a positive weight"):
            member_statistics([[1.0], [2.0]], 0.9, weights=[0.0, 0.0])

    def test_no_positive_weight_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive weight"):
            weighted_quantiles([1.0], [0.5], [0.0])


LEVEL_MESSAGE = (
    "confidence_level is strictly between 0 and 1, got 1.5; use 0.95 for a 95% interval"
)


def _sample() -> ParameterSample:
    return ParameterSample(
        names=["k1"],
        bounds=[(0.0, 0.5)],
        values=[[0.2], [0.3]],
        log_weights=[0.0, 0.0],
        method=SampleMethod(name="test"),
    )


def _scenarios() -> Scenarios:
    analysis = SensitivityAnalysis(
        Simulation(build(abc_builder())),
        InputSpace.uniform_on_bounds([bounded("k1", 0.0, 0.5)]),
    )
    return Scenarios(analysis, _sample(), {"base": {}, "slow": {"k2": 0.02}}, [0, 1])


ONE = [ValueAt(series="C", at=1)]

TAKES_A_LEVEL: dict[str, Callable[[float], object]] = {
    "ci_percentiles": ci_percentiles,
    "member_statistics": lambda c: member_statistics([1.0, 2.0], c),
    "ParameterSample.interval": lambda c: _sample().interval(c),
    "BootstrapResult.interval_error": (
        lambda c: BootstrapResult.model_construct().interval_error(c)
    ),
    "PropagationResult.summary": lambda c: PropagationResult(
        sample=_sample(),
        labels=["x"],
        values=[[1.0], [2.0]],
        weights=[0.5, 0.5],
        draw_seeds=[1, 2],
    ).summary(c),
    "Scenarios.summary": lambda c: _scenarios().summary("base", ONE, c),
    "Scenarios.compare": lambda c: _scenarios().compare("slow", "base", ONE, c),
    "Threshold": lambda c: Threshold(kind="beale", confidence_level=c),
    "Threshold.chi_square": Threshold.chi_square,
    "Threshold.beale": Threshold.beale,
    "Threshold.bootstrap_calibrated": lambda c: Threshold.bootstrap_calibrated(
        BayesianBootstrap(replicates=5, block_length=1), c
    ),
    "SplitConformal": lambda c: SplitConformal(
        calibration_data=[ObservedDataPoint(at=1, output="B", value=1.0)],
        positions=[2.0],
        score="absolute",
        by="all",
        confidence_level=c,
    ),
    "Sobol": lambda c: Sobol(draws=8, replicates=2, confidence_level=c, quantities=ONE),
    "BootstrapCoverageEta": lambda c: BootstrapCoverageEta(
        replicates=2, block_length=1, confidence_level=c
    ),
}


@pytest.mark.parametrize("takes_a_level", TAKES_A_LEVEL.values(), ids=TAKES_A_LEVEL)
def test_every_confidence_level_has_one_check(
    takes_a_level: Callable[[float], object],
) -> None:
    with pytest.raises(ValueError, match=re.escape(LEVEL_MESSAGE)):
        takes_a_level(1.5)
