"""Tests for the confidence-interval helpers."""

import numpy as np
import pytest

from commol import ci_percentiles, member_statistics


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

    @pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.5, 95.0])
    def test_out_of_range_confidence_level_is_rejected(
        self, confidence_level: float
    ) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            ci_percentiles(confidence_level)

    def test_percentage_mistake_is_called_out(self) -> None:
        with pytest.raises(ValueError, match="fraction, not a percentage"):
            ci_percentiles(95.0)


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
        values = list(np.linspace(0.0, 100.0, 401))
        lower_point, upper_point = ci_percentiles(0.9)
        stats = member_statistics(values, 0.9)

        assert stats["ci_lower"] == pytest.approx(
            float(np.percentile(values, lower_point))
        )
        assert stats["ci_upper"] == pytest.approx(
            float(np.percentile(values, upper_point))
        )

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

    def test_out_of_range_confidence_level_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            member_statistics([1.0, 2.0], 1.0)
