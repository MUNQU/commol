"""Tests for converting between physical time and simulation steps."""

import math

import pytest
from pydantic import ValidationError

from commol import TimeScale
from commol.api.time_scale import SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_WEEK

HOURLY = TimeScale(step_seconds=SECONDS_PER_HOUR)
DAILY = TimeScale(step_seconds=SECONDS_PER_DAY)


class TestConstruction:
    @pytest.mark.parametrize("step_seconds", [0.0, -1.0])
    def test_non_positive_step_is_rejected(self, step_seconds: float) -> None:
        with pytest.raises(ValidationError):
            TimeScale(step_seconds=step_seconds)

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            HOURLY.step_seconds = 1.0


class TestStepConversion:
    def test_hourly_steps(self) -> None:
        assert HOURLY.steps_per_hour == 1
        assert HOURLY.steps_per_day == 24
        assert HOURLY.steps_per_week == 168

    def test_daily_steps(self) -> None:
        assert DAILY.steps_per_day == 1
        assert DAILY.steps_per_week == 7

    def test_named_units_agree_with_seconds(self) -> None:
        assert HOURLY.steps_from_minutes(120) == HOURLY.steps_from_hours(2)
        assert HOURLY.steps_from_days(7) == HOURLY.steps_from_weeks(1)
        assert HOURLY.steps_from_days(1) == HOURLY.steps_from_seconds(SECONDS_PER_DAY)

    def test_a_period_that_is_not_a_whole_number_of_steps_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="whole number of steps"):
            DAILY.steps_from_hours(1)

    def test_an_hour_is_not_expressible_in_daily_steps(self) -> None:
        with pytest.raises(ValueError, match="whole number of steps"):
            _ = DAILY.steps_per_hour

    def test_a_step_longer_than_a_day_still_expresses_a_week(self) -> None:
        scale = TimeScale(step_seconds=SECONDS_PER_WEEK / 2)

        assert scale.steps_per_week == 2
        with pytest.raises(ValueError, match="whole number of steps"):
            _ = scale.steps_per_day


class TestRateFromMeanDuration:
    def test_matches_the_exponential_definition(self) -> None:
        rate = DAILY.rate_from_mean_duration(4 * SECONDS_PER_DAY)

        assert rate == pytest.approx(1.0 - math.exp(-0.25))

    def test_a_mean_duration_of_one_step_leaves_most_of_the_state(self) -> None:
        rate = DAILY.rate_from_mean_duration(SECONDS_PER_DAY)

        assert rate == pytest.approx(1.0 - math.exp(-1.0))

    def test_a_longer_mean_duration_gives_a_smaller_rate(self) -> None:
        slow = DAILY.rate_from_mean_duration(10 * SECONDS_PER_DAY)
        fast = DAILY.rate_from_mean_duration(2 * SECONDS_PER_DAY)

        assert 0.0 < slow < fast < 1.0

    def test_a_finer_step_gives_a_smaller_rate_for_the_same_duration(self) -> None:
        duration = 4 * SECONDS_PER_DAY

        assert HOURLY.rate_from_mean_duration(duration) < DAILY.rate_from_mean_duration(
            duration
        )

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_non_positive_duration_is_rejected(self, duration: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            DAILY.rate_from_mean_duration(duration)


class TestRateFromProbability:
    def test_accumulates_to_the_given_probability_over_the_period(self) -> None:
        probability = 0.3
        period = HOURLY.steps_per_week
        rate = HOURLY.rate_from_probability(probability, period)

        assert 1.0 - (1.0 - rate) ** period == pytest.approx(probability)

    def test_a_single_step_period_returns_the_probability(self) -> None:
        assert DAILY.rate_from_probability(0.3, 1) == pytest.approx(0.3)

    def test_certainty_and_impossibility_are_preserved(self) -> None:
        assert DAILY.rate_from_probability(0.0, 10) == pytest.approx(0.0)
        assert DAILY.rate_from_probability(1.0, 10) == pytest.approx(1.0)

    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_probability_outside_the_unit_interval_is_rejected(
        self, probability: float
    ) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            DAILY.rate_from_probability(probability, 7)

    @pytest.mark.parametrize("period_steps", [0, -1])
    def test_non_positive_period_is_rejected(self, period_steps: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            DAILY.rate_from_probability(0.3, period_steps)


class TestFromStepUnit:
    def test_a_unit_of_time_gives_its_length(self) -> None:
        assert TimeScale.from_step_unit("day").step_seconds == 86400.0
        assert TimeScale.from_step_unit("h").step_seconds == 3600.0

    def test_a_step_unit_that_is_not_time_is_named(self) -> None:
        with pytest.raises(ValueError, match="'meter'"):
            TimeScale.from_step_unit("meter")
