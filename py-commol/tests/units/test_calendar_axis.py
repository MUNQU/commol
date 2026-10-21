"""Tests for an axis anchored to calendar time: dates, positions and periods."""

import calendar
import json
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from commol import AxisUnits
from commol.context.model import Model

from tests.support import abc_builder, build

HOURS_FROM_2020 = AxisUnits(step_unit="hour", origin=date(2020, 1, 1))
DAYS_FROM_A_WEDNESDAY = AxisUnits(step_unit="day", origin=date(2020, 1, 1))


def test_dates_and_positions_are_inverse() -> None:
    assert HOURS_FROM_2020.position_of(date(2020, 1, 2)) == 24.0
    assert HOURS_FROM_2020.moment_of(24) == datetime(2020, 1, 2)
    for position in (-30.0, 0.0, 1.5, 24.0, 8760.25):
        moment = HOURS_FROM_2020.moment_of(position)
        assert HOURS_FROM_2020.position_of(moment) == position


def test_a_date_is_its_midnight() -> None:
    assert HOURS_FROM_2020.origin == datetime(2020, 1, 1)
    assert HOURS_FROM_2020.position_of(datetime(2020, 1, 1, 6, 30)) == 6.5
    assert HOURS_FROM_2020.position_of(date(2020, 1, 1)) == 0.0


def test_an_axis_without_origin_has_no_dates() -> None:
    axis = AxisUnits(step_unit="hour")

    with pytest.raises(ValueError, match="has no origin"):
        axis.position_of(date(2020, 1, 2))
    with pytest.raises(ValueError, match="has no origin"):
        axis.moment_of(24)
    with pytest.raises(ValueError, match="has no origin"):
        axis.day_ends(0, 48)


@pytest.mark.parametrize("step_unit", ["month", "year"])
def test_an_origin_needs_a_unit_of_fixed_calendar_length(step_unit: str) -> None:
    with pytest.raises(
        ValueError, match="fixed length, not the length of a calendar month or year"
    ):
        AxisUnits(step_unit=step_unit, origin=date(2020, 1, 1))


@pytest.mark.parametrize("step_unit", ["second", "minute", "hour", "day", "week"])
def test_a_step_of_whole_days_or_a_whole_fraction_of_a_day_takes_an_origin(
    step_unit: str,
) -> None:
    axis = AxisUnits(step_unit=step_unit, origin=date(2020, 1, 1))

    assert axis.moment_of(axis.steps(14, "day")) == datetime(2020, 1, 15)


def test_an_origin_needs_a_step_unit_of_time() -> None:
    with pytest.raises(ValidationError, match="must be a unit of time") as length:
        AxisUnits(step_unit="meter", origin=date(2020, 1, 1))
    with pytest.raises(ValidationError, match="Declare the unit of a step"):
        AxisUnits(step_unit="step", origin=date(2020, 1, 1))

    assert "Declare" not in str(length.value)


def test_a_moment_with_a_time_zone_is_rejected() -> None:
    aware = datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=1)))

    with pytest.raises(ValidationError, match=r"astimezone\(zone\)"):
        AxisUnits(step_unit="hour", origin=aware)
    with pytest.raises(ValueError, match="has no time zone"):
        HOURS_FROM_2020.position_of(aware)


@pytest.mark.parametrize("position", [float("nan"), float("inf")])
def test_a_moment_needs_a_finite_position(position: float) -> None:
    with pytest.raises(ValueError, match="is finite"):
        HOURS_FROM_2020.moment_of(position)
    with pytest.raises(ValueError, match="is finite"):
        HOURS_FROM_2020.day_ends(0, position)


def test_week_ends_follow_the_start_weekday() -> None:
    assert date(2020, 1, 1).weekday() == calendar.WEDNESDAY

    ends = DAYS_FROM_A_WEDNESDAY.week_ends(0, 30, first_weekday=calendar.SUNDAY)

    assert ends == [4.0, 11.0, 18.0, 25.0]
    for end in ends:
        moment = DAYS_FROM_A_WEDNESDAY.moment_of(end)
        assert moment.weekday() == calendar.SUNDAY
        assert moment.time() == datetime.min.time()


def test_a_period_that_ends_at_the_start_is_not_in_the_range() -> None:
    assert DAYS_FROM_A_WEDNESDAY.week_ends(4, 18, calendar.SUNDAY) == [11.0, 18.0]
    assert DAYS_FROM_A_WEDNESDAY.week_ends(0, 3, calendar.SUNDAY) == []


def test_day_ends_are_the_midnights_after_the_start() -> None:
    assert HOURS_FROM_2020.day_ends(6, 72) == [24.0, 48.0, 72.0]
    assert HOURS_FROM_2020.day_ends(24, 47.5) == []
    assert DAYS_FROM_A_WEDNESDAY.day_ends(0.5, 3) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("first_weekday", [-1, 7, 1.5])
def test_a_week_starts_on_a_day_from_monday_to_sunday(first_weekday: float) -> None:
    with pytest.raises(ValueError, match="from 0 \\(Monday\\) to 6 \\(Sunday\\)"):
        DAYS_FROM_A_WEDNESDAY.week_ends(0, 30, first_weekday)  # ty: ignore[invalid-argument-type]


class TestModelOrigin:
    def test_the_model_keeps_its_origin_for_its_axis(self, tmp_path) -> None:
        model = build(abc_builder(step_unit="day", origin=date(2020, 1, 1)))

        assert model.origin == datetime(2020, 1, 1)
        assert model.axis_units().moment_of(7) == datetime(2020, 1, 8)
        path = tmp_path / "model.json"
        model.to_json(path)
        assert json.loads(path.read_text())["origin"] == "2020-01-01T00:00:00"
        assert Model.from_json(path).origin == datetime(2020, 1, 1)

    def test_a_model_without_origin_has_none(self) -> None:
        assert build(abc_builder(step_unit="day")).axis_units().origin is None

    @pytest.mark.parametrize(
        ("step_unit", "message"),
        [("month", "not the length of a calendar month"), (None, "unit of time")],
    )
    def test_build_checks_the_step_unit_of_an_origin(
        self, step_unit: str | None, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            build(abc_builder(step_unit=step_unit, origin=date(2020, 1, 1)))
