"""Tests for values in units as steps, rates per engine, and the window grid."""

import math

import pytest
from pydantic import ValidationError

from commol import Aggregate, AxisUnits, Simulation
from commol.commol_rs import _commol_rs as commol_rs
from commol.utils.equations import UnitConsistencyError

from tests.support import (
    FactoryBuilder,
    TIGHT,
    a_to_b_builder,
    assert_close,
    build,
    build_ode,
)


class TestSteps:
    def test_a_time_axis_converts_any_unit_of_time(self) -> None:
        hours = AxisUnits(step_unit="hour")

        assert hours.steps(3, "day") == 72.0
        assert hours.steps(1, "week") == 168.0
        assert hours.steps(30, "minute") == 0.5

    def test_a_length_axis_converts_any_unit_of_length(self) -> None:
        meters = AxisUnits(step_unit="m")

        assert meters.steps(2, "km") == 2000.0
        assert meters.steps(50, "cm") == 0.5

    def test_a_result_near_an_integer_is_that_integer(self) -> None:
        assert AxisUnits(step_unit="minute").steps(4.1, "hour") == 246.0

    def test_a_large_result_is_near_an_integer_by_its_distance_alone(self) -> None:
        assert AxisUnits(step_unit="second").steps(1e10 + 0.5, "second") == 1e10 + 0.5

    def test_a_value_of_another_dimension_names_both_units(self) -> None:
        with pytest.raises(ValueError, match=r"'day'.*'meter'"):
            AxisUnits(step_unit="meter").steps(3, "day")

    def test_a_physical_unit_on_the_default_step_says_to_declare_one(self) -> None:
        with pytest.raises(ValueError) as error:
            AxisUnits().steps(3, "day")

        assert str(error.value).startswith("Cannot convert 3 day to steps of 'step'")
        assert str(error.value).endswith(
            "'step' has dimension [step]. Declare the unit of a step (step_unit in "
            "ModelBuilder) to convert values in that unit."
        )

    def test_a_unit_pint_does_not_define_is_its_own_dimension(self) -> None:
        generations = AxisUnits(step_unit="generation")

        assert generations.steps(4, "generation") == 4.0
        with pytest.raises(ValueError, match="'generation'"):
            generations.steps(4, "day")

    @pytest.mark.parametrize("unit", ["1/day", "m**2"])
    def test_the_step_unit_is_one_unit_of_one_dimension(self, unit: str) -> None:
        with pytest.raises(UnitConsistencyError, match="one dimension"):
            AxisUnits(step_unit=unit)

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            AxisUnits().step_unit = "day"  # type: ignore[misc]


class TestModelAxisUnits:
    def test_a_model_gives_its_step_unit_and_engine(self) -> None:
        builder = FactoryBuilder(name="Daily", step_unit="day")
        builder.add_bin(id="A", name="A")
        builder.add_bin(id="B", name="B")
        builder.add_transition(id="ab", source=["A"], target=["B"], rate=0.1)
        builder.set_structure(Aggregate(bin_amounts={"A": 1, "B": 0}))

        units = build_ode(builder).axis_units()

        assert units == AxisUnits(step_unit="day", engine="DifferentialEquations")

    def test_a_model_without_a_step_unit_has_steps(self) -> None:
        units = build(a_to_b_builder()).axis_units()

        assert units == AxisUnits(step_unit="step", engine="DifferenceEquations")


class TestWindowGrid:
    """The axis gives the window grid of a width in a unit, in steps."""

    def test_the_window_grid_of_the_axis_is_in_its_unit(self) -> None:
        hours = AxisUnits(step_unit="hour")
        week = 168

        assert hours.window_end(2, 1, "week") == 504
        assert (hours.window_start(2, 1, "week"), hours.window_end(2, 1, "week")) == (
            commol_rs.calibration.window_bounds(2, week)
        )
        assert hours.window_start(2, 7, "day") == 336
        assert [hours.window_index(at, 1, "week") for at in [0, 168, 168.5, 504]] == [
            0,
            0,
            1,
            2,
        ]
        assert hours.window_ends(1, "week", 600) == [168, 336, 504]
        assert hours.window_ends(1, "week", 600) == (
            commol_rs.calibration.window_ends(week, 600)
        )

    def test_a_width_in_steps_is_the_span_in_the_step_unit(self) -> None:
        assert AxisUnits().window_ends(5, "step", 20) == [5, 10, 15, 20]
        assert AxisUnits(step_unit="day").window_index(14, 7, "day") == 1

    @pytest.mark.parametrize(
        ("step_unit", "span", "unit", "steps"),
        [
            ("day", 12, "hour", "0.5"),
            ("step", 2.5, "step", "2.5"),
            ("day", 0, "day", "0"),
        ],
    )
    def test_a_window_must_be_whole_steps(
        self, step_unit: str, span: float, unit: str, steps: str
    ) -> None:
        with pytest.raises(ValueError) as error:
            AxisUnits(step_unit=step_unit).window_end(0, span, unit)

        assert str(error.value) == (
            f"A window must be a positive whole number of steps: {span} {unit} "
            f"is {steps} steps of '{step_unit}'."
        )

    def test_a_window_in_another_dimension_names_both_units(self) -> None:
        with pytest.raises(ValueError, match=r"'meter'.*'day'"):
            AxisUnits(step_unit="day").window_ends(1, "meter", 10)

    def test_a_negative_index_or_position_is_rejected(self) -> None:
        days = AxisUnits(step_unit="day")

        with pytest.raises(ValueError) as index:
            days.window_start(-1, 7, "day")
        with pytest.raises(ValueError) as at:
            days.window_index(-1, 7, "day")

        assert str(index.value) == "index is a whole number of at least 0, got -1"
        assert str(at.value) == "at is a finite number of at least 0, got -1"


DAILY_DIFFERENCE = AxisUnits(step_unit="day", engine="DifferenceEquations")
DAILY_DIFFERENTIAL = AxisUnits(step_unit="day", engine="DifferentialEquations")


class TestRates:
    def test_a_rate_needs_an_engine(self) -> None:
        with pytest.raises(ValueError) as error:
            AxisUnits(step_unit="day").rate_from_mean_span(4, "day")

        assert str(error.value) == (
            "A rate needs the engine that reads it: give AxisUnits an engine, "
            "or use Model.axis_units()."
        )

    def test_an_unknown_engine_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown engine"):
            AxisUnits(engine="Stochastic")

    def test_the_engine_says_what_a_rate_means(self) -> None:
        assert DAILY_DIFFERENCE.rate_meaning == "per_step"
        assert DAILY_DIFFERENTIAL.rate_meaning == "continuous"

    def test_a_mean_span_is_converted_to_steps_first(self) -> None:
        assert DAILY_DIFFERENTIAL.rate_from_mean_span(2, "week") == pytest.approx(
            1 / 14
        )
        assert DAILY_DIFFERENCE.rate_from_mean_span(12, "hour") == pytest.approx(
            1 - math.exp(-2)
        )

    def test_a_per_step_rate_from_a_mean_span_has_a_longer_mean(self) -> None:
        rate = DAILY_DIFFERENCE.rate_from_mean_span(10, "day")

        assert 1 / rate == pytest.approx(10.508, abs=1e-3)

    @pytest.mark.parametrize("span", [0.0, -1.0])
    def test_a_non_positive_span_is_rejected(self, span: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            DAILY_DIFFERENCE.rate_from_mean_span(span, "day")

    @pytest.mark.parametrize("probability", [-0.1, 1.1])
    def test_a_probability_outside_the_unit_interval_is_rejected(
        self, probability: float
    ) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            DAILY_DIFFERENCE.rate_from_probability(probability, 7, "day")

    def test_a_continuous_rate_cannot_move_a_probability_of_one(self) -> None:
        assert DAILY_DIFFERENCE.rate_from_probability(1.0, 7, "day") == 1.0
        with pytest.raises(ValueError, match="probability of 1"):
            DAILY_DIFFERENTIAL.rate_from_probability(1.0, 7, "day")

    def test_a_non_positive_period_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DAILY_DIFFERENCE.rate_from_probability(0.3, 0, "day")


PROBABILITY = 0.3
PERIOD_DAYS = 7
MEAN_DAYS = 10


def _outflow(units: AxisUnits, rate: float):
    builder = a_to_b_builder(parameters={"k": rate})
    if units.engine == "DifferenceEquations":
        return build(builder)
    return build_ode(builder, TIGHT)


class TestRatesMoveTheBinInBothEngines:
    @pytest.mark.parametrize("units", [DAILY_DIFFERENCE, DAILY_DIFFERENTIAL])
    def test_a_rate_from_a_probability_moves_it_over_the_period(
        self, units: AxisUnits
    ) -> None:
        rate = units.rate_from_probability(PROBABILITY, 1, "week")

        results = Simulation(_outflow(units, rate)).run(PERIOD_DAYS)

        assert_close(results["A"][PERIOD_DAYS], 1000 * (1 - PROBABILITY), rel=1e-7)

    @pytest.mark.parametrize("units", [DAILY_DIFFERENCE, DAILY_DIFFERENTIAL])
    def test_a_rate_from_a_mean_span_keeps_the_exponential_survival(
        self, units: AxisUnits
    ) -> None:
        rate = units.rate_from_mean_span(MEAN_DAYS, "day")

        results = Simulation(_outflow(units, rate)).run(20)

        for step in range(21):
            assert_close(
                results["A"][step], 1000 * math.exp(-step / MEAN_DAYS), rel=1e-7
            )

    def test_a_continuous_rate_from_a_mean_span_has_that_mean(self) -> None:
        assert DAILY_DIFFERENTIAL.rate_from_mean_span(MEAN_DAYS, "day") == 1 / MEAN_DAYS
