"""A DifferentialEquations model: the ODE engine, ``run_at`` and the rule of ``t``."""

import math

import pytest
from pydantic import ValidationError

from commol import AxisPattern, Simulation
from commol._model.model import Model
from commol.engines import ODESolver
from tests.support import (
    a_to_b_builder,
    abc_builder,
    assert_close,
    build,
    build_ode,
)

K = 0.1
"""Proportional rate of the default A → B model."""


def _decay(**kwargs) -> Model:
    return build_ode(a_to_b_builder(accumulator="moved", **kwargs))


class TestRun:
    def test_exponential_decay_matches_the_closed_form(self) -> None:
        results = Simulation(_decay()).run(10)

        for step, value in enumerate(results["A"]):
            assert_close(value, 1000 * math.exp(-K * step), rel=1e-7)

    def test_an_accumulator_is_the_integral_of_its_flow(self) -> None:
        results = Simulation(_decay()).run(10)

        assert_close(results["moved"][10], 1000 * (1 - math.exp(-K * 10)), rel=1e-7)

    def test_both_methods_agree(self) -> None:
        tsit45 = Simulation(_decay()).run(20)
        bdf = Simulation(
            build_ode(a_to_b_builder(), ODESolver(method="bdf", rtol=1e-10, atol=1e-8))
        ).run(20)

        assert_close(bdf["A"][20], tsit45["A"][20], rel=1e-6)

    def test_the_sir_final_size_relation_holds(self) -> None:
        results = Simulation(build_ode(abc_builder())).run(200)

        for step in range(0, 201, 20):
            s, r = results["A"][step], results["C"][step]
            assert math.log(s / 990) + 3.0 * r / 1000 == pytest.approx(0, abs=1e-7)

    def test_the_default_solver_is_used_without_settings(self) -> None:
        results = Simulation(build_ode(a_to_b_builder(), settings=None)).run(5)

        assert_close(results["A"][5], 1000 * math.exp(-K * 5), rel=1e-5)


class TestRunAt:
    def test_any_positions_match_the_closed_form(self) -> None:
        results = Simulation(_decay()).run_at([0, 0.5, 2.25])

        for position, value in zip([0, 0.5, 2.25], results["A"], strict=True):
            assert_close(value, 1000 * math.exp(-K * position), rel=1e-7)

    def test_a_window_is_the_integral_of_the_flow(self) -> None:
        results = Simulation(_decay()).run_at([0.5, 3.5])

        expected = 1000 * (math.exp(-K * 0.5) - math.exp(-K * 3.5))
        assert_close(results["moved"][1] - results["moved"][0], expected, rel=1e-7)

    def test_results_hold_the_requested_positions(self) -> None:
        results = Simulation(_decay()).run_at([1, 2.5])

        assert results.positions == [1.0, 2.5]
        rows = results.rows()
        assert len(rows) == 2 and len(rows[0]) == 3

    def test_a_difference_model_rejects_a_non_integer_step(self) -> None:
        simulation = Simulation(build(a_to_b_builder()))

        with pytest.raises(ValueError, match="0.5 is not one"):
            simulation.run_at([0.5])

    def test_a_difference_model_reports_at_integer_steps(self) -> None:
        simulation = Simulation(build(a_to_b_builder()))

        at = simulation.run_at([0, 3])
        full = simulation.run(3)

        assert at["A"] == [full["A"][0], full["A"][3]]

    def test_unsorted_steps_are_an_error(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            Simulation(_decay()).run_at([2, 1])


class TestTimeInsideAStep:
    def test_a_pulse_is_a_constant_rate_on_its_unit_interval(self) -> None:
        pulse = AxisPattern.pulse(at=2, value=0.2)
        ode = Simulation(build_ode(a_to_b_builder(rate=pulse))).run(4)
        difference = Simulation(build(a_to_b_builder(rate=pulse))).run(4)

        assert_close(ode["A"][2] - ode["A"][3], 1000 * (1 - math.exp(-0.2)), rel=1e-8)
        assert_close(difference["A"][2] - difference["A"][3], 200.0, rel=1e-12)

    def test_t_is_continuous_and_step_is_its_floor(self) -> None:
        # dA/dt = -t A: A(t) = 1000 exp(-t^2 / 2). With `step` the rate is n on
        # [n, n + 1), so A(2.5) = 1000 exp(-(0 + 1 + 2 * 0.5)).
        continuous = Simulation(
            build_ode(a_to_b_builder(rate="t", parameters={}))
        ).run_at([1.5])
        stepwise = Simulation(
            build_ode(a_to_b_builder(rate="step", parameters={}))
        ).run_at([2.5])

        assert_close(continuous["A"][0], 1000 * math.exp(-(1.5**2) / 2), rel=1e-8)
        assert_close(stepwise["A"][0], 1000 * math.exp(-2.0), rel=1e-8)

    @pytest.mark.parametrize(
        ("start", "end"), [("12.5", "14"), ("4 * pi", "4 * pi + 1.5")]
    )
    def test_a_window_on_t_is_not_stepped_over(self, start: str, end: str) -> None:
        # The rate is 0 before the window, so only a restart at its start keeps
        # the solver from stepping over it. A keeps exp(-0.5 * 1.5).
        rate = f"if(t >= {start}, if(t < {end}, 0.5, 0), 0)"
        model = build_ode(a_to_b_builder(rate=rate, parameters={}))
        results = Simulation(model).run(40)

        assert_close(results["A"][40], 1000 * math.exp(-0.75), rel=1e-7)


class TestSolverSettings:
    @pytest.mark.parametrize("field", ["rtol", "atol"])
    def test_a_tolerance_must_be_positive(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ODESolver(**{field: 0.0})
