"""Parameters: numbers, formulas, series, and values left for calibration."""

import math

import pytest

from commol import Simulation
from commol.context.parameter import Parameter
from tests.support import (
    ENGINE_REL,
    a_to_b_builder,
    assert_close,
    build,
    build_ode,
    sir_builder,
)


class TestFormulaParameters:
    def test_a_formula_of_other_parameters_runs_as_its_value(self) -> None:
        model = build(
            sir_builder(
                parameters={
                    "beta_base": 0.3,
                    "contact_multiplier": 2.0,
                    "beta": "beta_base * contact_multiplier",
                    "gamma": 0.1,
                }
            )
        )
        same_as_number = build(sir_builder(parameters={"beta": 0.6, "gamma": 0.1}))

        beta = next(p for p in model.parameters if p.id == "beta")
        assert beta.value == "beta_base * contact_multiplier"
        assert_close(
            Simulation(model).run(20),
            Simulation(same_as_number).run(20),
            rel=ENGINE_REL,
        )

    def test_a_formula_may_use_subpopulation_totals(self) -> None:
        model = build(
            sir_builder(
                parameters={
                    "young_fraction": "N_young / N",
                    "beta": 0.3,
                    "gamma": 0.1,
                },
                infection="beta * young_fraction * S * I / N",
                stratifications={"age": {"young": 0.3, "old": 0.7}},
            )
        )
        same_as_number = build(
            sir_builder(
                parameters={"beta": 0.3 * 0.3, "gamma": 0.1},
                stratifications={"age": {"young": 0.3, "old": 0.7}},
            )
        )

        # N_young / N stays 0.3: no transition moves people between ages.
        assert_close(
            Simulation(model).run(20),
            Simulation(same_as_number).run(20),
            rel=ENGINE_REL,
        )


class TestUncalibratedParameters:
    def test_a_parameter_may_have_no_value(self) -> None:
        parameter = Parameter(id="beta", value=None)

        assert parameter.value is None
        assert not parameter.is_calibrated()

    def test_a_parameter_with_a_value_is_calibrated(self) -> None:
        assert Parameter(id="gamma", value=0.05).is_calibrated()

    def test_the_model_lists_the_parameters_without_a_value(self) -> None:
        model = build(
            sir_builder(
                parameters={"beta": None, "gamma": 0.05, "delta": None},
            )
        )

        assert model.get_uncalibrated_parameters() == ["beta", "delta"]

    def test_update_parameters_sets_the_values(self) -> None:
        model = build(sir_builder(parameters={"beta": None, "gamma": None}))

        model.update_parameters({"beta": 0.1, "gamma": 0.05})

        assert [p.value for p in model.parameters] == [0.1, 0.05]
        assert model.get_uncalibrated_parameters() == []

    def test_update_parameters_rejects_an_unknown_id(self) -> None:
        model = build(sir_builder(parameters={"beta": None, "gamma": None}))

        with pytest.raises(ValueError, match="invalid_param") as error:
            model.update_parameters({"invalid_param": 0.1})

        assert "not found" in str(error.value).lower()


class TestSeriesParameters:
    def test_a_difference_model_reads_a_linear_series_at_each_step(self) -> None:
        # k(n) = n / 10, so one step moves k(n) A(n).
        builder = a_to_b_builder(parameters={}, rate="k")
        builder.add_parameter("k", [(0, 0.0), (10, 1.0)], mode="linear")

        a = Simulation(build(builder)).run(4)["A"]

        expected = [1000.0]
        for n in range(4):
            expected.append(expected[-1] * (1 - n / 10))
        assert_close(a, expected, rel=ENGINE_REL)

    def test_a_difference_engine_names_a_point_between_steps(self) -> None:
        builder = a_to_b_builder(parameters={}, rate="k")
        builder.add_parameter("k", [(2.5, 0.1)], mode="step_function")

        with pytest.raises(RuntimeError, match="'k' has a point at 2.5"):
            Simulation(build(builder))

    def test_a_linear_series_is_a_ramp_in_a_differential_model(self) -> None:
        # k(t) = t / 10 on [0, 10]: A(10) = 1000 exp(-5), not the staircase
        # 1000 exp(-4.5) of a series read at the step.
        builder = a_to_b_builder(parameters={}, rate="k")
        builder.add_parameter("k", [(0, 0.0), (10, 1.0)], mode="linear")

        a = Simulation(build_ode(builder)).run(10)["A"]

        assert_close(a[10], 1000 * math.exp(-5), rel=1e-7)

    def test_a_pulse_between_steps_names_its_parameter(self) -> None:
        builder = a_to_b_builder(parameters={}, rate="k")
        builder.add_parameter("k", [(1.5, 0.1)], mode="pulse")

        with pytest.raises(RuntimeError, match="'k' is a pulse at 1.5"):
            Simulation(build_ode(builder))
