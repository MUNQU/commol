"""Parameters: numbers, formulas, series, and values left for calibration."""

import math
import warnings

import pytest

from pydantic import ValidationError

from commol import ModelBuilder, SeriesValue, Simulation
from commol.context.engine import EngineName
from commol.context.parameter import Parameter
from tests.support import (
    ENGINE_REL,
    a_to_b_builder,
    assert_close,
    build,
    build_ode,
    abc_builder,
)


class TestFormulaParameters:
    def test_a_formula_of_other_parameters_runs_as_its_value(self) -> None:
        model = build(
            abc_builder(
                parameters={
                    "k1_base": 0.3,
                    "multiplier": 2.0,
                    "k1": "k1_base * multiplier",
                    "k2": 0.1,
                }
            )
        )
        same_as_number = build(abc_builder(parameters={"k1": 0.6, "k2": 0.1}))

        k1 = next(p for p in model.parameters if p.id == "k1")
        assert k1.value == "k1_base * multiplier"
        assert_close(
            Simulation(model).run(20),
            Simulation(same_as_number).run(20),
            rel=ENGINE_REL,
        )

    def test_a_formula_may_use_subgroup_totals(self) -> None:
        model = build(
            abc_builder(
                parameters={
                    "g1_fraction": "total_g1 / total",
                    "k1": 0.3,
                    "k2": 0.1,
                },
                ab="k1 * g1_fraction * A * B / total",
                stratifications={"group": {"g1": 0.3, "g2": 0.7}},
            )
        )
        same_as_number = build(
            abc_builder(
                parameters={"k1": 0.3 * 0.3, "k2": 0.1},
                stratifications={"group": {"g1": 0.3, "g2": 0.7}},
            )
        )

        # total_g1 / total stays 0.3: no transition moves amounts between groups.
        assert_close(
            Simulation(model).run(20),
            Simulation(same_as_number).run(20),
            rel=ENGINE_REL,
        )


class TestUncalibratedParameters:
    def test_a_parameter_may_have_no_value(self) -> None:
        parameter = Parameter(id="k1", value=None)

        assert parameter.value is None
        assert not parameter.is_calibrated()

    def test_a_parameter_with_a_value_is_calibrated(self) -> None:
        assert Parameter(id="k2", value=0.05).is_calibrated()

    def test_the_model_lists_the_parameters_without_a_value(self) -> None:
        model = build(
            abc_builder(
                parameters={"k1": None, "k2": 0.05, "delta": None},
            )
        )

        assert model.get_uncalibrated_parameters() == ["k1", "delta"]

    def test_update_parameters_sets_the_values(self) -> None:
        model = build(abc_builder(parameters={"k1": None, "k2": None}))

        model.update_parameters({"k1": 0.1, "k2": 0.05})

        assert [p.value for p in model.parameters] == [0.1, 0.05]
        assert model.get_uncalibrated_parameters() == []

    def test_update_parameters_rejects_an_unknown_id(self) -> None:
        model = build(abc_builder(parameters={"k1": None, "k2": None}))

        with pytest.raises(ValueError, match="invalid_param") as error:
            model.update_parameters({"invalid_param": 0.1})

        assert "not found" in str(error.value).lower()


@pytest.mark.parametrize("engine", ["DifferenceEquations", "StochasticSteps"])
def test_a_point_the_engine_cannot_run_is_rejected_at_build_and_update(
    engine: EngineName,
) -> None:
    between = SeriesValue(data=[(0, 0.1), (1.5, 0.2)], mode="step_function")
    point = (
        r"Series parameter 'k' has a point at 1\.5, which is not an integer step; "
        f"the {engine} engine steps only at integers"
    )
    with_point = a_to_b_builder(parameters={}, rate="k")
    with_point.add_parameter("k", between.data, mode=between.mode)
    with_impulse = a_to_b_builder(parameters={"k": 0.1})
    with_impulse.add_transition(
        id="jump", source=["A"], target=["B"], rate="0.5", impulse_at=[1.5]
    )

    with pytest.raises(ValueError, match=point):
        with_point.build(with_point.structure, engine=engine)
    with pytest.raises(ValueError, match=r"Impulse transition 'jump' acts at 1\.5"):
        with_impulse.build(with_impulse.structure, engine=engine)

    valid = a_to_b_builder(parameters={"k": 0.1})
    model = valid.build(valid.structure, engine=engine)
    saved = model.model_dump()
    with pytest.raises(ValueError, match=point):
        model.update_parameters({"k": between})
    assert model.model_dump() == saved
    Simulation(model)

    ode = build_ode(a_to_b_builder(parameters={"k": 0.1}))
    ode.update_parameters({"k": between})
    assert Simulation(ode).run_at([1.5])["A"][0] < 1000


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

        with pytest.raises(ValueError, match="'k' is a pulse at 1.5"):
            build_ode(builder)

    def test_update_parameters_replaces_a_series(self) -> None:
        # A pulse moves its value times A on the step of its point only; a
        # step function moves it on every step from its point on.
        builder = a_to_b_builder(parameters={}, rate="d")
        builder.add_parameter("d", [(1, 0.5)], mode="pulse")
        model = build(builder)

        model.update_parameters({"d": [(2, 0.25)]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pulse = Simulation(model).run(4)["B"]
        model.update_parameters(
            {"d": SeriesValue(data=[(3, 0.5)], mode="step_function")}
        )
        model.update_parameters({"d": [(2, 0.25)]})
        step_function = Simulation(model).run(4)["B"]

        assert_close(pulse, [0, 0, 0, 250, 250], rel=ENGINE_REL)
        assert_close(step_function, [0, 0, 0, 250, 437.5], rel=ENGINE_REL)

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({"k": [(1, 0.5)]}, "Parameter 'k' is not a series"),
            ({"d": []}, "Parameter 'd' cannot take .* at least one point"),
            (
                {"k": 0.3, "d": [(-1, 0.5)]},
                "Parameter 'd' cannot take .*series position is a finite number",
            ),
            ({"k": "  "}, "Parameter 'k' cannot take .* holds nothing"),
            ({"d": [1.0]}, "Parameter 'd' cannot take .* valid tuple"),
            ({"k": "total_g1"}, "Unknown subgroup total.*'total_g1'"),
        ],
    )
    def test_update_parameters_rejects_a_value_it_cannot_take(
        self,
        values: dict[str, float | str | list[tuple[float, float]] | SeriesValue | None],
        message: str,
    ) -> None:
        builder = a_to_b_builder(parameters={"k": 0.1}, rate="k * d")
        builder.add_parameter("d", [(1, 1.0)], mode="pulse")
        model = build(builder)
        saved = model.model_dump()

        with pytest.raises(ValueError, match=message):
            model.update_parameters(values)

        assert model.model_dump() == saved


class TestValues:
    def test_a_formula_is_not_blank(self) -> None:
        with pytest.raises(ValidationError, match="Parameter 'x': .* holds nothing"):
            ModelBuilder("M").add_parameter("x", "   ")

    def test_a_series_has_a_mode_and_only_a_series_has_one(self) -> None:
        with pytest.raises(ValueError, match="'x': 'mode' is required"):
            ModelBuilder("M").add_parameter("x", [(0, 1.0)])
        with pytest.raises(ValueError, match="'x': 'mode' is only valid"):
            ModelBuilder("M").add_parameter("x", 1.0, mode="pulse")

    def test_a_run_names_the_initial_conditions_left_to_calibrate(self) -> None:
        model = build(a_to_b_builder(amounts={"A": 500, "B": None}))

        with pytest.raises(
            ValueError, match="Initial conditions requiring calibration: B"
        ):
            Simulation(model).run(2)
