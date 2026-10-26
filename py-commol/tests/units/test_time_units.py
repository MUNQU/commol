"""Units of the axis: one step unit for every rate, the unit of the axis, a
position on the axis in a rate, and the units of the variables of a rate."""

import pytest

from commol import Aggregate
from commol._model.engine_state import formula_tree
from commol._model.helpers import ModelCompartmentHelper
from commol._model.model import Model
from commol._model.unit_checks import UnitConsistencyError
from tests.support import (
    FactoryBuilder,
    a_to_b_builder,
    assert_accepted,
    build,
    build_ode,
)

FAILED = "Unit consistency check failed"


def _two_rates(
    k1_unit: str,
    k2_unit: str,
    bin_unit: str | None = "item",
    *,
    step_unit: str | None = None,
    ode: bool = False,
) -> Model:
    """A → B → C, each transition with its own rate parameter."""
    return (build_ode if ode else build)(
        FactoryBuilder(name="Step unit", bin_unit=bin_unit, step_unit=step_unit)
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_bin(id="C", name="C")
        .add_parameter(id="k1", value=0.1, unit=k1_unit)
        .add_parameter(id="k2", value=0.1, unit=k2_unit)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .set_structure(Aggregate(bin_amounts={"A": 1000, "B": 0, "C": 0}))
    )


class TestStepUnit:
    """One step unit, the declared ``step_unit`` or else ``step``, for every rate."""

    @pytest.mark.parametrize("unit", ["day", "week", "d"])
    def test_a_model_in_its_step_unit_passes(self, unit: str) -> None:
        model = _two_rates(f"1/{unit}", f"1/{unit}", step_unit=unit)

        assert_accepted(model.check_unit_consistency)

    def test_a_model_in_steps_needs_no_step_unit(self) -> None:
        assert_accepted(_two_rates("1/step", "1/step").check_unit_consistency)

    def test_aliases_of_one_unit_are_one_unit(self) -> None:
        model = _two_rates("1/day", "1/d", step_unit="day")

        assert_accepted(model.check_unit_consistency)

    def test_rates_per_day_without_a_step_unit_say_to_declare_it(self) -> None:
        with pytest.raises(UnitConsistencyError, match="Declare step_unit"):
            _two_rates("1/day", "1/day").check_unit_consistency()

    @pytest.mark.parametrize("unit", ["week", "quarter", "mo"])
    def test_a_rate_in_another_unit_is_named(self, unit: str) -> None:
        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            _two_rates("1/day", f"1/{unit}", step_unit="day").check_unit_consistency()

        message = str(error.value)
        assert "t_bc" in message
        assert unit in message
        assert "t_ab" not in message

    def test_a_model_without_units_of_time_uses_step(self) -> None:
        model = build(
            a_to_b_builder(
                parameters={"share": 0.1},
                rate="share",
                bin_unit="item",
                units={"share": "dimensionless"},
            )
        )

        with pytest.raises(UnitConsistencyError, match="item/step"):
            model.check_unit_consistency()

    def test_a_bin_without_a_unit_is_named(self) -> None:
        with pytest.raises(UnitConsistencyError, match="no unit: A, B, C"):
            _two_rates(
                "1/day", "1/day", bin_unit=None, step_unit="day"
            ).check_unit_consistency()


class TestAxisUnit:
    """A declared step unit of any single dimension: time is one case."""

    def test_a_model_in_one_per_meter_passes_with_its_step_unit(self) -> None:
        model = _two_rates("1/meter", "1/meter", step_unit="meter")

        assert_accepted(model.check_unit_consistency)

    def test_without_a_declared_unit_a_length_is_not_the_step(self) -> None:
        with pytest.raises(UnitConsistencyError, match="item/step"):
            _two_rates("1/meter", "1/meter").check_unit_consistency()

    def test_one_per_meter_mixed_with_one_per_day_names_both(self) -> None:
        with pytest.raises(UnitConsistencyError) as error:
            _two_rates("1/meter", "1/day", step_unit="meter").check_unit_consistency()

        message = str(error.value)
        assert "item / day" in message or "item/day" in message
        assert "item/meter" in message

    @pytest.mark.parametrize("unit", ["1/day", "meter**2"])
    def test_the_step_unit_is_one_unit_of_one_dimension(self, unit: str) -> None:
        with pytest.raises(UnitConsistencyError, match="step unit"):
            _two_rates("1/day", "1/day", step_unit=unit).check_unit_consistency()

    def test_a_step_unit_that_is_no_unit_says_why(self) -> None:
        with pytest.raises(UnitConsistencyError) as error:
            _two_rates("1/day", "1/day", step_unit="day**").check_unit_consistency()

        assert (
            str(error.value) == "The step unit 'day**' is not a unit: missing operand"
        )

    def test_a_step_unit_pint_does_not_define_is_its_own_dimension(self) -> None:
        model = _two_rates("1/generation", "1/generation", step_unit="generation")

        assert_accepted(model.check_unit_consistency)

    def test_a_differential_model_in_one_per_day_passes(self) -> None:
        model = _two_rates("1/day", "1/day", step_unit="day", ode=True)

        assert_accepted(model.check_unit_consistency)

    def test_an_impulse_moves_an_amount_of_the_bin_unit(self) -> None:
        def impulse(unit: str) -> Model:
            return build(
                FactoryBuilder(name="impulse", bin_unit="item")
                .add_bin(id="A", name="A")
                .add_bin(id="B", name="B")
                .add_parameter(id="p", value=0.1, unit=unit)
                .add_transition(
                    id="t_ab", source=["A"], target=["B"], rate="p * A", impulse_at=[1]
                )
                .set_structure(Aggregate(bin_amounts={"A": 1000, "B": 0}))
            )

        impulse("dimensionless").check_unit_consistency()
        with pytest.raises(UnitConsistencyError, match="expected 'item'"):
            impulse("1/day").check_unit_consistency()


class TestPositionOnTheAxis:
    """``t`` is a position in the step unit; ``step`` counts increments."""

    def _proportional_to_t(self, k_unit: str) -> Model:
        return build(
            a_to_b_builder(
                parameters={"k": 0.001},
                rate="k * t * A",
                bin_unit="item",
                units={"k": k_unit},
                step_unit="day",
            )
        )

    def test_a_rate_proportional_to_t_needs_k_per_step_unit_squared(self) -> None:
        assert_accepted(self._proportional_to_t("1/day**2").check_unit_consistency)

    def test_a_rate_proportional_to_t_with_k_per_day_fails(self) -> None:
        with pytest.raises(UnitConsistencyError, match="k \\* t"):
            self._proportional_to_t("1/day").check_unit_consistency()

    def _windowed(self, rate: str) -> Model:
        return build(
            a_to_b_builder(
                parameters={"k": 0.1, "cap": 5},
                rate=rate,
                bin_unit="item",
                units={"k": "1/day", "cap": "item"},
                step_unit="day",
            )
        )

    def test_a_number_compared_with_t_is_read_in_the_step_unit(self) -> None:
        model = self._windowed("if(t >= 10, if(t < 20.5, k, 0), 0) * A")

        assert_accepted(model.check_unit_consistency)

    def test_a_number_added_to_t_is_read_in_the_step_unit(self) -> None:
        assert_accepted(self._windowed("k * (t - 3) * k * A").check_unit_consistency)

    def test_the_branches_of_if_share_a_unit(self) -> None:
        with pytest.raises(UnitConsistencyError, match=FAILED):
            self._windowed("if(t >= 10, k, cap) * A").check_unit_consistency()

    def test_a_computed_number_is_not_read_in_a_unit(self) -> None:
        with pytest.raises(UnitConsistencyError, match=FAILED):
            self._windowed("(k + exp(2)) * A").check_unit_consistency()


class TestVariableUnits:
    """The unit of every variable a rate may read."""

    @staticmethod
    def _model(c_unit: str) -> Model:
        return build(
            FactoryBuilder(name="units", step_unit="day")
            .add_bin(id="A", name="A", unit="item")
            .add_bin(id="C", name="C", unit=c_unit)
            .add_stratification(id="group", categories=["g1", "g2"])
            .add_parameter(id="k", value=0.1, unit="1/day")
            .add_parameter(id="g", value="f / k")
            .add_parameter(id="f", value="k * 2")
            .add_parameter(id="z", value="2")
            .add_transition(id="t", source=["g1"], target=["g2"], rate="f")
            .set_structure(
                Aggregate(
                    bin_amounts={"A": 500, "C": 500},
                    stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
                )
            )
        )

    def test_every_variable_has_its_unit(self) -> None:
        units = ModelCompartmentHelper(self._model("item")).build_variable_units()

        assert units == {
            "k": "1/day",
            "f": "1 / day",
            "g": "dimensionless",
            "z": "dimensionless",
            "A": "item",
            "C": "item",
            "A_g1": "item",
            "A_g2": "item",
            "C_g1": "item",
            "C_g2": "item",
            "total": "item",
            "total_g1": "item",
            "total_g2": "item",
            "neighbors": "item",
            "neighbors_A": "item",
            "neighbors_C": "item",
            "step": "dimensionless",
            "t": "day",
            "pi": "dimensionless",
            "e": "dimensionless",
        }

    def test_a_compartment_and_its_neighbors_have_the_unit_of_their_bin(
        self,
    ) -> None:
        units = ModelCompartmentHelper(self._model("kg")).build_variable_units()

        assert {name: units[name] for name in ["C", "C_g1", "C_g2", "neighbors_C"]} == {
            "C": "kg",
            "C_g1": "kg",
            "C_g2": "kg",
            "neighbors_C": "kg",
        }

    def test_a_total_over_bins_of_different_units_has_no_unit(self) -> None:
        model = self._model("kg")
        helper = ModelCompartmentHelper(model)

        units = helper.build_variable_units()

        assert helper.mixed_unit_totals() == {
            "total",
            "total_g1",
            "total_g2",
            "neighbors",
        }
        assert not helper.mixed_unit_totals() & set(units)
        assert ModelCompartmentHelper(self._model("item")).mixed_unit_totals() == set()

    @pytest.mark.parametrize("total", ["total", "total_g1"])
    def test_a_rate_that_adds_bins_of_different_units_is_rejected(
        self, total: str
    ) -> None:
        model = build(
            FactoryBuilder(name="units", step_unit="day")
            .add_bin(id="A", name="A", unit="item")
            .add_bin(id="C", name="C", unit="kg")
            .add_stratification(id="group", categories=["g1", "g2"])
            .add_parameter(id="k", value=0.1, unit="1/day")
            .add_transition(
                id="t", source=["A"], target=[], rate=f"k * A * {total} / {total}"
            )
            .set_structure(
                Aggregate(
                    bin_amounts={"A": 500, "C": 500},
                    stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
                )
            )
        )

        with pytest.raises(UnitConsistencyError) as error:
            model.check_unit_consistency()

        assert (
            f"Variable '{total}' in rate 'k * A * {total} / {total}' adds bins of "
            "different units (item, kg); read the bins or the compartments it "
            "sums instead"
        ) in str(error.value)

    def test_a_unit_named_like_a_variable_is_not_annotated_again(self) -> None:
        helper = ModelCompartmentHelper(self._model("item"))

        annotated = helper.annotate_rate_variables(
            formula_tree("k * A_g1 / g1"), {"k": "1/day", "A_g1": "g1", "g1": "item"}
        )

        assert annotated == "k(1/day) * A_g1(g1) / g1(item)"

    def test_a_rate_with_a_variable_without_a_unit_has_no_unit(self) -> None:
        helper = ModelCompartmentHelper(self._model("item"))
        units = {"k": "1/day"}

        rate = formula_tree("k * x")

        assert helper.format_rate_with_unit(rate, units) == "k(1/day) * x"
        assert helper.format_rate_with_unit(rate, units, latex=True) == (
            "k(\\text{1/day}) \\cdot x"
        )

    def test_a_formula_whose_unit_fails_says_why(self) -> None:
        model = build(
            FactoryBuilder(name="units", bin_unit="item")
            .add_bin(id="A", name="A")
            .add_parameter(id="k", value=0.1, unit="1/step")
            .add_parameter(id="f", value="exp(k)")
            .add_transition(id="t", source=["A"], target=[], rate="k")
            .set_structure(Aggregate(bin_amounts={"A": 100}))
        )

        with pytest.raises(UnitConsistencyError) as error:
            ModelCompartmentHelper(model).build_variable_units()

        assert str(error.value) == (
            "Cannot infer unit for formula parameter 'f' = 'exp(k)': Failed to "
            "evaluate equation 'exp(k)': Cannot convert from '1 / step' to "
            "'dimensionless'. Please provide an explicit unit for this parameter."
        )
