"""``Model.check_unit_consistency``: every flow in ``<bin unit>/<step unit>``."""

import pytest

from commol import ModelBuilder
from commol.context.model import Model
from commol.utils.equations import UnitConsistencyError
from tests.support import a_to_b_builder, build, build_ode, abcd_builder, abc_builder

DAILY = {"k1": "1/day", "k2": "1/day"}
FAILED = "Unit consistency check failed"


def _abc(**options) -> Model:
    """``abc_builder`` in items with rates per day; an option replaces a default."""
    options.setdefault("units", DAILY)
    options.setdefault("step_unit", "day")
    return build(abc_builder(bin_unit="item", **options))


def _per_group_ab(*rates: str, **options) -> Model:
    """``abc_builder`` whose ``ab`` has one rate per category of ``group``, in order."""
    categories = ("g1", "g2", "g3")[: len(rates)]
    builder = abc_builder(
        bin_unit="item",
        units=DAILY,
        step_unit="day",
        ab=None,
        stratifications={"group": dict.fromkeys(categories, 1 / len(rates))},
        **options,
    )
    return build(
        builder.add_transition(
            id="ab",
            source=["A"],
            target=["B"],
            stratified_rates=[
                {
                    "conditions": [{"stratification": "group", "category": category}],
                    "rate": rate,
                }
                for category, rate in zip(categories, rates, strict=True)
            ],
        )
    )


def _with_v(rate: str, threshold_unit: str = "item") -> Model:
    """A, V, B, C: A moves to V at ``rate`` besides the flows of ``abc_builder``."""
    return build(
        ModelBuilder(name="AVBC", bin_unit="item", step_unit="day")
        .add_bin(id="A", name="A")
        .add_bin(id="V", name="V")
        .add_bin(id="B", name="B")
        .add_bin(id="C", name="C")
        .add_parameter(id="k1", value=0.5, unit="1/day")
        .add_parameter(id="k2", value=0.1, unit="1/day")
        .add_parameter(id="k_v", value=0.01, unit="1/day")
        .add_parameter(id="threshold", value=100, unit=threshold_unit)
        .add_transition(id="av", source=["A"], target=["V"], rate=rate)
        .add_transition(id="ab", source=["A"], target=["B"], rate="k1 * A * B / total")
        .add_transition(id="bc", source=["B"], target=["C"], rate="k2 * B")
        .set_initial_conditions(
            total=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 0.98},
                {"bin": "V", "fraction": 0.0},
                {"bin": "B", "fraction": 0.02},
                {"bin": "C", "fraction": 0.0},
            ],
        )
    )


class TestConsistentModels:
    def test_abc(self) -> None:
        _abc().check_unit_consistency()

    def test_rates_divided_by_subgroup_totals(self) -> None:
        _per_group_ab(
            "k1 * A * B / total_g1", "k1 * A * B / total_g2"
        ).check_unit_consistency()

    def test_a_dimensionless_parameter(self) -> None:
        _abc(
            parameters={"factor": 0.5, "k1": 0.5, "k2": 0.1},
            units={**DAILY, "factor": "dimensionless"},
            ab="k1 * factor * A * B / total",
        ).check_unit_consistency()

    def test_a_unit_pint_does_not_define_is_its_own_dimension(self) -> None:
        build(
            abc_builder(
                bin_unit="widget",
                parameters={"k1": 0.1},
                units={"k1": "1/generation"},
                step_unit="generation",
                bc=None,
            )
        ).check_unit_consistency()

    def test_a_with_v_rate(self) -> None:
        _with_v("k_v * A", threshold_unit="dimensionless").check_unit_consistency()

    def test_exponential_decay_of_a_rate(self) -> None:
        _abc(
            parameters={"k1_0": 0.5, "decay_rate": 0.01, "k2": 0.1},
            units={"k1_0": "1/day", "decay_rate": "dimensionless", "k2": "1/day"},
            ab="k1_0 * exp(-decay_rate * step) * A * B / total",
        ).check_unit_consistency()

    def test_sin_of_step_forcing(self) -> None:
        _abc(
            parameters={"k1_avg": 0.5, "amplitude": 0.1, "k2": 0.1},
            units={
                "k1_avg": "1/day",
                "amplitude": "dimensionless",
                "k2": "1/day",
            },
            ab=("k1_avg * (1 + amplitude * sin(2 * pi * step / 365)) * A * B / total"),
        ).check_unit_consistency()

    def test_the_unit_of_a_formula_parameter_is_inferred(self) -> None:
        _abc(
            parameters={
                "k1_base": 0.3,
                "multiplier": 2.0,
                "k1": "k1_base * multiplier",
                "k2": 0.1,
            },
            units={
                "k1_base": "1/day",
                "multiplier": "dimensionless",
                "k2": "1/day",
            },
        ).check_unit_consistency()

    def test_a_formula_of_subgroup_totals_is_dimensionless(self) -> None:
        _abc(
            parameters={
                "g1_fraction": "total_g1 / total",
                "k1": 0.3,
                "k2": 0.1,
            },
            stratifications={"group": {"g1": 0.3, "g2": 0.7}},
        ).check_unit_consistency()

    def test_chained_formula_parameters(self) -> None:
        _abc(
            parameters={
                "base_rate": 0.1,
                "adjusted_rate": "base_rate * 2",
                "final_rate": "adjusted_rate * 3",
                "k2": 0.1,
            },
            units={"base_rate": "1/day", "k2": "1/day"},
            ab="final_rate * A * B / total",
        ).check_unit_consistency()


class TestMissingUnits:
    def test_parameters_without_units_are_named(self) -> None:
        with pytest.raises(UnitConsistencyError, match="missing units"):
            _abc(units={}).check_unit_consistency()

    def test_one_parameter_without_a_unit_is_named(self) -> None:
        with pytest.raises(UnitConsistencyError, match="missing units") as error:
            _abc(units={"k1": "1/day"}).check_unit_consistency()

        assert "k2" in str(error.value)

    def test_a_formula_of_an_unknown_variable_has_no_unit(self) -> None:
        model = _abc(
            parameters={
                "k1_base": 0.3,
                "multiplier": 2.0,
                "invalid_param": "nonexistent_var * 2",
                "k2": 0.1,
            },
            units={
                "k1_base": "1/day",
                "multiplier": "dimensionless",
                "k2": "1/day",
            },
            ab="invalid_param * A * B / total",
        )

        with pytest.raises(UnitConsistencyError, match="no defined unit") as error:
            model.check_unit_consistency()

        assert "invalid_param" in str(error.value)


class TestInconsistentModels:
    def test_a_rate_in_the_wrong_unit_names_its_transition(self) -> None:
        model = _abc(units={"k1": "day", "k2": "1/day"})

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "ab" in str(error.value)

    @pytest.mark.parametrize(
        ("rate", "extra"),
        [
            pytest.param("k1 * log(B) * A / total", {}, id="log-of-an-amount"),
            pytest.param(
                "k1 * (1 + 0.1 * sin(time_scale)) * A * B / total",
                {"time_scale": (365, "day")},
                id="sin-of-days",
            ),
            pytest.param(
                "k1 * (1 + 0.1 * atan2(B, threshold_time)) * A * B / total",
                {"threshold_time": (30, "day")},
                id="atan2-of-items-and-days",
            ),
            pytest.param(
                "min(k1, capacity) * A * B / total",
                {"capacity": (50, "item")},
                id="min-of-a-rate-and-items",
            ),
        ],
    )
    def test_a_function_of_a_wrong_unit_fails(
        self, rate: str, extra: dict[str, tuple[float, str]]
    ) -> None:
        model = _abc(
            parameters={"k1": 0.5, "k2": 0.1}
            | {name: value for name, (value, _) in extra.items()},
            units=DAILY | {name: unit for name, (_, unit) in extra.items()},
            ab=rate,
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert rate in str(error.value)

    def test_exp_of_days_fails(self) -> None:
        model = build(
            abcd_builder(
                bin_unit="item",
                step_unit="day",
                parameters={"k1": 0.5, "k3": 0.2, "k2": 0.1, "decay_time": 10},
                units={**DAILY, "k3": "1/day", "decay_time": "day"},
                ab="k1 * exp(decay_time) * A * C / total",
            )
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "exp(decay_time)" in str(error.value)

    def test_sin_of_steps_over_days_fails(self) -> None:
        model = build(
            abcd_builder(
                bin_unit="item",
                step_unit="day",
                parameters={
                    "k1_avg": 0.5,
                    "amplitude": 0.2,
                    "period": 365,
                    "k3": 0.2,
                    "k2": 0.1,
                },
                units={
                    "k1_avg": "1/day",
                    "amplitude": "dimensionless",
                    "period": "day",
                    "k3": "1/day",
                    "k2": "1/day",
                },
                ab=(
                    "k1_avg * (1 + amplitude * sin(2 * pi * step / period)) "
                    "* A * C / total"
                ),
            )
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "sin(2 * pi * step / period)" in str(error.value)

    def test_sqrt_of_an_amount_in_a_stratified_rate_fails(self) -> None:
        model = _per_group_ab(
            "k1 * sqrt(B) * A / total_g1", "k1 * 0.5 * sqrt(B) * A / total_g2"
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "sqrt(B)" in str(error.value)

    def test_floor_of_an_amount_in_a_stratified_rate_fails(self) -> None:
        model = _per_group_ab(
            "k1 * floor(A) * B / total_g1",
            "k1 * A * B / total_g2",
            "k1 * 0.6 * A * B / total_g3",
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "floor(A)" in str(error.value)

    def test_log_of_items_in_one_of_four_combinations_fails(self) -> None:
        builder = abc_builder(
            bin_unit="item",
            parameters={"k1": 0.5, "k2": 0.1, "scale": 50},
            units={**DAILY, "scale": "item"},
            ab=None,
            stratifications={
                "group": {"g1": 0.6, "g2": 0.4},
                "type": {"t1": 0.7, "t2": 0.3},
            },
        )
        rates = {
            (
                "g1",
                "t1",
            ): "k1 * exp(log(scale)) * A * B / total_g1_t1",
            ("g1", "t2"): "k1 * 0.5 * A * B / total_g1_t2",
            ("g2", "t1"): "k1 * 0.7 * A * B / total_g2_t1",
            ("g2", "t2"): "k1 * 0.3 * A * B / total_g2_t2",
        }
        model = build(
            builder.add_transition(
                id="ab",
                source=["A"],
                target=["B"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "group", "category": group},
                            {"stratification": "type", "category": kind},
                        ],
                        "rate": rate,
                    }
                    for (group, kind), rate in rates.items()
                ],
            )
        )

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "exp(log(scale))" in str(error.value)

    def test_pow_with_an_exponent_in_items_fails(self) -> None:
        model = _with_v("k_v * pow(B / total, threshold) * A")

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "pow(B / total, threshold)" in str(error.value)


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
        ModelBuilder(name="Step unit", bin_unit=bin_unit, step_unit=step_unit)
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_bin(id="C", name="C")
        .add_parameter(id="k1", value=0.1, unit=k1_unit)
        .add_parameter(id="k2", value=0.1, unit=k2_unit)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .set_initial_conditions(
            total=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 1.0},
                {"bin": "B", "fraction": 0.0},
                {"bin": "C", "fraction": 0.0},
            ],
        )
    )


class TestStepUnit:
    """One step unit, the declared ``step_unit`` or else ``step``, for every rate."""

    @pytest.mark.parametrize("unit", ["day", "week", "d"])
    def test_a_model_in_its_step_unit_passes(self, unit: str) -> None:
        _two_rates(f"1/{unit}", f"1/{unit}", step_unit=unit).check_unit_consistency()

    def test_a_model_in_steps_needs_no_step_unit(self) -> None:
        _two_rates("1/step", "1/step").check_unit_consistency()

    def test_aliases_of_one_unit_are_one_unit(self) -> None:
        _two_rates("1/day", "1/d", step_unit="day").check_unit_consistency()

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
        _two_rates("1/meter", "1/meter", step_unit="meter").check_unit_consistency()

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

    def test_a_step_unit_pint_does_not_define_is_its_own_dimension(self) -> None:
        _two_rates(
            "1/generation", "1/generation", step_unit="generation"
        ).check_unit_consistency()

    def test_a_differential_model_in_one_per_day_passes(self) -> None:
        _two_rates("1/day", "1/day", step_unit="day", ode=True).check_unit_consistency()

    def test_an_impulse_moves_an_amount_of_the_bin_unit(self) -> None:
        def impulse(unit: str) -> Model:
            return build(
                ModelBuilder(name="impulse", bin_unit="item")
                .add_bin(id="A", name="A")
                .add_bin(id="B", name="B")
                .add_parameter(id="p", value=0.1, unit=unit)
                .add_transition(
                    id="t_ab", source=["A"], target=["B"], rate="p * A", impulse_at=[1]
                )
                .set_initial_conditions(
                    total=1000,
                    bin_fractions=[
                        {"bin": "A", "fraction": 1.0},
                        {"bin": "B", "fraction": 0.0},
                    ],
                )
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
        self._proportional_to_t("1/day**2").check_unit_consistency()

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

        model.check_unit_consistency()

    def test_a_number_added_to_t_is_read_in_the_step_unit(self) -> None:
        self._windowed("k * (t - 3) * k * A").check_unit_consistency()

    def test_the_branches_of_if_share_a_unit(self) -> None:
        with pytest.raises(UnitConsistencyError, match=FAILED):
            self._windowed("if(t >= 10, k, cap) * A").check_unit_consistency()

    def test_a_computed_number_is_not_read_in_a_unit(self) -> None:
        with pytest.raises(UnitConsistencyError, match=FAILED):
            self._windowed("(k + exp(2)) * A").check_unit_consistency()
