"""``Model.check_unit_consistency``: every flow in ``<bin unit>/<step unit>``."""

import pytest

from commol import Aggregate, Complete, StratificationCondition, StratifiedRate
from commol._model.model import Model
from commol._model.unit_checks import UnitConsistencyError
from tests.support import (
    FactoryBuilder,
    abc_builder,
    abcd_builder,
    assert_accepted,
    build,
)

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
                StratifiedRate(
                    conditions=[
                        StratificationCondition(
                            stratification="group", category=category
                        )
                    ],
                    rate=rate,
                )
                for category, rate in zip(categories, rates, strict=True)
            ],
        )
    )


def _with_v(rate: str, threshold_unit: str = "item") -> Model:
    """A, V, B, C: A moves to V at ``rate`` besides the flows of ``abc_builder``."""
    return build(
        FactoryBuilder(name="AVBC", bin_unit="item", step_unit="day")
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
        .set_structure(Aggregate(bin_amounts={"A": 980, "V": 0, "B": 20, "C": 0}))
    )


class TestConsistentModels:
    def test_abc(self) -> None:
        assert_accepted(_abc().check_unit_consistency)

    def test_a_check_says_it_passed_only_when_verbose(self, capsys) -> None:
        _abc().check_unit_consistency()
        assert capsys.readouterr().out == ""

        _abc().check_unit_consistency(verbose=True)
        assert (
            capsys.readouterr().out == "Unit consistency check passed successfully.\n"
        )

    def test_rates_divided_by_subgroup_totals(self) -> None:
        model = _per_group_ab("k1 * A * B / total_g1", "k1 * A * B / total_g2")

        assert_accepted(model.check_unit_consistency)

    def test_a_proportional_rate_is_checked_as_its_flow(self) -> None:
        assert_accepted(_abc(bc="k2").check_unit_consistency)

    def test_a_proportional_sum_is_multiplied_by_its_source_as_a_whole(self) -> None:
        assert_accepted(_abc(bc="k1 + k2").check_unit_consistency)

    def test_a_proportional_rate_in_the_unit_of_a_flow_fails(self) -> None:
        model = _abc(bc="k2", units={"k1": "1/day", "k2": "item/day"})

        with pytest.raises(UnitConsistencyError, match="k2 \\* B"):
            model.check_unit_consistency()

    def test_the_neighbors_of_a_unit_have_the_bin_unit(self) -> None:
        builder = abc_builder(
            bin_unit="item",
            units=DAILY,
            step_unit="day",
            ab="k1 * neighbors_B / neighbors",
        )
        builder.structure = builder.structure.model_copy(
            update={"neighbors": Complete()}
        )

        assert_accepted(build(builder).check_unit_consistency)

    def test_a_dimensionless_parameter(self) -> None:
        model = _abc(
            parameters={"factor": 0.5, "k1": 0.5, "k2": 0.1},
            units={**DAILY, "factor": "dimensionless"},
            ab="k1 * factor * A * B / total",
        )

        assert_accepted(model.check_unit_consistency)

    def test_a_unit_pint_does_not_define_is_its_own_dimension(self) -> None:
        model = build(
            abc_builder(
                bin_unit="widget",
                parameters={"k1": 0.1},
                units={"k1": "1/generation"},
                step_unit="generation",
                bc=None,
            )
        )

        assert_accepted(model.check_unit_consistency)

    def test_a_with_v_rate(self) -> None:
        model = _with_v("k_v * A", threshold_unit="dimensionless")

        assert_accepted(model.check_unit_consistency)

    def test_exponential_decay_of_a_rate(self) -> None:
        model = _abc(
            parameters={"k1_0": 0.5, "decay_rate": 0.01, "k2": 0.1},
            units={"k1_0": "1/day", "decay_rate": "dimensionless", "k2": "1/day"},
            ab="k1_0 * exp(-decay_rate * step) * A * B / total",
        )

        assert_accepted(model.check_unit_consistency)

    def test_sin_of_step_forcing(self) -> None:
        model = _abc(
            parameters={"k1_avg": 0.5, "amplitude": 0.1, "k2": 0.1},
            units={
                "k1_avg": "1/day",
                "amplitude": "dimensionless",
                "k2": "1/day",
            },
            ab=("k1_avg * (1 + amplitude * sin(2 * pi * step / 365)) * A * B / total"),
        )

        assert_accepted(model.check_unit_consistency)

    def test_the_unit_of_a_formula_parameter_is_inferred(self) -> None:
        model = _abc(
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
        )

        assert_accepted(model.check_unit_consistency)

    def test_a_formula_of_subgroup_totals_is_dimensionless(self) -> None:
        model = _abc(
            parameters={
                "g1_fraction": "total_g1 / total",
                "k1": 0.3,
                "k2": 0.1,
            },
            stratifications={"group": {"g1": 0.3, "g2": 0.7}},
        )

        assert_accepted(model.check_unit_consistency)

    def test_chained_formula_parameters(self) -> None:
        model = _abc(
            parameters={
                "base_rate": 0.1,
                "adjusted_rate": "base_rate * 2",
                "final_rate": "adjusted_rate * 3",
                "k2": 0.1,
            },
            units={"base_rate": "1/day", "k2": "1/day"},
            ab="final_rate * A * B / total",
        )

        assert_accepted(model.check_unit_consistency)


class TestMissingUnits:
    def test_one_parameter_without_a_unit_is_named(self) -> None:
        with pytest.raises(UnitConsistencyError, match="missing units") as error:
            _abc(units={"k1": "1/day"}).check_unit_consistency()

        assert "k2" in str(error.value)

    def test_every_parameter_without_a_unit_is_named(self) -> None:
        with pytest.raises(UnitConsistencyError) as error:
            _abc(units={}).check_unit_consistency()

        assert "constant parameters are missing units: k1, k2. " in str(error.value)

    def test_a_formula_whose_unit_cannot_be_inferred_has_no_unit(self) -> None:
        model = _abc(
            parameters={
                "k2": 0.1,
                "f1": "2 * f2",
                "f2": "f1 / 2",
            },
            units={"k2": "1/day"},
            ab="f1 * A * B / total",
        )

        with pytest.raises(UnitConsistencyError, match="no defined unit") as error:
            model.check_unit_consistency()

        assert "Variable 'f1'" in str(error.value)


class TestInconsistentModels:
    def test_a_rate_in_the_wrong_unit_names_its_transition(self) -> None:
        model = _abc(units={"k1": "day", "k2": "1/day"})

        with pytest.raises(UnitConsistencyError, match=FAILED) as error:
            model.check_unit_consistency()

        assert "ab" in str(error.value)

    def test_every_wrong_flow_is_one_line_of_the_message(self) -> None:
        model = _abc(units={"k1": "day", "k2": "day"})

        with pytest.raises(UnitConsistencyError) as error:
            model.check_unit_consistency()

        lines = str(error.value).splitlines()
        assert lines[0] == "Unit consistency check failed:"
        assert [line[:20] for line in lines[1:]] == [
            "  - Transition 'ab':",
            "  - Transition 'bc':",
        ]

    def test_a_rate_shared_by_the_flows_of_a_transition_is_named_once(self) -> None:
        model = _abc(
            units={"k1": "day", "k2": "1/day"},
            stratifications={"group": {"g1": 0.5, "g2": 0.5}},
        )

        with pytest.raises(UnitConsistencyError) as error:
            model.check_unit_consistency()

        assert str(error.value).count("Transition 'ab'") == 1

    def test_a_flow_moves_the_unit_of_its_own_bin(self) -> None:
        def model(source: list[str], target: list[str], rate: str) -> Model:
            return build(
                FactoryBuilder(name="units", step_unit="day")
                .add_bin(id="A", name="A", unit="item")
                .add_bin(id="C", name="C", unit="kg")
                .add_stratification(id="group", categories=["g1", "g2"])
                .add_parameter(id="k", value=0.1, unit="1/day")
                .add_transition(id="t", source=source, target=target, rate=rate)
                .set_structure(
                    Aggregate(
                        bin_amounts={"A": 500, "C": 500},
                        stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
                    )
                )
            )

        assert_accepted(model(["g1"], ["g2"], "k").check_unit_consistency)
        assert_accepted(model(["C"], [], "k").check_unit_consistency)
        assert_accepted(model([], ["C"], "k * C").check_unit_consistency)
        with pytest.raises(UnitConsistencyError, match="expected 'kg/day'"):
            model(["C"], [], "k * A").check_unit_consistency()

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
                    StratifiedRate(
                        conditions=[
                            StratificationCondition(
                                stratification="group", category=group
                            ),
                            StratificationCondition(
                                stratification="type", category=kind
                            ),
                        ],
                        rate=rate,
                    )
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
