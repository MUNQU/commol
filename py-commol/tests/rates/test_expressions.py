"""Rate expressions: constants, operators, the total and the subgroup totals."""

import math

import pytest

from commol.context._model import engine_state
from commol.context.model import Model
from tests.support import a_to_b_builder, build, run

GROUP_AND_TYPE = {
    "group": {"g1": 0.5, "g2": 0.5},
    "type": {"t1": 0.5, "t2": 0.5},
}


def _ab_model(rate: str) -> Model:
    return build(a_to_b_builder(rate=rate))


class TestConstants:
    @pytest.mark.parametrize(
        ("rate", "constant"),
        [
            ("pi * k * A", math.pi),
            ("e * k * A", math.e),
            ("k * pi", math.pi),
            ("pi / 10", math.pi),
        ],
    )
    def test_pi_and_e_are_constants_in_a_rate(self, rate: str, constant: float) -> None:
        result = run(_ab_model(rate), 1)

        assert result["B"][1] == pytest.approx(constant * 0.1 * 1000)


class TestSubgroupTotals:
    def test_categories_out_of_declaration_order_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_t1_g1") as error:
            build(
                a_to_b_builder(
                    rate="k * total_t1_g1 / total", stratifications=GROUP_AND_TYPE
                )
            )

        assert "total_g1_t1" in str(error.value)

    def test_categories_in_declaration_order_are_accepted(self) -> None:
        result = run(
            a_to_b_builder(
                rate="k * total_g1_t1 / total", stratifications=GROUP_AND_TYPE
            ),
            1,
        )

        # 0.1 * 250 / 1000 moves from each A compartment.
        assert result["B_g1_t1"][1] == pytest.approx(0.025)

    def test_an_unknown_subgroup_total_in_a_formula_parameter_is_rejected(
        self,
    ) -> None:
        builder = a_to_b_builder(
            parameters={"share": "total_t1_g1 / total"},
            rate="share",
            stratifications=GROUP_AND_TYPE,
        )

        with pytest.raises(ValueError, match="'total_t1_g1'.*Valid names: total_g1, "):
            build(builder)

    def test_a_combination_a_conditional_stratification_never_creates(
        self,
    ) -> None:
        builder = a_to_b_builder(
            rate="k * total_g2_s1",
            stratifications={
                "group": {"g1": 0.5, "g2": 0.5},
                "sub": {"s1": 0.5, "s2": 0.5},
            },
            conditions={"sub": {"group": "g1"}},
        )

        with pytest.raises(ValueError, match="'total_g2_s1'"):
            build(builder)

    def test_every_name_the_engine_provides_is_accepted_in_a_rate(self) -> None:
        strata = {
            "stratifications": {
                "group": {"g1": 0.5, "g2": 0.5},
                "sub": {"s1": 0.5, "s2": 0.5},
            },
            "conditions": {"sub": {"group": "g1"}},
        }
        structure = build(a_to_b_builder(**strata)).structure
        names = engine_state.subgroup_total_names(structure)

        assert "total_g1_s1" in names
        assert "total_g2_s1" not in names
        for name in names:
            build(a_to_b_builder(rate=f"k * {name} / total", **strata))


class TestOperators:
    def test_modulo_alone_is_an_operator(self) -> None:
        result = run(_ab_model("step % 2"), 2)

        # Per-capita rate 0 at step 0, 1 at step 1.
        assert result["A"][1] == pytest.approx(1000.0)
        assert result["A"][2] == pytest.approx(0.0)

    def test_modulo_with_a_parameter(self) -> None:
        result = run(_ab_model("k * A % 7"), 1)

        assert result["B"][1] == pytest.approx((0.1 * 1000) % 7)
