"""The variables of a formula: what the engine reads from its tree, the
engine variables, and a bin id as the total of the compartments of that bin."""

import re

import pytest

from commol import Simulation
from commol.commol_rs import _commol_rs as commol_rs
from commol._model.constants import SUBGROUP_TOTAL_PREFIX, TOTAL_VARIABLE
from tests.support import (
    ENGINE_REL,
    assert_close,
    assert_total_conserved,
    build,
    reference_run,
    abc_builder,
    total,
)
from tests.support.reference import State

GROUP = {"group": {"g1": 0.6, "g2": 0.4}}


def _bin_total(bin_id: str):
    def bin_total(state: State) -> float:
        return state[f"{bin_id}_g1"] + state[f"{bin_id}_g2"]

    return bin_total


class TestBinTotals:
    def test_a_bin_id_in_a_rate_sums_its_compartments(self) -> None:
        model = build(abc_builder(stratifications=GROUP))

        result = Simulation(model).run(10)

        # Each group moves the same absolute flow, computed on the totals.
        amount_a, amount_b = _bin_total("A"), _bin_total("B")
        expected = reference_run(
            {
                "A_g1": 990 * 0.6,
                "A_g2": 990 * 0.4,
                "B_g1": 10 * 0.6,
                "B_g2": 10 * 0.4,
                "C_g1": 0.0,
                "C_g2": 0.0,
            },
            [
                (
                    f"A_{group}",
                    f"B_{group}",
                    lambda s, _: 0.3 * amount_a(s) * amount_b(s) / total(s),
                )
                for group in ("g1", "g2")
            ]
            + [
                (f"B_{group}", f"C_{group}", lambda s, _: 0.1 * amount_b(s))
                for group in ("g1", "g2")
            ],
            10,
        )
        assert_close(result, expected, rel=ENGINE_REL)
        assert_total_conserved(result, 1000.0)

    def test_a_bin_id_equals_the_explicit_sum(self) -> None:
        by_bin = build(abc_builder(stratifications=GROUP))
        by_sum = build(
            abc_builder(
                stratifications=GROUP,
                ab="k1 * (A_g1 + A_g2) * (B_g1 + B_g2) / total",
                bc="k2 * (B_g1 + B_g2)",
            )
        )

        assert_close(
            Simulation(by_bin).run(10), Simulation(by_sum).run(10), rel=ENGINE_REL
        )


def test_an_undefined_variable_is_rejected_by_the_engine() -> None:
    builder = abc_builder(parameters={"k1": 0.3, "k2": 0.1}, bc="k9 * B")
    message = (
        "Undefined variables in transition 'bc' rate 'k9 * B': k9. Available "
        "parameters: k1, k2. Available bins: A, B, C."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        build(builder)


def test_a_category_in_a_rate_is_rejected() -> None:
    builder = abc_builder(
        parameters={"k1": 0.3, "k2": 0.1},
        bc="k2 * B * g1",
        stratifications={"group": {"g1": 0.5, "g2": 0.5}},
    )
    message = (
        "Undefined variables in transition 'bc' rate 'k2 * B * g1': g1. Available "
        "parameters: k1, k2. Available bins: A, B, C."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        build(builder)


def test_the_variables_of_a_formula_come_from_the_engine() -> None:
    expression = commol_rs.core.MathExpression("k1 * exp(-e1 * step) + 1e5")

    assert expression.variables == ["k1", "e1"]


def test_the_engine_variables_come_from_the_engine() -> None:
    assert commol_rs.model.ENGINE_VARIABLES == ["total", "step", "t", "pi", "e"]
    assert (TOTAL_VARIABLE, SUBGROUP_TOTAL_PREFIX) == ("total", "total_")
