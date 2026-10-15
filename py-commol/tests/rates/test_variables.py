"""A bin id in a rate is the total of the compartments of that bin."""

from commol import Simulation
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
