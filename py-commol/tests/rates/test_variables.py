"""A bin id in a rate is the total of the compartments of that bin."""

from commol import Simulation
from tests.support import (
    ENGINE_REL,
    assert_close,
    assert_total_conserved,
    build,
    reference_run,
    sir_builder,
    total,
)
from tests.support.reference import State

AGE = {"age": {"young": 0.6, "old": 0.4}}


def _bin_total(bin_id: str):
    def bin_total(state: State) -> float:
        return state[f"{bin_id}_young"] + state[f"{bin_id}_old"]

    return bin_total


class TestBinTotals:
    def test_a_bin_id_in_a_rate_sums_its_compartments(self) -> None:
        model = build(sir_builder(stratifications=AGE))

        result = Simulation(model).run(10)

        # Each age group moves the same absolute flow, computed on the totals.
        susceptible, infected = _bin_total("S"), _bin_total("I")
        expected = reference_run(
            {
                "S_young": 990 * 0.6,
                "S_old": 990 * 0.4,
                "I_young": 10 * 0.6,
                "I_old": 10 * 0.4,
                "R_young": 0.0,
                "R_old": 0.0,
            },
            [
                (
                    f"S_{age}",
                    f"I_{age}",
                    lambda s, _: 0.3 * susceptible(s) * infected(s) / total(s),
                )
                for age in ("young", "old")
            ]
            + [
                (f"I_{age}", f"R_{age}", lambda s, _: 0.1 * infected(s))
                for age in ("young", "old")
            ],
            10,
        )
        assert_close(result, expected, rel=ENGINE_REL)
        assert_total_conserved(result, 1000.0)

    def test_a_bin_id_equals_the_explicit_sum(self) -> None:
        by_bin = build(sir_builder(stratifications=AGE))
        by_sum = build(
            sir_builder(
                stratifications=AGE,
                infection="beta * (S_young + S_old) * (I_young + I_old) / total",
                recovery="gamma * (I_young + I_old)",
            )
        )

        assert_close(
            Simulation(by_bin).run(10), Simulation(by_sum).run(10), rel=ENGINE_REL
        )
