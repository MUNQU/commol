"""Bin fractions and the total of the initial conditions."""

import math

import pytest
from pydantic import ValidationError

from commol.context.initial_conditions import BinFraction, InitialConditions
from tests.support import a_to_b_builder, build, initial_state


def _conditions() -> InitialConditions:
    return InitialConditions(
        total=1000,
        bin_fractions=[
            BinFraction(bin="A", fraction=0.5),
            BinFraction(bin="B", fraction=0.5),
        ],
    )


class TestBinFractionUpdates:
    def test_complete_update_must_sum_to_one(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.9, "B": 0.9})

    def test_partial_update_must_keep_the_sum(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.4})

    def test_rejected_update_leaves_the_bins_unchanged(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.9})

        assert [(bf.bin, bf.fraction) for bf in conditions.bin_fractions] == [
            ("A", 0.5),
            ("B", 0.5),
        ]

    def test_updated_bins_always_reload(self) -> None:
        conditions = _conditions()

        conditions.update_bin_fractions({"A": 0.4, "B": 0.6})

        reloaded = InitialConditions.model_validate_json(conditions.model_dump_json())
        assert [(bf.bin, bf.fraction) for bf in reloaded.bin_fractions] == [
            ("A", 0.4),
            ("B", 0.6),
        ]


class TestTotal:
    @pytest.mark.parametrize("total", [0, -10, math.nan, math.inf])
    def test_a_total_that_is_not_finite_and_positive_is_rejected(
        self, total: float
    ) -> None:
        with pytest.raises(
            ValidationError, match="total must be finite and greater than 0"
        ):
            InitialConditions(
                total=total,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
            )

    def test_a_real_total_starts_with_amounts_that_sum_to_it(self) -> None:
        model = build(a_to_b_builder(total=2.5))

        assert sum(initial_state(model).values()) == pytest.approx(2.5)
