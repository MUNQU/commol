"""Bin fractions and the total of the initial conditions."""

import math

import pytest
from pydantic import ValidationError

from commol import Aggregate
from tests.support import a_to_b_builder, build, initial_state


def _conditions() -> Aggregate:
    return Aggregate(total=1000, bin_fractions={"A": 0.5, "B": 0.5})


class TestBinFractionUpdates:
    def test_complete_update_must_sum_to_one(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.9, "B": 0.9})

    def test_an_unknown_bin_names_the_bins(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError) as error:
            conditions.update_bin_fractions({"C": 0.5, "A": 0.5})

        assert str(error.value) == (
            "Bin(s) ['C'] not found in initial conditions. Available bins: ['A', 'B']"
        )
        assert conditions.bin_fractions == {"A": 0.5, "B": 0.5}

    def test_rejected_update_leaves_the_bins_unchanged(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            conditions.update_bin_fractions({"A": 0.9})

        assert conditions.bin_fractions == {"A": 0.5, "B": 0.5}

    def test_updated_bins_always_reload(self) -> None:
        conditions = _conditions()

        conditions.update_bin_fractions({"A": 0.4, "B": 0.6})

        reloaded = Aggregate.model_validate_json(conditions.model_dump_json())
        assert reloaded.bin_fractions == {"A": 0.4, "B": 0.6}


class TestTotal:
    @pytest.mark.parametrize("total", [0, -10, math.nan, math.inf])
    def test_a_total_that_is_not_finite_and_positive_is_rejected(
        self, total: float
    ) -> None:
        with pytest.raises(
            ValidationError, match="total must be finite and greater than 0"
        ):
            Aggregate(total=total, bin_fractions={"A": 1.0})

    def test_a_real_total_starts_with_amounts_that_sum_to_it(self) -> None:
        model = build(a_to_b_builder(total=2.5))

        assert sum(initial_state(model).values()) == pytest.approx(2.5)


class TestNames:
    def test_calibrated_fractions_leave_room_for_the_others(self) -> None:
        with pytest.raises(ValidationError, match="must be LESS than 1.0"):
            Aggregate(total=10, bin_fractions={"A": 1.0, "B": None})

    def test_the_bin_fractions_name_every_bin(self) -> None:
        builder = a_to_b_builder()
        builder.set_structure(Aggregate(total=10, bin_fractions={"A": 1.0, "C": 0.0}))

        with pytest.raises(
            ValidationError, match=r"Missing ids: \['B'\], Extra ids: \['C'\]"
        ):
            build(builder)
