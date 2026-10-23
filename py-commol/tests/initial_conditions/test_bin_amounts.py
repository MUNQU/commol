"""Bin amounts: the initial amount of each bin, in the unit of the bin."""

import math

import pytest
from pydantic import ValidationError

from commol import Aggregate, Simulation
from tests.support import a_to_b_builder, build, initial_state, mixed_units_builder


def _conditions() -> Aggregate:
    return Aggregate(bin_amounts={"A": 500, "B": 500})


def test_each_bin_starts_at_its_amount_in_its_unit() -> None:
    model = build(mixed_units_builder())

    results = Simulation(model).run(num_steps=1)

    assert initial_state(model) == {"A": 990, "B": 10, "C": 1e5}
    assert [results[bin_id][0] for bin_id in ("A", "B", "C")] == [990, 10, 1e5]


def test_fractions_of_a_total_are_rejected_with_the_new_fields() -> None:
    with pytest.raises(ValidationError) as error:
        Aggregate.model_validate(
            {"total": 1000, "bin_fractions": {"A": 1.0}, "compartment_fractions": {}}
        )

    message = str(error.value)
    assert "not fractions of a total" in message
    assert "replace total, bin_fractions, compartment_fractions" in message
    assert "with bin_amounts and compartment_amounts" in message


class TestBinAmountUpdates:
    def test_an_update_sets_the_amounts_it_names(self) -> None:
        conditions = _conditions()

        conditions.update_bin_amounts({"A": 900, "B": None})

        assert conditions.bin_amounts == {"A": 900, "B": None}

    def test_an_unknown_bin_names_the_bins(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError) as error:
            conditions.update_bin_amounts({"C": 0.5, "A": 0.5})

        assert str(error.value) == (
            "Bin(s) ['C'] not found in initial conditions. Available bins: ['A', 'B']"
        )
        assert conditions.bin_amounts == {"A": 500, "B": 500}

    def test_rejected_update_leaves_the_bins_unchanged(self) -> None:
        conditions = _conditions()

        with pytest.raises(ValueError, match="bin 'B' must be finite and not neg"):
            conditions.update_bin_amounts({"A": 900, "B": -1})

        assert conditions.bin_amounts == {"A": 500, "B": 500}

    def test_updated_bins_always_reload(self) -> None:
        conditions = _conditions()

        conditions.update_bin_amounts({"A": 400, "B": 600})

        reloaded = Aggregate.model_validate_json(conditions.model_dump_json())
        assert reloaded.bin_amounts == {"A": 400, "B": 600}


class TestAmounts:
    @pytest.mark.parametrize("amount", [-10, math.nan, math.inf])
    def test_an_amount_that_is_not_finite_or_is_negative_is_rejected(
        self, amount: float
    ) -> None:
        with pytest.raises(
            ValidationError, match="bin 'A' must be finite and not negative"
        ):
            Aggregate(bin_amounts={"A": amount})

    def test_bins_that_all_start_at_zero_build(self) -> None:
        model = build(a_to_b_builder(amounts={"A": 0, "B": 0}))

        assert initial_state(model) == {"A": 0, "B": 0}

    def test_the_bin_amounts_name_every_bin(self) -> None:
        builder = a_to_b_builder()
        builder.set_structure(Aggregate(bin_amounts={"A": 10, "C": 0}))

        with pytest.raises(
            ValidationError, match=r"Missing ids: \['B'\], Extra ids: \['C'\]"
        ):
            build(builder)
