"""Bin fractions and the population size of the initial conditions."""

import pytest
from pydantic import ValidationError

from commol.context.initial_conditions import BinFraction, InitialConditions


def _conditions() -> InitialConditions:
    return InitialConditions(
        population_size=1000,
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


class TestPopulationSize:
    @pytest.mark.parametrize("population_size", [0, -10])
    def test_a_non_positive_population_size_is_rejected(
        self, population_size: int
    ) -> None:
        with pytest.raises(
            ValidationError, match="population_size must be greater than 0"
        ):
            InitialConditions(
                population_size=population_size,
                bin_fractions=[BinFraction(bin="A", fraction=1.0)],
            )
