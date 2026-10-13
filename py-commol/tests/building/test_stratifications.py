"""Stratifications expand each bin into compartments, fully or under conditions."""

import pytest

from commol import ModelBuilder, Simulation
from tests.support import assert_population_conserved, build, sir_builder

AGE = {"y60": 0.8, "oe60": 0.2}
VACCINATION = {"nv": 0.7, "v": 0.3}
COMORBIDITIES = {"nc": 0.8, "c": 0.2}


def _compartments(builder: ModelBuilder) -> set[str]:
    return set(Simulation(build(builder)).engine.compartments)


class TestFullProduct:
    def test_two_stratifications_give_every_combination(self) -> None:
        builder = sir_builder(
            stratifications={"age": AGE, "vaccination": VACCINATION},
        )

        assert _compartments(builder) == {
            f"{bin_id}_{age}_{vaccination}"
            for bin_id in ("S", "I", "R")
            for age in AGE
            for vaccination in VACCINATION
        }

    def test_a_run_of_a_two_stratification_model_conserves_the_population(
        self,
    ) -> None:
        model = build(
            sir_builder(
                parameters={"beta": 0.25, "gamma": 0.1},
                stratifications={
                    "age": {"young": 0.6, "old": 0.4},
                    "location": {"urban": 0.7, "rural": 0.3},
                },
            )
        )

        results = Simulation(model).run(10)

        assert len(results) == 12
        assert_population_conserved(results, 1000.0)


class TestConditionalStratification:
    """A conditional stratification splits only the compartments it names."""

    def test_vaccination_splits_only_the_oe60_compartments(self) -> None:
        builder = sir_builder(
            stratifications={"age": AGE, "vaccination": VACCINATION},
            conditions={"vaccination": {"age": "oe60"}},
        )

        assert _compartments(builder) == {
            f"{bin_id}_{suffix}"
            for bin_id in ("S", "I", "R")
            for suffix in ("y60", "oe60_nv", "oe60_v")
        }

    def test_two_conditional_stratifications_on_one_category(self) -> None:
        builder = sir_builder(
            fractions={"S": 0.98, "I": 0.02, "R": 0.0},
            stratifications={
                "age": AGE,
                "vaccination": VACCINATION,
                "comorbidities": COMORBIDITIES,
            },
            conditions={
                "vaccination": {"age": "oe60"},
                "comorbidities": {"age": "oe60"},
            },
        )

        assert _compartments(builder) == {
            f"{bin_id}_{suffix}"
            for bin_id in ("S", "I", "R")
            for suffix in (
                "y60",
                "oe60_nv_nc",
                "oe60_nv_c",
                "oe60_v_nc",
                "oe60_v_c",
            )
        }

    def test_a_run_conserves_the_population(self) -> None:
        model = build(
            sir_builder(
                fractions={"S": 0.9, "I": 0.1, "R": 0.0},
                stratifications={"age": AGE, "vaccination": VACCINATION},
                conditions={"vaccination": {"age": "oe60"}},
            )
        )

        assert_population_conserved(Simulation(model).run(20), 1000.0)

    def test_a_condition_on_an_undeclared_stratification_is_rejected(self) -> None:
        builder = ModelBuilder(name="invalid").add_bin(id="S", name="S")

        with pytest.raises(ValueError, match="not been declared"):
            builder.add_stratification(
                id="vaccination",
                categories=["nv", "v"],
                conditions=[{"stratification": "age", "category": "oe60"}],
            )

    def test_a_condition_without_a_category_is_rejected(self) -> None:
        builder = (
            ModelBuilder(name="invalid")
            .add_bin(id="A", name="A")
            .add_stratification(id="group", categories=["cat0", "cat1"])
        )

        with pytest.raises(ValueError, match="must include a 'category'"):
            builder.add_stratification(
                id="subgroup",
                categories=["sub0", "sub1"],
                conditions=[{"stratification": "group"}],
            )
