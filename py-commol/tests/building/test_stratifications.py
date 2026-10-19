"""Stratifications expand each bin into compartments, fully or under conditions."""

import pytest

from commol import ModelBuilder, Simulation
from commol.context.stratification import Stratification
from tests.support import assert_total_conserved, build, abc_builder

GROUP = {"g1": 0.8, "g2": 0.2}
STATUS = {"s1": 0.7, "s2": 0.3}
LEVEL = {"l1": 0.8, "l2": 0.2}


def _compartments(builder: ModelBuilder) -> set[str]:
    return set(Simulation(build(builder)).engine.compartments)


class TestFullProduct:
    def test_two_stratifications_give_every_combination(self) -> None:
        builder = abc_builder(
            stratifications={"group": GROUP, "status": STATUS},
        )

        assert _compartments(builder) == {
            f"{bin_id}_{group}_{status}"
            for bin_id in ("A", "B", "C")
            for group in GROUP
            for status in STATUS
        }

    def test_a_run_of_a_two_stratification_model_keeps_the_total(
        self,
    ) -> None:
        model = build(
            abc_builder(
                parameters={"k1": 0.25, "k2": 0.1},
                ab="k1",
                bc="k2",
                stratifications={
                    "group": {"g1": 0.6, "g2": 0.4},
                    "type": {"t1": 0.7, "t2": 0.3},
                },
            )
        )

        results = Simulation(model).run(10)

        assert len(results) == 12
        assert_total_conserved(results, 1000.0)


class TestConditionalStratification:
    """A conditional stratification splits only the compartments it names."""

    def test_status_splits_only_the_g2_compartments(self) -> None:
        builder = abc_builder(
            stratifications={"group": GROUP, "status": STATUS},
            conditions={"status": {"group": "g2"}},
        )

        assert _compartments(builder) == {
            f"{bin_id}_{suffix}"
            for bin_id in ("A", "B", "C")
            for suffix in ("g1", "g2_s1", "g2_s2")
        }

    def test_two_conditional_stratifications_on_one_category(self) -> None:
        builder = abc_builder(
            amounts={"A": 980, "B": 20, "C": 0},
            stratifications={
                "group": GROUP,
                "status": STATUS,
                "level": LEVEL,
            },
            conditions={
                "status": {"group": "g2"},
                "level": {"group": "g2"},
            },
        )

        assert _compartments(builder) == {
            f"{bin_id}_{suffix}"
            for bin_id in ("A", "B", "C")
            for suffix in (
                "g1",
                "g2_s1_l1",
                "g2_s1_l2",
                "g2_s2_l1",
                "g2_s2_l2",
            )
        }

    def test_a_run_keeps_the_total(self) -> None:
        model = build(
            abc_builder(
                ab="k1",
                bc="k2",
                amounts={"A": 900, "B": 100, "C": 0},
                stratifications={"group": GROUP, "status": STATUS},
                conditions={"status": {"group": "g2"}},
            )
        )

        assert_total_conserved(Simulation(model).run(20), 1000.0)

    def test_a_condition_on_an_undeclared_stratification_is_rejected(self) -> None:
        builder = ModelBuilder(name="invalid").add_bin(id="A", name="A")

        with pytest.raises(ValueError, match="not been declared"):
            builder.add_stratification(
                id="status",
                categories=["s1", "s2"],
                conditions=[{"stratification": "group", "category": "g2"}],
            )

    def test_a_condition_without_a_category_is_rejected(self) -> None:
        builder = (
            ModelBuilder(name="invalid")
            .add_bin(id="A", name="A")
            .add_stratification(id="group", categories=["cat0", "cat1"])
        )

        with pytest.raises(ValueError, match="'group' must include a category"):
            builder.add_stratification(
                id="subgroup",
                categories=["sub0", "sub1"],
                conditions=[{"stratification": "group"}],
            )


class TestCategories:
    @pytest.mark.parametrize(
        ("categories", "message"),
        [
            ([], "'g' must have at least one category"),
            (["g1", "g2", "g1"], r"must not be repeated. Found duplicates: \['g1'\]"),
        ],
    )
    def test_a_stratification_has_distinct_categories(
        self, categories: list[str], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            ModelBuilder(name="m").add_stratification(id="g", categories=categories)

    def test_stratifications_of_one_id_are_one_member_of_a_set(self) -> None:
        first = Stratification(id="g", categories=["g1"])
        second = Stratification(id="g", categories=["g1", "g2"])

        assert {first, second} == {first}
