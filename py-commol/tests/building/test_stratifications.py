"""Stratifications expand each bin into compartments, fully or under conditions."""

import re

import pytest

from pydantic import ValidationError

from commol import Aggregate, ModelBuilder, Simulation
from commol._model.stratification import Stratification
from tests.support import a_to_b_builder, abc_builder, assert_total_conserved, build

GROUP = {"g1": 0.8, "g2": 0.2}
STATUS = {"s1": 0.7, "s2": 0.3}
LEVEL = {"l1": 0.8, "l2": 0.2}
SHARING_A = {"group": {"a": 0.5, "b": 0.5}, "type": {"a": 0.5, "c": 0.5}}
"""Two stratifications that share the category id ``a``."""


def _compartments(builder: ModelBuilder) -> set[str]:
    return set(Simulation(build(builder)).engine.compartments)


def _moved(rate: str) -> float:
    """What a step of ``rate`` moves from A, of 1000, to B, both split by
    :data:`SHARING_A`."""
    results = Simulation(
        build(a_to_b_builder(rate=rate, stratifications=SHARING_A))
    ).run(1)
    return sum(series[-1] for name, series in results.items() if name.startswith("B_"))


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

    def test_a_condition_names_a_stratification_declared_before_it(self) -> None:
        builder = abc_builder(
            stratifications={"status": STATUS, "group": GROUP},
            conditions={"status": {"group": "g2"}},
        )

        with pytest.raises(
            ValueError,
            match="Stratification 'status': a condition names 'group', which is not "
            "a stratification declared before it",
        ):
            build(builder)


class TestCategories:
    @pytest.mark.parametrize(
        ("categories", "message"),
        [
            ([], "Stratification 'g' has no category; give it at least one"),
            (
                ["g1", "g2", "g1"],
                "Stratification 'g' names the category 'g1' twice; name each "
                "category once",
            ),
        ],
    )
    @pytest.mark.parametrize("fractions", [{}, {"g": {}}])
    def test_build_refuses_a_stratification_without_distinct_categories(
        self, categories: list[str], message: str, fractions: dict
    ) -> None:
        builder = a_to_b_builder()
        builder.add_stratification(id="g", categories=categories)
        builder.set_structure(
            Aggregate(
                bin_amounts={"A": 1000, "B": 0}, stratification_fractions=fractions
            )
        )

        with pytest.raises(ValidationError) as error:
            build(builder)

        assert error.value.errors()[0]["msg"] == f"Value error, {message}"

    def test_stratifications_of_one_id_are_one_member_of_a_set(self) -> None:
        first = Stratification(id="g", categories=["g1"])
        second = Stratification(id="g", categories=["g1", "g2"])

        assert {first, second} == {first}


def test_the_order_of_the_stratifications_sets_the_compartments() -> None:
    """The later stratification is conditional on a category of the earlier one."""
    group_first = abc_builder(
        stratifications={"group": GROUP, "status": STATUS},
        conditions={"status": {"group": "g2"}},
    )
    status_first = abc_builder(
        stratifications={"status": STATUS, "group": GROUP},
        conditions={"group": {"status": "s2"}},
    )

    assert Simulation(build(group_first)).engine.compartments == [
        f"{bin_id}_{suffix}"
        for bin_id in ("A", "B", "C")
        for suffix in ("g1", "g2_s1", "g2_s2")
    ]
    assert Simulation(build(status_first)).engine.compartments == [
        f"{bin_id}_{suffix}"
        for bin_id in ("A", "B", "C")
        for suffix in ("s1", "s2_g1", "s2_g2")
    ]


@pytest.mark.parametrize(
    ("rate", "read", "sums"),
    [
        pytest.param(
            "0.0001 * total_a",
            "'total_a' names the subgroup total of 'group:a' and the subgroup total "
            "of 'type:a'.",
            "'group:a' is 'total_a_a + total_a_c'; 'type:a' is 'total_a_a + total_b_a'",
            id="a subgroup total",
        ),
        pytest.param(
            "0.0001 * A_a",
            "'A_a' names the compartments of bin 'A' in 'group:a' and the "
            "compartments of bin 'A' in 'type:a'.",
            "'group:a' is 'A_a_a + A_a_c'; 'type:a' is 'A_a_a + A_b_a'",
            id="a partial bin total",
        ),
    ],
)
def test_a_rate_that_reads_a_name_of_a_shared_category_is_refused(
    rate: str, read: str, sums: str
) -> None:
    """``a`` gives ``total_a`` and ``A_a`` two meanings, the sum in ``group:a``
    and the sum in ``type:a``, and a rate reads one value. The message gives
    each meaning as a sum of names of one meaning."""
    builder = a_to_b_builder(rate=rate, stratifications=SHARING_A)
    name = read.split(" ", 1)[0]

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"transition 't_ab' reads {name}. {read} Rename category 'group:a' or "
            "category 'type:a', so that each has a name of its own. To keep the "
            f"ids, read a meaning as a sum of names of one meaning: {sums}."
        ),
    ):
        build(builder)


@pytest.mark.parametrize(
    "rate",
    [
        pytest.param("0.0001 * total_c", id="a name of one meaning"),
        pytest.param("0.0001 * (total_a_a + total_a_c)", id="group a, as two totals"),
        pytest.param("0.0001 * (total_a_a + total_b_a)", id="type a, as two totals"),
        pytest.param("0.0001 * (A_a_a + A_a_c)", id="group a of A, as compartments"),
    ],
)
def test_a_rate_reads_a_shared_category_through_names_of_one_meaning(
    rate: str,
) -> None:
    """Each of the 4 compartments of A moves 0.0001 * 500."""
    assert _moved(rate) == pytest.approx(0.2)
