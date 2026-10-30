"""A conditional stratification and a stratified rate whose condition names a
set of categories: the compartments it expands, the rates they take, and what
a build refuses."""

import math

import pytest

from commol import (
    Aggregate,
    ModelBuilder,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)
from commol._model.description import Table
from commol._model.model import Model
from tests.support import assert_accepted

GROUPS = {"g1": 0.2, "g2": 0.3, "g3": 0.5}

TYPES = {"t1": 0.5, "t2": 0.5}

ON_G2_G3 = StratificationCondition(stratification="group", categories=["g2", "g3"])
"""``type`` applies to ``g2`` and ``g3``."""


def _builder(
    condition: StratificationCondition = ON_G2_G3,
    rate: str = "k0",
    stratified_rates: list[StratifiedRate] | None = None,
) -> ModelBuilder:
    """Bins A and B, ``group`` of three categories and ``type`` on the
    categories of `condition`, A → B at `rate`."""
    return (
        ModelBuilder("sets")
        .add_bin("A", "A")
        .add_bin("B", "B")
        .add_stratification("group", list(GROUPS))
        .add_stratification("type", list(TYPES), conditions=[condition])
        .add_parameter("k0", 0.1)
        .add_parameter("k1", 0.2)
        .add_transition("ab", ["A"], ["B"], rate, stratified_rates=stratified_rates)
    )


def _build(
    builder: ModelBuilder, compartment_amounts: dict[str, float] | None = None
) -> Model:
    return builder.build(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
            stratification_fractions={"group": GROUPS, "type": TYPES},
            compartment_amounts=compartment_amounts or {},
        ),
        engine="DifferenceEquations",
    )


def _compartments(model: Model, bin_id: str = "A") -> list[str]:
    return [
        compartment.name
        for compartment in model.structure.initial_state(model.process)
        if compartment.name.split("_")[0] == bin_id
    ]


def _stratification_rows(model: Model) -> list[list[object]]:
    """The rows of the table of stratifications of the description."""
    return [
        row
        for block in model.describe().section("entities").blocks
        if isinstance(block, Table) and block.heading == "Stratifications"
        for row in block.rows
    ]


def test_a_condition_matches_any_category_of_a_set() -> None:
    model = _build(_builder())

    assert _compartments(model) == ["A_g1", "A_g2_t1", "A_g2_t2", "A_g3_t1", "A_g3_t2"]
    amounts = {
        compartment.name: compartment.amount
        for compartment in model.structure.initial_state(model.process)
    }
    assert amounts["A_g3_t2"] == pytest.approx(1000 * 0.5 * 0.5)
    assert model.structure.subgroup_total(model.process, ["g2", "g3"]) == (
        pytest.approx(800.0)
    )
    assert _stratification_rows(model)[1] == [
        "type",
        "t1, t2",
        "where group in {g2, g3}",
    ]


VALUES = {
    "g1": {"g1": 2.0, "g2": 0.5, "g3": 0.0},
    "g2": {"g1": 1.5, "g2": 3.0, "g3": 1.0},
    "g3": {"g1": 0.0, "g2": 0.5, "g3": 2.5},
}

START = {
    "A_g1": 200.0,
    "A_g2_t1": 100.0,
    "A_g2_t2": 200.0,
    "A_g3_t1": 300.0,
    "A_g3_t2": 100.0,
    "B_g1": 20.0,
    "B_g2_t1": 30.0,
    "B_g2_t2": 5.0,
    "B_g3_t1": 10.0,
    "B_g3_t2": 40.0,
}
"""Amounts whose ``B / total`` differs in each compartment, so a compartment
read for the sum of its group gives another flow."""


def test_a_matrix_couples_groups_that_a_set_splits() -> None:
    builder = _builder(rate="k1 * matrix_sum(M, B / total)")
    builder.add_matrix("M", "group", VALUES)
    model = _build(builder, compartment_amounts=START)

    run = Simulation(model).run(1)

    def of_group(bin_id: str, group: str) -> float:
        return sum(v for k, v in START.items() if k.startswith(f"{bin_id}_{group}"))

    share = {
        g: of_group("B", g) / (of_group("A", g) + of_group("B", g)) for g in GROUPS
    }
    for compartment in ("A_g1", "A_g2_t1", "A_g3_t2"):
        row = compartment.split("_")[1]
        flow = 0.2 * START[compartment] * sum(VALUES[row][j] * share[j] for j in GROUPS)
        assert math.isclose(run[compartment][1], START[compartment] - flow)


def _condition(**fields: object) -> StratificationCondition:
    return StratificationCondition.model_validate({"stratification": "group", **fields})


REFUSED = [
    (
        _condition(category="g2", categories=["g3"]),
        r"""the condition on 'group' gives both the category 'g2' and the """
        r"""categories \["g3"\]""",
        True,
    ),
    (_condition(categories=[]), "the condition on 'group' has an empty set", True),
    (
        _condition(categories=["g2", "g3", "g2"]),
        "the condition on 'group' names the category 'g2' twice",
        True,
    ),
    (
        _condition(categories=["g2", "g9"]),
        "the condition on 'group' names 'g9', which is not a category of 'group'",
        True,
    ),
    (
        _condition(categories=["g2", "g3"], to="g1"),
        "the condition on 'group' moves to 'g1' from a set of categories",
        True,
    ),
    (
        _condition(),
        "the condition on 'group' gives neither a category nor a set of categories",
        False,
    ),
]
"""Each condition, the message of its refusal, and whether a stratified rate
refuses it too: the condition of a rate without a category filters nothing."""


@pytest.mark.parametrize(("condition", "message", "of_a_rate_too"), REFUSED)
def test_a_condition_on_a_set_is_checked(
    condition: StratificationCondition, message: str, of_a_rate_too: bool
) -> None:
    of_a_stratification = _builder(condition=condition)
    of_a_rate = _builder(
        stratified_rates=[StratifiedRate(conditions=[condition], rate="k1")]
    )

    with pytest.raises(ValueError, match=f"Stratification 'type': {message}"):
        _build(of_a_stratification)
    if of_a_rate_too:
        with pytest.raises(
            ValueError, match=f"Transition 'ab', stratified rate 0: {message}"
        ):
            _build(of_a_rate)
    else:
        assert_accepted(_build, of_a_rate)


def test_a_condition_on_one_category_is_unchanged() -> None:
    on_g2 = StratificationCondition(stratification="group", category="g2")
    rate = StratifiedRate(
        conditions=[
            on_g2,
            StratificationCondition(stratification="type", category="t2"),
        ],
        rate="k1",
    )
    model = _build(_builder(condition=on_g2, stratified_rates=[rate]))

    assert _compartments(model) == ["A_g1", "A_g2_t1", "A_g2_t2", "A_g3"]
    run = Simulation(model).run(1)
    for compartment, k in [("A_g1", 0.1), ("A_g2_t1", 0.1), ("A_g2_t2", 0.2)]:
        start = run[compartment][0]
        assert math.isclose(run[compartment][1], start - k * start)
    assert _stratification_rows(model)[1] == ["type", "t1, t2", "where group = g2"]
    assert model.process.stratifications[1].conditions == [on_g2]
    assert "categories" not in on_g2.model_dump(exclude_none=True)
