"""A rate reads the categories of a stratification through a matrix:
``matrix_sum(M, e)`` is ``Σ_j M[i, j] · e_j`` in a compartment of category
``i``, on every engine."""

import math

import pytest

from commol import (
    Aggregate,
    ModelBuilder,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)
from commol._model.equations import latex_rate_expression
from commol._model.formula_tree import (
    Chain,
    MatrixSum,
    Variable,
    formula_text,
    variables,
)
from commol._model.engine_state import formula_tree
from commol.engines.engine import EngineName
from tests.support import (
    MEAN_FIELD,
    STOCHASTIC_ENGINES,
    TIGHT,
    FactoryBuilder,
    a_to_b_builder,
    build,
)

VALUES = {"g1": {"g1": 2.0, "g2": 0.5}, "g2": {"g1": 1.5, "g2": 3.0}}
"""Rows ``g1`` 2 and 0.5, ``g2`` 1.5 and 3: not symmetric, so a row read as a
column gives another flow."""

GROUP = {"group": {"g1": 0.4, "g2": 0.6}}

START = {"A_g1": 300.0, "A_g2": 600.0, "B_g1": 20.0, "B_g2": 80.0}
"""Amounts whose ``B / total`` differs by group: 1/16 in ``g1``, 2/17 in ``g2``."""

SUM = "matrix_sum(M, B / total)"


def _builder(
    stratifications: dict[str, dict[str, float]] = GROUP,
    compartment_amounts: dict[str, float] | None = None,
) -> FactoryBuilder:
    builder = a_to_b_builder(
        parameters={"k": 0.1},
        rate=f"k * {SUM}",
        amounts={"A": 900, "B": 100},
        stratifications=stratifications,
        compartment_amounts=START
        if compartment_amounts is None
        else compartment_amounts,
    )
    builder.add_matrix("M", "group", VALUES)
    return builder


def _flows(state: dict[str, float]) -> dict[str, float]:
    """The flow out of each compartment of A at `state`, by hand."""
    b = {g: state[f"B_{g}"] / (state[f"A_{g}"] + state[f"B_{g}"]) for g in VALUES}
    return {
        f"A_{i}": 0.1 * state[f"A_{i}"] * sum(VALUES[i][j] * b[j] for j in VALUES)
        for i in VALUES
    }


def test_a_rate_reads_the_sum_over_the_categories_through_the_matrix() -> None:
    # The rate of each group reads its compartment of A, so it is absolute.
    model = (
        ModelBuilder("Coupled")
        .add_bin("A", "A")
        .add_bin("B", "B")
        .add_stratification("group", list(VALUES))
        .add_matrix("M", "group", VALUES)
        .add_parameter("k", 0.1)
        .add_transition(
            "ab",
            ["A"],
            ["B"],
            "0",
            stratified_rates=[
                StratifiedRate(
                    conditions=[
                        StratificationCondition(stratification="group", category=g)
                    ],
                    rate=f"k * A_{g} * {SUM}",
                )
                for g in VALUES
            ],
        )
        .build(
            Aggregate(
                bin_amounts={"A": 900, "B": 100},
                stratification_fractions=GROUP,
                compartment_amounts=START,
            ),
            engine="DifferenceEquations",
        )
    )

    run = Simulation(model).run(1)

    for compartment, flow in _flows(START).items():
        assert math.isclose(run[compartment][1], START[compartment] - flow)
    assert math.isclose(
        _flows(START)["A_g1"], 0.1 * 300 * (2 * 20 / 320 + 0.5 * 80 / 680)
    )


def test_a_differential_model_moves_at_the_rate_of_the_sum() -> None:
    builder = _builder()
    model = builder.build(
        builder.structure, engine="DifferentialEquations", settings=TIGHT
    )

    run = Simulation(model).run_at([1e-6])

    for compartment, flow in _flows(START).items():
        change = (START[compartment] - run[compartment][0]) / 1e-6
        assert math.isclose(change, flow, rel_tol=1e-5)


@pytest.mark.parametrize("engine", STOCHASTIC_ENGINES)
def test_the_mean_of_a_stochastic_engine_is_its_mean_field_run(
    engine: EngineName,
) -> None:
    builder = _builder()
    settings = TIGHT if MEAN_FIELD[engine] == "DifferentialEquations" else None
    mean_field = Simulation(
        builder.build(builder.structure, engine=MEAN_FIELD[engine], settings=settings)
    ).run_at([2.0])
    simulation = Simulation(builder.build(builder.structure, engine=engine))

    runs = simulation.replicates(300, seed=5).run_at([2.0])

    values = [run["B_g1"][0] for run in runs]
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    assert abs(mean - mean_field["B_g1"][0]) < 4.5 * spread / math.sqrt(300)


SPLIT = {
    "A_g1_s1": 100.0,
    "A_g1_s2": 200.0,
    "A_g2_s1": 450.0,
    "A_g2_s2": 150.0,
    "B_g1_s1": 5.0,
    "B_g1_s2": 15.0,
    "B_g2_s1": 60.0,
    "B_g2_s2": 20.0,
}
"""Amounts that differ in each compartment, so one compartment read for the
sum of its group gives another flow."""


def test_a_split_group_reads_the_sum_of_its_compartments() -> None:
    builder = _builder(
        stratifications={**GROUP, "sub": {"s1": 0.5, "s2": 0.5}},
        compartment_amounts=SPLIT,
    )

    run = Simulation(build(builder)).run(1)

    state = {name: series[0] for name, series in run.items()}
    grouped = {
        f"{bin_id}_{g}": state[f"{bin_id}_{g}_s1"] + state[f"{bin_id}_{g}_s2"]
        for bin_id in "AB"
        for g in VALUES
    }
    for g in VALUES:
        for s in ("s1", "s2"):
            share = state[f"A_{g}_{s}"] / grouped[f"A_{g}"]
            flow = _flows(grouped)[f"A_{g}"] * share
            assert math.isclose(run[f"A_{g}_{s}"][1], state[f"A_{g}_{s}"] - flow)


def test_a_matrix_sum_is_a_node_of_the_tree_and_writes_back() -> None:
    tree = formula_tree(f"k * {SUM}")

    assert formula_text(tree) == f"k * {SUM}"
    assert variables(tree) == ["k", "B", "total"]
    assert tree == Chain(
        Variable("k"),
        (("Mul", MatrixSum("M", Chain(Variable("B"), (("Div", Variable("total")),)))),),
    )
    assert latex_rate_expression(tree) == (
        "k \\cdot \\sum_{j} {M}_{i,j} \\left(\\frac{B}{total}\\right)_{j}"
    )
