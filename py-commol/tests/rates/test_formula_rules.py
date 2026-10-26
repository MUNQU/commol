"""The rules of a formula: the parser of the engine checks every formula when
its object is built, and the error gives the message of the parser."""

from collections.abc import Callable
import re

import pytest

from commol import StratificationCondition, StratifiedRate
from commol import AxisPattern, CalibrationConstraint, ModelBuilder
from commol._model.engine_state import formula_tree
from commol._model.formula_tree import (
    Chain,
    Conditional,
    Constant,
    FunctionCall,
    Power,
    UnaryOp,
    Variable,
    read_tree,
    variables,
)
from tests.support import abc_builder, assert_accepted


def a_rate(formula: str) -> None:
    abc_builder(ab=formula)


def a_stratified_rate(formula: str) -> None:
    abc_builder(stratifications={"s": {"g1": 0.5, "g2": 0.5}}).add_transition(
        id="ac",
        source=["A"],
        target=["C"],
        rate="k1",
        stratified_rates=[
            StratifiedRate(
                conditions=[StratificationCondition(stratification="s", category="g1")],
                rate=formula,
            )
        ],
    )


def a_parameter(formula: str) -> None:
    ModelBuilder(name="m").add_parameter(id="k3", value=formula)


def a_constraint(formula: str) -> None:
    CalibrationConstraint(id="c", expression=formula)


def a_pattern(formula: str) -> None:
    AxisPattern.from_formula(formula)


OBJECTS: list[Callable[[str], None]] = [
    a_rate,
    a_stratified_rate,
    a_parameter,
    a_constraint,
    a_pattern,
]
"""Each object of the Python API that holds a formula, built with one."""

REJECTED = {
    "empty": ("", "the formula, or a pair of parentheses in it, holds nothing"),
    "malformed": ("k1 $ 2", "it does not parse: the character '$' at character"),
    "too deep": (
        "-" * 200 + "k1",
        "it nests deeper than 100 levels of parentheses, calls, signs and powers",
    ),
    "too many nodes": (
        " + ".join(["k1"] * 50_001),
        "it has 100001 nodes (numbers, variables, operators and calls), more than "
        "the limit of 100000",
    ),
    "unknown function": (
        "typeof(k1)",
        "'typeof' is not a function of the language, whose functions are sin,",
    ),
}
"""A formula of each kind the parser rejects, and the part of its message
that names why. A sum of n terms has 2n - 1 nodes."""


@pytest.mark.parametrize("build", OBJECTS)
@pytest.mark.parametrize(("formula", "reason"), REJECTED.values(), ids=REJECTED)
def test_a_formula_gets_the_message_of_the_engine_parser(
    build: Callable[[str], None], formula: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(reason)):
        build(formula)


@pytest.mark.parametrize("build", OBJECTS)
def test_a_formula_of_the_language_is_accepted(
    build: Callable[[str], None],
) -> None:
    formula = "k1 * if(step > 30, max(A, B) / total, sin(2 * pi * t / 365) ^ 2)"

    assert_accepted(build, formula)


def test_a_long_sum_is_one_level_of_nesting() -> None:
    assert_accepted(a_parameter, " + ".join(["k1"] * 20_000))


def test_the_error_of_a_formula_parameter_names_the_parameter() -> None:
    message = (
        "Parameter 'k3': Formula 'k1 * (step > 30)': the comparison '>' is only "
        "allowed in the condition of if(condition, a, b)"
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        a_parameter("k1 * (step > 30)")


def test_python_reads_the_tree_of_a_formula_from_the_engine() -> None:
    a, k = Variable("a"), Variable("k")

    assert formula_tree("if(!(a > 1e400), max(a, 0x10), -k ^ 2)") == Conditional(
        UnaryOp("Not", Chain(a, (("Gt", Constant(float("inf"))),))),
        FunctionCall("max", (a, Constant(16.0))),
        UnaryOp("Neg", Power(k, Constant(2.0))),
    )
    assert variables(formula_tree("k * a ^ x + if(x > 0, -k, max(a, 2))")) == [
        "k",
        "a",
        "x",
    ]
    with pytest.raises(ValueError, match="the boolean 'true' is not a number"):
        formula_tree("k * true")


def test_a_json_object_that_is_no_node_of_a_formula_is_rejected() -> None:
    with pytest.raises(ValueError, match="Not a node of the tree of a formula"):
        read_tree('{"Chain": {"first": {"Variable": "a"}, "rest": [["Mul", 2]]}}')
