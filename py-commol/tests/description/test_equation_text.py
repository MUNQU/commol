"""The text of a rate: its LaTeX form and the merge of complementary terms,
built from the tree of the engine."""

import pytest

from commol.context._model.engine_state import formula_tree
from commol.context._model.equations import (
    _factors,
    are_complementary_factors,
    latex_rate_expression,
    latex_variable,
    simplify_complementary_terms,
)
from commol.context._model.formula_tree import formula_text
from tests.support import abc_builder, build
from tests.support.description import equation_forms


@pytest.mark.parametrize(
    ("name", "latex"),
    [
        ("k", "k"),
        ("beta", "\\beta"),
        ("A_g1", "A_{g1}"),
        ("beta_g1_x", "\\beta_{g1,x}"),
    ],
)
def test_a_variable_is_a_letter_with_its_categories_as_subscripts(
    name: str, latex: str
) -> None:
    assert latex_variable(name) == latex


@pytest.mark.parametrize(
    ("rate", "latex"),
    [
        ("k * A", "k \\cdot A"),
        ("(a + b) / (c - d)", "\\frac{a + b}{c - d}"),
        ("a / b / c", "\\frac{\\frac{a}{b}}{c}"),
        ("k * A / (1 + B)", "\\frac{k \\cdot A}{1 + B}"),
        ("beta * A * (B / N)", "\\beta \\cdot A \\cdot (B / N)"),
        ("k * sin(t / T) * A", "k \\cdot sin(t / T) \\cdot A"),
        ("x - y", "x - y"),
        ("((a + b) * c) / d", "\\frac{(a + b) \\cdot c}{d}"),
        ("a / (b / c)", "\\frac{a}{b / c}"),
        ("(a / b) / c", "\\frac{a / b}{c}"),
        ("((a) + b) / c", "\\frac{a + b}{c}"),
        ("a / b * c", "\\frac{a}{b} \\cdot c"),
        ("x + a / b", "x + \\frac{a}{b}"),
    ],
)
def test_a_rate_in_latex_divides_with_frac_outside_parentheses(
    rate: str, latex: str
) -> None:
    assert latex_rate_expression(formula_tree(rate)) == latex


@pytest.mark.parametrize(
    ("rate", "latex"),
    [
        ("A ^ 12", "A^{12}"),
        ("(a + b) ^ -c", "(a + b)^{-c}"),
        ("-(a - b) % 2", "-(a - b) \\bmod 2"),
        (
            "if(!(a <= 0.5), max(a, 1e-05), 2)",
            "if(\\lnot (a \\leq 0.5), max(a, 1e-05), 2)",
        ),
        (
            "if(a >= b, 1, 0) + if(a == b, 1, 0)",
            "if(a \\geq b, 1, 0) + if(a = b, 1, 0)",
        ),
        ("if(a != b, a, b)", "if(a \\neq b, a, b)"),
    ],
)
def test_a_rate_in_latex_writes_each_operator_of_the_language(
    rate: str, latex: str
) -> None:
    assert latex_rate_expression(formula_tree(rate)) == latex


def test_a_rate_in_latex_carries_the_unit_of_each_variable() -> None:
    units = {"k": "1/day", "f": "1 / day", "A": "item"}

    assert latex_rate_expression(formula_tree("k * f * A / B"), units) == (
        "\\frac{k(\\text{1/day}) \\cdot f(\\text{1 / day}) \\cdot A(\\text{item})}{B}"
    )


def test_a_variable_inside_the_latex_of_another_is_not_replaced_again() -> None:
    units = {"k": "1/day", "A_g1": "item", "g1": "item"}

    assert latex_rate_expression(formula_tree("k * A_g1 / g1"), units) == (
        "\\frac{k(\\text{1/day}) \\cdot A_{g1}(\\text{item})}{g1(\\text{item})}"
    )


@pytest.mark.parametrize(
    ("rate", "text"),
    [
        ("k*A/N", "k * A / N"),
        ("(a) + (b)", "a + b"),
        ("((a + b) * c) / d", "(a + b) * c / d"),
        ("a - (b - c)", "a - (b - c)"),
        ("(a - b) - c", "(a - b) - c"),
        ("-2 ** 2 ^ x", "-2 ^ 2 ^ x"),
        ("(-2) ^ (2 ^ x)", "(-2) ^ 2 ^ x"),
        ("(a ^ b) ^ c", "(a ^ b) ^ c"),
        ("--a", "--a"),
        ("1.0 * 1e-5 * 0x10 * 0.25", "1 * 1e-05 * 16 * 0.25"),
        ("if(!(a<b),max(a,b),0)", "if(!(a < b), max(a, b), 0)"),
    ],
)
def test_a_rate_is_written_with_the_parentheses_its_tree_needs(
    rate: str, text: str
) -> None:
    assert formula_text(formula_tree(rate)) == text
    assert formula_tree(text) == formula_tree(rate)


def test_the_spaces_of_a_rate_do_not_change_its_description() -> None:
    spaced = build(abc_builder(ab="k1 * A * B / total", bc="k2 * (1 - k1)"))
    compact = build(abc_builder(ab="k1*A*B/total", bc="k2*(1-k1)"))

    assert compact.describe().section("dynamics") == spaced.describe().section(
        "dynamics"
    )
    assert latex_rate_expression(formula_tree("k*S/N")) == latex_rate_expression(
        formula_tree("k * S / N")
    )


def test_a_proportional_sum_is_multiplied_by_its_source_as_a_whole() -> None:
    output = equation_forms(build(abc_builder(bc="k1 + k2")))

    assert "B -> C: (k1 + k2) * B" in output
    assert "[C(t+Dt) - C(t)] / Dt = ((k1 + k2) * B)" in output


def test_a_rate_splits_into_its_factors_outside_parentheses() -> None:
    def factors(rate: str) -> list[str]:
        return [formula_text(factor) for factor in _factors(formula_tree(rate))]

    assert factors("k * (a * b) * c") == ["k", "a * b", "c"]
    assert factors("k*a") == ["k", "a"]
    assert factors("(a * b) * c") == ["a * b", "c"]
    assert factors("k * A / N * p") == ["k", "A / N", "p"]
    assert factors("k * A % 2") == ["k * A % 2"]
    assert factors("a + b") == ["a + b"]


@pytest.mark.parametrize(
    ("factors", "complementary"),
    [
        (["p", "(1 - p)"], True),
        (["p", "q", "(1 - p - q)"], True),
        (["(1 - q - p)", "p", "q"], True),
        (["p", "(1 - q)"], False),
        (["p", "(1 - p - q)"], False),
        (["p", "(2 - p)"], False),
        (["p", "(1 + p)"], False),
        (["p", "q"], False),
        (["sin(x)", "(1 - sin(x))"], True),
        (["p"], False),
    ],
)
def test_factors_are_complementary_when_one_is_one_minus_the_others(
    factors: list[str], complementary: bool
) -> None:
    trees = [formula_tree(factor) for factor in factors]

    assert are_complementary_factors(trees) is complementary


@pytest.mark.parametrize(
    ("terms", "merged"),
    [
        ([(1, "k * A * p"), (1, "k * A * (1 - p)")], [(1, "k * A")]),
        ([(1, "k*A*p"), (1, "k * A * (1-p)")], [(1, "k * A")]),
        (
            [(1, "x"), (1, "k * p"), (-1, "y"), (1, "k * (1 - p)")],
            [(1, "x"), (1, "k"), (-1, "y")],
        ),
        (
            [(-1, "k * p"), (-1, "k * (1 - p)"), (1, "z")],
            [(-1, "k"), (1, "z")],
        ),
        (
            [(1, "k * p"), (-1, "k * (1 - p)")],
            [(1, "k * p"), (-1, "k * (1 - p)")],
        ),
        (
            [(1, "k * p"), (1, "j * (1 - p)")],
            [(1, "k * p"), (1, "j * (1 - p)")],
        ),
        ([(1, "x")], [(1, "x")]),
        (
            [
                (1, "a * p"),
                (1, "x"),
                (1, "a * (1 - p)"),
                (1, "b * q"),
                (1, "y"),
                (1, "b * (1 - q)"),
            ],
            [(1, "a"), (1, "x"), (1, "b"), (1, "y")],
        ),
        (
            [(1, "a * p"), (1, "b * q"), (1, "a * (1 - p)"), (1, "b * (1 - q)")],
            [(1, "a"), (1, "b")],
        ),
        (
            [(1, "k * p"), (1, "k * (1 - p)"), (1, "j * q"), (1, "j * (1 - q)")],
            [(1, "k"), (1, "j")],
        ),
        (
            [(1, "k * p"), (1, "k * (1 - p)"), (1, "k * x * y")],
            [(1, "k"), (1, "k * x * y")],
        ),
        (
            [(1, "a * p"), (1, "(1 - p) * a"), (1, "p * k")],
            [(1, "a"), (1, "p * k")],
        ),
        (
            [(1, "(1 - p) * k"), (1, "k * p"), (1, "p * k")],
            [(1, "k"), (1, "p * k")],
        ),
        ([(1, "p"), (1, "(1 - p)")], [(1, "p"), (1, "(1 - p)")]),
        (
            [(1, "p * q"), (1, "(1 - p) * q"), (1, "p * (1 - q)")],
            [(1, "q"), (1, "p * (1 - q)")],
        ),
        (
            [(1, "k * A / N * p"), (1, "k * A / N * (1 - p)")],
            [(1, "k * A / N")],
        ),
    ],
)
def test_terms_of_one_sign_with_complementary_factors_merge_in_place(
    terms: list[tuple[int, str]], merged: list[tuple[int, str]]
) -> None:
    trees = [(sign, formula_tree(term)) for sign, term in terms]

    assert [
        (sign, formula_text(term)) for sign, term in simplify_complementary_terms(trees)
    ] == [(sign, formula_text(formula_tree(term))) for sign, term in merged]
