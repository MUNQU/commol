"""The text of a rate: its LaTeX form and the merge of complementary terms."""

import pytest

from commol.context._model.equations import (
    _convert_division_to_frac,
    _strip_outer_parens,
    are_complementary_factors,
    latex_rate_expression,
    latex_variable,
    simplify_complementary_terms,
    tokenize_rate,
)


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
        ("beta * S * (I / N)", "\\beta \\cdot S \\cdot (I / N)"),
        ("k * sin(t / T) * A", "k \\cdot sin(t / T) \\cdot A"),
        ("x - y", "x - y"),
        ("((a + b) * c) / d", "\\frac{(a + b) \\cdot c}{d}"),
        ("a / (b / c)", "\\frac{a}{b / c}"),
    ],
)
def test_a_rate_in_latex_divides_with_frac_outside_parentheses(
    rate: str, latex: str
) -> None:
    assert latex_rate_expression(rate) == latex


def test_a_rate_in_latex_carries_the_unit_of_each_variable() -> None:
    units = {"k": "1/day", "f": "1 / day", "A": "item"}

    assert latex_rate_expression("k * f * A / B", units) == (
        "\\frac{k(\\text{1/day}) \\cdot f(\\text{1 / day}) \\cdot A(\\text{item})}{B}"
    )


def test_a_variable_inside_the_latex_of_another_is_not_replaced_again() -> None:
    units = {"k": "1/day", "A_g1": "item", "g1": "item"}

    assert latex_rate_expression("k * A_g1 / g1", units) == (
        "\\frac{k(\\text{1/day}) \\cdot A_{g1}(\\text{item})}{g1(\\text{item})}"
    )


def test_a_division_inside_parentheses_stays_a_division() -> None:
    assert _convert_division_to_frac("a / (b / c)") == "\\frac{a}{b / c}"
    assert _convert_division_to_frac("((a) + b) / c") == "\\frac{(a) + b}{c}"


@pytest.mark.parametrize(
    ("expression", "stripped"),
    [
        (" (a + b) ", "a + b"),
        ("((a))", "(a)"),
        ("(a) + (b)", "(a) + (b)"),
        ("(a", "(a"),
        ("a)", "a)"),
        ("(a))", "(a))"),
    ],
)
def test_outer_parentheses_go_only_when_they_wrap_everything(
    expression: str, stripped: str
) -> None:
    assert _strip_outer_parens(expression) == stripped


def test_a_rate_splits_into_its_factors_outside_parentheses() -> None:
    assert tokenize_rate("k * (a * b) * c") == ["k", "(a * b)", "c"]
    assert tokenize_rate("k*a") == ["k*a"]
    assert tokenize_rate("(a * b) * c") == ["(a * b)", "c"]
    assert tokenize_rate("((a * b) * c) * d") == ["((a * b) * c)", "d"]


@pytest.mark.parametrize(
    ("factors", "complementary"),
    [
        (["p", "(1 - p)"], True),
        (["p", "q", "(1 - p - q)"], True),
        (["(1 - q - p)", "p", "q"], True),
        (["p", "(1 - q)"], False),
        (["p", "(1 - p - q)"], False),
        (["p", "(2 - p)"], False),
        (["p", "q"], False),
        (["f(x)", "(1 - f(x))"], True),
        (["p"], False),
    ],
)
def test_factors_are_complementary_when_one_is_one_minus_the_others(
    factors: list[str], complementary: bool
) -> None:
    assert are_complementary_factors(factors) is complementary


@pytest.mark.parametrize(
    ("terms", "merged"),
    [
        ([(1, "k * A * p"), (1, "k * A * (1 - p)")], [(1, "k * A")]),
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
    ],
)
def test_terms_of_one_sign_with_complementary_factors_merge_in_place(
    terms: list[tuple[int, str]], merged: list[tuple[int, str]]
) -> None:
    assert simplify_complementary_terms(terms) == merged
