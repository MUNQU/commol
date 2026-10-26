"""The equations of the description of an unstratified model, in text and in LaTeX."""

import pytest

from commol import Aggregate
from commol._model.equations import DynamicsDescription
from commol._model.model import Model
from tests.support import (
    FactoryBuilder,
    abc_builder,
    build,
    build_ode,
    build_stochastic,
)
from tests.support.description import equation_forms


def _without_ab_rate() -> Model:
    """The transition ab has no rate, so it has no flow."""
    return build(
        abc_builder(name="Test Model", parameters={"k2": 0.1}, ab=None, bc=None)
        .add_transition(id="ab", source=["A"], target=["B"], rate=None)
        .add_transition(id="bc", source=["B"], target=["C"], rate="k2 * B")
    )


class TestText:
    def test_to_console(self) -> None:
        output = equation_forms(build(abc_builder(name="ABC Model")))

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  A -> B: k1 * A * B / total\n"
            "\n"
            "bc (B -> C):\n"
            "  B -> C: k2 * B\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A(t+Dt) - A(t)] / Dt = - (k1 * A * B / total)\n"
            "[B(t+Dt) - B(t)] / Dt = (k1 * A * B / total) - (k2 * B)\n"
            "[C(t+Dt) - C(t)] / Dt = (k2 * B)"
        )
        assert output == expected

    def test_without_rate(self) -> None:
        output = equation_forms(_without_ab_rate())

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  (none)\n"
            "\n"
            "bc (B -> C):\n"
            "  B -> C: k2 * B\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A(t+Dt) - A(t)] / Dt = 0\n"
            "[B(t+Dt) - B(t)] / Dt = - (k2 * B)\n"
            "[C(t+Dt) - C(t)] / Dt = (k2 * B)"
        )
        assert output == expected

    def test_a_model_of_category_transitions_only_lists_them(self) -> None:
        builder = (
            FactoryBuilder(name="Moves")
            .add_bin(id="A", name="A")
            .add_stratification(id="group", categories=["g1", "g2"])
            .add_transition(id="move", source=["g1"], target=["g2"], rate="0.1")
        )
        builder.set_structure(
            Aggregate(
                bin_amounts={"A": 100},
                stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
            )
        )

        text = build(builder).describe().render("text")

        assert "Group Stratification Transitions:\nmove (g1 -> g2):\n" in text
        assert "Bin Transitions" not in text

    def test_an_engine_without_equations_has_no_expanded_form(self) -> None:
        model = build_stochastic(abc_builder(), "StochasticEvents")

        with pytest.raises(ValueError) as error:
            DynamicsDescription(model).expanded_form()

        assert str(error.value) == (
            "The StochasticEvents engine has no equations: its dynamics are "
            "stochastic_events"
        )

    def test_an_unknown_format_lists_the_formats(self) -> None:
        description = build(abc_builder(name="ABC Model")).describe()

        with pytest.raises(ValueError, match="markdown"):
            description.render("invalid")  # ty: ignore[invalid-argument-type]


class TestLatex:
    def test_latex_format(self) -> None:
        output = equation_forms(build(abc_builder(name="ABC Model")), latex=True)

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  $A \\to B: \\frac{k1 \\cdot A \\cdot B}{total}$\n"
            "\n"
            "bc (B -> C):\n"
            "  $B \\to C: k2 \\cdot B$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{A(t+\\Delta t) - A(t)}{\\Delta t} = - (\\frac{k1 \\cdot A "
            "\\cdot B}{total})\\]\n"
            "\\[\\frac{B(t+\\Delta t) - B(t)}{\\Delta t} = (\\frac{k1 \\cdot A "
            "\\cdot B}{total}) - (k2 \\cdot B)\\]\n"
            "\\[\\frac{C(t+\\Delta t) - C(t)}{\\Delta t} = (k2 \\cdot B)\\]"
        )
        assert output == expected

    def test_a_greek_letter_id_prints_as_its_latex_command(self) -> None:
        model = build(
            abc_builder(
                parameters={"alpha": 0.3, "k2": 0.1}, ab="alpha * A * B / total"
            )
        )

        output = equation_forms(model, latex=True)

        assert "$A \\to B: \\frac{\\alpha \\cdot A \\cdot B}{total}$" in output

    def test_a_differential_model_writes_its_derivatives(self) -> None:
        model = build_ode(abc_builder(name="ABC Model"), settings=None)

        output = equation_forms(model, latex=True)

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  $A \\to B: \\frac{k1 \\cdot A \\cdot B}{total}$\n"
            "\n"
            "bc (B -> C):\n"
            "  $B \\to C: k2 \\cdot B$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{dA}{dt} = - (\\frac{k1 \\cdot A \\cdot B}{total})\\]\n"
            "\\[\\frac{dB}{dt} = (\\frac{k1 \\cdot A \\cdot B}{total}) "
            "- (k2 \\cdot B)\\]\n"
            "\\[\\frac{dC}{dt} = (k2 \\cdot B)\\]"
        )
        assert output == expected

    def test_latex_without_rate(self) -> None:
        output = equation_forms(_without_ab_rate(), latex=True)

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  (none)\n"
            "\n"
            "bc (B -> C):\n"
            "  $B \\to C: k2 \\cdot B$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{A(t+\\Delta t) - A(t)}{\\Delta t} = 0\\]\n"
            "\\[\\frac{B(t+\\Delta t) - B(t)}{\\Delta t} = - (k2 \\cdot B)\\]\n"
            "\\[\\frac{C(t+\\Delta t) - C(t)}{\\Delta t} = (k2 \\cdot B)\\]"
        )
        assert output == expected

    def test_latex_no_text_wrapping_on_math_parentheses(self) -> None:
        """
        ``(1 - p_h)`` is grouping, not a unit annotation, so it is not wrapped
        in ``\\text{...}``.
        """
        model = build(
            abc_builder(
                name="ABC Parenthesized Rate",
                parameters={"k1": 0.3, "k2": 0.1, "p_h": 0.05},
                bc="(1 - p_h) * k2 * B",
            )
        )

        output = equation_forms(model, latex=True)

        expected = (
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  $A \\to B: "
            "\\frac{k1 \\cdot A \\cdot B}{total}$\n"
            "\n"
            "bc (B -> C):\n"
            "  $B \\to C: "
            "(1 - p_{h}) \\cdot k2 \\cdot B$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{A(t+\\Delta t) - A(t)}{\\Delta t} = "
            "- (\\frac{k1 \\cdot A \\cdot B}{total})\\]\n"
            "\\[\\frac{B(t+\\Delta t) - B(t)}{\\Delta t} = "
            "(\\frac{k1 \\cdot A \\cdot B}{total}) "
            "- ((1 - p_{h}) \\cdot k2 \\cdot B)\\]\n"
            "\\[\\frac{C(t+\\Delta t) - C(t)}{\\Delta t} = "
            "((1 - p_{h}) \\cdot k2 \\cdot B)\\]"
        )
        assert output == expected
