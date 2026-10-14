"""print_equations of an unstratified model, in text and in LaTeX."""

from pathlib import Path

import pytest

from commol.context.model import Model
from tests.support import build, seir_builder, sir_builder


def _printed(model: Model, capsys: pytest.CaptureFixture[str], **options) -> str:
    model.print_equations(**options)
    return capsys.readouterr().out


def _without_infection_rate() -> Model:
    """The infection transition has no rate, so it has no flow."""
    return build(
        sir_builder(
            name="Test Model", parameters={"gamma": 0.1}, infection=None, recovery=None
        )
        .add_transition(id="infection", source=["S"], target=["I"], rate=None)
        .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
    )


class TestText:
    def test_print_equations_to_console(self, capsys) -> None:
        output = _printed(build(sir_builder(name="SIR Model")), capsys)

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 2\n"
            "Number of Transitions: 2\n"
            "Bins: S, I, R\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S -> I: beta * S * I / N\n"
            "\n"
            "recovery (I -> R):\n"
            "  I -> R: gamma * I\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S(t+Dt) - S(t)] / Dt = - (beta * S * I / N)\n"
            "[I(t+Dt) - I(t)] / Dt = (beta * S * I / N) - (gamma * I)\n"
            "[R(t+Dt) - R(t)] / Dt = (gamma * I)\n"
        )
        assert output == expected

    def test_print_equations_to_file(self, tmp_path: Path) -> None:
        model = build(seir_builder(name="SEIR Model"))
        output_path = tmp_path / "equations.txt"

        model.print_equations(output_file=str(output_path))

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SEIR Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 4\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, E, I, R\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "exposure (S -> E):\n"
            "  S -> E: beta * S * I / N\n"
            "\n"
            "progression (E -> I):\n"
            "  E -> I: sigma * E\n"
            "\n"
            "recovery (I -> R):\n"
            "  I -> R: gamma * I\n"
            "\n"
            "Total System: 4 coupled equations (4 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S(t+Dt) - S(t)] / Dt = - (beta * S * I / N)\n"
            "[E(t+Dt) - E(t)] / Dt = (beta * S * I / N) - (sigma * E)\n"
            "[I(t+Dt) - I(t)] / Dt = (sigma * E) - (gamma * I)\n"
            "[R(t+Dt) - R(t)] / Dt = (gamma * I)"
        )
        assert output_path.read_text() == expected

    def test_print_equations_without_rate(self, capsys) -> None:
        output = _printed(_without_infection_rate(), capsys)

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Test Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 1\n"
            "Number of Transitions: 2\n"
            "Bins: S, I, R\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  (no flows)\n"
            "\n"
            "recovery (I -> R):\n"
            "  I -> R: gamma * I\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S(t+Dt) - S(t)] / Dt = 0\n"
            "[I(t+Dt) - I(t)] / Dt = - (gamma * I)\n"
            "[R(t+Dt) - R(t)] / Dt = (gamma * I)\n"
        )
        assert output == expected

    def test_print_equations_invalid_format(self) -> None:
        model = build(sir_builder(name="SIR Model"))

        with pytest.raises(ValueError, match="Invalid format"):
            model.print_equations(format="invalid")


class TestLatex:
    def test_print_equations_latex_format(self, capsys) -> None:
        output = _printed(build(sir_builder(name="SIR Model")), capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 2\n"
            "Number of Transitions: 2\n"
            "Bins: S, I, R\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  $S \\to I: \\frac{\\beta \\cdot S \\cdot I}{N}$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I \\to R: \\gamma \\cdot I$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{S(t+\\Delta t) - S(t)}{\\Delta t} = - (\\frac{\\beta \\cdot S "
            "\\cdot I}{N})\\]\n"
            "\\[\\frac{I(t+\\Delta t) - I(t)}{\\Delta t} = (\\frac{\\beta \\cdot S "
            "\\cdot I}{N}) - (\\gamma \\cdot I)\\]\n"
            "\\[\\frac{R(t+\\Delta t) - R(t)}{\\Delta t} = (\\gamma \\cdot I)\\]\n"
        )
        assert output == expected

    def test_print_equations_latex_to_file(self, tmp_path: Path) -> None:
        model = sir_builder(name="SIR Model").build(engine="DifferentialEquations")
        output_file = tmp_path / "equations_latex.txt"

        model.print_equations(output_file=str(output_file), format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Model\n"
            "Model Type: DifferentialEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 2\n"
            "Number of Transitions: 2\n"
            "Bins: S, I, R\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  $S \\to I: \\frac{\\beta \\cdot S \\cdot I}{N}$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I \\to R: \\gamma \\cdot I$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{dS}{dt} = - (\\frac{\\beta \\cdot S \\cdot I}{N})\\]\n"
            "\\[\\frac{dI}{dt} = (\\frac{\\beta \\cdot S \\cdot I}{N}) "
            "- (\\gamma \\cdot I)\\]\n"
            "\\[\\frac{dR}{dt} = (\\gamma \\cdot I)\\]"
        )
        assert output_file.read_text() == expected

    def test_print_equations_latex_without_rate(self, capsys) -> None:
        output = _printed(_without_infection_rate(), capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Test Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 1\n"
            "Number of Transitions: 2\n"
            "Bins: S, I, R\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  (no flows)\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I \\to R: \\gamma \\cdot I$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{S(t+\\Delta t) - S(t)}{\\Delta t} = 0\\]\n"
            "\\[\\frac{I(t+\\Delta t) - I(t)}{\\Delta t} = - (\\gamma \\cdot I)\\]\n"
            "\\[\\frac{R(t+\\Delta t) - R(t)}{\\Delta t} = (\\gamma \\cdot I)\\]\n"
        )
        assert output == expected

    def test_print_equations_latex_no_text_wrapping_on_math_parentheses(
        self, capsys
    ) -> None:
        """
        ``(1 - p_h)`` is grouping, not a unit annotation, so it is not wrapped
        in ``\\text{...}``.
        """
        model = build(
            sir_builder(
                name="SIR Parenthesized Rate",
                parameters={"beta": 0.3, "gamma": 0.1, "p_h": 0.05},
                recovery="(1 - p_h) * gamma * I",
            )
        )

        output = _printed(model, capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Parenthesized Rate\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 2\n"
            "Bins: S, I, R\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  $S \\to I: "
            "\\frac{\\beta \\cdot S \\cdot I}{N}$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I \\to R: "
            "(1 - p_{h}) \\cdot \\gamma \\cdot I$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{S(t+\\Delta t) - S(t)}{\\Delta t} = "
            "- (\\frac{\\beta \\cdot S \\cdot I}{N})\\]\n"
            "\\[\\frac{I(t+\\Delta t) - I(t)}{\\Delta t} = "
            "(\\frac{\\beta \\cdot S \\cdot I}{N}) "
            "- ((1 - p_{h}) \\cdot \\gamma \\cdot I)\\]\n"
            "\\[\\frac{R(t+\\Delta t) - R(t)}{\\Delta t} = "
            "((1 - p_{h}) \\cdot \\gamma \\cdot I)\\]\n"
        )
        assert output == expected
