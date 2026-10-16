"""print_equations of an unstratified model, in text and in LaTeX."""

from pathlib import Path

import pytest

from commol.context.model import Model
from tests.support import build, build_ode, abcd_builder, abc_builder


def _printed(model: Model, capsys: pytest.CaptureFixture[str], **options) -> str:
    model.print_equations(**options)
    return capsys.readouterr().out


def _without_ab_rate() -> Model:
    """The transition ab has no rate, so it has no flow."""
    return build(
        abc_builder(name="Test Model", parameters={"k2": 0.1}, ab=None, bc=None)
        .add_transition(id="ab", source=["A"], target=["B"], rate=None)
        .add_transition(id="bc", source=["B"], target=["C"], rate="k2 * B")
    )


class TestText:
    def test_print_equations_to_console(self, capsys) -> None:
        output = _printed(build(abc_builder(name="ABC Model")), capsys)

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: ABC Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 2\n"
            "Number of Transitions: 2\n"
            "Bins: A, B, C\n"
            "\n"
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
            "[C(t+Dt) - C(t)] / Dt = (k2 * B)\n"
        )
        assert output == expected

    def test_print_equations_to_file(self, tmp_path: Path) -> None:
        model = build(abcd_builder(name="ABCD Model"))
        output_path = tmp_path / "equations.txt"

        model.print_equations(output_file=str(output_path))

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: ABCD Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 4\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: A, B, C, D\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  A -> B: k1 * A * C / total\n"
            "\n"
            "bc (B -> C):\n"
            "  B -> C: k3 * B\n"
            "\n"
            "cd (C -> D):\n"
            "  C -> D: k2 * C\n"
            "\n"
            "Total System: 4 coupled equations (4 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A(t+Dt) - A(t)] / Dt = - (k1 * A * C / total)\n"
            "[B(t+Dt) - B(t)] / Dt = (k1 * A * C / total) - (k3 * B)\n"
            "[C(t+Dt) - C(t)] / Dt = (k3 * B) - (k2 * C)\n"
            "[D(t+Dt) - D(t)] / Dt = (k2 * C)"
        )
        assert output_path.read_text() == expected

    def test_print_equations_without_rate(self, capsys) -> None:
        output = _printed(_without_ab_rate(), capsys)

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
            "Bins: A, B, C\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  (no flows)\n"
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
            "[C(t+Dt) - C(t)] / Dt = (k2 * B)\n"
        )
        assert output == expected

    def test_print_equations_invalid_format(self) -> None:
        model = build(abc_builder(name="ABC Model"))

        with pytest.raises(ValueError, match="Invalid format"):
            model.print_equations(format="invalid")


class TestLatex:
    def test_print_equations_latex_format(self, capsys) -> None:
        output = _printed(build(abc_builder(name="ABC Model")), capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: ABC Model\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 2\n"
            "Number of Transitions: 2\n"
            "Bins: A, B, C\n"
            "\n"
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
            "\\[\\frac{C(t+\\Delta t) - C(t)}{\\Delta t} = (k2 \\cdot B)\\]\n"
        )
        assert output == expected

    def test_a_greek_letter_id_prints_as_its_latex_command(self, capsys) -> None:
        model = build(
            abc_builder(
                parameters={"alpha": 0.3, "k2": 0.1}, ab="alpha * A * B / total"
            )
        )

        output = _printed(model, capsys, format="latex")

        assert "$A \\to B: \\frac{\\alpha \\cdot A \\cdot B}{total}$" in output

    def test_print_equations_latex_to_file(self, tmp_path: Path) -> None:
        model = build_ode(abc_builder(name="ABC Model"), settings=None)
        output_file = tmp_path / "equations_latex.txt"

        model.print_equations(output_file=str(output_file), format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: ABC Model\n"
            "Model Type: DifferentialEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 2\n"
            "Number of Transitions: 2\n"
            "Bins: A, B, C\n"
            "\n"
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
        assert output_file.read_text() == expected

    def test_print_equations_latex_without_rate(self, capsys) -> None:
        output = _printed(_without_ab_rate(), capsys, format="latex")

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
            "Bins: A, B, C\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "ab (A -> B):\n"
            "  (no flows)\n"
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
            "\\[\\frac{C(t+\\Delta t) - C(t)}{\\Delta t} = (k2 \\cdot B)\\]\n"
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
            abc_builder(
                name="ABC Parenthesized Rate",
                parameters={"k1": 0.3, "k2": 0.1, "p_h": 0.05},
                bc="(1 - p_h) * k2 * B",
            )
        )

        output = _printed(model, capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: ABC Parenthesized Rate\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 2\n"
            "Bins: A, B, C\n"
            "\n"
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
            "((1 - p_{h}) \\cdot k2 \\cdot B)\\]\n"
        )
        assert output == expected
