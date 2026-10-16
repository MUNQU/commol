"""print_equations shows units when every parameter and bin has one."""

import pytest

from commol.context.model import Model
from tests.support import FactoryBuilder, build, build_ode, abc_builder
from commol import Aggregate

DAILY = {"k1": "1/day", "k2": "1/day"}


def _printed(model: Model, capsys: pytest.CaptureFixture[str], **options) -> str:
    model.print_equations(**options)
    return capsys.readouterr().out


def _ab_into_b(**options) -> Model:
    """``abc_builder`` whose ``ab`` names B as a source and a target."""
    return build(
        abc_builder(name="ABC Model", ab=None, bc=None, **options)
        .add_transition(
            id="ab",
            source=["A", "B"],
            target=["B", "B"],
            rate="k1 * A * B / total",
        )
        .add_transition(id="bc", source=["B"], target=["C"], rate="k2 * B")
    )


class TestUnits:
    def test_print_equations_latex_with_units(self, capsys) -> None:
        model = build_ode(
            abc_builder(name="ABC Model with Units", bin_unit="item", units=DAILY),
            settings=None,
        )

        output = _printed(model, capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: ABC Model with Units\n"
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
            "  $A \\to B: \\frac{k1(\\text{1/day}) \\cdot A(\\text{item}) "
            "\\cdot B(\\text{item})}{total(\\text{item})} [\\text{item / day}]$\n"
            "\n"
            "bc (B -> C):\n"
            "  $B \\to C: k2(\\text{1/day}) \\cdot B(\\text{item}) "
            "[\\text{item / day}]$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{dA}{dt} = - (\\frac{k1 \\cdot A \\cdot B}{total})\\]\n"
            "\\[\\frac{dB}{dt} = (\\frac{k1 \\cdot A \\cdot B}{total}) "
            "- (k2 \\cdot B)\\]\n"
            "\\[\\frac{dC}{dt} = (k2 \\cdot B)\\]\n"
        )
        assert output == expected

    def test_print_equations_compact_with_complete_units(self, capsys) -> None:
        output = _printed(_ab_into_b(bin_unit="item", units=DAILY), capsys)

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
            "ab (A, B -> B):\n"
            "  A -> B: k1(1/day) * A(item) * B(item) / total(item) "
            "[item / day]\n"
            "  B -> B: k1(1/day) * A(item) * B(item) / total(item) "
            "[item / day]\n"
            "\n"
            "bc (B -> C):\n"
            "  B -> C: k2(1/day) * B(item) [item / day]\n"
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

    def test_print_equations_compact_without_units(self, capsys) -> None:
        output = _printed(_ab_into_b(), capsys)

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
            "ab (A, B -> B):\n"
            "  A -> B: k1 * A * B / total\n"
            "  B -> B: k1 * A * B / total\n"
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

    def test_print_equations_compact_with_partial_units(self) -> None:
        model = _ab_into_b(bin_unit="item", units={"k1": "1/day"})

        with pytest.raises(ValueError, match="Some parameters have units but not all"):
            model.print_equations()

    def test_print_equations_compact_with_bin_unit_only(self) -> None:
        model = _ab_into_b(bin_unit="item")

        with pytest.raises(ValueError, match="Some parameters have units but not all"):
            model.print_equations()

    def test_print_equations_compact_with_different_bin_units(self, capsys) -> None:
        model = build(
            FactoryBuilder(name="Mole Units", version="1.0", bin_unit="mol")
            .add_bin(id="A", name="A")
            .add_bin(id="B", name="B")
            .add_bin(id="C", name="C")
            .add_parameter(id="k1", value=0.5, unit="1/s")
            .add_parameter(id="k2", value=0.2, unit="1/s")
            .add_transition(
                id="forward", source=["A", "B"], target=["C", "C"], rate="k1 * A * B"
            )
            .add_transition(id="backward", source=["C"], target=["A"], rate="k2 * C")
            .set_structure(
                Aggregate(total=100, bin_fractions={"A": 0.4, "B": 0.4, "C": 0.2})
            )
        )

        output = _printed(model, capsys)

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Mole Units\n"
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
            "forward (A, B -> C):\n"
            "  A -> C: k1(1/s) * A(mol) * B(mol) [mole ** 2 / second]\n"
            "  B -> C: k1(1/s) * A(mol) * B(mol) [mole ** 2 / second]\n"
            "\n"
            "backward (C -> A):\n"
            "  C -> A: k2(1/s) * C(mol) [mole / second]\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[A(t+Dt) - A(t)] / Dt = - (k1 * A * B) + (k2 * C)\n"
            "[B(t+Dt) - B(t)] / Dt = - (k1 * A * B)\n"
            "[C(t+Dt) - C(t)] / Dt = (2 * k1 * A * B) - (k2 * C)\n"
        )
        assert output == expected
