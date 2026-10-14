"""print_equations shows units when every parameter and bin has one."""

import pytest

from commol import ModelBuilder
from commol.context.model import Model
from tests.support import build, sir_builder

DAILY = {"beta": "1/day", "gamma": "1/day"}


def _printed(model: Model, capsys: pytest.CaptureFixture[str], **options) -> str:
    model.print_equations(**options)
    return capsys.readouterr().out


def _reinfection(**options) -> Model:
    """SIR whose infection names I as a source and a target."""
    return build(
        sir_builder(name="SIR Model", infection=None, recovery=None, **options)
        .add_transition(
            id="infection",
            source=["S", "I"],
            target=["I", "I"],
            rate="beta * S * I / N",
        )
        .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
    )


class TestUnits:
    def test_print_equations_latex_with_units(self, capsys) -> None:
        model = sir_builder(
            name="SIR Model with Units", bin_unit="person", units=DAILY
        ).build(engine="DifferentialEquations")

        output = _printed(model, capsys, format="latex")

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Model with Units\n"
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
            "  $S \\to I: \\frac{\\beta(\\text{1/day}) \\cdot S(\\text{person}) "
            "\\cdot I(\\text{person})}{N(\\text{person})} [\\text{person / day}]$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I \\to R: \\gamma(\\text{1/day}) \\cdot I(\\text{person}) "
            "[\\text{person / day}]$\n"
            "\n"
            "Total System: 3 coupled equations (3 bins)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{dS}{dt} = - (\\frac{\\beta \\cdot S \\cdot I}{N})\\]\n"
            "\\[\\frac{dI}{dt} = (\\frac{\\beta \\cdot S \\cdot I}{N}) "
            "- (\\gamma \\cdot I)\\]\n"
            "\\[\\frac{dR}{dt} = (\\gamma \\cdot I)\\]\n"
        )
        assert output == expected

    def test_print_equations_compact_with_complete_units(self, capsys) -> None:
        output = _printed(_reinfection(bin_unit="person", units=DAILY), capsys)

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
            "infection (I, S -> I):\n"
            "  S -> I: beta(1/day) * S(person) * I(person) / N(person) [person / day]\n"
            "  I -> I: beta(1/day) * S(person) * I(person) / N(person) [person / day]\n"
            "\n"
            "recovery (I -> R):\n"
            "  I -> R: gamma(1/day) * I(person) [person / day]\n"
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

    def test_print_equations_compact_without_units(self, capsys) -> None:
        output = _printed(_reinfection(), capsys)

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
            "infection (I, S -> I):\n"
            "  S -> I: beta * S * I / N\n"
            "  I -> I: beta * S * I / N\n"
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

    def test_print_equations_compact_with_partial_units(self) -> None:
        model = _reinfection(bin_unit="person", units={"beta": "1/day"})

        with pytest.raises(ValueError, match="Some parameters have units but not all"):
            model.print_equations()

    def test_print_equations_compact_with_bin_unit_only(self) -> None:
        model = _reinfection(bin_unit="person")

        with pytest.raises(ValueError, match="Some parameters have units but not all"):
            model.print_equations()

    def test_print_equations_compact_with_different_bin_units(self, capsys) -> None:
        model = build(
            ModelBuilder(name="Chemical Reaction", version="1.0", bin_unit="mol")
            .add_bin(id="A", name="Reactant A")
            .add_bin(id="B", name="Reactant B")
            .add_bin(id="C", name="Product C")
            .add_parameter(id="k1", value=0.5, unit="1/s")
            .add_parameter(id="k2", value=0.2, unit="1/s")
            .add_transition(
                id="forward", source=["A", "B"], target=["C", "C"], rate="k1 * A * B"
            )
            .add_transition(id="backward", source=["C"], target=["A"], rate="k2 * C")
            .set_initial_conditions(
                population_size=100,
                bin_fractions=[
                    {"bin": "A", "fraction": 0.4},
                    {"bin": "B", "fraction": 0.4},
                    {"bin": "C", "fraction": 0.2},
                ],
            )
        )

        output = _printed(model, capsys)

        expected = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Chemical Reaction\n"
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
