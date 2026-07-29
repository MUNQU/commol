import sys
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from commol.api.model_builder import ModelBuilder
from commol.constants import ModelTypes


class TestModel:
    def test_print_equations_to_console(self):
        """
        Test that print_equations outputs to console correctly in mathematical form.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations()
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
        assert output == expected_output

    def test_print_equations_latex_format(self):
        """
        Test that print_equations outputs to console in LaTeX format.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="latex")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
        assert output == expected_output

    def test_print_equations_invalid_format(self):
        """
        Test that print_equations raises error for invalid format.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with pytest.raises(ValueError, match="Invalid format"):
            model.print_equations(format="invalid")

    def test_print_equations_latex_with_stratification(self):
        """
        Test that print_equations outputs LaTeX format correctly with stratifications.
        """
        builder = (
            ModelBuilder(name="Age-Stratified SIR Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_parameter(id="aging_rate", value=0.01)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .add_transition(
                id="aging", source=["young"], target=["old"], rate="aging_rate"
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.7},
                            {"category": "old", "fraction": 0.3},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENTIAL_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="latex")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Age-Stratified SIR Model\n"
            "Model Type: DifferentialEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 1\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  $S_{young} \\to I_{young}: \\frac{\\beta \\cdot S \\cdot I}{N}$\n"
            "  $S_{old} \\to I_{old}: \\frac{\\beta \\cdot S \\cdot I}{N}$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I_{young} \\to R_{young}: \\gamma \\cdot I$\n"
            "  $I_{old} \\to R_{old}: \\gamma \\cdot I$\n"
            "\n"
            "Age Stratification Transitions (young -> old):\n"
            "For each bin X in {S, I, R}:\n"
            "  $X_{young} \\to X_{old}: aging_{rate} \\cdot X_{young}$\n"
            "\n"
            "Total System: 6 coupled equations (3 bins × 2 age)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{dS_{young}}{dt} = - (\\frac{\\beta \\cdot S \\cdot I}{N}) - "
            "(aging_{rate} \\cdot S_{young})\\]\n"
            "\\[\\frac{dS_{old}}{dt} = - (\\frac{\\beta \\cdot S \\cdot I}{N}) "
            "+ (aging_{rate} \\cdot S_{young})\\]\n"
            "\\[\\frac{dI_{young}}{dt} = (\\frac{\\beta \\cdot S \\cdot I}{N}) "
            "- (\\gamma \\cdot I) - (aging_{rate} \\cdot I_{young})\\]\n"
            "\\[\\frac{dI_{old}}{dt} = (\\frac{\\beta \\cdot S \\cdot I}{N}) "
            "- (\\gamma \\cdot I) + (aging_{rate} \\cdot I_{young})\\]\n"
            "\\[\\frac{dR_{young}}{dt} = (\\gamma \\cdot I) - (aging_{rate} "
            "\\cdot R_{young})\\]\n"
            "\\[\\frac{dR_{old}}{dt} = (\\gamma \\cdot I) + (aging_{rate} "
            "\\cdot R_{young})\\]\n"
        )
        assert output == expected_output

    def test_print_equations_latex_to_file(self):
        """
        Test that print_equations outputs LaTeX format to a file correctly.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENTIAL_EQUATIONS.value)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "equations_latex.txt"
            model.print_equations(output_file=str(output_file), format="latex")

            assert output_file.exists()
            content = output_file.read_text()

            expected_output = (
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
            assert content == expected_output

    def test_print_equations_latex_with_units(self):
        """
        Test that print_equations outputs LaTeX format correctly with unit annotations.
        """
        builder = (
            ModelBuilder(name="SIR Model with Units", version="1.0.0")
            .add_bin(id="S", name="Susceptible", unit="person")
            .add_bin(id="I", name="Infected", unit="person")
            .add_bin(id="R", name="Recovered", unit="person")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENTIAL_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="latex")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
        assert output == expected_output

    def test_print_equations_latex_without_rate(self):
        """
        Test LaTeX format with transitions without rate (LaTeX equivalent of
        test_print_equations_without_rate). Verifies that None rates are properly
        displayed as \\text{None} and result in 0 in expanded form.
        """
        builder = (
            ModelBuilder(name="Test Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate=None,  # No rate specified
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="latex")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
            "  $S \\to I: \\text{None}$\n"
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
        assert output == expected_output

    def test_parameter_with_formula_referencing_other_parameters(self):
        """
        Test that parameters can use formulas referencing other parameters.
        """
        builder = (
            ModelBuilder(name="SIR with Formula Parameters", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(
                id="beta_base", value=0.3, description="Base transmission rate"
            )
            .add_parameter(
                id="contact_multiplier", value=2.0, description="Contact multiplier"
            )
            .add_parameter(
                id="beta",
                value="beta_base * contact_multiplier",  # Formula parameter!
                description="Effective transmission rate",
            )
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Check that the parameter was created with a formula
        beta_param = next(p for p in model.parameters if p.id == "beta")
        assert isinstance(beta_param.value, str)
        assert beta_param.value == "beta_base * contact_multiplier"

    def test_parameter_with_formula_referencing_N(self):
        """
        Test that parameters can use formulas referencing N and N_category.
        """
        builder = (
            ModelBuilder(name="Model with N-based Parameters", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification("age", ["young", "old"])
            .add_parameter(
                id="young_fraction",
                value="N_young / N",  # Formula using population variables
                description="Fraction of young population",
            )
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.3},
                            {"category": "old", "fraction": 0.7},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Check that the parameter was created with a formula
        young_fraction_param = next(
            p for p in model.parameters if p.id == "young_fraction"
        )
        assert isinstance(young_fraction_param.value, str)
        assert young_fraction_param.value == "N_young / N"

    def test_print_equations_to_file(self):
        """
        Test that print_equations writes to a file correctly.
        """
        builder = (
            ModelBuilder(name="SEIR Model", version="2.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="E", name="Exposed")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.5)
            .add_parameter(id="sigma", value=0.2)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="Exposure",
                source=["S"],
                target=["E"],
                rate="beta * S * I / N",
            )
            .add_transition(
                id="Infection", source=["E"], target=["I"], rate="sigma * E"
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "E", "fraction": 0.01},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "equations.txt"

            model.print_equations(output_file=str(output_path))

            assert output_path.exists()

            content = output_path.read_text()

            expected_content = (
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
                "Exposure (S -> E):\n"
                "  S -> E: beta * S * I / N\n"
                "\n"
                "Infection (E -> I):\n"
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
            assert content == expected_content

    def test_print_equations_with_stratification(self):
        """
        Test that print_equations works with stratifications.
        """
        builder = (
            ModelBuilder(name="Age-Stratified SIR", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_parameter(id="aging_rate", value=0.01)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .add_transition(
                id="aging", source=["young"], target=["old"], rate="aging_rate"
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.7},
                            {"category": "old", "fraction": 0.3},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations()
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Age-Stratified SIR\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 1\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_young -> I_young: beta * S * I / N\n"
            "  S_old -> I_old: beta * S * I / N\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_young -> R_young: gamma * I\n"
            "  I_old -> R_old: gamma * I\n"
            "\n"
            "Age Stratification Transitions (young -> old):\n"
            "For each bin X in {S, I, R}:\n"
            "  X_young -> X_old: aging_rate * X_young\n"
            "\n"
            "Total System: 6 coupled equations (3 bins × 2 age)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_young(t+Dt) - S_young(t)] / Dt = - (beta * S * I / N) "
            "- (aging_rate * S_young)\n"
            "[S_old(t+Dt) - S_old(t)] / Dt = - (beta * S * I / N) "
            "+ (aging_rate * S_young)\n"
            "[I_young(t+Dt) - I_young(t)] / Dt = (beta * S * I / N) - (gamma * I) "
            "- (aging_rate * I_young)\n"
            "[I_old(t+Dt) - I_old(t)] / Dt = (beta * S * I / N) - (gamma * I) "
            "+ (aging_rate * I_young)\n"
            "[R_young(t+Dt) - R_young(t)] / Dt = (gamma * I) - (aging_rate * R_young)\n"
            "[R_old(t+Dt) - R_old(t)] / Dt = (gamma * I) + (aging_rate * R_young)\n"
        )
        assert output == expected_output

    def test_print_equations_without_rate(self):
        """
        Test that transitions without rate are properly ignored in equations.
        """
        builder = (
            ModelBuilder(name="Test Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate=None,  # No rate specified
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations()
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
            "  S -> I: None\n"
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
        assert output == expected_output

    def test_print_equations_with_multiple_stratifications(self):
        """
        Test that print_equations works with multiple stratifications (age and location)
        with different rates by category, different population summatories
        (N_young, N_old), and fallback rates for unspecified categories.
        """
        builder = (
            ModelBuilder(name="Age-Location Stratified SIR", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_stratification(id="location", categories=["urban", "rural"])
            # Transmission parameters with age-specific population denominators
            .add_parameter(id="beta_young_urban", value=0.4)
            .add_parameter(id="beta_young_rural", value=0.25)
            .add_parameter(id="beta_old_urban", value=0.35)
            # Note: beta_old_rural is intentionally not defined to test fallback rate
            # Recovery parameters - only define for young to test fallback
            .add_parameter(id="gamma_young", value=0.15)
            # Note: gamma_old is not defined - will use fallback rate
            # Stratification transition parameters
            .add_parameter(id="aging_rate", value=0.01)
            .add_parameter(id="migration_rate_young", value=0.02)
            .add_parameter(id="migration_rate_old", value=0.01)
            # Infection transition with stratified rates using age-specific populations
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="0.3 * S * I / N",  # Fallback rate (will be used for old_rural)
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "age", "category": "young"},
                            {"stratification": "location", "category": "urban"},
                        ],
                        "rate": "beta_young_urban * S * I / N_young",
                    },
                    {
                        "conditions": [
                            {"stratification": "age", "category": "young"},
                            {"stratification": "location", "category": "rural"},
                        ],
                        "rate": "beta_young_rural * S * I / N_young",
                    },
                    {
                        "conditions": [
                            {"stratification": "age", "category": "old"},
                            {"stratification": "location", "category": "urban"},
                        ],
                        "rate": "beta_old_urban * S * I / N_old",
                    },
                    # Note: old_rural is intentionally omitted to test fallback
                ],
            )
            # Recovery transition with stratified rate only for young
            # (old uses fallback)
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="0.1 * I",  # Fallback rate (will be used for old age group)
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "gamma_young * I",
                    },
                    # Note: old category is intentionally omitted to test fallback
                ],
            )
            # Aging transition (same for all locations)
            .add_transition(
                id="aging", source=["young"], target=["old"], rate="aging_rate"
            )
            # Migration transition with stratified rates by age
            .add_transition(
                id="migration",
                source=["urban"],
                target=["rural"],
                rate="0.015",  # Fallback rate (not used since all categories defined)
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "migration_rate_young",
                    },
                    {
                        "conditions": [{"stratification": "age", "category": "old"}],
                        "rate": "migration_rate_old",
                    },
                ],
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.7},
                            {"category": "old", "fraction": 0.3},
                        ],
                    },
                    {
                        "stratification": "location",
                        "fractions": [
                            {"category": "urban", "fraction": 0.6},
                            {"category": "rural", "fraction": 0.4},
                        ],
                    },
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations()
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: Age-Location Stratified SIR\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 7\n"
            "Number of Transitions: 4\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "  - location: [urban, rural]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_young_urban -> I_young_urban: beta_young_urban * S * I / N_young\n"
            "  S_young_rural -> I_young_rural: beta_young_rural * S * I / N_young\n"
            "  S_old_urban -> I_old_urban: beta_old_urban * S * I / N_old\n"
            "  S_old_rural -> I_old_rural: 0.3 * S * I / N\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_young_urban -> R_young_urban: gamma_young * I\n"
            "  I_young_rural -> R_young_rural: gamma_young * I\n"
            "  I_old_urban -> R_old_urban: 0.1 * I\n"
            "  I_old_rural -> R_old_rural: 0.1 * I\n"
            "\n"
            "Age Stratification Transitions (young -> old):\n"
            "For each bin X in {S, I, R} and each location "
            "in {urban, rural}:\n"
            "  X_young_urban -> X_old_urban: aging_rate * X_young_urban\n"
            "  X_young_rural -> X_old_rural: aging_rate * X_young_rural\n"
            "\n"
            "Location Stratification Transitions (urban -> rural):\n"
            "For each bin X in {S, I, R} and each age in {young, old}:\n"
            "  X_young_urban -> X_young_rural: migration_rate_young * X_young_urban\n"
            "  X_old_urban -> X_old_rural: migration_rate_old * X_old_urban\n"
            "\n"
            "Total System: 12 coupled equations "
            "(3 bins × 2 age × 2 location)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_young_urban(t+Dt) - S_young_urban(t)] / Dt = "
            "- (beta_young_urban * S * I / N_young) - (aging_rate * S_young_urban) "
            "- (migration_rate_young * S_young_urban)\n"
            "[S_young_rural(t+Dt) - S_young_rural(t)] / Dt = "
            "- (beta_young_rural * S * I / N_young) - (aging_rate * S_young_rural) "
            "+ (migration_rate_young * S_young_urban)\n"
            "[S_old_urban(t+Dt) - S_old_urban(t)] / Dt = "
            "- (beta_old_urban * S * I / N_old) + (aging_rate * S_young_urban) "
            "- (migration_rate_old * S_old_urban)\n"
            "[S_old_rural(t+Dt) - S_old_rural(t)] / Dt = - (0.3 * S * I / N) "
            "+ (aging_rate * S_young_rural) + (migration_rate_old * S_old_urban)\n"
            "[I_young_urban(t+Dt) - I_young_urban(t)] / Dt = "
            "(beta_young_urban * S * I / N_young) - (gamma_young * I) "
            "- (aging_rate * I_young_urban) "
            "- (migration_rate_young * I_young_urban)\n"
            "[I_young_rural(t+Dt) - I_young_rural(t)] / Dt = "
            "(beta_young_rural * S * I / N_young) "
            "- (gamma_young * I) - (aging_rate * I_young_rural) "
            "+ (migration_rate_young * I_young_urban)\n"
            "[I_old_urban(t+Dt) - I_old_urban(t)] / Dt = "
            "(beta_old_urban * S * I / N_old) - (0.1 * I) "
            "+ (aging_rate * I_young_urban) - (migration_rate_old * I_old_urban)\n"
            "[I_old_rural(t+Dt) - I_old_rural(t)] / Dt = "
            "(0.3 * S * I / N) - (0.1 * I) "
            "+ (aging_rate * I_young_rural) + (migration_rate_old * I_old_urban)\n"
            "[R_young_urban(t+Dt) - R_young_urban(t)] / Dt = "
            "(gamma_young * I) - (aging_rate * R_young_urban) "
            "- (migration_rate_young * R_young_urban)\n"
            "[R_young_rural(t+Dt) - R_young_rural(t)] / Dt = "
            "(gamma_young * I) - (aging_rate * R_young_rural) "
            "+ (migration_rate_young * R_young_urban)\n"
            "[R_old_urban(t+Dt) - R_old_urban(t)] / Dt = "
            "(0.1 * I) + (aging_rate * R_young_urban) "
            "- (migration_rate_old * R_old_urban)\n"
            "[R_old_rural(t+Dt) - R_old_rural(t)] / Dt = "
            "(0.1 * I) + (aging_rate * R_young_rural) "
            "+ (migration_rate_old * R_old_urban)\n"
        )
        assert output == expected_output

    def test_print_equations_compact_with_complete_units(self):
        """
        Test that print_equations shows units when all parameters and bins have units.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0", bin_unit="person")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
            .add_transition(
                id="infection",
                source=["S", "I"],
                target=["I", "I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations()
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
        assert output == expected_output

    def test_print_equations_compact_without_units(self):
        """
        Test that print_equations does not show units when no units are specified.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S", "I"],
                target=["I", "I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations()
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
        assert output == expected_output

    def test_print_equations_compact_with_partial_units(self):
        """
        Test that print_equations raises an error when only some parameters have units.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0", bin_unit="person")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1)  # No unit
            .add_transition(
                id="infection",
                source=["S", "I"],
                target=["I", "I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise ValueError for partial units
        with pytest.raises(ValueError, match="Some parameters have units but not all"):
            model.print_equations()

    def test_print_equations_compact_with_bin_unit_only(self):
        """
        Test that print_equations does not show units when only bin_unit is specified
        but parameters don't have units.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0", bin_unit="person")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S", "I"],
                target=["I", "I"],
                rate="beta * S * I / N",
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should raise ValueError for partial units (bins have units but params don't)
        with pytest.raises(ValueError, match="Some parameters have units but not all"):
            model.print_equations()

    def test_print_equations_compact_with_different_bin_units(self):
        """
        Test that print_equations works with any bin units.
        """
        builder = (
            ModelBuilder(name="Chemical Reaction", version="1.0.0", bin_unit="mol")
            .add_bin(id="A", name="Reactant A")
            .add_bin(id="B", name="Reactant B")
            .add_bin(id="C", name="Product C")
            .add_parameter(id="k1", value=0.5, unit="1/s")
            .add_parameter(id="k2", value=0.2, unit="1/s")
            .add_transition(
                id="forward",
                source=["A", "B"],
                target=["C", "C"],
                rate="k1 * A * B",
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

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations()
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
            "[A(t+Dt) - A(t)] / Dt = (k2 * C) - (k1 * A * B)\n"
            "[B(t+Dt) - B(t)] / Dt = - (k1 * A * B)\n"
            "[C(t+Dt) - C(t)] / Dt = (k1 * A * B) - (k2 * C)\n"
        )
        assert output == expected_output

    def _build_conditional_stratification_model(self):
        """Build model with conditional stratification and cross-category flows."""
        return (
            ModelBuilder(name="SIR Conditional", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["y", "o"])
            .add_stratification(
                id="vax",
                categories=["nv", "v"],
                conditions=[{"stratification": "age", "category": "o"}],
            )
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_parameter(id="aging", value=0.01)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
                per_compartment=True,
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="gamma",
                per_compartment=True,
            )
            .add_transition(
                id="aging_transition",
                source=["S"],
                target=["S"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "age", "category": "y", "to": "o"},
                            {"stratification": "vax", "category": "nv"},
                        ],
                        "rate": "aging * S_y",
                    }
                ],
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.005},
                    {"bin": "R", "fraction": 0.005},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "y", "fraction": 0.7},
                            {"category": "o", "fraction": 0.3},
                        ],
                    },
                    {
                        "stratification": "vax",
                        "fractions": [
                            {"category": "nv", "fraction": 0.6},
                            {"category": "v", "fraction": 0.4},
                        ],
                    },
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

    def test_print_equations_conditional_stratification_text(self):
        """
        Test print_equations with conditional stratifications and cross-category
        transitions in text format.

        Verifies that:
        - Compartments respect conditions (vax only applies to age=o)
        - Cross-category transitions (with ``to`` overrides) appear correctly
        - Expanded form shows correct inflow/outflow terms
        """
        model = self._build_conditional_stratification_model()

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="text")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Conditional\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [y, o]\n"
            "  - vax: [nv, v]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_y -> I_y: beta * S_y * I_y / N\n"
            "  S_o_nv -> I_o_nv: beta * S_o_nv * I_o_nv / N\n"
            "  S_o_v -> I_o_v: beta * S_o_v * I_o_v / N\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_y -> R_y: gamma\n"
            "  I_o_nv -> R_o_nv: gamma\n"
            "  I_o_v -> R_o_v: gamma\n"
            "\n"
            "aging_transition (S -> S):\n"
            "  S_y -> S_o_nv: aging * S_y\n"
            "\n"
            "Total System: 9 coupled equations "
            "(3 bins × 2 age × 2 vax)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_y(t+Dt) - S_y(t)] / Dt = "
            "- (beta * S_y * I_y / N) - (aging * S_y)\n"
            "[S_o_nv(t+Dt) - S_o_nv(t)] / Dt = "
            "- (beta * S_o_nv * I_o_nv / N) + (aging * S_y)\n"
            "[S_o_v(t+Dt) - S_o_v(t)] / Dt = "
            "- (beta * S_o_v * I_o_v / N)\n"
            "[I_y(t+Dt) - I_y(t)] / Dt = "
            "(beta * S_y * I_y / N) - (gamma)\n"
            "[I_o_nv(t+Dt) - I_o_nv(t)] / Dt = "
            "(beta * S_o_nv * I_o_nv / N) - (gamma)\n"
            "[I_o_v(t+Dt) - I_o_v(t)] / Dt = "
            "(beta * S_o_v * I_o_v / N) - (gamma)\n"
            "[R_y(t+Dt) - R_y(t)] / Dt = (gamma)\n"
            "[R_o_nv(t+Dt) - R_o_nv(t)] / Dt = (gamma)\n"
            "[R_o_v(t+Dt) - R_o_v(t)] / Dt = (gamma)\n"
        )
        assert output == expected_output

    def test_print_equations_conditional_stratification_latex(self):
        """
        Test print_equations with conditional stratifications and cross-category
        transitions in LaTeX format.

        Verifies LaTeX rendering of conditional compartments and cross-category
        transition arrows.
        """
        model = self._build_conditional_stratification_model()

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="latex")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Conditional\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [y, o]\n"
            "  - vax: [nv, v]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  $S_{y} \\to I_{y}: "
            "\\frac{\\beta \\cdot S_{y} \\cdot I_{y}}{N}$\n"
            "  $S_{o,nv} \\to I_{o,nv}: "
            "\\frac{\\beta \\cdot S_{o,nv} \\cdot I_{o,nv}}{N}$\n"
            "  $S_{o,v} \\to I_{o,v}: "
            "\\frac{\\beta \\cdot S_{o,v} \\cdot I_{o,v}}{N}$\n"
            "\n"
            "recovery (I -> R):\n"
            "  $I_{y} \\to R_{y}: \\gamma$\n"
            "  $I_{o,nv} \\to R_{o,nv}: \\gamma$\n"
            "  $I_{o,v} \\to R_{o,v}: \\gamma$\n"
            "\n"
            "aging_transition (S -> S):\n"
            "  $S_{y} \\to S_{o,nv}: aging \\cdot S_{y}$\n"
            "\n"
            "Total System: 9 coupled equations "
            "(3 bins × 2 age × 2 vax)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "\\[\\frac{S_{y}(t+\\Delta t) - S_{y}(t)}{\\Delta t} = "
            "- (\\frac{\\beta \\cdot S_{y} \\cdot I_{y}}{N}) "
            "- (aging \\cdot S_{y})\\]\n"
            "\\[\\frac{S_{o,nv}(t+\\Delta t) - S_{o,nv}(t)}{\\Delta t} = "
            "- (\\frac{\\beta \\cdot S_{o,nv} \\cdot I_{o,nv}}{N}) "
            "+ (aging \\cdot S_{y})\\]\n"
            "\\[\\frac{S_{o,v}(t+\\Delta t) - S_{o,v}(t)}{\\Delta t} = "
            "- (\\frac{\\beta \\cdot S_{o,v} \\cdot I_{o,v}}{N})\\]\n"
            "\\[\\frac{I_{y}(t+\\Delta t) - I_{y}(t)}{\\Delta t} = "
            "(\\frac{\\beta \\cdot S_{y} \\cdot I_{y}}{N}) "
            "- (\\gamma)\\]\n"
            "\\[\\frac{I_{o,nv}(t+\\Delta t) - I_{o,nv}(t)}{\\Delta t} = "
            "(\\frac{\\beta \\cdot S_{o,nv} \\cdot I_{o,nv}}{N}) "
            "- (\\gamma)\\]\n"
            "\\[\\frac{I_{o,v}(t+\\Delta t) - I_{o,v}(t)}{\\Delta t} = "
            "(\\frac{\\beta \\cdot S_{o,v} \\cdot I_{o,v}}{N}) "
            "- (\\gamma)\\]\n"
            "\\[\\frac{R_{y}(t+\\Delta t) - R_{y}(t)}{\\Delta t} = "
            "(\\gamma)\\]\n"
            "\\[\\frac{R_{o,nv}(t+\\Delta t) - R_{o,nv}(t)}{\\Delta t} = "
            "(\\gamma)\\]\n"
            "\\[\\frac{R_{o,v}(t+\\Delta t) - R_{o,v}(t)}{\\Delta t} = "
            "(\\gamma)\\]\n"
        )
        assert output == expected_output

    def test_print_equations_cross_category_multiple_targets(self):
        """
        Test cross-category transitions that split into multiple target
        compartments (e.g. aging from y to o with comorbidity split).
        """
        model = (
            ModelBuilder(name="SIR Aging Split", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["y", "o"])
            .add_stratification(
                id="comorbidity",
                categories=["nc", "c"],
                conditions=[{"stratification": "age", "category": "o"}],
            )
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_parameter(id="aging", value=0.01)
            .add_parameter(id="p_c", value=0.2)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
                per_compartment=True,
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="gamma",
                per_compartment=True,
            )
            .add_transition(
                id="aging_nc",
                source=["S"],
                target=["S"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "age", "category": "y", "to": "o"},
                            {"stratification": "comorbidity", "category": "nc"},
                        ],
                        "rate": "aging * (1 - p_c) * S_y",
                    }
                ],
            )
            .add_transition(
                id="aging_c",
                source=["S"],
                target=["S"],
                stratified_rates=[
                    {
                        "conditions": [
                            {"stratification": "age", "category": "y", "to": "o"},
                            {"stratification": "comorbidity", "category": "c"},
                        ],
                        "rate": "aging * p_c * S_y",
                    }
                ],
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.005},
                    {"bin": "R", "fraction": 0.005},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "y", "fraction": 0.7},
                            {"category": "o", "fraction": 0.3},
                        ],
                    },
                    {
                        "stratification": "comorbidity",
                        "fractions": [
                            {"category": "nc", "fraction": 0.8},
                            {"category": "c", "fraction": 0.2},
                        ],
                    },
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="text")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
            "========================================\n"
            "MODEL INFORMATION\n"
            "========================================\n"
            "Model: SIR Aging Split\n"
            "Model Type: DifferenceEquations\n"
            "Number of Bins: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 4\n"
            "Number of Transitions: 4\n"
            "Bins: S, I, R\n"
            "Stratifications:\n"
            "  - age: [y, o]\n"
            "  - comorbidity: [nc, c]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Bin Transitions:\n"
            "infection (S -> I):\n"
            "  S_y -> I_y: beta * S_y * I_y / N\n"
            "  S_o_nc -> I_o_nc: beta * S_o_nc * I_o_nc / N\n"
            "  S_o_c -> I_o_c: beta * S_o_c * I_o_c / N\n"
            "\n"
            "recovery (I -> R):\n"
            "  I_y -> R_y: gamma\n"
            "  I_o_nc -> R_o_nc: gamma\n"
            "  I_o_c -> R_o_c: gamma\n"
            "\n"
            "aging_nc (S -> S):\n"
            "  S_y -> S_o_nc: aging * (1 - p_c) * S_y\n"
            "\n"
            "aging_c (S -> S):\n"
            "  S_y -> S_o_c: aging * p_c * S_y\n"
            "\n"
            "Total System: 9 coupled equations "
            "(3 bins × 2 age × 2 comorbidity)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "[S_y(t+Dt) - S_y(t)] / Dt = "
            "- (beta * S_y * I_y / N) "
            "- (aging * S_y)\n"
            "[S_o_nc(t+Dt) - S_o_nc(t)] / Dt = "
            "- (beta * S_o_nc * I_o_nc / N) "
            "+ (aging * (1 - p_c) * S_y)\n"
            "[S_o_c(t+Dt) - S_o_c(t)] / Dt = "
            "- (beta * S_o_c * I_o_c / N) "
            "+ (aging * p_c * S_y)\n"
            "[I_y(t+Dt) - I_y(t)] / Dt = "
            "(beta * S_y * I_y / N) - (gamma)\n"
            "[I_o_nc(t+Dt) - I_o_nc(t)] / Dt = "
            "(beta * S_o_nc * I_o_nc / N) - (gamma)\n"
            "[I_o_c(t+Dt) - I_o_c(t)] / Dt = "
            "(beta * S_o_c * I_o_c / N) - (gamma)\n"
            "[R_y(t+Dt) - R_y(t)] / Dt = (gamma)\n"
            "[R_o_nc(t+Dt) - R_o_nc(t)] / Dt = (gamma)\n"
            "[R_o_c(t+Dt) - R_o_c(t)] / Dt = (gamma)\n"
        )
        assert output == expected_output

    def test_print_equations_latex_no_text_wrapping_on_math_parentheses(self):
        """
        Test that parenthesized math sub-expressions like ``(1 - p_h)`` are
        NOT wrapped in ``\\text{...}`` in LaTeX output.

        The ``\\text{}`` wrapper is only for unit annotations like ``(person)``
        or ``(1/day)``, not for mathematical grouping with operators or LaTeX
        commands.
        """
        model = (
            ModelBuilder(name="SIR Parenthesized Rate", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_parameter(id="p_h", value=0.05)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="(1 - p_h) * gamma * I",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
            .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        )

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            model.print_equations(format="latex")
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        expected_output = (
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
        assert output == expected_output


def _stratified_model():
    """Two bins split by one binary and one ternary stratification."""
    return (
        ModelBuilder(name="Stratified", version="1.0")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_stratification(id="group", categories=["group1", "group2"])
        .add_stratification(id="tier", categories=["low", "mid", "high"])
        .add_parameter(id="k1", value=0.3)
        .add_transition(id="flow", source=["A"], target=["B"], rate="k1")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[
                {"bin": "A", "fraction": 0.9},
                {"bin": "B", "fraction": 0.1},
            ],
            stratification_fractions=[
                {
                    "stratification": "group",
                    "fractions": [
                        {"category": "group1", "fraction": 0.6},
                        {"category": "group2", "fraction": 0.4},
                    ],
                },
                {
                    "stratification": "tier",
                    "fractions": [
                        {"category": "low", "fraction": 0.5},
                        {"category": "mid", "fraction": 0.3},
                        {"category": "high", "fraction": 0.2},
                    ],
                },
            ],
        )
        .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
    )


def _fractions_of(model, stratification_id: str) -> dict[str, float]:
    for stratification in model.population.initial_conditions.stratification_fractions:
        if stratification.stratification == stratification_id:
            return {f.category: f.fraction for f in stratification.fractions}
    raise AssertionError(f"stratification '{stratification_id}' not found")


class TestUpdateStratificationFractions:
    def test_writing_one_of_two_categories_fills_the_other(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions({"group1": 0.25})

        assert _fractions_of(model, "group") == {"group1": 0.25, "group2": 0.75}

    def test_writing_the_other_category_is_symmetric(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions({"group2": 0.25})

        assert _fractions_of(model, "group") == {"group1": 0.75, "group2": 0.25}

    def test_all_categories_of_a_ternary_stratification_can_be_set(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions({"low": 0.2, "mid": 0.3, "high": 0.5})

        assert _fractions_of(model, "tier") == {"low": 0.2, "mid": 0.3, "high": 0.5}

    def test_one_omitted_category_takes_the_remainder(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions({"low": 0.2, "mid": 0.3})

        assert _fractions_of(model, "tier") == {"low": 0.2, "mid": 0.3, "high": 0.5}

    def test_several_stratifications_update_in_one_call(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions({"group1": 0.25, "low": 0.1, "mid": 0.4})

        assert _fractions_of(model, "group") == {"group1": 0.25, "group2": 0.75}
        assert _fractions_of(model, "tier") == {"low": 0.1, "mid": 0.4, "high": 0.5}

    def test_fractions_still_sum_to_one(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions(
            {"group1": 0.123, "low": 0.077, "mid": 0.401}
        )

        assert sum(_fractions_of(model, "group").values()) == pytest.approx(1.0)
        assert sum(_fractions_of(model, "tier").values()) == pytest.approx(1.0)

    def test_two_omitted_categories_are_rejected(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="left unnamed"):
            model.update_stratification_fractions({"low": 0.4})

        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}

    def test_fully_specified_values_must_sum_to_one(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="must sum to 1.0"):
            model.update_stratification_fractions({"low": 0.2, "mid": 0.2, "high": 0.2})

    def test_values_exceeding_one_are_rejected(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="negative fraction"):
            model.update_stratification_fractions({"low": 0.7, "mid": 0.6})

    def test_unknown_category_is_rejected_and_lists_available(self) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="Category 'nope' not found") as excinfo:
            model.update_stratification_fractions({"nope": 0.4})

        assert "group1" in str(excinfo.value)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_out_of_range_fraction_is_rejected(self, value: float) -> None:
        model = _stratified_model()

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            model.update_stratification_fractions({"group1": value})

    def test_boundary_values_are_allowed(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions({"group1": 0.0})

        assert _fractions_of(model, "group") == {"group1": 0.0, "group2": 1.0}

    def test_untouched_stratifications_keep_their_fractions(self) -> None:
        model = _stratified_model()

        model.update_stratification_fractions({"group1": 0.25})

        assert _fractions_of(model, "tier") == {"low": 0.5, "mid": 0.3, "high": 0.2}
