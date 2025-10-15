import tempfile
import sys
from io import StringIO
from pathlib import Path

from epimodel.api.model_builder import ModelBuilder
from epimodel.constants import ModelTypes


class TestModel:
    def test_print_equations_to_console(self):
        """
        Test that print_equations outputs to console correctly in mathematical form.
        """
        builder = (
            ModelBuilder(name="SIR Model", version="1.0.0")
            .add_disease_state(id="S", name="Susceptible")
            .add_disease_state(id="I", name="Infected")
            .add_disease_state(id="R", name="Recovered")
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
                disease_state_fractions=[
                    {"disease_state": "S", "fraction": 0.99},
                    {"disease_state": "I", "fraction": 0.01},
                    {"disease_state": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

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
            "Number of Disease States: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 2\n"
            "Number of Transitions: 2\n"
            "Disease States: S, I, R\n"
            "\n"
            "========================================\n"
            "EQUATIONS\n"
            "========================================\n"
            "dS/dt = - (beta * S * I / N)\n"
            "dI/dt =  (beta * S * I / N) - (gamma * I)\n"
            "dR/dt =  (gamma * I)\n"
        )
        assert output == expected_output

    def test_print_equations_to_file(self):
        """
        Test that print_equations writes to a file correctly.
        """
        builder = (
            ModelBuilder(name="SEIR Model", version="2.0.0")
            .add_disease_state(id="S", name="Susceptible")
            .add_disease_state(id="E", name="Exposed")
            .add_disease_state(id="I", name="Infected")
            .add_disease_state(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.5)
            .add_parameter(id="sigma", value=0.2)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="exposure",
                source=["S"],
                target=["E"],
                rate="beta * S * I / N",
            )
            .add_transition(
                id="infection", source=["E"], target=["I"], rate="sigma * E"
            )
            .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
            .set_initial_conditions(
                population_size=1000,
                disease_state_fractions=[
                    {"disease_state": "S", "fraction": 0.98},
                    {"disease_state": "E", "fraction": 0.01},
                    {"disease_state": "I", "fraction": 0.01},
                    {"disease_state": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

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
                "Number of Disease States: 4\n"
                "Number of Stratifications: 0\n"
                "Number of Parameters: 3\n"
                "Number of Transitions: 3\n"
                "Disease States: S, E, I, R\n"
                "\n"
                "========================================\n"
                "EQUATIONS\n"
                "========================================\n"
                "dS/dt = - (beta * S * I / N)\n"
                "dE/dt =  (beta * S * I / N) - (sigma * E)\n"
                "dI/dt =  (sigma * E) - (gamma * I)\n"
                "dR/dt =  (gamma * I)"
            )
            assert content == expected_content

    def test_print_equations_with_stratification(self):
        """
        Test that print_equations works with stratifications.
        """
        builder = (
            ModelBuilder(name="Age-Stratified SIR", version="1.0.0")
            .add_disease_state(id="S", name="Susceptible")
            .add_disease_state(id="I", name="Infected")
            .add_disease_state(id="R", name="Recovered")
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
                disease_state_fractions=[
                    {"disease_state": "S", "fraction": 0.99},
                    {"disease_state": "I", "fraction": 0.01},
                    {"disease_state": "R", "fraction": 0.0},
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

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

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
            "Number of Disease States: 3\n"
            "Number of Stratifications: 1\n"
            "Number of Parameters: 3\n"
            "Number of Transitions: 3\n"
            "Disease States: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Stratification Transitions (age):\n"
            "For each disease state X in {S, I, R}:\n"
            "  dX_young/dt: - (aging_rate * X)\n"
            "  dX_old/dt:  (aging_rate * X)\n"
            "\n"
            "Disease State Transitions:\n"
            "For each stratification s in {young, old}:\n"
            "  dS_s/dt: - (beta * S * I / N)\n"
            "  dI_s/dt:  (beta * S * I / N) - (gamma * I)\n"
            "  dR_s/dt:  (gamma * I)\n"
            "\n"
            "Total System: 6 coupled equations (3 disease states × 2 stratification)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "dS/dt = - (beta * S * I / N)\n"
            "dI/dt =  (beta * S * I / N) - (gamma * I)\n"
            "dR/dt =  (gamma * I)\n"
            "dyoung/dt = - (aging_rate)\n"
            "dold/dt =  (aging_rate)\n"
        )
        assert output == expected_output

    def test_print_equations_without_rate(self):
        """
        Test that transitions without rate are properly ignored in equations.
        """
        builder = (
            ModelBuilder(name="Test Model", version="1.0.0")
            .add_disease_state(id="S", name="Susceptible")
            .add_disease_state(id="I", name="Infected")
            .add_disease_state(id="R", name="Recovered")
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
                disease_state_fractions=[
                    {"disease_state": "S", "fraction": 0.99},
                    {"disease_state": "I", "fraction": 0.01},
                    {"disease_state": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

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
            "Number of Disease States: 3\n"
            "Number of Stratifications: 0\n"
            "Number of Parameters: 1\n"
            "Number of Transitions: 2\n"
            "Disease States: S, I, R\n"
            "\n"
            "========================================\n"
            "EQUATIONS\n"
            "========================================\n"
            "dS/dt = - ()\n"
            "dI/dt =  () - (gamma * I)\n"
            "dR/dt =  (gamma * I)\n"
        )
        assert output == expected_output
