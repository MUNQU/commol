import tempfile
import sys
from io import StringIO
from pathlib import Path

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
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Disease State Transitions:\n"
            "Infection (S -> I):\n"
            "  S -> I: beta * S * I / N\n"
            "\n"
            "Recovery (I -> R):\n"
            "  I -> R: gamma * I\n"
            "\n"
            "Total System: 3 coupled equations (3 disease states)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "dS/dt = - (beta * S * I / N)\n"
            "dI/dt = (beta * S * I / N) - (gamma * I)\n"
            "dR/dt = (gamma * I)\n"
        )
        assert output == expected_output

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
                "COMPACT FORM\n"
                "========================================\n"
                "\n"
                "Disease State Transitions:\n"
                "Exposure (S -> E):\n"
                "  S -> E: beta * S * I / N\n"
                "\n"
                "Infection (E -> I):\n"
                "  E -> I: sigma * E\n"
                "\n"
                "Recovery (I -> R):\n"
                "  I -> R: gamma * I\n"
                "\n"
                "Total System: 4 coupled equations (4 disease states)\n"
                "\n"
                "========================================\n"
                "EXPANDED FORM\n"
                "========================================\n"
                "dS/dt = - (beta * S * I / N)\n"
                "dE/dt = (beta * S * I / N) - (sigma * E)\n"
                "dI/dt = (sigma * E) - (gamma * I)\n"
                "dR/dt = (gamma * I)"
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
            "Disease State Transitions:\n"
            "Infection (S -> I):\n"
            "  S_young -> I_young: beta * S * I / N\n"
            "  S_old -> I_old: beta * S * I / N\n"
            "\n"
            "Recovery (I -> R):\n"
            "  I_young -> R_young: gamma * I\n"
            "  I_old -> R_old: gamma * I\n"
            "\n"
            "Age Stratification Transitions (young -> old):\n"
            "For each disease state X in {S, I, R}:\n"
            "  X_young -> X_old: aging_rate * X_young\n"
            "\n"
            "Total System: 6 coupled equations (3 disease states × 2 age)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "dS_young/dt = - (beta * S * I / N) - (aging_rate * S_young)\n"
            "dS_old/dt = - (beta * S * I / N) + (aging_rate * S_young)\n"
            "dI_young/dt = (beta * S * I / N) - (gamma * I) - (aging_rate * I_young)\n"
            "dI_old/dt = (beta * S * I / N) - (gamma * I) + (aging_rate * I_young)\n"
            "dR_young/dt = (gamma * I) - (aging_rate * R_young)\n"
            "dR_old/dt = (gamma * I) + (aging_rate * R_young)\n"
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
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Disease State Transitions:\n"
            "Infection (S -> I):\n"
            "  S -> I: None\n"
            "\n"
            "Recovery (I -> R):\n"
            "  I -> R: gamma * I\n"
            "\n"
            "Total System: 3 coupled equations (3 disease states)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "dS/dt = 0\n"
            "dI/dt = - (gamma * I)\n"
            "dR/dt = (gamma * I)\n"
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
            "Model: Age-Location Stratified SIR\n"
            "Model Type: DifferenceEquations\n"
            "Number of Disease States: 3\n"
            "Number of Stratifications: 2\n"
            "Number of Parameters: 7\n"
            "Number of Transitions: 4\n"
            "Disease States: S, I, R\n"
            "Stratifications:\n"
            "  - age: [young, old]\n"
            "  - location: [urban, rural]\n"
            "\n"
            "========================================\n"
            "COMPACT FORM\n"
            "========================================\n"
            "\n"
            "Disease State Transitions:\n"
            "Infection (S -> I):\n"
            "  S_young_urban -> I_young_urban: beta_young_urban * S * I / N_young\n"
            "  S_young_rural -> I_young_rural: beta_young_rural * S * I / N_young\n"
            "  S_old_urban -> I_old_urban: beta_old_urban * S * I / N_old\n"
            "  S_old_rural -> I_old_rural: 0.3 * S * I / N\n"
            "\n"
            "Recovery (I -> R):\n"
            "  I_young_urban -> R_young_urban: gamma_young * I\n"
            "  I_young_rural -> R_young_rural: gamma_young * I\n"
            "  I_old_urban -> R_old_urban: 0.1 * I\n"
            "  I_old_rural -> R_old_rural: 0.1 * I\n"
            "\n"
            "Age Stratification Transitions (young -> old):\n"
            "For each disease state X in {S, I, R} and each location "
            "in {urban, rural}:\n"
            "  X_young_urban -> X_old_urban: aging_rate * X_young_urban\n"
            "  X_young_rural -> X_old_rural: aging_rate * X_young_rural\n"
            "\n"
            "Location Stratification Transitions (urban -> rural):\n"
            "For each disease state X in {S, I, R} and each age in {young, old}:\n"
            "  X_young_urban -> X_young_rural: migration_rate_young * X_young_urban\n"
            "  X_old_urban -> X_old_rural: migration_rate_old * X_old_urban\n"
            "\n"
            "Total System: 12 coupled equations "
            "(3 disease states × 2 age × 2 location)\n"
            "\n"
            "========================================\n"
            "EXPANDED FORM\n"
            "========================================\n"
            "dS_young_urban/dt = - (beta_young_urban * S * I / N_young) "
            "- (aging_rate * S_young_urban) "
            "- (migration_rate_young * S_young_urban)\n"
            "dS_young_rural/dt = - (beta_young_rural * S * I / N_young) "
            "- (aging_rate * S_young_rural) "
            "+ (migration_rate_young * S_young_urban)\n"
            "dS_old_urban/dt = - (beta_old_urban * S * I / N_old) "
            "+ (aging_rate * S_young_urban) "
            "- (migration_rate_old * S_old_urban)\n"
            "dS_old_rural/dt = - (0.3 * S * I / N) + (aging_rate * S_young_rural) "
            "+ (migration_rate_old * S_old_urban)\n"
            "dI_young_urban/dt = (beta_young_urban * S * I / N_young) "
            "- (gamma_young * I) - (aging_rate * I_young_urban) "
            "- (migration_rate_young * I_young_urban)\n"
            "dI_young_rural/dt = (beta_young_rural * S * I / N_young) "
            "- (gamma_young * I) - (aging_rate * I_young_rural) "
            "+ (migration_rate_young * I_young_urban)\n"
            "dI_old_urban/dt = (beta_old_urban * S * I / N_old) - (0.1 * I) "
            "+ (aging_rate * I_young_urban) - (migration_rate_old * I_old_urban)\n"
            "dI_old_rural/dt = (0.3 * S * I / N) - (0.1 * I) "
            "+ (aging_rate * I_young_rural) + (migration_rate_old * I_old_urban)\n"
            "dR_young_urban/dt = (gamma_young * I) - (aging_rate * R_young_urban) "
            "- (migration_rate_young * R_young_urban)\n"
            "dR_young_rural/dt = (gamma_young * I) - (aging_rate * R_young_rural) "
            "+ (migration_rate_young * R_young_urban)\n"
            "dR_old_urban/dt = (0.1 * I) + (aging_rate * R_young_urban) "
            "- (migration_rate_old * R_old_urban)\n"
            "dR_old_rural/dt = (0.1 * I) + (aging_rate * R_young_rural) "
            "+ (migration_rate_old * R_old_urban)\n"
        )
        assert output == expected_output
