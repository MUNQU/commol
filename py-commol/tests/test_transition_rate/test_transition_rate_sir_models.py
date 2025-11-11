import math

import pytest

from commol.api.model_builder import ModelBuilder
from commol.api.simulation import Simulation
from commol.constants import ModelTypes


class TestBasicSIRRateFormulas:
    """Test basic SIR models with various rate formula types."""

    def create_base_sir_builder(self, name: str = "Test SIR") -> ModelBuilder:
        """Create a basic SIR model builder without transitions."""
        return (
            ModelBuilder(name=name, version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

    def test_constant_rate_sir_model(self):
        """Test SIR model with constant transition rates."""
        try:
            builder = (
                self.create_base_sir_builder("Constant Rate SIR")
                .add_parameter(id="beta", value=0.3)
                .add_parameter(id="gamma", value=0.1)
                .add_transition(id="infection", source=["S"], target=["I"], rate=0.3)
                .add_transition(id="recovery", source=["I"], target=["R"], rate=0.1)
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_parameter_reference_rates(self):
        """Test SIR model with parameter references as rates."""
        try:
            builder = (
                self.create_base_sir_builder("Parameter Reference SIR")
                .add_parameter(id="transmission_rate", value=0.25)
                .add_parameter(id="recovery_rate", value=0.05)
                .add_transition(
                    id="infection", source=["S"], target=["I"], rate="transmission_rate"
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="recovery_rate"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_mass_action_kinetics(self):
        """Test SIR model with mass action kinetics (S*I interaction)."""
        try:
            builder = (
                self.create_base_sir_builder("Mass Action SIR")
                .add_parameter(id="beta", value=0.0003)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="infection", source=["S"], target=["I"], rate="beta * S * I"
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_frequency_dependent_transmission(self):
        """Test SIR model with frequency-dependent transmission (S*I/N)."""
        try:
            builder = (
                self.create_base_sir_builder("Frequency Dependent SIR")
                .add_parameter(id="beta", value=0.3)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="infection",
                    source=["S"],
                    target=["I"],
                    rate="beta * S * I / N",
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")


class TestStepVaryingRates:
    """Test SIR models with step-varying transition rates."""

    def create_base_sir_builder(self) -> ModelBuilder:
        """Create a basic SIR model builder."""
        return (
            ModelBuilder(name="Time Varying SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

    def test_seasonal_transmission(self):
        """Test SIR model with seasonal variation in transmission."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta_avg", value=0.25)
                .add_parameter(id="seasonal_amp", value=0.2)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="seasonal_infection",
                    source=["S"],
                    target=["I"],
                    rate=(
                        "(beta_avg * (1 + seasonal_amp * sin(2 * pi * step / 365))) "
                        "* S * I / N"
                    ),
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_exponential_decay_transmission(self):
        """Test SIR model with exponentially decaying transmission rate."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta_0", value=0.4)
                .add_parameter(id="decay_rate", value=0.01)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="decaying_infection",
                    source=["S"],
                    target=["I"],
                    rate="beta_0 * exp(-decay_rate * step) * S * I / N",
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_step_function_intervention(self):
        """Test SIR model with step function representing intervention."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta_pre", value=0.3)
                .add_parameter(id="beta_post", value=0.1)
                .add_parameter(id="intervention_time", value=5)
                .add_parameter(id="gamma", value=0.05)
                # Using sigmoid approximation for step function
                .add_transition(
                    id="intervention_infection",
                    source=["S"],
                    target=["I"],
                    rate=(
                        "(beta_pre + (beta_post - beta_pre) * "
                        "(1 / (1 + exp(-10 * (t - intervention_time))))) * S * I / N"
                    ),
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")


class TestComplexMathematicalFormulas:
    """Test SIR models with complex mathematical formulations."""

    def create_base_sir_builder(self) -> ModelBuilder:
        """Create a basic SIR model builder."""
        return (
            ModelBuilder(name="Complex Math SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

    def test_logistic_transmission_rate(self):
        """Test SIR model with logistic function for transmission rate."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="max_beta", value=0.4)
                .add_parameter(id="steepness", value=0.1)
                .add_parameter(id="midpoint", value=30)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="logistic_infection",
                    source=["S"],
                    target=["I"],
                    rate=(
                        "max_beta * (1 / (1 + exp(-steepness * (step - midpoint))))"
                        " * S * I / N"
                    ),
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_power_law_recovery_rate(self):
        """Test SIR model with power law recovery rate."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta", value=0.25)
                .add_parameter(id="gamma_0", value=0.05)
                .add_parameter(id="power", value=0.5)
                .add_transition(
                    id="infection",
                    source=["S"],
                    target=["I"],
                    rate="beta * S * I / N",
                )
                .add_transition(
                    id="power_recovery",
                    source=["I"],
                    target=["R"],
                    rate="gamma_0 * pow(I, power)",
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_trigonometric_rate_combinations(self):
        """Test SIR model with combinations of trigonometric functions."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta_base", value=0.2)
                .add_parameter(id="amplitude_1", value=0.1)
                .add_parameter(id="amplitude_2", value=0.05)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="complex_seasonal_infection",
                    source=["S"],
                    target=["I"],
                    rate=(
                        "(beta_base + amplitude_1 * sin(2 * pi * step / 365) + "
                        "amplitude_2 * cos(4 * pi * step / 365)) * S * I / N"
                    ),
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            _ = simulation.run(10)

            # TODO: Check model dynamics (valid with only one output type)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")


class TestMathematicalEdgeCases:
    """Test edge cases and boundary conditions in mathematical formulas."""

    def create_base_sir_builder(self) -> ModelBuilder:
        """Create a basic SIR model builder."""
        return (
            ModelBuilder(name="Edge Case SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

    def test_very_small_rates(self):
        """Test SIR model with very small transition rates."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="tiny_beta", value=1e-6)
                .add_parameter(id="tiny_gamma", value=1e-7)
                .add_transition(
                    id="slow_infection",
                    source=["S"],
                    target=["I"],
                    rate="tiny_beta * S * I / N",
                )
                .add_transition(
                    id="slow_recovery",
                    source=["I"],
                    target=["R"],
                    rate="tiny_gamma * I",
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            results = simulation.run(1000)  # Longer simulation for small rates

            # Should handle small rates without numerical issues
            assert len(results["S"]) == 1001

            # Population should be conserved
            total = results["S"][-1] + results["I"][-1] + results["R"][-1]
            assert math.isclose(total, 1000, rel_tol=1e-3)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_zero_rate_scenarios(self):
        """Test SIR model with zero rates in some transitions."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta", value=0.2)
                .add_transition(
                    id="infection",
                    source=["S"],
                    target=["I"],
                    rate="beta * S * I / N",
                )
                # No recovery transition (zero recovery rate)
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            results = simulation.run(50)

            # Should handle zero recovery rate
            assert len(results["I"]) == 51
            # R should remain zero throughout
            assert all(r == 0.0 for r in results["R"])
            # S + I should equal total population
            for i in range(51):
                assert math.isclose(
                    results["S"][i] + results["I"][i], 1000, rel_tol=1e-6
                )

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_max_min_functions_in_rates(self):
        """Test SIR model using max/min functions in transition rates."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta_max", value=0.4)
                .add_parameter(id="beta_min", value=0.1)
                .add_parameter(id="threshold", value=100)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="threshold_infection",
                    source=["S"],
                    target=["I"],
                    rate=(
                        "max(beta_min, min(beta_max, beta_max * (1 - I / threshold)))"
                        " * S * I / N"
                    ),
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            results = simulation.run(100)

            # Should complete without mathematical errors
            assert len(results["I"]) == 101

            # All values should be non-negative
            for i in range(101):
                assert results["S"][i] >= 0
                assert results["I"][i] >= 0
                assert results["R"][i] >= 0

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_floor_ceil_round_functions(self):
        """Test SIR model with floor, ceil, and round functions."""
        try:
            builder = (
                self.create_base_sir_builder()
                .add_parameter(id="beta", value=0.25)
                .add_parameter(id="gamma_base", value=0.05)
                .add_parameter(id="step_scale", value=10)
                .add_transition(
                    id="infection",
                    source=["S"],
                    target=["I"],
                    rate="beta * S * I / N",
                )
                .add_transition(
                    id="stepped_recovery",
                    source=["I"],
                    target=["R"],
                    rate="gamma_base * (1 + 0.1 * floor(step / step_scale)) * I",
                )
            )

            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            simulation = Simulation(model)
            results = simulation.run(50)

            # Should handle discrete functions
            assert len(results["R"]) == 51
            assert results["R"][-1] > results["R"][0]  # Recovery should increase

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")


class TestRateFormulaErrorHandling:
    """Test error handling and edge cases in rate formula processing."""

    def test_invalid_mathematical_expressions(self):
        """Test that invalid mathematical expressions are properly rejected."""
        invalid_formulas = [
            "S * I /",  # Incomplete expression
            "beta * * gamma",  # Invalid operator sequence
            "sin()",  # Function with no arguments
            "S +",  # Incomplete addition
            "/ beta",  # Leading division operator
        ]

        for formula in invalid_formulas:
            with pytest.raises((ValueError, SyntaxError)):
                # This should fail during validation or model building
                builder = (
                    ModelBuilder(name="Invalid Formula Test", version="1.0")
                    .add_bin(id="S", name="Susceptible")
                    .add_bin(id="I", name="Infected")
                    .add_bin(id="R", name="Recovered")
                    .add_parameter(id="beta", value=0.1)
                    .add_parameter(id="gamma", value=0.05)
                    .add_transition(
                        id="invalid_transition",
                        source=["S"],
                        target=["I"],
                        rate=formula,
                    )
                    .set_initial_conditions(
                        population_size=1000,
                        bin_fractions=[
                            {"bin": "S", "fraction": 1.0},
                            {"bin": "I", "fraction": 0.0},
                            {"bin": "R", "fraction": 0.0},
                        ],
                    )
                )
                _ = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

    def test_undefined_parameters_in_formulas(self):
        """Test handling of undefined parameters in formulas."""
        # This test checks if undefined parameters are caught
        try:
            builder = (
                ModelBuilder(name="Undefined Param Test", version="1.0")
                .add_bin(id="S", name="Susceptible")
                .add_bin(id="I", name="Infected")
                .add_bin(id="R", name="Recovered")
                .add_parameter(id="beta", value=0.1)
                # Note: gamma parameter is NOT defined
                .add_transition(
                    id="infection", source=["S"], target=["I"], rate="beta * S * I"
                )
                .add_transition(
                    id="recovery",
                    source=["I"],
                    target=["R"],
                    rate="undefined_gamma * I",  # This parameter doesn't exist
                )
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
            simulation = Simulation(model)

            # TODO: Check
            # This might succeed at build time but fail at simulation time
            # depending on when parameter validation occurs
            with pytest.raises((ValueError, KeyError, RuntimeError)):
                _ = simulation.run(10)

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")
        except (ValueError, KeyError, RuntimeError):
            # Expected behavior - undefined parameter should be caught
            pass

    def test_division_by_zero_scenarios(self):
        """Test handling of potential division by zero in formulas."""
        try:
            # Create a model that might have division by zero issues
            builder = (
                ModelBuilder(name="Division Test", version="1.0")
                .add_bin(id="S", name="Susceptible")
                .add_bin(id="I", name="Infected")
                .add_bin(id="R", name="Recovered")
                .add_parameter(id="beta", value=0.1)
                .add_parameter(id="gamma", value=0.05)
                .add_transition(
                    id="infection",
                    source=["S"],
                    target=["I"],
                    # Avoid division by zero
                    rate="beta * S * I / max(1, S + I + R)",
                )
                .add_transition(
                    id="recovery", source=["I"], target=["R"], rate="gamma * I"
                )
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
            simulation = Simulation(model)
            results = simulation.run(10)

            # Should complete without division by zero errors
            assert len(results["S"]) == 11

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")


class TestMathematicalFunctionComprehensive:
    """Comprehensive tests for all supported mathematical functions in rate formulas."""

    def create_function_test_sir_builder(self) -> ModelBuilder:
        """Create base SIR builder for function testing."""
        return (
            ModelBuilder(name="Function Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="base_rate", value=0.1)
            .add_parameter(id="scale", value=100)
            .add_parameter(id="offset", value=10)
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "I", "fraction": 0.01},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

    def test_exponential_and_logarithmic_functions(self):
        """Test exponential and logarithmic functions in rate formulas."""
        test_cases = [
            ("exp", "base_rate * exp(-I / scale) * S * I / N"),
            ("log", "base_rate * log(1 + I / offset) * S * I / N"),
            ("log10", "base_rate * log10(1 + I / offset) * S * I / N"),
            ("log2", "base_rate * log2(1 + I / offset) * S * I / N"),
            ("ln", "base_rate * ln(1 + I / offset) * S * I / N"),
        ]

        try:
            for func_name, formula in test_cases:
                builder = (
                    self.create_function_test_sir_builder()
                    .add_transition(
                        id=f"{func_name}_infection",
                        source=["S"],
                        target=["I"],
                        rate=formula,
                    )
                    .add_transition(
                        id="recovery",
                        source=["I"],
                        target=["R"],
                        rate="base_rate * I",
                    )
                )

                model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
                simulation = Simulation(model)
                results = simulation.run(50)

                # Should complete without mathematical errors
                assert len(results["S"]) == 51, f"{func_name} function failed"

                # Check for reasonable behavior
                assert all(values >= -1e-10 for values in results["S"]), (
                    f"Negative S with {func_name}"
                )
                assert all(values >= -1e-10 for values in results["I"]), (
                    f"Negative I with {func_name}"
                )
                assert all(values >= -1e-10 for values in results["R"]), (
                    f"Negative R with {func_name}"
                )

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_trigonometric_functions(self):
        """Test trigonometric functions in rate formulas."""
        trig_functions = [
            ("sin", "base_rate * (1 + 0.1 * sin(I / scale)) * S * I / N"),
            ("cos", "base_rate * (1 + 0.1 * cos(I / scale)) * S * I / N"),
            ("tan", "base_rate * (1 + 0.01 * tan(I / (10 * scale))) * S * I / N"),
        ]

        try:
            for func_name, formula in trig_functions:
                builder = (
                    self.create_function_test_sir_builder()
                    .add_transition(
                        id=f"{func_name}_infection",
                        source=["S"],
                        target=["I"],
                        rate=formula,
                    )
                    .add_transition(
                        id="recovery",
                        source=["I"],
                        target=["R"],
                        rate="base_rate * I",
                    )
                )

                model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
                simulation = Simulation(model)
                results = simulation.run(30)

                assert len(results["S"]) == 31, f"{func_name} function failed"

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_hyperbolic_functions(self):
        """Test hyperbolic functions in rate formulas."""
        hyperbolic_functions = [
            ("sinh", "base_rate * (1 + 0.01 * sinh(I / scale)) * S * I / N"),
            ("cosh", "base_rate * cosh(I / scale) * S * I / N"),
            ("tanh", "base_rate * tanh(I / scale) * S * I / N"),
        ]

        try:
            for func_name, formula in hyperbolic_functions:
                builder = (
                    self.create_function_test_sir_builder()
                    .add_transition(
                        id=f"{func_name}_infection",
                        source=["S"],
                        target=["I"],
                        rate=formula,
                    )
                    .add_transition(
                        id="recovery",
                        source=["I"],
                        target=["R"],
                        rate="base_rate * I",
                    )
                )

                model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
                simulation = Simulation(model)
                results = simulation.run(30)

                assert len(results["S"]) == 31, f"{func_name} function failed"

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_utility_mathematical_functions(self):
        """Test utility mathematical functions like abs, pow, sqrt, etc."""
        utility_functions = [
            ("abs", "base_rate * abs(I - offset) * S / N"),
            ("sqrt", "base_rate * sqrt(I + 1) * S / N"),
            ("pow", "base_rate * pow(I + 1, 0.5) * S / N"),
            ("floor", "base_rate * (1 + floor(I / offset) * 0.01) * S * I / N"),
            ("ceil", "base_rate * (1 + ceil(I / offset) * 0.01) * S * I / N"),
            ("round", "base_rate * (1 + round(I / offset) * 0.01) * S * I / N"),
        ]

        try:
            for func_name, formula in utility_functions:
                builder = (
                    self.create_function_test_sir_builder()
                    .add_transition(
                        id=f"{func_name}_infection",
                        source=["S"],
                        target=["I"],
                        rate=formula,
                    )
                    .add_transition(
                        id="recovery",
                        source=["I"],
                        target=["R"],
                        rate="base_rate * I",
                    )
                )

                model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
                simulation = Simulation(model)
                results = simulation.run(30)

                assert len(results["S"]) == 31, f"{func_name} function failed"

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")

    def test_inverse_trigonometric_functions(self):
        """Test inverse trigonometric functions in rate formulas."""
        inverse_trig_functions = [
            (
                "asin",
                "base_rate * asin(I / (10 * scale)) * S / N",
            ),  # Ensure input is in [-1,1]
            (
                "acos",
                "base_rate * acos(I / (10 * scale)) * S / N",
            ),  # Ensure input is in [-1,1]
            ("atan", "base_rate * atan(I / scale) * S / N"),
            ("atan2", "base_rate * atan2(I, S) * I / N"),
        ]

        try:
            for func_name, formula in inverse_trig_functions:
                builder = (
                    self.create_function_test_sir_builder()
                    .add_transition(
                        id=f"{func_name}_infection",
                        source=["S"],
                        target=["I"],
                        rate=formula,
                    )
                    .add_transition(
                        id="recovery",
                        source=["I"],
                        target=["R"],
                        rate="base_rate * I",
                    )
                )

                model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
                simulation = Simulation(model)
                results = simulation.run(20)

                assert len(results["S"]) == 21, f"{func_name} function failed"

        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")
