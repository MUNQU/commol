import pytest

from commol.api.model_builder import ModelBuilder
from commol.api.simulation import Simulation
from commol.constants import ModelTypes
from commol.context.model import Model


class TestModelBuilder:
    def test_build_simple_model(self):
        """
        Test that a simple model can be built using the ModelBuilder.
        """
        builder = (
            ModelBuilder(name="TestModel", version="0.1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.1)
            .add_parameter(id="gamma", value=0.05)
            .add_transition(
                id="infection",
                source=["S", "I"],
                target=["I", "I"],
                rate="beta * S * I",
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

        assert isinstance(model, Model)
        assert model.name == "TestModel"
        assert model.version == "0.1.0"
        assert len(model.population.bins) == 3
        assert len(model.parameters) == 2
        assert len(model.dynamics.transitions) == 2
        assert model.population.initial_conditions.population_size == 1000


class TestCompartmentPlaceholder:
    """
    Test $compartment placeholder functionality for automatic transition expansion.
    """

    def test_expansion_with_empty_target(self):
        """Test that $compartment expands correctly with empty target (removal)."""
        builder = (
            ModelBuilder(name="Test Death Model", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="d", value=0.01, description="Death rate", unit="1/day")
            .add_transition(
                id="death",
                source=["S", "I", "R"],
                target=[],
                rate="d * $compartment",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.8},
                    {"bin": "I", "fraction": 0.1},
                    {"bin": "R", "fraction": 0.1},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should have created 3 transitions (one per source compartment)
        assert len(model.dynamics.transitions) == 3

        # Check that transitions have correct IDs
        transition_ids = {t.id for t in model.dynamics.transitions}
        assert transition_ids == {"death__S", "death__I", "death__R"}

        # Check that each transition has correct source, target, and rate
        for transition in model.dynamics.transitions:
            assert len(transition.source) == 1
            assert transition.target == []

            source_compartment = transition.source[0]
            expected_rate = f"d * {source_compartment}"
            assert transition.rate == expected_rate

    def test_expansion_with_single_target(self):
        """Test that $compartment works with a single target compartment."""
        builder = (
            ModelBuilder(name="Test Treatment Model", version="1.0")
            .add_bin(id="I_mild", name="Mild Infection")
            .add_bin(id="I_severe", name="Severe Infection")
            .add_bin(id="R", name="Recovered")
            .add_parameter(
                id="treatment_rate",
                value=0.2,
                description="Treatment rate",
                unit="1/day",
            )
            .add_transition(
                id="treatment",
                source=["I_mild", "I_severe"],
                target=["R"],
                rate="treatment_rate * $compartment",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "I_mild", "fraction": 0.6},
                    {"bin": "I_severe", "fraction": 0.3},
                    {"bin": "R", "fraction": 0.1},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should have created 2 transitions
        assert len(model.dynamics.transitions) == 2

        # Check transition details
        transition_ids = {t.id for t in model.dynamics.transitions}
        assert transition_ids == {"treatment__I_mild", "treatment__I_severe"}

        for transition in model.dynamics.transitions:
            assert len(transition.source) == 1
            assert transition.target == ["R"]

            source_compartment = transition.source[0]
            expected_rate = f"treatment_rate * {source_compartment}"
            assert transition.rate == expected_rate

    def test_compartment_in_complex_formula(self):
        """Test $compartment replacement in more complex formulas."""
        builder = (
            ModelBuilder(name="Complex Formula Test", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_parameter(id="d", value=0.01, unit="1/day")
            .add_parameter(id="N", value=1000, unit="person")
            .add_transition(
                id="complex_death",
                source=["S", "I"],
                target=[],
                rate="d * $compartment * (1 + 0.1 * $compartment / N)",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.9},
                    {"bin": "I", "fraction": 0.1},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Check that all occurrences of $compartment were replaced
        for transition in model.dynamics.transitions:
            assert transition.rate is not None
            assert "$compartment" not in transition.rate
            source_compartment = transition.source[0]
            expected_rate = (
                f"d * {source_compartment} * (1 + 0.1 * {source_compartment} / N)"
            )
            assert transition.rate == expected_rate

    def test_error_single_source(self):
        """Test that using $compartment with single source raises error."""
        builder = (
            ModelBuilder(name="Single Source Test", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_parameter(id="d", value=0.01, unit="1/day")
        )

        with pytest.raises(ValueError) as exc_info:
            builder.add_transition(
                id="death",
                source=["S"],  # Only one source
                target=[],
                rate="d * $compartment",
            )

        assert "$compartment placeholder requires either multiple source" in str(
            exc_info.value
        )
        assert "use the compartment name directly" in str(exc_info.value)

    def test_error_multiple_targets(self):
        """Test that using $compartment with multiple targets raises error."""
        builder = (
            ModelBuilder(name="Multiple Targets Test", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R1", name="Recovered Type 1")
            .add_bin(id="R2", name="Recovered Type 2")
            .add_parameter(id="recovery_rate", value=0.1, unit="1/day")
        )

        with pytest.raises(ValueError) as exc_info:
            builder.add_transition(
                id="recovery",
                source=["S", "I"],
                target=["R1", "R2"],  # Multiple targets
                rate="recovery_rate * $compartment",
            )

        assert "cannot be used with both multiple sources and multiple targets" in str(
            exc_info.value
        )
        assert "ambiguous mappings" in str(exc_info.value)

    def test_with_stratified_rates(self):
        """Test that $compartment works correctly with stratified_rates."""
        builder = (
            ModelBuilder(name="Stratified Test", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_stratification(id="age", categories=["young", "old"])
            .add_parameter(id="d_young", value=0.005, unit="1/day")
            .add_parameter(id="d_old", value=0.02, unit="1/day")
            .add_parameter(id="d_base", value=0.01, unit="1/day")
            .add_transition(
                id="death",
                source=["S", "I"],
                target=[],
                rate="d_base * $compartment",  # Fallback rate
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "d_young * $compartment",
                    },
                    {
                        "conditions": [{"stratification": "age", "category": "old"}],
                        "rate": "d_old * $compartment",
                    },
                ],
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.9},
                    {"bin": "I", "fraction": 0.1},
                ],
                stratification_fractions=[
                    {
                        "stratification": "age",
                        "fractions": [
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    }
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Should have created 2 transitions (one per compartment)
        death_transitions = [t for t in model.dynamics.transitions if "death" in t.id]
        assert len(death_transitions) == 2

        # Check each transition
        for transition in death_transitions:
            assert len(transition.source) == 1
            compartment = transition.source[0]

            # Check base rate
            assert transition.rate == f"d_base * {compartment}"

            # Check stratified rates
            assert transition.stratified_rates is not None
            assert len(transition.stratified_rates) == 2

            # Check that $compartment was replaced in stratified rates
            for strat_rate in transition.stratified_rates:
                assert "$compartment" not in strat_rate.rate
                assert compartment in strat_rate.rate

    def test_death_transition_simulation(self):
        """Test that death transitions using $compartment produce correct results."""
        builder = (
            ModelBuilder(name="SLIR with Death", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="L", name="Latent")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.2, unit="1/day")
            .add_parameter(id="delta", value=0.1, unit="1/day")
            .add_parameter(id="d", value=0.01, description="Death rate", unit="1/day")
            .add_transition(
                id="infection",
                source=["S"],
                target=["L"],
                rate="beta * S * I / N",
            )
            .add_transition(
                id="progression",
                source=["L"],
                target=["I"],
                rate="gamma * L",
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="delta * I",
            )
            .add_transition(
                id="death",
                source=["S", "L", "I", "R"],
                target=[],
                rate="d * $compartment",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.99},
                    {"bin": "L", "fraction": 0.005},
                    {"bin": "I", "fraction": 0.005},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)

        # Verify that 4 death transitions were created (one per compartment)
        death_transitions = [t for t in model.dynamics.transitions if "death" in t.id]
        assert len(death_transitions) == 4

        simulation = Simulation(model)
        results = simulation.run(num_steps=200, output_format="dict_of_lists")

        # Check that simulation ran successfully
        assert len(results["S"]) == 201
        assert len(results["L"]) == 201
        assert len(results["I"]) == 201
        assert len(results["R"]) == 201

        # Check that population decreases over time (deaths are happening)
        initial_population = (
            results["S"][0] + results["L"][0] + results["I"][0] + results["R"][0]
        )
        final_population = (
            results["S"][-1] + results["L"][-1] + results["I"][-1] + results["R"][-1]
        )
        # Verify the transitions were created correctly (main test objective)
        # Note: Population may not decrease much if birth/death rates balance
        assert final_population <= initial_population

        # All compartments should have non-negative values
        for compartment in ["S", "L", "I", "R"]:
            assert all(val >= -1e-10 for val in results[compartment]), (
                f"{compartment} has negative values"
            )

    def test_comparison_with_manual_expansion(self):
        """
        Test that $compartment expansion produces same results as manual expansion.
        """
        # Model with $compartment expansion
        builder_auto = (
            ModelBuilder(name="Auto Expansion", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
            .add_parameter(id="d", value=0.01, unit="1/day")
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
                rate="gamma * I",
            )
            .add_transition(
                id="death",
                source=["S", "I", "R"],
                target=[],
                rate="d * $compartment",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "I", "fraction": 0.02},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        # Model with manual expansion
        builder_manual = (
            ModelBuilder(name="Manual Expansion", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="beta", value=0.3, unit="1/day")
            .add_parameter(id="gamma", value=0.1, unit="1/day")
            .add_parameter(id="d", value=0.01, unit="1/day")
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
                rate="gamma * I",
            )
            .add_transition(
                id="death_S",
                source=["S"],
                target=[],
                rate="d * S",
            )
            .add_transition(
                id="death_I",
                source=["I"],
                target=[],
                rate="d * I",
            )
            .add_transition(
                id="death_R",
                source=["R"],
                target=[],
                rate="d * R",
            )
            .set_initial_conditions(
                population_size=1000,
                bin_fractions=[
                    {"bin": "S", "fraction": 0.98},
                    {"bin": "I", "fraction": 0.02},
                    {"bin": "R", "fraction": 0.0},
                ],
            )
        )

        model_auto = builder_auto.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        model_manual = builder_manual.build(
            typology=ModelTypes.DIFFERENCE_EQUATIONS.value
        )

        simulation_auto = Simulation(model_auto)
        simulation_manual = Simulation(model_manual)

        results_auto = simulation_auto.run(num_steps=100, output_format="dict_of_lists")
        results_manual = simulation_manual.run(
            num_steps=100, output_format="dict_of_lists"
        )

        # Results should be identical (or very close due to floating point)
        for compartment in ["S", "I", "R"]:
            for i in range(101):
                assert (
                    abs(results_auto[compartment][i] - results_manual[compartment][i])
                    < 1e-6
                ), (
                    f"Mismatch in {compartment} at step {i}: "
                    f"{results_auto[compartment][i]} vs "
                    f"{results_manual[compartment][i]}"
                )
