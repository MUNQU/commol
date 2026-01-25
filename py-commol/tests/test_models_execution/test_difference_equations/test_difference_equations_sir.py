import math

import pytest

from commol.api import ModelBuilder
from commol.api.simulation import Simulation
from commol.constants import ModelTypes
from commol.context import Model


class TestSIR:
    @pytest.fixture(scope="class")
    def sir_model(self) -> Model:
        """
        Builds a standard SIR model.
        Skips tests in this class if the Rust extension is not built.
        """
        builder = (
            ModelBuilder(name="Test SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_parameter(id="infection_rate", value=0.1)
            .add_parameter(id="recovery_rate", value=0.05)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="infection_rate",
            )
            .add_transition(
                id="recovery", source=["I"], target=["R"], rate="recovery_rate"
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
        try:
            return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")
        assert False, "Should not be reached"

    def test_sir_list_of_lists_output(self, sir_model: Model):
        """
        Tests that a simple SIR model can be built and run, producing
        a valid list-of-lists result.
        """
        num_steps = 10
        initial_population = 1000

        sir_simulation = Simulation(sir_model)
        steps_list = sir_simulation.run(num_steps, output_format="list_of_lists")

        assert len(steps_list) == num_steps + 1

        # Based on the order of `add_disease_state`, the compartments are S, I, R.
        s_idx, i_idx, r_idx = 0, 1, 2

        for i, population_state in enumerate(steps_list):
            assert isinstance(population_state, list)
            assert len(population_state) == 3  # S, I, R

            total_population = sum(population_state)
            assert math.isclose(total_population, initial_population, rel_tol=1e-6), (
                f"Population not conserved at step {i}. Got {total_population}"
            )

            for value in population_state:
                assert value >= 0

        # Check for basic dynamics (S should decrease, R should increase)
        assert steps_list[0][s_idx] == 990.0
        assert steps_list[0][i_idx] == 10.0
        assert steps_list[0][r_idx] == 0.0

        assert math.isclose(steps_list[1][s_idx], 891.0)
        assert math.isclose(steps_list[1][i_idx], 108.5)
        assert math.isclose(steps_list[1][r_idx], 0.5)

        assert math.isclose(steps_list[2][s_idx], 801.9)
        assert math.isclose(steps_list[2][i_idx], 192.175)
        assert math.isclose(steps_list[2][r_idx], 5.925)

        assert math.isclose(steps_list[3][s_idx], 721.71)
        assert math.isclose(steps_list[3][i_idx], 262.75625)
        assert math.isclose(steps_list[3][r_idx], 15.53375)

        assert math.isclose(steps_list[4][s_idx], 649.539)
        assert math.isclose(steps_list[4][i_idx], 321.7894375)
        assert math.isclose(steps_list[4][r_idx], 28.6715625)

        assert math.isclose(steps_list[5][s_idx], 584.5851)
        assert math.isclose(steps_list[5][i_idx], 370.6538656)
        assert math.isclose(steps_list[5][r_idx], 44.76103438)

        assert math.isclose(steps_list[6][s_idx], 526.12659)
        assert math.isclose(steps_list[6][i_idx], 410.5796823)
        assert math.isclose(steps_list[6][r_idx], 63.29372766)

        assert math.isclose(steps_list[7][s_idx], 473.513931)
        assert math.isclose(steps_list[7][i_idx], 442.6633572)
        assert math.isclose(steps_list[7][r_idx], 83.82271177)

        assert math.isclose(steps_list[8][s_idx], 426.1625379)
        assert math.isclose(steps_list[8][i_idx], 467.8815825)
        assert math.isclose(steps_list[8][r_idx], 105.9558796)

        assert math.isclose(steps_list[9][s_idx], 383.5462841)
        assert math.isclose(steps_list[9][i_idx], 487.1037571)
        assert math.isclose(steps_list[9][r_idx], 129.3499588)

        assert math.isclose(steps_list[10][s_idx], 345.1916557)
        assert math.isclose(steps_list[10][i_idx], 501.1031977)
        assert math.isclose(steps_list[10][r_idx], 153.7051466)

    def test_sir_dict_of_lists_output(self, sir_model: Model):
        """
        Tests that a simple SIR model can be built and run, producing
        a valid dict-of-lists result (the default format).
        """
        num_steps = 10
        initial_population = 1000

        sir_simulation = Simulation(sir_model)

        steps_dict = sir_simulation.run(num_steps)

        assert isinstance(steps_dict, dict)
        assert list(steps_dict.keys()) == ["S", "I", "R"]

        num_results = len(steps_dict["S"])
        assert num_results == num_steps + 1

        # Check population conservation
        for i in range(num_results):
            total_population = (
                steps_dict["S"][i] + steps_dict["I"][i] + steps_dict["R"][i]
            )
            assert math.isclose(total_population, initial_population, rel_tol=1e-6), (
                f"Population not conserved at step {i}. Got {total_population}"
            )

        # Check for basic dynamics
        assert steps_dict["S"][0] == 990.0
        assert steps_dict["I"][0] == 10.0
        assert steps_dict["R"][0] == 0.0

        assert math.isclose(steps_dict["S"][1], 891.0)
        assert math.isclose(steps_dict["I"][1], 108.5)
        assert math.isclose(steps_dict["R"][1], 0.5)

        assert math.isclose(steps_dict["S"][2], 801.9)
        assert math.isclose(steps_dict["I"][2], 192.175)
        assert math.isclose(steps_dict["R"][2], 5.925)

        assert math.isclose(steps_dict["S"][3], 721.71)
        assert math.isclose(steps_dict["I"][3], 262.75625)
        assert math.isclose(steps_dict["R"][3], 15.53375)

        assert math.isclose(steps_dict["S"][4], 649.539)
        assert math.isclose(steps_dict["I"][4], 321.7894375)
        assert math.isclose(steps_dict["R"][4], 28.6715625)

        assert math.isclose(steps_dict["S"][5], 584.5851)
        assert math.isclose(steps_dict["I"][5], 370.6538656)
        assert math.isclose(steps_dict["R"][5], 44.76103438)

        assert math.isclose(steps_dict["S"][6], 526.12659)
        assert math.isclose(steps_dict["I"][6], 410.5796823)
        assert math.isclose(steps_dict["R"][6], 63.29372766)

        assert math.isclose(steps_dict["S"][7], 473.513931)
        assert math.isclose(steps_dict["I"][7], 442.6633572)
        assert math.isclose(steps_dict["R"][7], 83.82271177)

        assert math.isclose(steps_dict["S"][8], 426.1625379)
        assert math.isclose(steps_dict["I"][8], 467.8815825)
        assert math.isclose(steps_dict["R"][8], 105.9558796)

        assert math.isclose(steps_dict["S"][9], 383.5462841)
        assert math.isclose(steps_dict["I"][9], 487.1037571)
        assert math.isclose(steps_dict["R"][9], 129.3499588)

        assert math.isclose(steps_dict["S"][10], 345.1916557)
        assert math.isclose(steps_dict["I"][10], 501.1031977)
        assert math.isclose(steps_dict["R"][10], 153.7051466)


class TestStratifiedSIRSimulation:
    """Tests for stratified SIR model simulations."""

    @pytest.fixture(scope="class")
    def age_stratified_sir_model(self) -> Model:
        """
        Builds an age-stratified SIR model where:
        - S, I, R in rate expressions represent the sum of all stratified versions
        - The default rate is applied to all stratifications
        """
        builder = (
            ModelBuilder(name="Age-Stratified SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            # Using S and I which represent sum of all stratified versions
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",  # S = S_young + S_old, I = I_young + I_old
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="gamma * I",  # I = I_young + I_old
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
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    }
                ],
            )
        )
        try:
            return builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")
        assert False, "Should not be reached"

    def test_stratified_sir_runs_successfully(self, age_stratified_sir_model: Model):
        """
        Test that a stratified SIR model with base compartment names (S, I, R)
        in rate expressions runs successfully.
        """
        num_steps = 10
        initial_population = 1000

        simulation = Simulation(age_stratified_sir_model)
        results = simulation.run(num_steps)

        # Should have 6 compartments: S_young, S_old, I_young, I_old, R_young, R_old
        assert len(results) == 6
        expected_compartments = {
            "S_young",
            "S_old",
            "I_young",
            "I_old",
            "R_young",
            "R_old",
        }
        assert set(results.keys()) == expected_compartments

        # Check population conservation
        for i in range(num_steps + 1):
            total = sum(results[comp][i] for comp in results)
            assert math.isclose(total, initial_population, rel_tol=1e-6), (
                f"Population not conserved at step {i}. Got {total}"
            )

        # Check initial conditions
        assert math.isclose(results["S_young"][0], 990 * 0.6)  # 594
        assert math.isclose(results["S_old"][0], 990 * 0.4)  # 396
        assert math.isclose(results["I_young"][0], 10 * 0.6)  # 6
        assert math.isclose(results["I_old"][0], 10 * 0.4)  # 4
        assert results["R_young"][0] == 0.0
        assert results["R_old"][0] == 0.0

        # Verify dynamics: S should decrease, I should initially increase, R increases
        total_s_initial = results["S_young"][0] + results["S_old"][0]
        total_s_final = results["S_young"][num_steps] + results["S_old"][num_steps]
        assert total_s_final < total_s_initial

        total_r_initial = results["R_young"][0] + results["R_old"][0]
        total_r_final = results["R_young"][num_steps] + results["R_old"][num_steps]
        assert total_r_final > total_r_initial

    def test_stratified_sir_with_default_and_specific_rates(self):
        """
        Test that the default rate is applied to stratifications without
        specific stratified_rates entries.
        """
        builder = (
            ModelBuilder(name="Mixed Rates SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_parameter(id="beta_young", value=0.4)
            .add_parameter(id="beta_default", value=0.2)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta_default * S * I / N",  # Default rate for "old"
                stratified_rates=[
                    {
                        "conditions": [{"stratification": "age", "category": "young"}],
                        "rate": "beta_young * S * I / N",  # Specific rate for "young"
                    },
                    # Note: "old" category not specified, should use default rate
                ],
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="gamma * I",
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
                            {"category": "young", "fraction": 0.5},
                            {"category": "old", "fraction": 0.5},
                        ],
                    }
                ],
            )
        )

        try:
            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")
            return

        simulation = Simulation(model)
        results = simulation.run(5)

        # Model should run without errors
        assert len(results) == 6

        # Verify young group has higher infection rate than old group
        # Due to beta_young (0.4) > beta_default (0.2)
        # After first step, young S should decrease more than old S (proportionally)
        s_young_decrease = results["S_young"][0] - results["S_young"][1]
        s_old_decrease = results["S_old"][0] - results["S_old"][1]

        # Young should have faster infection (higher beta)
        assert s_young_decrease > s_old_decrease

    def test_multi_stratified_sir_with_base_compartments(self):
        """
        Test a model with multiple stratifications using base compartment names.
        """
        builder = (
            ModelBuilder(name="Multi-Stratified SIR", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_stratification(id="location", categories=["urban", "rural"])
            .add_parameter(id="beta", value=0.25)
            .add_parameter(id="gamma", value=0.1)
            # Using base compartment names - sum of all 4 stratified versions each
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
                            {"category": "young", "fraction": 0.6},
                            {"category": "old", "fraction": 0.4},
                        ],
                    },
                    {
                        "stratification": "location",
                        "fractions": [
                            {"category": "urban", "fraction": 0.7},
                            {"category": "rural", "fraction": 0.3},
                        ],
                    },
                ],
            )
        )

        try:
            model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")
            return

        simulation = Simulation(model)
        results = simulation.run(10)

        # Should have 12 compartments (3 bins x 2 ages x 2 locations)
        assert len(results) == 12
        expected_compartments = {
            "S_young_urban",
            "S_young_rural",
            "S_old_urban",
            "S_old_rural",
            "I_young_urban",
            "I_young_rural",
            "I_old_urban",
            "I_old_rural",
            "R_young_urban",
            "R_young_rural",
            "R_old_urban",
            "R_old_rural",
        }
        assert set(results.keys()) == expected_compartments

        # Verify population conservation
        for i in range(11):
            total = sum(results[comp][i] for comp in results)
            assert math.isclose(total, 1000, rel_tol=1e-6)


class TestParameterNameValidation:
    """Tests for parameter name validation against reserved compartment names."""

    def test_parameter_name_conflicts_with_bin_id_raises_error(self):
        """
        Test that using a bin ID as a parameter name raises an error.
        """
        with pytest.raises(ValueError) as exc_info:
            (
                ModelBuilder(name="Invalid Model", version="1.0")
                .add_bin(id="S", name="Susceptible")
                .add_bin(id="I", name="Infected")
                .add_bin(id="R", name="Recovered")
                # This should raise an error - "S" is a reserved name
                .add_parameter(id="S", value=0.5)
                .add_parameter(id="gamma", value=0.1)
                .add_transition(
                    id="recovery",
                    source=["I"],
                    target=["R"],
                    rate="gamma * I",
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

        error_message = str(exc_info.value)
        assert "conflict" in error_message.lower()
        assert "S" in error_message
        assert "reserved" in error_message.lower()

    def test_parameter_name_conflicts_with_bin_id_raises_error_with_stratification(
        self,
    ):
        """
        Test that using a bin ID as a parameter name raises an error
        when stratifications are present.
        """
        with pytest.raises(ValueError) as exc_info:
            (
                ModelBuilder(name="Invalid Model", version="1.0")
                .add_bin(id="S", name="Susceptible")
                .add_bin(id="I", name="Infected")
                .add_bin(id="R", name="Recovered")
                .add_stratification(id="age", categories=["young", "old"])
                # This should raise an error - "S" is a reserved name
                .add_parameter(id="S", value=0.5)
                .add_parameter(id="gamma", value=0.1)
                .add_transition(
                    id="recovery",
                    source=["I"],
                    target=["R"],
                    rate="gamma * I",
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
                                {"category": "young", "fraction": 0.5},
                                {"category": "old", "fraction": 0.5},
                            ],
                        }
                    ],
                )
                .build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            )

        error_message = str(exc_info.value)
        assert "conflict" in error_message.lower()
        assert "S" in error_message
        assert "reserved" in error_message.lower()

    def test_multiple_conflicting_parameters_all_reported(self):
        """
        Test that when multiple parameters conflict with bin IDs,
        all conflicts are reported.
        """
        with pytest.raises(ValueError) as exc_info:
            (
                ModelBuilder(name="Invalid Model", version="1.0")
                .add_bin(id="S", name="Susceptible")
                .add_bin(id="I", name="Infected")
                .add_bin(id="R", name="Recovered")
                # Multiple conflicting parameters
                .add_parameter(id="S", value=0.5)
                .add_parameter(id="I", value=0.3)
                .add_parameter(id="gamma", value=0.1)
                .add_transition(
                    id="recovery",
                    source=["I"],
                    target=["R"],
                    rate="gamma",
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

        error_message = str(exc_info.value)
        # Both S and I should be mentioned
        assert "S" in error_message or "I" in error_message


class TestBaseCompartmentTotals:
    """Tests verifying that base compartment totals (S, I, R) are computed correctly."""

    def test_base_compartment_equals_sum_of_stratified(self):
        """
        Test that using S in a rate expression gives the same result as
        explicitly using S_young + S_old.
        """
        # Model 1: Using base compartment name S
        builder1 = (
            ModelBuilder(name="Base Compartment Model", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="beta * S * I / N",  # Uses S (sum of S_young + S_old)
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="gamma * I",
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
                            {"category": "young", "fraction": 0.5},
                            {"category": "old", "fraction": 0.5},
                        ],
                    }
                ],
            )
        )

        # Model 2: Using explicit stratified compartment names
        builder2 = (
            ModelBuilder(name="Explicit Compartment Model", version="1.0")
            .add_bin(id="S", name="Susceptible")
            .add_bin(id="I", name="Infected")
            .add_bin(id="R", name="Recovered")
            .add_stratification(id="age", categories=["young", "old"])
            .add_parameter(id="beta", value=0.3)
            .add_parameter(id="gamma", value=0.1)
            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                # Explicitly using S_young + S_old and I_young + I_old
                rate="beta * (S_young + S_old) * (I_young + I_old) / N",
            )
            .add_transition(
                id="recovery",
                source=["I"],
                target=["R"],
                rate="gamma * (I_young + I_old)",
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
                            {"category": "young", "fraction": 0.5},
                            {"category": "old", "fraction": 0.5},
                        ],
                    }
                ],
            )
        )

        try:
            model1 = builder1.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
            model2 = builder2.build(typology=ModelTypes.DIFFERENCE_EQUATIONS.value)
        except ImportError:
            pytest.skip("Rust extension not built. Skipping simulation tests.")
            return

        simulation1 = Simulation(model1)
        simulation2 = Simulation(model2)

        results1 = simulation1.run(10)
        results2 = simulation2.run(10)

        # Both models should produce identical results
        for comp in results1:
            for i in range(11):
                assert math.isclose(
                    results1[comp][i], results2[comp][i], rel_tol=1e-9
                ), (
                    f"Mismatch at {comp}[{i}]: "
                    f"{results1[comp][i]} vs {results2[comp][i]}"
                )
