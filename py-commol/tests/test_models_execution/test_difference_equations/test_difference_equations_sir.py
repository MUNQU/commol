import math

import pytest

from commol.api import ModelBuilder
from commol.api.simulation import Simulation
from commol.context import Model
from commol.constants import ModelTypes


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
