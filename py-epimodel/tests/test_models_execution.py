import math

from epimodel.api import ModelBuilder


class TestModelsExecution:

    def test_sir(self):
        """
        Tests that a simple SIR model can be built, the engine created,
        and a simulation run, producing valid results.
        """
        builder = (

            ModelBuilder(name="Test Sir", version="1.0")

            .add_disease_state(id="S", name="Susceptible")
            .add_disease_state(id="I", name="Infected")
            .add_disease_state(id="R", name="Recovered")

            .add_parameter(id="infection_rate", value=0.1)
            .add_parameter(id="recovery_rate", value=0.05)

            .add_transition(
                id="infection",
                source=["S"],
                target=["I"],
                rate="infection_rate",
            )
            .add_transition(
                id="recovery", 
                source=["I"], 
                target=["R"], 
                rate="recovery_rate"
            )

            .set_initial_conditions(
                population_size=1000,
                disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0},
            )
        )

        try:
            engine = builder.build_engine()
        except ImportError:
            raise ImportError("Rust extension not built. Skipping simulation test.")

        num_steps = 10
        initial_population = 1000
        history = engine.run(num_steps)

        assert len(history) == num_steps + 1

        # Based on the order of `add_disease_state`, the compartments are S, I, R.
        s_idx, _, r_idx = 0, 1, 2

        for i, population_state in enumerate(history):
            assert isinstance(population_state, list)
            assert len(population_state) == 3  # S, I, R
    
            total_population = sum(population_state)
            assert math.isclose(total_population, initial_population, rel_tol=1e-6), (
                f"Population not conserved at step {i}. Got {total_population}"
            )
    
            for value in population_state:
                assert value >= 0
    
        # Check for basic dynamics (S should decrease, R should increase)
        s_initial = history[0][s_idx]
        s_final = history[-1][s_idx]
        assert s_final < s_initial
        
        r_initial = history[0][r_idx]
        r_final = history[-1][r_idx]
        assert r_final > r_initial
