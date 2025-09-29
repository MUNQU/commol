import pytest
import math
from epimodel.api import ModelBuilder


def test_simulation_run():
    """
    Tests that a simple SIR model can be built, the engine created,
    and a simulation run, producing valid results.
    """
    # 1. Use the builder to define a simple SIR model
    builder = (
        ModelBuilder(name="TestSimSIR", version="1.0")
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
        .add_transition(id="recovery", source=["I"], target=["R"], rate="recovery_rate")
        .set_initial_conditions(
            population_size=1000,
            disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0},
        )
    )

    # 2. Build the Rust engine
    # This requires the project to be compiled. If not, skip the test.
    try:
        engine = builder.build_engine()
    except ImportError:
        pytest.skip("Rust extension not built. Skipping simulation test.")

    # 3. Run the simulation
    num_steps = 10
    initial_population = 1000
    history = engine.run(num_steps)

    # 4. Assert properties of the results
    assert len(history) == num_steps + 1

    # Based on the order of `add_disease_state`, the compartments are S, I, R.
    s_idx, i_idx, r_idx = 0, 1, 2

    # Check that population is conserved
    for i, population_state in enumerate(history):
        assert isinstance(population_state, list)
        assert len(population_state) == 3  # S, I, R

        total_population = sum(population_state)
        assert math.isclose(total_population, initial_population, rel_tol=1e-6), (
            f"Population not conserved at step {i}. Got {total_population}"
        )

        # Check for non-negative values
        for value in population_state:
            assert value >= 0

    # Check for basic dynamics (S should decrease, R should increase)
    s_initial = history[0][s_idx]
    s_final = history[-1][s_idx]
    r_initial = history[0][r_idx]
    r_final = history[-1][r_idx]

    assert s_final < s_initial
    assert r_final > r_initial
