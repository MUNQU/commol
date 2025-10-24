# Examples

Complete examples demonstrating different modeling scenarios.

## Example 1: Basic SIR Model

Classic Susceptible-Infected-Recovered model:

```python
from commol import ModelBuilder, Simulation
from commol.constants import ModelTypes
import matplotlib.pyplot as plt

# Build model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Simulate
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 2: SEIR Model

Adding an exposed (incubation) period:

```python
model = (
    ModelBuilder(name="SEIR Model", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="E", name="Exposed")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.4, description="Transmission rate")
    .add_parameter(id="sigma", value=0.2, description="Incubation rate")
    .add_parameter(id="gamma", value=0.1, description="Recovery rate")
    .add_transition(id="exposure", source=["S"], target=["E"], rate="beta * S * I / N")
    .add_transition(id="infection", source=["E"], target=["I"], rate="sigma")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.999},
            {"bin": "E", "fraction": 0.0},
            {"bin": "I", "fraction": 0.001},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 3: Seasonal Transmission

Modeling seasonal variation in transmission:

```python
model = (
    ModelBuilder(name="Seasonal SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta_mean", value=0.3)
    .add_parameter(id="beta_amp", value=0.2)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(
        id="seasonal_infection",
        source=["S"],
        target=["I"],
        # Seasonal forcing: peaks in winter (day 0, 365, ...)
        rate="beta_mean * (1 + beta_amp * sin(2 * pi * step / 365)) * S * I / N"
    )
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Run for 3 years to see seasonal pattern
simulation = Simulation(model)
results = simulation.run(num_steps=365 * 3)
```

## Example 4: Age-Stratified Model

Different age groups with varying recovery rates:

```python
model = (
    ModelBuilder(name="Age-Stratified SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_stratification(id="age", categories=["child", "adult", "elderly"])
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma_child", value=0.15)
    .add_parameter(id="gamma_adult", value=0.12)
    .add_parameter(id="gamma_elderly", value=0.08)
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        stratified_rates=[
            {
                "conditions": [{"stratification": "age", "category": "child"}],
                "rate": "gamma_child"
            },
            {
                "conditions": [{"stratification": "age", "category": "adult"}],
                "rate": "gamma_adult"
            },
            {
                "conditions": [{"stratification": "age", "category": "elderly"}],
                "rate": "gamma_elderly"
            },
        ]
    )
    .set_initial_conditions(
        population_size=10000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ],
        stratification_fractions=[
            {
                "stratification": "age",
                "fractions": [
                    {"category": "child", "fraction": 0.25},
                    {"category": "adult", "fraction": 0.55},
                    {"category": "elderly", "fraction": 0.20}
                ]
            }
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 5: Vaccination Campaign

Adding vaccination to an SIR model:

```python
model = (
    ModelBuilder(name="SIR with Vaccination", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.4)
    .add_parameter(id="gamma", value=0.1)
    .add_parameter(id="vax_rate", value=0.01)  # 1% per day
    .add_parameter(id="vax_eff", value=0.9)     # 90% effective
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .add_transition(
        id="vaccination",
        source=["S"],
        target=["R"],
        rate="vax_rate * vax_eff"  # Effective vaccination rate
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Compare with and without vaccination
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 6: Healthcare Capacity

Modeling reduced recovery when hospitals are overwhelmed:

```python
model = (
    ModelBuilder(name="SIR with Healthcare Capacity", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.5)
    .add_parameter(id="gamma_max", value=0.15)
    .add_parameter(id="hospital_cap", value=100.0)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(
        id="recovery_saturated",
        source=["I"],
        target=["R"],
        # Recovery slows as infections approach hospital capacity
        rate="gamma_max * (1 - max(0, (I - hospital_cap) / hospital_cap))"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.95},
            {"bin": "I", "fraction": 0.05},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 7: Waning Immunity

Recovered individuals gradually become susceptible again:

```python
model = (
    ModelBuilder(name="SIRS Model", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_parameter(id="omega", value=0.01)  # Waning immunity rate
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .add_transition(id="waning", source=["R"], target=["S"], rate="omega")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Run for longer to see endemic equilibrium
simulation = Simulation(model)
results = simulation.run(num_steps=1000)
```

## Example 8: Multi-Stratified Model

This example demonstrates a model with multiple stratifications (age and risk) and defines a transition rate for a specific intersection of these stratifications.

```python
import numpy as np

# Build model
model = (
    ModelBuilder(name="Multi-Stratified SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_stratification(id="age", categories=["young", "old"])
    .add_stratification(id="risk", categories=["low", "high"])
    .add_parameter(id="beta_low", value=0.3)
    .add_parameter(id="beta_high", value=0.6)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta_low * S * I / N", # Default rate
        stratified_rates=[
            {
                "conditions": [{"stratification": "risk", "category": "high"}],
                "rate": "beta_high * S * I / N"
            }
        ]
    )
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ],
        stratification_fractions=[
            {
                "stratification": "age",
                "fractions": [
                    {"category": "young", "fraction": 0.6},
                    {"category": "old", "fraction": 0.4}
                ]
            },
            {
                "stratification": "risk",
                "fractions": [
                    {"category": "low", "fraction": 0.8},
                    {"category": "high", "fraction": 0.2}
                ]
            }
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Simulate
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 9: Subpopulation-Dependent Transmission

This example shows how to model frequency-dependent transmission within a specific subpopulation using the automatically calculated `N_{category}` variable. The infection rate for the `young` category is normalized by the total `young` population (`N_young`), while the `old` category uses the global population `N`.

```python
# Build model
model = (
    ModelBuilder(name="Subpopulation-Dependent SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_stratification(id="age", categories=["young", "old"])
    .add_parameter(id="beta", value=0.4)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N", # Default rate for old
        stratified_rates=[
            {
                "conditions": [{"stratification": "age", "category": "young"}],
                "rate": "beta * S * I / N_young" # Rate normalized by subpopulation
            }
        ]
    )
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ],
        stratification_fractions=[
            {
                "stratification": "age",
                "fractions": [
                    {"category": "young", "fraction": 0.5},
                    {"category": "old", "fraction": 0.5}
                ]
            }
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Simulate
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Example 10: Parameter Calibration

Calibrate model parameters to match observed data:

```python
from commol import (
    ModelBuilder,
    Simulation,
    Calibrator,
    CalibrationProblem,
    CalibrationParameter,
    ObservedDataPoint,
    LossConfig,
    LossFunction,
    OptimizationConfig,
    OptimizationAlgorithm,
    ParticleSwarmConfig,
)
from commol.constants import ModelTypes

# Build SIR model with initial parameter guesses
model = (
    ModelBuilder(name="SIR for Calibration", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)   # Initial guess
    .add_parameter(id="gamma", value=0.1)  # Initial guess
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="gamma * I"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Define observed data (e.g., from real outbreak surveillance)
observed_data = [
    ObservedDataPoint(step=0, compartment="I", value=10.0),
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
    ObservedDataPoint(step=40, compartment="I", value=38.1),
    ObservedDataPoint(step=50, compartment="I", value=18.7),
    ObservedDataPoint(step=60, compartment="I", value=8.2),
]

# Define parameters to calibrate
parameters = [
    CalibrationParameter(
        id="beta",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3
    ),
    CalibrationParameter(
        id="gamma",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.1
    ),
]

# Configure calibration problem
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.SSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig(
            num_particles=30,
            max_iterations=500,
            verbose=True
        ),
    ),
)

# Run calibration
simulation = Simulation(model)
calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Display results
print("\n=== Calibration Results ===")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final loss: {result.final_loss:.6f}")
print(f"Calibrated beta: {result.best_parameters['beta']:.6f}")
print(f"Calibrated gamma: {result.best_parameters['gamma']:.6f}")
print(f"Termination reason: {result.termination_reason}")
```

## Next Steps

- [API Reference](../api/model-builder.md) - Complete API documentation
- [Calibration Guide](calibration.md) - Comprehensive calibration documentation
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Contributing](../development/contributing.md) - Build your own examples
