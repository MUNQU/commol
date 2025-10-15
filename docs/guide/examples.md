# Examples

Complete examples demonstrating different modeling scenarios.

## Example 1: Basic SIR Model

Classic Susceptible-Infected-Recovered model:

```python
from epimodel import ModelBuilder, Simulation
from epimodel.constants import ModelTypes
import matplotlib.pyplot as plt

# Build model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="E", name="Exposed")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.4, description="Transmission rate")
    .add_parameter(id="sigma", value=0.2, description="Incubation rate")
    .add_parameter(id="gamma", value=0.1, description="Recovery rate")
    .add_transition(id="exposure", source=["S"], target=["E"], rate="beta * S * I / N")
    .add_transition(id="infection", source=["E"], target=["I"], rate="sigma")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.999},
            {"disease_state": "E", "fraction": 0.0},
            {"disease_state": "I", "fraction": 0.001},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.95},
            {"disease_state": "I", "fraction": 0.05},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_parameter(id="omega", value=0.01)  # Waning immunity rate
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma")
    .add_transition(id="waning", source=["R"], target=["S"], rate="omega")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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

## Next Steps

- [API Reference](../api/model-builder.md) - Complete API documentation
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Contributing](../development/contributing.md) - Build your own examples
