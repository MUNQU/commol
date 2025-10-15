# Quick Start

This guide will help you build and run your first epidemiological model with EpiModel.

## Your First SIR Model

Let's create a basic SIR (Susceptible-Infected-Recovered) model:

```python
from epimodel import ModelBuilder, Simulation
from epimodel.constants import ModelTypes

# Build the model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)   # Transmission rate
    .add_parameter(id="gamma", value=0.1)  # Recovery rate
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"  # Mathematical formula
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="gamma"
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

# Run simulation
simulation = Simulation(model)
results = simulation.run(num_steps=100)

# Display results
print(f"Susceptible at day 100: {results['S'][-1]:.0f}")
print(f"Infected at day 100: {results['I'][-1]:.0f}")
print(f"Recovered at day 100: {results['R'][-1]:.0f}")
```

## Understanding the Code

### 1. Import Required Classes

```python
from epimodel import ModelBuilder, Simulation
from epimodel.constants import ModelTypes
```

- `ModelBuilder`: Fluent API for constructing models
- `Simulation`: Runs the model simulation
- `ModelTypes`: Enumeration of available model types

### 2. Define Disease States

```python
.add_disease_state(id="S", name="Susceptible")
.add_disease_state(id="I", name="Infected")
.add_disease_state(id="R", name="Recovered")
```

Disease states represent compartments in your model.

### 3. Add Parameters

```python
.add_parameter(id="beta", value=0.3)   # Transmission rate
.add_parameter(id="gamma", value=0.1)  # Recovery rate
```

Parameters are constants used in transition rate formulas.

### 4. Define Transitions

```python
.add_transition(
    id="infection",
    source=["S"],
    target=["I"],
    rate="beta * S * I / N"
)
```

Transitions move populations between states using mathematical formulas.

### 5. Set Initial Conditions

```python
.set_initial_conditions(
    population_size=1000,
    disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
)
```

Define the starting population distribution.

### 6. Build and Run

```python
model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Next Steps

Now that you've built your first model, explore:

- [Core Concepts](../guide/core-concepts.md) - Deep dive into EpiModel concepts
- [Building Models](../guide/building-models.md) - Advanced model construction
- [Mathematical Expressions](../guide/mathematical-expressions.md) - Complex rate formulas
- [Examples](../guide/examples.md) - More complete examples
