# Core Concepts

Commol is built around several key concepts that work together to create compartment models.

## Disease States

Disease states (also called compartments) represent different stages of infection in your population.

```python
builder.add_bin(id="S", name="Susceptible")
builder.add_bin(id="E", name="Exposed")
builder.add_bin(id="I", name="Infected")
builder.add_bin(id="R", name="Recovered")
```

### Common Disease States

| ID  | Name        | Description                         |
| --- | ----------- | ----------------------------------- |
| S   | Susceptible | Individuals who can become infected |
| E   | Exposed     | Infected but not yet infectious     |
| I   | Infected    | Actively infected and infectious    |
| R   | Recovered   | No longer infectious, immune        |
| D   | Dead        | Deceased from disease               |

## Stratifications

Stratifications allow you to model population subgroups (age, location, risk factors, etc.):

```python
# Age stratification
builder.add_stratification(
    id="age_group",
    categories=["young", "adult", "elderly"]
)

# Location stratification
builder.add_stratification(
    id="region",
    categories=["urban", "rural"]
)
```

### Initial Conditions with Stratifications

```python
builder.set_initial_conditions(
    population_size=10000,
    bin_fractions=[
        {"bin": "S", "fraction": 0.99},
        {"bin": "I", "fraction": 0.01},
        {"bin": "R", "fraction": 0.0}
    ],
    stratification_fractions=[
        {
            "stratification": "age_group",
            "fractions": [
                {"category": "young", "fraction": 0.3},
                {"category": "adult", "fraction": 0.5},
                {"category": "elderly", "fraction": 0.2}
            ]
        }
    ]
)
```

This creates compartments: `S_young`, `S_adult`, `S_elderly`, `I_young`, etc.

## Parameters

Parameters are global constants used throughout your model:

```python
builder.add_parameter(
    id="beta",
    value=0.3,
    description="Transmission rate per contact"
)

builder.add_parameter(
    id="gamma",
    value=0.1,
    description="Recovery rate (1/infectious_period)"
)
```

### Common Parameters

| Parameter | Meaning                               | Typical Range  |
| --------- | ------------------------------------- | -------------- |
| `beta`    | Transmission rate                     | 0.1 - 1.0      |
| `gamma`   | Recovery rate                         | 0.05 - 0.5     |
| `sigma`   | Incubation rate (1/incubation_period) | 0.1 - 0.5      |
| `mu`      | Birth/death rate                      | 0.0001 - 0.001 |
| `R0`      | Basic reproduction number             | 1.0 - 10.0     |

## Transitions

Transitions define how populations move between disease states:

### Simple Transitions

```python
# Recovery: I → R
builder.add_transition(
    id="recovery",
    source=["I"],
    target=["R"],
    rate="gamma"
)
```

### Formula-Based Transitions

```python
# Infection: S → I (with force of infection)
builder.add_transition(
    id="infection",
    source=["S"],
    target=["I"],
    rate="beta * S * I / N"
)
```

### Multi-Source Transitions

```python
# Natural death from any state
builder.add_transition(
    id="death",
    source=["S", "I", "R"],
    target=[],  # Empty = removal from system
    rate="0.000027"  # Daily death rate
)
```

## Initial Conditions

Initial conditions define the starting state of your model:

```python
builder.set_initial_conditions(
    population_size=1000,
    bin_fractions=[
        {"bin": "S", "fraction": 0.99},  # 99% susceptible
        {"bin": "I", "fraction": 0.01},  # 1% infected
        {"bin": "R", "fraction": 0.0}    # 0% recovered
    ]
)
```

### Validation Rules

- Disease state fractions must sum to 1.0
- Stratification fractions must sum to 1.0 for each stratification
- Population size must be positive
- All disease states must have initial fractions defined

## Model Types

EpiModel currently supports:

```python
from commol.constants import ModelTypes

model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
```

### Difference Equations

- Discrete time steps
- Deterministic dynamics
- Fast computation
- Best for: Population-level modeling, policy analysis

## The Model Building Process

A typical workflow:

1. **Define disease states** - What compartments exist?
2. **Add stratifications** (optional) - What subgroups matter?
3. **Define parameters** - What rates and constants?
4. **Create transitions** - How do populations flow?
5. **Set initial conditions** - What's the starting state?
6. **Build the model** - Validate and construct
7. **Run simulation** - Execute and analyze

## Next Steps

- [Building Models](building-models.md) - Detailed ModelBuilder API
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Examples](examples.md) - Complete model examples
