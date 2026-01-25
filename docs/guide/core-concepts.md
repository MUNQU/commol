# Core Concepts

Commol is built around several key concepts that work together to create compartment models.

## Compartments

Compartments (also called bins or states) represent distinct states that entities in your model can occupy. In the API, these are added using `add_bin()`.

```python
builder.add_bin(id="S", name="Susceptible")
builder.add_bin(id="E", name="Exposed")
builder.add_bin(id="I", name="Infected")
builder.add_bin(id="R", name="Recovered")
```

### Example: Epidemiological Compartments

The following table shows common compartments used in epidemiological models:

| ID  | Name        | Description                         |
| --- | ----------- | ----------------------------------- |
| S   | Susceptible | Individuals who can become infected |
| E   | Exposed     | Infected but not yet infectious     |
| I   | Infected    | Actively infected and infectious    |
| R   | Recovered   | No longer infectious, immune        |
| D   | Dead        | Deceased from disease               |

Note: Compartments can represent any type of state depending on your application domain (e.g., customer segments, chemical species, population groups).

## Stratifications

Stratifications divide your population into subgroups based on characteristics like age, location, or risk factors. When you add a stratification, Commol automatically creates separate compartments for each subgroup, tracking disease dynamics independently within each stratum.

### How Stratification Works

When you define compartments **without** stratification, each compartment represents the entire population in that state:

```
Base compartments: S, I, R
Total: 3 compartments
```

When you add a stratification, Commol **expands** each compartment by creating one version per category. The naming pattern is `{compartment}_{category}`:

```
Add stratification: age = [young, old]

Expanded compartments:
  S → S_young, S_old
  I → I_young, I_old
  R → R_young, R_old

Total: 6 compartments (3 bins × 2 categories)
```

With **multiple stratifications**, compartments are expanded using the Cartesian product of all categories. Each additional stratification multiplies the number of compartments:

```
Add stratifications:
  age = [young, old]
  location = [urban, rural]

Expanded compartments:
  S → S_young_urban, S_young_rural, S_old_urban, S_old_rural
  I → I_young_urban, I_young_rural, I_old_urban, I_old_rural
  R → R_young_urban, R_young_rural, R_old_urban, R_old_rural

Total: 12 compartments (3 bins × 2 ages × 2 locations)
```

The order of category suffixes matches the order in which stratifications are added to the model.

### Defining Stratifications

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

When setting initial conditions, you specify:

1. **Bin fractions**: How the population is distributed across disease states
2. **Stratification fractions**: How each disease state is distributed across categories

These fractions are applied multiplicatively:

```python
builder.set_initial_conditions(
    population_size=10000,
    bin_fractions=[
        {"bin": "S", "fraction": 0.99},   # 9900 susceptible
        {"bin": "I", "fraction": 0.01},   # 100 infected
        {"bin": "R", "fraction": 0.0}     # 0 recovered
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

**Resulting initial populations:**

| Compartment | Calculation        | Initial Value |
| ----------- | ------------------ | ------------- |
| `S_young`   | 10000 × 0.99 × 0.3 | 2970          |
| `S_adult`   | 10000 × 0.99 × 0.5 | 4950          |
| `S_elderly` | 10000 × 0.99 × 0.2 | 1980          |
| `I_young`   | 10000 × 0.01 × 0.3 | 30            |
| `I_adult`   | 10000 × 0.01 × 0.5 | 50            |
| `I_elderly` | 10000 × 0.01 × 0.2 | 20            |
| `R_young`   | 10000 × 0.0 × 0.3  | 0             |
| `R_adult`   | 10000 × 0.0 × 0.5  | 0             |
| `R_elderly` | 10000 × 0.0 × 0.2  | 0             |

### Why Use Stratifications?

Stratifications are essential when:

- **Different groups have different rates**: Young people may recover faster, elderly may have higher mortality
- **Modeling heterogeneous mixing**: Urban populations may have higher contact rates than rural ones
- **Policy analysis**: Evaluate interventions targeting specific subgroups (e.g., vaccinating the elderly first)
- **Data fitting**: Match model outputs to age-stratified surveillance data

## Parameters

Parameters are global constants used throughout your model:

```python
builder.add_parameter(
    id="beta",
    value=0.3,
    description="Transition rate coefficient"
)

builder.add_parameter(
    id="gamma",
    value=0.1,
    description="Rate constant"
)
```

### Example: Epidemiological Parameters

The following table shows common parameters used in epidemiological models:

| Parameter | Meaning                               | Typical Range  |
| --------- | ------------------------------------- | -------------- |
| `beta`    | Transmission rate                     | 0.1 - 1.0      |
| `gamma`   | Recovery rate                         | 0.05 - 0.5     |
| `sigma`   | Incubation rate (1/incubation_period) | 0.1 - 0.5      |
| `mu`      | Birth/death rate                      | 0.0001 - 0.001 |
| `R0`      | Basic reproduction number             | 1.0 - 10.0     |

## Transitions

Transitions define how populations move between compartments:

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
        {"bin": "S", "fraction": 0.99},  # 99% in state S
        {"bin": "I", "fraction": 0.01},  # 1% in state I
        {"bin": "R", "fraction": 0.0}    # 0% in state R
    ]
)
```

### Validation Rules

- Compartment fractions must sum to 1.0
- Stratification fractions must sum to 1.0 for each stratification
- Population size must be positive
- All compartments must have initial fractions defined

## Model Types

Commol currently supports:

```python
model = builder.build(typology="DifferenceEquations")
```

### Difference Equations

- Discrete time steps
- Deterministic dynamics
- Fast computation
- Best for: Population-level modeling, policy analysis

## The Model Building Process

A typical workflow:

1. **Define compartments** - What states exist in your model?
2. **Add stratifications** (optional) - What subgroups matter?
3. **Define parameters** - What rates and constants?
4. **Create transitions** - How do populations flow between compartments?
5. **Set initial conditions** - What's the starting state?
6. **Build the model** - Validate and construct
7. **Run simulation** - Execute and analyze

## Next Steps

- [Building Models](building-models.md) - Detailed ModelBuilder API
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Examples](examples.md) - Complete model examples
