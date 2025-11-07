# Building Models

The `ModelBuilder` class provides a fluent API for constructing compartment models.

## ModelBuilder Basics

### Creating a Builder

```python
from commol import ModelBuilder

builder = ModelBuilder(
    name="My Model",
    version="1.0",
    description="Optional description"
)
```

### Chaining Methods

The builder uses method chaining for a clean, readable API:

```python
model = (
    ModelBuilder(name="SIR Model")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .build(ModelTypes.DIFFERENCE_EQUATIONS)
)
```

## Adding Disease States

```python
builder.add_bin(
    id="S",                    # Required: Unique identifier
    name="Susceptible",        # Required: Display name
    description="Population susceptible to infection"  # Optional
)
```

### Best Practices

- Use short, clear IDs (S, I, R, E, etc.)
- Provide descriptive names

## Adding Stratifications

Stratifications create population subgroups:

```python
builder.add_stratification(
    id="age_group",
    categories=["0-17", "18-64", "65+"],
    description="Age-based stratification"
)
```

### Multiple Stratifications

```python
builder.add_stratification(id="age", categories=["young", "old"])
builder.add_stratification(id="location", categories=["urban", "rural"])
```

This creates compartments: `S_young_urban`, `S_young_rural`, `S_old_urban`, `S_old_rural`, etc.

## Adding Parameters

Parameters are global constants used in formulas:

```python
builder.add_parameter(
    id="beta",
    value=0.3,
    description="Transmission rate per contact per day"
)
```

### Parameters with Units

You can specify units for automatic dimensional analysis and validation:

```python
builder.add_parameter(
    id="beta",
    value=0.5,
    description="Transmission rate",
    unit="1/day"  # Rate unit
)

builder.add_parameter(
    id="seasonal_amplitude",
    value=0.2,
    description="Seasonal variation amplitude",
    unit="dimensionless"  # Pure number
)
```

When **all parameters have units**, the model will automatically validate dimensional consistency. See [Unit Checking](#unit-checking) below.

**Tip:** To mark a parameter as unitless (dimensionless) for unit checking, use `unit="dimensionless"`. This is useful for ratios, fractions, scaling factors, and amplitudes. Dimensionless parameters are also required as arguments to mathematical functions like `sin()`, `cos()`, `exp()`, `sqrt()`, `pow()`, etc.

### Parameter Guidelines

- Use meaningful IDs (beta, gamma, R0, etc.)
- Document units and meaning
- Ensure values are realistic for your model
- Specify units for automatic validation (recommended)

## Adding Transitions

Transitions move populations between states.

### Understanding Transition Rates

The `rate` parameter accepts **mathematical expressions** that can include:

- **Parameters**: Reference parameter IDs (e.g., `"gamma"`)
- **Disease states**: Use state populations (e.g., `"S"`, `"I"`)
- **Special variables**: `N` (total population), `step` or `t` (current time step), `pi`, `e`
- **Mathematical operations**: `+`, `-`, `*`, `/`, `**` (power)
- **Functions**: `sin()`, `cos()`, `exp()`, `log()`, `sqrt()`, `max()`, `min()`, etc.

For the complete list of functions and advanced examples, see [Mathematical Expressions](mathematical-expressions.md).

### Simple Parameter-Based Rates

```python
builder.add_transition(
    id="recovery",
    source=["I"],
    target=["R"],
    rate="gamma"  # References parameter id
)
```

### Formula-Based Rates

```python
builder.add_transition(
    id="infection",
    source=["S"],
    target=["I"],
    rate="beta * S * I / N"  # Mathematical expression
)
```

### Constant Rates

```python
builder.add_transition(
    id="birth",
    source=[],      # Empty = enters system
    target=["S"],
    rate="0.001"    # Fixed rate
)
```

### Time-Dependent Rates

```python
builder.add_transition(
    id="seasonal_infection",
    source=["S"],
    target=["I"],
    rate="beta * (1 + 0.3 * sin(2 * pi * t / 365)) * S * I / N"
)
```

See [Mathematical Expressions](mathematical-expressions.md) for more complex rate formulas.

### Multi-State Transitions

```python
# Death from any compartment
builder.add_transition(
    id="death",
    source=["S", "I", "R"],
    target=[],  # Empty = leaves system
    rate="mu"
)
```

### Stratified Transitions

When a model includes stratifications, you often need different transition rates for different subgroups. The `add_transition` method supports this via the `stratified_rates` parameter.

This parameter takes a list of dictionaries, where each dictionary defines a rate for a specific combination of stratification categories.

#### Single Stratification

Let's define different recovery rates for different age groups.

```python
builder.add_stratification(id="age", categories=["child", "adult", "elderly"])
builder.add_parameter(id="gamma_child", value=0.15)
builder.add_parameter(id="gamma_adult", value=0.1)
builder.add_parameter(id="gamma_elderly", value=0.08)

builder.add_transition(
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
```

#### Multi-Stratification Transitions

To define rates for intersections of multiple stratifications, add multiple conditions to a single rate entry.

For example, let's model different infection rates for high-risk adults in urban areas.

```python
builder.add_stratification(id="age", categories=["child", "adult"])
builder.add_stratification(id="risk", categories=["low", "high"])
builder.add_stratification(id="location", categories=["urban", "rural"])

builder.add_parameter(id="beta_urban_adult_high_risk", value=0.8)
builder.add_parameter(id="beta_default", value=0.3)

builder.add_transition(
    id="infection",
    source=["S"],
    target=["I"],
    rate="beta_default * S * I / N",  # Fallback rate
    stratified_rates=[
        {
            "conditions": [
                {"stratification": "age", "category": "adult"},
                {"stratification": "risk", "category": "high"},
                {"stratification": "location", "category": "urban"},
            ],
            "rate": "beta_urban_adult_high_risk * S * I / N"
        }
    ]
)
```

In this example:

- The `rate` parameter acts as a **fallback** for any compartment that doesn't match a specific stratified rate.
- The `stratified_rates` entry defines a high infection rate that only applies to compartments matching all three conditions (e.g., `S_adult_high_urban`).

## Setting Initial Conditions

### Basic Setup

```python
builder.set_initial_conditions(
    population_size=1000,
    bin_fractions=[
        {"bin": "S", "fraction": 0.99},
        {"bin": "I", "fraction": 0.01},
        {"bin": "R", "fraction": 0.0}
    ]
)
```

### With Stratifications

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
```

## Building the Model

Once all components are added, build the model:

```python
from commol.constants import ModelTypes

model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
```

### Validation

The build process validates:

- All disease state fractions sum to 1.0
- All stratification fractions sum to 1.0
- Transition sources/targets reference valid states
- Mathematical expressions are syntactically correct
- No security issues in formulas

If validation fails, a descriptive error is raised.

## Unit Checking

EpiModel provides automatic dimensional analysis to catch unit errors in your model equations. This validates that rate expressions produce the correct units and that mathematical functions receive dimensionally correct arguments.

### Enabling Unit Checking

Unit checking is enabled when **all parameters have units**:

```python
# Build model with units
builder = ModelBuilder(name="SIR with Units", version="1.0")

builder.add_bin("S", "Susceptible")
builder.add_bin("I", "Infected")
builder.add_bin("R", "Recovered")

# Specify units for all parameters
builder.add_parameter("beta", 0.5, "Transmission rate", unit="1/day")
builder.add_parameter("gamma", 0.1, "Recovery rate", unit="1/day")

builder.add_transition(
    "infection", ["S"], ["I"],
    rate="beta * S * I / N"
)
builder.add_transition("recovery", ["I"], ["R"], rate="gamma * I")

builder.set_initial_conditions(
    population_size=1000,
    bin_fractions=[
        {"bin": "S", "fraction": 0.99},
        {"bin": "I", "fraction": 0.01},
        {"bin": "R", "fraction": 0.0},
    ],
)

model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)

# Validate dimensional consistency
model.check_unit_consistency()  # Raises error if units are inconsistent
```

### Common Units

```python
# Rate units
unit="1/day"         # Per-day rates
unit="1/week"        # Per-week rates

# Population units (automatically assigned to disease states)
unit="person"        # Population count

# Dimensionless quantities
unit="dimensionless" # Ratios, fractions, amplitudes
```

### Mathematical Functions

All standard math functions work with unit checking and validate their arguments:

```python
# Seasonal forcing (sin requires dimensionless argument)
builder.add_parameter("beta_avg", 0.5, unit="1/day")
builder.add_parameter("seasonal_amp", 0.2, unit="dimensionless")

builder.add_transition(
    "infection", ["S"], ["I"],
    rate="beta_avg * (1 + seasonal_amp * sin(2 * pi * step / 365)) * S * I / N"
)

# Exponential decay (exp requires dimensionless argument)
builder.add_parameter("beta_0", 0.5, unit="1/day")
builder.add_parameter("decay_rate", 0.01, unit="dimensionless")

builder.add_transition(
    "infection", ["S"], ["I"],
    rate="beta_0 * exp(-decay_rate * step) * S * I / N"
)
```

**Supported functions**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `pow`, `min`, `max`, `abs`, and more.

### Automatic Unit Assignment

The system automatically assigns units to:

- **Disease states**: All have units of `person` (S, I, R, etc.)
- **Population variables**: `N`, `N_young`, `N_urban`, etc. have units of `person`
- **Time variables**: `t` and `step` are dimensionless
- **Constants**: `pi` and `e` are dimensionless

### Error Detection

Unit checking catches common errors:

```python
# Wrong parameter units
builder.add_parameter("beta", 0.5, unit="day")  # Should be "1/day"!
# Error: Unit mismatch: equation has unit 'day * person' but expected 'person/day'

# Dimensional argument to math function
rate="beta * sin(I) * S"  # I has units of person!
# Error: Cannot convert from 'person' to 'dimensionless'

# Incompatible units in operations
rate="min(beta, threshold) * S"  # beta is 1/day, threshold is person
# Error: Cannot compare incompatible units
```

### Best Practices

1. **Always specify units** for physical quantities
2. **Use "dimensionless"** for ratios and fractions
3. **Ensure math function arguments are dimensionless** (divide by appropriate quantities)
4. **Use consistent time units** throughout your model

### Unit Display in Equations

When you print equations using `model.print_equations()`, unit annotations are displayed based on unit completeness:

```python
# Model with complete units - shows annotations
model = (
    ModelBuilder(name="SIR", bin_unit="person")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_parameter(id="beta", value=0.5, unit="1/day")
    .add_parameter(id="gamma", value=0.1, unit="1/day")
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .build()
)

model.print_equations()
# Output:
#   S -> I: beta(1/day) * S(person) * I(person) / N(person) [person/day]

# Model without units - no annotations
model = (
    ModelBuilder(name="SIR")
    .add_bin(id="S", name="Susceptible")
    .add_parameter(id="beta", value=0.5)
    .build()
)

model.print_equations()
# Output:
#   S -> I: beta * S * I / N
```

### Partial Unit Definitions

**Important**: You must define units for ALL parameters and bins, or for NONE. Partial unit definitions will raise a `ValueError`:

```python
# This will raise an error!
model = (
    ModelBuilder(name="SIR", bin_unit="person")
    .add_parameter(id="beta", value=0.5, unit="1/day")  # Has unit
    .add_parameter(id="gamma", value=0.1)  # No unit - INCONSISTENT!
    .build()
)

model.print_equations()  # ValueError: Some parameters have units but not all
```

This prevents accidentally mixing unit systems or forgetting to specify units for some parameters.

## Advanced: Conditional Transitions

Create transitions that only occur under certain conditions:

```python
# Create a condition
condition = builder.create_condition(
    logic="and",
    rules=[
        {"variable": "state:I", "operator": "gt", "value": 100},
        {"variable": "step", "operator": "gt", "value": 30}
    ]
)

# Add conditional transition
builder.add_transition(
    id="intervention",
    source=["S"],
    target=["S"],
    rate="0.5 * beta",  # Reduced transmission
    condition=condition
)
```

## Loading from JSON

Load pre-defined models from JSON files:

```python
from commol import ModelLoader

model = ModelLoader.from_json("path/to/model.json")
```

### JSON Structure

```json
{
  "name": "SIR Model",
  "version": "1.0",
  "population": {
    "disease_states": [
      { "id": "S", "name": "Susceptible" },
      { "id": "I", "name": "Infected" },
      { "id": "R", "name": "Recovered" }
    ]
  },
  "parameters": [
    { "id": "beta", "value": 0.3 },
    { "id": "gamma", "value": 0.1 }
  ],
  "dynamics": {
    "typology": "difference_equations",
    "transitions": [
      {
        "id": "infection",
        "source": ["S"],
        "target": ["I"],
        "rate": "beta * S * I / N"
      }
    ]
  }
}
```

## Complete Example

```python
from commol import ModelBuilder, Simulation
from commol.constants import ModelTypes

# Build SEIR model
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

# Run simulation
simulation = Simulation(model)
results = simulation.run(num_steps=200)
```

## Next Steps

- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Simulations](simulations.md) - Running and analyzing models
- [Examples](examples.md) - Complete model examples
