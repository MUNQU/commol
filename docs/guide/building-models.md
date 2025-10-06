# Building Models

The `ModelBuilder` class provides a fluent API for constructing epidemiological models.

## ModelBuilder Basics

### Creating a Builder

```python
from epimodel import ModelBuilder

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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .build(ModelTypes.DIFFERENCE_EQUATIONS)
)
```

## Adding Disease States

```python
builder.add_disease_state(
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

### Parameter Guidelines

- Use meaningful IDs (beta, gamma, R0, etc.)
- Document units and meaning
- Ensure values are realistic for your model

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

## Setting Initial Conditions

### Basic Setup

```python
builder.set_initial_conditions(
    population_size=1000,
    disease_state_fractions={
        "S": 0.99,
        "I": 0.01,
        "R": 0.0
    }
)
```

### With Stratifications

```python
builder.set_initial_conditions(
    population_size=10000,
    disease_state_fractions={
        "S": 0.99,
        "I": 0.01,
        "R": 0.0
    },
    stratification_fractions={
        "age_group": {
            "young": 0.3,
            "adult": 0.5,
            "elderly": 0.2
        },
        "risk": {
            "low": 0.8,
            "high": 0.2
        }
    }
)
```

## Building the Model

Once all components are added, build the model:

```python
from epimodel.constants import ModelTypes

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
from epimodel import ModelLoader

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
from epimodel import ModelBuilder, Simulation
from epimodel.constants import ModelTypes

# Build SEIR model
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
        disease_state_fractions={"S": 0.999, "E": 0.0, "I": 0.001, "R": 0.0}
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
