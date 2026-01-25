# Building Models

The `ModelBuilder` class provides a fluent API for constructing compartment models.

## ModelBuilder Basics

### Creating a Builder

```python
from commol import ModelBuilder

builder = ModelBuilder(
    name="My Model",
    version="1.0",
    description="Optional description",
    bin_unit="person"  # Optional: default unit for all bins
)
```

#### Parameters

- **`name`** (required): Unique identifier for your model
- **`version`** (optional): Version string for tracking model changes
- **`description`** (optional): Human-readable description of the model
- **`bin_unit`** (optional): Default unit for all bins (compartments). When specified, this enables:
  - Automatic unit assignment to bins, predefined population variables (`N`, `N_young`, etc.), and stratification categories
  - Unit checking via `model.check_unit_consistency()`
  - Unit annotations in `model.print_equations()` output

  Common values: `"person"`, `"individual"`, `"molecule"`, or any custom unit.

  **Note**: Individual bins can override this with their own `unit` parameter in `add_bin()`.

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
    .build("DifferenceEquations")
)
```

## Adding Compartments

Compartments (also called bins or states) represent the different states in your model:

```python
builder.add_bin(
    id="S",                    # Required: Unique identifier
    name="Susceptible",        # Required: Display name
    description="Initial population state"  # Optional
)
```

### Best Practices

- Use short, clear IDs (S, I, R, A, B, etc.)
- Provide descriptive names

## Adding Stratifications

Stratifications divide your population into distinct subgroups, allowing different rates and dynamics for each group. When you add stratifications, Commol automatically expands your compartments to track each subgroup separately.

```python
builder.add_stratification(
    id="age_group",
    categories=["0-17", "18-64", "65+"],
    description="Age-based stratification"
)
```

### How Compartment Expansion Works

When you add a stratification, every compartment is expanded by appending each category as a suffix:

```
Before: S, I, R (3 compartments)
After adding age=[young, old]: S_young, S_old, I_young, I_old, R_young, R_old (6 compartments)
```

### Multiple Stratifications

With multiple stratifications, compartments are expanded using the **Cartesian product**—every combination of categories is created:

```python
builder.add_stratification(id="age", categories=["young", "old"])
builder.add_stratification(id="location", categories=["urban", "rural"])
```

This creates 12 compartments for a 3-bin model:

| Base Bin | Expanded Compartments                                          |
| -------- | -------------------------------------------------------------- |
| S        | `S_young_urban`, `S_young_rural`, `S_old_urban`, `S_old_rural` |
| I        | `I_young_urban`, `I_young_rural`, `I_old_urban`, `I_old_rural` |
| R        | `R_young_urban`, `R_young_rural`, `R_old_urban`, `R_old_rural` |

**Key points:**

- Category suffixes are added in the **order stratifications are defined**
- With 3 bins, 2 age categories, and 2 location categories: 3 × 2 × 2 = 12 compartments
- Compartment names are case-sensitive: `S_young` ≠ `S_Young`

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
- **Compartments**: Use compartment populations (e.g., `"S"`, `"I"`)
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

### Using `$compartment` Placeholder for Per-Compartment Rates

When applying the same type of transition to multiple compartments with per-compartment rates, use the `$compartment` placeholder to avoid repetitive code:

```python
# Instead of writing 4 separate transitions:
# .add_transition("death_S", ["S"], [], rate="d * S")
# .add_transition("death_L", ["L"], [], rate="d * L")
# .add_transition("death_I", ["I"], [], rate="d * I")
# .add_transition("death_R", ["R"], [], rate="d * R")

# Write one transition that automatically expands:
builder.add_transition(
    id="death",
    source=["S", "L", "I", "R"],
    target=[],
    rate="d * $compartment"  # $compartment gets replaced with S, L, I, R
)
```

**How it works:**

- The system detects `$compartment` in the rate formula
- Automatically creates one transition per source compartment
- Replaces `$compartment` with the actual compartment name in each transition
- Generated transition IDs use the pattern: `{id}__{compartment}` (e.g., `death__S`, `death__L`)

**Complex formulas with multiple occurrences:**

```python
builder.add_transition(
    id="nonlinear_death",
    source=["S", "I", "R"],
    target=[],
    rate="d * $compartment * (1 + 0.1 * $compartment / N)"
)
# Expands to:
# death__S: rate = "d * S * (1 + 0.1 * S / N)"
# death__I: rate = "d * I * (1 + 0.1 * I / N)"
# death__R: rate = "d * R * (1 + 0.1 * R / N)"
```

**With single target (transfers):**

```python
builder.add_transition(
    id="treatment",
    source=["I_mild", "I_severe"],
    target=["R"],  # All transfer to same compartment
    rate="treatment_rate * $compartment"
)
```

**With stratified rates:**

```python
builder.add_stratification(id="age", categories=["young", "old"])

builder.add_transition(
    id="death",
    source=["S", "I", "R"],
    target=[],
    rate="d_base * $compartment",  # Fallback rate
    stratified_rates=[
        {
            "conditions": [{"stratification": "age", "category": "young"}],
            "rate": "d_young * $compartment"  # Lower death rate for young
        },
        {
            "conditions": [{"stratification": "age", "category": "old"}],
            "rate": "d_old * $compartment"  # Higher death rate for old
        }
    ]
)
# Expands to death__S, death__I, death__R, each with their own stratified rates
```

**Restrictions:**

- Only valid with multiple source compartments (2 or more)
- Target must be empty `[]` or contain exactly one compartment
- Cannot be used if you want different targets for different sources

**Comparison with standard multi-source transitions:**

Standard multi-source transitions (without `$compartment`) create a **single** transition that affects all sources simultaneously:

```python
# This creates ONE transition
.add_transition(
    id="interaction",
    source=["S", "I"],
    target=["I", "I"],
    rate="beta * S * I"
)
# Resulting equations:
# dS/dt = ... - (beta*S*I)
# dI/dt = ... - (beta*S*I) + 2*(beta*S*I) = ... + (beta*S*I)
```

With `$compartment`, you create **multiple independent** transitions:

```python
# This creates TWO separate transitions
.add_transition(
    id="death",
    source=["S", "I"],
    target=[],
    rate="d * $compartment"
)
# Resulting equations:
# dS/dt = ... - (d*S)
# dI/dt = ... - (d*I)
```

### Stratified Transitions

When a model includes stratifications, you often need different transition rates for different subgroups. The `add_transition` method supports this via the `stratified_rates` parameter.

#### How Stratified Rate Matching Works

When a transition is applied to a stratified compartment, Commol determines which rate to use by:

1. **Extracting categories** from the compartment name (e.g., `I_young_urban` → age=young, location=urban)
2. **Finding the best match** among stratified rates based on how many conditions match
3. **Falling back to the default rate** if no stratified rate matches

The system uses a **most-specific-match** strategy: if multiple stratified rates match, the one with the most matching conditions is used.

```
Compartment: I_young_urban
Stratified rates:
  1. [age=young] → matches 1 condition
  2. [age=young, location=urban] → matches 2 conditions ← SELECTED
  3. [age=old] → matches 0 conditions

Result: Rate #2 is used because it's most specific
```

#### Single Stratification Example

Define different recovery rates for different age groups:

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

This creates three transition flows:

- `I_child → R_child` with rate `gamma_child` (0.15)
- `I_adult → R_adult` with rate `gamma_adult` (0.1)
- `I_elderly → R_elderly` with rate `gamma_elderly` (0.08)

#### Multi-Stratification Transitions

To define rates for intersections of multiple stratifications, add multiple conditions to a single rate entry. Conditions within the same entry use **AND** logic—all must match.

```python
builder.add_stratification(id="age", categories=["child", "adult"])
builder.add_stratification(id="risk", categories=["low", "high"])
builder.add_stratification(id="location", categories=["urban", "rural"])

builder.add_parameter(id="beta_urban_adult_high_risk", value=0.8)
builder.add_parameter(id="beta_default", value=0.3)

builder.add_transition(
    id="transition_S_to_I",
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

**Rate assignment for each compartment:**

| Compartment          | Matching Conditions | Rate Used                                |
| -------------------- | ------------------- | ---------------------------------------- |
| `S_adult_high_urban` | 3 (all match)       | `beta_urban_adult_high_risk * S * I / N` |
| `S_adult_high_rural` | 2 (age, risk)       | `beta_default * S * I / N` (fallback)    |
| `S_child_low_urban`  | 1 (location only)   | `beta_default * S * I / N` (fallback)    |
| `S_child_low_rural`  | 0                   | `beta_default * S * I / N` (fallback)    |

Only `S_adult_high_urban` matches all three conditions, so it gets the special high-risk rate. All other compartments use the fallback rate.

#### Fallback Rate Behavior

- The `rate` parameter acts as a **fallback** for any compartment that doesn't match a specific stratified rate
- If you define `stratified_rates` for all categories, the fallback rate is never used
- It's good practice to always provide a fallback rate for defensive coding

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


model = builder.build(typology="DifferenceEquations")
```

### Validation

The build process validates:

- All compartment fractions sum to 1.0
- All stratification fractions sum to 1.0
- Transition sources/targets reference valid compartments
- Mathematical expressions are syntactically correct
- No security issues in formulas

If validation fails, a descriptive error is raised.

## Unit Checking

Commol provides automatic dimensional analysis to catch unit errors in your model equations. This validates that rate expressions produce the correct units and that mathematical functions receive dimensionally correct arguments.

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

model = builder.build(typology="DifferenceEquations")

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

- **Compartments**: All have the specified bin_unit (e.g., `person`, `molecule`)
- **Population variables**: `N`, `N_young`, `N_urban`, etc. have the same unit
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
    ModelBuilder(name="Model", bin_unit="person")
    .add_parameter(id="beta", value=0.5, unit="1/day")  # Has unit
    .add_parameter(id="gamma", value=0.1)  # No unit - INCONSISTENT!
    .build()
)

model.print_equations()  # ValueError: Some parameters have units but not all
```

This prevents accidentally mixing unit systems or forgetting to specify units for some parameters.

### LaTeX Output Format

Export equations in LaTeX format for inclusion in documents and publications:

```python
# Default text format
model.print_equations()
# Output: dS/dt = - (beta * S * I / N)

# LaTeX format
model.print_equations(format="latex")
# Output: \[\frac{dS}{dt} = - (\beta \cdot S \cdot I / N)\]

# Save to file
model.print_equations(output_file="equations.txt", format="latex")
```

**LaTeX features:**

- Compact form uses inline math: `$S \to I: \beta \cdot S \cdot I / N$`
- Expanded form uses display math: `\[\frac{dS}{dt} = ...\]`
- Equations are copy-paste ready into LaTeX documents
- Subscripts formatted as: `S_{young,urban}`
- Multiplication shown as: `\cdot`

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
from commol import Model

model = Model.from_json("path/to/model.json")
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
    .build(typology="DifferenceEquations")
)

# Run simulation
simulation = Simulation(model)
results = simulation.run(num_steps=200)
```

## Next Steps

- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Simulations](simulations.md) - Running and analyzing models
- [Examples](examples.md) - Complete model examples
