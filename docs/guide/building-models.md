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

  Common values: `"unit"`, `"item"`, or any custom unit string.

  **Note**: Individual bins can override this with their own `unit` parameter in `add_bin()`.

### Chaining Methods

The builder uses method chaining for a clean, readable API:

```python
model = (
    ModelBuilder(name="My Model")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / N")
    .build("DifferenceEquations")
)
```

## Adding Compartments

Compartments (also called bins or states) represent the different states in your model:

```python
builder.add_bin(
    id="A",                    # Required: Unique identifier
    name="State A",            # Required: Display name
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
    id="group",
    categories=["g1", "g2", "g3"],
    description="Primary grouping dimension"
)
```

### How Compartment Expansion Works

When you add a stratification, every compartment is expanded by appending each category as a suffix:

```
Before: A, B, C (3 compartments)
After adding group=[g1, g2]: A_g1, A_g2, B_g1, B_g2, C_g1, C_g2 (6 compartments)
```

### Multiple Stratifications

With multiple stratifications, compartments are expanded using the **Cartesian product** by default — every combination of categories is created:

```python
builder.add_stratification(id="group", categories=["g1", "g2"])
builder.add_stratification(id="type", categories=["t1", "t2"])
```

This creates 12 compartments for a 3-bin model:

| Base Bin | Expanded Compartments                                    |
| -------- | -------------------------------------------------------- |
| A        | `A_g1_t1`, `A_g1_t2`, `A_g2_t1`, `A_g2_t2`             |
| B        | `B_g1_t1`, `B_g1_t2`, `B_g2_t1`, `B_g2_t2`             |
| C        | `C_g1_t1`, `C_g1_t2`, `C_g2_t1`, `C_g2_t2`             |

**Key points:**

- Category suffixes are added in the **order stratifications are defined**
- With 3 bins, 2 group categories, and 2 type categories: 3 × 2 × 2 = 12 compartments
- Compartment names are case-sensitive: `A_g1` ≠ `A_G1`

### Conditional Stratifications

A stratification can be marked as conditional so that it only expands compartments whose already-assigned categories satisfy a given set of conditions. Compartments that do not satisfy the conditions are kept as-is — the conditional stratification is simply not applied to them.

```python
builder.add_stratification(id="group", categories=["g1", "g2"])

# Only expand g2 compartments further
builder.add_stratification(
    id="subtype",
    categories=["s1", "s2"],
    conditions=[{"stratification": "group", "category": "g2"}]
)
```

Result for a 2-bin model with bins A and B:

| Compartment  | How it was created                        |
| ------------ | ----------------------------------------- |
| `A_g1`       | group=g1 → subtype condition not met      |
| `A_g2_s1`    | group=g2 → subtype condition met → s1     |
| `A_g2_s2`    | group=g2 → subtype condition met → s2     |
| `B_g1`       | group=g1 → subtype condition not met      |
| `B_g2_s1`    | group=g2 → subtype condition met → s1     |
| `B_g2_s2`    | group=g2 → subtype condition met → s2     |

**Rules:**
- `conditions` is a list of `{"stratification": str, "category": str}` dicts — all must match (AND logic)
- Conditions may only reference stratifications **declared before** this one; a `ValueError` is raised otherwise
- Initial condition fractions for the conditional stratification apply only to the compartments where it was expanded; population in non-expanded compartments is unchanged

## Adding Parameters

Parameters are global constants used in formulas:

```python
builder.add_parameter(
    id="k1",
    value=0.3,
    description="Forward rate constant"
)
```

### Parameters with Units

You can specify units for automatic dimensional analysis and validation:

```python
builder.add_parameter(
    id="k1",
    value=0.5,
    description="Forward rate constant",
    unit="1/day"  # Rate unit
)

builder.add_parameter(
    id="amp",
    value=0.2,
    description="Periodic amplitude",
    unit="dimensionless"  # Pure number
)
```

When **all parameters have units**, the model will automatically validate dimensional consistency. See [Unit Checking](#unit-checking) below.

**Tip:** To mark a parameter as unitless (dimensionless) for unit checking, use `unit="dimensionless"`. This is useful for ratios, fractions, scaling factors, and amplitudes. Dimensionless parameters are also required as arguments to mathematical functions like `sin()`, `cos()`, `exp()`, `sqrt()`, `pow()`, etc.

### Parameter Guidelines

- Use meaningful IDs that reflect their role in the model
- Document units and meaning
- Ensure values are realistic for your model
- Specify units for automatic validation (recommended)

## Adding Accumulators

Accumulators are cumulative counters that track the total flow through one or more transitions over time. They appear in simulation output but are not part of the population (they never subtract from compartment values).

```python
builder.add_accumulator(
    id="cum_ab",    # Required: unique identifier
    name="Cumulative A→B events"  # Required: display name
)
```

To increment an accumulator, add it to the `accumulators` list of one or more transitions:

```python
builder.add_transition(
    id="flow_ab",
    source=["A"],
    target=["B"],
    rate="k1 * A * B / N",
    accumulators=["cum_ab"],  # incremented by every A→B flow
)
```

Multiple transitions can share an accumulator, and one transition can increment multiple accumulators:

```python
builder.add_accumulator(id="total_out", name="Total outflow from A")
builder.add_accumulator(id="cum_ab",    name="Cumulative A→B")

builder.add_transition(
    id="flow_ab",
    source=["A"],
    target=["B"],
    rate="k1 * A * B / N",
    accumulators=["cum_ab", "total_out"],
)
builder.add_transition(
    id="flow_ac",
    source=["A"],
    target=["C"],
    rate="k2",
    accumulators=["total_out"],
)
```

With stratifications, accumulators are expanded identically to bins. `cum_ab` with a stratification `group=[g1, g2]` produces output columns `cum_ab_g1` and `cum_ab_g2`.

## Adding Transitions

Transitions move populations between states.

### Understanding Transition Rates

The `rate` parameter accepts **mathematical expressions** that can include:

- **Parameters**: Reference parameter IDs (e.g., `"k1"`)
- **Compartments**: Use compartment populations (e.g., `"S"`, `"I"`)
- **Special variables**: `N` (total population), `step` or `t` (current time step), `pi`, `e`
- **Mathematical operations**: `+`, `-`, `*`, `/`, `**` (power)
- **Functions**: `sin()`, `cos()`, `exp()`, `log()`, `sqrt()`, `max()`, `min()`, etc.

For the complete list of functions and advanced examples, see [Mathematical Expressions](mathematical-expressions.md).

### Simple Parameter-Based Rates

```python
builder.add_transition(
    id="forward",
    source=["A"],
    target=["B"],
    rate="k1"  # References parameter id
)
```

### Formula-Based Rates

```python
builder.add_transition(
    id="transfer",
    source=["A"],
    target=["B"],
    rate="k1 * A * B / N"  # Mathematical expression
)
```

### Constant Rates

```python
builder.add_transition(
    id="inflow",
    source=[],      # Empty = enters system
    target=["A"],
    rate="0.001"    # Fixed rate
)
```

### Time-Dependent Rates

```python
builder.add_transition(
    id="periodic_flow",
    source=["A"],
    target=["B"],
    rate="k1 * (1 + 0.3 * sin(2 * pi * t / 365)) * A * B / N"
)
```

See [Mathematical Expressions](mathematical-expressions.md) for more complex rate formulas.

### Multi-State Transitions

```python
# Outflow from multiple compartments
builder.add_transition(
    id="outflow",
    source=["A", "B", "C"],
    target=[],  # Empty = leaves system
    rate="mu"
)
```

### Using `$compartment` Placeholder for Per-Compartment Rates

When applying the same type of transition to multiple compartments with per-compartment rates, use the `$compartment` placeholder to avoid repetitive code:

```python
# Instead of writing separate transitions:
# .add_transition("outflow_A", ["A"], [], rate="mu * A")
# .add_transition("outflow_B", ["B"], [], rate="mu * B")
# .add_transition("outflow_C", ["C"], [], rate="mu * C")

# Write one transition that automatically expands:
builder.add_transition(
    id="outflow",
    source=["A", "B", "C"],
    target=[],
    rate="mu * $compartment"  # $compartment gets replaced with A, B, C
)
```

**How it works:**

- The system detects `$compartment` in the rate formula
- Automatically creates one transition per source compartment
- Replaces `$compartment` with the actual compartment name in each transition
- Generated transition IDs use the pattern: `{id}__{compartment}` (e.g., `outflow__A`, `outflow__B`)

**Complex formulas with multiple occurrences:**

```python
builder.add_transition(
    id="nonlinear_outflow",
    source=["A", "B", "C"],
    target=[],
    rate="mu * $compartment * (1 + 0.1 * $compartment / N)"
)
# Expands to:
# nonlinear_outflow__A: rate = "mu * A * (1 + 0.1 * A / N)"
# nonlinear_outflow__B: rate = "mu * B * (1 + 0.1 * B / N)"
# nonlinear_outflow__C: rate = "mu * C * (1 + 0.1 * C / N)"
```

**With single target (transfers):**

```python
builder.add_transition(
    id="merge",
    source=["B1", "B2"],
    target=["C"],  # All transfer to same compartment
    rate="k * $compartment"
)
```

**With stratified rates:**

```python
builder.add_stratification(id="group", categories=["g1", "g2"])

builder.add_transition(
    id="outflow",
    source=["A", "B", "C"],
    target=[],
    rate="mu_base * $compartment",  # Fallback rate
    stratified_rates=[
        {
            "conditions": [{"stratification": "group", "category": "g1"}],
            "rate": "mu_g1 * $compartment"
        },
        {
            "conditions": [{"stratification": "group", "category": "g2"}],
            "rate": "mu_g2 * $compartment"
        }
    ]
)
# Expands to outflow__A, outflow__B, outflow__C, each with their own stratified rates
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
    source=["A", "B"],
    target=["B", "B"],
    rate="k1 * A * B"
)
# Resulting equations:
# dA/dt = ... - (k1*A*B)
# dB/dt = ... - (k1*A*B) + 2*(k1*A*B) = ... + (k1*A*B)
```

With `$compartment`, you create **multiple independent** transitions:

```python
# This creates TWO separate transitions
.add_transition(
    id="outflow",
    source=["A", "B"],
    target=[],
    rate="mu * $compartment"
)
# Resulting equations:
# dA/dt = ... - (mu*A)
# dB/dt = ... - (mu*B)
```

### Per-Compartment Rates with `per_compartment`

When a model has stratifications, base compartment names like `A` or `B` in rate expressions resolve to the **total** across all stratified versions (e.g., `A = A_g1 + A_g2 + ...`). This is correct when a transition depends on the global total, but **incorrect** when each subpopulation should evolve independently based on its own value.

The `per_compartment` flag solves this by automatically replacing base compartment names with the specific stratified compartment name for each expanded transition flow:

```python
builder.add_transition(
    id="forward",
    source=["A"],
    target=["B"],
    rate="k1 * A",
    per_compartment=True  # A becomes A_g1, A_g2, etc. per flow
)
```

**Without** `per_compartment` (default):

| Flow           | Rate Expression | `A` resolves to          |
| -------------- | --------------- | ------------------------ |
| `A_g1 → B_g1` | `k1 * A`        | `A_g1 + A_g2` (total)   |
| `A_g2 → B_g2` | `k1 * A`        | `A_g1 + A_g2` (total)   |

**With** `per_compartment=True`:

| Flow           | Rate Expression | Meaning                |
| -------------- | --------------- | ---------------------- |
| `A_g1 → B_g1` | `k1 * A_g1`     | Only this subgroup     |
| `A_g2 → B_g2` | `k1 * A_g2`     | Only this subgroup     |

Both source and target bin names are replaced. For example, with `rate="k1 * A + k2 * B"` and `per_compartment=True`, the flow `A_g1 → B_g1` uses `k1 * A_g1 + k2 * B_g1`.

**Works with stratified rates:**

```python
builder.add_transition(
    id="forward",
    source=["A"],
    target=["B"],
    rate="k1 * A",  # Fallback
    stratified_rates=[
        {
            "conditions": [{"stratification": "group", "category": "g1"}],
            "rate": "k1_g1 * A"
        },
        {
            "conditions": [{"stratification": "group", "category": "g2"}],
            "rate": "k1_g2 * A"
        }
    ],
    per_compartment=True  # A is replaced in each stratified rate
)
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

Define different rates for different categories:

```python
builder.add_stratification(id="group", categories=["g1", "g2", "g3"])
builder.add_parameter(id="k_g1", value=0.15)
builder.add_parameter(id="k_g2", value=0.10)
builder.add_parameter(id="k_g3", value=0.08)

builder.add_transition(
    id="forward",
    source=["A"],
    target=["B"],
    stratified_rates=[
        {
            "conditions": [{"stratification": "group", "category": "g1"}],
            "rate": "k_g1"
        },
        {
            "conditions": [{"stratification": "group", "category": "g2"}],
            "rate": "k_g2"
        },
        {
            "conditions": [{"stratification": "group", "category": "g3"}],
            "rate": "k_g3"
        },
    ]
)
```

This creates three transition flows:

- `A_g1 → B_g1` with rate `k_g1` (0.15)
- `A_g2 → B_g2` with rate `k_g2` (0.10)
- `A_g3 → B_g3` with rate `k_g3` (0.08)

#### Multi-Stratification Transitions

To define rates for intersections of multiple stratifications, add multiple conditions to a single rate entry. Conditions within the same entry use **AND** logic — all must match.

```python
builder.add_stratification(id="group", categories=["g1", "g2"])
builder.add_stratification(id="type", categories=["t1", "t2"])
builder.add_stratification(id="variant", categories=["v1", "v2"])

builder.add_parameter(id="k_special", value=0.8)
builder.add_parameter(id="k_default", value=0.3)

builder.add_transition(
    id="flow_AB",
    source=["A"],
    target=["B"],
    rate="k_default * A * B / N",  # Fallback rate
    stratified_rates=[
        {
            "conditions": [
                {"stratification": "group", "category": "g2"},
                {"stratification": "type", "category": "t1"},
                {"stratification": "variant", "category": "v2"},
            ],
            "rate": "k_special * A * B / N"
        }
    ]
)
```

**Rate assignment for each compartment:**

| Compartment         | Matching Conditions | Rate Used                     |
| ------------------- | ------------------- | ----------------------------- |
| `A_g2_t1_v2`        | 3 (all match)       | `k_special * A * B / N`       |
| `A_g2_t1_v1`        | 2 (group, type)     | `k_default * A * B / N`       |
| `A_g1_t2_v2`        | 1 (variant only)    | `k_default * A * B / N`       |
| `A_g1_t1_v1`        | 0                   | `k_default * A * B / N`       |

Only `A_g2_t1_v2` matches all three conditions, so it gets the special rate. All others use the fallback.

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
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
        {"bin": "C", "fraction": 0.0},
    ]
)
```

### With Stratifications

```python
builder.set_initial_conditions(
    population_size=10000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
        {"bin": "C", "fraction": 0.0},
    ],
    stratification_fractions=[
        {
            "stratification": "group",
            "fractions": [
                {"category": "g1", "fraction": 0.3},
                {"category": "g2", "fraction": 0.5},
                {"category": "g3", "fraction": 0.2},
            ]
        },
        {
            "stratification": "type",
            "fractions": [
                {"category": "t1", "fraction": 0.8},
                {"category": "t2", "fraction": 0.2},
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
builder = ModelBuilder(name="Model with Units", version="1.0", bin_unit="unit")

builder.add_bin("A", "State A")
builder.add_bin("B", "State B")
builder.add_bin("C", "State C")

builder.add_parameter("k1", 0.5, "Forward rate", unit="1/day")
builder.add_parameter("k2", 0.1, "Reverse rate", unit="1/day")

builder.add_transition("flow_AB", ["A"], ["B"], rate="k1 * A * B / N")
builder.add_transition("flow_BC", ["B"], ["C"], rate="k2 * B")

builder.set_initial_conditions(
    population_size=1000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
        {"bin": "C", "fraction": 0.0},
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

# Entity count units (automatically assigned to bins)
unit="unit"          # Generic entity count

# Dimensionless quantities
unit="dimensionless" # Ratios, fractions, amplitudes
```

### Mathematical Functions

All standard math functions work with unit checking and validate their arguments:

```python
# Periodic forcing (sin requires dimensionless argument)
builder.add_parameter("k_avg", 0.5, unit="1/day")
builder.add_parameter("amp", 0.2, unit="dimensionless")

builder.add_transition(
    "periodic_flow", ["A"], ["B"],
    rate="k_avg * (1 + amp * sin(2 * pi * step / 365)) * A * B / N"
)

# Exponential decay (exp requires dimensionless argument)
builder.add_parameter("k0", 0.5, unit="1/day")
builder.add_parameter("decay", 0.01, unit="dimensionless")

builder.add_transition(
    "decaying_flow", ["A"], ["B"],
    rate="k0 * exp(-decay * step) * A * B / N"
)
```

**Supported functions**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `pow`, `min`, `max`, `abs`, and more.

### Automatic Unit Assignment

The system automatically assigns units to:

- **Compartments**: All have the specified `bin_unit`
- **Population variables**: `N`, `N_g1`, `N_g1_t1`, etc. have the same unit
- **Time variables**: `t` and `step` are dimensionless
- **Constants**: `pi` and `e` are dimensionless

### Error Detection

Unit checking catches common errors:

```python
# Wrong parameter units
builder.add_parameter("k1", 0.5, unit="day")  # Should be "1/day"!
# Error: Unit mismatch: equation has unit 'day * unit' but expected 'unit/day'

# Dimensional argument to math function
rate="k1 * sin(B) * A"  # B has units of 'unit'!
# Error: Cannot convert from 'unit' to 'dimensionless'

# Incompatible units in operations
rate="min(k1, threshold) * A"  # k1 is 1/day, threshold is 'unit'
# Error: Cannot compare incompatible units
```

### Best Practices

1. **Always specify units** for all quantities when using unit checking
2. **Use "dimensionless"** for ratios and fractions
3. **Ensure math function arguments are dimensionless** (divide by appropriate quantities)
4. **Use consistent time units** throughout your model

### Unit Display in Equations

When you print equations using `model.print_equations()`, unit annotations are displayed when all units are defined:

```python
# Model with complete units - shows annotations
model = (
    ModelBuilder(name="Model", bin_unit="unit")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.5, unit="1/day")
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / N")
    .build()
)

model.print_equations()
# Output:
#   A -> B: k1(1/day) * A(unit) * B(unit) / N(unit) [unit/day]

# Model without units - no annotations
model = (
    ModelBuilder(name="Model")
    .add_bin(id="A", name="State A")
    .add_parameter(id="k1", value=0.5)
    .build()
)

model.print_equations()
# Output:
#   A -> B: k1 * A * B / N
```

### Partial Unit Definitions

**Important**: You must define units for ALL parameters and bins, or for NONE. Partial unit definitions will raise a `ValueError`:

```python
# This will raise an error!
model = (
    ModelBuilder(name="Model", bin_unit="unit")
    .add_parameter(id="k1", value=0.5, unit="1/day")  # Has unit
    .add_parameter(id="k2", value=0.1)  # No unit - INCONSISTENT!
    .build()
)

model.print_equations()  # ValueError: Some parameters have units but not all
```

### LaTeX Output Format

Export equations in LaTeX format for inclusion in documents and publications:

```python
# Default text format
model.print_equations()
# Output: dA/dt = - (k1 * A * B / N)

# LaTeX format
model.print_equations(format="latex")
# Output: \[\frac{dA}{dt} = - (k_1 \cdot A \cdot B / N)\]

# Save to file
model.print_equations(output_file="equations.txt", format="latex")
```

**LaTeX features:**

- Compact form uses inline math: `$A \to B: k_1 \cdot A \cdot B / N$`
- Expanded form uses display math: `\[\frac{dA}{dt} = ...\]`
- Equations are copy-paste ready into LaTeX documents
- Subscripts formatted as: `A_{g1,t1}`
- Multiplication shown as: `\cdot`

## Advanced: Conditional Transitions

Create transitions that only occur under certain conditions:

```python
# Create a condition
condition = builder.create_condition(
    logic="and",
    rules=[
        {"variable": "state:B", "operator": "gt", "value": 100},
        {"variable": "step", "operator": "gt", "value": 30}
    ]
)

# Add conditional transition
builder.add_transition(
    id="gated_flow",
    source=["A"],
    target=["C"],
    rate="0.5 * k1",
    condition=condition
)
```

## Saving and Loading JSON

Load pre-defined models from JSON files, and save a model back:

```python
from commol import Model

model = Model.from_json("path/to/model.json")
model.to_json("path/to/model.json")
```

`to_json` writes every field, so the file always reloads through `from_json` to
an equal model, whether or not the model has been calibrated. Long numeric
arrays, such as a [time-series parameter](#time-series-parameters), are written
on one line to keep the file readable; `indent` controls the rest.

### JSON Structure

```json
{
  "name": "Three-State Model",
  "version": "1.0",
  "population": {
    "bins": [
      { "id": "A", "name": "State A" },
      { "id": "B", "name": "State B" },
      { "id": "C", "name": "State C" }
    ]
  },
  "parameters": [
    { "id": "k1", "value": 0.3 },
    { "id": "k2", "value": 0.1 }
  ],
  "dynamics": {
    "typology": "DifferenceEquations",
    "transitions": [
      {
        "id": "flow_AB",
        "source": ["A"],
        "target": ["B"],
        "rate": "k1 * A * B / N"
      },
      {
        "id": "flow_BC",
        "source": ["B"],
        "target": ["C"],
        "rate": "k2"
      }
    ]
  }
}
```

## Complete Example

```python
from commol import ModelBuilder, Simulation

model = (
    ModelBuilder(name="Four-State Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_bin(id="D", name="State D")
    .add_parameter(id="k1", value=0.4, description="A to B rate")
    .add_parameter(id="k2", value=0.2, description="B to C rate")
    .add_parameter(id="k3", value=0.1, description="C to D rate")
    .add_transition(id="flow_AB", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="flow_BC", source=["B"], target=["C"], rate="k2")
    .add_transition(id="flow_CD", source=["C"], target=["D"], rate="k3")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
            {"bin": "C", "fraction": 0.0},
            {"bin": "D", "fraction": 0.0},
        ]
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=200)
```

## Time-Varying Rates and Conditional Schedules

For transitions whose rate depends on the simulation step (pulses, periodic
events, seasonal forcing, sliding windows, Gaussian bumps, linear ramps),
pass a [`TimePattern`](time-patterns.md) instance directly as the
`rate=` argument:

```python
from commol import TimePattern

builder.add_transition(
    "flow",
    ["A"], ["B"],
    rate=TimePattern.periodic(period=7, amount=0.05),
)
```

When different stratification sub-groups require different schedules, build
the schedule with `TimePattern.add_group(...)` and chain. The result is
itself a `TimePattern` and passes straight into `rate=` — there is no
separate schedule class.

Each group can also choose absolute-flow handling independently with
`absolute=True`, `absolute=False`, or the default `absolute=None` inference.

## Next Steps

- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Time Patterns](time-patterns.md) - Time-varying rate helpers
- [Simulations](simulations.md) - Running and analyzing models
- [Examples](examples.md) - Complete model examples
