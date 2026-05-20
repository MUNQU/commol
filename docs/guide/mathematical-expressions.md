# Mathematical Expressions

Commol supports rich mathematical expressions in transition rates, enabling complex and realistic model dynamics.

## Basic Syntax

Transition rates are specified as **string expressions** that are evaluated during simulation:

```python
.add_transition(
    id="flow_AB",
    source=["A"],
    target=["B"],
    rate="k1 * A * B / N"  # Mathematical expression as a string
)
```

## Arithmetic Operations

### Basic Operations

| Operation      | Operator | Example             | Description       |
| -------------- | -------- | ------------------- | ----------------- |
| Addition       | `+`      | `"k1 + k2"`         | Sum of two values |
| Subtraction    | `-`      | `"k1 - k2"`         | Difference        |
| Multiplication | `*`      | `"k1 * A"`          | Product           |
| Division       | `/`      | `"B / N"`           | Division          |
| Exponentiation | `**`     | `"k1 ** 2"`         | Power             |
| Parentheses    | `()`     | `"(k1 + k2) * B"`   | Grouping          |

### Examples

```python
rate = "k1 * A * B"            # Multiply rate by two compartment populations
rate = "k1 + k2"               # Sum of two parameters
rate = "k1 / N"                # Divide by total population
rate = "(k1 + k2) / 2"         # Average with parentheses
rate = "k1 ** 2"               # Square a parameter value
```

## Available Variables

### Compartment Variables

Reference any compartment by its ID:

```python
# In a model with compartments A, B, C
rate = "k1 * A * B"    # Use A and B populations
rate = "k2 * B"        # Use B population
rate = "0.01 * C"      # Use C population
```

### Special Variables

| Variable        | Type  | Description                                                                |
| --------------- | ----- | -------------------------------------------------------------------------- |
| `N`             | float | Total population (automatic sum of all compartments)                       |
| `N_{category}`  | float | Total population for a specific stratification category (e.g., `N_young`)  |
| `N_{cat1_cat2}` | float | Total population for an intersection of categories (e.g., `N_young_urban`) |
| `step`          | int   | Current simulation step (0, 1, 2, ...)                                     |
| `t`             | int   | Alias for `step`                                                           |
| `$compartment`  | str   | Placeholder that expands to each source compartment name (see below)       |

**Examples:**

```python
rate = "k1 * B / N"               # Frequency-dependent flow
rate = "k1 * sin(2 * pi * t)"     # Periodic variation
rate = "k2 * exp(-0.01 * step)"   # Exponential decay over time
```

### Subpopulation Variables

When you use stratifications, Commol automatically creates several types of **subpopulation sum variables** that aggregate compartment populations. These are updated at every simulation step and can be used in any rate expression.

#### Types of Subpopulation Variables

Given a model with bins and stratifications, Commol generates:

| Variable pattern                  | Description                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------- |
| `N`                               | Total population across all compartments                                                          |
| `N_{cat}`                         | Total population matching a stratification category (e.g., `N_young`)                             |
| `N_{cat1_cat2}`                   | Total population matching an intersection of categories (e.g., `N_young_urban`)                   |
| `{bin}`                           | Base bin total — sum of all stratified versions of a bin (e.g., `S` = sum of all `S_*`)           |
| `{bin}_{cat}`                     | Partial bin sum — sum of all compartments for a bin matching a category subset (e.g., `S_young`)  |
| `{bin}_{cat1}_{cat2}_{...}_{catN}` | Full stratified compartment name (e.g., `S_young_urban`) — the individual compartment population |

**Partial bin sums** (e.g., `A_g1`) are available when the model has **two or more stratifications**. They represent the sum of a bin's compartments across one or more stratification categories, summing over the remaining dimensions. For instance, with stratifications `group` and `type`, the variable `A_g1` equals the sum of all `A` compartments where `group=g1`, regardless of type.

Category suffixes in variable names must follow the **declaration order** of stratifications. For example, if `group` is declared before `type`, then `A_g1` (summing over type) is valid, but the intersection must be written as `A_g1_t1` (not `A_t1_g1`).

**Note on conditional stratifications**: partial bin-sum and `N_*` variables are generated from the Cartesian product of all declared categories regardless of conditions. A combination that was never actually generated as a compartment (because conditions were not met) will always evaluate to 0 at runtime.

### The `$compartment` Placeholder

The `$compartment` special variable is a **template placeholder** that automatically expands multi-compartment transitions into individual per-compartment transitions:

```python
# Instead of writing repetitive code:
.add_transition("outflow_A", ["A"], [], rate="mu * A")
.add_transition("outflow_B", ["B"], [], rate="mu * B")
.add_transition("outflow_C", ["C"], [], rate="mu * C")

# Use $compartment to write it once:
.add_transition(
    id="outflow",
    source=["A", "B", "C"],
    target=[],
    rate="mu * $compartment"  # Automatically expands
)
```

**How it works:**

1. System detects `$compartment` in the rate formula
2. Creates one transition per source compartment
3. Replaces `$compartment` with the actual compartment name in each formula

**Generated transitions:**

- `outflow__A`: `rate = "mu * A"`
- `outflow__B`: `rate = "mu * B"`
- `outflow__C`: `rate = "mu * C"`

**Multiple occurrences in complex formulas:**

```python
rate = "mu * $compartment * (1 + 0.1 * $compartment / N)"

# Expands to:
# outflow__A: "mu * A * (1 + 0.1 * A / N)"
# outflow__B: "mu * B * (1 + 0.1 * B / N)"
# outflow__C: "mu * C * (1 + 0.1 * C / N)"
```

**With stratified rates:**

```python
.add_transition(
    id="outflow",
    source=["A", "B", "C"],
    target=[],
    rate="mu_base * $compartment",
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
```

**When to use:**

- Per-capita rates that apply to multiple compartments
- Transitions from multiple source compartments to a single target
- Any time you need the same formula pattern applied to different compartments

**When NOT to use:**

- Standard multi-compartment interactions (use regular syntax)
- Single source compartment (use compartment name directly)
- Different formulas for different compartments (write separate transitions)

See [Building Models - Using $compartment Placeholder](building-models.md#using-compartment-placeholder-for-per-compartment-rates) for more details.

### The `per_compartment` Flag

When a model uses stratifications, base compartment names in rate expressions (like `A`, `B`) resolve to the **sum** of all their stratified versions. The `per_compartment=True` flag on a transition changes this behavior, replacing base compartment names with the specific stratified compartment name for each expanded flow:

```python
# Without per_compartment: A means total A (A_g1 + A_g2 + ...)
.add_transition(id="forward", source=["A"], target=["B"], rate="k1 * A")

# With per_compartment: A is replaced with A_g1, A_g2, etc.
.add_transition(id="forward", source=["A"], target=["B"], rate="k1 * A", per_compartment=True)
```

Both source and target bin names are replaced. Use this for transitions where each subgroup should evolve based on its own value, not for terms that explicitly depend on the total across all subgroups.

See [Building Models - Per-Compartment Rates](building-models.md#per-compartment-rates-with-per_compartment) for detailed examples.

### Parameter References

Reference any parameter by its ID:

```python
.add_parameter(id="k1", value=0.3)
.add_parameter(id="k2", value=0.1)

.add_transition(rate="k1 * A * B / N")  # Uses k1 parameter
.add_transition(rate="k2")              # Uses k2 parameter
```

## Mathematical Functions

### Trigonometric Functions

| Function      | Description              | Example                 |
| ------------- | ------------------------ | ----------------------- |
| `sin(x)`      | Sine                     | `sin(2 * pi * t / 365)` |
| `cos(x)`      | Cosine                   | `cos(t)`                |
| `tan(x)`      | Tangent                  | `tan(x)`                |
| `asin(x)`     | Arc sine                 | `asin(x)`               |
| `acos(x)`     | Arc cosine               | `acos(x)`               |
| `atan(x)`     | Arc tangent              | `atan(x)`               |
| `atan2(y, x)` | Two-argument arc tangent | `atan2(y, x)`           |

**Example:**

```python
# Periodic variation (annual cycle)
rate = "k1 * sin(2 * pi * step / 365)"
```

### Exponential and Logarithmic

| Function   | Description       | Example          |
| ---------- | ----------------- | ---------------- |
| `exp(x)`   | Exponential (e^x) | `exp(-0.01 * t)` |
| `log(x)`   | Natural logarithm | `log(I + 1)`     |
| `ln(x)`    | Alias for `log`   | `ln(I + 1)`      |
| `log10(x)` | Base-10 logarithm | `log10(I)`       |
| `log2(x)`  | Base-2 logarithm  | `log2(I)`        |

**Example:**

```python
# Exponential decay over time
rate = "k2 * exp(-0.01 * step)"
```

### Power and Root

| Function    | Description    | Example     |
| ----------- | -------------- | ----------- |
| `sqrt(x)`   | Square root    | `sqrt(I)`   |
| `pow(x, y)` | Power (x^y)    | `pow(I, 2)` |
| `x ** y`    | Power operator | `I ** 2`    |

**Example:**

```python
# Square root relationship
rate = "k1 * sqrt(B)"
```

### Comparison Functions

| Function    | Description           | Example               |
| ----------- | --------------------- | --------------------- |
| `max(a, b)` | Maximum of two values | `max(0, k1 - 0.01)` |
| `min(a, b)` | Minimum of two values | `min(k2, 0.5)`      |
| `abs(x)`    | Absolute value        | `abs(x)`             |

**Example:**

```python
# Ensure rate stays positive
rate = "max(0, k1 - 0.001 * step)"
```

### Rounding Functions

| Function   | Description      | Example           |
| ---------- | ---------------- | ----------------- |
| `floor(x)` | Round down       | `floor(k1 * B)`   |
| `ceil(x)`  | Round up         | `ceil(k2 * B)`    |
| `round(x)` | Round to nearest | `round(k1 * B)`   |

### Hyperbolic Functions

| Function  | Description        | Example   |
| --------- | ------------------ | --------- |
| `sinh(x)` | Hyperbolic sine    | `sinh(x)` |
| `cosh(x)` | Hyperbolic cosine  | `cosh(x)` |
| `tanh(x)` | Hyperbolic tangent | `tanh(x)` |

## Common Patterns

### Time-Dependent Rates

Use `step` or `t` for time-varying rates:

```python
# Linear increase
rate = "0.1 + 0.001 * step"

# Exponential decay
rate = "k1 * exp(-0.01 * t)"

# Periodic pattern (annual cycle)
rate = "k1 * (1 + 0.3 * sin(2 * pi * step / 365))"
```

### Population-Dependent Rates

```python
# Mass action
rate = "k1 * A * B"

# Frequency-dependent (normalized by total population)
rate = "k1 * A * B / N"

# Saturation effect
rate = "k1 * B / (1 + B)"
```

### Threshold Effects

```python
# Activate when condition is met
rate = "max(0, k1) * (B > 100)"  # Only when B > 100

# Reduce flow above threshold
rate = "k1 * min(1, 100 / B)"   # Reduces when B > 100
```

### Composite Expressions

Combine multiple effects:

```python
# Periodic flow with saturation
rate = "k1 * (1 + 0.3 * sin(2 * pi * t / 365)) * A * B / (N + B)"

# Time-varying rate with minimum
rate = "k1 * max(0.2, 1 - 0.01 * step) * A * B / N"
```

## Operator Precedence

From highest to lowest priority:

1. **Parentheses**: `()`
2. **Exponentiation**: `**`
3. **Multiplication/Division**: `*`, `/`
4. **Addition/Subtraction**: `+`, `-`

**Examples:**

```python
rate = "2 + 3 * 4"        # = 14 (multiplication first)
rate = "(2 + 3) * 4"      # = 20 (parentheses first)
rate = "2 ** 3 * 4"       # = 32 (exponentiation first)
rate = "2 * 3 ** 4"       # = 162 (exponentiation before multiplication)
```

## Best Practices

### 1. Use Meaningful Parameters

```python
# Good: Named parameters
.add_parameter(id="k1", value=0.3)
rate = "k1 * A * B / N"

# Avoid: Magic numbers
rate = "0.3 * A * B / N"
```

### 2. Prevent Division by Zero

```python
# Good: Add small constant
rate = "k1 * B / (N + 1)"

# Good: Use max
rate = "k1 * B / max(N, 1)"
```

### 3. Keep Expressions Simple

```python
# Good: Simple and readable
rate = "k1 * A * B / N"

# Acceptable but complex:
rate = "k1 * (1 + 0.3 * sin(2 * pi * t / 365)) * A * B / N"

# Better: Split into parameters
.add_parameter(id="k1_periodic", value="k1 * (1 + 0.3 * sin(2 * pi * t / 365))")
rate = "k1_periodic * A * B / N"
```

### 4. Document Complex Expressions

```python
.add_transition(
    id="periodic_flow",
    source=["A"],
    target=["B"],
    rate="k1 * (1 + 0.3 * sin(2 * pi * step / 365)) * A * B / N",
    description="Flow with 30% periodic amplitude, annual cycle"
)
```

## Security Features

Commol validates all expressions for security:

### Safe Operations

```python
rate = "k1 * A * B / N"                      # Mathematical operations
rate = "sin(2 * pi * t)"                     # Mathematical functions
rate = "max(0, k2 - 0.01 * step)"           # Built-in functions
```

### Blocked Operations

```python
rate = "__import__('os')"                    # Python imports
rate = "eval('code')"                        # Code evaluation
rate = "exec('code')"                        # Code execution
rate = "open('file')"                        # File operations
```

The parser:

- Only allows mathematical operations and approved functions
- Blocks all Python/Rust code execution
- Validates syntax before simulation
- Prevents code injection attacks

## Performance Considerations

### Expression Complexity

- **Fast**: `"k2"` (parameter lookup)
- **Fast**: `"k1 * B"` (simple arithmetic)
- **Medium**: `"k1 * A * B / N"` (multiple operations)
- **Slower**: `"k1 * exp(-0.01 * step) * sin(2 * pi * t / 365)"` (functions + operations)

### Optimization Tips

1. **Use parameters for constants**:

   ```python
   # Slower: recalculates each step
   rate = "0.3 * A * B / 1000"

   # Faster: parameter lookup
   .add_parameter(id="k1", value=0.3)
   rate = "k1 * A * B / N"
   ```

2. **Simplify when possible**:

   ```python
   # Complex
   rate = "k1 * B / N + k2 * B / N"

   # Simplified
   rate = "(k1 + k2) * B / N"
   ```

3. **Profile complex models**: Use `time` module to measure performance with different expressions.

## Complete Function Reference

### All Available Functions

| Category            | Functions                                            |
| ------------------- | ---------------------------------------------------- |
| **Trigonometric**   | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| **Exponential/Log** | `exp`, `log`, `ln`, `log10`, `log2`                  |
| **Power/Root**      | `sqrt`, `pow`                                        |
| **Comparison**      | `max`, `min`, `abs`                                  |
| **Rounding**        | `floor`, `ceil`, `round`                             |
| **Hyperbolic**      | `sinh`, `cosh`, `tanh`                               |

### All Available Variables

| Variable                    | Type  | Description                                                            |
| --------------------------- | ----- | ---------------------------------------------------------------------- |
| **Compartments**            | float | Any full compartment name (e.g., `S`, `S_young`, `S_young_urban`)      |
| **Parameters**              | float | Any parameter ID                                                       |
| `N`                         | float | Total population                                                       |
| `N_{cat}`, `N_{cat1_cat2}`  | float | Subpopulation totals by category (e.g., `N_young`, `N_young_urban`)    |
| `{bin}`                     | float | Base bin total (e.g., `S` = sum of all `S_*` when stratified)          |
| `{bin}_{cat}`               | float | Partial bin sum (e.g., `S_young`); requires 2+ stratifications         |
| `step`                      | int   | Current time step                                                      |
| `t`                         | int   | Alias for step                                                         |
| `pi`                        | float | π constant                                                             |
| `e`                         | float | e constant                                                             |

### All Operators

| Operator | Description    |
| -------- | -------------- |
| `+`      | Addition       |
| `-`      | Subtraction    |
| `*`      | Multiplication |
| `/`      | Division       |
| `**`     | Exponentiation |
| `()`     | Grouping       |

## Next Steps

- [Building Models](building-models.md) - Use expressions in transitions
- [Simulations](simulations.md) - Run models with mathematical expressions
- [Examples](examples.md) - See complex expressions in complete models
