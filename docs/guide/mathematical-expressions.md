# Mathematical Expressions

EpiModel supports rich mathematical expressions in transition rates, enabling complex and realistic disease dynamics.

## Basic Syntax

Transition rates are specified as **string expressions** that are evaluated during simulation:

```python
.add_transition(
    id="infection",
    source=["S"],
    target=["I"],
    rate="beta * S * I / N"  # Mathematical expression as a string
)
```

## Arithmetic Operations

### Basic Operations

| Operation      | Operator | Example                | Description       |
| -------------- | -------- | ---------------------- | ----------------- |
| Addition       | `+`      | `"alpha + beta"`       | Sum of two values |
| Subtraction    | `-`      | `"alpha - beta"`       | Difference        |
| Multiplication | `*`      | `"beta * S"`           | Product           |
| Division       | `/`      | `"I / N"`              | Division          |
| Exponentiation | `**`     | `"beta ** 2"`          | Power             |
| Parentheses    | `()`     | `"(alpha + beta) * I"` | Grouping          |

### Examples

```python
rate = "beta * S * I"           # Multiply transmission rate by populations
rate = "gamma + delta"          # Add two parameters
rate = "beta / N"               # Divide by total population
rate = "(beta + gamma) / 2"     # Average with parentheses
rate = "beta ** 2"              # Square a value
```

## Available Variables

### Disease State Variables

Reference any disease state by its ID:

```python
# In a model with states S, I, R
rate = "beta * S * I"    # Use S and I populations
rate = "gamma * I"       # Use I population
rate = "0.01 * R"        # Use R population
```

### Special Variables

| Variable | Type  | Description                                          |
| -------- | ----- | ---------------------------------------------------- |
| `N`      | float | Total population (automatic sum of all compartments) |
| `step`   | int   | Current simulation step (0, 1, 2, ...)               |
| `t`      | int   | Alias for `step`                                     |
| `pi`     | float | Mathematical constant π ≈ 3.14159                    |
| `e`      | float | Mathematical constant e ≈ 2.71828                    |

**Examples:**

```python
rate = "beta * I / N"              # Frequency-dependent transmission
rate = "beta * sin(2 * pi * t)"    # Periodic variation
rate = "gamma * exp(-0.01 * step)" # Exponential decay over time
```

### Parameter References

Reference any parameter by its ID:

```python
.add_parameter(id="beta", value=0.3)
.add_parameter(id="gamma", value=0.1)

.add_transition(rate="beta * S * I / N")  # Uses beta parameter
.add_transition(rate="gamma")             # Uses gamma parameter
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
# Seasonal variation (peaks annually)
rate = "beta * sin(2 * pi * step / 365)"
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
rate = "gamma * exp(-0.01 * step)"
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
rate = "beta * sqrt(I)"
```

### Comparison Functions

| Function    | Description           | Example               |
| ----------- | --------------------- | --------------------- |
| `max(a, b)` | Maximum of two values | `max(0, beta - 0.01)` |
| `min(a, b)` | Minimum of two values | `min(gamma, 0.5)`     |
| `abs(x)`    | Absolute value        | `abs(x)`              |

**Example:**

```python
# Ensure rate stays positive
rate = "max(0, beta - 0.001 * step)"
```

### Rounding Functions

| Function   | Description      | Example            |
| ---------- | ---------------- | ------------------ |
| `floor(x)` | Round down       | `floor(beta * I)`  |
| `ceil(x)`  | Round up         | `ceil(gamma * I)`  |
| `round(x)` | Round to nearest | `round(alpha * I)` |

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
rate = "beta * exp(-0.01 * t)"

# Seasonal pattern (annual cycle)
rate = "beta * (1 + 0.3 * sin(2 * pi * step / 365))"
```

### Population-Dependent Rates

```python
# Standard mass action
rate = "beta * S * I"

# Frequency-dependent (normalized by population)
rate = "beta * S * I / N"

# Saturation effect
rate = "beta * I / (1 + I)"
```

### Threshold Effects

```python
# Activate when condition is met
rate = "max(0, beta) * (I > 100)"  # Only when I > 100

# Reduce transmission above threshold
rate = "beta * min(1, 100 / I)"    # Reduces when I > 100
```

### Composite Expressions

Combine multiple effects:

```python
# Seasonal transmission with saturation
rate = "beta * (1 + 0.3 * sin(2 * pi * t / 365)) * S * I / (N + I)"

# Time-varying intervention
rate = "beta * max(0.2, 1 - 0.01 * step) * S * I / N"
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
# Good: Clear parameter names
.add_parameter(id="beta", value=0.3)
rate = "beta * S * I / N"

# Avoid: Magic numbers
rate = "0.3 * S * I / N"
```

### 2. Prevent Division by Zero

```python
# Good: Add small constant
rate = "beta * I / (N + 1)"

# Good: Use max
rate = "beta * I / max(N, 1)"
```

### 3. Keep Expressions Simple

```python
# Good: Simple and readable
rate = "beta * S * I / N"

# Acceptable but complex:
rate = "beta * (1 + 0.3 * sin(2 * pi * t / 365)) * S * I / N"

# Better: Split into parameters
.add_parameter(id="seasonal_beta", value="beta * (1 + 0.3 * sin(2 * pi * t / 365))")
rate = "seasonal_beta * S * I / N"
```

### 4. Document Complex Expressions

```python
.add_transition(
    id="seasonal_infection",
    source=["S"],
    target=["I"],
    rate="beta * (1 + 0.3 * sin(2 * pi * step / 365)) * S * I / N",
    description="Infection with 30% seasonal amplitude, annual cycle"
)
```

## Security Features

EpiModel validates all expressions for security:

### Safe Operations

```python
rate = "beta * S * I / N"                    # Mathematical operations
rate = "sin(2 * pi * t)"                     # Mathematical functions
rate = "max(0, gamma - 0.01 * step)"        # Built-in functions
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

- **Fast**: `"gamma"` (parameter lookup)
- **Fast**: `"beta * I"` (simple arithmetic)
- **Medium**: `"beta * S * I / N"` (multiple operations)
- **Slower**: `"beta * exp(-0.01 * step) * sin(2 * pi * t / 365)"` (functions + operations)

### Optimization Tips

1. **Use parameters for constants**:

   ```python
   # Slower: recalculates each step
   rate = "0.3 * S * I / 1000"

   # Faster: parameter lookup
   .add_parameter(id="beta", value=0.3)
   rate = "beta * S * I / N"
   ```

2. **Simplify when possible**:

   ```python
   # Complex
   rate = "beta * I / N + gamma * I / N"

   # Simplified
   rate = "(beta + gamma) * I / N"
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

| Variable           | Type  | Description                  |
| ------------------ | ----- | ---------------------------- |
| **Disease states** | float | Any state ID (S, I, R, etc.) |
| **Parameters**     | float | Any parameter ID             |
| `N`                | float | Total population             |
| `step`             | int   | Current time step            |
| `t`                | int   | Alias for step               |
| `pi`               | float | π constant                   |
| `e`                | float | e constant                   |

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
