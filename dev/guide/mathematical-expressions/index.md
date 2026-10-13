# Mathematical Expressions

[Back to Home](https://munqu.github.io/commol/latest/index.md)

Rates and formula parameters are written as expressions, such as `"k1 * A * B / N"`. This page lists everything an expression can contain: operators, variables, functions, and the security limits. The engine compiles each expression to machine code (Cranelift) once, and reuses it at every step.

## Quick path

```python
from commol import ModelBuilder, Simulation

model = (
    ModelBuilder(name="Expressions")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k1_seasonal", value="k1 * (1 + 0.3 * sin(2 * pi * step / 365))")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1_seasonal * A * B / N")
    .add_transition(id="t_ba", source=["B"], target=["A"], rate="if(step > 30, 0.05, 0)")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 0.99}, {"bin": "B", "fraction": 0.01}],
    )
    .build(typology="DifferenceEquations")
)

results = Simulation(model).run(num_steps=60)
```

| Where                               | Example            | Evaluated                                                                                 |
| ----------------------------------- | ------------------ | ----------------------------------------------------------------------------------------- |
| Transition `rate`                   | `"k1 * A * B / N"` | For every flow, at every step                                                             |
| Formula parameter `value`           | `"k1 * 2"`         | At every step, in declaration order, before the rates                                     |
| Stratified rate `rate`              | `"k_g1 * A"`       | Like a transition rate                                                                    |
| Calibration constraint `expression` | `"k1 - k2"`        | See [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md) |

Whether a rate is the flow itself or a rate per member of the source depends on the variables it names; see [Absolute and per-capita rates](https://munqu.github.io/commol/latest/guide/transitions/#absolute-and-per-capita-rates).

## Operators

| Operator                         | Meaning                         | Example                     | Value  |
| -------------------------------- | ------------------------------- | --------------------------- | ------ |
| `+`, `-`                         | Addition, subtraction, negation | `"k1 - k2"`, `"-k1"`        |        |
| `*`, `/`                         | Multiplication, division        | `"k1 * B / N"`              |        |
| `%`                              | Remainder                       | `"step % 7"`                | 0 to 6 |
| `**` or `^`                      | Power                           | `"2 * 3 ** 4"`              | 162    |
| `( )`                            | Grouping                        | `"(2 + 3) * 4"`             | 20     |
| `>`, `<`, `>=`, `<=`, `==`, `!=` | Comparison, only inside `if`    | `"if(step > 30, k1, 0)"`    |        |
| `!`                              | Not, only inside `if`           | `"if(!(step > 30), k1, 0)"` |        |

Precedence, from high to low: parentheses, power, multiplication, division and remainder, addition and subtraction. So `"2 ** 3 * 4"` is 32 and `"2 + 3 * 4"` is 14.

A comparison gives a boolean, not a number, so it cannot be multiplied: `"k1 * (step > 30)"` is rejected. Write `"if(step > 30, k1, 0)"`. Rates do not accept `&&` or `||`; nest `if` calls instead:

```python
in_window = "if(step >= 10, if(step <= 20, 0.05, 0), 0)"   # 0.05 for steps 10 to 20
outside = "if(step < 10, 0.05, if(step > 20, 0.05, 0))"    # 0.05 outside that window
```

## Variables

| Variable                           | Value                                                                                                                                                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A parameter id                     | The parameter's value at this step                                                                                                                                                                                                   |
| A bin id, without stratifications  | The bin's population                                                                                                                                                                                                                 |
| A compartment name, such as `A_g1` | That compartment's population                                                                                                                                                                                                        |
| A bin id, with stratifications     | The total of all its compartments                                                                                                                                                                                                    |
| `N`                                | The total population                                                                                                                                                                                                                 |
| `t`                                | The current position on the axis. A difference model reads it at each step, so there it equals `step`; a differential model reads it anywhere                                                                                        |
| `step`                             | The step that holds the current position, `floor(t)`: 0, 1, 2, … It is constant on each unit interval of a differential model; see [Time inside a step](https://munqu.github.io/commol/latest/guide/simulations/#time-inside-a-step) |
| `pi`, `e`                          | π and e                                                                                                                                                                                                                              |

### Subpopulation variables

With stratifications, the engine also provides sums of compartments. Category suffixes must follow the order in which the stratifications were declared.

| Variable       | Sum of                                       | Example (`group` then `type`)                                           |
| -------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| `N_<cat>…`     | Every compartment with these categories      | `N_g1`, `N_t1`, `N_g1_t1`                                               |
| `<bin>`        | Every compartment of the bin                 | `A`                                                                     |
| `<bin>_<cat>…` | The bin's compartments with these categories | `A_g1`, `A_t1`; with every category, the compartment itself (`A_g1_t1`) |

```python
from commol import ModelBuilder

def value_of(expression: str) -> float:
    """The value of an expression at step 0, read through a formula parameter."""
    model = (
        ModelBuilder(name="Variables")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_stratification(id="type", categories=["t1", "t2"])
        .add_parameter(id="probe", value=expression)
        .add_transition(id="probe_flow", source=[], target=["B"], rate="probe")
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 0.6}, {"bin": "B", "fraction": 0.4}],
            stratification_fractions=[
                {
                    "stratification": "group",
                    "fractions": [
                        {"category": "g1", "fraction": 0.25},
                        {"category": "g2", "fraction": 0.75},
                    ],
                },
                {
                    "stratification": "type",
                    "fractions": [
                        {"category": "t1", "fraction": 0.5},
                        {"category": "t2", "fraction": 0.5},
                    ],
                },
            ],
        )
        .build(typology="DifferenceEquations")
    )
    results = Simulation(model).run(num_steps=1)
    inflow = sum(series[1] - series[0] for series in results.values())
    return inflow / 4  # the probe flows into each of the 4 B compartments

print(value_of("N + 0"))        # 1000.0
print(value_of("N_g1 + 0"))     # 250.0  = 1000 x 0.25
print(value_of("N_g1_t1 + 0"))  # 125.0  = 1000 x 0.25 x 0.5
print(value_of("A + 0"))        # 600.0
print(value_of("A_g1 + 0"))     # 150.0  = 600 x 0.25
print(value_of("A_t1 + 0"))     # 300.0  = 600 x 0.5
```

| Pitfall                                                       | Result                                          |
| ------------------------------------------------------------- | ----------------------------------------------- |
| Suffixes out of declaration order, as in `A_t1_g1`            | Error: the variable is not found                |
| `N_…` out of order, as in `N_t1_g1`                           | `build` fails and lists the valid `N_…` names   |
| A combination that a conditional stratification never creates | `build` fails for `N_…`; `<bin>_…` is not found |

## Functions

| Group                     | Functions                                                     |
| ------------------------- | ------------------------------------------------------------- |
| Trigonometric             | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y, x)`    |
| Hyperbolic                | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`             |
| Exponential and logarithm | `exp`, `log` (natural), `ln` (natural), `log10`, `log2`       |
| Power and root            | `sqrt`, `cbrt`, `pow(x, y)`, `hypot(x, y)`                    |
| Comparison                | `min(a, b)`, `max(a, b)`, `abs`                               |
| Rounding                  | `floor`, `ceil`, `round` (halves round up: `round(2.5)` is 3) |
| Conditional               | `if(condition, value_if_true, value_if_false)`                |

For rates that change over time, [`AxisPattern`](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md) builds tested expressions for pulses, windows, ramps and seasons.

## Common patterns

```python
patterns = {
    "mass action": "k1 * A * B",
    "frequency dependent": "k1 * A * B / N",
    "saturation": "k1 * B / (1 + B)",
    "linear growth over time": "0.1 + 0.001 * step",
    "exponential decay over time": "k1 * exp(-0.01 * step)",
    "annual cycle": "k1 * (1 + 0.3 * sin(2 * pi * step / 365))",
    "never negative": "max(0, k1 - 0.001 * step)",
    "capped above a threshold": "k1 * min(1, 100 / max(B, 1))",
    "only after step 30": "if(step > 30, k1, 0)",
    "safe division": "k1 * B / max(N, 1)",
}
```

To keep a rate readable, move part of it into a formula parameter: `add_parameter(id="k1_seasonal", value="k1 * (1 + 0.3 * sin(2 * pi * step / 365))")` and then `rate="k1_seasonal * A * B / N"`.

## Security

Every rate and constraint expression is checked before it reaches the engine (`commol.utils.security.validate_expression_security`). An expression that is not plain mathematics is rejected when the transition is added:

```python
try:
    ModelBuilder(name="Unsafe").add_bin(id="A", name="State A").add_transition(
        id="t", source=["A"], target=[], rate="__import__('os')"
    )
except ValueError as error:
    print("rejected")
```

| Limit (`SecurityConfig`) | Value            |
| ------------------------ | ---------------- |
| Expression length        | 10000 characters |
| Nesting depth            | 10 levels        |
| Function calls           | 500              |
| Variable name length     | 50 characters    |
| Numeric literal          | 1e12             |
| Complexity score         | 1000             |

## Next steps

| To learn                      | Read                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| Ready-made time-varying rates | [Axis Patterns](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md)     |
| How rates become flows        | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)         |
| Running a model               | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md) |
