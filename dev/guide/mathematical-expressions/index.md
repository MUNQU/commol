# Mathematical Expressions

[Back to Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md)

Rates and formula parameters are written as expressions, such as `"k1 * A * B / total"`. This page lists everything an expression can contain: operators, variables, functions, and the limits of the parser. The engine checks each expression against this language, compiles it to machine code (Cranelift) once, and reuses it at every step.

## Quick path

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Expressions")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k1_periodic", value="k1 * (1 + 0.3 * sin(2 * pi * step / 365))")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1_periodic * A * B / total")
    .add_transition(id="t_ba", source=["B"], target=["A"], rate="if(step > 30, 0.05, 0)")
    .build(
        Aggregate(
            bin_amounts={"A": 990, "B": 10},
        ),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=60)
```

| Where                               | Example                | Evaluated                                                                                 |
| ----------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| Transition `rate`                   | `"k1 * A * B / total"` | For every flow, at every step                                                             |
| Formula parameter `value`           | `"k1 * 2"`             | At every step, in declaration order, before the rates                                     |
| Stratified rate `rate`              | `"k_g1 * A"`           | Like a transition rate                                                                    |
| Calibration constraint `expression` | `"k1 - k2"`            | See [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md) |

The variables a rate names decide whether it is absolute or proportional; see [Absolute and proportional rates](https://munqu.github.io/commol/latest/guide/transitions/#absolute-and-proportional-rates).

## Operators

| Operator                         | Meaning                         | Example                     | Value  |
| -------------------------------- | ------------------------------- | --------------------------- | ------ |
| `+`, `-`                         | Addition, subtraction, negation | `"k1 - k2"`, `"-k1"`        |        |
| `*`, `/`                         | Multiplication, division        | `"k1 * B / total"`          |        |
| `%`                              | Remainder                       | `"step % 7"`                | 0 to 6 |
| `**` or `^`                      | Power                           | `"2 * 3 ** 4"`              | 162    |
| `( )`                            | Grouping                        | `"(2 + 3) * 4"`             | 20     |
| `>`, `<`, `>=`, `<=`, `==`, `!=` | Comparison, only inside `if`    | `"if(step > 30, k1, 0)"`    |        |
| `!`                              | Not, only inside `if`           | `"if(!(step > 30), k1, 0)"` |        |

Precedence, from high to low: parentheses, power, the sign `-`, multiplication, division and remainder, addition and subtraction. So `"2 ** 3 * 4"` is 32, `"2 + 3 * 4"` is 14 and `"-2 ** 2"` is -4. Operators of one level apply from the left: `"8 - 4 - 2"` is 2 and `"8 / 4 * 2"` is 4. A chain of powers applies from the right, as in Python: `"2 ** 3 ** 2"` is `"2 ** (3 ** 2)"`, 512.

A comparison gives a boolean, not a number, so it cannot be multiplied: `"k1 * (step > 30)"` is rejected. Write `"if(step > 30, k1, 0)"`. Rates do not accept `&&` or `||`; nest `if` calls instead:

```python
in_window = "if(step >= 10, if(step <= 20, 0.05, 0), 0)"   # 0.05 for steps 10 to 20
outside = "if(step < 10, 0.05, if(step > 20, 0.05, 0))"    # 0.05 outside that window
```

## Variables

An id starts with a letter or `_` and can hold digits after it, such as `e1` or `c1star`. A number such as `1e5` or `2.5E-3` is a number: its exponent is never a variable.

| Variable                           | Value                                                                                                                                                                                                                                                                         |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A parameter id                     | The parameter's value at this step                                                                                                                                                                                                                                            |
| A bin id, without stratifications  | The bin's amount                                                                                                                                                                                                                                                              |
| A compartment name, such as `A_g1` | That compartment's amount                                                                                                                                                                                                                                                     |
| A bin id, with stratifications     | The total of all its compartments                                                                                                                                                                                                                                             |
| `total`                            | The total: the sum of every compartment, accumulators excluded                                                                                                                                                                                                                |
| `neighbors_<bin>`, `neighbors`     | In a transition rate only: what the neighbors of a unit hold, of one bin or of every bin. The structure says who they are; a rate that reads them and no compartment is proportional. See [Neighbors](https://munqu.github.io/commol/latest/guide/building-models/#neighbors) |
| `<M>_<i>_<j>`, `<M>_radius`        | The entry of row `i` and column `j` of the matrix `M`, and its spectral radius, both scaled by the parameters of its rows and columns. See [Matrix sums](#matrix-sums)                                                                                                        |
| `t`                                | The current position on the axis. A difference model reads it at each step, so there it equals `step`; a differential model reads it anywhere                                                                                                                                 |
| `step`                             | The step that holds the current position, `floor(t)`: 0, 1, 2, … It is constant on each unit interval of a differential model; see [Time inside a step](https://munqu.github.io/commol/latest/guide/simulations/#time-inside-a-step)                                          |
| `pi`, `e`                          | π and e                                                                                                                                                                                                                                                                       |

### Subgroup totals

With stratifications, the engine also provides sums of compartments, the subgroup totals. Category suffixes must follow the order in which the stratifications were declared. `commol_rs.model.subgroup_total_names` (in `crates/commol-model/src/builder.rs`) lists them for a process.

| Variable       | Sum of                                       | Example (`group` then `type`)                                           |
| -------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| `total_<cat>…` | Every compartment with these categories      | `total_g1`, `total_t1`, `total_g1_t1`                                   |
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
        .build(
            Aggregate(
                bin_amounts={"A": 600, "B": 400},
                stratification_fractions={
                    "group": {"g1": 0.25, "g2": 0.75},
                    "type": {"t1": 0.5, "t2": 0.5},
                },
            ),
            engine="DifferenceEquations",
        )
    )
    results = Simulation(model).run(num_steps=1)
    inflow = sum(series[1] - series[0] for series in results.values())
    return inflow / 4  # the probe flows into each of the 4 B compartments

print(value_of("total + 0"))        # 1000.0
print(value_of("total_g1 + 0"))     # 250.0  = 1000 x 0.25
print(value_of("total_g1_t1 + 0"))  # 125.0  = 1000 x 0.25 x 0.5
print(value_of("A + 0"))        # 600.0
print(value_of("A_g1 + 0"))     # 150.0  = 600 x 0.25
print(value_of("A_t1 + 0"))     # 300.0  = 600 x 0.5
```

| Pitfall                                                       | Result                                              |
| ------------------------------------------------------------- | --------------------------------------------------- |
| Suffixes out of declaration order, as in `A_t1_g1`            | Error: the variable is not found                    |
| `total_…` out of order, as in `total_t1_g1`                   | `build` fails and lists the valid `total_…` names   |
| A combination that a conditional stratification never creates | `build` fails for `total_…`; `<bin>_…` is not found |

### Matrix sums

A matrix between the categories of a stratification (`ModelBuilder.add_matrix`, see [Matrices between categories](https://munqu.github.io/commol/latest/guide/building-models/#matrices-between-categories)) couples the categories in one rate. In the rate of a compartment of category `i`, `matrix_sum(M, e)` is

```text
Σ_j M[i, j] · e_j
```

where `e_j` is the expression `e` read in category `j` of the stratification of `M`:

| In `e`                                                      | Reads in category `j`                                                                                                                     |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| A bin id, such as `B`                                       | The amount of the bin in `j`: the compartment `B_j`, or the sum of the compartments of `B` in `j` when a later stratification splits them |
| `total`                                                     | The total of `j`, `total_j`                                                                                                               |
| Anything else: a parameter, a compartment, a subgroup total | The same value as outside the sum                                                                                                         |

```python
coupled = (
    ModelBuilder(name="Coupled groups")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_matrix(
        id="M",
        stratification="group",
        values={"g1": {"g1": 2, "g2": 1}, "g2": {"g1": 1, "g2": 3}},
    )
    .add_parameter(id="k1", value=0.1)
    .add_transition(
        id="t_ab", source=["A"], target=["B"], rate="k1 * matrix_sum(M, B / total)"
    )
    .build(
        Aggregate(
            bin_amounts={"A": 900, "B": 100},
            stratification_fractions={"group": {"g1": 0.4, "g2": 0.6}},
        ),
        engine="DifferenceEquations",
    )
)

coupled_run = Simulation(coupled).run(num_steps=1)
# A_g1 starts at 360: k1 x 360 x (2 x 40 / 400 + 1 x 60 / 600) = 10.8 moves.
print(coupled_run["A_g1"][1])  # 349.2
```

The engine writes each matrix sum as the sum it is before it compiles the rate, so every engine runs it: in `g1` the rate above is `k1 * (M_g1_g1 * (B_g1 / total_g1) + M_g1_g2 * (B_g2 / total_g2))`, and `Model.describe` shows that sum. A term whose entry is 0 is left out. A row of zeros is `if(M_g1_g1 != 0, M_g1_g1 * (B_g1 / total_g1), 0)` in `g1`: its value is 0, even where `B_g1 / total_g1` is not a number, and it keeps the unit of the sum.

| Rule                                                                                           | Why                                                                                                                                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What `e` reads does not make the rate absolute; what the rate reads outside the sum does       | `k1 * matrix_sum(M, B / total)` is a rate per unit of its source, as a rate that reads the neighbors of a unit is. `k1 * A_g1 * matrix_sum(M, B / total)` reads a compartment, so it is absolute. See [Absolute and proportional rates](https://munqu.github.io/commol/latest/guide/transitions/#absolute-and-proportional-rates) |
| Only a transition rate reads `matrix_sum`                                                      | A formula parameter or a constraint has no compartment, so no category `i`                                                                                                                                                                                                                                                        |
| A matrix sum holds no other matrix sum, and `e` does not read `neighbors_<bin>` or `neighbors` | Each sum has one category `j`                                                                                                                                                                                                                                                                                                     |
| Each compartment of the transition has a category of the stratification of `M`                 | Else the row `i` is not defined; a conditional stratification can leave a compartment without one                                                                                                                                                                                                                                 |

A rate, a formula parameter or a calibration constraint reads the entries of `M` as `M_<i>_<j>` (`M_g1_g2` is 1 above) and its spectral radius as `M_radius`: the largest modulus of its eigenvalues, real or complex. When parameters [scale the rows or the columns](https://munqu.github.io/commol/latest/guide/building-models/#matrices-between-categories) of `M`, the sum, the entries and the radius are those of the scaled matrix. The engine computes them when it builds the model and again when a caller sets a scale, such as a calibration at each point; they do not change during a run. Above, `M_radius` is `(5 + √5) / 2`, about 3.618. A [constraint](https://munqu.github.io/commol/latest/guide/calibration-constraints/#a-bound-on-a-matrix) reads them at the values of the point it checks. A rate scaled by `1 / M_radius` reads the matrix normalized to a spectral radius of 1, so its scale follows a change of the values:

```python
normalized = "k1 / M_radius * matrix_sum(M, B / total)"
```

## Functions

| Group                     | Functions                                                                                      |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| Trigonometric             | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y, x)`                                     |
| Hyperbolic                | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`                                              |
| Exponential and logarithm | `exp`, `log` (natural), `ln` (natural), `log10`, `log2`                                        |
| Power and root            | `sqrt`, `cbrt`, `pow(x, y)`, `hypot(x, y)`                                                     |
| Comparison                | `min(a, b, ...)`, `max(a, b, ...)` (two arguments or more), `abs`                              |
| Rounding                  | `floor`, `ceil`, `round` (halves round away from zero: `round(2.5)` is 3, `round(-2.5)` is -3) |
| Conditional               | `if(condition, value_if_true, value_if_false)`                                                 |
| Matrix                    | `matrix_sum(M, e)`, in a transition rate: see [Matrix sums](#matrix-sums)                      |

No other function is in the language: `erf(x)` or `trunc(x)` is rejected.

For rates that change along the axis, [`AxisPattern`](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md) builds tested expressions for pulses, windows, ramps and sinusoids.

## Common patterns

```python
patterns = {
    "product of two amounts": "k1 * A * B",
    "frequency dependent": "k1 * A * B / total",
    "saturation": "k1 * B / (1 + B)",
    "linear growth along the axis": "0.1 + 0.001 * step",
    "exponential decay along the axis": "k1 * exp(-0.01 * step)",
    "cycle of 365 steps": "k1 * (1 + 0.3 * sin(2 * pi * step / 365))",
    "never negative": "max(0, k1 - 0.001 * step)",
    "capped above a threshold": "k1 * min(1, 100 / max(B, 1))",
    "only after step 30": "if(step > 30, k1, 0)",
    "safe division": "k1 * B / max(total, 1)",
}
```

To keep a rate readable, move part of it into a formula parameter: `add_parameter(id="k1_periodic", value="k1 * (1 + 0.3 * sin(2 * pi * step / 365))")` and then `rate="k1_periodic * A * B / total"`.

## Rejected expressions

The engine checks each expression when it is added: a transition rate, a stratified rate, a formula parameter, a calibration constraint or an axis pattern. An expression outside the language of this page fails with a `ValueError` that names the expression and its first part outside the language: an unknown function, a call with the wrong number of arguments, `&&` or `||`, a comparison or `!` outside the condition of an `if`, or a `matrix_sum` that does not name a matrix and an expression, or that holds another one.

```python
try:
    ModelBuilder(name="Rejected").add_parameter(id="k2", value="k1 * (step > 30)")
except ValueError as error:
    print(error)  # ... Parameter 'k2': Formula 'k1 * (step > 30)': the comparison '>' ...
```

When the model is built, the engine checks the variables each rate and each formula parameter reads. A variable that no parameter, bin, compartment, subgroup total, variable of a matrix or engine variable defines fails the build, with a message that names the transition or the parameter, the expression, the undefined variables, and the parameters and bins of the model: `Undefined variables in transition 'bc' rate 'k9 * B': k9. Available parameters: k1, k2. Available bins: A, B, C.` A category id, such as `g1`, is not a variable: write `total_g1` for its compartments.

## Limits of the parser

The parser of the engine is the only check of an expression. It checks each expression when its object is built: a rate when the transition is added, a formula parameter, a calibration constraint and an `AxisPattern`. An expression outside the language raises `ValueError` with the message of the parser, which quotes the expression (its first 80 characters, then `…`) and names the first part of it that is not in the language:

```python
try:
    ModelBuilder(name="Outside").add_bin(id="A", name="State A").add_transition(
        id="t", source=["A"], target=[], rate="__import__('os')"
    )
except ValueError as error:
    print("rejected")
```

The parser has two limits, and no setting changes them:

| Limit                                                                                                                                 | Value      | Message                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------- |
| Nesting: each pair of parentheses, each call, each sign and each power is one level; a sum or a product of any number of terms is one | 100 levels | `it nests deeper than 100 levels of parentheses, calls, signs and powers`                 |
| Nodes of the tree: numbers, variables, operators and calls; a sum of n terms has 2n - 1                                               | 100 000    | `it has … nodes (numbers, variables, operators and calls), more than the limit of 100000` |

## Next steps

| To learn                                    | Read                                                                                    |
| ------------------------------------------- | --------------------------------------------------------------------------------------- |
| Ready-made rates that change along the axis | [Axis Patterns](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md)     |
| How rates become flows                      | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)         |
| Running a model                             | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md) |
