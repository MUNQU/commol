# Units and Equations

[Back to Building Models](building-models.md)

Commol can check that every rate has the right units, and can print the
equations that the engine simulates, as text or LaTeX. Both work on a built
`Model`.

## Quick path

```python
from commol import Aggregate, ModelBuilder

model = (
    ModelBuilder(name="Model with Units", bin_unit="item", step_unit="day")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.5, unit="1/day")
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .build(
        Aggregate(
            total=1000,
            bin_fractions={"A": 0.9, "B": 0.1, "C": 0.0},
        ),
        engine="DifferenceEquations",
    )
)

model.check_unit_consistency(verbose=True)  # Unit consistency check passed successfully.
model.print_equations()
```

## Unit checking

`Model.check_unit_consistency(verbose=False)` checks that every flow has the
unit `<bin unit>/<step unit>`, every [impulse](transitions.md#impulses) the
unit `<bin unit>`, and that every function receives a plain number. It checks
`DifferenceEquations` and `DifferentialEquations` models alike: in both, a rate
is per unit of the axis, and only what its value means differs (see
[What a rate means](core-concepts.md#what-a-rate-means)).
It raises `UnitConsistencyError` (from `commol.utils.equations`) and lists every
transition that fails.

### What needs a unit

| Item | How it gets a unit |
|---|---|
| Bins | `ModelBuilder(bin_unit=...)`, or `add_bin(unit=...)` for one bin. Required: the check names every bin without one. |
| `total`, `total_<category>…` | The bin unit, automatically |
| Constant parameters | `add_parameter(unit=...)`. Required: the check raises if one is missing. |
| Formula parameters | Inferred from the formula; a unit is optional |
| `t` | The step unit, automatically: it is a position on the axis |
| `step`, `pi`, `e` | Dimensionless, automatically: `step` counts increments |

| Unit | Use for |
|---|---|
| `"1/day"`, `"1/week"`, `"1/step"` | Rates |
| `"dimensionless"` | Ratios, fractions, amplitudes, function arguments |
| Any word that `pint` does not define, such as `"item"` | A unit of its own dimension; Commol registers it |

A unit means what [`pint`](https://pint.readthedocs.io/) defines, for time as
for any other quantity: `month`, `year` and `fortnight` are units of time, but
`quarter` is a unit of mass and `decade` is a number. A word that `pint` does
not define, such as `mo`, becomes a unit of its own dimension, which no other
unit converts to.

### The step unit

The engine converts no value between units. So one step has one unit, and every
rate must use it (`UnitChecker._step_unit` in
`py-commol/commol/context/_model/units.py`).

The axis of a model is its independent variable: the quantity the model steps
along or differentiates over. Time is one case; a length, a dose or any other
quantity is another. So the step unit may have any single dimension. Declare it
with `ModelBuilder(..., step_unit="day")` (`Model.step_unit`); `check_step_unit`
accepts one unit of one dimension with exponent 1, such as `day` or `meter`, and
rejects `1/day` or `meter**2`. A model that declares no step unit has the
step unit `step`, the default increment of any axis, of its own dimension
`[step]`. The check never guesses the step unit from the parameters.

| Model | Step unit | Flows must be |
|---|---|---|
| `step_unit="day"` | `day` | `<bin unit>/day`: a rate in `1/d` passes, a rate in `1/week` or `1/quarter` fails |
| `step_unit="meter"` | `meter` | `<bin unit>/meter` |
| No `step_unit`, rates in `1/step` or none | `step` | `<bin unit>/step` |
| No `step_unit`, rates in `1/day` | `step` | Fails; the error says to declare `step_unit` |

Two names of one unit, such as `d` and `day`, are one unit. A flow must have
exactly the expected unit: `item/week` does not match `item/day`, because the
engine would read a weekly rate as a daily one.

```python
along_a_line = (
    ModelBuilder(name="Along a line", bin_unit="item", step_unit="meter")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.01, unit="1/meter")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A")
    .build(
        Aggregate(
            total=1000,
            bin_fractions={"A": 1.0, "B": 0.0},
        ),
        engine="DifferentialEquations",
    )
)
along_a_line.check_unit_consistency(verbose=True)  # Unit consistency check passed successfully.
```

`Model.axis_units()` converts values in units of the step dimension to steps,
and gives the rates the engine reads; see
[Units of the axis](core-concepts.md#units-of-the-axis).

### Positions and numbers in a rate

`t` is a position on the axis, so it has the step unit: with `step_unit="day"`,
`k * t * A` passes when `k` is in `1/day**2`. `step` counts increments and is
dimensionless.

A number written in a rate and added to, subtracted from or compared with a
quantity is read in the unit of that quantity, as the engine reads it:
`if(t >= 10, k, 0)` compares `t` with 10 days, and `t - 3` is 3 days before
`t`. The two branches of `if` and the arguments of `min` and `max` must share
a unit, and a written `0` takes the unit of the other branch. A number that a
function computes, such as `exp(2)`, stays dimensionless.

### Impulses

An impulse moves an amount at once, so its rate must have the bin unit:
`p * A` with `p` in `"dimensionless"` is an amount of `<bin unit>`; `k * A`
with `k` in `"1/day"` is a flow and fails with `expected '<bin unit>'`.

### What the check catches

```python
from commol.utils.equations import UnitConsistencyError

wrong_units = (
    ModelBuilder(name="Wrong units", bin_unit="item", step_unit="day")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.5, unit="day")  # should be "1/day"
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_ba", source=["B"], target=["A"], rate="k2 * sin(B) * B")
    .build(
        Aggregate(
            total=1000,
            bin_fractions={"A": 0.9, "B": 0.1},
        ),
        engine="DifferenceEquations",
    )
)

try:
    wrong_units.check_unit_consistency()
except UnitConsistencyError as error:
    print(error)
```

| Error | Cause |
|---|---|
| `Unit mismatch: equation '…' has unit 'day * item' but expected 'item/day'` | A rate with the wrong unit |
| `Cannot convert from 'item' to 'dimensionless'` | A function such as `sin`, `exp` or `log` received a quantity with a unit |
| `The following constant parameters are missing units: …` | A constant parameter has no unit |
| `The following bins have no unit: …` | A bin has no unit (set `bin_unit`) |
| `The model declares no step unit, so it is 'step'` | Rates in a unit such as `1/day` without `step_unit`; see [The step unit](#the-step-unit) |
| `Variable '…' … has no defined unit` | A variable of a rate has no unit |

To pass a quantity to a function, make it dimensionless first, for example
`sin(2 * pi * step / 365)` or `exp(-decay * step)` with `decay` in
`"dimensionless"`, or `sin(2 * pi * t / period)` with `period` in the step
unit.

## Printing equations

`Model.print_equations(output_file=None, format="text")` prints the flows that
the engine plans, after it has decided which rates are proportional (see
[Transitions](transitions.md#absolute-and-proportional-rates)). It has three
parts:

| Part | Content |
|---|---|
| Model information | Name, engine, and the number of bins, stratifications, parameters and transitions |
| Compact form | Each transition and its flows, `source -> target: rate`. With units, each symbol shows its unit and each flow its resulting unit. |
| Expanded form | One difference equation per compartment: `[A(t+Dt) - A(t)] / Dt = …` |

For the model above, the compact form is:

```text
t_ab (A -> B):
  A -> B: k1(1/day) * A(item) * B(item) / total(item) [item / day]
t_bc (B -> C):
  B -> C: k2(1/day) * B(item) [item / day]
```

and the expanded form is:

```text
[A(t+Dt) - A(t)] / Dt = - (k1 * A * B / total)
[B(t+Dt) - B(t)] / Dt = (k1 * A * B / total) - (k2 * B)
[C(t+Dt) - C(t)] / Dt = (k2 * B)
```

| Argument | Values | Effect |
|---|---|---|
| `format` | `"text"` (default), `"latex"` | LaTeX writes inline math for the compact form and display math (`\[…\]`) for the expanded form. |
| `output_file` | `None` (default) or a path | Writes to the file instead of printing. |

```python
model.print_equations(format="latex", output_file="equations.tex")
```

Units appear only when every bin and every constant parameter has one. If only
some parameters have a unit, `print_equations` raises `ValueError` ("Some
parameters have units but not all").

## Next steps

| To learn | Read |
|---|---|
| What a formula can contain | [Mathematical Expressions](mathematical-expressions.md) |
| Running the model | [Running Simulations](simulations.md) |
