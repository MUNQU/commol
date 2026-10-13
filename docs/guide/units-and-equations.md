# Units and Equations

[Back to Building Models](building-models.md)

Commol can check that every rate has the right units, and can print the
equations that the engine simulates, as text or LaTeX. Both work on a built
`Model`.

## Quick path

```python
from commol import ModelBuilder

model = (
    ModelBuilder(name="Model with Units", bin_unit="item")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.5, unit="1/day")
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)

model.check_unit_consistency(verbose=True)  # Unit consistency check passed successfully.
model.print_equations()
```

## Unit checking

`Model.check_unit_consistency(verbose=False)` checks that every flow has the
unit `<bin unit>/<step unit>`, and that every function receives a plain number.
It raises `UnitConsistencyError` (from `commol.utils.equations`) and lists every
transition that fails.

### What needs a unit

| Item | How it gets a unit |
|---|---|
| Bins | `ModelBuilder(bin_unit=...)`, or `add_bin(unit=...)` for one bin. Required: the check names every bin without one. |
| `N`, `N_<category>…` | The bin unit, automatically |
| Constant parameters | `add_parameter(unit=...)`. Required: the check raises if one is missing. |
| Formula parameters | Inferred from the formula; a unit is optional |
| `step`, `t`, `pi`, `e` | Dimensionless, automatically |

| Unit | Use for |
|---|---|
| `"1/day"`, `"1/week"`, `"1/step"` | Rates |
| `"dimensionless"` | Ratios, fractions, amplitudes, function arguments |
| Any word, such as `"item"` | A custom unit for bins; Commol registers it |

### The step unit

The engine applies every flow once per step and converts no value between
units. So one step has one unit, and every rate must use it
(`UnitChecker._step_unit` in `py-commol/commol/context/_model/units.py`):

| Parameter units | Step unit | Result |
|---|---|---|
| All in one unit of time, such as `"1/day"` and `"day"` | `day` | Flows must be `<bin unit>/day` |
| All in `"1/week"` | `week` | Accepted: one step is one week |
| All in `"1/step"` | `step` | Accepted |
| No unit of time | `step` | Flows must be `<bin unit>/step` |
| `"1/day"` and `"1/week"`, or `"1/day"` and `"1/step"` | — | Error that names each parameter and its unit |

A unit of time is any unit of dimension `[time]` (`second`, `hour`, `day`,
`week`, `month`, `year`, …) or `step`. Two names of one unit, such as `d` and
`day`, are one unit. A flow must have exactly the expected unit: `person/week`
does not match `person/day`, because the engine would apply a weekly rate
every day.

### What the check catches

```python
from commol.utils.equations import UnitConsistencyError

wrong_units = (
    ModelBuilder(name="Wrong units", bin_unit="item")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.5, unit="day")  # should be "1/day"
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_ba", source=["B"], target=["A"], rate="k2 * sin(B) * B")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 0.9}, {"bin": "B", "fraction": 0.1}],
    )
    .build(typology="DifferenceEquations")
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
| `the parameters use more than one unit of time ('k1' in '1/day', 'k2' in '1/week')` | Rates in two units of time; see [The step unit](#the-step-unit) |
| `Variable '…' … has no defined unit` | A variable of a rate has no unit |

To pass a quantity to a function, make it dimensionless first, for example
`sin(2 * pi * step / 365)` or `exp(-decay * step)` with `decay` in
`"dimensionless"`.

## Printing equations

`Model.print_equations(output_file=None, format="text")` prints the flows that
the engine plans, after it has decided which rates are per-capita (see
[Transitions](transitions.md#absolute-and-per-capita-rates)). It has three
parts:

| Part | Content |
|---|---|
| Model information | Name, typology, and the number of bins, stratifications, parameters and transitions |
| Compact form | Each transition and its flows, `source -> target: rate`. With units, each symbol shows its unit and each flow its resulting unit. |
| Expanded form | One difference equation per compartment: `[A(t+Dt) - A(t)] / Dt = …` |

For the model above, the compact form is:

```text
t_ab (A -> B):
  A -> B: k1(1/day) * A(item) * B(item) / N(item) [item / day]
t_bc (B -> C):
  B -> C: k2(1/day) * B(item) [item / day]
```

and the expanded form is:

```text
[A(t+Dt) - A(t)] / Dt = - (k1 * A * B / N)
[B(t+Dt) - B(t)] / Dt = (k1 * A * B / N) - (k2 * B)
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
