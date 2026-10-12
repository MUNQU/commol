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
unit `<bin unit>/<time unit>`, and that every function receives a plain number.
It raises `UnitConsistencyError` (from `commol.utils.equations`) and lists every
transition that fails.

### What needs a unit

| Item | How it gets a unit |
|---|---|
| Bins | `ModelBuilder(bin_unit=...)`, or `add_bin(unit=...)` for one bin |
| `N`, `N_<category>…` | The bin unit, automatically |
| Constant parameters | `add_parameter(unit=...)`. Required: the check raises if one is missing. |
| Formula parameters | Inferred from the formula; a unit is optional |
| `step`, `t`, `pi`, `e` | Dimensionless, automatically |

| Unit | Use for |
|---|---|
| `"1/day"`, `"1/week"`, `"1/hour"` | Rates |
| `"dimensionless"` | Ratios, fractions, amplitudes, function arguments |
| Any word, such as `"item"` | A custom unit for bins; Commol registers it |

The time unit of the flows comes from the parameters: the first parameter whose
unit has a known time unit (`second`, `minute`, `hour`, `day`, `week`, `month`,
`year`, …) sets it, and `day` is the default.

!!! warning "Per-step units"
    `step` is not a time unit for the check: a model whose rates are in
    `"1/step"` fails with "expected `<bin unit>/day`". Give rates a physical
    time unit, such as `"1/day"`.

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
| `Variable '…' … has no defined unit` | A bin has no unit (set `bin_unit`) |

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
