# Units

[Back to Building Models](building-models.md)

Commol checks that every rate of a built `Model` has the right units. To see
the equations the engine simulates, with their units, see
[Describing a Model](describing-models.md).

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
            bin_amounts={"A": 900, "B": 100, "C": 0},
        ),
        engine="DifferenceEquations",
    )
)

model.check_unit_consistency(verbose=True)  # Unit consistency check passed successfully.
```

## Unit checking

`Model.check_unit_consistency(verbose=False)` checks that every flow has the
unit `<bin unit>/<step unit>`, every [impulse](transitions.md#impulses) the
unit `<bin unit>`, and that every function receives a plain number. It checks
`DifferenceEquations` and `DifferentialEquations` models alike: in both, a rate
is per unit of the axis, and only what its value means differs (see
[What a rate means](core-concepts.md#what-a-rate-means)).
It reads the flows the engine plans (`transition_flows`), so a proportional
rate is checked as its flow, the rate times its source: `k2` in `1/day` from a
bin in `item` gives `item/day`. The bin unit of a flow is the unit of the bin
it leaves, or of the bin it enters when it has no source; so a category
transition in a model of bins in `item` and `kg` moves `item/day` in one bin
and `kg/day` in the other. In such a model `total`, a subgroup total
`total_<categories>`, a category and `neighbors` add amounts of different
units, so they have none, and a rate that reads one fails the check; read the
bins or the compartments instead. A rate that reads the
[neighbors of a unit](building-models.md#neighbors) is checked as the sums
they are. It raises `UnitConsistencyError` (from `commol.utils.equations`) and
lists every transition that fails.

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
            bin_amounts={"A": 1000, "B": 0},
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
            bin_amounts={"A": 900, "B": 100},
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

## Dates on the axis

An axis of time can start at a calendar moment: `ModelBuilder(origin=...)`
takes a `datetime.date` (read as its midnight) or a `datetime.datetime` with
no time zone. `Model.axis_units()` then converts between dates and positions,
and gives the positions where calendar days and weeks end.

```python
import calendar
from datetime import date, datetime

dated = (
    ModelBuilder(name="Dated", step_unit="hour", origin=date(2020, 1, 1))
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.01)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A")
    .build(Aggregate(bin_amounts={"A": 1000, "B": 0}), engine="DifferenceEquations")
)
axis = dated.axis_units()

print(axis.position_of(date(2020, 1, 2)))         # 24.0
print(axis.position_of(datetime(2020, 1, 1, 6)))  # 6.0
print(axis.moment_of(36))                         # 2020-01-02 12:00:00
print(axis.day_ends(6, 72))                       # [24.0, 48.0, 72.0]
# 2020-01-01 is a Wednesday: the first week that starts on Sunday ends on 5 January.
print(axis.week_ends(0, 400, first_weekday=calendar.SUNDAY))  # [96.0, 264.0]
```

| Method | Gives |
|---|---|
| `position_of(moment)` | `(moment - origin)` in steps; negative before the origin. |
| `moment_of(position)` | `origin + position` steps, a `datetime` with no time zone. The inverse of `position_of`. |
| `day_ends(start, end)` | The positions of the midnights in `(start, end]`. |
| `week_ends(start, end, first_weekday)` | The positions in `(start, end]` where a week starts, at midnight of `first_weekday`: 0 (Monday) to 6 (Sunday), as `date.weekday()` and `calendar.SUNDAY` number them. |

A period ends where the next one starts, so its end is the midnight that
starts the next day or week. The range `(start, end]` is the range of a
[window](calibration-observations.md#counts-per-period-windows): a period that
ends at `start` is not in it, and the first end can close a partial period
that starts before `start`. The number of a period (a week number of a
national rule) is not part of the axis: label a period from its end with your
own function.

Rules of the origin, each a `ValueError`:

| Rule | Why |
|---|---|
| A conversion needs an origin. | An axis without one has no dates. |
| The moment has no time zone. | A step has no time zone, so `(moment - origin) / step` stays exact through a change of summer time. Convert an aware moment with `moment.astimezone(zone).replace(tzinfo=None)`. |
| The step unit is a unit of time whose length is a whole number of days or a whole fraction of a day, such as `hour`, `day` or `week`. | `month` and `year` are units of a fixed length (a month is 1/12 of a year, about 30.44 days), not the length of a calendar month or year, so their positions would not fall on calendar dates. |

With an origin, [`results.plot.series()`](simulations.md#plotting)
labels its ticks with dates when it has no `position_label`.

## Next steps

| To learn | Read |
|---|---|
| The equations the engine simulates | [Describing a Model](describing-models.md) |
| What a formula can contain | [Mathematical Expressions](mathematical-expressions.md) |
| Running the model | [Running Simulations](simulations.md) |
