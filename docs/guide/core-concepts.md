# Core Concepts

[Back to Home](../index.md)

A Commol model is a set of **compartments** that each hold an **amount**, and
**transitions** that move amounts between them along the **axis**, the
independent variable of the model (time is the most common one). The amounts of
every compartment together are the **state** of the model, and their sum is the
**total**. An amount can be a count, a mass or a concentration. This page explains each
building block once, with a runnable example. For every option of every method,
see [Building Models](building-models.md).

## The concepts at a glance

| Concept | What it is | Builder method |
|---|---|---|
| Compartment (bin) | A state that holds an amount | `add_bin` |
| Stratification | A dimension that splits every compartment into categories | `add_stratification` |
| Parameter | A named constant or formula used in rates | `add_parameter` |
| Transition | A flow between compartments, with a rate | `add_transition` |
| Accumulator | A counter of what a transition moved; not part of the state | `add_accumulator` |
| Process | The bins, stratifications, transitions and accumulators: what changes and how, written once | the `add_*` methods |
| Structure | How the process is laid out; `Aggregate` holds amounts per compartment and says how much starts where | `Aggregate(...)` |
| Model | The validated definition: a header, a process, a structure and an engine | `build(structure, engine)` |
| Simulation | Runs a model for a number of steps | `Simulation(model).run` |

## Compartments

A compartment (a *bin*) holds an amount, in the unit of the bin. A model can have
any number of them, with any ids.

```python
from commol import Aggregate, ModelBuilder, Simulation

builder = (
    ModelBuilder(name="Concepts")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
)
```

## Parameters

A parameter is a named value that rates can use. Its value is a number, a
formula (a string that can use other parameters and the
[special variables](mathematical-expressions.md)), or `None` when calibration
will find it.

```python
builder = (
    builder
    .add_parameter(id="k1", value=0.3, description="A to B rate")
    .add_parameter(id="k2", value=0.1, description="B to C rate")
)
```

## Transitions

A transition moves an amount from its `source` compartments to its `target`
compartments along the axis. How much it moves depends on its rate:

| The rate… | Kind | Flow | Example |
|---|---|---|---|
| names a compartment or a bin total | **Absolute** | the rate itself | `"k1 * A * B / total"` |
| names no compartment | **Proportional** | rate × source | `"k2"` moves `k2 × B` |

```python
builder = (
    builder
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
)
```

The engine decides the kind when the model is built, and
[`model.describe()`](describing-models.md) shows the result: the proportional
rate above shows as `k2 * B`. [Transitions](transitions.md#absolute-and-proportional-rates)
gives the exact rule, with its edge cases.

## Accumulators

An accumulator counts what a transition has moved since step 0. It is an output
of the simulation, not part of the state: nothing is ever taken from it, and
it is not part of the total.

```python
builder = builder.add_accumulator(id="cum_ab", name="Total moved from A to B")
builder = builder.add_transition(
    id="t_ab_counted",
    source=["A"],
    target=["B"],
    rate="0.01 * A",
    accumulators=["cum_ab"],
)
```

With stratifications, an accumulator expands like a bin (`cum_ab_g1`,
`cum_ab_g2`, …). Resetting a simulation sets it back to 0.

## Initial conditions: the structure

The calls above write the **process**: what changes and how. The
**structure** lays the process out. `Aggregate` holds an amount per
compartment, and gives the amount that each bin starts with, in the unit of
the bin. An amount is a real number, finite and not negative:

```python
structure = Aggregate(bin_amounts={"A": 900, "B": 100, "C": 0})
```

Every bin needs an amount. An amount may be `None` when calibration will find
it ([Model Calibration](calibration.md)). [The structure](building-models.md#the-structure-aggregate)
lists every argument of `Aggregate` and when each rule is checked.

## Build and run

`build` joins the process with the structure and an engine, validates the
definition and returns a `Model`. `Simulation` runs it in the Rust engine:

```python
model = builder.build(structure, engine="DifferenceEquations")

results = Simulation(model).run(num_steps=50)

print(sorted(results))   # ['A', 'B', 'C', 'cum_ab']
print(len(results["A"]))  # 51: step 0 to step 50
```

A `Model` has a header (its name, parameters and step unit), the `process`
(bins, stratifications, transitions and accumulators), the `structure` that
lays it out, and the `engine` that runs it. The process is written once; the
structure is what changes between kinds of model, so `ModelBuilder` builds
every kind its process fits. Here `model.structure` is an `Aggregate`:

```python
print(model.engine.type)                    # DifferenceEquations
print(model.structure.kind)                 # aggregate
print([b.id for b in model.process.bins])   # ['A', 'B', 'C']
print(model.structure.bin_amounts)          # {'A': 900.0, 'B': 100.0, 'C': 0.0}
```

Four engines run an aggregate model, with the same process and structure:

| Engine | Time | Crate |
|---|---|---|
| `DifferenceEquations` | Advances in whole steps; each step applies every flow once. | `commol-difference` |
| `DifferentialEquations` | Continuous: the engine integrates `dX/dt = Σ flows` with an ODE solver. | `commol-differential` |
| `StochasticEvents` | Continuous, at random: whole units move one event at a time. | `commol-stochastic` |
| `StochasticSteps` | Whole steps, at random: each step draws how many units each flow moves. | `commol-stochastic` |

See [Differential equations](simulations.md#differential-equations) and
[Stochastic Simulation](stochastic.md).

### What a rate means

A flow is per unit of the axis in every engine, and the unit check is the same
for every engine. What its value means is the engine's (`rate_meaning` of the engine's
list, `EngineType` in `crates/commol-core/src/types/engine.rs`):

| Engine | `rate_meaning` | A flow of value `f` | Equation |
|---|---|---|---|
| `DifferenceEquations` | `per_step` | Moves `f` in one step: the flow of an absolute rate, or `f × source` for a proportional rate | `X(n+1) = X(n) + Σ flows` |
| `DifferentialEquations` | `continuous` | Is the instantaneous rate of change: over a short span `Δ` it moves about `f × Δ` | `dX/dt = Σ flows` |
| `StochasticEvents` | `continuous` | Is the propensity of its events: the expected number of units it moves per unit of the axis | `dX/dt = Σ flows`, in the mean |
| `StochasticSteps` | `per_step` | Is the expected number of units it moves in one step; a proportional rate is the probability that one unit moves | `X(n+1) = X(n) + Σ flows`, in the mean |

So one rate gives two different dynamics: a proportional rate `k` keeps
`(1 - k)^n` of its source after `n` steps in a difference model, and
`exp(-k n)` in a differential model. To find the rate that gives a mean span
or a probability over a period in the engine of a model, use
[`AxisUnits`](#units-of-the-axis): it asks the engine.

## Stratifications

A stratification splits every compartment into one compartment per category.
The expanded name is `{bin}_{category}`, with one suffix per stratification, in
the order the stratifications are declared:

```text
Bins: A, B           Stratification group = [g1, g2]

A -> A_g1, A_g2
B -> B_g1, B_g2      4 compartments (2 bins x 2 categories)
```

With several stratifications, the expansion is the product of all their
categories:

```text
Stratifications group = [g1, g2] and type = [t1, t2]

A -> A_g1_t1, A_g1_t2, A_g2_t1, A_g2_t2      8 compartments for 2 bins
```

The initial amount of each compartment is the amount of its bin times the
product of the fractions of its categories:

```python
stratified = (
    ModelBuilder(name="Stratified")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .build(
        Aggregate(
            bin_amounts={"A": 9000, "B": 1000},
            stratification_fractions={"group": {"g1": 0.6, "g2": 0.4}},
        ),
        engine="DifferenceEquations",
    )
)

for compartment in stratified.structure.initial_state(stratified.process):
    print(compartment.name, compartment.amount)
```

| Compartment | Calculation | Initial amount |
|---|---|---|
| `A_g1` | 9000 × 0.6 | 5400 |
| `A_g2` | 9000 × 0.4 | 3600 |
| `B_g1` | 1000 × 0.6 | 600 |
| `B_g2` | 1000 × 0.4 | 400 |

`model.structure.initial_state(model.process)` returns this table as the
engine builds it: each compartment with its bin, its categories and its
amount.

Use a stratification when subgroups have different rates, when you need output
per subgroup, or when the data you fit is broken down by category.

### Conditional stratifications

A stratification with `conditions` only splits the compartments whose earlier
categories satisfy all of them. A condition may only name a stratification
declared before it.

```text
Stratifications group = [g1, g2] and subtype = [s1, s2] (only where group = g2)

A -> A_g1, A_g2_s1, A_g2_s2      3 compartments per bin instead of 4
```

```python
conditional = (
    ModelBuilder(name="Conditional")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_stratification(
        id="subtype",
        categories=["s1", "s2"],
        conditions=[{"stratification": "group", "category": "g2"}],
    )
    .add_parameter(id="k1", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .build(
        Aggregate(
            bin_amounts={"A": 9000, "B": 1000},
            stratification_fractions={
                "group": {"g1": 0.6, "g2": 0.4},
                "subtype": {"s1": 0.5, "s2": 0.5},
            },
        ),
        engine="DifferenceEquations",
    )
)

print([c.name for c in conditional.structure.initial_state(conditional.process)])
# ['A_g1', 'A_g2_s1', 'A_g2_s2', 'B_g1', 'B_g2_s1', 'B_g2_s2']
```

The fractions of a conditional stratification are fractions of the subgroup it
splits, not of the whole bin, so they multiply along the chain.
`subgroup_total` sums the initial state over the compartments of a subgroup,
and `get_conditioning_categories` names the subgroup:

```python
aggregate, process = conditional.structure, conditional.process
print(aggregate.subgroup_total(process))                 # 10000
print(aggregate.subgroup_total(process, ["g2"]))         # 10000 x 0.4 = 4000
print(aggregate.subgroup_total(process, ["g2", "s1"]))   # 4000 x 0.5 = 2000
print(aggregate.get_conditioning_categories(process, "subtype"))  # ('g2',)
```

See [Conditional Stratifications](building-models.md#stratifications)
for the full API.

### Per-compartment overrides

When the product of fractions cannot express the starting state, for example a
state that is empty in only one category, `compartment_amounts` overrides
single expanded compartments. Each amount is in the unit of the bin of its
compartment and replaces the product for that compartment only:

```python
overridden = (
    ModelBuilder(name="Override")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .build(
        Aggregate(
            bin_amounts={"A": 9000, "B": 1000},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
            compartment_amounts={"B_g1": 0},
        ),
        engine="DifferenceEquations",
    )
)

print({c.name: c.amount for c in overridden.structure.initial_state(overridden.process)})
# B_g1 starts empty; the other compartments keep their product
```

| Rule | Checked when |
|---|---|
| Each amount is finite and not negative | `Aggregate(...)` |
| Each name is an expanded compartment | The model is built |

An override may hold more than the product gave its compartment: the bins keep
no sum. `model.structure.update_compartment_amounts(model.process, {"B_g1": 50})`
changes or adds overrides on a built model; the engine checks them against
the expansion of the process before they are stored.

### Updating the initial state after building

```python
overridden.structure.update_bin_amounts({"A": 8500, "B": 1500})
overridden.structure.update_stratification_fractions({"g1": 0.7})  # g2 becomes 0.3
```

| Method | Rule |
|---|---|
| `update_bin_amounts` | Each amount is finite and not negative, or `None` for calibration. The other bins keep their amounts. |
| `update_stratification_fractions` | Within each stratification you may omit one category; it receives what remains of 1.0. Omitting two raises. |

A category name that two stratifications declare must be qualified with its
stratification, as in `{"group:g1": 0.5}`. Calibration writes its results back
through these same methods.

## Several sources or targets

A transition can have several sources or several targets. It then moves
**one** flow, which each source loses and each target gains, so its rate must
be absolute. For independent flows of one formula from several compartments,
the `$compartment` placeholder makes one transition per compartment. See
[Several sources or targets](transitions.md#several-sources-or-targets).

## Units of the axis

A model knows only steps: a rate is per step, and a position or a period is a
number of steps. `AxisUnits` converts values in units to steps, and gives the
rate that the engine of a model reads. `Model.axis_units()` gives the one of a
model: its `step_unit` (or `step` when it declares none), its `origin` and its
engine.

```python
from commol import AxisUnits

hourly = AxisUnits(step_unit="hour")

print(hourly.steps(1, "day"))     # 24.0
print(hourly.steps(3, "day"))     # 72.0
print(hourly.steps(30, "minute")) # 0.5

along_a_line = AxisUnits(step_unit="meter")
print(along_a_line.steps(2, "km"))  # 2000.0
```

`steps` works for any dimension, and raises `ValueError` for a unit of another
dimension than the step, naming both units. It returns a `float`: whether a
position must be a whole number of steps is the engine's check, made where the
position is used. A result within `1e-9` of an integer is that integer.

The rate conversions ask the engine what a rate means
([What a rate means](#what-a-rate-means)):

| Method | `DifferenceEquations` (`per_step`) | `DifferentialEquations` (`continuous`) |
|---|---|---|
| `rate_from_mean_span(span, unit)` | `1 - exp(-1 / n)`, with `span` equal to `n` steps | `1 / n` |
| `rate_from_probability(p, period, unit)` | `1 - (1 - p) ** (1 / n)`, with `period` equal to `n` steps | `-ln(1 - p) / n` |

```python
difference = AxisUnits(step_unit="day", engine="DifferenceEquations")
differential = AxisUnits(step_unit="day", engine="DifferentialEquations")

print(difference.rate_from_mean_span(4, "day"))    # 0.2212 = 1 - exp(-1/4)
print(differential.rate_from_mean_span(4, "day"))  # 0.25
print(difference.rate_from_probability(0.3, 1, "week"))
print(differential.rate_from_probability(0.3, 1, "week"))
```

With either rate, one outflow moves `p` of its bin over the period, and the bin
keeps `exp(-n / span)` at each step `n`. In the difference model the mean
residence is then `1 / rate` steps (`4.52` for a span of 4), not the span: a
per-step rate cannot keep both the survival and the mean of the continuous
process, and these conversions keep the survival.

An axis of time with an origin also converts between dates and positions,
and gives the ends of calendar days and weeks; see
[Dates on the axis](units-and-equations.md#dates-on-the-axis).

The axis also gives the window grid. A window has a width, a span in a unit,
which must be a whole number of steps; any other width raises `ValueError`,
naming the span, its unit and the step unit. A width in steps is the span in
the step unit, such as `(7, "day")` for 7 steps of a day, or `(7, "step")`
when the model declares no step unit.

| Method | Returns |
|---|---|
| `window_start(i, span, unit)`, `window_end(i, span, unit)` | The positions that bound window `i`: `i · width` and `(i + 1) · width`, with the width in steps |
| `window_index(at, span, unit)` | The window that holds a position, `max(ceil(at / width) − 1, 0)`; the inverse of `window_end`. Reports place observations with this rule. |
| `window_ends(span, unit, end)` | Every complete window end up to the position `end` |

```python
print(hourly.window_start(0, 1, "week"), hourly.window_end(0, 1, "week"))  # 0.0 168.0
print(hourly.window_end(2, 1, "week"))       # 504.0
print(hourly.window_index(168, 1, "week"))   # 0
print(hourly.window_index(168.5, 1, "week")) # 1
print(hourly.window_ends(1, "day", 100))     # [24.0, 48.0, 72.0, 96.0]
```

A window covers the positions `(start, end]`, the same interval that
[windowed observations](calibration-observations.md#counts-per-period-windows)
use. [Windowed totals](simulations.md#windowed-totals) reads the gain of a
series over each window.

## Next steps

| To learn | Read |
|---|---|
| Every builder option | [Building Models](building-models.md) |
| What rate formulas can contain | [Mathematical Expressions](mathematical-expressions.md) |
| Rates that change along the axis | [Axis Patterns](axis-patterns.md) |
| Running and reading simulations | [Running Simulations](simulations.md) |
