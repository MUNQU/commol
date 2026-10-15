# Core Concepts

[Back to Home](https://munqu.github.io/commol/latest/index.md)

A Commol model is a set of **compartments** that each hold an **amount**, and **transitions** that move amounts between them along the **axis**, the independent variable of the model (time is the most common one). The amounts of every compartment together are the **state** of the model, and their sum is the **total**. An amount can be a count, a mass or a concentration. This page explains each building block once, with a runnable example. For every option of every method, see [Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md).

## The concepts at a glance

| Concept            | What it is                                                  | Builder method           |
| ------------------ | ----------------------------------------------------------- | ------------------------ |
| Compartment (bin)  | A state that holds an amount                                | `add_bin`                |
| Stratification     | A dimension that splits every compartment into categories   | `add_stratification`     |
| Parameter          | A named constant or formula used in rates                   | `add_parameter`          |
| Transition         | A flow between compartments, with a rate                    | `add_transition`         |
| Accumulator        | A counter of what a transition moved; not part of the state | `add_accumulator`        |
| Initial conditions | The total and how it is split at the start                  | `set_initial_conditions` |
| Model              | The validated definition                                    | `build`                  |
| Simulation         | Runs a model for a number of steps                          | `Simulation(model).run`  |

## Compartments

A compartment (a *bin*) holds an amount: a part of the total. A model can have any number of them, with any ids.

```python
from commol import ModelBuilder, Simulation

builder = (
    ModelBuilder(name="Concepts")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
)
```

## Parameters

A parameter is a named value that rates can use. Its value is a number, a formula (a string that can use other parameters and the [special variables](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md)), or `None` when calibration will find it.

```python
builder = (
    builder
    .add_parameter(id="k1", value=0.3, description="A to B rate")
    .add_parameter(id="k2", value=0.1, description="B to C rate")
)
```

## Transitions

A transition moves an amount from its `source` compartments to its `target` compartments along the axis. How much it moves depends on its rate:

| The rate…                                   | Kind             | Flow            | Example                |
| ------------------------------------------- | ---------------- | --------------- | ---------------------- |
| names a compartment, a bin total or `total` | **Absolute**     | the rate itself | `"k1 * A * B / total"` |
| names no compartment                        | **Proportional** | rate × source   | `"k2"` moves `k2 × B`  |

```python
builder = (
    builder
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
)
```

The engine decides the kind when the model is built, and `model.print_equations()` shows the result: the proportional rate above prints as `k2 * B`. See [Design decisions](https://munqu.github.io/commol/latest/codebase/design-decisions/#absolute-and-proportional-rates) for the exact rule.

## Accumulators

An accumulator counts what a transition has moved since step 0. It is an output of the simulation, not part of the state: nothing is ever taken from it, and it is not part of the total.

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

With stratifications, an accumulator expands like a bin (`cum_ab_g1`, `cum_ab_g2`, …). Resetting a simulation sets it back to 0.

## Initial conditions

The initial conditions give the total and the fraction of it that starts in each bin. The total is a real number, finite and greater than 0:

```python
builder = builder.set_initial_conditions(
    total=1000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
        {"bin": "C", "fraction": 0.0},
    ],
)
```

| Rule                                                    | Checked when           |
| ------------------------------------------------------- | ---------------------- |
| `total` is finite and greater than 0                    | The conditions are set |
| Each fraction is between 0.0 and 1.0                    | The conditions are set |
| The bin fractions sum to 1.0                            | The conditions are set |
| Every bin has a fraction                                | The model is built     |
| Every stratification has fractions, and they sum to 1.0 | The model is built     |

A fraction may be `None` when calibration will find it; see [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md).

## Build and run

`build` validates the definition and returns a `Model`. `Simulation` runs it in the Rust engine:

```python
model = builder.build(engine="DifferenceEquations")

results = Simulation(model).run(num_steps=50)

print(sorted(results))   # ['A', 'B', 'C', 'cum_ab']
print(len(results["A"]))  # 51: step 0 to step 50
```

A `Model` has a header (its name, parameters and step unit), the `engine` that runs it, and its `structure`: the part of its kind. `ModelBuilder` builds the compartmental kind, so `model.structure` is a `CompartmentalStructure` that holds the bins, stratifications, transitions and initial conditions:

```python
print(model.engine.type)                    # DifferenceEquations
print(model.structure.kind)                 # compartmental
print([b.id for b in model.structure.bins])  # ['A', 'B', 'C']
```

Two engines run a compartmental model, with the same structure:

| Engine                  | Time                                                                    | Crate                 |
| ----------------------- | ----------------------------------------------------------------------- | --------------------- |
| `DifferenceEquations`   | Advances in whole steps; each step applies every flow once.             | `commol-difference`   |
| `DifferentialEquations` | Continuous: the engine integrates `dX/dt = Σ flows` with an ODE solver. | `commol-differential` |

See [Differential equations](https://munqu.github.io/commol/latest/guide/simulations/#differential-equations).

### What a rate means

A flow is per unit of the axis in every engine, and the unit check is the same for both. What its value means is the engine's (`rate_meaning` of the engine's list, `EngineType` in `crates/commol-core/src/types/engine.rs`):

| Engine                  | `rate_meaning` | A flow of value `f`                                                                          | Equation                  |
| ----------------------- | -------------- | -------------------------------------------------------------------------------------------- | ------------------------- |
| `DifferenceEquations`   | `per_step`     | Moves `f` in one step: the flow of an absolute rate, or `f × source` for a proportional rate | `X(n+1) = X(n) + Σ flows` |
| `DifferentialEquations` | `continuous`   | Is the instantaneous rate of change: over a short span `Δ` it moves about `f × Δ`            | `dX/dt = Σ flows`         |

So one rate gives two different dynamics: a proportional rate `k` keeps `(1 - k)^n` of its source after `n` steps in a difference model, and `exp(-k n)` in a differential model. To find the rate that gives a mean span or a probability over a period in the engine of a model, use [`AxisUnits`](#units-of-the-axis): it asks the engine.

## Stratifications

A stratification splits every compartment into one compartment per category. The expanded name is `{bin}_{category}`, with one suffix per stratification, in the order the stratifications are declared:

```text
Bins: A, B           Stratification group = [g1, g2]

A -> A_g1, A_g2
B -> B_g1, B_g2      4 compartments (2 bins x 2 categories)
```

With several stratifications, the expansion is the product of all their categories:

```text
Stratifications group = [g1, g2] and type = [t1, t2]

A -> A_g1_t1, A_g1_t2, A_g2_t1, A_g2_t2      8 compartments for 2 bins
```

The initial amount of each compartment is the total times the product of its fractions:

```python
stratified = (
    ModelBuilder(name="Stratified")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .set_initial_conditions(
        total=10000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.6},
                    {"category": "g2", "fraction": 0.4},
                ],
            }
        ],
    )
    .build(engine="DifferenceEquations")
)

for compartment in stratified.structure.initial_state():
    print(compartment.name, compartment.amount)
```

| Compartment | Calculation       | Initial amount |
| ----------- | ----------------- | -------------- |
| `A_g1`      | 10000 × 0.9 × 0.6 | 5400           |
| `A_g2`      | 10000 × 0.9 × 0.4 | 3600           |
| `B_g1`      | 10000 × 0.1 × 0.6 | 600            |
| `B_g2`      | 10000 × 0.1 × 0.4 | 400            |

`CompartmentalStructure.initial_state()` returns this table as the engine builds it: each compartment with its bin, its categories and its amount.

Use a stratification when subgroups have different rates, when you need output per subgroup, or when the data you fit is broken down by category.

### Conditional stratifications

A stratification with `conditions` only splits the compartments whose earlier categories satisfy all of them. A condition may only name a stratification declared before it.

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
    .set_initial_conditions(
        total=10000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.6},
                    {"category": "g2", "fraction": 0.4},
                ],
            },
            {
                "stratification": "subtype",
                "fractions": [
                    {"category": "s1", "fraction": 0.5},
                    {"category": "s2", "fraction": 0.5},
                ],
            },
        ],
    )
    .build(engine="DifferenceEquations")
)

print([c.name for c in conditional.structure.initial_state()])
# ['A_g1', 'A_g2_s1', 'A_g2_s2', 'B_g1', 'B_g2_s1', 'B_g2_s2']
```

The fractions of a conditional stratification are fractions of the subgroup it splits, not of the total. `subgroup_total` multiplies along the chain to give the total of a subgroup, and `get_conditioning_categories` names the subgroup:

```python
print(conditional.structure.subgroup_total())                # 10000
print(conditional.structure.subgroup_total(["g2"]))          # 10000 x 0.4 = 4000
print(conditional.structure.subgroup_total(["g2", "s1"]))    # 4000 x 0.5 = 2000
print(conditional.structure.get_conditioning_categories("subtype"))  # ('g2',)
```

See [Conditional Stratifications](https://munqu.github.io/commol/latest/guide/building-models/#stratifications) for the full API.

### Per-compartment overrides

When the product of fractions cannot express the starting state, for example a state that is empty in only one category, `compartment_fractions` overrides single expanded compartments. Each fraction is a fraction of the **total** and replaces the product for that compartment only:

```python
overridden = (
    ModelBuilder(name="Override")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .set_initial_conditions(
        total=10000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.5},
                    {"category": "g2", "fraction": 0.5},
                ],
            }
        ],
        compartment_fractions=[{"compartment": "B_g1", "fraction": 0.0}],
    )
    .build(engine="DifferenceEquations")
)

print({c.name: c.amount for c in overridden.structure.initial_state()})
# B_g1 starts empty; the other compartments keep their product
```

| Rule                                                             | Checked when           |
| ---------------------------------------------------------------- | ---------------------- |
| Each name appears once, and each fraction is between 0.0 and 1.0 | The conditions are set |
| Each name is an expanded compartment                             | The model is built     |
| The initial amounts do not sum to more than `total`              | The model is built     |

`model.structure.update_compartment_fractions({"B_g1": 0.05})` changes or adds overrides on a built model.

### Updating fractions after building

```python
overridden.structure.update_initial_conditions({"A": 0.85, "B": 0.15})
overridden.structure.update_stratification_fractions({"g1": 0.7})  # g2 becomes 0.3
```

| Method                            | Rule                                                                                                        |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `update_initial_conditions`       | After the update, the bin fractions sum to 1.0. Move an amount between bins in one call.                    |
| `update_stratification_fractions` | Within each stratification you may omit one category; it receives what remains of 1.0. Omitting two raises. |

A category name that two stratifications declare must be qualified with its stratification, as in `{"group:g1": 0.5}`. Calibration writes its results back through these same methods.

## Several sources or targets

A transition with several sources or targets moves **one** flow: each source loses it and each target gains it. Its rate must be absolute, because a proportional rate has no single source to multiply.

```python
merge = (
    ModelBuilder(name="Merge")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k", value=0.001)
    .add_transition(id="t1", source=["A", "B"], target=["C", "C"], rate="k * A * B")
    .set_initial_conditions(
        total=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.5},
            {"bin": "B", "fraction": 0.5},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(engine="DifferenceEquations")
)
# Each step: A loses k*A*B, B loses k*A*B, C gains 2*k*A*B
```

For **independent** flows from several compartments, use the `$compartment` placeholder: the builder creates one transition per source. An empty `target` removes the flow from the system.

```python
outflow = (
    ModelBuilder(name="Outflow")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="mu", value=0.01)
    .add_transition(id="outflow", source=["A", "B"], target=[], rate="mu * $compartment")
    .set_initial_conditions(
        total=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.5},
            {"bin": "B", "fraction": 0.5},
        ],
    )
    .build(engine="DifferenceEquations")
)
print([t.id for t in outflow.structure.transitions])  # ['outflow__A', 'outflow__B']
```

## Units of the axis

A model knows only steps: a rate is per step, and a position or a period is a number of steps. `AxisUnits` converts values in units to steps, and gives the rate that the engine of a model reads. `Model.axis_units()` gives the one of a model: its `step_unit` (or `step` when it declares none) and its engine.

```python
from commol import AxisUnits

hourly = AxisUnits(step_unit="hour")

print(hourly.steps(1, "day"))     # 24.0
print(hourly.steps(3, "day"))     # 72.0
print(hourly.steps(30, "minute")) # 0.5

along_a_line = AxisUnits(step_unit="meter")
print(along_a_line.steps(2, "km"))  # 2000.0
```

`steps` works for any dimension, and raises `ValueError` for a unit of another dimension than the step, naming both units. It returns a `float`: whether a position must be a whole number of steps is the engine's check, made where the position is used. A result within `1e-9` of an integer is that integer.

The rate conversions ask the engine what a rate means ([What a rate means](#what-a-rate-means)):

| Method                                   | `DifferenceEquations` (`per_step`)                         | `DifferentialEquations` (`continuous`) |
| ---------------------------------------- | ---------------------------------------------------------- | -------------------------------------- |
| `rate_from_mean_span(span, unit)`        | `1 - exp(-1 / n)`, with `span` equal to `n` steps          | `1 / n`                                |
| `rate_from_probability(p, period, unit)` | `1 - (1 - p) ** (1 / n)`, with `period` equal to `n` steps | `-ln(1 - p) / n`                       |

```python
difference = AxisUnits(step_unit="day", engine="DifferenceEquations")
differential = AxisUnits(step_unit="day", engine="DifferentialEquations")

print(difference.rate_from_mean_span(4, "day"))    # 0.2212 = 1 - exp(-1/4)
print(differential.rate_from_mean_span(4, "day"))  # 0.25
print(difference.rate_from_probability(0.3, 1, "week"))
print(differential.rate_from_probability(0.3, 1, "week"))
```

With either rate, one outflow moves `p` of its bin over the period, and the bin keeps `exp(-n / span)` at each step `n`. In the difference model the mean residence is then `1 / rate` steps (`4.52` for a span of 4), not the span: a per-step rate cannot keep both the survival and the mean of the continuous process, and these conversions keep the survival.

The window grid (`window_start`, `window_end`, `window_index`) is in `py-commol/commol/api/windows.py`; see [Windowed totals](https://munqu.github.io/commol/latest/guide/simulations/#windowed-totals).

```python
from commol import window_end, window_index, window_start

week = int(hourly.steps(1, "week"))
print(window_start(0, week), window_end(0, week))  # 0 168
print(window_index(168, week))                     # 0
```

A window covers the positions `(start, end]`, the same interval that [windowed observations](https://munqu.github.io/commol/latest/guide/calibration-observations/#counts-per-period-windows) use.

## Next steps

| To learn                         | Read                                                                                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Every builder option             | [Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md)                   |
| What rate formulas can contain   | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) |
| Rates that change along the axis | [Axis Patterns](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md)                       |
| Running and reading simulations  | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md)                   |
