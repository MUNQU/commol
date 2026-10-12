# Core Concepts

[Back to Home](../index.md)

A Commol model is a set of **compartments** that hold a population, and
**transitions** that move it between them at each step. This page explains each
building block once, with a runnable example. For every option of every method,
see [Building Models](building-models.md).

## The concepts at a glance

| Concept | What it is | Builder method |
|---|---|---|
| Compartment (bin) | A state that holds part of the population | `add_bin` |
| Stratification | A dimension that splits every compartment into categories | `add_stratification` |
| Parameter | A named constant or formula used in rates | `add_parameter` |
| Transition | A flow between compartments, with a rate | `add_transition` |
| Accumulator | A counter of what a transition moved; not part of the population | `add_accumulator` |
| Initial conditions | The population and how it starts out | `set_initial_conditions` |
| Model | The validated definition | `build` |
| Simulation | Runs a model for a number of steps | `Simulation(model).run` |

## Compartments

A compartment (a *bin*) is a state that part of the population is in. A model
can have any number of them, with any ids.

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

A transition moves population from its `source` compartments to its `target`
compartments at each step. How much it moves depends on its rate:

| The rate… | Kind | Flow per step | Example |
|---|---|---|---|
| names a compartment, a bin total or `N` | **Absolute** | the rate itself | `"k1 * A * B / N"` |
| names no compartment | **Per-capita** | rate × source | `"k2"` moves `k2 × B` |

```python
builder = (
    builder
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
)
```

The engine decides the kind when the model is built, and
`model.print_equations()` shows the result: the per-capita rate above prints as
`k2 * B`. See [Design decisions](../codebase/design-decisions.md#absolute-and-per-capita-rates)
for the exact rule.

## Accumulators

An accumulator counts what a transition has moved since step 0. It is an output
of the simulation, not part of the population: nothing is ever taken from it.

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

## Initial conditions

The initial conditions give the total population and the fraction that starts
in each bin:

```python
builder = builder.set_initial_conditions(
    population_size=1000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
        {"bin": "C", "fraction": 0.0},
    ],
)
```

| Rule | Checked when |
|---|---|
| `population_size` is greater than 0 | The conditions are set |
| Each fraction is between 0.0 and 1.0 | The conditions are set |
| The bin fractions sum to 1.0 | The conditions are set |
| Every bin has a fraction | The model is built |
| Every stratification has fractions, and they sum to 1.0 | The model is built |

A fraction may be `None` when calibration will find it; see
[Model Calibration](calibration.md).

## Build and run

`build` validates the definition and returns a `Model`. `Simulation` runs it in
the Rust engine:

```python
model = builder.build(typology="DifferenceEquations")

results = Simulation(model).run(num_steps=50)

print(sorted(results))   # ['A', 'B', 'C', 'cum_ab']
print(len(results["A"]))  # 51: step 0 to step 50
```

`DifferenceEquations` is the only typology that runs: time advances in whole
steps, and each step applies every flow once.

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

The initial population of each compartment is the product of its fractions:

```python
stratified = (
    ModelBuilder(name="Stratified")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .set_initial_conditions(
        population_size=10000,
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
    .build(typology="DifferenceEquations")
)

for compartment in stratified.initial_state():
    print(compartment.name, compartment.population)
```

| Compartment | Calculation | Initial population |
|---|---|---|
| `A_g1` | 10000 × 0.9 × 0.6 | 5400 |
| `A_g2` | 10000 × 0.9 × 0.4 | 3600 |
| `B_g1` | 10000 × 0.1 × 0.6 | 600 |
| `B_g2` | 10000 × 0.1 × 0.4 | 400 |

`Model.initial_state()` returns this table as the engine builds it: each
compartment with its bin, its categories and its population.

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
    .set_initial_conditions(
        population_size=10000,
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
    .build(typology="DifferenceEquations")
)

print([c.name for c in conditional.initial_state()])
# ['A_g1', 'A_g2_s1', 'A_g2_s2', 'B_g1', 'B_g2_s1', 'B_g2_s2']
```

The fractions of a conditional stratification are fractions of the subgroup it
splits, not of the whole population. `subgroup_population` multiplies along the
chain, and `get_conditioning_categories` names the subgroup:

```python
print(conditional.subgroup_population())                # 10000
print(conditional.subgroup_population(["g2"]))          # 10000 x 0.4 = 4000
print(conditional.subgroup_population(["g2", "s1"]))    # 4000 x 0.5 = 2000
print(conditional.get_conditioning_categories("subtype"))  # ('g2',)
```

See [Conditional Stratifications](building-models.md#stratifications)
for the full API.

### Per-compartment overrides

When the product of fractions cannot express the starting state, for example a
state that is empty in only one category, `compartment_fractions` overrides
single expanded compartments. Each fraction is a fraction of the **total**
population and replaces the product for that compartment only:

```python
overridden = (
    ModelBuilder(name="Override")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .set_initial_conditions(
        population_size=10000,
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
    .build(typology="DifferenceEquations")
)

print({c.name: c.population for c in overridden.initial_state()})
# B_g1 starts empty; the other compartments keep their product
```

| Rule | Checked when |
|---|---|
| Each name appears once, and each fraction is between 0.0 and 1.0 | The conditions are set |
| Each name is an expanded compartment | The model is built |
| The initial population does not exceed `population_size` | The model is built |

`Model.update_compartment_fractions({"B_g1": 0.05})` changes or adds overrides
on a built model.

### Updating fractions after building

```python
overridden.update_initial_conditions({"A": 0.85, "B": 0.15})
overridden.update_stratification_fractions({"g1": 0.7})  # g2 becomes 0.3
```

| Method | Rule |
|---|---|
| `update_initial_conditions` | After the update, the bin fractions sum to 1.0. Move population between bins in one call. |
| `update_stratification_fractions` | Within each stratification you may omit one category; it receives what remains of 1.0. Omitting two raises. |

A category name that two stratifications declare must be qualified with its
stratification, as in `{"group:g1": 0.5}`. Calibration writes its results back
through these same methods.

## Several sources or targets

A transition with several sources or targets moves **one** flow, like a
reaction: each source loses it and each target gains it. Its rate must be
absolute, because a per-capita rate has no single source to multiply.

```python
reaction = (
    ModelBuilder(name="Reaction")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k", value=0.001)
    .add_transition(id="reaction", source=["A", "B"], target=["C", "C"], rate="k * A * B")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.5},
            {"bin": "B", "fraction": 0.5},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)
# Each step: A loses k*A*B, B loses k*A*B, C gains 2*k*A*B
```

For **independent** flows from several compartments, use the `$compartment`
placeholder: the builder creates one transition per source. An empty `target`
removes the flow from the system.

```python
outflow = (
    ModelBuilder(name="Outflow")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="mu", value=0.01)
    .add_transition(id="outflow", source=["A", "B"], target=[], rate="mu * $compartment")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.5},
            {"bin": "B", "fraction": 0.5},
        ],
    )
    .build(typology="DifferenceEquations")
)
print([t.id for t in outflow.dynamics.transitions])  # ['outflow__A', 'outflow__B']
```

## Steps and physical time

A model knows only steps: a rate is per step and a period is a number of steps.
`TimeScale` gives a step a duration, and converts physical quantities to the
step counts and per-step rates a model needs:

```python
from commol import TimeScale

scale = TimeScale(step_seconds=60 * 60)  # one hour per step

print(scale.steps_per_day)        # 24
print(scale.steps_per_week)       # 168
print(scale.steps_from_days(3))   # 72
```

A period that is not a whole number of steps raises instead of rounding, so a
step size that cannot express a period fails early:

```python
daily = TimeScale(step_seconds=24 * 60 * 60)
try:
    daily.steps_from_hours(1)
except ValueError as error:
    print(error)
```

| Method | Returns |
|---|---|
| `rate_from_mean_duration(seconds)` | `1 - exp(-step / duration)`: the per-step rate for a mean time in a state |
| `rate_from_probability(p, period_steps)` | `1 - (1 - p) ** (1 / period_steps)`: applied over the period, it gives back `p` |
| `window_start(i, period)`, `window_end(i, period)` | The steps that bound window `i` |
| `window_index(step, period)` | The window that contains a step; the inverse of `window_end` |

```python
print(scale.rate_from_mean_duration(4 * 24 * 60 * 60))  # mean of 4 days
print(scale.rate_from_probability(0.3, scale.steps_per_week))

week = scale.steps_per_week
print(scale.window_start(0, week), scale.window_end(0, week))  # 0 168
print(scale.window_index(168, week))                             # 0
```

A window covers the steps `(start, end]`, the same interval that
[windowed observations](calibration.md#calibrating-against-cumulative-outputs)
use.

## Next steps

| To learn | Read |
|---|---|
| Every builder option | [Building Models](building-models.md) |
| What rate formulas can contain | [Mathematical Expressions](mathematical-expressions.md) |
| Rates that change over time | [Time Patterns](time-patterns.md) |
| Running and reading simulations | [Running Simulations](simulations.md) |
