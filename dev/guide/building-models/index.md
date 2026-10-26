# Building Models

[Back to Home](https://munqu.github.io/commol/latest/index.md)

`ModelBuilder` writes the **process** of a model: its bins, parameters, stratifications, accumulators and transitions. `build` joins the process with a **structure**, which lays it out, and an **engine**, which runs it, and returns a validated `Model` ([Build and run](https://munqu.github.io/commol/latest/guide/core-concepts/#build-and-run)). The structure of an aggregate model is `Aggregate`: amounts per compartment. This page documents every builder method except transitions, which have their own page:

| Topic                                                                                              | Page                                                                                         |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Bins, parameters, stratifications, accumulators, the structure (`Aggregate`), build, save and load | This page                                                                                    |
| Transitions and their rates                                                                        | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)              |
| Units                                                                                              | [Units](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md)            |
| The equations and a description of a model                                                         | [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md) |
| What the concepts mean                                                                             | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md)          |

## Quick path

```python
from commol import Aggregate, Model, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Four-State Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_bin(id="D", name="State D")
    .add_parameter(id="k1", value=0.4, description="A to B rate")
    .add_parameter(id="k2", value=0.2, description="B to C rate")
    .add_parameter(id="k3", value=0.1, description="C to D rate")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .add_transition(id="t_cd", source=["C"], target=["D"], rate="k3")
    .build(
        Aggregate(bin_amounts={"A": 900, "B": 100, "C": 0, "D": 0}),
        engine="DifferenceEquations",
    )
)

results = Simulation(model).run(num_steps=200)
```

1. Create a builder.
1. Add bins, parameters and transitions, and optionally stratifications and accumulators: the process.
1. Call `build` with a structure, such as `Aggregate` with the initial amount of each bin, and an engine.

## The builder

```python
builder = ModelBuilder(
    name="My Model",
    version="1.0",
    description="Optional description",
    bin_unit="item",
)
```

| Argument      | Type                       | Default  | Meaning                                                                                                                                                                                                                                 |
| ------------- | -------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | `str`                      | required | The model name.                                                                                                                                                                                                                         |
| `description` | `str \| None`              | `None`   | A free-text description.                                                                                                                                                                                                                |
| `version`     | `str \| None`              | `None`   | A version label for your own tracking.                                                                                                                                                                                                  |
| `bin_unit`    | `str \| None`              | `None`   | The default unit of every bin. It enables [unit checking](https://munqu.github.io/commol/latest/guide/units-and-equations/#unit-checking).                                                                                              |
| `step_unit`   | `str \| None`              | `None`   | The unit of one step of the axis, such as `"day"` or `"meter"`. The unit check reads each flow as `<bin unit>/<step_unit>`. `None` means `step`. See [Units](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md). |
| `origin`      | `date \| datetime \| None` | `None`   | The calendar moment of position 0, with no time zone, for an axis of time. See [Dates on the axis](https://munqu.github.io/commol/latest/guide/units-and-equations/#dates-on-the-axis).                                                 |

Every `add_*` method returns the builder, so calls chain. The other methods:

| Method                                    | Returns                             | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build(structure, engine, settings=None)` | `Model` of the class of `structure` | Joins the process with a [structure](#the-structure-aggregate) and an engine, validates the model and returns it. The engine must run the kind of the structure: `"DifferenceEquations"` (discrete time), `"DifferentialEquations"` (continuous time), `"StochasticEvents"` and `"StochasticSteps"` ([at random](https://munqu.github.io/commol/latest/guide/stochastic/index.md)) run `Aggregate`. `settings` are the settings of that engine (`ODESolver` for `"DifferentialEquations"`, `StochasticEventsSettings` and `StochasticStepsSettings` for the stochastic engines); None uses its defaults. One builder builds the model for each engine and each structure. See [Differential equations](https://munqu.github.io/commol/latest/guide/simulations/#differential-equations). |
| `process()`                               | `Process`                           | The bins, accumulators, stratifications and transitions written so far.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `clone()`                                 | `ModelBuilder`                      | A deep copy of everything, accumulators included, to build variants of one model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `reset()`                                 | `ModelBuilder`                      | Removes bins, accumulators, stratifications, transitions and parameters; keeps the name, description, version, `bin_unit`, `step_unit` and `origin`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `get_summary()`                           | `dict`                              | The name, description, version, and the count and ids of the bins, stratifications, transitions and parameters added so far.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

```python
variant = builder.clone()
print(builder.get_summary()["name"])  # My Model
```

## Bins

```python
builder = (
    builder
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B", unit="item")
)
```

| Argument | Type          | Default  | Meaning                                                       |
| -------- | ------------- | -------- | ------------------------------------------------------------- |
| `id`     | `str`         | required | The identifier used in rates, initial conditions and results. |
| `name`   | `str`         | required | A display name.                                               |
| `unit`   | `str \| None` | `None`   | The unit of this bin; `None` uses the builder's `bin_unit`.   |

A bin id is reserved: no parameter may use it. With stratifications, a bin id in a rate means the total of all its expanded compartments.

## Parameters

A parameter is a named value for rate formulas:

```python
builder = (
    builder
    .add_parameter(id="k1", value=0.3, description="A constant")
    .add_parameter(id="k2", value="k1 / 2", description="A formula")
    .add_parameter(id="k3", value=None, description="Found by calibration")
)
```

| Argument      | Type                                                  | Default  | Meaning                                                                                                        |
| ------------- | ----------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `id`          | `str`                                                 | required | The identifier. It must be unique and must not be a [name the engine sets](#names-the-engine-sets).            |
| `value`       | `float`, `str`, `list[tuple[float, float]]` or `None` | required | See the table below.                                                                                           |
| `description` | `str \| None`                                         | `None`   | A free-text description.                                                                                       |
| `unit`        | `str \| None`                                         | `None`   | The unit, for [unit checking](https://munqu.github.io/commol/latest/guide/units-and-equations/#unit-checking). |
| `mode`        | `str \| None`                                         | `None`   | How a series is read; required for a series and rejected otherwise.                                            |

| `value`                  | Kind         | Behavior                                                                                                                                                                                                         |
| ------------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0.3`                    | Constant     | The same value at every step.                                                                                                                                                                                    |
| `"k1 / 2"`               | Formula      | Evaluated at every step, in declaration order, so it can use earlier formulas, `total`, `step` and the other [special variables](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md). |
| `[(position, value), …]` | Series       | Values at positions on the axis, read with `mode` (below).                                                                                                                                                       |
| `None`                   | Uncalibrated | Calibration must set it before a simulation runs.                                                                                                                                                                |

### Names the engine sets

The engine sets these variables, so `build` rejects a parameter with one of these ids (`Aggregate.check_model` and `check_model` in `crates/commol-model/src/builder.rs`):

| Name                                                                     | Example                                       |
| ------------------------------------------------------------------------ | --------------------------------------------- |
| `total`, `step`, `t`, `pi`, `e`                                          | `total`                                       |
| `neighbors` and `neighbors_<bin>`, the [neighbors of a unit](#neighbors) | `neighbors_A`                                 |
| A bin id                                                                 | `A`                                           |
| An expanded compartment                                                  | `A_g1`                                        |
| A subgroup total                                                         | `total_g1`, `A_t1` (with two stratifications) |

A bin or an accumulator cannot take an id of the first two rows either, because the engine variable would hide it. A bin whose names read as a subgroup total is rejected too: a bin `total_y` stratified by `group = [y, o]` and `type = [y, o]` expands into `total_y_y`, which is also the total of the `y` of both stratifications.

An id that starts with `total_` but names no subgroup total, such as `total_max`, is accepted. `N` is not reserved, so a bin or a parameter may use it.

### Series parameters

A series gives a parameter its value from `(position, value)` points on the axis. `mode` says how the positions between the points are read, and whether the series reads `step` or `t` (`SeriesParameter::evaluate` in `crates/commol-model/src/types.rs`):

| `mode`            | Reads  | Before the first point | At a point                  | Between points              | After the last point        |
| ----------------- | ------ | ---------------------- | --------------------------- | --------------------------- | --------------------------- |
| `"pulse"`         | `step` | 0                      | the value on the whole step | 0                           | 0                           |
| `"step_function"` | `t`    | 0                      | the value                   | the value of the last point | the value of the last point |
| `"linear"`        | `t`    | 0                      | the value                   | linear interpolation in `t` | 0                           |

A series has at least one point, each position finite, not negative and in one point only, and each value finite. `SeriesValue(...)` checks its points with the rule of the engine (`check_series` in `crates/commol-model/src/rules.rs`), which the model compiler also reads.

The engine decides which positions its model type accepts. `ModelBuilder.build` and `Model.update_parameters` reject a position the engine cannot run with `ValueError`, whose message names the parameter (`check_model` in `crates/commol-model/src/builder.rs`). A rejected update leaves the model unchanged.

| Model                                       | Accepted positions                                                                                                                                                                                                                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DifferenceEquations`, `StochasticSteps`    | Integer steps only, for every mode. `t` equals `step`, so `"linear"` is read at each step.                                                                                                                                                                                           |
| `DifferentialEquations`, `StochasticEvents` | Any non-negative position for `"step_function"` and `"linear"`; the engine stops and restarts at each point. A `"pulse"` keeps integer positions: it is a rate on one unit step ([Time inside a step](https://munqu.github.io/commol/latest/guide/simulations/#time-inside-a-step)). |

```python
import numpy as np

def inflow_per_step(mode: str) -> list[float]:
    model = (
        ModelBuilder(name="Series")
        .add_bin(id="A", name="State A")
        .add_parameter(id="k_in", value=[(2, 1.0), (5, 4.0)], mode=mode)
        .add_transition(id="inflow", source=[], target=["A"], rate="k_in")
        .build(
            Aggregate(
                bin_amounts={"A": 100},
            ),
            engine="DifferenceEquations",
        )
    )
    series = Simulation(model).run(num_steps=7)["A"]
    return np.diff(series).round(3).tolist()

print(inflow_per_step("pulse"))          # [0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0]
print(inflow_per_step("step_function"))  # [0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0]
print(inflow_per_step("linear"))         # [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
```

`Model.update_parameters` changes a series of a built model, such as a schedule of a scenario. A list of points keeps the mode of the series; a `SeriesValue` sets the points and the mode. The model takes every value or none: a value it cannot take raises `ValueError` at the call, with the id of the parameter, and the model stays as it was (`tests/building/test_parameters.py::test_update_parameters_rejects_a_value_it_cannot_take`):

```python
from commol import SeriesValue

schedule = (
    ModelBuilder(name="Schedule")
    .add_bin(id="A", name="State A")
    .add_parameter(id="k_in", value=[(2, 1.0)], mode="pulse")
    .add_transition(id="inflow", source=[], target=["A"], rate="k_in")
    .build(Aggregate(bin_amounts={"A": 100}), engine="DifferenceEquations")
)

schedule.update_parameters({"k_in": [(3, 2.0)]})  # still a pulse
print(np.diff(Simulation(schedule).run(num_steps=5)["A"]).tolist())
# [0.0, 0.0, 0.0, 2.0, 0.0]
schedule.update_parameters(
    {"k_in": SeriesValue(data=[(3, 2.0)], mode="step_function")}
)
print(np.diff(Simulation(schedule).run(num_steps=5)["A"]).tolist())
# [0.0, 0.0, 0.0, 2.0, 2.0]
```

`Model.to_json` writes a long series on one line, so model files stay readable.

## Stratifications

```python
builder = builder.add_stratification(
    id="group",
    categories=["g1", "g2"],
    description="Primary grouping",
)
builder = builder.add_stratification(
    id="subtype",
    categories=["s1", "s2"],
    conditions=[{"stratification": "group", "category": "g2"}],
)
```

| Argument      | Type                 | Default  | Meaning                                                                         |
| ------------- | -------------------- | -------- | ------------------------------------------------------------------------------- |
| `id`          | `str`                | required | The identifier, used in initial conditions and conditions.                      |
| `categories`  | `list[str]`          | required | The categories. They must be unique and not empty.                              |
| `description` | `str \| None`        | `None`   | A free-text description.                                                        |
| `conditions`  | `list[dict] \| None` | `None`   | Splits only the compartments whose earlier categories match **all** conditions. |

Each condition is `{"stratification": <id>, "category": <category>}`, and may only name a stratification declared before this one; otherwise the call raises `ValueError`.

| Rule                       | Effect                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| Expansion                  | Every bin and accumulator gets one compartment per category: `A_g1`, `A_g2`.               |
| Several stratifications    | The expansion is the product of their categories, in declaration order: `A_g1_s1`.         |
| Conditional stratification | Only matching compartments are split: with the code above, `A_g1`, `A_g2_s1`, `A_g2_s2`.   |
| Order                      | Changing the order of `add_stratification` calls changes the compartments and their names. |
| Case                       | Names are case-sensitive: `A_g1` is not `A_G1`.                                            |

See [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/#stratifications) for worked examples, and [Transitions](https://munqu.github.io/commol/latest/guide/transitions/#stratified-rates) for rates that depend on the category.

## Accumulators

An [accumulator](https://munqu.github.io/commol/latest/guide/core-concepts/#accumulators) counts what one or more transitions move.

```python
counted = (
    ModelBuilder(name="Counted")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.2)
    .add_parameter(id="k2", value=0.1)
    .add_accumulator(id="cum_ab", name="Total moved from A to B")
    .add_accumulator(id="total_out", name="Total that left A")
    .add_transition(
        id="t_ab", source=["A"], target=["B"], rate="k1", accumulators=["cum_ab", "total_out"]
    )
    .add_transition(id="t_ac", source=["A"], target=["C"], rate="k2", accumulators=["total_out"])
    .build(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0, "C": 0},
        ),
        engine="DifferenceEquations",
    )
)

results = Simulation(counted).run(num_steps=10)
print(round(results["total_out"][-1], 3) == round(1000 - results["A"][-1], 3))  # True
```

| Argument | Type  | Meaning                         |
| -------- | ----- | ------------------------------- |
| `id`     | `str` | The identifier and output name. |
| `name`   | `str` | A display name.                 |

A transition adds its flow to every accumulator in its `accumulators` list, and several transitions can share one accumulator. With stratifications, an accumulator expands like a bin.

## The structure: `Aggregate`

`Aggregate` lays the process out as amounts per compartment: the bins expand over the stratifications into compartments, and each compartment holds an amount. It says how much starts where and who the neighbors of a unit are. Each bin starts at an amount in its own unit, so there is no total over bins of different units.

```python
structure = Aggregate(
    bin_amounts={"A": 9000, "B": 1000},
    stratification_fractions={
        "group": {"g1": 0.3, "g2": 0.7},
        "subtype": {"s1": 0.8, "s2": 0.2},
    },
    compartment_amounts={"B_g1": 0},
)
```

| Argument                   | Type                          | Default  | Meaning                                                                                                                                                                            |
| -------------------------- | ----------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bin_amounts`              | `dict[str, float \| None]`    | required | The amount of each bin at the start, in the unit of the bin: a count, a mass or a concentration. An amount may be `None` when calibration finds it.                                |
| `stratification_fractions` | `dict[str, dict[str, float]]` | `{}`     | For each stratification, the fraction in each category: the split of the amount of every bin it applies to. For a conditional stratification, the split of the subgroup it splits. |
| `compartment_amounts`      | `dict[str, float]`            | `{}`     | Overrides single expanded compartments with an amount in the unit of their bin.                                                                                                    |
| `neighbors`                | `Complete \| Groups \| None`  | `None`   | Who the [neighbors](#neighbors) of a unit are. Required when a rate reads them.                                                                                                    |

| Rule                                                                                                                                                                           | Checked when                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| Every amount is finite and not negative; `0` is an amount. The engine holds the rule (`check_bin_amount` and `check_compartment_amount` in `crates/commol-model/src/rules.rs`) | `Aggregate(...)` and each update of an amount               |
| Every bin has an amount, and no other id                                                                                                                                       | `build`                                                     |
| Every stratification has a fraction for each of its categories, and no other category                                                                                          | `build`                                                     |
| Every fraction is from 0 to 1, and the fractions of a stratification sum to 1 within 1e-6. The engine holds the rule (`check_fractions` in `crates/commol-model/src/rules.rs`) | `Aggregate(...)` and each `update_stratification_fractions` |
| Override names are expanded compartments                                                                                                                                       | `build`                                                     |
| With a stochastic engine, the amount of each bin is a whole number                                                                                                             | `build`                                                     |

The initial amount of a compartment is `bin amount × the fraction of each of its categories`, unless an override names it. `model.structure.initial_state(model.process)` shows the result. To change amounts or fractions on a built model, see [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/#updating-the-initial-state-after-building).

A structure that states `total`, `bin_fractions` or `compartment_fractions` is rejected, and the message names `bin_amounts` and `compartment_amounts`.

### Bins of different units

A bin may hold a count while another holds a concentration. Each bin states its amount in its own unit:

```python
mixed = (
    ModelBuilder(name="Mixed units", step_unit="day")
    .add_bin(id="A", name="State A", unit="item")
    .add_bin(id="B", name="State B", unit="item")
    .add_bin(id="C", name="State C", unit="mg/l")
    .add_parameter(id="k1", value=0.5, unit="1/day")
    .add_parameter(id="k2", value=1e6, unit="mg/l")
    .add_transition(
        id="t_ab", source=["A"], target=["B"], rate="k1 * A * C / (C + k2)"
    )
    .build(
        Aggregate(bin_amounts={"A": 990, "B": 10, "C": 1e5}),
        engine="DifferenceEquations",
    )
)

print({c.name: c.amount for c in mixed.structure.initial_state(mixed.process)})
# {'A': 990.0, 'B': 10.0, 'C': 100000.0}
mixed.check_unit_consistency()
```

A rate that reads `total`, a subgroup total or `neighbors` adds bins of different units here, so the [unit check](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md) rejects it. A model whose bins all start at `0` builds and runs: its state stays at `0` when nothing flows in.

### Neighbors

A rate may read the neighbors of a unit: `neighbors_<bin>` is the amount of a bin among them, and `neighbors` is their total. A rate that reads them and no compartment is **proportional**, a rate per unit of its source, so the same process carries over to structures where units have their own neighbors. Under `Aggregate`, the structure says who they are (`resolve_neighbors` in `crates/commol-model/src/flows.rs`):

| `neighbors=`                           | `neighbors_<bin>` in a compartment of category `g`                                 | `neighbors`                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------ |
| `Complete()`                           | The bin total: every unit is a neighbor of every other                             | `total`                                          |
| `Groups(stratification, weights=None)` | The sum over the categories `g'` of `w[g][g']` times the amount of the bin in `g'` | The same sum, of the amount of every bin in `g'` |

`weights[g][g']` is the weight of category `g'` for a unit in `g`: one number of at least 0 for every pair of categories. No weights is the identity: a unit sees only its own category. A rate that reads the neighbors needs `neighbors`; there is no default, and `build` lists the options.

Here `B` starts only in `g1` (the override pins `B_g2` to 0), and the flow into `B_g2` depends on whom the units of `g2` see:

```python
from commol import Complete, Groups

def into_b_g2(neighbors: Complete | Groups) -> list[float]:
    model = (
        ModelBuilder(name="Neighbors")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_stratification(id="group", categories=["g1", "g2"])
        .add_parameter(id="k1", value=0.3)
        .add_transition(
            id="t_ab", source=["A"], target=["B"], rate="k1 * neighbors_B / neighbors"
        )
        .build(
            Aggregate(
                bin_amounts={"A": 990, "B": 10},
                stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
                compartment_amounts={"B_g2": 0},
                neighbors=neighbors,
            ),
            engine="DifferenceEquations",
        )
    )
    return [round(value, 3) for value in Simulation(model).run(num_steps=3)["B_g2"]]

half = {"g1": {"g1": 1.0, "g2": 0.5}, "g2": {"g1": 0.5, "g2": 1.0}}
print(into_b_g2(Complete()))                       # [0.0, 0.746, 1.714, 2.967]
print(into_b_g2(Groups("group")))                  # [0.0, 0.0, 0.0, 0.0]
print(into_b_g2(Groups("group", weights=half)))    # [0.0, 0.498, 1.194, 2.151]
```

Under `Complete()`, `k1 * neighbors_B / neighbors` from `A` moves `k1 × A × B / total`, the flow of the absolute rate `k1 * A * B / total`. Under `Groups("group")`, a unit of `g2` sees only `g2`, where there is no `B`.

## Build

```python
builder = (
    builder
    .add_parameter(id="k_out", value=0.05)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k_out")
)
stratified_model = builder.build(structure, engine="DifferenceEquations")

print([c.name for c in stratified_model.structure.initial_state(stratified_model.process)])
# ['A_g1', 'A_g2_s1', 'A_g2_s2', 'B_g1', 'B_g2_s1', 'B_g2_s2']
```

`build` raises `ValueError` (or Pydantic's `ValidationError`, a subclass) when the definition is not valid:

| Check                                                                                                                                                                     | Example of a rejected definition                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| The engine runs the kind of the structure                                                                                                                                 | An `Aggregate` with an engine of another kind; the message lists the engines of `Aggregate`      |
| At least one bin and one transition                                                                                                                                       | A model with no transition                                                                       |
| The structure follows the rules above                                                                                                                                     | A bin with no amount in `bin_amounts`                                                            |
| Parameter ids are unique and are not [names the engine sets](#names-the-engine-sets)                                                                                      | A parameter named `A`, `total` or `A_g1`                                                         |
| Every `total_…` variable names a subgroup total                                                                                                                           | `total_t1_g1` when `group` is declared before `type`                                             |
| A rate that reads the neighbors has them                                                                                                                                  | `k1 * neighbors_B` with no `neighbors=`                                                          |
| Every rate expression is in the language of formulas ([Limits of the parser](https://munqu.github.io/commol/latest/guide/mathematical-expressions/#limits-of-the-parser)) | A formula with `__import__('os')`                                                                |
| Every transition can be planned by the engine                                                                                                                             | See [Transitions](https://munqu.github.io/commol/latest/guide/transitions/#what-building-checks) |

## Save and load

```python
stratified_model.to_json("model.json")
reloaded = Model.from_json("model.json")
print(reloaded == stratified_model)  # True
```

| Method                               | What it does                                                                                                                                                                                          |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Model.to_json(file_path, indent=2)` | Writes every field, including nulls, so an uncalibrated model reloads too: the header, the `process` and the `structure`, with its `kind` and `variant`. Long numeric arrays are written on one line. |
| `Model.from_json(file_path)`         | Reads and validates a model file.                                                                                                                                                                     |

## Next steps

| To learn                                   | Read                                                                                                      |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Transitions and rates                      | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)                           |
| Units                                      | [Units](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md)                         |
| The equations and a description of a model | [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md)              |
| What rate formulas can contain             | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) |
| Running a model                            | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md)                   |
