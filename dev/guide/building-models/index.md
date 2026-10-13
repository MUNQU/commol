# Building Models

[Back to Home](https://munqu.github.io/commol/latest/index.md)

`ModelBuilder` turns a model description into a validated `Model`. This page documents every builder method except transitions, which have their own page:

| Topic                                                                                     | Page                                                                                            |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Bins, parameters, stratifications, accumulators, initial conditions, build, save and load | This page                                                                                       |
| Transitions and their rates                                                               | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)                 |
| Units and printing equations                                                              | [Units and Equations](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md) |
| What the concepts mean                                                                    | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md)             |

## Quick path

```python
from commol import Model, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Four-State Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_bin(id="D", name="State D")
    .add_parameter(id="k1", value=0.4, description="A to B rate")
    .add_parameter(id="k2", value=0.2, description="B to C rate")
    .add_parameter(id="k3", value=0.1, description="C to D rate")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .add_transition(id="t_cd", source=["C"], target=["D"], rate="k3")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.9},
            {"bin": "B", "fraction": 0.1},
            {"bin": "C", "fraction": 0.0},
            {"bin": "D", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)

results = Simulation(model).run(num_steps=200)
```

1. Create a builder.
1. Add bins, parameters and transitions, and optionally stratifications and accumulators.
1. Set the initial conditions.
1. Call `build`.

## The builder

```python
builder = ModelBuilder(
    name="My Model",
    version="1.0",
    description="Optional description",
    bin_unit="item",
)
```

| Argument      | Type          | Default  | Meaning                                                                                                                                    |
| ------------- | ------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`        | `str`         | required | The model name.                                                                                                                            |
| `description` | `str \| None` | `None`   | A free-text description.                                                                                                                   |
| `version`     | `str \| None` | `None`   | A version label for your own tracking.                                                                                                     |
| `bin_unit`    | `str \| None` | `None`   | The default unit of every bin. It enables [unit checking](https://munqu.github.io/commol/latest/guide/units-and-equations/#unit-checking). |

Every `add_*` and `set_*` method returns the builder, so calls chain. The other methods:

| Method            | Returns        | What it does                                                                                                                                      |
| ----------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build(typology)` | `Model`        | Validates the definition and returns the model. `typology` must be `"DifferenceEquations"`.                                                       |
| `clone()`         | `ModelBuilder` | A deep copy of everything, accumulators included, to build variants of one model.                                                                 |
| `reset()`         | `ModelBuilder` | Removes bins, accumulators, stratifications, transitions, parameters and initial conditions; keeps the name, description, version and `bin_unit`. |
| `get_summary()`   | `dict`         | The name, description, version and the ids added so far.                                                                                          |

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

| Argument      | Type                                                | Default  | Meaning                                                                                                        |
| ------------- | --------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `id`          | `str`                                               | required | The identifier. It must be unique and must not be a [name the engine sets](#names-the-engine-sets).            |
| `value`       | `float`, `str`, `list[tuple[int, float]]` or `None` | required | See the table below.                                                                                           |
| `description` | `str \| None`                                       | `None`   | A free-text description.                                                                                       |
| `unit`        | `str \| None`                                       | `None`   | The unit, for [unit checking](https://munqu.github.io/commol/latest/guide/units-and-equations/#unit-checking). |
| `mode`        | `str \| None`                                       | `None`   | How a time series is read; required for a time series and rejected otherwise.                                  |

| `value`              | Kind         | Behavior                                                                                                                                                                                                     |
| -------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `0.3`                | Constant     | The same value at every step.                                                                                                                                                                                |
| `"k1 / 2"`           | Formula      | Evaluated at every step, in declaration order, so it can use earlier formulas, `N`, `step` and the other [special variables](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md). |
| `[(step, value), …]` | Time series  | A value per step, read with `mode` (below).                                                                                                                                                                  |
| `None`               | Uncalibrated | Calibration must set it before a simulation runs.                                                                                                                                                            |

### Names the engine sets

The engine sets these variables, so `build` rejects a parameter with one of these ids (`Model.validate_parameter_names_not_reserved` and `check_names` in `crates/commol-model/src/builder.rs`):

| Name                        | Example                                   |
| --------------------------- | ----------------------------------------- |
| `N`, `step`, `t`, `pi`, `e` | `N`                                       |
| A bin id                    | `A`                                       |
| An expanded compartment     | `A_g1`                                    |
| A subpopulation total       | `N_g1`, `A_t1` (with two stratifications) |

An id that starts with `N_` but names no subpopulation, such as `N_max`, is accepted.

### Time-series parameters

A time series gives a parameter its value from `(step, value)` pairs. `mode` says how the steps between the pairs are read:

| `mode`            | Before the first pair | At a listed step | Between listed steps  | After the last pair   |
| ----------------- | --------------------- | ---------------- | --------------------- | --------------------- |
| `"pulse"`         | 0                     | the listed value | 0                     | 0                     |
| `"step_function"` | 0                     | the listed value | the last listed value | the last listed value |
| `"linear"`        | 0                     | the listed value | linear interpolation  | 0                     |

```python
import numpy as np

def inflow_per_step(mode: str) -> list[float]:
    model = (
        ModelBuilder(name="Time series")
        .add_bin(id="A", name="State A")
        .add_parameter(id="k_in", value=[(2, 1.0), (5, 4.0)], mode=mode)
        .add_transition(id="inflow", source=[], target=["A"], rate="k_in")
        .set_initial_conditions(
            population_size=100, bin_fractions=[{"bin": "A", "fraction": 1.0}]
        )
        .build(typology="DifferenceEquations")
    )
    series = Simulation(model).run(num_steps=7)["A"]
    return np.diff(series).round(3).tolist()

print(inflow_per_step("pulse"))          # [0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0]
print(inflow_per_step("step_function"))  # [0.0, 0.0, 1.0, 1.0, 1.0, 4.0, 4.0]
print(inflow_per_step("linear"))         # [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
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

An accumulator counts what one or more transitions move. It is an output, not part of the population.

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
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 1.0},
            {"bin": "B", "fraction": 0.0},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)

results = Simulation(counted).run(num_steps=10)
print(round(results["total_out"][-1], 3) == round(1000 - results["A"][-1], 3))  # True
```

| Argument | Type  | Meaning                         |
| -------- | ----- | ------------------------------- |
| `id`     | `str` | The identifier and output name. |
| `name`   | `str` | A display name.                 |

A transition adds its flow to every accumulator in its `accumulators` list, and several transitions can share one accumulator. With stratifications, an accumulator expands like a bin.

## Initial conditions

```python
builder = builder.set_initial_conditions(
    population_size=10000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
    ],
    stratification_fractions=[
        {
            "stratification": "group",
            "fractions": [
                {"category": "g1", "fraction": 0.3},
                {"category": "g2", "fraction": 0.7},
            ],
        },
        {
            "stratification": "subtype",
            "fractions": [
                {"category": "s1", "fraction": 0.8},
                {"category": "s2", "fraction": 0.2},
            ],
        },
    ],
    compartment_fractions=[{"compartment": "B_g1", "fraction": 0.0}],
)
```

| Argument                   | Type                                            | Default  | Meaning                                                                                                                        |
| -------------------------- | ----------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `population_size`          | `int`                                           | required | The total population. It must be greater than 0.                                                                               |
| `bin_fractions`            | `list[{"bin", "fraction"}]`                     | required | The fraction of the population in each bin. A fraction may be `None` when calibration finds it.                                |
| `stratification_fractions` | `list[{"stratification", "fractions"}] \| None` | `None`   | For each stratification, the fraction in each category. For a conditional stratification, fractions of the subgroup it splits. |
| `compartment_fractions`    | `list[{"compartment", "fraction"}] \| None`     | `None`   | Keyword-only. Overrides single expanded compartments with a fraction of the total population.                                  |

| Rule                                                                                               | Checked when             |
| -------------------------------------------------------------------------------------------------- | ------------------------ |
| `population_size > 0`; each fraction in `[0.0, 1.0]`; override names unique                        | `set_initial_conditions` |
| The bin fractions sum to 1.0; with `None` fractions, the given ones sum to less than 1.0           | `set_initial_conditions` |
| Every bin has a fraction                                                                           | `build`                  |
| Every stratification has fractions that sum to 1.0                                                 | `build`                  |
| Override names are expanded compartments; the initial population does not exceed `population_size` | `build`                  |

The initial population of a compartment is `population_size × bin fraction × the fraction of each of its categories`, unless an override names it. `Model.initial_state()` shows the result. To change fractions on a built model, see [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/#updating-fractions-after-building).

## Build

```python
builder = (
    builder
    .add_parameter(id="k_out", value=0.05)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k_out")
)
stratified_model = builder.build(typology="DifferenceEquations")

print([c.name for c in stratified_model.initial_state()])
# ['A_g1', 'A_g2_s1', 'A_g2_s2', 'B_g1', 'B_g2_s1', 'B_g2_s2']
```

`build` raises `ValueError` (or Pydantic's `ValidationError`, a subclass) when the definition is not valid:

| Check                                                                                | Example of a rejected definition                                                                 |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Initial conditions are set                                                           | `build` called before `set_initial_conditions`                                                   |
| At least one bin and one transition                                                  | A model with no transition                                                                       |
| Initial conditions follow the rules above                                            | Bin fractions that sum to 0.9                                                                    |
| Parameter ids are unique and are not [names the engine sets](#names-the-engine-sets) | A parameter named `A`, `N` or `A_g1`                                                             |
| Every `N_…` variable names a subpopulation                                           | `N_t1_g1` when `group` is declared before `type`                                                 |
| Every rate expression passes the security check                                      | A formula with `__import__`                                                                      |
| Every transition can be planned by the engine                                        | See [Transitions](https://munqu.github.io/commol/latest/guide/transitions/#what-building-checks) |

## Save and load

```python
stratified_model.to_json("model.json")
reloaded = Model.from_json("model.json")
print(reloaded == stratified_model)  # True
```

| Method                               | What it does                                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `Model.to_json(file_path, indent=2)` | Writes every field, including nulls, so an uncalibrated model reloads too. Long numeric arrays are written on one line. |
| `Model.from_json(file_path)`         | Reads and validates a model file.                                                                                       |

## Next steps

| To learn                       | Read                                                                                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Transitions and rates          | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md)                           |
| Units and equations            | [Units and Equations](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md)           |
| What rate formulas can contain | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) |
| Running a model                | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md)                   |
