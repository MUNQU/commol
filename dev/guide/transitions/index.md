# Transitions

[Back to Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md)

A transition moves population at every step: from its `source` compartments to its `target` compartments, at the rate you give. This page documents `ModelBuilder.add_transition` completely: rates, several sources, transitions between categories, stratified rates, impulses, and what `build` checks.

## Quick path

```python
from commol import ModelBuilder, Simulation

model = (
    ModelBuilder(name="Transitions")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
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

model.print_equations()  # shows every flow as the engine computes it
```

## `add_transition`

| Argument           | Type                                    | Default  | Meaning                                                                                                                                             |
| ------------------ | --------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | `str`                                   | required | The identifier.                                                                                                                                     |
| `source`           | `list[str]`                             | required | Bins (or one category) that lose the flow. `[]` means the flow enters the system.                                                                   |
| `target`           | `list[str]`                             | required | Bins (or one category) that gain the flow. `[]` means the flow leaves the system.                                                                   |
| `rate`             | `str`, `float`, `TimePattern` or `None` | `None`   | The rate formula; see [Rates](#rates). With `stratified_rates`, the fallback rate. `None` with no stratified rate gives a transition with no flows. |
| `stratified_rates` | `list[dict] \| None`                    | `None`   | Rates for some categories; see [Stratified rates](#stratified-rates).                                                                               |
| `per_compartment`  | `bool \| None`                          | `None`   | Replaces bin names in the rate with each flow's own compartment; see [per_compartment](#per_compartment).                                           |
| `accumulators`     | `list[str] \| None`                     | `None`   | Accumulators that count the flow; see [Building Models](https://munqu.github.io/commol/latest/guide/building-models/#accumulators).                 |
| `impulse_at`       | `list[float] \| None`                   | `None`   | Makes the transition an impulse at these steps; see [Impulses](#impulses).                                                                          |
| `impulse_every`    | `tuple[float, float] \| None`           | `None`   | `(period, offset)`: an impulse at `offset`, `offset + period`, …; see [Impulses](#impulses).                                                        |

## Rates

A rate is a formula (a string), a number, or a [`TimePattern`](https://munqu.github.io/commol/latest/guide/time-patterns/index.md). A formula can use parameters, compartments, bin totals, `N`, `step`, constants and functions; see [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md).

### Absolute and per-capita rates

The engine decides for each flow whether its rate is the flow itself or a rate per member of the source:

| Case                                                          | Kind       | Flow per step   |
| ------------------------------------------------------------- | ---------- | --------------- |
| The rate names a compartment, a bin total or an `N…` variable | Absolute   | `rate`          |
| The transition has no source                                  | Absolute   | `rate`          |
| A stratified rate sets `"absolute": True` or `False`          | As set     | —               |
| Otherwise                                                     | Per-capita | `rate × source` |

| Rate                     | Kind       | Flow from `B` to `C`    |
| ------------------------ | ---------- | ----------------------- |
| `"k2"`                   | Per-capita | `k2 * B`                |
| `"k2 * B"`               | Absolute   | `k2 * B`                |
| `"k1 * A * B / N"`       | Absolute   | `k1 * A * B / N`        |
| `"0.5"` with `source=[]` | Absolute   | `0.5` enters the target |

`print_equations()` prints every flow after this decision, so a per-capita rate appears multiplied by its source.

### Time-dependent rates

A formula may use `step` (or `t`), the current step:

```python
seasonal = (
    ModelBuilder(name="Seasonal")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.3)
    .add_transition(
        id="t_ab",
        source=["A"],
        target=["B"],
        rate="k1 * (1 + 0.3 * sin(2 * pi * step / 365)) * A * B / N",
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 0.99}, {"bin": "B", "fraction": 0.01}],
    )
    .build(typology="DifferenceEquations")
)
```

For pulses, windows, ramps and schedules per group, use [Time Patterns](https://munqu.github.io/commol/latest/guide/time-patterns/index.md).

## Several sources or targets

A transition with several sources or targets moves **one** flow, like a reaction: every entry in `source` loses it and every entry in `target` gains it. A bin listed twice loses or gains it twice. The rate must be absolute.

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
reaction.print_equations()
# A loses k * A * B, B loses k * A * B, C gains 2 * (k * A * B)
```

### The `$compartment` placeholder

For **independent** flows with one formula, write `$compartment` in the rate. The builder creates one transition per source (or per target), named `{id}__{compartment}`, with `$compartment` replaced by that compartment:

```python
outflow = (
    ModelBuilder(name="Outflow")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="mu", value=0.01)
    .add_transition(
        id="outflow",
        source=["A", "B", "C"],
        target=[],
        rate="mu * $compartment * (1 + 0.1 * $compartment / N)",
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.4},
            {"bin": "B", "fraction": 0.3},
            {"bin": "C", "fraction": 0.3},
        ],
    )
    .build(typology="DifferenceEquations")
)

for transition in outflow.dynamics.transitions:
    print(transition.id, transition.rate)
# outflow__A mu * A * (1 + 0.1 * A / N)
# outflow__B mu * B * (1 + 0.1 * B / N)
# outflow__C mu * C * (1 + 0.1 * C / N)
```

| Rule                                                                            | Otherwise                                        |
| ------------------------------------------------------------------------------- | ------------------------------------------------ |
| Several sources with one or no target, or several targets with one or no source | `ValueError`: the mapping would be ambiguous     |
| At least two sources or two targets                                             | `ValueError`: write the compartment name instead |
| A `rate` is given                                                               | `ValueError`                                     |

`stratified_rates` may also use `$compartment`; each generated transition gets its own copy.

## Transitions between categories

When `source` and `target` are categories of one stratification, the transition moves population between them **in every bin**:

```python
moving = (
    ModelBuilder(name="Moving")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k", value=0.1)
    .add_transition(id="move", source=["g1"], target=["g2"], rate="k")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 0.5}, {"bin": "B", "fraction": 0.5}],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.5},
                    {"category": "g2", "fraction": 0.5},
                ],
            }
        ],
    )
    .build(typology="DifferenceEquations")
)
moving.print_equations()
# move (g1 -> g2):  A_g1 -> A_g2: k * A_g1   and   B_g1 -> B_g2: k * B_g1
```

A category transition needs exactly one source and one target category. If the target category is split by a conditional stratification, every flow must say which of its categories it enters, with a `to` condition (see [Redirecting the target with `to`](#redirecting-the-target-with-to)); otherwise `build` raises.

## `per_compartment`

With stratifications, a bin id in a rate means the **total** of the bin. In a flow from `A_g1`, `"k * A"` uses `A_g1 + A_g2`. `per_compartment=True` replaces each source and target bin id with the flow's own compartment:

| Flow           | `rate="k * A"`      | with `per_compartment=True` |
| -------------- | ------------------- | --------------------------- |
| `A_g1 -> B_g1` | `k * A` (the total) | `k * A_g1`                  |
| `A_g2 -> B_g2` | `k * A` (the total) | `k * A_g2`                  |

```python
own_compartment = (
    ModelBuilder(name="Per compartment")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k", value=0.1)
    .add_transition(id="t", source=["A"], target=["B"], rate="k * A", per_compartment=True)
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 0.5}, {"bin": "B", "fraction": 0.5}],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.5},
                    {"category": "g2", "fraction": 0.5},
                ],
            }
        ],
    )
    .build(typology="DifferenceEquations")
)
own_compartment.print_equations()
# A_g1 -> B_g1: k * A_g1   and   A_g2 -> B_g2: k * A_g2
```

It applies to the fallback rate and to every stratified rate, and only to transitions between bins.

## Stratified rates

`stratified_rates` gives some categories their own rate. Each entry has `conditions` and a `rate`, and may set `absolute`:

```python
group_rates = (
    ModelBuilder(name="Group rates")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2", "g3"])
    .add_parameter(id="k_g1", value=0.15)
    .add_parameter(id="k_g2", value=0.10)
    .add_parameter(id="k_default", value=0.05)
    .add_transition(
        id="t_ab",
        source=["A"],
        target=["B"],
        rate="k_default",
        stratified_rates=[
            {"conditions": [{"stratification": "group", "category": "g1"}], "rate": "k_g1"},
            {"conditions": [{"stratification": "group", "category": "g2"}], "rate": "k_g2"},
        ],
    )
    .set_initial_conditions(
        population_size=900,
        bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.3},
                    {"category": "g2", "fraction": 0.3},
                    {"category": "g3", "fraction": 0.4},
                ],
            }
        ],
    )
    .build(typology="DifferenceEquations")
)
group_rates.print_equations()
# A_g1 -> B_g1: k_g1 * A_g1,  A_g2 -> B_g2: k_g2 * A_g2,  A_g3 -> B_g3: k_default * A_g3
```

| Entry key    | Type             | Meaning                                       |
| ------------ | ---------------- | --------------------------------------------- |
| `conditions` | `list[dict]`     | All must match the source compartment (AND).  |
| `rate`       | `str` or `float` | The rate for matching compartments.           |
| `absolute`   | `bool`, optional | Forces the rate to be absolute or per-capita. |

| Condition key    | Meaning                                                        |
| ---------------- | -------------------------------------------------------------- |
| `stratification` | The stratification id.                                         |
| `category`       | Optional. The category the **source** must have.               |
| `to`             | Optional. The category the **target** gets instead; see below. |

**Which rate applies:** for each source compartment, the entry that matches with the **most** conditions wins; on a tie, the first one in the list. With no match, the fallback `rate` applies. With three stratifications and one entry for `g2`, `t1`, `v2`:

| Source compartment | Conditions matched | Rate      |
| ------------------ | ------------------ | --------- |
| `A_g2_t1_v2`       | 3 of 3             | The entry |
| `A_g2_t1_v1`       | not all            | Fallback  |
| `A_g1_t1_v1`       | not all            | Fallback  |

### Redirecting the target with `to`

A condition with `category` filters the **source**. A condition with `to` changes the **target** category. A condition with only `to` filters nothing, so it can send the flow into a stratification that the source does not have:

```python
redirect = (
    ModelBuilder(name="Redirect")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_stratification(
        id="subtype",
        categories=["s1", "s2"],
        conditions=[{"stratification": "group", "category": "g2"}],
    )
    .add_parameter(id="k", value=0.1)
    .add_transition(
        id="move",
        source=["g1"],
        target=["g2"],
        stratified_rates=[{"conditions": [{"stratification": "subtype", "to": "s1"}], "rate": "k"}],
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 0.5}, {"bin": "B", "fraction": 0.5}],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.5},
                    {"category": "g2", "fraction": 0.5},
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
redirect.print_equations()
# A_g1 -> A_g2_s1: k * A_g1   and   B_g1 -> B_g2_s1: k * B_g1
```

## Limiting when a transition runs

A transition runs at every step. To limit a flow, write the condition in the rate with `if`, as in `"if(step > 30, k, 0)"`, or use a [`TimePattern`](https://munqu.github.io/commol/latest/guide/time-patterns/index.md), a [time-series parameter](https://munqu.github.io/commol/latest/guide/building-models/#time-series-parameters) or a [stratified rate](#stratified-rates). A comparison alone is a boolean, so `"k * (step > 30)"` is rejected.

## Impulses

An impulse moves its amount at once, at given steps, and nothing between them. Give the steps with `impulse_at=[…]` or `impulse_every=(period, offset)`. The rate is then a fraction of the source (per-capita) or an amount (absolute), with the same [rules](#absolute-and-per-capita-rates), sources, targets, stratified rates and accumulators as any transition.

```python
impulse_model = (
    ModelBuilder(name="Impulses")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_accumulator(id="moved", name="Moved by the impulse")
    .add_parameter(id="p", value=0.25)
    .add_parameter(id="k", value=0.1)
    .add_transition(id="jump", source=["A"], target=["B"], rate="p", impulse_at=[1], accumulators=["moved"])
    .add_transition(id="flow", source=["A"], target=["B"], rate="k")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
    )
    .build(typology="DifferenceEquations")
)

jumped = Simulation(impulse_model).run(num_steps=2)
print([round(a, 1) for a in jumped["A"]])  # [1000.0, 675.0, 607.5]
print(jumped["moved"][2])  # 225.0: a quarter of the 900 left at step 1
```

| Rule                         | In both engines                                                                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Order                        | At step `n` the state jumps first; then the flows of the step run from the new state. The output at `n` shows the state after the jump (`CompiledModel::apply_impulses` in `crates/commol-model/src/system.rs`).   |
| Several impulses at one step | Every jump is computed from the state before any of them.                                                                                                                                                          |
| Steps                        | A `DifferentialEquations` model takes any steps, such as `0.5`: the engine stops there, applies the jump and restarts. A `DifferenceEquations` model needs integer steps (`ValidationError` names the transition). |
| Units                        | The rate of an impulse is an amount, `<bin unit>`, not a flow per step ([Units and Equations](https://munqu.github.io/commol/latest/guide/units-and-equations/#impulses)).                                         |
| `print_equations`            | Lists impulses apart from the equations, with their steps.                                                                                                                                                         |

An impulse is not a pulse rate. A [`TimePattern.pulse`](https://munqu.github.io/commol/latest/guide/time-patterns/index.md) is a rate: a difference model applies it for one step with the other flows, and a differential model applies it as a constant rate for one time unit, so it moves `1 − e^(−p)` of its source. An impulse moves exactly `p` of its source in both engines.

## What building checks

`build` plans every flow in the engine and rejects a transition it cannot simulate as written:

| Rejected                                                                                     | Message names                                            |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| A source or target that is neither a bin nor a category of one stratification                | The transition                                           |
| A category transition without exactly one source and one target                              | The transition                                           |
| A rate variable that is not a parameter, bin, compartment or special variable                | The variable and the available names                     |
| Several sources with a per-capita rate                                                       | The transition; use an absolute rate or `$compartment`   |
| A stratified rate whose `category` conditions match no source compartment                    | The categories; use `to` to reach another stratification |
| A flow that ends in categories no compartment has                                            | The transition; add a `to` condition                     |
| `impulse_at` and `impulse_every` together, a negative step, or a period that is not positive | The transition                                           |
| A non-integer impulse step in a `DifferenceEquations` model                                  | The transition and the step                              |

## Next steps

| To learn                         | Read                                                                                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Rates that change over time      | [Time Patterns](https://munqu.github.io/commol/latest/guide/time-patterns/index.md)                       |
| Everything a formula can contain | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) |
| Units and equation output        | [Units and Equations](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md)           |
