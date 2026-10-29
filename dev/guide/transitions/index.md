# Transitions

[Back to Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md)

A transition moves an amount at every step: from its `source` compartments to its `target` compartments, at the rate you give. This page documents `ModelBuilder.add_transition` completely: rates, several sources, transitions between categories, stratified rates, impulses, amounts below zero, and what `build` checks.

## Quick path

```python
from commol import (
    Aggregate,
    ModelBuilder,
    Simulation,
    StratificationCondition,
    StratifiedRate,
)

model = (
    ModelBuilder(name="Transitions")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .build(
        Aggregate(
            bin_amounts={"A": 900, "B": 100, "C": 0},
        ),
        engine="DifferenceEquations",
    )
)

print(model.describe().render("text"))  # every flow, as the engine computes it
```

## `add_transition`

| Argument           | Type                                    | Default  | Meaning                                                                                                                                             |
| ------------------ | --------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`               | `str`                                   | required | The identifier.                                                                                                                                     |
| `source`           | `list[str]`                             | required | Bins (or one category) that lose the flow. `[]` means the flow enters the system.                                                                   |
| `target`           | `list[str]`                             | required | Bins (or one category) that gain the flow. `[]` means the flow leaves the system.                                                                   |
| `rate`             | `str`, `float`, `AxisPattern` or `None` | `None`   | The rate formula; see [Rates](#rates). With `stratified_rates`, the fallback rate. `None` with no stratified rate gives a transition with no flows. |
| `stratified_rates` | `list[StratifiedRate] \| None`          | `None`   | Rates for some categories; see [Stratified rates](#stratified-rates).                                                                               |
| `per_compartment`  | `bool \| None`                          | `None`   | Replaces bin names in the rate with each flow's own compartment; see [per_compartment](#per_compartment).                                           |
| `accumulators`     | `list[str] \| None`                     | `None`   | Accumulators that count the flow; see [Building Models](https://munqu.github.io/commol/latest/guide/building-models/#accumulators).                 |
| `impulse_at`       | `list[float] \| None`                   | `None`   | Makes the transition an impulse at these steps; see [Impulses](#impulses).                                                                          |
| `impulse_every`    | `tuple[float, float] \| None`           | `None`   | `(period, offset)`: an impulse at `offset`, `offset + period`, …; see [Impulses](#impulses).                                                        |

## Rates

A rate is a formula (a string), a number, or a [`AxisPattern`](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md). A formula can use parameters, compartments, bin totals, `total`, `step`, constants and functions; see [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md).

### Absolute and proportional rates

The engine decides for each flow whether its rate is the flow itself or a rate per member of the source:

| Case                                                                                           | Kind         | Flow            |
| ---------------------------------------------------------------------------------------------- | ------------ | --------------- |
| The rate names, outside every `matrix_sum`, a compartment, a bin total or a `total_…` variable | Absolute     | `rate`          |
| No source                                                                                      | Absolute     | `rate`          |
| A stratified rate sets `"absolute": True` or `False`                                           | As set       | —               |
| Otherwise                                                                                      | Proportional | `rate × source` |

A rate that reads `total` and no compartment, such as `k1 / total`, is proportional (`plan_transition_flows` in `crates/commol-model/src/flows.rs`).

| Rate                     | Kind         | Flow from `B` to `C`    |
| ------------------------ | ------------ | ----------------------- |
| `"k2"`                   | Proportional | `k2 * B`                |
| `"k2 * B"`               | Absolute     | `k2 * B`                |
| `"k1 * A * B / total"`   | Absolute     | `k1 * A * B / total`    |
| `"0.5"` with `source=[]` | Absolute     | `0.5` enters the target |

The [description](https://munqu.github.io/commol/latest/guide/describing-models/index.md) of the model shows every flow after this decision, so a proportional rate appears multiplied by its source.

### Rates that change along the axis

A formula may use `step` (or `t`), the current step:

```python
periodic = (
    ModelBuilder(name="Periodic")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k1", value=0.3)
    .add_transition(
        id="t_ab",
        source=["A"],
        target=["B"],
        rate="k1 * (1 + 0.3 * sin(2 * pi * step / 365)) * A * B / total",
    )
    .build(
        Aggregate(
            bin_amounts={"A": 990, "B": 10},
        ),
        engine="DifferenceEquations",
    )
)
```

For pulses, windows, ramps and schedules per group, use [Axis Patterns](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md).

## Several sources or targets

A transition with several sources or targets moves **one** flow: every entry in `source` loses it and every entry in `target` gains it. A bin listed twice loses or gains it twice. The rate must be absolute.

```python
merge = (
    ModelBuilder(name="Merge")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k", value=0.001)
    .add_transition(id="t1", source=["A", "B"], target=["C", "C"], rate="k * A * B")
    .build(
        Aggregate(
            bin_amounts={"A": 500, "B": 500, "C": 0},
        ),
        engine="DifferenceEquations",
    )
)
print(merge.describe().render("text"))
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
        rate="mu * $compartment * (1 + 0.1 * $compartment / total)",
    )
    .build(
        Aggregate(
            bin_amounts={"A": 400, "B": 300, "C": 300},
        ),
        engine="DifferenceEquations",
    )
)

for transition in outflow.process.transitions:
    print(transition.id, transition.rate)
# outflow__A mu * A * (1 + 0.1 * A / total)
# outflow__B mu * B * (1 + 0.1 * B / total)
# outflow__C mu * C * (1 + 0.1 * C / total)
```

| Rule                                                                            | Otherwise                                        |
| ------------------------------------------------------------------------------- | ------------------------------------------------ |
| Several sources with one or no target, or several targets with one or no source | `ValueError`: the mapping would be ambiguous     |
| At least two sources or two targets                                             | `ValueError`: write the compartment name instead |
| A `rate` is given                                                               | `ValueError`                                     |

`stratified_rates` may also use `$compartment`; each generated transition gets its own copy. Each generated transition keeps `per_compartment`, `accumulators` and the impulse schedule of the call. With `per_compartment=True`, `$compartment` reads the compartment that flows: in a flow from `A_g1`, `"k * $compartment"` is `k * A_g1`.

## Transitions between categories

When `source` and `target` are categories of one stratification, the transition moves amounts between them **in every bin**:

```python
moving = (
    ModelBuilder(name="Moving")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k", value=0.1)
    .add_transition(id="move", source=["g1"], target=["g2"], rate="k")
    .build(
        Aggregate(
            bin_amounts={"A": 500, "B": 500},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        ),
        engine="DifferenceEquations",
    )
)
print(moving.describe().render("text"))
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
    .build(
        Aggregate(
            bin_amounts={"A": 500, "B": 500},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        ),
        engine="DifferenceEquations",
    )
)
print(own_compartment.describe().render("text"))
# A_g1 -> B_g1: k * A_g1   and   A_g2 -> B_g2: k * A_g2
```

It applies to the fallback rate and to every stratified rate, and only to transitions between bins. Any other bin in the rate keeps its total: in the flow `A_g1 -> B_g1`, `"k * A * C / total"` reads `A_g1` and the total of `C`.

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
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g1")
                ],
                rate="k_g1",
            ),
            StratifiedRate(
                conditions=[
                    StratificationCondition(stratification="group", category="g2")
                ],
                rate="k_g2",
            ),
        ],
    )
    .build(
        Aggregate(
            bin_amounts={"A": 900, "B": 0},
            stratification_fractions={"group": {"g1": 0.3, "g2": 0.3, "g3": 0.4}},
        ),
        engine="DifferenceEquations",
    )
)
print(group_rates.describe().render("text"))
# A_g1 -> B_g1: k_g1 * A_g1,  A_g2 -> B_g2: k_g2 * A_g2,  A_g3 -> B_g3: k_default * A_g3
```

| Field of `StratifiedRate` | Type                            | Meaning                                         |
| ------------------------- | ------------------------------- | ----------------------------------------------- |
| `conditions`              | `list[StratificationCondition]` | All must match the source compartment (AND).    |
| `rate`                    | `str` or `float`                | The rate for matching compartments.             |
| `absolute`                | `bool`, optional                | Forces the rate to be absolute or proportional. |

| Field of `StratificationCondition` | Meaning                                                        |
| ---------------------------------- | -------------------------------------------------------------- |
| `stratification`                   | The stratification id.                                         |
| `category`                         | Optional. The category the **source** must have.               |
| `to`                               | Optional. The category the **target** gets instead; see below. |

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
        conditions=[StratificationCondition(stratification="group", category="g2")],
    )
    .add_parameter(id="k", value=0.1)
    .add_transition(
        id="move",
        source=["g1"],
        target=["g2"],
        stratified_rates=[
            StratifiedRate(
                conditions=[StratificationCondition(stratification="subtype", to="s1")],
                rate="k",
            )
        ],
    )
    .build(
        Aggregate(
            bin_amounts={"A": 500, "B": 500},
            stratification_fractions={
                "group": {"g1": 0.5, "g2": 0.5},
                "subtype": {"s1": 0.5, "s2": 0.5},
            },
        ),
        engine="DifferenceEquations",
    )
)
print(redirect.describe().render("text"))
# A_g1 -> A_g2_s1: k * A_g1   and   B_g1 -> B_g2_s1: k * B_g1
```

A transition without a source starts from the compartments it enters, so its `category` conditions choose them and `build` rejects a `to`.

## Limiting when a transition runs

A transition runs at every step. To limit a flow, write the condition in the rate with `if`, as in `"if(step > 30, k, 0)"`, or use a [`AxisPattern`](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md), a [series parameter](https://munqu.github.io/commol/latest/guide/building-models/#series-parameters) or a [stratified rate](#stratified-rates). A comparison alone is a boolean, so `"k * (step > 30)"` is rejected.

## Impulses

An impulse moves its amount at once, at given steps, and nothing between them. Give the steps with `impulse_at=[…]` or `impulse_every=(period, offset)`. The rate is then a fraction of the source (proportional) or an amount (absolute), with the same [rules](#absolute-and-proportional-rates), sources, targets, stratified rates and accumulators as any transition.

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
    .build(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
        ),
        engine="DifferenceEquations",
    )
)

jumped = Simulation(impulse_model).run(num_steps=2)
print([round(a, 1) for a in jumped["A"]])  # [1000.0, 675.0, 607.5]
print(jumped["moved"][2])  # 225.0: a quarter of the 900 left at step 1
```

| Rule                         | In every engine                                                                                                                                                                                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Order                        | At step `n` the state jumps first; then the flows of the step run from the new state. The output at `n` shows the state after the jump (`CompiledModel::apply_impulses` in `crates/commol-model/src/system.rs`).                                                                                |
| Several impulses at one step | Every jump is computed from the state before any of them.                                                                                                                                                                                                                                       |
| Steps                        | A `DifferentialEquations` or `StochasticEvents` model takes any steps, such as `0.5`: the engine stops there, applies the jump and restarts. A `DifferenceEquations` or `StochasticSteps` model needs integer steps: `ModelBuilder.build` raises `ValidationError`, which names the transition. |
| Units                        | The rate of an impulse is an amount, `<bin unit>`, not a flow per step ([Units](https://munqu.github.io/commol/latest/guide/units-and-equations/#impulses)).                                                                                                                                    |
| `describe()`                 | Lists impulses apart from the equations, with their steps ([Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/#the-equations)).                                                                                                                                 |

An impulse is not a pulse rate. An [`AxisPattern.pulse`](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md) is a rate, which acts with the other flows for one step, and in a differential model moves less than `p` ([Time inside a step](https://munqu.github.io/commol/latest/guide/simulations/#time-inside-a-step)). An impulse moves exactly `p` of its source in both deterministic engines; a [stochastic engine](https://munqu.github.io/commol/latest/guide/stochastic/index.md) moves each unit with probability `p`.

## Amounts below zero

A `DifferenceEquations` step never leaves a compartment below zero. After the impulses of a step and after its flows, the engine checks every compartment. An amount below `−10⁻¹² × total` (the total at the start of the step) stops the run with `InfeasibleStateError`, a `ValueError`. Its message gives the compartment, the step and the amount before the step, and lists the flows out of the compartment, largest first. The tolerance lets a flow that empties a compartment exactly pass. Accumulators are not checked: a net flow may count negative (`DifferenceEquations::step` in `crates/commol-difference/src/simulation.rs`).

```python
from commol import InfeasibleStateError

too_fast = (
    ModelBuilder(name="Too fast")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_parameter(id="k", value=1.5)
    .add_transition(id="t", source=["A"], target=["B"], rate="k")
    .build(
        Aggregate(
            bin_amounts={"A": 1000, "B": 0},
        ),
        engine="DifferenceEquations",
    )
)

try:
    Simulation(too_fast).run(num_steps=3)
except InfeasibleStateError as error:
    print(error)  # The flows of step 0 leave compartment 'A' at -500, ...
```

A step moves what the flows give, so the usual causes are in the rates:

| Cause                               | Example                                                                                                                  | Fix                                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Proportional rates that sum above 1 | `rate="k"` with `k = 1.5` moves 1.5 times its source in one step                                                         | A smaller rate per step, or a shorter step                                                                              |
| An absolute rate on a bin total     | With stratifications, `"k1 * A * B / total"` moves from each compartment of `A` an amount computed from the total of `A` | `per_compartment=True`, so each compartment loses an amount from its own amount ([`per_compartment`](#per_compartment)) |
| A rate that turns negative          | `"k * B * (1 - C / capacity)"` once `C > capacity`: the flow runs backwards and drains its target                        | Bound the factor: `"k * B * max(0, 1 - C / capacity)"`                                                                  |

A calibration gives a run that stops this way a penalty and goes on; see [Calibration](https://munqu.github.io/commol/latest/guide/calibration/#losses). A `DifferentialEquations` model is not checked: its solver may step below zero by its tolerance.

## What building checks

`build` plans every flow in the engine and rejects a transition it cannot simulate as written:

| Rejected                                                                                                                                                                                                                                                                                         | Message names                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| A source or target that is neither a bin nor a category of one stratification                                                                                                                                                                                                                    | The transition                                                     |
| A category transition without exactly one source and one target                                                                                                                                                                                                                                  | The transition                                                     |
| A rate variable that is not a parameter, bin, compartment or special variable                                                                                                                                                                                                                    | The variable and the available names                               |
| Several sources with a proportional rate                                                                                                                                                                                                                                                         | The transition; use an absolute rate or `$compartment`             |
| A stratified rate whose `category` conditions match no source compartment                                                                                                                                                                                                                        | The transition; use `to` to reach another stratification           |
| A `to` condition on a transition without a source                                                                                                                                                                                                                                                | The transition; use `category` to choose the compartment it enters |
| A flow that ends in categories no compartment has                                                                                                                                                                                                                                                | The transition; add a `to` condition                               |
| `impulse_at` and `impulse_every` together, an `impulse_at` with no step, a step that is negative or not finite, a period that is not positive, or a negative offset; the `Transition` fails when it is made, with the rule of the engine (`check_impulse` in `crates/commol-model/src/rules.rs`) | The transition                                                     |
| A non-integer impulse step in a `DifferenceEquations` or `StochasticSteps` model                                                                                                                                                                                                                 | The transition and the step                                        |

## Next steps

| To learn                         | Read                                                                                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Rates that change along the axis | [Axis Patterns](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md)                       |
| Everything a formula can contain | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) |
| Units                            | [Units](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md)                         |
| The equations of a model         | [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md)              |
