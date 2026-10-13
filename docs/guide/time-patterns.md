# Time Patterns

[Back to Transitions](transitions.md)

`TimePattern` builds rates that change with the step: pulses, periodic events,
windows, seasons, ramps and bell curves, and schedules that give each subgroup
its own pattern. Pass a pattern as `rate=` to `add_transition`. Every pattern is
a tested, parenthesised expression, so patterns combine safely.

## Quick path

```python
from commol import ModelBuilder, Simulation, TimePattern

def flow_per_step(rate: TimePattern, num_steps: int = 10) -> list[float]:
    """The flow into B at each step, for a transition with no source."""
    model = (
        ModelBuilder(name="Pattern")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_transition(id="inflow", source=[], target=["B"], rate=rate)
        .set_initial_conditions(
            population_size=100,
            bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
        )
        .build(typology="DifferenceEquations")
    )
    series = Simulation(model).run(num_steps=num_steps)["B"]
    return [round(series[i + 1] - series[i], 4) for i in range(num_steps)]

print(flow_per_step(TimePattern.window(start=3, end=6, amount=1.0)))
# [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
```

A transition with no source uses the pattern as the flow. With a source, a
pattern that names no compartment is per-capita: `amount=0.1` moves 10% of the
source. See [Absolute and per-capita rates](transitions.md#absolute-and-per-capita-rates).

## Single patterns

| Pattern | Value at step `s` | Arguments |
|---|---|---|
| `pulse(at, amount)` | `amount` when `s == at`, else 0 | `at >= 0` |
| `pulses(at, amount)` | `amount` at each step in `at`, else 0 | unique steps, not empty; `amount` may be one value per step |
| `periodic(period, amount, offset=0)` | `amount` when `s = offset, offset + period, …`, else 0 | `period > 0`, `0 <= offset < period` |
| `window(start, end, amount)` | `amount` for `start <= s < end`, else 0 | `start < end` |
| `linear_ramp(start, end, start_value, end_value)` | From `start_value` at `start` towards `end_value`, for `start <= s < end`; else 0 | `start < end` |
| `seasonal(amplitude, period, phase=0, baseline=0)` | `baseline + amplitude × sin(2π (s − phase) / period)` | numeric `period != 0` |
| `gaussian_pulse(center, width, peak)` | `peak × exp(−(s − center)² / (2 width²))` | `width > 0` |
| `from_formula(expr)` | Any expression, wrapped in parentheses | not empty; passes the security check |

An invalid argument raises `pydantic.ValidationError` (a `ValueError`). These
arguments also accept a parameter name, so calibration can fit them: `amount`
of `pulse`, `pulses`, `periodic` and `window`; every argument of `seasonal`; and
`peak` of `gaussian_pulse`.

```python
print(flow_per_step(TimePattern.pulse(at=3, amount=1.0)))
# [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(flow_per_step(TimePattern.pulses(at=[2, 5], amount=1.0)))
# [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
print(flow_per_step(TimePattern.periodic(period=4, amount=1.0, offset=1)))
# [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
print(flow_per_step(TimePattern.linear_ramp(start=2, end=6, start_value=0.0, end_value=1.0)))
# [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.0, 0.0, 0.0, 0.0]
print(flow_per_step(TimePattern.seasonal(amplitude=1.0, period=8, baseline=1.0)))
# [1.0, 1.7071, 2.0, 1.7071, 1.0, 0.2929, 0.0, 0.2929, 1.0, 1.7071]
print(flow_per_step(TimePattern.gaussian_pulse(center=5.0, width=1.0, peak=1.0)))
# [0.0, 0.0003, 0.0111, 0.1353, 0.6065, 1.0, 0.6065, 0.1353, 0.0111, 0.0003]
```

`str(pattern)` and `pattern.formula` give the expression:

```python
print(TimePattern.pulse(at=10, amount=0.05).formula)  # (if(step == 10, 0.05, 0))
print(f"2 * {TimePattern.from_formula('k1 + k2')}")    # 2 * (k1 + k2)
```

The parentheses make composition safe: the `2 *` applies to the whole
expression.

## Combining patterns

`TimePattern.combine(*patterns, mode=TimePattern.SUM)` merges patterns:

| `mode` | Value |
|---|---|
| `TimePattern.SUM` (default) | The sum of the patterns |
| `TimePattern.MAX` | The largest value at each step |
| `TimePattern.MIN` | The smallest value at each step |

```python
peak_or_window = TimePattern.combine(
    TimePattern.pulse(at=3, amount=2.0),
    TimePattern.window(start=2, end=5, amount=1.0),
    mode=TimePattern.MAX,
)
print(flow_per_step(peak_or_window))
# [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

| Rule | Otherwise |
|---|---|
| At least one pattern | `ValueError` |
| All patterns have the same group binding (or none); the result keeps it | `ValueError` |
| The result is not longer than `SecurityConfig.max_expression_length` (10000 characters by default) | `ValueError` |

`combine(p)` with one pattern returns `p`.

## Patterns per subgroup

With stratifications, a schedule gives each subgroup its own pattern. Start
with the class method `TimePattern.add_group(...)` and chain more groups on the
result:

```python
def stratified_flows(rate: TimePattern, num_steps: int = 7) -> dict[str, list[float]]:
    """The change of each compartment at each step, for a transition A -> B."""
    model = (
        ModelBuilder(name="Groups")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_stratification(id="group", categories=["cat1", "cat2"])
        .add_transition(id="flow", source=["A"], target=["B"], rate=rate)
        .set_initial_conditions(
            population_size=1000,
            bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
            stratification_fractions=[
                {
                    "stratification": "group",
                    "fractions": [
                        {"category": "cat1", "fraction": 0.5},
                        {"category": "cat2", "fraction": 0.5},
                    ],
                }
            ],
        )
        .build(typology="DifferenceEquations")
    )
    results = Simulation(model).run(num_steps=num_steps)
    return {
        name: [round(series[i + 1] - series[i], 2) for i in range(num_steps)]
        for name, series in results.items()
        if name.startswith("B")
    }

schedule = (
    TimePattern.add_group(
        conditions=[{"stratification": "group", "category": "cat1"}],
        schedule=TimePattern.periodic(period=3, amount=0.1),
    )
    .add_group(
        conditions=[{"stratification": "group", "category": "cat2"}],
        schedule=TimePattern.window(start=2, end=4, amount=0.2),
    )
)
print(stratified_flows(schedule))
# {'B_cat1': [50.0, 0.0, 0.0, 45.0, 0.0, 0.0, 40.5],
#  'B_cat2': [0.0, 0.0, 100.0, 80.0, 0.0, 0.0, 0.0]}
```

| Method | What it does |
|---|---|
| `TimePattern.add_group(conditions, schedule, *, source_compartment=None, absolute=None)` | Class method: starts a schedule with its first group. On a schedule: adds a group. On a single pattern: `TypeError`. |
| `schedule.set_default(pattern)` | The pattern for compartments that match no group. At most once; the pattern may not have conditions or a `source_compartment`. |
| `pattern.for_group(conditions, *, source_compartment=None)` | A single pattern for one subgroup only. |

| Rule | Behavior |
|---|---|
| A compartment that matches no group, with no default | No flow |
| A compartment that matches several groups | The group with the most conditions wins |
| `conditions` | Not empty; each entry has `stratification` and `category`, and may have `to` |

```python
only_cat1 = TimePattern.add_group(
    conditions=[{"stratification": "group", "category": "cat1"}],
    schedule=TimePattern.pulse(at=1, amount=0.1),
)
with_default = only_cat1.set_default(TimePattern.from_formula("0.01"))
print(stratified_flows(with_default, num_steps=3))
# {'B_cat1': [0.0, 50.0, 0.0], 'B_cat2': [5.0, 4.95, 4.9]}

print(stratified_flows(TimePattern.pulse(at=1, amount=0.1).for_group(
    [{"stratification": "group", "category": "cat1"}]
), num_steps=3))
# {'B_cat1': [0.0, 50.0, 0.0], 'B_cat2': [0.0, 0.0, 0.0]}
```

### Absolute flows and `source_compartment`

| Argument of `add_group` | Effect |
|---|---|
| `absolute=None` (default) | Inferred: a pattern that names a compartment is absolute, otherwise per-capita |
| `absolute=False` | Per-capita: `amount=0.1` moves 10% of each matching source compartment |
| `absolute=True` | Absolute: `amount=100.0` moves exactly 100 |
| `source_compartment="A_cat1"` | Multiplies the pattern by that compartment, which makes it absolute |

```python
mixed = (
    TimePattern.add_group(
        conditions=[{"stratification": "group", "category": "cat1"}],
        schedule=TimePattern.pulse(at=1, amount=0.1),
        absolute=False,
    )
    .add_group(
        conditions=[{"stratification": "group", "category": "cat2"}],
        schedule=TimePattern.pulse(at=1, amount=100.0),
        absolute=True,
    )
)
print(stratified_flows(mixed, num_steps=3))
# {'B_cat1': [0.0, 50.0, 0.0], 'B_cat2': [0.0, 100.0, 0.0]}
```

### Moving between categories with `to`

A condition with `to` sends the flow to another category of the same
stratification. With the same bin as source and target, the bin's total stays
the same:

```python
routing = (
    ModelBuilder(name="Routing")
    .add_bin(id="A", name="State A")
    .add_stratification(id="status", categories=["s0", "s1"])
    .add_transition(
        id="flow",
        source=["A"],
        target=["A"],
        rate=TimePattern.add_group(
            conditions=[{"stratification": "status", "category": "s0", "to": "s1"}],
            schedule=TimePattern.pulse(at=1, amount=0.5),
        ),
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[{"bin": "A", "fraction": 1.0}],
        stratification_fractions=[
            {
                "stratification": "status",
                "fractions": [
                    {"category": "s0", "fraction": 0.5},
                    {"category": "s1", "fraction": 0.5},
                ],
            }
        ],
    )
    .build(typology="DifferenceEquations")
)
routed = Simulation(routing).run(num_steps=2)
print(routed["A_s0"], routed["A_s1"])  # [500.0, 500.0, 250.0] [500.0, 500.0, 750.0]
```

## Patterns in a differential model

A pattern reads `step`. In a `DifferentialEquations` model, `step` is constant
on each unit interval `[n, n + 1)`, so a pattern is a rate that holds for one
time unit. A per-capita `pulse` of `amount` moves `1 − e^(−amount)` of its
source there, and `amount` in a difference model. To move a fraction at once in
both engines, use an [impulse](transitions.md#impulses). See
[Time inside a step](simulations.md#time-inside-a-step).

## How patterns are stored

Numeric `pulse` and `pulses` patterns become
[time-series parameters](building-models.md#time-series-parameters) instead of
long formulas, so equations stay readable and many pulses stay fast:

| Pattern | Generated parameter id |
|---|---|
| Not in a group | `_ts_<n>` |
| In a group | `{transition}_{source bin}_{source categories}_to_{target bin}_{target categories}`, such as `flow_A_cat1_to_B_cat1` |

A generated id must not clash with a parameter of yours; building the model
raises if it does.

## Next steps

| To learn | Read |
|---|---|
| Everything a formula can contain | [Mathematical Expressions](mathematical-expressions.md) |
| How rates become flows | [Transitions](transitions.md) |
| Signatures | [API reference: TimePattern](../api/time-patterns.md) |
