# Axis Patterns

[Back to Transitions](transitions.md)

`AxisPattern` builds rates that change along the axis: pulses, periodic
events, intervals, seasons, ramps and bell curves, and schedules that give each
subgroup its own pattern. Pass a pattern as `rate=` to `add_transition`. Every pattern is
a tested, parenthesised expression, so patterns combine safely.

## Quick path

```python
from commol import ModelBuilder, Simulation, AxisPattern

def flow_per_step(rate: AxisPattern, num_steps: int = 10) -> list[float]:
    """The flow into B at each step, for a transition with no source."""
    model = (
        ModelBuilder(name="Pattern")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_transition(id="inflow", source=[], target=["B"], rate=rate)
        .set_initial_conditions(
            total=100,
            bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
        )
        .build(engine="DifferenceEquations")
    )
    series = Simulation(model).run(num_steps=num_steps)["B"]
    return [round(series[i + 1] - series[i], 4) for i in range(num_steps)]

print(flow_per_step(AxisPattern.interval(start=3, end=6, value=1.0)))
# [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
```

A transition with no source uses the pattern as the flow. With a source, a
pattern that names no compartment is per-capita: `value=0.1` moves 10% of the
source. See [Absolute and per-capita rates](transitions.md#absolute-and-per-capita-rates).

## Single patterns

Each pattern reads `step` or `t` (see
[Patterns in a differential model](#patterns-in-a-differential-model)). In a
difference model both are the step `s`.

| Pattern | Reads | Value | Arguments |
|---|---|---|---|
| `pulse(at, value)` | `step` | `value` when `step == at`, else 0 | integer `at >= 0` |
| `pulses(at, value)` | `step` | `value` at each step in `at`, else 0 | unique integer steps, not empty; `value` may be one value per step |
| `periodic(period, value, offset=0)` | `step` | `value` when `step = offset, offset + period, …`, else 0 | integer `period > 0`, `0 <= offset < period` |
| `interval(start, end, value)` | `t` | `value` for `start <= t < end`, else 0 | `start < end`, any finite positions |
| `linear_ramp(start, end, start_value, end_value)` | `t` | From `start_value` at `start` towards `end_value`, for `start <= t < end`; else 0 | `start < end`, any finite positions |
| `seasonal(amplitude, period, phase=0, baseline=0)` | `t` | `baseline + amplitude × sin(2π (t − phase) / period)` | numeric `period != 0` |
| `gaussian_pulse(center, width, peak)` | `t` | `peak × exp(−(t − center)² / (2 width²))` | `width > 0` |
| `from_formula(expr)` | Any expression, wrapped in parentheses | not empty; passes the security check |

An invalid argument raises `pydantic.ValidationError` (a `ValueError`). These
arguments also accept a parameter name, so calibration can fit them: `value`
of `pulse`, `pulses`, `periodic` and `interval`; every argument of `seasonal`; and
`peak` of `gaussian_pulse`.

```python
print(flow_per_step(AxisPattern.pulse(at=3, value=1.0)))
# [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(flow_per_step(AxisPattern.pulses(at=[2, 5], value=1.0)))
# [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
print(flow_per_step(AxisPattern.periodic(period=4, value=1.0, offset=1)))
# [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
print(flow_per_step(AxisPattern.linear_ramp(start=2, end=6, start_value=0.0, end_value=1.0)))
# [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.0, 0.0, 0.0, 0.0]
print(flow_per_step(AxisPattern.seasonal(amplitude=1.0, period=8, baseline=1.0)))
# [1.0, 1.7071, 2.0, 1.7071, 1.0, 0.2929, 0.0, 0.2929, 1.0, 1.7071]
print(flow_per_step(AxisPattern.gaussian_pulse(center=5.0, width=1.0, peak=1.0)))
# [0.0, 0.0003, 0.0111, 0.1353, 0.6065, 1.0, 0.6065, 0.1353, 0.0111, 0.0003]
```

`str(pattern)` and `pattern.formula` give the expression:

```python
print(AxisPattern.pulse(at=10, value=0.05).formula)  # (if(step == 10, 0.05, 0))
print(f"2 * {AxisPattern.from_formula('k1 + k2')}")    # 2 * (k1 + k2)
```

The parentheses make composition safe: the `2 *` applies to the whole
expression.

## Combining patterns

`AxisPattern.combine(*patterns, mode=AxisPattern.SUM)` merges patterns:

| `mode` | Value |
|---|---|
| `AxisPattern.SUM` (default) | The sum of the patterns |
| `AxisPattern.MAX` | The largest value at each position |
| `AxisPattern.MIN` | The smallest value at each position |

```python
peak_or_interval = AxisPattern.combine(
    AxisPattern.pulse(at=3, value=2.0),
    AxisPattern.interval(start=2, end=5, value=1.0),
    mode=AxisPattern.MAX,
)
print(flow_per_step(peak_or_interval))
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
with the class method `AxisPattern.add_group(...)` and chain more groups on the
result:

```python
def stratified_flows(rate: AxisPattern, num_steps: int = 7) -> dict[str, list[float]]:
    """The change of each compartment at each step, for a transition A -> B."""
    model = (
        ModelBuilder(name="Groups")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_stratification(id="group", categories=["cat1", "cat2"])
        .add_transition(id="flow", source=["A"], target=["B"], rate=rate)
        .set_initial_conditions(
            total=1000,
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
        .build(engine="DifferenceEquations")
    )
    results = Simulation(model).run(num_steps=num_steps)
    return {
        name: [round(series[i + 1] - series[i], 2) for i in range(num_steps)]
        for name, series in results.items()
        if name.startswith("B")
    }

schedule = (
    AxisPattern.add_group(
        conditions=[{"stratification": "group", "category": "cat1"}],
        schedule=AxisPattern.periodic(period=3, value=0.1),
    )
    .add_group(
        conditions=[{"stratification": "group", "category": "cat2"}],
        schedule=AxisPattern.interval(start=2, end=4, value=0.2),
    )
)
print(stratified_flows(schedule))
# {'B_cat1': [50.0, 0.0, 0.0, 45.0, 0.0, 0.0, 40.5],
#  'B_cat2': [0.0, 0.0, 100.0, 80.0, 0.0, 0.0, 0.0]}
```

| Method | What it does |
|---|---|
| `AxisPattern.add_group(conditions, schedule, *, source_compartment=None, absolute=None)` | Class method: starts a schedule with its first group. On a schedule: adds a group. On a single pattern: `TypeError`. |
| `schedule.set_default(pattern)` | The pattern for compartments that match no group. At most once; the pattern may not have conditions or a `source_compartment`. |
| `pattern.for_group(conditions, *, source_compartment=None)` | A single pattern for one subgroup only. |

| Rule | Behavior |
|---|---|
| A compartment that matches no group, with no default | No flow |
| A compartment that matches several groups | The group with the most conditions wins |
| `conditions` | Not empty; each entry has `stratification` and `category`, and may have `to` |

```python
only_cat1 = AxisPattern.add_group(
    conditions=[{"stratification": "group", "category": "cat1"}],
    schedule=AxisPattern.pulse(at=1, value=0.1),
)
with_default = only_cat1.set_default(AxisPattern.from_formula("0.01"))
print(stratified_flows(with_default, num_steps=3))
# {'B_cat1': [0.0, 50.0, 0.0], 'B_cat2': [5.0, 4.95, 4.9]}

print(stratified_flows(AxisPattern.pulse(at=1, value=0.1).for_group(
    [{"stratification": "group", "category": "cat1"}]
), num_steps=3))
# {'B_cat1': [0.0, 50.0, 0.0], 'B_cat2': [0.0, 0.0, 0.0]}
```

### Absolute flows and `source_compartment`

| Argument of `add_group` | Effect |
|---|---|
| `absolute=None` (default) | Inferred: a pattern that names a compartment is absolute, otherwise per-capita |
| `absolute=False` | Per-capita: `value=0.1` moves 10% of each matching source compartment |
| `absolute=True` | Absolute: `value=100.0` moves exactly 100 |
| `source_compartment="A_cat1"` | Multiplies the pattern by that compartment, which makes it absolute |

```python
mixed = (
    AxisPattern.add_group(
        conditions=[{"stratification": "group", "category": "cat1"}],
        schedule=AxisPattern.pulse(at=1, value=0.1),
        absolute=False,
    )
    .add_group(
        conditions=[{"stratification": "group", "category": "cat2"}],
        schedule=AxisPattern.pulse(at=1, value=100.0),
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
        rate=AxisPattern.add_group(
            conditions=[{"stratification": "status", "category": "s0", "to": "s1"}],
            schedule=AxisPattern.pulse(at=1, value=0.5),
        ),
    )
    .set_initial_conditions(
        total=1000,
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
    .build(engine="DifferenceEquations")
)
routed = Simulation(routing).run(num_steps=2)
print(routed["A_s0"], routed["A_s1"])  # [500.0, 500.0, 250.0] [500.0, 500.0, 750.0]
```

## Patterns in a differential model

In a `DifferentialEquations` model `step` is constant on each unit interval
`[n, n + 1)` and `t` moves continuously.

| Pattern | In a differential model |
|---|---|
| `pulse`, `pulses`, `periodic` (read `step`) | A rate that holds for one unit: a per-capita `pulse` of `value` moves `1 − e^(−value)` of its source, and `value` in a difference model. To move a fraction at once in both engines, use an [impulse](transitions.md#impulses). |
| `interval`, `linear_ramp` (read `t`) | Open and close at any position, such as `interval(start=2.5, end=4, …)`; the ramp is continuous. The engine restarts its integration at both bounds. |
| `seasonal`, `gaussian_pulse` (read `t`) | Continuous in `t`. |

```python
def remaining_a(rate: AxisPattern, at: float) -> float:
    """A of a per-capita A -> B transition, in a differential model."""
    model = (
        ModelBuilder(name="Pattern ODE")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_transition(id="flow", source=["A"], target=["B"], rate=rate)
        .set_initial_conditions(
            total=1000,
            bin_fractions=[{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}],
        )
        .build(engine="DifferentialEquations")
    )
    return Simulation(model).run_at([at])["A"][0]

# The rate 0.5 acts on [2.5, 4): A keeps exp(-0.75) of its start.
print(round(remaining_a(AxisPattern.interval(start=2.5, end=4, value=0.5), 6), 2))
# 472.37
```

See [Time inside a step](simulations.md#time-inside-a-step).

## How patterns are stored

Numeric `pulse` and `pulses` patterns become
[series parameters](building-models.md#series-parameters) instead of
long formulas, so equations stay readable and many pulses stay fast:

| Pattern | Generated parameter id |
|---|---|
| Not in a group | `_series_<n>` |
| In a group | `{transition}_{source bin}_{source categories}_to_{target bin}_{target categories}`, such as `flow_A_cat1_to_B_cat1` |

A generated id must not clash with a parameter of yours; building the model
raises if it does.

## Next steps

| To learn | Read |
|---|---|
| Everything a formula can contain | [Mathematical Expressions](mathematical-expressions.md) |
| How rates become flows | [Transitions](transitions.md) |
| Signatures | [API reference: AxisPattern](../api/axis-patterns.md) |
