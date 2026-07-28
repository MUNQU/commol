# Time-Varying Rate Patterns

Commol provides the `TimePattern` class to express time-dependent transition
rates — discrete pulses, periodic events, sliding windows, seasonal curves,
linear ramps, Gaussian bell pulses, and arbitrary user formulas. Every pattern
is parenthesised so it composes safely; security validation runs at
construction time.

`TimePattern` is the **only** user-facing class for this feature. Multi-group
schedules are built by calling `TimePattern.add_group(...)` directly on the
class and chaining further calls on the returned instance — there is no
separate schedule type to import or instantiate.

A `TimePattern` is consumed directly by `ModelBuilder.add_transition` via
the `rate=` argument:

```python
builder.add_transition("flow", ["A"], ["B"], rate=TimePattern.pulse(at=5, amount=0.1))
```

The same call works whether `rate=` is a single pattern, a single grouped
pattern, or a multi-group schedule.

---

## Single patterns

### Pulse

`TimePattern.pulse(at, amount)` fires exactly once at step `at`.

```python
from commol import TimePattern

rate = TimePattern.pulse(at=10, amount=0.05)
# Formula: (if(step == 10, 0.05, 0))
```

### Multi-step pulses

`TimePattern.pulses(at=[steps], amount)` fires at each step in `at`. Steps
must be unique and non-negative.

```python
rate = TimePattern.pulses(at=[3, 7, 14], amount=0.1)
```

### Periodic

`TimePattern.periodic(period, amount, offset=0)` fires every `period` steps,
first at step `offset`. Floor-based modulo is used so the formula compiles on
both the evalexpr fallback and the Cranelift JIT path.

```python
weekly = TimePattern.periodic(period=7, amount=0.05)
monthly_from_step_3 = TimePattern.periodic(period=30, amount=0.05, offset=3)
```

### Window

`TimePattern.window(start, end, amount)` is constant at `amount` for
`step in [start, end)` and 0 elsewhere.

```python
rate = TimePattern.window(start=5, end=12, amount=0.2)
```

### Seasonal (sinusoidal)

`TimePattern.seasonal(amplitude, period, phase=0, baseline=0)` is the
continuous formula `baseline + amplitude * sin(2*pi*(t - phase) / period)`.

```python
rate = TimePattern.seasonal(amplitude=0.005, period=365, baseline=0.01)
```

All four arguments may be parameter names (strings) — useful when the values
are calibrated.

### Gaussian pulse

`TimePattern.gaussian_pulse(center, width, peak)` is a smooth bell-shaped
pulse centred at step `center` with standard deviation `width`.

```python
rate = TimePattern.gaussian_pulse(center=20.0, width=3.0, peak=0.1)
```

### Linear ramp

`TimePattern.linear_ramp(start, end, start_value, end_value)` linearly
interpolates between `start_value` at `start` and `end_value` at `end`
(exclusive). Outside `[start, end)` the rate is 0.

```python
rate = TimePattern.linear_ramp(start=0, end=10, start_value=0.0, end_value=0.5)
```

### Arbitrary formula

`TimePattern.from_formula(expr)` wraps any rate formula string in parens and
runs the full security validator at construction time.

```python
rate = TimePattern.from_formula("beta * t / 100")
```

The wrapping makes composition safe:

```python
composed = f"2 * {TimePattern.from_formula('a + b')}"
# Composed: "2 * (a + b)" — multiplication binds to the whole expression.
```

---

## Composing patterns: `combine`

`TimePattern.combine(*patterns, mode=TimePattern.SUM)` merges multiple
patterns. Modes:

- `TimePattern.SUM` (default) — add the sub-formulas.
- `TimePattern.MAX` — pointwise maximum.
- `TimePattern.MIN` — pointwise minimum.

```python
rate = TimePattern.combine(
    TimePattern.pulse(at=0, amount=0.3),
    TimePattern.periodic(period=30, amount=0.05),
    TimePattern.seasonal(amplitude=0.005, period=365, baseline=0.0),
    mode=TimePattern.SUM,
)
```

Rules:

- `combine(p)` (single input) is the identity — returns `p` unchanged.
- All inputs must share the same `conditions` and `source_compartment`
  (or have none). The result preserves them.
- The composed formula must not exceed the security length cap
  (`SecurityConfig.max_expression_length`, default 500 chars). When it
  would, `combine` raises a clear `ValueError`.

---

## Sub-group binding: `for_group`

`pattern.for_group(conditions, source_compartment=None)` returns a copy of
the pattern restricted to compartments matching `conditions`. Use this for a
**single** sub-group:

```python
rate = TimePattern.pulse(at=5, amount=0.1).for_group(
    [{"stratification": "group", "category": "cat1"}]
)
builder.add_transition("flow", ["A"], ["B"], rate=rate)
```

For multiple sub-groups, use `TimePattern.add_group(...)` (next section).

---

## Sub-group schedules

When different stratification sub-groups should receive different patterns,
chain `add_group(...)` calls starting from the class itself:

```python
rate = (
    TimePattern.add_group(
        conditions=[{"stratification": "group", "category": "cat1"}],
        schedule=TimePattern.periodic(period=7, amount=0.05),
    )
    .add_group(
        conditions=[{"stratification": "group", "category": "cat2"}],
        schedule=TimePattern.periodic(period=30, amount=0.05),
    )
)

builder.add_transition("flow", ["A"], ["B"], rate=rate)
```

- The **first** `add_group` is a class-level call that creates a fresh
  schedule and registers the first group.
- Subsequent `add_group` calls are instance methods that append additional
  groups to the same schedule.
- Calling `add_group` on a non-schedule pattern (a single `pulse`, `periodic`,
  etc.) raises `TypeError`. Use the class-level call instead.

### Conditional exclusion by default

If you do not register a fallback, compartments not matched by any
`add_group` receive **zero flow**:

```python
# Only cat1 receives the schedule; cat2 compartments do not transition.
rate = TimePattern.add_group(
    conditions=[{"stratification": "group", "category": "cat1"}],
    schedule=TimePattern.periodic(period=7, amount=0.05),
)
```

This makes schedules opt-in by group, exactly what is needed for targeted
interventions.

### Fallback via `set_default`

Use `set_default` to register a pattern that applies to every unmatched
sub-group:

```python
rate = (
    TimePattern.add_group(
        conditions=[{"stratification": "group", "category": "cat1"}],
        schedule=TimePattern.periodic(period=7, amount=0.05),
    )
    .set_default(TimePattern.from_formula("0.01"))
)
```

The default pattern must not carry `conditions` or a `source_compartment`.
`set_default` can be called at most once per schedule.

### Per-capita vs absolute flows

For schedule groups, `absolute` controls whether the group formula is
interpreted as a per-capita rate or as an absolute count per step:

- `absolute=None` (default): infer from the formula. Expressions that
  reference compartment or sub-population variables are treated as absolute;
  other expressions are treated as per-capita.
- `absolute=False`: force per-capita behavior.
- `absolute=True`: force absolute-flow behavior.

```python
rate = (
    TimePattern.add_group(
        conditions=[{"stratification": "group", "category": "cat1"}],
        schedule=TimePattern.pulse(at=5, amount=0.1),
        absolute=False,  # 10% of the matching source compartment
    )
    .add_group(
        conditions=[{"stratification": "group", "category": "cat2"}],
        schedule=TimePattern.pulse(at=5, amount=100.0),
        absolute=True,  # exactly 100 units at step 5
    )
)
```

For advanced cases, `source_compartment` multiplies the group formula by a
specific compartment variable:

```python
rate = TimePattern.add_group(
    conditions=[{"stratification": "group", "category": "cat1"}],
    schedule=TimePattern.pulse(at=5, amount=0.25),
    source_compartment="A_cat1",
)
```

With the default `absolute=None`, that compartment reference is inferred as
an absolute flow. If you need a per-capita expression that references
population variables, pass `absolute=False` explicitly.

`set_default` rejects `source_compartment` (the default applies to every
unmatched compartment, so binding it to one name is almost always a bug).

### Numeric pulse schedules as parameters

Numeric `TimePattern.pulse(...)` and `TimePattern.pulses(...)` schedules are
stored internally as `TimeSeries` parameters instead of long generated
formula strings. This keeps extended equations readable and makes large
pulse schedules efficient to evaluate.

For grouped schedules, generated parameter names use the transition name and
the source/target group labels:

```text
{transition_id}_{source_bin}_{source_categories}_to_{target_bin}_{target_categories}
```

Generated parameter names must be unique. If a user-defined parameter already
uses the same name, model construction raises an error instead of silently
renaming it.

### Cross-category routing (`to:`)

A condition entry may include a `to` key to move population **between
categories** of the same stratification:

```python
rate = TimePattern.add_group(
    conditions=[{"stratification": "status", "category": "s0", "to": "s1"}],
    schedule=TimePattern.pulse(at=5, amount=0.5),
)

builder.add_transition("flow", ["A"], ["A"], rate=rate)
```

This creates a transition `A_s0 → A_s1` (the bin is the same, only the
category changes). Total population in bin `A` is conserved.

### Multi-stratification conditions

A group can pin multiple stratifications simultaneously:

```python
rate = TimePattern.add_group(
    conditions=[
        {"stratification": "group", "category": "cat1"},
        {"stratification": "status", "category": "s0"},
    ],
    schedule=TimePattern.pulse(at=5, amount=0.5),
)
```

The schedule only applies to compartments matching **all** listed
conditions. Most-specific match wins, so a single-condition group is
overridden by a multi-condition group whenever both apply.

---

## Special variables

All formulas have access to the standard engine variables:

- `step` — current simulation step (integer-valued, starts at 0).
- `t` — alias for `step`.
- `pi`, `e` — mathematical constants.
- `N`, `N_{combo}` — total / sub-population totals.
- Compartment and bin names (when stratifications are present).

See [mathematical-expressions.md](mathematical-expressions.md) for the full
list of available variables and operators.

---

## Validation rules

| Pattern          | Constraints                                                  |
|------------------|--------------------------------------------------------------|
| `pulse`          | `at >= 0`. `amount` strings must not contain `,`.            |
| `pulses`         | Non-empty unique step list.                                  |
| `periodic`       | `period > 0`; `0 <= offset < period`.                        |
| `window`         | `start < end`.                                               |
| `seasonal`       | Numeric `period != 0` (strings accepted as parameter names). |
| `gaussian_pulse` | `width > 0`.                                                 |
| `linear_ramp`    | `start < end`.                                               |
| `combine`        | At least one input; consistent group bindings; under length cap. |
| `from_formula`   | Non-empty expression that survives the full security validator. |
| `add_group`      | Non-empty conditions; condition dicts must have `stratification` and `category`. |
| `set_default`    | Pattern with no conditions and no `source_compartment`. At most one default per schedule. |

Bad inputs raise `pydantic.ValidationError` (or plain `ValueError` for
explicit pre-validation paths). See `tests/test_time_patterns.py` for the
canonical set of accepted and rejected cases.

---

## Where to next

- [Examples](examples.md) — fully worked patterns and schedules in abstract
  models.
- [Mathematical Expressions](mathematical-expressions.md) — full list of
  available variables, operators, and functions.
- [API reference: TimePattern](../api/time-patterns.md) — auto-generated
  signature reference.
