# Core Concepts

Commol is built around several key concepts that work together to create compartment models.

## Compartments

Compartments (also called bins or states) represent distinct states that entities in your model can occupy. In the API, these are added using `add_bin()`.

```python
builder.add_bin(id="A", name="State A")
builder.add_bin(id="B", name="State B")
builder.add_bin(id="C", name="State C")
```

A model can have any number of compartments with any names. Compartments represent whatever discrete states are meaningful in your system.

## Stratifications

Stratifications divide your population into subgroups along a dimension that matters for the model. When you add a stratification, Commol automatically creates separate compartments for each category, tracking dynamics independently within each subgroup.

### How Stratification Works

When you define compartments **without** stratification, each compartment represents the entire population in that state:

```
Base compartments: A, B, C
Total: 3 compartments
```

When you add a stratification, Commol **expands** each compartment by creating one version per category. The naming pattern is `{compartment}_{category}`:

```
Add stratification: group = [g1, g2]

Expanded compartments:
  A → A_g1, A_g2
  B → B_g1, B_g2
  C → C_g1, C_g2

Total: 6 compartments (3 bins × 2 categories)
```

With **multiple stratifications**, compartments are expanded using the Cartesian product of all categories by default. Each additional stratification multiplies the number of compartments:

```
Add stratifications:
  group = [g1, g2]
  type  = [t1, t2]

Expanded compartments:
  A → A_g1_t1, A_g1_t2, A_g2_t1, A_g2_t2
  B → B_g1_t1, B_g1_t2, B_g2_t1, B_g2_t2
  C → C_g1_t1, C_g1_t2, C_g2_t1, C_g2_t2

Total: 12 compartments (3 bins × 2 groups × 2 types)
```

The order of category suffixes matches the order in which stratifications are added to the model.

### Conditional Stratifications

Sometimes a second stratification only makes sense for a subset of the first. You can express this with the `conditions` parameter: the stratification is only applied to compartments whose already-assigned categories satisfy all conditions.

```
Add stratifications:
  group = [g1, g2]
  subtype = [s1, s2]   (conditions: group = g2)

Expanded compartments:
  A_g1          ← subtype not applied (condition not met)
  A_g2_s1
  A_g2_s2
  B_g1
  B_g2_s1
  B_g2_s2
  C_g1
  C_g2_s1
  C_g2_s2

Total: 9 compartments instead of 12
```

Conditions may only reference stratifications declared **before** the conditional one. See [Conditional Stratifications](building-models.md#conditional-stratifications) in the Building Models guide for the full API.

### Defining Stratifications

```python
# Unconditional stratification
builder.add_stratification(
    id="group",
    categories=["g1", "g2"],
    description="Primary grouping dimension"
)

# Conditional stratification — only expands g2 compartments
builder.add_stratification(
    id="subtype",
    categories=["s1", "s2"],
    conditions=[{"stratification": "group", "category": "g2"}]
)
```

### Initial Conditions with Stratifications

When setting initial conditions, you specify:

1. **Bin fractions**: How the population is distributed across states
2. **Stratification fractions**: How each state is distributed across categories

These fractions are applied multiplicatively:

```python
builder.set_initial_conditions(
    population_size=10000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
        {"bin": "C", "fraction": 0.0},
    ],
    stratification_fractions=[
        {
            "stratification": "group",
            "fractions": [
                {"category": "g1", "fraction": 0.6},
                {"category": "g2", "fraction": 0.4},
            ]
        }
    ]
)
```

**Resulting initial populations (single stratification):**

| Compartment | Calculation       | Initial Value |
| ----------- | ----------------- | ------------- |
| `A_g1`      | 10000 × 0.9 × 0.6 | 5400          |
| `A_g2`      | 10000 × 0.9 × 0.4 | 3600          |
| `B_g1`      | 10000 × 0.1 × 0.6 | 600           |
| `B_g2`      | 10000 × 0.1 × 0.4 | 400           |
| `C_g1`      | 10000 × 0.0 × 0.6 | 0             |
| `C_g2`      | 10000 × 0.0 × 0.4 | 0             |

#### Updating fractions after building

Both kinds of fraction can be changed on a built model:

```python
model.update_initial_conditions({"A": 0.85, "B": 0.15})
model.update_stratification_fractions({"g1": 0.7})   # g2 becomes 0.3
```

Within each stratification, at most one category may be omitted. The omitted
category receives whatever remains of 1.0, so the fractions always sum to 1.0:

```python
model.update_stratification_fractions({"low": 0.2, "mid": 0.3})
# a third category `high` becomes 0.5
```

Naming every category is also allowed, in which case the given values must
already sum to 1.0. Omitting two or more categories raises, because the split
between them would be undetermined.

Both methods are also what [calibration](calibration.md) writes back when a bin
fraction or a stratification split is fitted.

#### Subgroup head counts

Category fractions are relative to the group their stratification applies to, so
a conditional stratification's fractions are *not* fractions of the whole
population. `subgroup_population` composes the chain for you:

```python
model.subgroup_population()                    # whole population
model.subgroup_population(["group1"])          # 10000 x 0.6
model.subgroup_population(["group1", "sub1"])  # ... x the sub1 fraction
```

`get_conditioning_categories` reports which subgroup a stratification
subdivides, empty when it applies to everyone:

```python
model.get_conditioning_categories("group")   # ()
model.get_conditioning_categories("sub")     # ("group1",)
```

This is the same composition the engine uses to initialize compartments
(`bin_fraction x stratification_fractions`), so the head counts agree with the
populations a simulation starts from.

### Why Use Stratifications?

Stratifications are useful when:

- **Different subgroups have different rates**: transition rates vary by category
- **Tracking subgroup dynamics independently**: you need per-category output over time
- **Fitting to stratified data**: model outputs must match data broken down by category
- **Modelling targeted flows**: some transitions only apply to specific categories

## Parameters

Parameters are global constants (or formulas) used throughout your model:

```python
builder.add_parameter(
    id="k1",
    value=0.3,
    description="Forward rate constant"
)

builder.add_parameter(
    id="k2",
    value=0.1,
    description="Reverse rate constant"
)
```

Parameters can be constants (`float`), formulas (`str`) that reference other parameters or special variables, or `None` when the value is to be determined by calibration.

## Transitions

Transitions define how populations move between compartments:

### Simple Transitions

```python
# Constant-rate flow: A → B
builder.add_transition(
    id="forward",
    source=["A"],
    target=["B"],
    rate="k1"
)
```

### Formula-Based Transitions

```python
# Population-dependent flow: A → B
builder.add_transition(
    id="transfer",
    source=["A"],
    target=["B"],
    rate="k1 * A * B / N"
)
```

### Multi-Source Transitions

```python
# Outflow from multiple states
builder.add_transition(
    id="outflow",
    source=["A", "B", "C"],
    target=[],  # Empty = removal from system
    rate="mu"
)
```

## Accumulators

Accumulators are cumulative event counters that are tracked as simulation outputs but are **not part of the population**. Unlike compartments, the value accumulated is never subtracted — it only grows as transitions fire.

A common use case is cumulative incidence: total number of events that have occurred since time zero.

```python
# Define an accumulator
builder.add_accumulator(id="cum_events", name="Cumulative events A→B")

# Attach it to a transition — every flow from A to B increments the accumulator
builder.add_transition(
    id="transfer",
    source=["A"],
    target=["B"],
    rate="k1 * A * B / N",
    accumulators=["cum_events"]
)
```

After running the simulation, accumulator values appear in the output alongside compartments:

```python
results = simulation.run(100)
# results["cum_events"] contains the cumulative total at each step
```

With stratifications, accumulators are expanded identically to bins:

```
Accumulator: cum_events
Stratification: group = [g1, g2]

Output columns: cum_events_g1, cum_events_g2
```

Accumulators are reset to 0 when the simulation is reset.

## Initial Conditions

Initial conditions define the starting state of your model:

```python
builder.set_initial_conditions(
    population_size=1000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.9},
        {"bin": "B", "fraction": 0.1},
        {"bin": "C", "fraction": 0.0},
    ]
)
```

### Validation Rules

- Bin fractions must sum to 1.0
- Stratification fractions must sum to 1.0 for each stratification
- Population size must be positive
- All bins must have initial fractions defined

## Model Types

Commol currently supports:

```python
model = builder.build(typology="DifferenceEquations")
```

### Difference Equations

- Discrete time steps
- Deterministic dynamics
- Fast computation
- Best for: Population-level modeling, policy analysis

## Steps and Physical Time

A model is defined purely in steps. Nothing in it records how long a step lasts,
so a rate is always "per step" and a period is always a step count.

`TimeScale` attaches a duration to a step, which turns physical quantities into
the step counts and per-step rates a model needs:

```python
from commol import TimeScale

scale = TimeScale(step_seconds=60 * 60)   # one hour per step

scale.steps_per_day                        # 24
scale.steps_per_week                       # 168
scale.steps_from_days(3)                   # 72
```

If a period is not a whole number of steps, the conversion raises rather than
rounding, which catches a step size that cannot express the periods a model
relies on:

```python
daily = TimeScale(step_seconds=24 * 60 * 60)
daily.steps_from_hours(1)   # ValueError: not a whole number of steps
```

### Converting rates

Two conversions cover most transition rates:

```python
# Mean residence time of 4 days -> per-step rate
scale.rate_from_mean_duration(4 * 24 * 60 * 60)

# 0.3 probability accumulated over a week -> per-step rate
scale.rate_from_probability(0.3, scale.steps_per_week)
```

`rate_from_mean_duration` returns `1 - exp(-step / duration)`, and
`rate_from_probability` returns `1 - (1 - p) ** (1 / period_steps)`, so applying
the latter over its period reproduces the original probability.

### Window grids

For a quantity reported per period, `window_start`, `window_end` and
`window_index` give one grid usable in both directions:

```python
week = scale.steps_per_week

scale.window_start(0, week)   # 0
scale.window_end(0, week)     # 168, the start of window 1
scale.window_index(168, week) # 0
```

`window_index` is the exact inverse of `window_end`, and returns the containing
window for any step in between. A window covers `(start, end]`, which is the
interval [windowed observations](calibration.md#calibrating-against-cumulative-outputs)
are compared over.

## The Model Building Process

A typical workflow:

1. **Define compartments** - What states exist in your model?
2. **Add stratifications** (optional) - What subgroups matter?
3. **Define parameters** - What rates and constants?
4. **Add accumulators** (optional) - What cumulative counters do you need?
5. **Create transitions** - How do populations flow between compartments?
6. **Set initial conditions** - What's the starting state?
7. **Build the model** - Validate and construct
8. **Run simulation** - Execute and analyze

## Next Steps

- [Building Models](building-models.md) - Detailed ModelBuilder API
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Examples](examples.md) - Complete model examples
