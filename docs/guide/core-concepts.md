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

## The Model Building Process

A typical workflow:

1. **Define compartments** - What states exist in your model?
2. **Add stratifications** (optional) - What subgroups matter?
3. **Define parameters** - What rates and constants?
4. **Create transitions** - How do populations flow between compartments?
5. **Set initial conditions** - What's the starting state?
6. **Build the model** - Validate and construct
7. **Run simulation** - Execute and analyze

## Next Steps

- [Building Models](building-models.md) - Detailed ModelBuilder API
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
- [Examples](examples.md) - Complete model examples
