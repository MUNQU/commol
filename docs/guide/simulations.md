# Running Simulations

Once you've built a model, use the `Simulation` class to run it and analyze results.

## Basic Simulation

```python
from commol import Simulation

# Create simulation from model
simulation = Simulation(model)

# Run for 100 time steps
results = simulation.run(num_steps=100)
```

## Output Formats

Commol supports two output formats for simulation results.

### Dictionary of Lists (Default)

Each compartment maps to a list of values over time:

```python
results = simulation.run(num_steps=100, output_format="dict_of_lists")

# Access results
state_a = results["A"]  # [900, 880, 855, ...]
state_b = results["B"]  # [100, 120, 145, ...]
state_c = results["C"]  # [0, 0, 0, ...]

# Get final values
final_A = results["A"][-1]
final_B = results["B"][-1]
final_C = results["C"][-1]

print(f"Final state: A={final_A:.0f}, B={final_B:.0f}, C={final_C:.0f}")
```

**Best for**: Plotting, time series analysis, accessing specific compartments

### List of Lists

Each time step is a list of all compartment values:

```python
results = simulation.run(num_steps=100, output_format="list_of_lists")

# results[time][compartment_index]
initial_state = results[0]     # [900, 100, 0]
midpoint_state = results[50]   # [450, 300, 250]
final_state = results[-1]      # [200, 100, 700]

# Iterate over time steps
for t, state in enumerate(results):
    total = sum(state)
    print(f"Step {t}: Total population = {total}")
```

**Best for**: Matrix operations, comparing states, exporting to CSV

## Accumulator Outputs

If the model defines [accumulators](core-concepts.md#accumulators), they appear as extra columns in the simulation output alongside compartments. Accumulator columns start at 0 and grow monotonically; they are never subtracted from.

```python
model = (
    ModelBuilder("Model with accumulator")
    .add_bin("A", "State A")
    .add_bin("B", "State B")
    .add_accumulator("cum_ab", "Cumulative A→B")
    .add_parameter("k1", 0.1)
    .add_transition("flow", ["A"], ["B"], rate="k1", accumulators=["cum_ab"])
    .set_initial_conditions(1000, [{"bin": "A", "fraction": 1.0}, {"bin": "B", "fraction": 0.0}])
    .build("DifferenceEquations")
)

sim = Simulation(model)
results = sim.run(100)

# Population outputs
a_values = results["A"]     # decreasing
b_values = results["B"]     # increasing

# Accumulator — equals total flow into B since t=0
cum = results["cum_ab"]
```

To enumerate all output names (compartments + accumulators), use `simulation.simulation_outputs`:

```python
print(simulation.simulation_outputs)
# ['A', 'B', 'cum_ab']
```

## Working with Stratified Results

When you use stratifications, your simulation results contain separate time series for each stratified compartment. Understanding how to navigate, access, and aggregate these results is essential for analysis.

### Understanding the Naming Convention

Commol creates stratified compartment names by joining the base compartment ID with category names using underscores. The categories appear in the **order stratifications were added** to the model.

**Pattern**: `{base_compartment}_{category1}_{category2}_...`

```python
# Model definition
.add_bin(id="A", name="State A")
.add_bin(id="B", name="State B")
.add_stratification(id="group", categories=["g1", "g2"])   # Added first
.add_stratification(id="type", categories=["t1", "t2"])    # Added second

# Resulting compartment names (8 total):
# A_g1_t1, A_g1_t2, A_g2_t1, A_g2_t2
# B_g1_t1, B_g1_t2, B_g2_t1, B_g2_t2
```

**Important**: The order is `{bin}_{group}_{type}` because group was added before type. If you reverse the order of `add_stratification` calls, names would be `{bin}_{type}_{group}`.

### Accessing Stratified Results

#### 1. List All Compartments

```python
results = simulation.run(num_steps=100)

# See all compartment names
print("All compartments:", list(results.keys()))
# Output: ['A_g1', 'A_g2', 'B_g1', 'B_g2', 'C_g1', 'C_g2']

# Count compartments
print(f"Total compartments: {len(results)}")
```

#### 2. Access Specific Strata

```python
# Access specific categories
g1_b = results["B_g1"]
g2_b = results["B_g2"]

# Access specific combinations (multiple stratifications)
g1_t1_b = results["B_g1_t1"]
g2_t2_b = results["B_g2_t2"]
```

#### 3. List the Outputs of a Bin or Accumulator

`outputs_for` returns the outputs a bin or accumulator expands into:

```python
# Every output of bin B
print(simulation.outputs_for("B"))
# Output: ['B_g1', 'B_g2']

# Several at once, keyed by the id they belong to
grouped = simulation.group_outputs(["A", "B"])
```

!!! warning "Do not match output names by prefix"

    `[k for k in results if k.startswith("B_")]` also matches any other output
    whose name begins with `B_`, such as an accumulator declared as `B_events`.
    It also breaks under conditional stratifications, where an output may carry
    fewer category suffixes than its siblings. `outputs_for` resolves names from
    the model structure, so neither case can go wrong.

Filtering by *category* still requires matching names, since a category is not a
declared output on its own:

```python
# Get all g1 outputs
g1_keys = [key for key in results.keys() if "_g1" in key]
```

### Aggregating Stratified Results

A common task is computing totals across one or more stratification dimensions.

#### Sum Across One Stratification

```python
import numpy as np

# Total B across all groups
total_b = np.array(results["B_g1"]) + np.array(results["B_g2"])

# Or using list comprehension (no NumPy required)
total_b = [a + b for a, b in zip(results["B_g1"], results["B_g2"])]
```

#### Sum Across Multiple Stratifications

```python
# Model has group=[g1, g2] and type=[t1, t2]
# Get total B (sum across all strata)
total_B = (
    np.array(results["B_g1_t1"]) +
    np.array(results["B_g1_t2"]) +
    np.array(results["B_g2_t1"]) +
    np.array(results["B_g2_t2"])
)

# Or let the simulation resolve the outputs of B
total_B = simulation.total_series(results, ["B"])
```

`total_series` accepts several ids and sums all of their outputs together, which
is how you total a set of bins:

```python
# Combined population of A and B at each step
total_ab = simulation.total_series(results, ["A", "B"])
```

#### Sum by Stratification Category

```python
# Total g1 population (across all states)
g1_keys = [k for k in results.keys() if "_g1" in k]
total_g1 = sum(np.array(results[k]) for k in g1_keys)

# Total t2 population
t2_keys = [k for k in results.keys() if "_t2" in k]
total_t2 = sum(np.array(results[k]) for k in t2_keys)
```

#### Create Aggregated DataFrame

```python
import pandas as pd
import numpy as np

# Convert results to DataFrame
df = pd.DataFrame(results)

# Add aggregated columns
df["B_total"] = df[[c for c in df.columns if c.startswith("B_")]].sum(axis=1)
df["g1_total"] = df[[c for c in df.columns if "_g1" in c]].sum(axis=1)

# Compute proportions
df["B_proportion"] = df["B_total"] / df[[c for c in df.columns if not c.endswith("_total")]].sum(axis=1)
```

### Common Pitfalls

#### 1. Case Sensitivity

```python
# Compartment names use exact category names as declared
.add_stratification(id="group", categories=["G1", "G2"])  # Capital G
# Access with: results["B_G1"], results["B_G2"]
```

#### 2. Order of Stratifications

```python
# Categories combine in the order stratifications are added
.add_stratification(id="group", categories=["g1", "g2"])
.add_stratification(id="type", categories=["t1", "t2"])

# Creates: B_g1_t1, B_g1_t2, B_g2_t1, B_g2_t2
# NOT: B_t1_g1, B_t2_g1, etc.
```

#### 3. Missing Compartments

```python
# Always check compartments exist before accessing
key = "B_g1"
if key in results:
    data = results[key]
else:
    print(f"Compartment {key} not found. Available: {list(results.keys())}")
```

## Visualizing Results

Commol provides the `SimulationPlotter` class for visualizing simulation results with automatic subplot organization and Seaborn styling.

### Basic Plotting

```python
from commol import SimulationPlotter

# After running simulation
results = simulation.run(num_steps=100)

# Create plotter
plotter = SimulationPlotter(simulation, results)

# Plot time series (one subplot per compartment)
plotter.plot_series(output_file="results.png")

# Plot cumulative results
plotter.plot_cumulative(output_file="cumulative.png")
```

### Customizing Plots

```python
from commol import PlotConfig, SeabornStyleConfig

# Custom configuration
config = PlotConfig(
    figsize=(16, 10),
    dpi=150,
    layout=(2, 2),  # 2x2 subplot grid
    seaborn=SeabornStyleConfig(
        style="darkgrid",      # darkgrid, whitegrid, dark, white, ticks
        palette="Set2",        # Color palette
        context="talk"         # paper, notebook, talk, poster
    )
)

plotter.plot_series(
    output_file="custom.png",
    config=config,
    bins=["B", "C"],  # Only plot specific compartments
    linewidth=2.5,
    alpha=0.8
)
```

### Overlaying Observed Data

```python
from commol import ObservedDataPoint

observed_data = [
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
    ObservedDataPoint(step=30, compartment="B", value=62.3),
]

plotter.plot_series(
    output_file="with_data.png",
    observed_data=observed_data
)
```

## Next Steps

- [Examples](examples.md) - Complete model examples with analysis
- [API Reference](../api/simulation.md) - Detailed Simulation API
- [Mathematical Expressions](mathematical-expressions.md) - Advanced formulas
