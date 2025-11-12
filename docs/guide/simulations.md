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

EpiModel supports two output formats for simulation results.

### Dictionary of Lists (Default)

Each disease state maps to a list of values over time:

```python
results = simulation.run(num_steps=100, output_format="dict_of_lists")

# Access results
susceptible = results["S"]  # [990, 985, 978, ...]
infected = results["I"]     # [10, 15, 22, ...]
recovered = results["R"]    # [0, 0, 0, ...]

# Get final values
final_S = results["S"][-1]
final_I = results["I"][-1]
final_R = results["R"][-1]

print(f"Final state: S={final_S:.0f}, I={final_I:.0f}, R={final_R:.0f}")
```

**Best for**: Plotting, time series analysis, accessing specific compartments

### List of Lists

Each time step is a list of all compartment values:

```python
results = simulation.run(num_steps=100, output_format="list_of_lists")

# results[time][compartment_index]
initial_state = results[0]     # [990, 10, 0]
midpoint_state = results[50]   # [450, 200, 350]
final_state = results[-1]      # [340, 50, 610]

# Iterate over time steps
for t, state in enumerate(results):
    total = sum(state)
    print(f"Step {t}: Total population = {total}")
```

**Best for**: Matrix operations, comparing states, exporting to CSV

## Working with Stratifications

Stratifications create multiple compartments by combining disease states with stratification categories. Understanding how to access and analyze stratified results is crucial for complex models.

### Understanding Stratified Compartment Names

When you add stratifications, EpiModel creates compartments by combining disease state IDs with stratification category names using underscore notation:

**Pattern**: `{disease_state}_{category1}_{category2}_...`

```python
# Model with one stratification (age)
.add_bin(id="S", name="Susceptible")
.add_bin(id="I", name="Infected")
.add_stratification(id="age", categories=["young", "old"])

# Creates compartments: S_young, S_old, I_young, I_old
```

```python
# Model with two stratifications (age and location)
.add_bin(id="I", name="Infected")
.add_stratification(id="age", categories=["child", "adult"])
.add_stratification(id="location", categories=["urban", "rural"])

# Creates compartments:
# I_child_urban, I_child_rural, I_adult_urban, I_adult_rural
```

### Accessing Stratified Results

#### 1. List All Compartments

```python
results = simulation.run(num_steps=100)

# See all compartment names
print("All compartments:", list(results.keys()))
# Output: ['S_young', 'S_old', 'I_young', 'I_old', 'R_young', 'R_old']

# Count compartments
print(f"Total compartments: {len(results)}")
```

#### 2. Access Specific Strata

```python
# Access specific age groups
young_infected = results["I_young"]
old_infected = results["I_old"]

# Access specific combinations (multiple stratifications)
urban_child_infected = results["I_child_urban"]
rural_adult_infected = results["I_adult_rural"]
```

#### 3. Filter Compartments by Pattern

```python
# Get all infected compartments
infected_keys = [key for key in results.keys() if key.startswith("I_")]
print("Infected compartments:", infected_keys)

# Get all young compartments
young_keys = [key for key in results.keys() if "_young" in key]
print("Young compartments:", young_keys)

# Get all urban compartments (multi-stratification)
urban_keys = [key for key in results.keys() if "_urban" in key]
```

### Aggregating Stratified Results

#### Sum Across One Stratification

```python
# Total infected across all age groups
total_infected = np.array(results["I_young"]) + np.array(results["I_old"])

# Or using list comprehension
total_infected = [y + o for y, o in zip(results["I_young"], results["I_old"])]
```

### Common Pitfalls

#### 1. Case Sensitivity

```python
# Compartment names use exact category names
.add_stratification(id="age", categories=["Young", "Old"])  # Capital Y and O
# Access with: results["I_Young"], results["I_Old"]
```

#### 2. Order of Stratifications

```python
# Categories combine in the order stratifications are added
.add_stratification(id="age", categories=["young", "old"])
.add_stratification(id="location", categories=["urban", "rural"])

# Creates: I_young_urban, I_young_rural, I_old_urban, I_old_rural
# NOT: I_urban_young, I_rural_young, etc.
```

#### 3. Missing Compartments

```python
# Always check compartments exist before accessing
key = "I_young"
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
    bins=["I", "R"],  # Only plot specific compartments
    linewidth=2.5,
    alpha=0.8
)
```

### Overlaying Observed Data

```python
from commol import ObservedDataPoint

observed_data = [
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
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
