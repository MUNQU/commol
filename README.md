# EpiModel

A high-performance mathematical epidemiology library for modeling infectious disease spread using difference equations. EpiModel provides a clean Python API backed by a fast Rust engine for numerical computations.

## Features

- **Intuitive Model Building**: Fluent API for constructing epidemiological models
- **Mathematical Expressions**: Support for complex mathematical formulas in transition rates
- **High Performance**: Rust-powered simulation engine for fast computations
- **Flexible Architecture**: Support for stratified populations and conditional transitions
- **Type Safety**: Comprehensive validation using Pydantic models
- **Multiple Output Formats**: Get results as dictionaries or lists for easy analysis

## Installation

```bash
# Install from source (requires Rust toolchain)
pip install epimodel
```

## Quick Start

Here's a simple SIR (Susceptible-Infected-Recovered) model:

```python
from epimodel import ModelBuilder, Simulation
from epimodel.constants import ModelTypes

# Build the model using the fluent API
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected") 
    .add_disease_state(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)  # Transmission rate
    .add_parameter(id="gamma", value=0.1)  # Recovery rate
    .add_parameter(id="N", value=1000.0)  # Population size
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"  # Mathematical formula
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="gamma"  # Parameter reference
    )
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Run the simulation
simulation = Simulation(model)
results = simulation.run(num_steps=100)

# Access results
print(f"Susceptible over time: {results['S']}")
print(f"Infected over time: {results['I']}")
print(f"Recovered over time: {results['R']}")
```

## Core Concepts

### Disease States

Disease states represent different stages of infection (e.g., Susceptible, Infected, Recovered):

```python
builder.add_disease_state(id="S", name="Susceptible")
builder.add_disease_state(id="E", name="Exposed")
builder.add_disease_state(id="I", name="Infected")
builder.add_disease_state(id="R", name="Recovered")
```

### Stratifications

Stratifications allow modeling of population subgroups:

```python
# Age stratification
builder.add_stratification(
    id="age_group",
    categories=["young", "adult", "elderly"]
)

# Set initial conditions for stratifications
builder.set_initial_conditions(
    population_size=10000,
    disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0},
    stratification_fractions={
        "age_group": {
            "young": 0.3,
            "adult": 0.5, 
            "elderly": 0.2
        }
    }
)
```

### Parameters

Parameters are global constants used in transition rate formulas:

```python
builder.add_parameter(id="beta", value=0.3, description="Transmission rate")
builder.add_parameter(id="gamma", value=0.1, description="Recovery rate")
builder.add_parameter(id="N", value=1000.0, description="Total population")
```

### Transitions

Transitions define how populations move between states:

```python
# Simple parameter-based rate
builder.add_transition(
    id="recovery",
    source=["I"],
    target=["R"], 
    rate="gamma"
)

# Mathematical formula
builder.add_transition(
    id="infection",
    source=["S"],
    target=["I"],
    rate="beta * S * I / N"
)

# Constant rate
builder.add_transition(
    id="birth",
    source=["R"],
    target=["S"],
    rate="0.001"  # 0.1% daily birth rate
)
```

## Mathematical Expressions

EpiModel supports rich mathematical expressions in transition rates:

### Basic Operations
```python
rate="beta * S * I"           # Multiplication
rate="gamma + delta"          # Addition  
rate="alpha / N"              # Division
rate="(beta + gamma) * I"     # Parentheses
```

### Mathematical Functions
```python
rate="beta * sin(2 * 3.14159 * step / 365)"  # Seasonal variation
rate="gamma * exp(-0.01 * step)"             # Exponential decay
rate="0.1 * sqrt(I)"                         # Square root
rate="max(0, beta - 0.001 * step)"           # Maximum function
```

### Available Functions
- Variables: `pi`, `e`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- Exponential/Logarithmic: `exp`, `log`, `ln`, `log2`, `log10`
- Power/Root: `sqrt`, `pow`
- Rounding: `floor`, `ceil`, `round`
- Comparison: `max`, `min`
- Other: `abs`, `sinh`, `cosh`, `tanh`

### Population Expression
Use the `N` variable for refering to the total population of the model.
```python
"beta * S * I / N"
```

### Time-Dependent Expressions
Use the `step` variable for time-dependent rates:
```python
# Seasonal transmission
rate="beta * (1 + 0.2 * sin(2 * 3.14159 * step / 365))"

# Declining effectiveness over time
rate="gamma * exp(-0.01 * step)"
```

### Healthcare Saturation
Model healthcare system capacity:
```python
# Recovery rate decreases as infections increase
rate="gamma * (1.0 - I / (2.0 * N))"
```

## Advanced Features

### Formula Transitions
Use the dedicated method for complex mathematical expressions:

```python
builder.add_formula_transition(
    id="seasonal_infection",
    source=["S"],
    target=["I"],
    formula="beta * S * I / N * (1 + 0.3 * sin(step * 2 * 3.14159 / 365))"
)
```

### Constant Rate Transitions
For simple constant rates:

```python
builder.add_constant_rate_transition(
    id="natural_death",
    source=["S", "I", "R"],  # Can affect multiple compartments
    target=[],  # Empty target means removal from system
    rate=0.000027  # Daily natural death rate
)
```

### Conditional Transitions
Add conditions to transitions (advanced feature):

```python
condition = builder.create_condition(
    logic="and",
    rules=[
        {"variable": "state:I", "operator": "gt", "value": 100},
        {"variable": "step", "operator": "gt", "value": 30}
    ]
)

builder.add_transition(
    id="lockdown_effect",
    source=["S"],
    target=["S"],  # No state change, just rate modification
    rate="0.5 * beta",  # Reduced transmission
    condition=condition
)
```

## Loading Models from Files

Load pre-defined models from JSON:

```python
from epimodel import ModelLoader

model = ModelLoader.from_json("path/to/model.json")
simulation = Simulation(model)
results = simulation.run(num_steps=365)
```

## Simulation and Results

### Running Simulations

```python
simulation = Simulation(model)

# Run for specific number of steps
results = simulation.run(num_steps=100)
```

### Output Formats

**Dictionary of Lists (default)**:
```python
results = simulation.run(num_steps=100, output_format="dict_of_lists")
# Results: {"S": [990, 980, ...], "I": [10, 20, ...], "R": [0, 0, ...]}

# Easy plotting
import matplotlib.pyplot as plt
plt.plot(results["S"], label="Susceptible")
plt.plot(results["I"], label="Infected") 
plt.plot(results["R"], label="Recovered")
plt.legend()
```

**List of Lists**:
```python
results = simulation.run(num_steps=100, output_format="list_of_lists")
# Results: [[990, 10, 0], [980, 20, 0], ...]

# Access specific time points
initial_state = results[0]    # [990, 10, 0]
final_state = results[-1]     # [340, 50, 610]
```

## Model Validation

EpiModel provides comprehensive validation:

```python
# Validation happens automatically during build()
try:
    model = builder.build(ModelTypes.DIFFERENCE_EQUATIONS)
except ValueError as e:
    print(f"Validation error: {e}")
```

### Common Validation Checks
- Disease state fractions must sum to 1.0
- Stratification fractions must sum to 1.0 for each stratification
- Transition sources/targets must reference valid states
- Mathematical expressions must be syntactically correct
- Security validation prevents code injection

## Security Features

EpiModel includes built-in security validation for mathematical expressions:

```python
# Safe expressions
rate="beta * S * I"  # ✓ Valid
rate="sin(step)"     # ✓ Valid

# Dangerous expressions (automatically rejected)
rate="__import__('os').system('rm -rf /')"  # ✗ Blocked
rate="eval('malicious_code')"               # ✗ Blocked
```

## Performance Tips

1. **Batch Processing**: Run longer simulations rather than many short ones
2. **Simple Expressions**: Complex mathematical expressions are slower than parameter references

## Model Structure

A complete model consists of:

```python
model = Model(
    name="Model Name",
    description="Optional description", 
    version="1.0.0",
    population=Population(
        disease_states=[...],
        stratifications=[...],
        transitions=[...],
        initial_conditions=InitialConditions(...)
    ),
    parameters=[...],
    dynamics=Dynamics(
        typology=ModelTypes.DIFFERENCE_EQUATIONS,
        transitions=[...]
    )
)
```

## Examples

### SEIR Model
```python
model = (
    ModelBuilder(name="SEIR Model")
    .add_disease_state("S", "Susceptible")
    .add_disease_state("E", "Exposed") 
    .add_disease_state("I", "Infected")
    .add_disease_state("R", "Recovered")
    .add_parameter("beta", 0.4)     # Transmission rate
    .add_parameter("sigma", 0.2)    # Incubation rate (1/incubation_period)
    .add_parameter("gamma", 0.1)    # Recovery rate (1/infectious_period)
    .add_parameter("N", 1000.0)
    .add_transition("exposure", ["S"], ["E"], "beta * S * I / N")
    .add_transition("infection", ["E"], ["I"], "sigma")
    .add_transition("recovery", ["I"], ["R"], "gamma")
    .set_initial_conditions(
        population_size=1000,
        disease_state_fractions={"S": 0.999, "E": 0.0, "I": 0.001, "R": 0.0}
    )
    .build(ModelTypes.DIFFERENCE_EQUATIONS)
)
```

### Age-Stratified Model
```python
model = (
    ModelBuilder(name="Age-Stratified SIR")
    .add_disease_state("S", "Susceptible")
    .add_disease_state("I", "Infected")
    .add_disease_state("R", "Recovered")
    .add_stratification("age", ["child", "adult", "elderly"])
    .add_parameter("beta_cc", 0.3)  # Child-to-child transmission
    .add_parameter("beta_ca", 0.2)  # Child-to-adult transmission
    .add_parameter("beta_aa", 0.25) # Adult-to-adult transmission
    .add_parameter("gamma", 0.1)
    .add_parameter("N", 10000.0)
    # Add age-specific transitions...
    .set_initial_conditions(
        population_size=10000,
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0},
        stratification_fractions={
            "age": {"child": 0.2, "adult": 0.6, "elderly": 0.2}
        }
    )
    .build(ModelTypes.DIFFERENCE_EQUATIONS)
)
```

## Contributing

EpiModel is built with Rust and Python. To contribute:

1. Install Rust toolchain
2. Install Python development dependencies
3. Build the project: `maturin develop --release`
4. Run tests: `pytest`
