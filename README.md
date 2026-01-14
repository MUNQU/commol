# Commol

A high-performance compartment modelling library for mathematical modeling using difference equations. Commol provides a clean Python API backed by a fast Rust engine for numerical computations.

> ⚠️ **Alpha Stage Warning**: Commol is currently in alpha development. The API is not yet stable and may change between versions without backward compatibility guarantees. Use in production at your own risk.

## Features

- **Intuitive Model Building**: Fluent API for constructing compartment models
- **Mathematical Expressions**: Support for complex mathematical formulas in transition rates (sin, cos, exp, log, etc.)
- **Unit Checking**: Automatic dimensional analysis to catch unit errors before simulation
- **High Performance**: Rust-powered simulation engine for fast computations
- **Flexible Architecture**: Support for stratified populations and conditional transitions
- **Type Safety**: Comprehensive validation using Pydantic models
- **Multiple Output Formats**: Get results as dictionaries or lists for easy analysis

## Installation

```bash
# Install from PyPI (once published)
pip install commol

# Or install from source
git clone https://github.com/MUNQU/commol.git
cd commol/py-commol
pip install maturin

# If using a virtual environment, activate it first
# source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate     # On Windows

maturin develop --release
```

> **⚠️ Important**: The project directory path must not contain tildes (`~`) or spaces. Maturin may fail with paths like `~/projects/commol` or `/home/my projects/commol`. Use full paths like `/home/username/projects/commol` instead.

## Quick Start

```python
from commol import ModelBuilder, Simulation

# Build a simple SIR model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="gamma"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology="DifferenceEquations")
)

# Run simulation
simulation = Simulation(model)
results = simulation.run(num_steps=100)

# Display results
print(f"Final infected: {results['I'][-1]:.0f}")

# Visualize results
from commol import SimulationPlotter

plotter = SimulationPlotter(simulation, results)
plotter.plot_series(output_file="sir_model.png")
```

### Using $compartment Placeholder for Multiple Transitions

When you need to apply the same transition to multiple compartments (like death rates), use the `$compartment` placeholder instead of writing repetitive code:

```python
model = (
    ModelBuilder(name="SLIR with Deaths", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="L", name="Latent")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3, unit="1/day")
    .add_parameter(id="gamma", value=0.2, unit="1/day")
    .add_parameter(id="delta", value=0.1, unit="1/day")
    .add_parameter(id="d", value=0.01, unit="1/day")  # Death rate
    .add_transition(
        id="infection",
        source=["S"],
        target=["L"],
        rate="beta * S * I / N"
    )
    .add_transition(
        id="progression",
        source=["L"],
        target=["I"],
        rate="gamma * L"
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="delta * I"
    )
    # Single transition automatically expands to 4 separate death transitions
    .add_transition(
        id="death",
        source=["S", "L", "I", "R"],
        target=[],
        rate="d * $compartment"  # Expands to: d*S, d*L, d*I, d*R
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "L", "fraction": 0.005},
            {"bin": "I", "fraction": 0.005},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology="DifferenceEquations")
)
```

**The `$compartment` placeholder:**

- Automatically expands to multiple transitions (one per source compartment)
- Replaces `$compartment` with the actual compartment name in the rate formula
- Works with stratified rates for age-structured or location-based models
- Reduces code duplication and improves maintainability

**Example with stratified rates:**

```python
.add_transition(
    id="death",
    source=["S", "I", "R"],
    target=[],
    rate="d_base * $compartment",  # Fallback rate
    stratified_rates=[
        {
            "conditions": [{"stratification": "age", "category": "young"}],
            "rate": "d_young * $compartment"  # Lower death rate for young
        },
        {
            "conditions": [{"stratification": "age", "category": "old"}],
            "rate": "d_old * $compartment"  # Higher death rate for old
        }
    ]
)
```

### With Unit Checking

Add units to parameters and bins for automatic dimensional validation and annotated equation display:

```python
model = (
    ModelBuilder(name="SIR with Units", version="1.0", bin_unit="person")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.5, unit="1/day")  # Rate with units
    .add_parameter(id="gamma", value=0.1, unit="1/day")
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="gamma * I"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology="DifferenceEquations")
)

# Validate dimensional consistency
model.check_unit_consistency()  # Ensures all equations have correct units

# Print equations with unit annotations
model.print_equations()
# Output shows:
#   S -> I: beta(1/day) * S(person) * I(person) / N(person) [person/day]
#   I -> R: gamma(1/day) * I(person) [person/day]

# Export equations in LaTeX format for publications
model.print_equations(format="latex")
# Output: \[\frac{dS}{dt} = - (\beta \cdot S \cdot I / N)\]
```

**Note**: Units must be defined for ALL parameters and bins, or for NONE. Partial unit definitions will raise a `ValueError` to prevent inconsistent models.

### Model Calibration

Fit model parameters to observed data using optimization algorithms. Parameters to be calibrated should be set to `None`:

```python
from commol import (
    ModelBuilder,
    Simulation,
    Calibrator,
    CalibrationProblem,
    CalibrationParameter,
    ObservedDataPoint,
    ParticleSwarmConfig,
)

# Build model with unknown parameters
model = (
    ModelBuilder(name="SIR Model", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=None)   # To be calibrated
    .add_parameter(id="gamma", value=None)  # To be calibrated
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="gamma * I"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology="DifferenceEquations")
)

# Define observed data from real outbreak
observed_data = [
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
]

# Simulation can be created with None values for calibration
simulation = Simulation(model)

# Specify parameters to calibrate with bounds and initial guesses
parameters = [
    CalibrationParameter(
        id="beta",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3  # Starting point
    ),
    CalibrationParameter(
        id="gamma",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
    ),
]

# Configure optimization algorithm (config type determines the algorithm)
pso_config = ParticleSwarmConfig(
    num_particles=40,
    max_iterations=300,
    verbose=True
)

# Configure calibration problem
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=pso_config,  # ParticleSwarmConfig or NelderMeadConfig
)

# Run calibration
calibrator = Calibrator(simulation, problem)
result = calibrator.run()

print(f"Calibrated beta: {result.best_parameters['beta']:.4f}")
print(f"Calibrated gamma: {result.best_parameters['gamma']:.4f}")

# Update model with calibrated parameters
model.update_parameters(result.best_parameters)

# Create new simulation with calibrated model
calibrated_simulation = Simulation(model)
calibrated_results = calibrated_simulation.run(num_steps=100)
```

**Calibrating with Scale Parameters:**

When observed data is underreported, use scale parameters to estimate the reporting rate:

```python
# Reported cases (potentially underreported)
reported_cases = [10, 15, 25, 40, 60, 75, 85, 70, 50, 30]

# Link observed data to scale parameter
observed_data = [
    ObservedDataPoint(
        step=idx,
        compartment="I",
        value=cases,
        scale_id="reporting_rate"  # Links to scale parameter
    )
    for idx, cases in enumerate(reported_cases)
]

parameters = [
    CalibrationParameter(
        id="beta",
        parameter_type="parameter",
        min_bound=0.1,
        max_bound=1.0
    ),
    CalibrationParameter(
        id="gamma",
        parameter_type="parameter",
        min_bound=0.05,
        max_bound=0.5
    ),
    CalibrationParameter(
        id="reporting_rate",
        parameter_type="scale",
        min_bound=0.01,
        max_bound=1.0
    ),
]

# Run calibration
result = calibrator.run()

# Separate parameters by type
scale_values = {
    param.id: result.best_parameters[param.id]
    for param in problem.parameters
    if param.parameter_type == "scale"
}

print(f"Calibrated reporting rate: {scale_values['reporting_rate']:.2%}")

# Visualize with scale_values for correct display
plotter.plot_series(observed_data=observed_data, scale_values=scale_values)
```

**Constraining Parameters:**

Apply constraints to enforce biological knowledge during calibration.

```python
from commol import CalibrationConstraint

# Add constraint: beta/gamma <= 5 (written as 5 - beta/gamma >= 0)
constraints = [
    CalibrationConstraint(
        id="r0_bound",
        expression="5.0 - beta/gamma",
        description="R0 <= 5",
    )
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    constraints=constraints,  # Include constraints
    loss_function="sse",
    optimization_config=pso_config,
)

result = calibrator.run()
```

**Probabilistic Calibration:**

For uncertainty quantification, use probabilistic calibration to get an ensemble of parameter sets:

```python
from commol import ProbabilisticCalibrationConfig

# Configure probabilistic calibration
prob_config = ProbabilisticCalibrationConfig(
    n_runs=20,  # Number of calibration runs
    confidence_level=0.95
)

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=pso_config,
    probabilistic_config=prob_config,  # Enable probabilistic mode
)

# Run probabilistic calibration
calibrator = Calibrator(simulation, problem)
prob_result = calibrator.run_probabilistic()
```

## Documentation

**[Full Documentation](https://munqu.github.io/commol)**

- [Installation Guide](https://munqu.github.io/commol/getting-started/installation/) - Setup and installation
- [Quick Start](https://munqu.github.io/commol/getting-started/quickstart/) - Build your first model
- [User Guide](https://munqu.github.io/commol/guide/core-concepts/) - Core concepts and tutorials
- [Model Calibration](https://munqu.github.io/commol/guide/calibration/) - Parameter fitting and optimization
- [API Reference](https://munqu.github.io/commol/api/model-builder/) - Complete API documentation
- [Examples](https://munqu.github.io/commol/guide/examples/) - SIR, SEIR, and advanced models

## Development

For contributors and developers:

- [Development Workflow](https://munqu.github.io/commol/development/workflow/) - Setup, branching, CI/CD
- [Contributing Guidelines](https://munqu.github.io/commol/development/contributing/) - How to contribute
- [Release Process](https://munqu.github.io/commol/development/release/) - Version management

### Local Development

```bash
# Clone repository
git clone https://github.com/MUNQU/commol.git
cd commol

# Create and activate virtual environment
python -m venv venv

source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate   # On Windows

# Install Python dependencies
cd py-commol
pip install -e ".[dev,docs]"

# Build Rust workspace
cd ..
cargo build --workspace

# Build Python extension (with virtual environment activated)
cd py-commol
maturin develop --release

# Run tests
pytest
cd ..
cargo test --workspace

# Build documentation locally
cd py-commol
mkdocs serve
```

> **⚠️ Path Requirements**: Ensure the project path contains no tildes (`~`) or spaces. Maturin may fail otherwise.
>
> **💡 Tip**: Make sure your virtual environment is activated before running `maturin develop`.

## License

Commol is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- Rafael J. Villanueva Micó - [rjvillan@imm.upv.es](mailto:rjvillan@imm.upv.es)
- Carlos Andreu Vilarroig - [caranvi1@upv.es](mailto:caranvi1@upv.es)
- David Martínez Rodríguez - [damarro3@upv.es](mailto:damarro3@upv.es)

## Citation

If you use Commol in your research, please cite:

```bibtex
@software{commol2025,
  title = {Commol: A High-Performance Compartment Modelling Library},
  author = {
    Villanueva Micó, Rafael J.
    and Andreu Vilarroig, Carlos
    and Martínez Rodríguez, David
  },
  year = {2025},
  url = {https://github.com/MUNQU/commol}
}
```

## Support

- Documentation: https://munqu.github.io/commol
- Issue Tracker: https://github.com/MUNQU/commol/issues
- Discussions: https://github.com/MUNQU/commol/discussions
