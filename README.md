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
maturin develop --release
```

## Quick Start

```python
from commol import ModelBuilder, Simulation
from commol.constants import ModelTypes

# Build a simple SIR model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma", value=0.1)
    .add_parameter(id="N", value=1000.0)
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
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Run simulation
simulation = Simulation(model)
results = simulation.run(num_steps=100)

# Display results
print(f"Final infected: {results['I'][-1]:.0f}")
```

### With Unit Checking

Add units to parameters for automatic dimensional validation:

```python
model = (
    ModelBuilder(name="SIR with Units", version="1.0")
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
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Validate dimensional consistency
model.check_unit_consistency()  # Ensures all equations have correct units
```

### Model Calibration

Fit model parameters to observed data using optimization algorithms:

```python
from commol import (
    Calibrator,
    CalibrationProblem,
    CalibrationParameter,
    ObservedDataPoint,
    LossConfig,
    LossFunction,
    OptimizationConfig,
    OptimizationAlgorithm,
    ParticleSwarmConfig,
)

# Define observed data from real outbreak
observed_data = [
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
]

# Specify parameters to calibrate with bounds
parameters = [
    CalibrationParameter(id="beta", min_bound=0.0, max_bound=1.0),
    CalibrationParameter(id="gamma", min_bound=0.0, max_bound=1.0),
]

# Configure calibration problem
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.SSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig(max_iterations=300),
    ),
)

# Run calibration
calibrator = Calibrator(simulation, problem)
result = calibrator.run()

print(f"Calibrated beta: {result.best_parameters['beta']:.4f}")
print(f"Calibrated gamma: {result.best_parameters['gamma']:.4f}")
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

# Install Python dependencies
cd py-commol
poetry install --with dev,docs

# Build Rust workspace
cd ..
cargo build --workspace

# Build Python extension
cd py-commol
maturin develop --release

# Run tests
poetry run pytest
cd ..
cargo test --workspace

# Build documentation locally
cd py-commol
poetry run mkdocs serve
```

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
  author = {Villanueva Micó, Rafael J. and Andreu Vilarroig, Carlos and Martínez Rodríguez, David},
  year = {2025},
  url = {https://github.com/MUNQU/commol}
}
```

## Support

- Documentation: https://munqu.github.io/commol
- Issue Tracker: https://github.com/MUNQU/commol/issues
- Discussions: https://github.com/MUNQU/commol/discussions
