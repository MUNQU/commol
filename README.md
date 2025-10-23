# EpiModel

A high-performance mathematical epidemiology library for modeling infectious disease spread using difference equations. EpiModel provides a clean Python API backed by a fast Rust engine for numerical computations.

> ⚠️ **Alpha Stage Warning**: EpiModel is currently in alpha development. The API is not yet stable and may change between versions without backward compatibility guarantees. Use in production at your own risk.

## Features

- **Intuitive Model Building**: Fluent API for constructing epidemiological models
- **Mathematical Expressions**: Support for complex mathematical formulas in transition rates (sin, cos, exp, log, etc.)
- **Unit Checking**: Automatic dimensional analysis to catch unit errors before simulation
- **High Performance**: Rust-powered simulation engine for fast computations
- **Flexible Architecture**: Support for stratified populations and conditional transitions
- **Type Safety**: Comprehensive validation using Pydantic models
- **Multiple Output Formats**: Get results as dictionaries or lists for easy analysis

## Installation

```bash
# Install from PyPI (once published)
pip install epimodel

# Or install from source
git clone https://github.com/MUNQU/epimodel.git
cd epimodel/py-epimodel
pip install maturin
maturin develop --release
```

## Quick Start

```python
from epimodel import ModelBuilder, Simulation
from epimodel.constants import ModelTypes

# Build a simple SIR model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
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
    .add_disease_state(id="S", name="Susceptible")
    .add_disease_state(id="I", name="Infected")
    .add_disease_state(id="R", name="Recovered")
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
        disease_state_fractions=[
            {"disease_state": "S", "fraction": 0.99},
            {"disease_state": "I", "fraction": 0.01},
            {"disease_state": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Validate dimensional consistency
model.check_unit_consistency()  # Ensures all equations have correct units
```

## Documentation

**[Full Documentation](https://munqu.github.io/epimodel)**

- [Installation Guide](https://munqu.github.io/epimodel/getting-started/installation/) - Setup and installation
- [Quick Start](https://munqu.github.io/epimodel/getting-started/quickstart/) - Build your first model
- [User Guide](https://munqu.github.io/epimodel/guide/core-concepts/) - Core concepts and tutorials
- [API Reference](https://munqu.github.io/epimodel/api/model-builder/) - Complete API documentation
- [Examples](https://munqu.github.io/epimodel/guide/examples/) - SIR, SEIR, and advanced models

## Development

For contributors and developers:

- [Development Workflow](https://munqu.github.io/epimodel/development/workflow/) - Setup, branching, CI/CD
- [Contributing Guidelines](https://munqu.github.io/epimodel/development/contributing/) - How to contribute
- [Release Process](https://munqu.github.io/epimodel/development/release/) - Version management

### Local Development

```bash
# Clone repository
git clone https://github.com/MUNQU/epimodel.git
cd epimodel

# Install Python dependencies
cd py-epimodel
poetry install --with dev,docs

# Build Rust workspace
cd ..
cargo build --workspace

# Build Python extension
cd py-epimodel
maturin develop --release

# Run tests
poetry run pytest
cd ..
cargo test --workspace

# Build documentation locally
cd py-epimodel
poetry run mkdocs serve
```

## License

EpiModel is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- Rafael J. Villanueva Micó - [rjvillan@imm.upv.es](mailto:rjvillan@imm.upv.es)
- Carlos Andreu Vilarroig - [caranvi1@upv.es](mailto:caranvi1@upv.es)
- David Martínez Rodríguez - [damarro3@upv.es](mailto:damarro3@upv.es)

## Citation

If you use EpiModel in your research, please cite:

```bibtex
@software{epimodel2025,
  title = {EpiModel: A High-Performance Epidemiological Modeling Library},
  author = {Villanueva Micó, Rafael J. and Andreu Vilarroig, Carlos and Martínez Rodríguez, David},
  year = {2025},
  url = {https://github.com/MUNQU/epimodel}
}
```

## Support

- Documentation: https://munqu.github.io/epimodel
- Issue Tracker: https://github.com/MUNQU/epimodel/issues
- Discussions: https://github.com/MUNQU/epimodel/discussions
