# EpiModel

A high-performance mathematical epidemiology library for modeling infectious disease spread using difference equations. EpiModel provides a clean Python API backed by a fast Rust engine for numerical computations.

!!! warning "Alpha Stage - API Unstable"
**EpiModel is currently in alpha development.** The API is not yet stable and may change between versions without backward compatibility guarantees.

    - Breaking changes may occur in any release
    - Not recommended for production use yet
    - API will stabilize in version 1.0.0

    We welcome feedback and contributions as we work toward a stable release!

## Features

- **Intuitive Model Building**: Fluent API for constructing epidemiological models
- **Mathematical Expressions**: Support for complex mathematical formulas in transition rates
- **High Performance**: Rust-powered simulation engine for fast computations
- **Flexible Architecture**: Support for stratified populations and conditional transitions
- **Type Safety**: Comprehensive validation using Pydantic models
- **Multiple Output Formats**: Get results as dictionaries or lists for easy analysis

## Quick Example

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
        disease_state_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Run simulation
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Getting Started

- [Installation Guide](getting-started/installation.md) - Install EpiModel
- [Quick Start](getting-started/quickstart.md) - Build your first model
- [Core Concepts](guide/core-concepts.md) - Understand the fundamentals

## Documentation Sections

### User Guide

Learn how to build and run epidemiological models:

- [Core Concepts](guide/core-concepts.md) - Disease states, stratifications, parameters
- [Building Models](guide/building-models.md) - Using the ModelBuilder API
- [Mathematical Expressions](guide/mathematical-expressions.md) - Advanced rate formulas
- [Running Simulations](guide/simulations.md) - Execute models and analyze results
- [Examples](guide/examples.md) - Complete model examples (SIR, SEIR, stratified)

### API Reference

Complete API documentation for all public interfaces:

- [Model Builder](api/model-builder.md) - ModelBuilder class documentation
- [Models](api/models.md) - Model and population structures
- [Simulation](api/simulation.md) - Simulation runner
- [Constants](api/constants.md) - Model types and enumerations

### Development

Contributing to EpiModel:

- [Development Workflow](development/workflow.md) - Setup, branching, CI/CD
- [Contributing Guidelines](development/contributing.md) - How to contribute
- [Release Process](development/release.md) - Version management and releases

## License

EpiModel is licensed under the MIT License. See [License](about/license.md) for details.

## Authors

MUNQU Team

- Rafael J. Villanueva Micó
- Carlos Andreu Vilarroig
- David Martínez Rodríguez
