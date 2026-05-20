# Commol

A high-performance compartmental modelling library for mathematical modeling using difference equations. Commol provides a clean Python API backed by a fast Rust engine for numerical computations.

!!! warning "Alpha Stage - API Unstable"
**Commol is currently in alpha development.** The API is not yet stable and may change between versions without backward compatibility guarantees.

    - Breaking changes may occur in any release
    - Not recommended for production use yet
    - API will stabilize in version 1.0.0

    We welcome feedback and contributions as we work toward a stable release!

## Features

- **Intuitive Model Building**: Fluent API for constructing compartment models
- **Mathematical Expressions**: Support for complex mathematical formulas in transition rates
- **High Performance**: Rust-powered simulation engine for fast computations
- **Flexible Architecture**: Support for stratified populations and conditional transitions
- **Type Safety**: Comprehensive validation using Pydantic models
- **Multiple Output Formats**: Get results as dictionaries or lists for easy analysis

## Quick Example

```python
from commol import ModelBuilder, Simulation

# Build a simple 3-compartment model
model = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(
        id="t_ab",
        source=["A"],
        target=["B"],
        rate="k1 * A * B / N"
    )
    .add_transition(
        id="t_bc",
        source=["B"],
        target=["C"],
        rate="k2"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0}
        ]
    )
    .build(typology="DifferenceEquations")
)

# Run simulation
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Getting Started

- [Installation Guide](getting-started/installation.md) - Install Commol
- [Quick Start](getting-started/quickstart.md) - Build your first model
- [Core Concepts](guide/core-concepts.md) - Understand the fundamentals

## Documentation Sections

### User Guide

Learn how to build and run compartment models:

- [Core Concepts](guide/core-concepts.md) - Compartments, stratifications, parameters
- [Building Models](guide/building-models.md) - Using the ModelBuilder API
- [Mathematical Expressions](guide/mathematical-expressions.md) - Advanced rate formulas
- [Running Simulations](guide/simulations.md) - Execute models and analyze results
- [Model Calibration](guide/calibration.md) - Fit models to observed data
- [Examples](guide/examples.md) - Complete examples (basic, stratified, conditional models)

### API Reference

Complete API documentation for all public interfaces:

- [Model Builder](api/model-builder.md) - ModelBuilder class documentation
- [Simulation](api/simulation.md) - Simulation runner
- [Calibrator](api/calibrator.md) - Parameter calibration and optimization

### Development

Contributing to Commol:

- [Development Workflow](development/workflow.md) - Setup, branching, CI/CD
- [Contributing Guidelines](development/contributing.md) - How to contribute
- [Release Process](development/release.md) - Version management and releases

## License

Commol is licensed under the MIT License. See [License](about/license.md) for details.

## Authors

MUNQU Team

- Rafael J. Villanueva Micó
- Carlos Andreu Vilarroig
- David Martínez Rodríguez
