# Quick Start

This guide will help you build and run your first compartment model with Commol.

## Your First Compartment Model

Let's create a basic 3-compartment model where entities flow from state A through B into C:

```python
from commol import ModelBuilder, Simulation

# Build the model
model = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)   # Rate of A→B transition
    .add_parameter(id="k2", value=0.1)   # Rate of B→C transition
    .add_transition(
        id="t_ab",
        source=["A"],
        target=["B"],
        rate="k1 * A * B / N"  # Mathematical formula
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

# Display results
print(f"State A at step 100: {results['A'][-1]:.0f}")
print(f"State B at step 100: {results['B'][-1]:.0f}")
print(f"State C at step 100: {results['C'][-1]:.0f}")
```

## Understanding the Code

### 1. Import Required Classes

```python
from commol import ModelBuilder, Simulation
```

- `ModelBuilder`: Fluent API for constructing models
- `Simulation`: Runs the model simulation

### 2. Define Compartments

```python
.add_bin(id="A", name="State A")
.add_bin(id="B", name="State B")
.add_bin(id="C", name="State C")
```

Compartments (also called bins or states) represent the different states in your model.

### 3. Add Parameters

```python
.add_parameter(id="k1", value=0.3)   # Rate for A→B
.add_parameter(id="k2", value=0.1)   # Rate for B→C
```

Parameters are constants used in transition rate formulas.

### 4. Define Transitions

```python
.add_transition(
    id="t_ab",
    source=["A"],
    target=["B"],
    rate="k1 * A * B / N"
)
```

Transitions move populations between states using mathematical formulas.

### 5. Set Initial Conditions

```python
.set_initial_conditions(
    population_size=1000,
    bin_fractions=[
        {"bin": "A", "fraction": 0.99},
        {"bin": "B", "fraction": 0.01},
        {"bin": "C", "fraction": 0.0}
    ]
)
```

Define the starting population distribution.

### 6. Build and Run

```python
model = builder.build(typology="DifferenceEquations")
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Adding Unit Checking

Improve model safety by adding units to your parameters:

```python
model = (
    ModelBuilder(name="Model with Units", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3, unit="1/step")
    .add_parameter(id="k2", value=0.1, unit="1/step")
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
        rate="k2 * B"
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

# Validate dimensional consistency
model.check_unit_consistency()
```

**Benefits**:

- Catches unit errors before simulation (e.g., mixing different time scales)
- Validates mathematical functions receive correct dimensional arguments
- Documents the physical meaning of parameters

See the [Unit Checking](../guide/building-models.md#unit-checking) section for details.

## Calibrating Model Parameters

When parameters are unknown, set them to `None` and calibrate them to match observed data:

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
calibration_model = (
    ModelBuilder(name="Calibration Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)   # To be calibrated
    .add_parameter(id="k2", value=None)   # To be calibrated
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
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

# Observed values of compartment B at different time steps
observed_data = [
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
    ObservedDataPoint(step=30, compartment="B", value=62.3),
    ObservedDataPoint(step=40, compartment="B", value=38.1),
]

# Create simulation (allowed with None values for calibration)
cal_simulation = Simulation(calibration_model)

# Define parameters to calibrate with bounds and initial guesses
parameters = [
    CalibrationParameter(
        id="k1",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3
    ),
    CalibrationParameter(
        id="k2",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.1
    ),
]

# Configure the calibration problem
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=ParticleSwarmConfig(
        num_particles=30,
        max_iterations=300,
        verbose=False
    ),
)

# Run calibration
calibrator = Calibrator(cal_simulation, problem)
result = calibrator.run()

# Display and apply calibrated parameters
print(f"Calibrated k1: {result.best_parameters['k1']:.4f}")
print(f"Calibrated k2: {result.best_parameters['k2']:.4f}")
print(f"Final loss: {result.final_loss:.6f}")

# Update model with calibrated values
calibration_model.update_parameters(result.best_parameters)

# Now run simulation with calibrated model
final_sim = Simulation(calibration_model)
final_results = final_sim.run(num_steps=100)
```

**Key concepts**:

- `ObservedDataPoint`: Measurements to fit against
- `CalibrationParameter`: Parameters to optimize with bounds
- `loss_function`: How to measure fit quality ("sse", "rmse", "mae", etc.)
- `optimization_config`: Optimization algorithm (ParticleSwarmConfig or NelderMeadConfig)

See the [Calibration Guide](../guide/calibration.md) for advanced techniques.

## Next Steps

Now that you've built your first model, explore:

- [Core Concepts](../guide/core-concepts.md) - Deep dive into Commol concepts
- [Building Models](../guide/building-models.md) - Advanced model construction
- [Mathematical Expressions](../guide/mathematical-expressions.md) - Complex rate formulas
- [Model Calibration](../guide/calibration.md) - Comprehensive calibration guide
- [Examples](../guide/examples.md) - More complete examples
