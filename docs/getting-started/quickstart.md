# Quick Start

This guide will help you build and run your first compartment model with Commol.

## Your First SIR Model

Let's create a basic SIR (Susceptible-Infected-Recovered) model:

```python
from commol import ModelBuilder, Simulation
from commol.constants import ModelTypes

# Build the model
model = (
    ModelBuilder(name="Basic SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)   # Transmission rate
    .add_parameter(id="gamma", value=0.1)  # Recovery rate
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
print(f"Susceptible at day 100: {results['S'][-1]:.0f}")
print(f"Infected at day 100: {results['I'][-1]:.0f}")
print(f"Recovered at day 100: {results['R'][-1]:.0f}")
```

## Understanding the Code

### 1. Import Required Classes

```python
from commol import ModelBuilder, Simulation
from commol.constants import ModelTypes
```

- `ModelBuilder`: Fluent API for constructing models
- `Simulation`: Runs the model simulation
- `ModelTypes`: Enumeration of available model types

### 2. Define Disease States

```python
.add_bin(id="S", name="Susceptible")
.add_bin(id="I", name="Infected")
.add_bin(id="R", name="Recovered")
```

Disease states represent compartments in your model.

### 3. Add Parameters

```python
.add_parameter(id="beta", value=0.3)   # Transmission rate
.add_parameter(id="gamma", value=0.1)  # Recovery rate
```

Parameters are constants used in transition rate formulas.

### 4. Define Transitions

```python
.add_transition(
    id="infection",
    source=["S"],
    target=["I"],
    rate="beta * S * I / N"
)
```

Transitions move populations between states using mathematical formulas.

### 5. Set Initial Conditions

```python
.set_initial_conditions(
    population_size=1000,
    bin_fractions={"S": 0.99, "I": 0.01, "R": 0.0}
)
```

Define the starting population distribution.

### 6. Build and Run

```python
model = builder.build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
simulation = Simulation(model)
results = simulation.run(num_steps=100)
```

## Adding Unit Checking

Improve model safety by adding units to your parameters:

```python
model = (
    ModelBuilder(name="SIR with Units", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.5, unit="1/day")    # Rate per day
    .add_parameter(id="gamma", value=0.1, unit="1/day")   # Rate per day
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
model.check_unit_consistency()
```

**Benefits**:

- Catches unit errors before simulation (e.g., mixing days and weeks)
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
    CalibrationParameterType,
    ObservedDataPoint,
    LossConfig,
    LossFunction,
    OptimizationConfig,
    OptimizationAlgorithm,
    ParticleSwarmConfig,
)
from commol.constants import ModelTypes

# Build model with unknown parameters
calibration_model = (
    ModelBuilder(name="SIR Calibration", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=None)   # To be calibrated
    .add_parameter(id="gamma", value=None)  # To be calibrated
    .add_transition(id="infection", source=["S"], target=["I"], rate="beta * S * I / N")
    .add_transition(id="recovery", source=["I"], target=["R"], rate="gamma * I")
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

# Observed infected counts at different time steps
observed_data = [
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
    ObservedDataPoint(step=40, compartment="I", value=38.1),
]

# Create simulation (allowed with None values for calibration)
cal_simulation = Simulation(calibration_model)

# Define parameters to calibrate with bounds and initial guesses
parameters = [
    CalibrationParameter(
        id="beta",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3
    ),
    CalibrationParameter(
        id="gamma",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.1
    ),
]

# Configure the calibration problem
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.SSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig(max_iterations=300, verbose=False),
    ),
)

# Run calibration
calibrator = Calibrator(cal_simulation, problem)
result = calibrator.run()

# Display and apply calibrated parameters
print(f"Calibrated beta: {result.best_parameters['beta']:.4f}")
print(f"Calibrated gamma: {result.best_parameters['gamma']:.4f}")
print(f"Final loss: {result.final_loss:.6f}")

# Update model with calibrated values
calibration_model.update_parameters(result.best_parameters)

# Now run simulation with calibrated model
final_sim = Simulation(calibration_model)
final_results = final_sim.run(num_steps=100)
```

**Key concepts**:

- `ObservedDataPoint`: Real-world measurements to fit against
- `CalibrationParameter`: Parameters to optimize with bounds
- `LossFunction`: How to measure fit quality (SSE, RMSE, MAE, etc.)
- `OptimizationAlgorithm`: Optimization method (Particle Swarm or Nelder-Mead)

See the [Calibration Guide](../guide/calibration.md) for advanced techniques.

## Next Steps

Now that you've built your first model, explore:

- [Core Concepts](../guide/core-concepts.md) - Deep dive into EpiModel concepts
- [Building Models](../guide/building-models.md) - Advanced model construction
- [Mathematical Expressions](../guide/mathematical-expressions.md) - Complex rate formulas
- [Model Calibration](../guide/calibration.md) - Comprehensive calibration guide
- [Examples](../guide/examples.md) - More complete examples
