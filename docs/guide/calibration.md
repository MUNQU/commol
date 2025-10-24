# Model Calibration

Calibration is the process of adjusting model parameters to match observed data. EpiModel provides a powerful calibration framework that uses optimization algorithms to find the best parameter values that minimize the difference between model predictions and real-world observations.

## Overview

The calibration process involves:

1. **Defining observed data**: Real-world measurements at specific time steps
2. **Selecting parameters to calibrate**: Which model parameters to optimize
3. **Choosing a loss function**: How to measure fit quality (SSE, RMSE, MAE, etc.)
4. **Selecting an optimization algorithm**: Nelder-Mead or Particle Swarm
5. **Running the calibration**: Finding optimal parameter values
6. **Creating a new model**: Rebuilding your model with the calibrated parameters

!!! note "Calibrator Returns Parameter Values Only"
The `Calibrator.run()` method returns a `CalibrationResult` object containing the optimized parameter values, but **does not** return or modify your model. To use the calibrated parameters, you must create a new model using `ModelBuilder` with the fitted values.

## Basic Example

Here's a simple calibration of an SIR model's transmission and recovery rates:

```python
from commol import (
    ModelBuilder,
    Simulation,
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
from commol.constants import ModelTypes

# Build the model
model = (
    ModelBuilder(name="SIR Model", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=0.3)   # Initial guess
    .add_parameter(id="gamma", value=0.1)  # Initial guess
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

# Define observed data (e.g., from real outbreak data)
observed_data = [
    ObservedDataPoint(step=0, compartment="I", value=10.0),
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
    ObservedDataPoint(step=40, compartment="I", value=38.1),
    ObservedDataPoint(step=50, compartment="I", value=18.7),
]

# Define parameters to calibrate with bounds
parameters = [
    CalibrationParameter(
        id="beta",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3  # Optional starting point
    ),
    CalibrationParameter(
        id="gamma",
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
        config=ParticleSwarmConfig(
            num_particles=30,
            max_iterations=500,
            verbose=True  # Show optimization progress
        ),
    ),
)

# Run calibration
simulation = Simulation(model)
calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Display results
print(result)
print(f"Calibrated beta: {result.best_parameters['beta']:.4f}")
print(f"Calibrated gamma: {result.best_parameters['gamma']:.4f}")
```

## Loss Functions

EpiModel supports multiple loss functions for measuring fit quality:

### Sum of Squared Errors (SSE)

Default loss function. Minimizes the sum of squared differences:

```python
loss_config = LossConfig(function=LossFunction.SSE)
```

$$\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)

Provides error in the same units as the data:

```python
loss_config = LossConfig(function=LossFunction.RMSE)
```

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Mean Absolute Error (MAE)

Less sensitive to outliers:

```python
loss_config = LossConfig(function=LossFunction.MAE)
```

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Weighted SSE

Allows different importance for different observations:

```python
# Give more weight to early observations
observed_data = [
    ObservedDataPoint(step=0, compartment="I", value=10.0, weight=2.0),
    ObservedDataPoint(step=10, compartment="I", value=45.2, weight=1.5),
    ObservedDataPoint(step=20, compartment="I", value=78.5, weight=1.0),
]

loss_config = LossConfig(function=LossFunction.WEIGHTED_SSE)
```

$$\text{WSSE} = \sum_{i=1}^{n} w_i (y_i - \hat{y}_i)^2$$

## Optimization Algorithms

### Particle Swarm Optimization (PSO)

Population-based algorithm inspired by social behavior. Good for global optimization:

```python
from commol import ParticleSwarmConfig, OptimizationConfig, OptimizationAlgorithm

optimization_config = OptimizationConfig(
    algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
    config=ParticleSwarmConfig(
        num_particles=40,        # Number of particles in swarm
        max_iterations=1000,     # Maximum iterations
        target_cost=0.01,        # Stop if loss below this (optional)
        inertia_factor=0.7,      # Velocity inertia (optional)
        cognitive_factor=1.5,    # Attraction to personal best (optional)
        social_factor=1.5,       # Attraction to global best (optional)
        verbose=True,            # Print progress
        header_interval=50       # Print header every N iterations
    ),
)
```

**When to use**:

- Multiple local minima expected
- Robust global search needed
- Multiple parameters to calibrate

### Nelder-Mead Simplex

Derivative-free simplex algorithm. Fast convergence for smooth problems:

```python
from commol import NelderMeadConfig, OptimizationConfig, OptimizationAlgorithm

optimization_config = OptimizationConfig(
    algorithm=OptimizationAlgorithm.NELDER_MEAD,
    config=NelderMeadConfig(
        max_iterations=1000,     # Maximum iterations
        sd_tolerance=1e-6,       # Convergence tolerance
        alpha=1.0,               # Reflection coefficient (optional)
        gamma=2.0,               # Expansion coefficient (optional)
        rho=0.5,                 # Contraction coefficient (optional)
        sigma=0.5,               # Shrink coefficient (optional)
        verbose=True,            # Print progress
        header_interval=50       # Print header every N iterations
    ),
)
```

**When to use**:

- Smooth, unimodal objective function
- Few parameters (< 10)
- Fast convergence desired

## Advanced Examples

### Multi-Compartment Calibration

Calibrate against observations from multiple compartments:

```python
# Observed data from both infected and recovered compartments
observed_data = [
    # Infected observations
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
    # Recovered observations
    ObservedDataPoint(step=10, compartment="R", value=12.5),
    ObservedDataPoint(step=20, compartment="R", value=35.8),
    ObservedDataPoint(step=30, compartment="R", value=68.2),
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="beta", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="gamma", min_bound=0.0, max_bound=1.0),
    ],
    loss_config=LossConfig(function=LossFunction.RMSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig(max_iterations=500, verbose=False),
    ),
)
```

### Calibrating Stratified Models

Calibrate parameters specific to age groups or other stratifications:

```python
# Build age-stratified model
model = (
    ModelBuilder(name="Age-Stratified SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_stratification(id="age", categories=["young", "old"])
    .add_parameter(id="beta", value=0.3)
    .add_parameter(id="gamma_young", value=0.12)
    .add_parameter(id="gamma_old", value=0.08)
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
        stratified_rates=[
            {
                "conditions": [{"stratification": "age", "category": "young"}],
                "rate": "gamma_young * I"
            },
            {
                "conditions": [{"stratification": "age", "category": "old"}],
                "rate": "gamma_old * I"
            },
        ]
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ],
        stratification_fractions=[
            {
                "stratification": "age",
                "fractions": [
                    {"category": "young", "fraction": 0.6},
                    {"category": "old", "fraction": 0.4}
                ]
            }
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Calibrate stratification-specific parameters
parameters = [
    CalibrationParameter(id="beta", min_bound=0.0, max_bound=1.0),
    CalibrationParameter(id="gamma_young", min_bound=0.05, max_bound=0.2),
    CalibrationParameter(id="gamma_old", min_bound=0.05, max_bound=0.2),
]

# Note: Observed data uses stratified compartment names like "I_young", "I_old"
observed_data = [
    ObservedDataPoint(step=10, compartment="I_young", value=30.5),
    ObservedDataPoint(step=10, compartment="I_old", value=15.2),
    ObservedDataPoint(step=20, compartment="I_young", value=52.3),
    ObservedDataPoint(step=20, compartment="I_old", value=26.1),
]
```

### Using Calibration Results

Once calibrated, you need to create a new model with the fitted parameters to run predictions:

```python
# Run calibration
result = Calibrator(simulation, problem).run()

# Check if calibration converged
if result.converged:
    print(f"Calibration converged after {result.iterations} iterations")
    print(f"Final loss: {result.final_loss:.6f}")
    print(f"Calibrated parameters: {result.best_parameters}")

    # Create a new model with the calibrated parameters
    # You need to rebuild the model with the new parameter values
    calibrated_model = (
        ModelBuilder(name="Calibrated SIR", version="1.0")
        .add_bin(id="S", name="Susceptible")
        .add_bin(id="I", name="Infected")
        .add_bin(id="R", name="Recovered")
        # Use calibrated parameter values
        .add_parameter(id="beta", value=result.best_parameters["beta"])
        .add_parameter(id="gamma", value=result.best_parameters["gamma"])
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

    # Run predictions with calibrated model
    calibrated_simulation = Simulation(calibrated_model)
    prediction_results = calibrated_simulation.run(num_steps=200)

    print(f"Predicted infections at day 200: {prediction_results['I'][-1]:.0f}")
else:
    print(f"Calibration did not converge: {result.termination_reason}")
```

**Important**: The `Calibrator` returns a `CalibrationResult` object containing the optimized parameter values, but does not automatically update your model. You must manually create a new model with the calibrated parameters to use them for predictions.

## Best Practices

### 1. Choose Appropriate Bounds

Set realistic bounds based on biological knowledge:

```python
# Too wide: allows unrealistic values
CalibrationParameter(id="beta", min_bound=0.0, max_bound=100.0)  # Bad

# Reasonable: based on disease characteristics
CalibrationParameter(id="beta", min_bound=0.1, max_bound=0.8)    # Good
```

### 2. Provide Good Initial Guesses

Initial guesses can speed up convergence:

```python
# Let optimizer choose random
CalibrationParameter(id="beta", min_bound=0.1, max_bound=0.8)

# Provide informed starting point
CalibrationParameter(
    id="beta",
    min_bound=0.1,
    max_bound=0.8,
    initial_guess=0.4  # Based on literature
)
```

### 3. Use Appropriate Loss Functions

- **SSE/RMSE**: When all observations equally important
- **MAE**: When outliers present in data
- **Weighted SSE**: When some observations more reliable

### 4. Choose the Right Algorithm

- **Particle Swarm**: Complex landscapes, global search, multiple parameters
- **Nelder-Mead**: Smooth problems, fewer parameters, faster convergence

### 5. Validate Results

Always check biological plausibility:

```python
result = calibrator.run()

# Check parameter values make sense
if result.best_parameters['beta'] > 1.0:
    print("Warning: Unusually high transmission rate")

# Check final loss
if result.final_loss > 1000:
    print("Warning: Poor fit quality")

# Verify convergence
if not result.converged:
    print(f"Warning: Did not converge - {result.termination_reason}")
```

## Troubleshooting

### Calibration Not Converging

- Increase `max_iterations`
- Widen parameter bounds
- Try different optimization algorithm
- Check if model can actually produce observed patterns

### Poor Fit Quality

- Verify observed data is correct
- Check model structure matches reality
- Add more parameters if underfitting
- Use weighted loss to prioritize important observations

## See Also

- [API Reference](../api/calibrator.md) - Complete Calibrator API documentation
- [Examples](examples.md) - Additional calibration examples
- [Simulation](simulations.md) - Running model simulations
