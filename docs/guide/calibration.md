# Model Calibration

Calibration is the process of adjusting model parameters to match observed data. Commol provides a calibration framework that uses optimization algorithms to find the best parameter values that minimize the difference between model predictions and real-world observations.

## Overview

The calibration process involves:

1. **Defining observed data**: Real-world measurements at specific time steps
2. **Selecting parameters to calibrate**: Which model parameters or initial conditions to optimize
3. **Choosing a loss function**: How to measure fit quality (SSE, RMSE, MAE, etc.)
4. **Selecting an optimization algorithm**: Nelder-Mead or Particle Swarm
5. **Running the calibration**: Finding optimal parameter values
6. **Updating the model**: Applying calibrated values to your model

!!! note "Working with Uncalibrated Parameters"
Parameters and initial conditions can be set to `None` to indicate they need calibration. A `Simulation` can be created with `None` values for calibration purposes, but attempting to call `run()` on a simulation with uncalibrated values will raise a `ValueError`. After calibration, use `model.update_parameters(result.best_parameters)` or `model.update_initial_conditions(result.best_parameters)` to update your model with the calibrated values.

## Basic Example

Here's a simple calibration of an SIR model's transmission and recovery rates. Parameters to be calibrated should be set to `None`, and initial guesses are provided in the `CalibrationParameter` configuration:

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
    NelderMeadConfig,
)
from commol.constants import ModelTypes

# Build model with parameters to be calibrated set to None
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

# Simulation can be created with None values for calibration
simulation = Simulation(model)

# But attempting to run without calibrated values raises an error
try:
    simulation.run(100)
except ValueError as e:
    print(f"Error: {e}")  # Cannot run Simulation: Parameters requiring calibration...

# Define parameters to calibrate with bounds and initial guesses
parameters = [
    CalibrationParameter(
        id="beta",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3  # Starting point for optimization
    ),
    CalibrationParameter(
        id="gamma",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.1
    ),
]

# Configure the calibration problem with Particle Swarm Optimization
# Use the builder pattern for clearer configuration
pso_config = ParticleSwarmConfig.create(
    num_particles=30,
    max_iterations=500,
    verbose=True  # Show optimization progress
)

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.SSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=pso_config,
    ),
)

# Run calibration
calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Display results
print(result)
print(f"Calibrated beta: {result.best_parameters['beta']:.4f}")
print(f"Calibrated gamma: {result.best_parameters['gamma']:.4f}")

# Update model with calibrated parameters
model.update_parameters(result.best_parameters)

# Create new simulation with calibrated model for predictions
calibrated_simulation = Simulation(model)
prediction_results = calibrated_simulation.run(num_steps=200)
print(f"Predicted infections at day 200: {prediction_results['I'][-1]:.0f}")
```

### Calibrating Initial Conditions

You can also calibrate initial population fractions:

```python
# Build model with unknown initial infected fraction
model = (
    ModelBuilder(name="SIR Model", version="1.0")
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
        rate="gamma * I"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.98},
            {"bin": "I", "fraction": None},  # Unknown - to be calibrated
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Calibrate initial infected fraction
parameters = [
    CalibrationParameter(
        id="I",
        parameter_type=CalibrationParameterType.INITIAL_CONDITION,
        min_bound=0.0,
        max_bound=0.1,
        initial_guess=0.02  # Starting point for optimization
    )
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.SSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.NELDER_MEAD,
        config=NelderMeadConfig(max_iterations=1000),
    ),
)

# Simulation works with None initial conditions for calibration
simulation = Simulation(model)
calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Update initial conditions with calibrated values
model.update_initial_conditions(result.best_parameters)

# Now create final simulation
final_simulation = Simulation(model)
results = final_simulation.run(num_steps=100)
```

## Loss Functions

Commol supports multiple loss functions for measuring fit quality:

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

Population-based algorithm inspired by social behavior. Good for global optimization.

**Basic Configuration:**

Use the `create()` factory method and builder methods for clear configuration:

```python
from commol import ParticleSwarmConfig, OptimizationConfig, OptimizationAlgorithm

# Basic PSO configuration
pso_config = ParticleSwarmConfig.create(
    num_particles=40,        # Number of particles in swarm
    max_iterations=1000,     # Maximum iterations
    target_cost=0.01,        # Stop if loss below this (optional)
    verbose=True,            # Print progress
    header_interval=50       # Print header every N iterations
)

optimization_config = OptimizationConfig(
    algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
    config=pso_config,
)
```

**Advanced Configuration with Builder Methods:**

```python
# PSO with chaotic inertia and TVAC
pso_config = (
    ParticleSwarmConfig.create(num_particles=40, max_iterations=1000, verbose=True)
    .with_chaotic_inertia(w_min=0.4, w_max=0.9)  # Dynamic inertia weight
    .with_tvac(c1_initial=2.5, c1_final=0.5,     # Time-varying acceleration
               c2_initial=0.5, c2_final=2.5)
    .with_initialization_strategy("latin_hypercube")  # Better initial distribution
)

# PSO with mutation to escape local optima
pso_config = (
    ParticleSwarmConfig.create(num_particles=50, max_iterations=2000)
    .with_acceleration(cognitive=2.0, social=2.0)  # Constant acceleration
    .with_velocity_clamping(0.2)  # Prevent explosive velocities
    .with_mutation(
        strategy="gaussian",      # Gaussian mutation
        scale=0.1,               # Mutation intensity
        probability=0.05,        # 5% mutation chance per iteration
        application="global_best"  # Apply to best particle only
    )
)
```

**Builder Methods:**

- `.with_inertia(factor)` - Set constant inertia weight
- `.with_chaotic_inertia(w_min, w_max)` - Enable chaotic inertia (conflicts with `with_inertia`)
- `.with_acceleration(cognitive, social)` - Set constant acceleration factors
- `.with_tvac(c1_i, c1_f, c2_i, c2_f)` - Time-varying acceleration (conflicts with `with_acceleration`)
- `.with_initialization_strategy(strategy)` - "uniform", "latin_hypercube", or "opposition_based"
- `.with_velocity_clamping(factor)` - Clamp velocities (typically 0.1-0.2)
- `.with_velocity_mutation(threshold)` - Reinitialize near-zero velocities
- `.with_mutation(strategy, scale, probability, application)` - Enable mutation

**When to use PSO:**

- Multiple local minima expected
- Robust global search needed
- Multiple parameters to calibrate
- Complex parameter landscapes

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
# Build model with parameters to calibrate
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
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

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

simulation = Simulation(model)

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
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
    ],
    loss_config=LossConfig(function=LossFunction.RMSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig.create(max_iterations=500, verbose=False),
    ),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Update model with calibrated values
model.update_parameters(result.best_parameters)
```

### Calibrating Stratified Models

Calibrate parameters specific to age groups or other stratifications:

```python
# Build age-stratified model with parameters to calibrate
model = (
    ModelBuilder(name="Age-Stratified SIR", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_stratification(id="age", categories=["young", "old"])
    .add_parameter(id="beta", value=None)          # To be calibrated
    .add_parameter(id="gamma_young", value=None)   # To be calibrated
    .add_parameter(id="gamma_old", value=None)     # To be calibrated
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

# Note: Observed data uses stratified compartment names like "I_young", "I_old"
observed_data = [
    ObservedDataPoint(step=10, compartment="I_young", value=30.5),
    ObservedDataPoint(step=10, compartment="I_old", value=15.2),
    ObservedDataPoint(step=20, compartment="I_young", value=52.3),
    ObservedDataPoint(step=20, compartment="I_old", value=26.1),
]

simulation = Simulation(model)

# Calibrate stratification-specific parameters
parameters = [
    CalibrationParameter(
        id="beta",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3
    ),
    CalibrationParameter(
        id="gamma_young",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.05,
        max_bound=0.2,
        initial_guess=0.12
    ),
    CalibrationParameter(
        id="gamma_old",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.05,
        max_bound=0.2,
        initial_guess=0.08
    ),
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.SSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig.create(max_iterations=500, verbose=False),
    ),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Update model with calibrated values
model.update_parameters(result.best_parameters)
```

### Calibrating Scale Parameters

When observed data represents only a fraction of true cases (e.g., due to underreporting or detection limits), scale parameters allow you to calibrate the reporting or detection rate alongside model parameters.

**How Scale Parameters Work:**

During calibration, model predictions are multiplied by the scale factor before comparing with observed data. This allows simultaneous estimation of true disease dynamics and the observation process.

**Basic Example:**

```python
# Build SIR model with parameters to calibrate
model = (
    ModelBuilder(name="SIR Model", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=None)
    .add_parameter(id="gamma", value=None)
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

# Reported cases (potentially underreported)
reported_cases = [10, 15, 25, 40, 60, 75, 85, 70, 50, 30]

# Link observed data to scale parameter
observed_data = [
    ObservedDataPoint(
        step=idx,
        compartment="I",
        value=cases,
        scale_id="reporting_rate"
    )
    for idx, cases in enumerate(reported_cases)
]

simulation = Simulation(model)

# Define parameters including scale
parameters = [
    CalibrationParameter(
        id="beta",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.1,
        max_bound=1.0
    ),
    CalibrationParameter(
        id="gamma",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.05,
        max_bound=0.5
    ),
    CalibrationParameter(
        id="reporting_rate",
        parameter_type=CalibrationParameterType.SCALE,
        min_bound=0.01,
        max_bound=1.0
    ),
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.MAE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig.create(num_particles=40, max_iterations=500),
    ),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Separate parameters by type
parameters = {}
scale_values = {}

for param in problem.parameters:
    value = result.best_parameters[param.id]

    if param.parameter_type == CalibrationParameterType.PARAMETER:
        parameters[param.id] = value
    elif param.parameter_type == CalibrationParameterType.SCALE:
        scale_values[param.id] = value

print(f"Calibrated reporting rate: {scale_values['reporting_rate']:.2%}")

# Update model and run simulation
model.update_parameters(parameters)
calibrated_simulation = Simulation(model)
results = calibrated_simulation.run(num_steps=len(observed_data) - 1)

# Visualize with scale_values to correctly display observed data
from commol import SimulationPlotter

plotter = SimulationPlotter(calibrated_simulation, results)
plotter.plot_series(
    observed_data=observed_data,
    scale_values=scale_values,
)
```

**Multiple Scale Parameters:**

Different compartments can have different detection rates:

```python
observed_data = [
    ObservedDataPoint(step=10, compartment="I", value=45.0, scale_id="case_detection"),
    ObservedDataPoint(step=10, compartment="R", value=120.0, scale_id="recovery_detection"),
]

parameters = [
    CalibrationParameter(
        id="case_detection",
        parameter_type=CalibrationParameterType.SCALE,
        min_bound=0.05,
        max_bound=0.5
    ),
    CalibrationParameter(
        id="recovery_detection",
        parameter_type=CalibrationParameterType.SCALE,
        min_bound=0.7,
        max_bound=1.0
    ),
]
```

**Key Considerations:**

- **Identifiability**: Calibrating transmission parameters and scale simultaneously may cause identifiability issues
- **Visualization**: Always pass `scale_values` to plotter for correct display
- **Interpretation**: Calibrated scale = fraction of true cases observed (e.g., 0.15 = 15% detection)

### Using Calibration Results

Once calibrated, update your model with the fitted parameters and create a new simulation:

```python
# Run calibration
result = Calibrator(simulation, problem).run()

# Check if calibration converged
if result.converged:
    print(f"Calibration converged after {result.iterations} iterations")
    print(f"Final loss: {result.final_loss:.6f}")
    print(f"Calibrated parameters: {result.best_parameters}")

    # Update the existing model with calibrated parameters
    model.update_parameters(result.best_parameters)

    # Create a new simulation with the updated model
    calibrated_simulation = Simulation(model)
    prediction_results = calibrated_simulation.run(num_steps=200)

    print(f"Predicted infections at day 200: {prediction_results['I'][-1]:.0f}")
else:
    print(f"Calibration did not converge: {result.termination_reason}")
```

**Important**: The `Calibrator` returns a `CalibrationResult` object containing the optimized parameter values, but does not automatically update your model. Use `model.update_parameters(result.best_parameters)` to update the model in place, then create a new `Simulation` object to run predictions with the calibrated parameters.

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
# Without initial guess - optimizer uses midpoint
CalibrationParameter(
    id="beta",
    parameter_type=CalibrationParameterType.PARAMETER,
    min_bound=0.1,
    max_bound=0.8
)

# Provide informed starting point (recommended)
CalibrationParameter(
    id="beta",
    parameter_type=CalibrationParameterType.PARAMETER,
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
