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
Parameters and initial conditions can be set to `None` to indicate they need calibration. A `Simulation` can be created with `None` values for calibration purposes, but attempting to call `run()` on a simulation with uncalibrated values will raise a `ValueError`. After calibration, use `model.apply_calibration_parameters(result, problem)` to write every calibrated value back to the model at once. It routes each value by the `parameter_type` declared in the problem, so model parameters, bin fractions and stratification categories all land in the right place. `model.update_parameters(...)` and `model.update_initial_conditions(...)` remain available when you want to set one kind directly.

## Basic Example

Here's a simple calibration of a model's rate parameters. Parameters to be calibrated should be set to `None`, and initial guesses are provided in the `CalibrationParameter` configuration:

```python
from commol import (
    ModelBuilder,
    Simulation,
    Calibrator,
    CalibrationProblem,
    CalibrationParameter,
    ObservedDataPoint,
    ParticleSwarmConfig,
    NelderMeadConfig,
)


# Build model with parameters to be calibrated set to None
model = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)   # To be calibrated
    .add_parameter(id="k2", value=None)   # To be calibrated
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

# Define observed data
observed_data = [
    ObservedDataPoint(step=0, compartment="B", value=10.0),
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
    ObservedDataPoint(step=30, compartment="B", value=62.3),
    ObservedDataPoint(step=40, compartment="B", value=38.1),
    ObservedDataPoint(step=50, compartment="B", value=18.7),
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
        id="k1",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3  # Starting point for optimization
    ),
    CalibrationParameter(
        id="k2",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.1
    ),
]

# Configure the calibration problem with Particle Swarm Optimization
pso_config = ParticleSwarmConfig(
    num_particles=30,
    max_iterations=500,
    verbose=True  # Show optimization progress
)

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=pso_config,
)

# Run calibration
calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Display results
print(result)
print(f"Calibrated k1: {result.best_parameters['k1']:.4f}")
print(f"Calibrated k2: {result.best_parameters['k2']:.4f}")

# Update model with calibrated parameters
model.update_parameters(result.best_parameters)

# Create new simulation with calibrated model for predictions
calibrated_simulation = Simulation(model)
prediction_results = calibrated_simulation.run(num_steps=200)
print(f"State B at step 200: {prediction_results['B'][-1]:.0f}")
```

### Calibrating Initial Conditions

You can also calibrate initial population fractions:

```python
# Build model with unknown initial fraction for compartment B
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
        rate="k2 * B"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.98},
            {"bin": "B", "fraction": None},  # Unknown - to be calibrated
            {"bin": "C", "fraction": 0.0}
        ]
    )
    .build(typology="DifferenceEquations")
)

# Calibrate initial fraction for B
parameters = [
    CalibrationParameter(
        id="B",
        parameter_type="initial_condition",
        min_bound=0.0,
        max_bound=0.1,
        initial_guess=0.02  # Starting point for optimization
    )
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=1000),
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
loss_function = "sse"
```

$$\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)

Provides error in the same units as the data:

```python
loss_function = "rmse"
```

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Mean Absolute Error (MAE)

Less sensitive to outliers:

```python
loss_function = "mae"
```

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Weighted SSE

Allows different importance for different observations:

```python
# Give more weight to early observations
observed_data = [
    ObservedDataPoint(step=0, compartment="B", value=10.0, weight=2.0),
    ObservedDataPoint(step=10, compartment="B", value=45.2, weight=1.5),
    ObservedDataPoint(step=20, compartment="B", value=78.5, weight=1.0),
]

loss_function = "weighted_sse"
```

$$\text{WSSE} = \sum_{i=1}^{n} w_i (y_i - \hat{y}_i)^2$$

## Optimization Algorithms

### Particle Swarm Optimization (PSO)

Population-based algorithm inspired by social behavior. Good for global optimization.

**Basic Configuration:**

```python
from commol import ParticleSwarmConfig

# Basic PSO configuration
pso_config = ParticleSwarmConfig(
    num_particles=40,        # Number of particles in swarm
    max_iterations=1000,     # Maximum iterations
    verbose=True,            # Print progress
)
```

**Advanced Configuration:**

```python
# PSO with chaotic inertia and TVAC (time-varying acceleration)
pso_config = (
    ParticleSwarmConfig(
        num_particles=40,
        max_iterations=1000,
        verbose=True,
        initialization="latin_hypercube",  # Better initial distribution
    )
    .inertia("chaotic", w_min=0.4, w_max=0.9)  # Dynamic inertia weight
    .acceleration("time_varying",              # Time-varying acceleration
                  c1_initial=2.5, c1_final=0.5,
                  c2_initial=0.5, c2_final=2.5)
)

# PSO with mutation to escape local optima
pso_config = (
    ParticleSwarmConfig(num_particles=50, max_iterations=2000)
    .acceleration("constant", cognitive=2.0, social=2.0)  # Constant acceleration
    .velocity(clamp_factor=0.2)  # Prevent explosive velocities
    .mutation(
        "gaussian",              # Gaussian mutation
        scale=0.1,               # Mutation intensity
        probability=0.05,        # 5% mutation chance per iteration
        application="global_best"  # Apply to best particle only
    )
)
```

**Fluent Methods:**

- `.inertia("constant", factor=...)` - Set constant inertia weight
- `.inertia("chaotic", w_min=..., w_max=...)` - Enable chaotic inertia
- `.acceleration("constant", cognitive=..., social=...)` - Set constant acceleration factors
- `.acceleration("time_varying", c1_initial=..., c1_final=..., c2_initial=..., c2_final=...)` - Time-varying acceleration
- `.velocity(clamp_factor=..., mutation_threshold=...)` - Velocity control
- `.mutation(strategy, scale=..., probability=..., application=...)` - Enable mutation

**When to use PSO:**

- Multiple local minima expected
- Robust global search needed
- Multiple parameters to calibrate
- Complex parameter landscapes

### Nelder-Mead Simplex

Derivative-free simplex algorithm. Fast convergence for smooth problems:

```python
from commol import NelderMeadConfig

# Config type determines the algorithm (Nelder-Mead in this case)
optimization_config = NelderMeadConfig(
    max_iterations=1000,     # Maximum iterations
    sd_tolerance=1e-6,       # Convergence tolerance
    alpha=1.0,               # Reflection coefficient (optional)
    gamma=2.0,               # Expansion coefficient (optional)
    rho=0.5,                 # Contraction coefficient (optional)
    sigma=0.5,               # Shrink coefficient (optional)
    verbose=True,            # Print progress
    header_interval=50       # Print header every N iterations
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
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)   # To be calibrated
    .add_parameter(id="k2", value=None)   # To be calibrated
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

# Observed data from both B and C compartments
observed_data = [
    # B observations
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
    ObservedDataPoint(step=30, compartment="B", value=62.3),
    # C observations
    ObservedDataPoint(step=10, compartment="C", value=12.5),
    ObservedDataPoint(step=20, compartment="C", value=35.8),
    ObservedDataPoint(step=30, compartment="C", value=68.2),
]

simulation = Simulation(model)

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
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
    ],
    loss_function="rmse",
    optimization_config=ParticleSwarmConfig(max_iterations=500, verbose=False),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Update model with calibrated values
model.update_parameters(result.best_parameters)
```

### Aggregate Observations

Use `compartments` when a single observation corresponds to the sum of multiple
simulation outputs. The `compartment` field remains the label used for grouping
and plotting; the outputs listed in `compartments` are summed before loss,
coverage, and plotting comparisons.

```python
observed_data = [
    ObservedDataPoint(
        step=10,
        compartment="reported_total",
        compartments=["B_g1", "B_g2"],
        value=72.0,
    ),
]
```

Aggregate observations can reference compartments or accumulator outputs. They
can also be combined with `scale_id` and `window_steps`.

### Calibrating Stratified Models

Calibrate parameters specific to individual stratification categories:

```python
# Build stratified model with group-specific parameters to calibrate
model = (
    ModelBuilder(name="Stratified Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=None)       # To be calibrated
    .add_parameter(id="k2_g1", value=None)    # To be calibrated
    .add_parameter(id="k2_g2", value=None)    # To be calibrated
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
        stratified_rates=[
            {
                "conditions": [{"stratification": "group", "category": "g1"}],
                "rate": "k2_g1 * B"
            },
            {
                "conditions": [{"stratification": "group", "category": "g2"}],
                "rate": "k2_g2 * B"
            },
        ]
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0}
        ],
        stratification_fractions=[
            {
                "stratification": "group",
                "fractions": [
                    {"category": "g1", "fraction": 0.6},
                    {"category": "g2", "fraction": 0.4}
                ]
            }
        ]
    )
    .build(typology="DifferenceEquations")
)

# Observed data uses stratified compartment names like "B_g1", "B_g2"
observed_data = [
    ObservedDataPoint(step=10, compartment="B_g1", value=30.5),
    ObservedDataPoint(step=10, compartment="B_g2", value=15.2),
    ObservedDataPoint(step=20, compartment="B_g1", value=52.3),
    ObservedDataPoint(step=20, compartment="B_g2", value=26.1),
]

simulation = Simulation(model)

# Calibrate stratification-specific parameters
parameters = [
    CalibrationParameter(
        id="k1",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3
    ),
    CalibrationParameter(
        id="k2_g1",
        parameter_type="parameter",
        min_bound=0.05,
        max_bound=0.2,
        initial_guess=0.12
    ),
    CalibrationParameter(
        id="k2_g2",
        parameter_type="parameter",
        min_bound=0.05,
        max_bound=0.2,
        initial_guess=0.08
    ),
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=ParticleSwarmConfig(max_iterations=500, verbose=False),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Update model with calibrated values
model.update_parameters(result.best_parameters)
```

### Calibrating Scale Parameters

When observed data represents only a fraction of true values (e.g., due to partial detection or measurement limits), scale parameters allow you to calibrate the detection rate alongside model parameters.

**How Scale Parameters Work:**

During calibration, model predictions are multiplied by the scale factor before comparing with observed data. This allows simultaneous estimation of true dynamics and the observation process.

**Basic Example:**

```python
# Build model with parameters to calibrate
model = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)
    .add_parameter(id="k2", value=None)
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

# Observed values (potentially partially detected)
observed_values = [10, 15, 25, 40, 60, 75, 85, 70, 50, 30]

# Link observed data to scale parameter
observed_data = [
    ObservedDataPoint(
        step=idx,
        compartment="B",
        value=val,
        scale_id="detection_rate"
    )
    for idx, val in enumerate(observed_values)
]

simulation = Simulation(model)

# Define parameters including scale
parameters = [
    CalibrationParameter(
        id="k1",
        parameter_type="parameter",
        min_bound=0.1,
        max_bound=1.0
    ),
    CalibrationParameter(
        id="k2",
        parameter_type="parameter",
        min_bound=0.05,
        max_bound=0.5
    ),
    CalibrationParameter(
        id="detection_rate",
        parameter_type="scale",
        min_bound=0.01,
        max_bound=1.0
    ),
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="mae",
    optimization_config=ParticleSwarmConfig(num_particles=40, max_iterations=500),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Separate parameters by type
parameters = {}
scale_values = {}

for param in problem.parameters:
    value = result.best_parameters[param.id]

    if param.parameter_type == "parameter":
        parameters[param.id] = value
    elif param.parameter_type == "scale":
        scale_values[param.id] = value

print(f"Calibrated detection rate: {scale_values['detection_rate']:.2%}")

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
    ObservedDataPoint(step=10, compartment="B", value=45.0, scale_id="b_detection"),
    ObservedDataPoint(step=10, compartment="C", value=120.0, scale_id="c_detection"),
]

parameters = [
    CalibrationParameter(
        id="b_detection",
        parameter_type="scale",
        min_bound=0.05,
        max_bound=0.5
    ),
    CalibrationParameter(
        id="c_detection",
        parameter_type="scale",
        min_bound=0.7,
        max_bound=1.0
    ),
]
```

**Key Considerations:**

- **Identifiability**: Calibrating rate parameters and scale simultaneously may cause identifiability issues
- **Visualization**: Always pass `scale_values` to plotter for correct display
- **Interpretation**: Calibrated scale = fraction of true compartment value observed (e.g., 0.15 = 15% detection)

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

    print(f"State B at step 200: {prediction_results['B'][-1]:.0f}")
else:
    print(f"Calibration did not converge: {result.termination_reason}")
```

**Important**: The `Calibrator` returns a `CalibrationResult` object containing the optimized parameter values, but does not automatically update your model. Use `model.apply_calibration_parameters(result, problem)` to update the model in place, then create a new `Simulation` object to run predictions with the calibrated parameters.

### Reporting what the calibration applied

`applied_parameters_report` renders the calibrated model as a plain structure:
every numeric parameter and every initial condition, each flagged with whether
the calibration set it.

```python
from commol import applied_parameters_report

report = applied_parameters_report(model, result, problem)
```

Initial conditions carry the `fraction` the model was built from, the head
`value` it works out to, and the `group` that fraction is taken of — which for a
[conditional stratification](core-concepts.md#subgroup-head-counts) is its
conditioning categories rather than the whole population.

Pass `problem` to include `scale` parameters, which are calibrated but have no
place in the model. For a `ProbabilisticCalibrationResult` every entry also
gains an `interval` block, and the report is preceded by `confidence_level`:

```json
{
  "confidence_level": 0.95,
  "parameters": {
    "k1": {"value": 0.31, "calibrated": true,
           "interval": {"mean": 0.31, "median": 0.31, "ci_lower": 0.28,
                        "ci_upper": 0.34, "min": 0.27, "max": 0.35, "std": 0.02}}
  },
  "initial_conditions": {
    "N": {"group": "population", "fraction": 1.0, "value": 10000.0,
          "calibrated": false}
  }
}
```

Intervals are in the unit the entry was calibrated in, so for an initial
condition that is its fraction, not its head count.

### Writing a result back to the model

```python
model.apply_calibration_parameters(result, problem)
```

This accepts a `CalibrationResult`, a `ProbabilisticCalibrationResult`, or a bare
mapping of parameter id to value (one ensemble member's parameters, for
instance). For a probabilistic result it applies `point_parameters` of the
selected ensemble, which is the only parameter set that describes a single
calibrated model.

Each value is routed by its declared `parameter_type`:

| `parameter_type` | Written to |
|---|---|
| `parameter` | the model parameter of that id |
| `initial_condition`, id is a bin | that bin's fraction |
| `initial_condition`, id is a stratification category | that category's fraction, and the remaining category |
| `scale` | nothing — it scales observations, not the model |

!!! warning "Per-compartment initial conditions cannot be written back"

    A calibration parameter naming an expanded compartment (`A_group1`) is
    accepted by the calibrator, but the model stores initial conditions only as
    bin and stratification fractions, so there is nowhere to put it.
    `apply_calibration_parameters` raises rather than skipping it silently.

#### Calibrated stratification splits

An `initial_condition` parameter may name a category of a binary
stratification, in which case calibration fits the split between the two
categories. Writing it back sets the named category and leaves the other to take
the remaining fraction:

```python
model.update_stratification_fractions({"group1": 0.25})
# group1 -> 0.25, group2 -> 0.75
```

See [updating fractions after building](core-concepts.md#updating-fractions-after-building)
for the general rule.

## Calibrating Against Cumulative Outputs

When your observed data represents incremental counts over a period (e.g., new events recorded each week), you can calibrate against [accumulator](core-concepts.md#accumulators) outputs using the `window_steps` parameter on `ObservedDataPoint`.

`window_steps=W` tells the calibration engine that the observation at step `t` corresponds to the **accumulated change** from step `t - W` to step `t`:

```
predicted_at(t) = accumulator(t) - accumulator(t - W)
```

`windowed_totals` applies that same reduction outside the calibration, so a
report can show the quantity the loss actually compared against:

```python
from commol import window_end_steps, windowed_totals

results = simulation.run(num_steps=70)

# Every complete window of the run
weekly = simulation.windowed_totals(results, ["cum_ab"], 7)

# Or exactly the steps the observations use
steps = window_end_steps(7, 70)
weekly = simulation.windowed_totals(results, ["cum_ab"], 7, steps)
```

`window_end_steps(W, num_steps)` lists the steps closing each complete window,
dropping a trailing partial one. `windowed_totals` also accepts steps that are
not multiples of the window, matching observations anchored anywhere.

For a probabilistic run the ensemble equivalents are on the selected ensemble:
`windowed_prediction_median`, `windowed_prediction_ci_lower` and
`windowed_prediction_ci_upper`, with `windowed_prediction_steps` giving the step
each value belongs to.

### Example: Windowed Calibration

```python
from commol import ModelBuilder, Simulation, Calibrator
from commol.context.calibration import (
    ObservedDataPoint,
    CalibrationParameter,
    CalibrationProblem,
    ParticleSwarmConfig,
)

# Model with an accumulator tracking flows A→B
model = (
    ModelBuilder("Model")
    .add_bin("A", "State A")
    .add_bin("B", "State B")
    .add_accumulator("cum_ab", "Cumulative A→B")
    .add_parameter("k1", None)       # to calibrate
    .add_transition("flow", ["A"], ["B"], rate="k1 * A * B / N",
                    accumulators=["cum_ab"])
    .set_initial_conditions(10000, [
        {"bin": "A", "fraction": 0.99},
        {"bin": "B", "fraction": 0.01},
    ])
    .build("DifferenceEquations")
)

# Weekly incidence data: new A→B events each 7-step window
weekly_new_ab = [45, 82, 130, 174, 205, 190, 160, 118, 85, 55]
observed_data = [
    ObservedDataPoint(
        step=7 * (i + 1),
        compartment="cum_ab",
        value=val,
        window_steps=7,  # compare week-over-week change
    )
    for i, val in enumerate(weekly_new_ab)
]

parameters = [
    CalibrationParameter(id="k1", parameter_type="parameter",
                         min_bound=0.0, max_bound=1.0)
]
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    constraints=[],
    optimization_config=ParticleSwarmConfig(num_particles=30, max_iterations=500),
)

calibrator = Calibrator(Simulation(model), problem)
result = calibrator.run()
print(f"k1 = {result.best_parameters['k1']:.4f}, loss = {result.final_loss:.2f}")
```

### Rules for `window_steps`

- `window_steps` must be ≤ the observation step (`step ≥ window_steps`).
- The compartment must be a simulation output (a compartment or an accumulator).
- `window_steps` can be combined with `scale_id`.
- Multiple observations at different steps can use different `window_steps` values.

## Best Practices

### 1. Choose Appropriate Bounds

Set realistic bounds based on domain knowledge:

```python
# Too wide: allows unrealistic values
CalibrationParameter(id="k1", min_bound=0.0, max_bound=100.0)  # Bad

# Reasonable: based on expected parameter ranges
CalibrationParameter(id="k1", min_bound=0.1, max_bound=0.8)    # Good
```

### 2. Provide Good Initial Guesses

Initial guesses can speed up convergence:

```python
# Without initial guess - optimizer uses midpoint
CalibrationParameter(
    id="k1",
    parameter_type="parameter",
    min_bound=0.1,
    max_bound=0.8
)

# Provide informed starting point (recommended)
CalibrationParameter(
    id="k1",
    parameter_type="parameter",
    min_bound=0.1,
    max_bound=0.8,
    initial_guess=0.4  # Based on prior knowledge
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

Always check plausibility of results:

```python
result = calibrator.run()

# Check parameter values make sense
if result.best_parameters['k1'] > 1.0:
    print("Warning: Unusually high rate parameter")

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

## Constraints on Calibration Parameters

Constraints allow you to enforce mathematical relationships between parameters during calibration.

### How Constraints Work

Constraints are mathematical expressions that must evaluate to **≥ 0** for the constraint to be satisfied. When violated (expression < 0), a penalty is added to the loss function, guiding the optimizer toward feasible parameter values.

$$\text{Total Loss} = \text{Data Loss} + \sum_{i} w_i \cdot \max(0, -c_i)^2$$

where $c_i$ is the constraint expression and $w_i$ is the constraint weight.

### Basic Constraint Example

```python
from commol import CalibrationConstraint

# Build model with parameters to calibrate
model = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)
    .add_parameter(id="k2", value=None)
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

observed_data = [
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
    ObservedDataPoint(step=30, compartment="B", value=62.3),
]

simulation = Simulation(model)

parameters = [
    CalibrationParameter(
        id="k1",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=1.0,
    ),
    CalibrationParameter(
        id="k2",
        parameter_type="parameter",
        min_bound=0.0,
        max_bound=0.5,
    ),
]

# Add constraint: k1/k2 <= 5, which is equivalent to 5 - k1/k2 >= 0
constraints = [
    CalibrationConstraint(
        id="ratio_bound",
        expression="5.0 - k1/k2",
        description="k1/k2 <= 5",
        weight=1.0,
    )
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    constraints=constraints,  # Add constraints here
    loss_function="sse",
    optimization_config=ParticleSwarmConfig(num_particles=30, max_iterations=500),
)

calibrator = Calibrator(simulation, problem)
result = calibrator.run()

# Verify constraint is satisfied
ratio = result.best_parameters["k1"] / result.best_parameters["k2"]
print(f"Calibrated k1/k2 ratio: {ratio:.2f}")  # Should be <= 5
```

### Types of Constraints

#### 1. Parameter Relationship Constraints

Enforce mathematical relationships between parameters:

```python
# Forward rate must be faster than reverse rate
CalibrationConstraint(
    id="k_forward_ge_reverse",
    expression="k_forward - k_reverse",
    description="Forward rate >= reverse rate",
)

# Birth rate limited relative to death rate
CalibrationConstraint(
    id="birth_death_ratio",
    expression="3.0 - birth_rate/death_rate",
    description="Birth/death ratio <= 3",
)

# Sum of transition probabilities must not exceed 1.0
CalibrationConstraint(
    id="probability_sum",
    expression="1.0 - (p_move + p_stay + p_exit)",
    description="Total probability <= 1.0",
)
```

#### 2. Ordering Constraints

Enforce relative ordering of parameters:

```python
# Production rate faster than defect rate
CalibrationConstraint(
    id="production_ordering",
    expression="production_rate - defect_rate",
    description="Production rate >= defect rate",
)

# Inflow rate greater than outflow rate
CalibrationConstraint(
    id="flow_order",
    expression="inflow_rate - outflow_rate",
    description="Inflow >= outflow",
)

# Fast decay rate exceeds slow decay rate
CalibrationConstraint(
    id="decay_ordering",
    expression="k_fast - k_slow",
    description="Fast decay >= slow decay",
)
```

#### 3. Time-Dependent Constraints

Constrain compartment values at specific time steps. These constraints can reference both parameters and compartment states:

```python
# Stock level must not exceed capacity
CalibrationConstraint(
    id="capacity_limit",
    expression="1000.0 - Stock",
    description="Stock never exceeds capacity",
    time_steps=[10, 20, 30, 40, 50],
)

# Minimum product amount by time 30
CalibrationConstraint(
    id="min_product",
    expression="Product - 50.0",
    description="At least 50 units of product by step 30",
    time_steps=[30],
)

# Total amount must stay below threshold
CalibrationConstraint(
    id="total_limit",
    expression="100.0 - (A + B)",
    description="Total A + B <= 100",
    time_steps=[30, 50, 70],
)
```

**Note**: Time-dependent constraints are evaluated at each specified time step during each simulation run. This allows you to constrain the system dynamics, not just the parameters.

### Multiple Constraints

You can apply multiple constraints simultaneously:

```python
constraints = [
    # k1/k2 ratio must be between 2 and 5
    CalibrationConstraint(
        id="ratio_min",
        expression="k1/k2 - 2.0",
        description="k1/k2 >= 2",
        weight=1.0,
    ),
    CalibrationConstraint(
        id="ratio_max",
        expression="5.0 - k1/k2",
        description="k1/k2 <= 5",
        weight=1.0,
    ),
    # k1 must be greater than k2
    CalibrationConstraint(
        id="ordering",
        expression="k1 - k2",
        description="k1 >= k2",
        weight=0.5,
    ),
    # Peak B below 500
    CalibrationConstraint(
        id="peak_limit",
        expression="500.0 - B",
        description="Peak B <= 500",
        time_steps=[10, 20, 30, 40, 50],
        weight=2.0,  # Higher weight = stricter enforcement
    ),
]
```

### Constraint Weights

The `weight` parameter controls how strictly a constraint is enforced. Higher weights make violations more costly:

```python
# Strict enforcement - large penalty for violations
CalibrationConstraint(
    id="critical_constraint",
    expression="10.0 - k1/k2",
    weight=10.0,  # High weight
)

# Soft enforcement - smaller penalty
CalibrationConstraint(
    id="preferred_constraint",
    expression="k1 - k2",
    weight=0.5,  # Low weight
)
```

**Default**: `weight=1.0`

### Best Practices for Constraints

1. **Write expressions that evaluate to ≥ 0 when satisfied**:
   - For `k1 <= 0.5`, use `"0.5 - k1"`
   - For `k1 >= k2`, use `"k1 - k2"`
   - For ratio `k1/k2 <= 5`, use `"5.0 - k1/k2"`

2. **Use descriptive IDs and descriptions**:

   ```python
   CalibrationConstraint(
       id="ratio_upper_bound",
       expression="5.0 - k1/k2",
       description="k1/k2 ratio must be <= 5",
   )
   ```

3. **Start with unit weights and adjust**:
   - Begin with `weight=1.0` for all constraints
   - Increase weight if constraint is frequently violated
   - Decrease weight if it prevents convergence

4. **Balance constraints and data fit**:
   - Too many or too strict constraints may prevent finding good fits
   - Monitor both data loss and constraint violations
   - Consider if constraints are realistic given your data

5. **Validate constraint expressions**:
   ```python
   # Test constraint expression with sample parameters
   k1, k2 = 0.3, 0.1
   ratio_constraint = 5.0 - k1/k2
   print(f"Ratio constraint value: {ratio_constraint}")  # Should be >= 0 if satisfied
   ```

### Troubleshooting Constraints

**Calibration fails to converge with constraints:**

- Check if constraints are too restrictive
- Widen parameter bounds
- Lower constraint weights
- Verify constraints don't contradict each other

**Constraints frequently violated:**

- Increase `max_iterations`
- Increase constraint weights
- Use Particle Swarm instead of Nelder-Mead for better global search
- Check if constraints are realistic given the data

**Example of conflicting constraints:**

```python
# These constraints conflict!
constraints = [
    CalibrationConstraint(id="c1", expression="k1 - 0.5"),  # k1 >= 0.5
    CalibrationConstraint(id="c2", expression="0.4 - k1"),  # k1 <= 0.4
]
```

## See Also

- [API Reference](../api/calibrator.md) - Complete Calibrator API documentation
- [Examples](examples.md) - Additional calibration examples
- [Simulation](simulations.md) - Running model simulations
