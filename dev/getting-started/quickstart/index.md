# Quick Start

[Back to Home](https://munqu.github.io/commol/latest/index.md)

Build, run, check and calibrate your first compartment model in five minutes. This page assumes Commol is installed (see [Installation](https://munqu.github.io/commol/latest/getting-started/installation/index.md)).

## Quick path

1. Build a model with `ModelBuilder`.
1. Run it with `Simulation`.
1. Read the results by compartment name.

```python
from commol import ModelBuilder, Simulation

model = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)

print(f"A at step 100: {results['A'][-1]:.0f}")
print(f"B at step 100: {results['B'][-1]:.0f}")
print(f"C at step 100: {results['C'][-1]:.0f}")
```

`results` maps each compartment name to its values, one per step, from step 0 to step 100.

## What each call does

| Call                                          | What it does                                                                                                               |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `ModelBuilder(name=..., version=...)`         | Starts a model definition. Every method returns the builder, so calls chain.                                               |
| `.add_bin(id="A", name="State A")`            | Adds a compartment (a *bin*): a state that holds part of the population.                                                   |
| `.add_parameter(id="k1", value=0.3)`          | Adds a named constant for rate formulas. `value=None` marks it for calibration.                                            |
| `.add_transition(..., rate="k1 * A * B / N")` | Moves population from `source` to `target` each step. This rate names compartments, so it is the flow itself (*absolute*). |
| `.add_transition(..., rate="k2")`             | This rate names no compartment, so it is *per-capita*: the flow is `k2 × B`.                                               |
| `.set_initial_conditions(...)`                | Sets the total population and the fraction that starts in each bin. The fractions must sum to 1.                           |
| `.build(typology="DifferenceEquations")`      | Validates the definition and returns a `Model`.                                                                            |
| `Simulation(model).run(num_steps=100)`        | Runs the Rust engine and returns the results.                                                                              |

See [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md) for bins, transitions and the two kinds of rate.

## Check units

Give every bin and parameter a unit, and Commol checks that each rate gives population per unit of time:

```python
model_with_units = (
    ModelBuilder(name="Model with Units", version="1.0", bin_unit="item")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3, unit="1/day")
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)

model_with_units.check_unit_consistency()  # raises UnitConsistencyError on a mismatch
```

The check catches rates that mix time scales or pass a quantity with units to a function that needs a plain number. See [Unit checking](https://munqu.github.io/commol/latest/guide/units-and-equations/#unit-checking).

## Calibrate unknown parameters

Set unknown parameters to `None`, describe the observations, and let an optimizer find the values:

```python
from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ObservedDataPoint,
    ParticleSwarmConfig,
)

calibration_model = (
    ModelBuilder(name="Calibration Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)
    .add_parameter(id="k2", value=None)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / N")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "A", "fraction": 0.99},
            {"bin": "B", "fraction": 0.01},
            {"bin": "C", "fraction": 0.0},
        ],
    )
    .build(typology="DifferenceEquations")
)

observed_data = [
    ObservedDataPoint(step=10, compartment="B", value=45.2),
    ObservedDataPoint(step=20, compartment="B", value=78.5),
    ObservedDataPoint(step=30, compartment="B", value=62.3),
    ObservedDataPoint(step=40, compartment="B", value=38.1),
]

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(
            id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0
        ),
        CalibrationParameter(
            id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0
        ),
    ],
    loss_function="sse",
    optimization_config=ParticleSwarmConfig(num_particles=30, max_iterations=300),
    seed=42,
)

# A Simulation of a model with uncalibrated parameters can be calibrated, not run.
result = Calibrator(Simulation(calibration_model), problem).run()

print(f"k1 = {result.best_parameters['k1']:.4f}")
print(f"k2 = {result.best_parameters['k2']:.4f}")
print(f"loss = {result.final_loss:.4f}")

calibration_model.update_parameters(result.best_parameters)
calibrated_results = Simulation(calibration_model).run(num_steps=100)
```

| Object                         | Role                                                                                                            |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `ObservedDataPoint`            | One measurement: a step, a compartment and a value.                                                             |
| `CalibrationParameter`         | One unknown, with its bounds. `initial_guess` is optional.                                                      |
| `CalibrationProblem`           | The observations, the unknowns, the loss (`"sse"`, `"rmse"`, `"mae"`, …), the optimizer and an optional `seed`. |
| `Calibrator(...).run()`        | Runs the optimizer in Rust and returns a `CalibrationResult` (`best_parameters`, `final_loss`, `converged`, …). |
| `Model.update_parameters(...)` | Writes the calibrated values into the model.                                                                    |

See [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md) for initial conditions, constraints, scale factors and the optimizers.

## Checklist

- [ ] The model builds without errors.
- [ ] `Simulation(model).run(...)` returns one series per compartment.
- [ ] `check_unit_consistency()` passes when you give units.
- [ ] Calibration returns `best_parameters` within the bounds you set.

## Next steps

| To learn                        | Read                                                                                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- |
| The concepts behind models      | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md)                       |
| Every builder option            | [Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md)                   |
| What rate formulas can contain  | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) |
| Running and reading simulations | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/index.md)                   |
| Calibration in depth            | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)                     |
| Complete models                 | [Examples](https://munqu.github.io/commol/latest/guide/examples/index.md)                                 |
