# Quick Start

[Back to Home](../index.md)

Build, run, check and calibrate your first compartment model in five minutes.
This page assumes Commol is installed (see [Installation](installation.md)).

## Quick path

1. Write the process with `ModelBuilder`, and build it with a structure
   (`Aggregate`) and an engine.
2. Run it with `Simulation`.
3. Read the results by compartment name.

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
    .build(
        Aggregate(
            bin_amounts={"A": 990, "B": 10, "C": 0},
        ),
        engine="DifferenceEquations",
    )
)

simulation = Simulation(model)
results = simulation.run(num_steps=100)

print(f"A at step 100: {results['A'][-1]:.0f}")
print(f"B at step 100: {results['B'][-1]:.0f}")
print(f"C at step 100: {results['C'][-1]:.0f}")
```

`results` maps each compartment name to its values, one per step, from step 0
to step 100.

## What each call does

| Call | What it does |
|---|---|
| `ModelBuilder(name=..., version=...)` | Starts a model definition. Every method returns the builder, so calls chain. |
| `.add_bin(id="A", name="State A")` | Adds a compartment (a *bin*): a state that holds an amount, in the unit of the bin. |
| `.add_parameter(id="k1", value=0.3)` | Adds a named constant for rate formulas. `value=None` marks it for calibration. |
| `.add_transition(..., rate="k1 * A * B / total")` | Moves an amount from `source` to `target` along the axis. This rate names compartments, so it is the flow itself (*absolute*). A `DifferenceEquations` model moves the flow once per step; see [What a rate means](../guide/core-concepts.md#what-a-rate-means). |
| `.add_transition(..., rate="k2")` | This rate names no compartment, so it is *proportional*: the flow is `k2 × B`. |
| `.build(Aggregate(...), engine="DifferenceEquations")` | Joins what you wrote, the *process*, with a *structure* and an engine, validates it and returns a `Model`. `Aggregate` holds an amount per compartment: `bin_amounts` gives the amount each bin starts with. |
| `Simulation(model).run(num_steps=100)` | Runs the Rust engine and returns the results. |

See [Core Concepts](../guide/core-concepts.md) for bins, transitions and the
two kinds of rate.

## Check units

Give every bin and parameter a unit, and the unit of one step, and Commol
checks that each rate gives the bin unit per step unit:

```python
model_with_units = (
    ModelBuilder(name="Model with Units", version="1.0", bin_unit="item", step_unit="day")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3, unit="1/day")
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .build(
        Aggregate(
            bin_amounts={"A": 990, "B": 10, "C": 0},
        ),
        engine="DifferenceEquations",
    )
)

model_with_units.check_unit_consistency()  # raises UnitConsistencyError on a mismatch
```

The check catches rates in another unit than the step unit, and quantities
with units passed to a function that needs a plain number. See
[Unit checking](../guide/units-and-equations.md#unit-checking).

## Calibrate unknown parameters

Set unknown parameters to `None`, describe the observations, and let an
optimizer find the values:

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
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .build(
        Aggregate(
            bin_amounts={"A": 990, "B": 10, "C": 0},
        ),
        engine="DifferenceEquations",
    )
)

observed_data = [
    ObservedDataPoint(at=10, output="B", value=45.2),
    ObservedDataPoint(at=20, output="B", value=78.5),
    ObservedDataPoint(at=30, output="B", value=62.3),
    ObservedDataPoint(at=40, output="B", value=38.1),
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

| Object | Role |
|---|---|
| `ObservedDataPoint` | One measurement: a step, a compartment and a value. |
| `CalibrationParameter` | One unknown, with its bounds. `initial_guess` is optional. |
| `CalibrationProblem` | The observations, the unknowns, the loss (`"sse"`, `"rmse"`, `"mae"`, …), the optimizer and an optional `seed`. |
| `Calibrator(...).run()` | Runs the optimizer in Rust and returns a `CalibrationResult` (`best_parameters`, `final_loss`, `converged`, …). |
| `Model.update_parameters(...)` | Writes the calibrated values into the model. |

See [Model Calibration](../guide/calibration.md) for initial conditions,
constraints, scale factors and the optimizers.

## Log messages

Commol logs with the standard `logging` module, to the logger `commol` and its
children, such as `commol.api.simulation` and `commol.api.calibrator`. Commol
adds no handler of its own: your application chooses where the messages go.
Python shows only warnings and errors until you configure logging.

To see the messages of commol and of every other library, configure the root
logger once, at the start of your program:

```python
import logging

logging.basicConfig(level=logging.INFO)

Simulation(model).run(num_steps=10)
```

Each message shows its level and its logger, such as
`INFO:commol.api.simulation:Running simulation for 10 steps.`

| To | Call |
|---|---|
| See the messages of every library | `logging.basicConfig(level=logging.INFO)` |
| Set a level for commol only | `logging.getLogger("commol").setLevel(logging.DEBUG)` |
| Send the messages of commol to a place of their own | `logging.getLogger("commol").addHandler(logging.FileHandler("commol.log"))` |
| Read the messages of one part only | `logging.getLogger("commol.api.calibrator")`, with a level or a handler |

See the [logging HOWTO](https://docs.python.org/3/howto/logging.html) for
handlers, formats and levels.

## Checklist

- [ ] The model builds without errors.
- [ ] `Simulation(model).run(...)` returns one series per compartment.
- [ ] `check_unit_consistency()` passes when you give units.
- [ ] Calibration returns `best_parameters` within the bounds you set.

## Next steps

| To learn | Read |
|---|---|
| The concepts behind models | [Core Concepts](../guide/core-concepts.md) |
| Every builder option | [Building Models](../guide/building-models.md) |
| What rate formulas can contain | [Mathematical Expressions](../guide/mathematical-expressions.md) |
| Running and reading simulations | [Running Simulations](../guide/simulations.md) |
| Runs at random, with whole units | [Stochastic Simulation](../guide/stochastic.md) |
| Calibration in depth | [Model Calibration](../guide/calibration.md) |
| Complete models | [Examples](../guide/examples.md) |
