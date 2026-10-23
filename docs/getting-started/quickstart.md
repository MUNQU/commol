# Quick Start

[Back to Home](../index.md)

Build a model, run it, read and plot its results, run the same process with
another engine, and fit it to data. Each step takes a few lines. This page
assumes Commol is installed with its figures (`pip install 'commol[plot]'`,
see [Installation](installation.md)).

## Quick path

1. Write the process with `ModelBuilder`: bins, parameters and transitions.
2. Build it with a structure (`Aggregate`, the amount each bin starts with)
   and an engine.
3. Run it with `Simulation`, read a series by its output name, and draw it
   with `results.plot`.

```python
from commol import Aggregate, ModelBuilder, Simulation

process = (
    ModelBuilder(name="Basic Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.3)
    .add_parameter(id="k2", value=0.1)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2")
)
structure = Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0})

model = process.build(structure, engine="DifferenceEquations")
simulation = Simulation(model)
results = simulation.run(num_steps=100)

print(simulation.simulation_outputs)  # ['A', 'B', 'C']
print(results.positions[-1], round(results["C"][-1]))  # 100.0 and the amount of C
results.plot.series(output_file="quickstart.png")  # one panel per bin
```

`results` maps each output name to its values, one per position from step 0
to step 100; `results.positions` holds the positions. `results.plot.series`
returns the matplotlib `Figure`, and saves it when you give `output_file`.

## What each call does

| Call | What it does |
|---|---|
| `ModelBuilder(name=..., version=...)` | Starts a model definition. Every method returns the builder, so calls chain. |
| `.add_bin(id="A", name="State A")` | Adds a compartment (a *bin*): a state that holds an amount, in the unit of the bin. |
| `.add_parameter(id="k1", value=0.3)` | Adds a named value for rate formulas. `value=None` leaves it for calibration. |
| `.add_transition(..., rate="k1 * A * B / total")` | Moves an amount from `source` to `target` along the axis. This rate names compartments, so it is the flow itself (*absolute*). |
| `.add_transition(..., rate="k2")` | This rate names no compartment, so it is *proportional*: the flow is `k2 × B`. |
| `Aggregate(bin_amounts=...)` | The *structure*: the amount each bin starts with. |
| `.build(structure, engine=...)` | Joins the *process* (what you wrote) with the structure and an engine, checks the definition and returns a `Model`. |
| `Simulation(model).run(num_steps=100)` | Runs the model in the Rust engine and returns its `SimulationResults`. |
| `results["B"]`, `results.positions` | The values of one output, and the positions they were read at. |
| `results.plot.series(...)` | Draws each series along the axis. |

See [Core Concepts](../guide/core-concepts.md) for bins, transitions and
[what a rate means](../guide/core-concepts.md#what-a-rate-means) in each
engine.

## One process, several engines

The process is written once. The same builder and structure build a model of
another engine: differential equations in continuous time, or a stochastic
process of whole units. `plot_compared` draws the runs on the same panels:

```python
from commol import plot_compared

differential = Simulation(process.build(structure, engine="DifferentialEquations"))
stochastic = Simulation(process.build(structure, engine="StochasticEvents"))

continuous = differential.run(num_steps=100)
replicates = stochastic.replicates(50, seed=1).run(num_steps=100)

print(len(replicates))  # 50 runs of whole units, one seed each
plot_compared(
    {"difference": results, "differential": continuous, "stochastic": replicates},
    series=["B"],
    output_file="engines.png",
)
```

A difference model applies each flow once per step; a differential model
integrates the flows; a stochastic engine moves whole units at random, so
`replicates` holds one run per seed and the figure shows their mean and band.
See [Running Simulations](../guide/simulations.md) and
[Stochastic Simulation](../guide/stochastic.md).

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
    .build(structure, engine="DifferenceEquations")
)

model_with_units.check_unit_consistency()  # raises UnitConsistencyError on a mismatch
```

See [Unit checking](../guide/units-and-equations.md#unit-checking).

## Calibrate unknown parameters

Set the unknown parameters to `None`, describe the observations and the
unknowns in a `CalibrationProblem`, and let an optimizer find the values:

```python
from commol import (
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    NelderMeadConfig,
    ObservedDataPoint,
)

observed_data = [
    ObservedDataPoint(at=step, output="B", value=value)
    for step, value in [(10, 45.2), (20, 78.5), (30, 62.3), (40, 38.1)]
]

calibration_model = (
    ModelBuilder(name="Calibration Model", version="1.0")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=None)
    .add_parameter(id="k2", value=None)
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .build(structure, engine="DifferenceEquations")
)

problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=42,
)

# A simulation of a model with a parameter at None can be calibrated, not run.
calibration = Simulation(calibration_model)
result = Calibrator(calibration, problem).run()
print({name: round(value, 3) for name, value in result.best_parameters.items()})

calibration_model.apply_calibration_parameters(result, problem)
fitted = calibration.run(num_steps=60)  # the simulation follows its model
fitted.plot.series(series=["B"], observed_data=observed_data, output_file="fit.png")
```

| Object | Role |
|---|---|
| `ObservedDataPoint` | One observation: a position (`at`), an output and a value. |
| `CalibrationParameter` | One unknown, with its bounds. |
| `CalibrationProblem` | The observations, the unknowns, the loss, the optimizer and the `seed` of its random numbers. |
| `Calibrator(...).run()` | Runs the optimizer in Rust and returns a `CalibrationResult` (`best_parameters`, `final_loss`, `converged`, …). |
| `Model.apply_calibration_parameters(result, problem)` | Writes the calibrated values into the model. |

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
- [ ] `Simulation(model).run(...)` returns one series per output, and
      `results.plot.series()` draws them.
- [ ] Another engine runs the same process when you change only `engine`.
- [ ] `check_unit_consistency()` passes when you give units.
- [ ] Calibration returns `best_parameters` within the bounds you set.

## Next steps

| To learn | Read |
|---|---|
| The concepts behind models | [Core Concepts](../guide/core-concepts.md) |
| Every builder option | [Building Models](../guide/building-models.md) |
| Running, reading and plotting runs, and comparing them | [Running Simulations](../guide/simulations.md) |
| Calibration in depth | [Model Calibration](../guide/calibration.md) |
| How sure a fit is: samples, predictive bands, scenarios | [Uncertainty](../guide/uncertainty/index.md) |
| How many threads the parallel work uses | [Threads and Parallelism](../guide/parallelism.md) |
| Complete models | [Examples](../guide/examples.md) |
