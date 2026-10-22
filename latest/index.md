# Commol

Commol builds, simulates and calibrates compartment models: amounts move between states (compartments) at rates you write as formulas. An amount can be a count, a mass or a concentration. You describe the model in Python; a Rust engine expands it over your stratifications, compiles every rate to machine code, and runs it or fits it to your data.

Before version 1.0

The API can change between minor versions (`0.2` → `0.3`). The [changelog](https://munqu.github.io/commol/latest/about/changelog/index.md) marks every change that breaks code as **Breaking**.

## Where to go

| You are…                       | Start with                                                                               | Then                                                                                                                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| New to Commol                  | [Quick Start](https://munqu.github.io/commol/latest/getting-started/quickstart/index.md) | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/index.md)                                                                                                        |
| Building a model               | [Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md)  | [Transitions](https://munqu.github.io/commol/latest/guide/transitions/index.md), [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md) |
| Fitting a model to data        | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)    | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)                                                                                                            |
| Looking up a class or argument | [API Reference](https://munqu.github.io/commol/latest/api/index.md)                      | [Model Structure](https://munqu.github.io/commol/latest/advanced/model-structure/index.md)                                                                                                 |
| Contributing to Commol         | [Codebase Guide](https://munqu.github.io/commol/latest/codebase/index.md)                | [Development Workflow](https://munqu.github.io/commol/latest/development/workflow/index.md)                                                                                                |
| An AI agent using Commol       | [`llms.txt`](https://munqu.github.io/commol/latest/llms.txt): an index of the site       | [`llms-full.txt`](https://munqu.github.io/commol/latest/llms-full.txt): the whole site in one file                                                                                         |
| An AI agent changing Commol    | [`AGENTS.md`](https://github.com/MUNQU/commol/blob/main/AGENTS.md)                       | [Codebase Guide](https://munqu.github.io/commol/latest/codebase/index.md)                                                                                                                  |

## How Commol works

```
flowchart LR
    A[ModelBuilder] -->|build| B[Model]
    B --> C[Simulation]
    C -->|run| D[Results]
    B --> E[CalibrationProblem]
    E --> F[Calibrator]
    F -->|run| G[CalibrationResult]
    G -->|apply_calibration_parameters| B
```

1. **Build** a `Model` with `ModelBuilder`: bins, parameters, transitions, stratifications and initial conditions. Commol validates it as you go.
1. **Simulate** it with `Simulation`: one series per compartment and accumulator.
1. **Calibrate** unknown values against observations with `Calibrator`, and write the result back into the model.

## Quick example

```python
from commol import Aggregate, ModelBuilder, Simulation

model = (
    ModelBuilder(name="Basic Model")
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

results = Simulation(model).run(num_steps=100)
```

## Features

| Feature                                                                                                | Guide                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Fluent, validated model building                                                                       | [Building Models](https://munqu.github.io/commol/latest/guide/building-models/index.md)                                                                |
| Stratifications, conditional stratifications, per-compartment initial conditions                       | [Core Concepts](https://munqu.github.io/commol/latest/guide/core-concepts/#stratifications)                                                            |
| Rate formulas compiled with Cranelift                                                                  | [Mathematical Expressions](https://munqu.github.io/commol/latest/guide/mathematical-expressions/index.md)                                              |
| Pulses, windows, ramps, sinusoids and schedules per subgroup                                           | [Axis Patterns](https://munqu.github.io/commol/latest/guide/axis-patterns/index.md)                                                                    |
| Unit checking                                                                                          | [Units](https://munqu.github.io/commol/latest/guide/units-and-equations/index.md)                                                                      |
| One process run as difference equations, differential equations or a stochastic process of whole units | [Stochastic Simulation](https://munqu.github.io/commol/latest/guide/stochastic/index.md)                                                               |
| A description of a model: equations, state graphs, the ODD protocol (text, Markdown, LaTeX)            | [Describing a Model](https://munqu.github.io/commol/latest/guide/describing-models/index.md)                                                           |
| Calibration of parameters, initial conditions and scale factors                                        | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)                                                                  |
| Parameter samples, predictive bands and scores                                                         | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)                                                                        |
| JSON-ready reports and matplotlib figures                                                              | [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md), [Plotting](https://munqu.github.io/commol/latest/guide/simulations/#plotting) |
| Parallel starts, replicates, particles and draws, on the threads you choose                            | [Threads and Parallelism](https://munqu.github.io/commol/latest/guide/parallelism/index.md)                                                            |

## License and authors

Commol is licensed under the MIT License; see [License](https://munqu.github.io/commol/latest/about/license/index.md). It is developed by the MUNQU team: Rafael J. Villanueva Micó, Carlos Andreu Vilarroig and David Martínez Rodríguez.
