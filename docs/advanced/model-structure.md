# Model Structure

[Back to API Reference](../api/index.md)

A `Model` is a tree of Pydantic models. `ModelBuilder` creates it, `Model.to_json`
writes it, and the Rust engine reads the same tree from JSON
(`crates/commol-core/src/types/`). Most users never build these types by hand;
read this page to understand model files or to work on Commol.

A model has a header, a **process** written once, a **structure** that lays the
process out, and an **engine** that runs it. The process (bins, accumulators,
stratifications, transitions) is the same for every kind of model; the
structure changes between kinds and is tagged in JSON by `kind`, and by
`variant` for a narrower form of a structure. The one kind is `aggregate`:
amounts per compartment.

```text
Model                      name, description, version, step_unit, origin
├── parameters: [Parameter]                (a value may be a SeriesValue)
├── engine: ModelEngine
│   ├── type: "DifferenceEquations", "DifferentialEquations",
│   │         "StochasticEvents" or "StochasticSteps"
│   └── settings: ODESolver | StochasticEventsSettings | StochasticStepsSettings | None
│                                           (settings of that engine; None: its defaults)
├── process: Process
│   ├── bins: [Bin]
│   ├── accumulators: [Accumulator]
│   ├── stratifications: [Stratification]
│   │   └── conditions: [StratificationCondition]
│   └── transitions: [Transition]          (impulse_at, impulse_every for impulses)
│       └── stratified_rates: [StratifiedRate]
│           └── conditions: [StratificationCondition]
└── structure: Aggregate                   (kind: "aggregate", variant: None)
    ├── bin_amounts: {bin: amount | None}          (in the unit of the bin)
    ├── stratification_fractions: {stratification: {category: fraction}}
    ├── compartment_amounts: {compartment: amount}
    └── neighbors: Complete | Groups | None
```

In JSON the structure is an object with its `kind`, its `variant` and the
fields of its class:

```json
{
  "name": "Model",
  "parameters": [],
  "engine": {"type": "DifferenceEquations", "settings": null},
  "process": {"bins": [], "accumulators": [], "stratifications": [], "transitions": []},
  "structure": {
    "kind": "aggregate",
    "variant": null,
    "bin_amounts": {"A": 990.0, "B": 10.0},
    "stratification_fractions": {},
    "compartment_amounts": {},
    "neighbors": null
  },
  "step_unit": null,
  "origin": null
}
```

`Model`, `Process`, `ModelStructure`, `Aggregate`, `Complete` and `Groups` are
documented in [Model API](../api/model.md). The engine must run the kind of the
structure: a model whose engine runs another kind is rejected when it is made,
with an error that names the engine, both kinds and the engines of the kind of
the model. A variant keeps the kind, the fields and the engines of its
structure, and loads back as its own class
([Design decisions](../codebase/decisions/model-kinds.md#a-model-is-a-header-a-process-a-structure-and-an-engine)).

## Bins and accumulators

::: commol.context.bin.Bin

::: commol.context.bin.Accumulator

## Stratifications

::: commol.context.stratification.Stratification

A condition of a stratification or of a stratified rate,
`StratificationCondition`, is documented in
[ModelBuilder API](../api/model-builder.md).

## Parameters

::: commol.context.parameter.Parameter

The value of a series parameter, `SeriesValue`, is documented in
[Model API](../api/model.md).

## Transitions

::: commol.context.transition.Transition

The rate of a transition in the compartments that match its conditions,
`StratifiedRate`, is documented in [ModelBuilder API](../api/model-builder.md).

## Engine

The engine's list of engines (`crates/commol-core/src/types/engine.rs`)
validates the settings: settings of another engine raise an error that names
both. `ODESolver` is documented in [Simulation API](../api/simulation.md).

::: commol.context.engine.ModelEngine

## Initial conditions

`Aggregate` holds the fields and methods of `InitialConditions`, and checks
them against the process when the model is built.

::: commol.context.initial_conditions.InitialConditions
