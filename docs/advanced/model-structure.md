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
│   ├── matrices: [CategoryMatrix]         (values: {row: {column: value}}, row_scales, column_scales)
│   └── transitions: [Transition]          (impulse_at, impulse_every for impulses)
│       └── stratified_rates: [StratifiedRate]
│           └── conditions: [StratificationCondition]
└── structure: Aggregate                   (kind: "aggregate", variant: None)
    ├── bin_amounts: {bin: amount | None}          (in the unit of the bin)
    ├── stratification_fractions: {stratification: {category: fraction}
    │                                  | {"by": stratification, category: {category: fraction}}}
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
  "process": {"bins": [], "accumulators": [], "stratifications": [], "matrices": [], "transitions": []},
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

## The parts of a model

`Model.process` holds the parts that the builder makes, and `Model.parameters`
holds the parameters. The root `commol` exports the class of each part, so a
typed function that reads a part names its class. The
[Model API](../api/model.md#the-parts-of-a-model) documents them.

| Part | Added by | Read with |
|---|---|---|
| `Bin`, `Accumulator` | `ModelBuilder.add_bin`, `ModelBuilder.add_accumulator` | `Process.bins`, `Process.accumulators` |
| `Stratification` | `ModelBuilder.add_stratification`, with a `StratificationCondition` ([ModelBuilder API](../api/model-builder.md)) | `Process.stratifications` |
| `CategoryMatrix` | `ModelBuilder.add_matrix`; a rate reads it through `matrix_sum` ([Matrix sums](../guide/mathematical-expressions.md#matrix-sums)) | `Process.matrices` |
| `Parameter` | `ModelBuilder.add_parameter`; a series parameter holds a `SeriesValue` ([Model API](../api/model.md)) | `Model.parameters` |
| `Transition` | `ModelBuilder.add_transition`, with a `StratifiedRate` ([ModelBuilder API](../api/model-builder.md)) | `Process.transitions` |

## Engine

The engine's list of engines (`crates/commol-core/src/types/engine.rs`)
validates the settings: settings of another engine raise an error that names
both. `Model.engine` is a `ModelEngine`, which the
[Engines API](../api/engines.md) documents with `ODESolver`.

## Initial conditions

`Aggregate` holds the fields and methods of `InitialConditions`, and checks
them against the process when the model is built.

::: commol._model.initial_conditions.InitialConditions
