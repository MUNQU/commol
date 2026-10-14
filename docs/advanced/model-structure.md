# Model Structure

[Back to API Reference](../api/index.md)

A `Model` is a tree of Pydantic models. `ModelBuilder` creates it, `Model.to_json`
writes it, and the Rust engine reads the same structure from JSON
(`crates/commol-core/src/types/`). Most users never build these types by hand;
read this page to understand model files or to work on Commol.

A model has a header that every kind of model has, an engine, and a
`structure`: the part of its kind, tagged in JSON by `kind`. The only kind is
`compartmental`, which `ModelBuilder` builds.

```text
Model                      name, description, version, step_unit
├── parameters: [Parameter]                (a value may be a SeriesValue)
├── engine: ModelEngine
│   ├── type: "DifferenceEquations" or "DifferentialEquations"
│   └── settings: OdeSolver | None          (settings of that engine; None: its defaults)
└── structure: CompartmentalStructure      (kind: "compartmental")
    ├── bins: [Bin]
    ├── accumulators: [Accumulator]
    ├── stratifications: [Stratification]
    │   └── conditions: [StratificationCondition]
    ├── transitions: [Transition]          (impulse_at, impulse_every for impulses)
    │   └── stratified_rates: [StratifiedRate]
    │       └── conditions: [StratificationCondition]
    └── initial_conditions: InitialConditions
        ├── bin_fractions: [BinFraction]
        ├── stratification_fractions: [StratificationFractions]
        │   └── fractions: [StratificationFraction]
        └── compartment_fractions: [CompartmentFraction]
```

In JSON the structure is an object with its `kind` and the fields of its
class:

```json
{
  "name": "Model",
  "parameters": [],
  "engine": {"type": "DifferenceEquations", "settings": null},
  "structure": {"kind": "compartmental", "bins": [], "...": "..."},
  "step_unit": null
}
```

`Model` itself is documented in [Model API](../api/model.md). The engine must
run the kind of the structure: a model whose engine runs another kind is
rejected when it is made, with an error that names the engine and both kinds.

## Compartmental structure

`CompartmentalStructure` and `ModelStructure`, the interface every kind gives,
are documented in [Model API](../api/model.md).

::: commol.context.bin.Bin

::: commol.context.bin.Accumulator

## Stratifications

::: commol.context.stratification.Stratification

::: commol.context.stratification_condition.StratificationCondition

## Parameters

::: commol.context.parameter.Parameter

::: commol.context.parameter.SeriesValue

## Transitions

::: commol.context.transition.Transition

::: commol.context.transition.StratifiedRate

## Engine

The engine's list of engines (`crates/commol-core/src/types/engine.rs`)
validates the settings: settings of another engine raise an error that names
both. `OdeSolver` is documented in [Simulation API](../api/simulation.md).

::: commol.context.engine.ModelEngine

## Initial conditions

::: commol.context.initial_conditions.InitialConditions

::: commol.context.initial_conditions.BinFraction

::: commol.context.initial_conditions.StratificationFractions

::: commol.context.initial_conditions.StratificationFraction

::: commol.context.initial_conditions.CompartmentFraction
