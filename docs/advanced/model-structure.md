# Model Structure

[Back to API Reference](../api/index.md)

A `Model` is a tree of Pydantic models. `ModelBuilder` creates it, `Model.to_json`
writes it, and the Rust engine reads the same structure from JSON
(`crates/commol-core/src/types/`). Most users never build these types by hand;
read this page to understand model files or to work on Commol.

```text
Model                      name, description, version, step_unit
├── population: Population
│   ├── bins: [Bin]
│   ├── accumulators: [Accumulator]
│   ├── stratifications: [Stratification]
│   │   └── conditions: [StratificationCondition]
│   ├── transitions: [Transition]          (the engine reads dynamics.transitions)
│   └── initial_conditions: InitialConditions
│       ├── bin_fractions: [BinFraction]
│       ├── stratification_fractions: [StratificationFractions]
│       │   └── fractions: [StratificationFraction]
│       └── compartment_fractions: [CompartmentFraction]
├── parameters: [Parameter]                (a value may be a SeriesValue)
├── dynamics: Dynamics
│   └── transitions: [Transition]          (impulse_at, impulse_every for impulses)
│       └── stratified_rates: [StratifiedRate]
│           └── conditions: [StratificationCondition]
└── engine: ModelEngine
    ├── type: "DifferenceEquations" or "DifferentialEquations"
    └── settings: OdeSolver | None          (settings of that engine; None: its defaults)
```

`Model` itself is documented in [Model API](../api/model.md).

## Population

::: commol.context.population.Population

::: commol.context.bin.Bin

::: commol.context.bin.Accumulator

## Stratifications

::: commol.context.stratification.Stratification

::: commol.context.stratification_condition.StratificationCondition

## Parameters

::: commol.context.parameter.Parameter

::: commol.context.parameter.SeriesValue

## Dynamics

::: commol.context.dynamics.Dynamics

::: commol.context.dynamics.Transition

::: commol.context.dynamics.StratifiedRate

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
