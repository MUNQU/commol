# Model Structure

[Back to API Reference](../api/index.md)

A `Model` is a tree of Pydantic models. `ModelBuilder` creates it, `Model.to_json`
writes it, and the Rust engine reads the same structure from JSON
(`crates/commol-core/src/types/`). Most users never build these types by hand;
read this page to understand model files or to work on Commol.

```text
Model                      name, description, version
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
├── parameters: [Parameter]                (a value may be a TimeSeriesValue)
└── dynamics: Dynamics
    ├── typology: "DifferenceEquations"
    └── transitions: [Transition]
        ├── stratified_rates: [StratifiedRate]
        │   └── conditions: [StratificationCondition]
        └── condition: Condition           (stored, not evaluated)
            └── rules: [Rule]
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

::: commol.context.parameter.TimeSeriesValue

## Dynamics

::: commol.context.dynamics.Dynamics

::: commol.context.dynamics.Transition

::: commol.context.dynamics.StratifiedRate

::: commol.context.dynamics.Condition

::: commol.context.dynamics.Rule

## Initial conditions

::: commol.context.initial_conditions.InitialConditions

::: commol.context.initial_conditions.BinFraction

::: commol.context.initial_conditions.StratificationFractions

::: commol.context.initial_conditions.StratificationFraction

::: commol.context.initial_conditions.CompartmentFraction
