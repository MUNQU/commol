# Model API

[Back to API Reference](index.md)

The validated model that `ModelBuilder.build` returns: a header, the process
written once, the structure that lays it out (`Aggregate`, with its neighbors
`Complete` and `Groups`), and the entries of `Aggregate.initial_state()`.
Guide: [Core Concepts](../guide/core-concepts.md),
[Building Models](../guide/building-models.md#the-structure-aggregate),
[Units and Equations](../guide/units-and-equations.md). The tree of types is in
[Model Structure](../advanced/model-structure.md).

::: commol.context.model.Model

::: commol.context.process.Process

::: commol.context.structure.ModelStructure

::: commol.context.aggregate.Aggregate

::: commol.context.neighbors.Complete

::: commol.context.neighbors.Groups

::: commol.context._model.engine_state.InitialCompartment

::: commol.utils.equations.UnitConsistencyError
