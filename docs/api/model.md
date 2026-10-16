# Model API

[Back to API Reference](index.md)

The validated model that `ModelBuilder.build` returns: a header, the process
written once, the structure that lays it out (`Aggregate`, with its neighbors
`Complete` and `Groups`), and the entries of `Aggregate.initial_state()`.
Guide: [Core Concepts](../guide/core-concepts.md),
[Building Models](../guide/building-models.md#the-structure-aggregate),
[Describing a Model](../guide/describing-models.md),
[Units](../guide/units-and-equations.md). The tree of types is in
[Model Structure](../advanced/model-structure.md).

::: commol.context.model.Model

::: commol.context.process.Process

::: commol.context.structure.ModelStructure

::: commol.context.aggregate.Aggregate

::: commol.context.neighbors.Complete

::: commol.context.neighbors.Groups

::: commol.context._model.engine_state.InitialCompartment

## Descriptions

`Model.describe` returns a `ModelDescription`; its blocks are in
`commol.context.description`.

::: commol.context.description.ModelDescription

::: commol.context.description.Section

::: commol.context.description.Text

::: commol.context.description.Table

::: commol.context.description.Equations

::: commol.context.description.EquationGroup

::: commol.context.description.Equation

::: commol.context.description.StateGraph

::: commol.context.description.GraphNode

::: commol.context.description.GraphEdge

::: commol.context.description.Matrix

::: commol.utils.equations.UnitConsistencyError
