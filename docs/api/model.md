# Model API

[Back to API Reference](index.md)

The validated model that `ModelBuilder.build` returns: a header, the process
written once, the structure that lays it out (`Aggregate`, with its neighbors
`Complete` and `Groups`), the entries of `Aggregate.initial_state()`, the
`SeriesValue` of a series parameter, and `UnitConsistencyError`, the error of
`Model.check_unit_consistency`.
Guide: [Core Concepts](../guide/core-concepts.md),
[Building Models](../guide/building-models.md#the-structure-aggregate),
[Describing a Model](../guide/describing-models.md),
[Units](../guide/units-and-equations.md). The tree of types is in
[Model Structure](../advanced/model-structure.md).

::: commol.Model

::: commol._model.process.Process

::: commol._model.structure.ModelStructure

::: commol.Aggregate

::: commol.Complete

::: commol.Groups

::: commol._model.engine_state.InitialCompartment

::: commol.SeriesValue

::: commol._model.unit_checks.UnitConsistencyError

## Descriptions

`Model.describe` returns a `ModelDescription`; its blocks are in
`commol._model.description`.

::: commol._model.description.ModelDescription

::: commol._model.description.Section

::: commol._model.description.Text

::: commol._model.description.Table

::: commol._model.description.Equations

::: commol._model.description.EquationGroup

::: commol._model.description.Equation

::: commol._model.description.StateGraph

::: commol._model.description.GraphNode

::: commol._model.description.GraphEdge

::: commol._model.description.Matrix
