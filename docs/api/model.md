# Model API

[Back to API Reference](index.md)

The validated model that `ModelBuilder.build` returns: a header, the process
written once, the structure that lays it out (`Aggregate`, with its neighbors
`Complete` and `Groups`), the entries of `Aggregate.initial_state()`, the
`SeriesValue` of a series parameter, the parts of the process, and
`UnitConsistencyError`, the error of `Model.check_unit_consistency`.
Guide: [Core Concepts](../guide/core-concepts.md),
[Building Models](../guide/building-models.md#the-structure-aggregate),
[Describing a Model](../guide/describing-models.md),
[Units](../guide/units-and-equations.md). The tree of types is in
[Model Structure](../advanced/model-structure.md).

::: commol.Model

::: commol.Process

::: commol._model.structure.ModelStructure

::: commol.Aggregate

::: commol.Complete

::: commol.Groups

::: commol.InitialCompartment

::: commol.SeriesLabel

::: commol.ReportSections

::: commol.SeriesValue

::: commol._model.unit_checks.UnitConsistencyError

## The parts of a model

The parts that the builder makes, which `Model.process` and
`Model.parameters` hold. [Model Structure](../advanced/model-structure.md#the-parts-of-a-model)
names the method that adds each one.

::: commol.Bin

::: commol.Accumulator

::: commol.Stratification

::: commol.CategoryMatrix

::: commol.Parameter

::: commol.Transition

## Descriptions

`Model.describe` returns a `ModelDescription`, a list of sections of
blocks.

::: commol.ModelDescription

::: commol.Section

::: commol.Text

::: commol.Table

::: commol.Equations

::: commol.EquationGroup

::: commol.Equation

::: commol.StateGraph

::: commol.GraphNode

::: commol.GraphEdge

::: commol.Matrix
