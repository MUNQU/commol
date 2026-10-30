# Reports API

[Back to API Reference](index.md)

Plain-dictionary reports, each a method of the object whose data it
describes. Guide: [Reports](../guide/reports.md).

::: commol.Model.applied_parameters

::: commol.SimulationResults.report

::: commol.SimulationReplicates.report

::: commol.uncertainty.Predictive.report

::: commol.identifiability.IdentifiabilityReport.report

## The types of the reports

Each report is a `TypedDict`: a plain dictionary whose keys a type checker
knows. The root `commol` exports the types of `Model.applied_parameters`, of
`Aggregate.report_sections` and of a simulation report; the [Identifiability API](identifiability.md) holds
the types of the identifiability report.

::: commol.AppliedParametersReport

::: commol.ParameterEntry

::: commol.EntrySpread

::: commol.InitialAmountEntry

::: commol.CategorySplitEntry

::: commol.SimulationReport

::: commol.ObservedSeries

::: commol.AccumulatorReport

::: commol.WindowedAccumulator

::: commol.PredictionIntervalsReport

::: commol.SeriesIntervalsReport
