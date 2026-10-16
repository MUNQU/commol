# API Reference

[Back to Home](../index.md)

Every public name of `commol`, generated from the docstrings. The guide pages
explain how to use them; these pages list every argument, attribute and error.

| Page | Names |
|---|---|
| [ModelBuilder](model-builder.md) | `ModelBuilder` |
| [Model](model.md) | `Model`, `Process`, `ModelStructure`, `Aggregate`, `Complete`, `Groups`, `ModelDescription`, `InitialCompartment`, `UnitConsistencyError` |
| [Simulation](simulation.md) | `Simulation`, `SimulationResults`, `ReplicateRuns`, `SimulationReplicates`, `OdeSolver`, `StochasticEventsSettings`, `StochasticStepsSettings`, `InfeasibleStateError`, `window_start`, `window_end`, `window_index`, `window_ends`, `windowed_totals` |
| [Plotting](plotting.md) | `SimulationPlotter`, `PlotConfig` |
| [AxisPattern](axis-patterns.md) | `AxisPattern` |
| [AxisUnits](axis-units.md) | `AxisUnits` |
| [Calibration](calibrator.md) | `Calibrator`, `CalibrationProblem`, `Quantile`, `ObservedDataPoint`, `CalibrationParameter`, `CalibrationConstraint`, `NelderMeadConfig`, `ParticleSwarmConfig`, `CalibrationResult`, `StartSummary`, `CalibrationHistory`, `CalibrationEvaluation`, `LossFunction`, `OptimizationAlgorithm` |
| [Uncertainty](uncertainty.md) | `InferenceMethod`, `UncertaintyMethod`, `ParameterSample`, `SampleMethod`, `ParameterSummary`, `Predictive`, `PredictiveBand`, `SeriesBand`, `ObservationDraws`, `commol.scores`, `weighted_quantiles`, `member_statistics`, `ci_percentiles` |
| [Identifiability](identifiability.md) | `IdentifiabilityMethod`, `StructuralRank`, `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability`, `IdentifiabilityReport` |
| [Reports](reports.md) | `applied_parameters_report`, `simulation_report`, `identifiability_report` |

Import every name from the package root, as in `from commol import Simulation`.
`UnitConsistencyError` is in `commol.utils.equations`; the scores are the
module `commol.scores` (`from commol import scores`).
