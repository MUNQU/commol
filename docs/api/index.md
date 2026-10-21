# API Reference

[Back to Home](../index.md)

Every public name of `commol`, generated from the docstrings. The guide pages
explain how to use them; these pages list every argument, attribute and error.

| Page | Names |
|---|---|
| [ModelBuilder](model-builder.md) | `ModelBuilder` |
| [Model](model.md) | `Model`, `Process`, `ModelStructure`, `Aggregate`, `Complete`, `Groups`, `ModelDescription`, `InitialCompartment`, `SeriesValue`, `UnitConsistencyError` |
| [Simulation](simulation.md) | `Simulation`, `SimulationResults`, `ReplicateRuns`, `SimulationReplicates`, `ODESolver`, `StochasticEventsSettings`, `StochasticStepsSettings`, `InfeasibleStateError`, `windowed_totals`, `sample` |
| [Plotting](plotting.md) | `SimulationPlotter`, `PlotConfig` |
| [AxisPattern](axis-patterns.md) | `AxisPattern` |
| [AxisUnits](axis-units.md) | `AxisUnits` |
| [Calibration](calibrator.md) | `Calibrator`, `CalibrationProblem`, `Quantile`, `ObservedDataPoint`, `CalibrationParameter`, `CalibrationConstraint`, `NelderMeadConfig`, `ParticleSwarmConfig`, `CalibrationResult`, `StartSummary`, `CalibrationHistory`, `CalibrationEvaluation`, `LossFunction`, `OptimizationAlgorithm` |
| [Uncertainty](uncertainty.md) | `InferenceMethod`, `UncertaintyMethod`, `ParameterSample`, `SampleMethod`, `ParameterSummary`, `Predictive`, `PredictiveBand`, `SeriesBand`, `ObservationDraws`, `commol.scores`, `weighted_quantiles`, `member_statistics`, `ci_percentiles` |
| [Sensitivity](sensitivity.md) | `SensitivityAnalysis`, `InputSpace`, `SensitivityMethod`, `Propagation`, `PropagationResult`, `QuantitySummary`, `PRCC`, `PRCCResult`, `Morris`, `MorrisResult`, `Sobol`, `SobolResult`, `PolynomialChaos`, `ChaosResult`, `ValueAt`, `Gain`, `Peak`, `PeakPosition`, `Trajectory` |
| [Identifiability](identifiability.md) | `IdentifiabilityMethod`, `StructuralRank`, `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability`, `IdentifiabilityReport` |
| [Reports](reports.md) | `Model.applied_parameters`, `SimulationResults.report`, `SimulationReplicates.report`, `Predictive.report`, `IdentifiabilityReport.report` |

Import every name from the package root, as in `from commol import Simulation`.
`UnitConsistencyError` is in `commol.utils.equations`; the scores are the
module `commol.scores` (`from commol import scores`).
