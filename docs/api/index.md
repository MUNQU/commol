# API Reference

[Back to Home](../index.md)

Every public name of `commol`, generated from the docstrings. The guide pages
explain how to use them; these pages list every argument, attribute and error.

| Page | Names |
|---|---|
| [ModelBuilder](model-builder.md) | `ModelBuilder` |
| [Model](model.md) | `Model`, `Process`, `ModelStructure`, `Aggregate`, `Complete`, `Groups`, `ModelDescription`, `InitialCompartment`, `UnitConsistencyError` |
| [Simulation](simulation.md) | `Simulation`, `SimulationResults`, `ReplicateRuns`, `SimulationReplicates`, `OdeSolver`, `InfeasibleStateError`, `window_start`, `window_end`, `window_index`, `window_ends`, `windowed_totals` |
| [Plotting](plotting.md) | `SimulationPlotter`, `PlotConfig` |
| [AxisPattern](axis-patterns.md) | `AxisPattern` |
| [AxisUnits](axis-units.md) | `AxisUnits` |
| [Calibration](calibrator.md) | `Calibrator`, `CalibrationProblem`, `Quantile`, `ObservedDataPoint`, `CalibrationParameter`, `CalibrationConstraint`, `NelderMeadConfig`, `ParticleSwarmConfig`, `CalibrationResult`, `LossFunction`, `OptimizationAlgorithm` |
| [Probabilistic Calibration](probabilistic-calibrator.md) | `ProbabilisticCalibrationConfig`, `ProbEvaluationFilterConfig`, `ProbClusteringConfig`, `ProbRepresentativeConfig`, `ProbNsga2Config`, `ProbGreedyLocalSearchConfig`, `ProbabilisticCalibrationResult`, `EnsembleSolution`, `EnsembleSelectionSummary`, `ParameterSetStatistics`, `CalibrationEvaluation`, `ci_percentiles`, `member_statistics` |
| [Reports](reports.md) | `applied_parameters_report`, `simulation_report` |

Import every name from the package root, as in `from commol import Simulation`.
`UnitConsistencyError` is in `commol.utils.equations`.
