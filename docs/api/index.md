# API Reference

[Back to Home](../index.md)

Every public name of `commol`, generated from the docstrings. The guide pages
explain how to use them; these pages list every argument, attribute and error.

The table lists each name that the package root exports (`commol.__all__`),
on the page that documents it. Import these names from the package root, as
in `from commol import Simulation`.

| Page | Names |
|---|---|
| [ModelBuilder](model-builder.md) | `ModelBuilder` |
| [Model](model.md) | `Model`, `Process`, `ModelStructure`, `Aggregate`, `Complete`, `Groups`, `SeriesValue`, `ModelDescription` |
| [AxisPattern](axis-patterns.md) | `AxisPattern` |
| [AxisUnits](axis-units.md) | `AxisUnits` |
| [Simulation](simulation.md) | `Simulation`, `SimulationResults`, `ReplicateRuns`, `SimulationReplicates`, `ODESolver`, `StochasticEventsSettings`, `StochasticStepsSettings`, `InfeasibleStateError`, `windowed_totals`, `sample` |
| [Plotting](plotting.md) | `plot_compared`, `PlotConfig`; the methods `SimulationResults.plot`, `SimulationReplicates.plot`, `Predictive.plot`, `PRCCResult.plot`, `MorrisResult.plot`, `SobolResult.plot`, `ChaosResult.plot`, `ProfileResult.plot` |
| [Calibration](calibrator.md) | `Calibrator`, `CalibrationProblem`, `Quantile`, `ObservedDataPoint`, `CalibrationParameter`, `CalibrationConstraint`, `NelderMeadConfig`, `ParticleSwarmConfig`, `CalibrationResult`, `StartSummary`, `CalibrationHistory`, `CalibrationEvaluation` |
| [Observation models](calibrator.md#observation-models) | `ObservationModel`, `NormalObservations`, `LogNormalObservations`, `StudentTObservations`, `PoissonObservations`, `NegativeBinomialObservations`, `BinomialObservations` |
| [Reports](reports.md) | the methods `Model.applied_parameters`, `SimulationResults.report`, `SimulationReplicates.report`, `Predictive.report`, `IdentifiabilityReport.report` |
| [Uncertainty](uncertainty.md) | `InferenceMethod`, `UncertaintyMethod`, `ParameterSample`, `SampleMethod`, `ParameterSummary`, `Predictive`, `PredictiveBand`, `SeriesBand`, `ObservationDraws`, `Scenarios`, `ScenarioSummary`, `NoNoise`, `ResidualNoise`, `DistributionNoise` |
| [Resampling](uncertainty.md#resampling) | `BayesianBootstrap`, `ResidualBootstrap`, `WildBootstrap`, `BlockJackknife`, `ParametricBootstrap`, `BootstrapResult`, `JackknifeResult`, `ResamplingDiagnostics`, `BoundError` |
| [Profile likelihood](uncertainty.md#profile-likelihood) | `ProfileLikelihood`, `PredictionProfile`, `PredictionSpec`, `Threshold`, `ProfileResult`, `Profile`, `ProfileInterval` |
| [Sequential Monte Carlo](uncertainty.md#sequential-monte-carlo) | `Priors`, `Uniform`, `LogUniform`, `Beta`, `Normal`, `LogNormal`, `Gamma`, `GibbsPosterior`, `BayesianPosterior`, `InformationMatchingEta`, `BootstrapCoverageEta`, `ABCSMC`, `ABCRejection`, `LossDistance`, `SeriesDistance`, `SMCResult`, `GibbsResult`, `SMCDiagnostics` |
| [Local approximations](uncertainty.md#local-approximations) | `Laplace`, `Sandwich`, `LaplaceMixture`, `LikelihoodTarget`, `GibbsTarget`, `LocalResult`, `LaplaceResult`, `SandwichResult`, `DeltaBand`, `ImportanceResult`, `MixtureComponent` |
| [Conformal prediction](uncertainty.md#conformal-prediction) | `SplitConformal`, `RollingConformal`, `JackknifePlus`, `EnbPI`, `AdaptiveConformal`, `ConformalResult`, `PredictionIntervals`, `SeriesIntervals` |
| [Scores and intervals](uncertainty.md#scores-and-intervals) | `weighted_quantiles`, `member_statistics`, `ci_percentiles` |
| [Identifiability](identifiability.md) | `IdentifiabilityMethod`, `StructuralRank`, `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability`, `IdentifiabilityReport` |
| [Sensitivity](sensitivity.md) | `SensitivityAnalysis`, `InputSpace`, `SensitivityMethod`, `Propagation`, `PropagationResult`, `QuantitySummary`, `PRCC`, `PRCCResult`, `Morris`, `MorrisResult`, `Sobol`, `SobolResult`, `PolynomialChaos`, `ChaosResult`, `Level`, `Increments`, `ValueAt`, `Gain`, `Peak`, `PeakPosition`, `Trajectory`, `Ratio`, `quantity_schedule` |
| [Parallelism](parallelism.md) | `set_threads`, `threads`, `get_threads`, `get_levels` |
| [Package](package.md) | `__version__`, `build_info`, `BuildInfo` |

The pages also document names that are not in the package root:

- The scores are the module `commol.scores` (`from commol import scores`),
  on the [Uncertainty](uncertainty.md#scores-and-intervals) page.
- `LossFunction`, the values of `CalibrationProblem.loss_function`, is in
  `commol.context.constants`.
- `UnitConsistencyError` is in `commol.utils.equations`.
- Some types are only read from what a method returns, and you do not import
  them: `InitialCompartment` (an entry of `Aggregate.initial_state()`), the
  blocks of a `ModelDescription`, the parts of an `IdentifiabilityReport`, and
  the types of a model in [Model Structure](../advanced/model-structure.md).
