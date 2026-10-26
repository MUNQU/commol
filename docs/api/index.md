# API Reference

[Back to Home](../index.md)

Every public name of `commol`, generated from the docstrings. The guide pages
explain how to use them; these pages list every argument, attribute and error.

The table lists each name that the package root exports (`commol.__all__`),
on the page that documents it. Import these names from the package root, as
in `from commol import Simulation`. A name you only read is not exported: a
result is typed as the return of the member that gives it, and it is on the
page of that object, as `SimulationResults` is on the Simulation page. A
choice among names is a string, such as `loss_function="sse"`.

| Page | Names |
|---|---|
| [ModelBuilder](model-builder.md) | `ModelBuilder`, `StratificationCondition`, `StratifiedRate` |
| [Model](model.md) | `Model`, `Aggregate`, `Complete`, `Groups`, `SeriesValue`; the results `Process`, `ModelDescription` and its blocks |
| [AxisPattern](axis-patterns.md) | `AxisPattern` |
| [AxisUnits](axis-units.md) | `AxisUnits` |
| [Simulation](simulation.md) | `Simulation`, `ODESolver`, `StochasticEventsSettings`, `StochasticStepsSettings`, `InfeasibleStateError`; the results `SimulationResults`, `ReplicateRuns`, `SimulationReplicates` and their figures |
| [Plotting](plotting.md) | `plot_compared`, `PlotConfig`; the methods `PRCCResult.plot`, `MorrisResult.plot`, `SobolResult.plot`, `ChaosResult.plot`, `ProfileResult.plot` |
| [Calibration](calibrator.md) | `Calibrator`, `CalibrationProblem`, `Quantile`, `ObservedDataPoint`, `CalibrationParameter`, `CalibrationConstraint`, `NelderMeadConfig`, `ParticleSwarmConfig`; the results `CalibrationResult`, `StartSummary`, `CalibrationHistory`, `CalibrationEvaluation`, `DeltaBand` |
| [Observation models](calibrator.md#observation-models) | `NormalObservations`, `LogNormalObservations`, `StudentTObservations`, `PoissonObservations`, `NegativeBinomialObservations`, `BinomialObservations` |
| [Reports](reports.md) | the methods `Model.applied_parameters`, `SimulationResults.report`, `SimulationReplicates.report`, `Predictive.report`, `IdentifiabilityReport.report` |
| [Uncertainty](uncertainty.md) | `ParameterSample`, `SampleMethod`, `Predictive`, `Scenarios`, `NoNoise`, `ResidualNoise`, `DistributionNoise`; the results `ParameterSummary`, `PredictiveBand`, `SeriesBand`, `ObservationDraws`, `ScenarioSummary` and the figures of a predictive |
| [Resampling](uncertainty.md#resampling) | `BayesianBootstrap`, `ResidualBootstrap`, `WildBootstrap`, `BlockJackknife`, `ParametricBootstrap`; the results `BootstrapResult`, `JackknifeResult`, `ResamplingDiagnostics`, `BoundError` |
| [Profile likelihood](uncertainty.md#profile-likelihood) | `ProfileLikelihood`, `PredictionProfile`, `PredictionSpec`, `Threshold`; the results `ProfileResult`, `Profile`, `ProfileInterval`, `ProfileStructure` |
| [Sequential Monte Carlo](uncertainty.md#sequential-monte-carlo) | `Priors`, `Uniform`, `LogUniform`, `Beta`, `Normal`, `LogNormal`, `Gamma`, `GibbsPosterior`, `BayesianPosterior`, `InformationMatchingEta`, `BootstrapCoverageEta`, `ABCSMC`, `ABCRejection`, `LossDistance`, `SeriesDistance`; the results `SMCResult`, `GibbsResult`, `SMCDiagnostics` |
| [Local approximations](uncertainty.md#local-approximations) | `Laplace`, `Sandwich`, `LaplaceMixture`, `LikelihoodTarget`, `GibbsTarget`; the results `LocalResult`, `LaplaceResult`, `SandwichResult`, `ImportanceResult`, `MixtureComponent` |
| [Conformal prediction](uncertainty.md#conformal-prediction) | `SplitConformal`, `RollingConformal`, `JackknifePlus`, `EnbPI`, `AdaptiveConformal`; the results `ConformalResult`, `PredictionIntervals`, `SeriesIntervals` |
| [Scores and intervals](uncertainty.md#scores-and-intervals) | `weighted_quantiles`, `member_statistics` |
| [Identifiability](identifiability.md) | `StructuralRank`, `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability`; the result `IdentifiabilityReport` and its parts |
| [Sensitivity](sensitivity.md) | `SensitivityAnalysis`, `InputSpace`, `Propagation`, `PRCC`, `Morris`, `Sobol`, `PolynomialChaos`, `Level`, `Increments`, `ValueAt`, `Gain`, `Peak`, `PeakPosition`, `Trajectory`, `Ratio`, `quantity_schedule`; the results `PropagationResult`, `QuantitySummary`, `PRCCResult`, `MorrisResult`, `SobolResult`, `ChaosResult` |
| [Parallelism](parallelism.md) | `set_threads`, `threads`, `get_threads`, `get_levels` |
| [Package](package.md) | `__version__`, `build_info`; the result `BuildInfo` |

The pages also document names that are not in the package root:

- The scores are the module `commol.scores` (`from commol import scores`),
  on the [Uncertainty](uncertainty.md#scores-and-intervals) page.
- `UnitConsistencyError` is in `commol.utils.equations`.
- The types of a model, such as `ModelStructure`, the type of
  `Model.structure`, are in [Model Structure](../advanced/model-structure.md).
