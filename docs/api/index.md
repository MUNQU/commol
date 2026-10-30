# API Reference

[Back to Home](../index.md)

Every public name of `commol`, generated from the docstrings. The guide pages
explain how to use them; these pages list every argument, attribute and error.

The package root `commol` holds the main flow: build a model, choose its
structure, simulate it and calibrate it, as in `from commol import
Simulation`. Every other name lives in the module of its page of the guide,
as in `from commol.resampling import BayesianBootstrap`; each module has its
page here. Every class that a public member returns is exported too, so a
typed function names it, as in `def fit(...) -> CalibrationResult` with
`from commol import CalibrationResult`. A result of the main flow is in the
root, the result of a method in the module of the method, and it is on the
page of the object that returns it, as `SimulationResults` is on the
Simulation page. A result is the return of its member, not a class to build.
A choice among names is a string, such as `loss_function="sse"`.

## The root `commol`

| Page | Names |
|---|---|
| [ModelBuilder](model-builder.md) | `ModelBuilder`, `StratificationCondition`, `StratifiedRate` |
| [Model](model.md) | `Model`, `Aggregate`, `Complete`, `Groups`, `SeriesValue`; the results `Process`, `InitialCompartment`, `SeriesLabel`, `ReportSections`; the parts `Bin`, `Accumulator`, `Stratification`, `CategoryMatrix`, `Parameter`, `Transition`; the description `ModelDescription` and its blocks `Section`, `Text`, `Table`, `Equations`, `EquationGroup`, `Equation`, `StateGraph`, `GraphNode`, `GraphEdge`, `Matrix` |
| [AxisPattern](axis-patterns.md) | `AxisPattern` |
| [AxisUnits](axis-units.md) | `AxisUnits` |
| [Simulation](simulation.md) | `Simulation`, `InfeasibleStateError`; the results `SimulationResults`, `ReplicateRuns`, `SimulationReplicates` and their figures `ResultsPlot`, `ReplicatesPlot` |
| [Calibration](calibrator.md) | `Calibrator`, `CalibrationProblem`, `Quantile`, `ObservedDataPoint`, `CalibrationParameter`, `CalibrationConstraint`, `NelderMeadConfig`, `ParticleSwarmConfig`; the results `CalibrationResult`, `StartSummary`, `CalibrationHistory`, `CalibrationEvaluation` |
| [Reports](reports.md) | the methods `Model.applied_parameters`, `SimulationResults.report`, `SimulationReplicates.report`, `Predictive.report`, `IdentifiabilityReport.report`; the types `AppliedParametersReport`, `ParameterEntry`, `Interval`, `SimulationReport`, `ObservedSeries`, `AccumulatorReport`, `WindowedAccumulator`, `PredictionIntervalsReport`, `SeriesIntervalsReport` |
| [Package](package.md) | `__version__`, `build_info`; the result `BuildInfo` |

## One module per page of the guide

| Module | Names |
|---|---|
| [`commol.engines`](engines.md) | `ODESolver`, `StochasticEventsSettings`, `StochasticStepsSettings`; the result `ModelEngine` |
| [`commol.plotting`](plotting.md) | `plot_compared`, `PlotConfig`; the methods `PRCCResult.plot`, `MorrisResult.plot`, `SobolResult.plot`, `ChaosResult.plot`, `ProfileResult.plot` |
| [`commol.parallelism`](parallelism.md) | `set_threads`, `threads`, `get_threads`, `get_levels` |
| [`commol.distributions`](distributions.md) | `Priors`, `Uniform`, `LogUniform`, `Beta`, `Normal`, `LogNormal`, `Gamma` |
| [`commol.quantities`](quantities.md) | `Level`, `Increments`, `ValueAt`, `Gain`, `Peak`, `PeakPosition`, `Trajectory`, `Ratio`, `quantity_schedule` |
| [`commol.observations`](observations.md) | `NormalObservations`, `LogNormalObservations`, `StudentTObservations`, `PoissonObservations`, `NegativeBinomialObservations`, `BinomialObservations`, `NoNoise`, `ResidualNoise`, `DistributionNoise` |
| [`commol.uncertainty`](uncertainty.md) | `ParameterSample`, `SampleMethod`, `Predictive`, `Scenarios`, `weighted_quantiles`, `member_statistics`; the results `ParameterSummary`, `PredictiveBand`, `SeriesBand`, `ObservationDraws`, `ScenarioSummary` and the figures of a predictive, `PredictivePlot` |
| [`commol.scores`](scores.md) | `coverage`, `interval_score`, `weighted_interval_score`, `crps`, `pit`, `log_score`; the result `Score` |
| [`commol.resampling`](resampling.md) | `BayesianBootstrap`, `ResidualBootstrap`, `WildBootstrap`, `BlockJackknife`, `ParametricBootstrap`; the results `BootstrapResult`, `JackknifeResult`, `ResamplingDiagnostics`, `BoundError` |
| [`commol.profiles`](profiles.md) | `ProfileLikelihood`, `PredictionProfile`, `PredictionSpec`, `Threshold`; the results `ProfileResult`, `Profile`, `ProfileInterval`, `ProfileStructure` |
| [`commol.local`](local.md) | `Laplace`, `Sandwich`, `LaplaceMixture`, `LikelihoodTarget`, `GibbsTarget`; the results `LaplaceResult`, `SandwichResult`, `ImportanceResult`, `MixtureComponent`, `DeltaBand` |
| [`commol.smc`](smc.md) | `GibbsPosterior`, `BayesianPosterior`, `InformationMatchingEta`, `BootstrapCoverageEta`, `ABCSMC`, `ABCRejection`, `LossDistance`, `SeriesDistance`; the results `SMCResult`, `GibbsResult`, `SMCDiagnostics` |
| [`commol.conformal`](conformal.md) | `SplitConformal`, `RollingConformal`, `JackknifePlus`, `EnbPI`, `AdaptiveConformal`; the results `ConformalResult`, `PredictionIntervals`, `SeriesIntervals` |
| [`commol.identifiability`](identifiability.md) | `StructuralRank`, `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability`, `ProfileIdentifiability`; the result `IdentifiabilityReport` and its parts `ParameterVerdicts`, `StructuralVerdict`, `PracticalVerdict`, `StructuralSection`, `PointRank`, `TighterRank`, `Accuracy`, `Combination`, `CollinearitySection`, `SubsetIndex`, `SampleSection`, `CorrelatedPair`, `StartsSection`, `ProfileSection`; the types of its report `IdentifiabilityReportData`, `VerdictsEntry`, `StructuralVerdictEntry`, `PracticalVerdictEntry`, `IdentifiabilitySections`, `StructuralSectionData`, `PointRankData`, `TighterRankData`, `AccuracyData`, `CombinationData`, `CollinearitySectionData`, `SubsetIndexData`, `SampleSectionData`, `CorrelatedPairData`, `StartsSectionData`, `ProfileSectionData` |
| [`commol.sensitivity`](sensitivity.md) | `SensitivityAnalysis`, `InputSpace`, `Propagation`, `PRCC`, `Morris`, `Sobol`, `PolynomialChaos`; the results `PropagationResult`, `QuantitySummary`, `PRCCResult`, `MorrisResult`, `SobolResult`, `ChaosResult` |

The pages also document names that no module exports, because no public
member returns them:

- `UnitConsistencyError`, which `Model.check_unit_consistency` raises, is in
  `commol._model.unit_checks`.
- The bases `ModelStructure` (of `Aggregate`), `LocalResult` (of
  `LaplaceResult` and `SandwichResult`) and `ConformalMethod` (of the
  conformal methods) are under the path of their module.
- `InitialConditions`, whose fields and methods `Aggregate` holds, is in
  [Model Structure](../advanced/model-structure.md).
