# Uncertainty API

[Back to API Reference](index.md)

What every uncertainty method takes and returns: the configuration bases, the
weighted parameter sample, the predictive and its observation noise, and the
scenarios. Then the methods of each family with their results (resampling,
profile likelihood, sequential Monte Carlo, local approximations and conformal
prediction), the scores and the one quantile.
Guide: [Uncertainty](../guide/uncertainty/index.md).

::: commol.context.inference.method.InferenceMethod

::: commol.context.inference.method.UncertaintyMethod

::: commol.context.inference.sample.ParameterSample

::: commol.context.inference.sample.SampleMethod

::: commol.context.inference.sample.ParameterSummary

::: commol.api.inference.predictive.Predictive

::: commol.api.inference.predictive.PredictiveBand

::: commol.api.inference.predictive.SeriesBand

::: commol.api.inference.predictive.ObservationDraws

::: commol.api.inference.scenarios.Scenarios

::: commol.api.inference.scenarios.ScenarioSummary

::: commol.context.inference.noise.NoNoise

::: commol.context.inference.noise.ResidualNoise

::: commol.context.inference.noise.DistributionNoise

## Resampling

Guide: [Resampling](../guide/uncertainty/resampling.md).

::: commol.context.inference.resampling.BayesianBootstrap

::: commol.context.inference.resampling.ResidualBootstrap

::: commol.context.inference.resampling.WildBootstrap

::: commol.context.inference.resampling.BlockJackknife

::: commol.context.inference.resampling.ParametricBootstrap

::: commol.context.inference.resampling.BootstrapResult

::: commol.context.inference.resampling.JackknifeResult

::: commol.context.inference.resampling.ResamplingDiagnostics

::: commol.context.inference.resampling.BoundError

## Profile likelihood

Guide: [Profile Likelihood](../guide/uncertainty/profiles.md).

::: commol.context.inference.profiles.ProfileLikelihood

::: commol.context.inference.profiles.PredictionProfile

::: commol.context.inference.profiles.PredictionSpec

::: commol.context.inference.profiles.Threshold

::: commol.context.inference.profiles.ProfileResult

::: commol.context.inference.profiles.Profile

::: commol.context.inference.profiles.ProfileInterval

::: commol.context.inference.profiles.ProfileStructure

## Sequential Monte Carlo

Guide: [Sequential Monte Carlo](../guide/uncertainty/smc.md).

::: commol.context.inference.distributions.Priors

::: commol.context.inference.distributions.Uniform

::: commol.context.inference.distributions.LogUniform

::: commol.context.inference.distributions.Beta

::: commol.context.inference.distributions.Normal

::: commol.context.inference.distributions.LogNormal

::: commol.context.inference.distributions.Gamma

::: commol.context.inference.smc.GibbsPosterior

::: commol.context.inference.smc.BayesianPosterior

::: commol.context.inference.smc.InformationMatchingEta

::: commol.context.inference.smc.BootstrapCoverageEta

::: commol.context.inference.smc.ABCSMC

::: commol.context.inference.smc.ABCRejection

::: commol.context.inference.smc.LossDistance

::: commol.context.inference.smc.SeriesDistance

::: commol.context.inference.smc.SMCResult

::: commol.context.inference.smc.GibbsResult

::: commol.context.inference.smc.SMCDiagnostics

## Local approximations

Guide: [Local Approximations](../guide/uncertainty/local.md).

::: commol.context.inference.local.Laplace

::: commol.context.inference.local.Sandwich

::: commol.context.inference.local.LaplaceMixture

::: commol.context.inference.local.LikelihoodTarget

::: commol.context.inference.local.GibbsTarget

::: commol.context.inference.local.LocalResult

::: commol.context.inference.local.LaplaceResult

::: commol.context.inference.local.SandwichResult

::: commol.context.inference.local.DeltaBand

::: commol.context.inference.local.ImportanceResult

::: commol.context.inference.local.MixtureComponent

## Conformal prediction

Guide: [Conformal Prediction Intervals](../guide/uncertainty/conformal.md).

::: commol.context.inference.conformal.ConformalMethod

::: commol.context.inference.conformal.SplitConformal

::: commol.context.inference.conformal.RollingConformal

::: commol.context.inference.conformal.JackknifePlus

::: commol.context.inference.conformal.EnbPI

::: commol.context.inference.conformal.AdaptiveConformal

::: commol.context.inference.conformal.ConformalResult

::: commol.context.inference.conformal.PredictionIntervals

::: commol.context.inference.conformal.SeriesIntervals

## Scores and intervals

::: commol.api.inference.scores

::: commol.api.inference.intervals.weighted_quantiles

::: commol.api.inference.intervals.member_statistics

::: commol.api.inference.intervals.ci_percentiles
