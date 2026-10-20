# Uncertainty API

[Back to API Reference](index.md)

What every uncertainty method takes and returns: the configuration base, the
weighted parameter sample, the predictive and its observation noise, the
resampling and conformal methods, the scores and the one quantile.
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

::: commol.context.inference.smc.EtaCalibration

::: commol.context.inference.smc.InformationMatching

::: commol.context.inference.smc.BootstrapCoverage

::: commol.context.inference.smc.AbcSmc

::: commol.context.inference.smc.AbcRejection

::: commol.context.inference.smc.LossDistance

::: commol.context.inference.smc.SeriesDistance

::: commol.context.inference.smc.SmcResult

::: commol.context.inference.smc.GibbsResult

::: commol.context.inference.smc.SmcDiagnostics

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

::: commol.api.inference.registry
