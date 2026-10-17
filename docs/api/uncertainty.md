# Uncertainty API

[Back to API Reference](index.md)

What every uncertainty method takes and returns: the configuration base, the
weighted parameter sample, the predictive and its observation noise, the
resampling methods, the scores and the one quantile.
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

## Resampling

Guide: [Resampling](../guide/uncertainty/resampling.md).

::: commol.context.inference.resampling.BayesianBootstrap

::: commol.context.inference.resampling.ResidualBootstrap

::: commol.context.inference.resampling.WildBootstrap

::: commol.context.inference.resampling.BlockJackknife

::: commol.context.inference.resampling.BootstrapResult

::: commol.context.inference.resampling.JackknifeResult

::: commol.context.inference.resampling.ResamplingDiagnostics

::: commol.context.inference.resampling.BoundError

## Scores and intervals

::: commol.api.inference.scores

::: commol.api.inference.intervals.weighted_quantiles

::: commol.api.inference.intervals.member_statistics

::: commol.api.inference.intervals.ci_percentiles

::: commol.api.inference.registry
