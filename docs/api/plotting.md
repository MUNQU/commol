# Plotting API

[Back to API Reference](index.md)

The figures are read from the results they draw, and return a matplotlib
`Figure`. `results.plot`, `replicates.plot` and `predictive.plot` give the
figures of a run, of replicates and of a predictive: `plot.series()`,
`plot.cumulative()` and `plot.observations()`. The results of PRCC, Morris,
Sobol and polynomial chaos, and the result of a profile likelihood, draw
their figure with `plot()`. `plot_compared` draws several runs, replicates or
predictives on the same panels. The figures need the optional extra
`commol[plot]`. Guide: [Plotting](../guide/simulations.md#plotting).

::: commol.api.plotting.accessors.ResultsPlot

::: commol.api.plotting.accessors.ReplicatesPlot

::: commol.api.plotting.accessors.PredictivePlot

::: commol.api.plotting.compared.plot_compared

::: commol.context.inference.global_sensitivity.PRCCResult.plot

::: commol.context.inference.global_sensitivity.MorrisResult.plot

::: commol.context.inference.global_sensitivity.SobolResult.plot

::: commol.context.inference.global_sensitivity.ChaosResult.plot

::: commol.context.inference.profiles.ProfileResult.plot

::: commol.context.visualization.PlotConfig
