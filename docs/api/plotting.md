# Plotting API

[Back to API Reference](index.md)

The figures are read from the results they draw, and return a matplotlib
`Figure`: `results.plot`, `replicates.plot` and `predictive.plot` give the
figures of a run, of replicates and of a predictive, and the result of a
sensitivity method or of a profile likelihood draws its figure with `plot()`.
They need the optional extra `commol[plot]`. Guide:
[Plotting](../guide/simulations.md#plotting).

::: commol.api.plotting.accessors.ResultsPlot

::: commol.api.plotting.accessors.ReplicatesPlot

::: commol.api.plotting.accessors.PredictivePlot

::: commol.context.inference.global_sensitivity.PRCCResult.plot

::: commol.context.inference.global_sensitivity.MorrisResult.plot

::: commol.context.inference.global_sensitivity.SobolResult.plot

::: commol.context.inference.global_sensitivity.ChaosResult.plot

::: commol.context.inference.profiles.ProfileResult.plot

::: commol.context.visualization.PlotConfig
