# Plotting API

[Back to API Reference](index.md)

The module `commol.plotting`: `plot_compared` and the style of a figure,
`PlotConfig`. The figures are read from the results they draw, and return a matplotlib
`Figure`. `results.plot`, `replicates.plot` and `predictive.plot` give the
figures of a run, of replicates and of a predictive: `plot.series()`,
`plot.cumulative()` and `plot.observations()`; each accessor is on the page
of the object that gives it: [Simulation](simulation.md) and
[Uncertainty](uncertainty.md). The results of PRCC, Morris,
Sobol and polynomial chaos, and the result of a profile likelihood, draw
their figure with `plot()`. `plot_compared` draws several runs, replicates or
predictives on the same panels. The figures need the optional extra
`commol[plot]`. Guide: [Plotting](../guide/simulations.md#plotting).

::: commol.plotting.plot_compared

::: commol.sensitivity.results.PRCCResult.plot

::: commol.sensitivity.results.MorrisResult.plot

::: commol.sensitivity.results.SobolResult.plot

::: commol.sensitivity.results.ChaosResult.plot

::: commol.profiles.results.ProfileResult.plot

::: commol.plotting.PlotConfig
