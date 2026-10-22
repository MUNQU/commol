"""Figures of results.

The `plot` accessors of `SimulationResults`, `SimulationReplicates` and
`Predictive` (`accessors`), the `plot` method of a result with one figure,
and `plot_compared` (`compared`), the figure of several results on the same
panels, are the public way to draw a figure. Each calls a private drawing
function of `figures`, which draws the panels that `panels` computes without
matplotlib.
"""
