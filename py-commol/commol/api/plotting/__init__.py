"""Figures of results.

The `plot` accessors of `SimulationResults`, `SimulationReplicates` and
`Predictive` (`accessors`), and the `plot` method of a result with one figure,
are the public way to draw a figure. Each calls a private drawing function of
`figures`, which draws the panels that `panels` computes without matplotlib.
"""
