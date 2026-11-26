# Descriptions

[Back to Design decisions](../design-decisions.md)

How a model is described: the blocks of a description and their renderings.
Reports have [a section of their own](results.md#reports).

## A description is data, rendered apart

`Model.describe(layout)` returns a `ModelDescription`
(`description/blocks.py`): sections with a key, a title and blocks from a
closed set of Pydantic models (`Text`, `Table`, `Equations`, `StateGraph`,
`Matrix`). The sections every kind shares come from `Model`
(`_model/describe.py`); a kind adds its own through
`ModelStructure.describe` (`Aggregate`: `_model/aggregate_description.py`).
The renderers (`description/render.py`) read only blocks, so a new kind renders
in every format with no new renderer. The form of the dynamics and the
schedule come from the engine's list (`EngineType::dynamics_form`), because
one structure runs as equations or as events depending on its engine.

| Part | Rule |
|---|---|
| Equations | Computed by the code of the planner's flows (`transition_flows`), so a description shows what the engine simulates. The text rendering keeps the compact and expanded forms line by line. |
| Diagrams | Mermaid in the Markdown rendering, only for a graph that is not `expanded`; Graphviz DOT for every graph in a `.dot` file. Text, LaTeX and JSON draw none. |
| State graphs | Built from the flows of the engine (`DynamicsDescription.state_graph`, `strata_graph`). The graph of the bins holds the moves between bins, the inflows and the outflows; the graph of the strata, only when a move changes categories, holds the categories that each move changes, joined when a move changes several stratifications. One edge per pair of nodes, one node `none` per side. A label is the formula of the one rate of one transition, and otherwise the parameters that the rates read, in declaration order. |
| Layouts | `"default"`: overview, entities, dynamics, schedule, initialization, parameters, outputs. `"odd"`: the ODD protocol; an element commol cannot fill is marked `TO_BE_WRITTEN`. A section with no blocks is left out of every rendering. |

| Alternative | Why not |
|---|---|
| A printer per kind and per format | Every kind writes every format, and no caller can read or export the description. |
| A free-form dict | The renderers would guess its shape; typed blocks are checked by Pydantic and `ty`. |
| An image (PNG, SVG) of a graph | Needs Graphviz or a browser at run time, a dependency for a view. |
| One graph of the bins and the categories | A move between categories crosses every bin: it is an edge from a bin to itself, which hides the categories, or one edge per bin and category. |
| A graph per stratification | A move can change several stratifications at once; one graph of the strata draws it as one edge. |
| The formulas, or the transition ids, as the label of a shared edge | The formulas are one per stratum, too long for an edge, and the graph of the compartments and the equations hold them; an id names a transition, not what drives it. |
| A node `none` per inflow or outflow | A model with an outflow from every bin draws as many `none` nodes as bins. |
| TikZ in the LaTeX rendering | Needs node coordinates, so a layout algorithm; a PDF made from the DOT file does the same. |

**Guards:** `tests/description/` (`test_equations*.py` check the lines of the
equations through `equation_forms`; `test_rendering.py` checks the snapshots of
`description/snapshots/`, parses the DOT with `pydot` and compiles the LaTeX
with `latexmk` when they are installed), `tests/building/test_engine.py`
(every engine gives a form).
