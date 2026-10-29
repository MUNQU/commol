# Describing a Model

[Back to Building Models](building-models.md)

`Model.describe()` returns a description of a model as data: sections of
tables, equations, state graphs and text. You can read it in Python, render it
as text, Markdown, LaTeX or JSON, save it to a file, or draw its state graphs.
The description works for every kind of model, and it is built from what the
engine simulates, so the equations it shows are the equations that run. Use
this page to document a model, to check the equations Commol derived, or to
write the methods section of a paper.

## Quick path

```python
from commol import Aggregate, ModelBuilder

model = (
    ModelBuilder(name="Described", bin_unit="item", step_unit="day")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_bin(id="C", name="State C")
    .add_parameter(id="k1", value=0.5, unit="1/day")
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
    .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
    .build(
        Aggregate(bin_amounts={"A": 900, "B": 100, "C": 0}),
        engine="DifferenceEquations",
    )
)

description = model.describe()
print(description.render("text"))   # every section, in plain text
description.save("model.md")        # Markdown, with the state graphs as Mermaid diagrams
description.save("model.tex")       # a LaTeX document
description.save("model.dot")       # every state graph, in Graphviz DOT
```

In a Jupyter notebook, `model.describe()` as the last line of a cell shows the
Markdown rendering, with the state graphs drawn (JupyterLab draws Mermaid from
version 4.1).

## What a description holds

A `ModelDescription` has a `title`, a `layout` and a list of `sections`. Each
section has a `key`, a `title` and `blocks`. `Model.describe()` builds the
sections every kind shares from the model, and the structure of the model
adds the sections of its kind (`ModelStructure.describe`):

| Key | Content | For an `Aggregate` model |
|---|---|---|
| `overview` | Name, description, version, kind, engine, its settings and the step unit | From `Model` |
| `entities` | What holds the state | Bins, accumulators, stratifications and their categories, the expanded compartments |
| `dynamics` | How the state changes | The state graphs, the compact and expanded forms of the equations, the impulses, the neighbors |
| `schedule` | How the engine advances, and the settings it advances with | From the engine (`EngineType.schedule`) and `Model.engine.settings` |
| `initialization` | The state the model starts from | Bin amounts, category fractions, overrides, and the initial state the engine builds |
| `parameters` | Every parameter, then each matrix between categories | Value, unit, whether calibration sets it, description; for a matrix, its stratification, unit, description, spectral radius and values |
| `outputs` | What a run returns | The outputs of each bin and accumulator |

A section with no blocks is left out of every rendering.

```python
dynamics = description.section("dynamics")
print([type(block).__name__ for block in dynamics.blocks])
# ['StateGraph', 'StateGraph', 'Equations', 'Text', 'Equations']
```

The blocks are a closed set (`commol._model.description`):

| Block | Holds |
|---|---|
| `Text` | A paragraph |
| `Table` | Columns and rows of cells |
| `Equations` | Groups of equations, each in plain text and in LaTeX |
| `StateGraph` | Nodes and edges; `expanded` marks the graph of the expanded compartments |
| `Matrix` | Labelled rows and columns of numbers, such as the weights of neighbors |

## Formats

`render(format)` returns a string; `save(path, format=None)` writes it and,
without `format`, takes the format from the suffix of the path.

| Format | Suffix | Equations | Diagrams |
|---|---|---|---|
| `"text"` | `.txt` | Compact and expanded forms in plain text | None |
| `"markdown"` | `.md` | LaTeX math | The graph of the bins and the graph of the strata as Mermaid diagrams |
| `"latex"` | `.tex` | `align*` environments in a full document | None |
| `"json"` | `.json` | As data | The graphs as data; `ModelDescription.model_validate_json` reads it back |
| `"dot"` | `.dot` | None | Every state graph, the expanded one included |

The Markdown rendering does not draw the graph of the expanded compartments:
a stratified model can have hundreds of edges, more than a Mermaid diagram
draws. It gives the size of the graph instead. The DOT file holds every
graph; `dot -Tpdf model.dot -o model.pdf` draws them with
[Graphviz](https://graphviz.org/).

## The state graphs

`description.graphs()` gives the state graphs of the `dynamics` section, in
order:

| Graph | `name` | Nodes | Edges |
|---|---|---|---|
| The bins | `states` | Every bin | The moves between bins, the inflows and the outflows |
| The strata | `strata` | The categories that a move changes | The moves between categories, in any bin |
| The compartments | `compartments` | Every expanded compartment (`expanded`) | One edge per flow, labelled with its formula |

The graph of the strata is there only when the model moves amounts between
categories: with a transition between categories (`source=["g1"]`,
`target=["g2"]`) or with a stratified rate whose condition has `to`. A move
inside one bin is an edge of the graph of the strata, not an edge from the bin
to itself. A node of the graph of the strata holds the categories of the
stratifications that the move changes: `group:g1`, or `group:g1, size:s1`
for a move that changes two stratifications at once.

The graphs of the bins and of the strata hold one edge per pair of nodes,
whatever the number of transitions between them, and every inflow starts at
one node `none` and every outflow ends at another. The label of an edge says
what drives the move:

| The edge comes from | Its label |
|---|---|
| One transition with one rate | The formula of the rate, such as `k1 * A` |
| One transition with stratified rates, or several transitions | The parameters that the rates read, in the order the model declares them, such as `k1, k2, m`; compartments, totals, `t` and `step` are left out |
| A transition with no rate | `no rate` |

In the graph of the strata, the label first names the bins that move, such as
`A, B, C: k1, k2`. The formula of each flow is in the graph of the
compartments and in the [equations](#the-equations).

```python
from commol import Aggregate, ModelBuilder

moving = (
    ModelBuilder(name="Moving", bin_unit="item", step_unit="day")
    .add_bin(id="A", name="State A")
    .add_bin(id="B", name="State B")
    .add_stratification(id="group", categories=["g1", "g2"])
    .add_parameter(id="k1", value=0.5, unit="1/day")
    .add_parameter(id="k2", value=0.1, unit="1/day")
    .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1")
    .add_transition(
        id="t_move",
        source=["A"],
        target=["A"],
        stratified_rates=[
            {
                "conditions": [
                    {"stratification": "group", "category": "g1", "to": "g2"}
                ],
                "rate": "k2",
            }
        ],
    )
    .build(
        Aggregate(
            bin_amounts={"A": 900, "B": 100},
            stratification_fractions={"group": {"g1": 0.5, "g2": 0.5}},
        ),
        engine="DifferenceEquations",
    )
)

bins, strata, compartments = moving.describe().graphs()
print([(edge.source, edge.target, edge.label) for edge in bins.edges])
# [('A', 'B', 'k1')]
print([(edge.source, edge.target, edge.label) for edge in strata.edges])
# [('group:g1', 'group:g2', 'A: k2')]
```

The LaTeX document uses only standard packages (`amsmath`, `amssymb`,
`longtable`, `fontenc`, `geometry`), so `latexmk -pdf model.tex` builds it.

## The equations

The `dynamics` section of an aggregate model holds the flows that the engine
plans, after it has decided which rates are proportional (see
[Transitions](transitions.md#absolute-and-proportional-rates)):

| Part | Content |
|---|---|
| Compact form | Each transition and its flows, `source -> target: rate`. With units, each symbol shows its unit and each flow its resulting unit. |
| Expanded form | One equation per compartment. The left-hand side comes from the engine: `[A(t+Dt) - A(t)] / Dt` for `DifferenceEquations`, `dA/dt` for `DifferentialEquations`. |
| Events | For a stochastic engine, instead of the expanded form: one row per flow, with the change of the state of one event and its propensity (`StochasticEvents`) or its draw in a step (`StochasticSteps`). See [Stochastic Simulation](stochastic.md#how-a-rate-becomes-an-event). |
| Impulses | Each impulse transition, the positions where it acts and the amounts it moves; an impulse is not part of the equations. |

Each rate and each flow is written from the tree that the parser of the
engine reads, not from the text of the rate, so `k1*A*B/total` and
`k1 * A * B / total` give the same description. The text has one space around
each operator, only the parentheses that its grouping needs, `^` for a power,
and each number as its value: `1e-5` is `1e-05` and `2.0` is `2`. The flow
of a proportional rate is the whole rate times its source: `(k1 + k2) * B`.
In LaTeX, a product is `\cdot`, a division outside parentheses is a fraction
and a power raises its exponent.

For the model above, the text rendering holds:

```text
Bin Transitions:
t_ab (A -> B):
  A -> B: k1(1/day) * A(item) * B(item) / total(item) [item / day]

t_bc (B -> C):
  B -> C: k2(1/day) * B(item) [item / day]

Total System: 3 coupled equations (3 bins)
```

```text
[A(t+Dt) - A(t)] / Dt = - (k1 * A * B / total)
[B(t+Dt) - B(t)] / Dt = (k1 * A * B / total) - (k2 * B)
[C(t+Dt) - C(t)] / Dt = (k2 * B)
```

Units appear only when every bin and every constant parameter has one; with
some units missing, the equations show none. To check the units, use
[`check_unit_consistency`](units-and-equations.md#unit-checking).

## The ODD layout

`model.describe(layout="odd")` follows the **ODD protocol** (Overview, Design
concepts, Details), the standard description of individual-based and
agent-based models, used for other simulation models too (Grimm et al. 2006,
updated in 2010 and 2020). The same layout works for every kind of model, so a
paper can describe an aggregate model and an agent model in one format.

```python
odd = model.describe(layout="odd")
print([section.title for section in odd.sections])
# ['Purpose and patterns', 'Entities, state variables and scales',
#  'Process overview and scheduling', 'Design concepts', 'Initialization',
#  'Input data', 'Submodels']
odd.save("model_odd.tex")
```

| Element | What Commol fills for an `Aggregate` model |
|---|---|
| Purpose and patterns | `Model.description`; without one, the marker |
| Entities, state variables and scales | Bins, accumulators, stratifications, compartments; the step unit and how the engine advances |
| Process overview and scheduling | Each transition, where it moves, whether it is an impulse; the schedule of the engine and its settings |
| Design concepts | Interaction (when the structure has [neighbors](building-models.md#neighbors)), stochasticity (from the engine) and observation (the outputs); the marker for the others |
| Initialization | The bin amounts and their units, the category fractions, the overrides and the initial state |
| Input data | The series parameters, or "None" |
| Submodels | The state graphs and equations, the parameters and the outputs |

An element Commol cannot fill says "To be written by the modeller.": write it
in the saved file. The layout of a description changes which sections it has
and their order; the format of `render` is independent of it.

References:

- Grimm, V., et al. (2006). A standard protocol for describing
  individual-based and agent-based models. *Ecological Modelling*, 198,
  115–126. [doi:10.1016/j.ecolmodel.2006.04.023](https://doi.org/10.1016/j.ecolmodel.2006.04.023)
- Grimm, V., et al. (2010). The ODD protocol: A review and first update.
  *Ecological Modelling*, 221, 2760–2768.
  [doi:10.1016/j.ecolmodel.2010.08.019](https://doi.org/10.1016/j.ecolmodel.2010.08.019)
- Grimm, V., et al. (2020). The ODD protocol for describing agent-based and
  other simulation models: A second update to improve clarity, replication,
  and structural realism. *Journal of Artificial Societies and Social
  Simulation*, 23(2), 7. [doi:10.18564/jasss.4259](https://doi.org/10.18564/jasss.4259)

## Checklist

- [ ] `model.describe().render("text")` shows the equations you expect.
- [ ] Saved files use a suffix of the table above, or pass `format=`.
- [ ] For a paper, `layout="odd"` and the markers it leaves are filled in.

## Next steps

| To learn | Read |
|---|---|
| How rates become flows | [Transitions](transitions.md) |
| Unit checking | [Units](units-and-equations.md) |
| Running the model | [Running Simulations](simulations.md) |
