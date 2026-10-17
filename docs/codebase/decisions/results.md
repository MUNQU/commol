# Results

[Back to Design decisions](../design-decisions.md)

How intervals, descriptions, reports and model files are made from a model and its runs.

## One confidence-interval convention

Every interval is a weighted quantile: the inverse of the weighted empirical
distribution, interpolated between the midpoints of the cumulative weights
(`statistics::quantiles` in `crates/commol-inference/src/statistics.rs`). With
equal weights it equals NumPy's `method="hazen"`; a weight of zero removes a
value. `ci_percentiles` (`api/inference/intervals.py`) turns a confidence level
into the two percentile points. `member_statistics` reduces each member to the
quantity of interest **first**, and then takes weighted quantiles across
members; taking quantiles first gives a band that describes no member. The
replicates of a stochastic run (`SimulationReplicates.quantiles`), a sample
and a predictive use the same quantile.

| Alternative | Why not |
|---|---|
| NumPy's default (`linear`) with equal weights and another rule with weights | Two conventions: a sample would change its interval when it gains weights. |
| Harrell–Davis | Smoother, but with no simple weighted form, and slower. |

**Rule:** every new interval goes through `weighted_quantiles` or
`member_statistics`.

**Guards:** `crates/commol-inference/src/statistics.rs` tests,
`tests/uncertainty/test_intervals.py`.

## A description is data, rendered apart

`Model.describe(layout)` returns a `ModelDescription`
(`context/description.py`): sections with a key, a title and blocks from a
closed set of Pydantic models (`Text`, `Table`, `Equations`, `StateGraph`,
`Matrix`). The sections every kind shares come from `Model`
(`context/_model/describe.py`); a kind adds its own through
`ModelStructure.describe` (`Aggregate`: `context/_model/aggregate_description.py`).
The renderers (`api/description.py`) read only blocks, so a new kind renders
in every format with no new renderer. The form of the dynamics and the
schedule come from the engine's list (`EngineType::dynamics_form`), because
one structure runs as equations or as events depending on its engine.

| Part | Rule |
|---|---|
| Equations | Computed by the code of the planner's flows (`transition_flows`), so a description shows what the engine simulates. The text rendering keeps the compact and expanded forms line by line. |
| Diagrams | Mermaid in the Markdown rendering, only for a graph that is not `expanded`; Graphviz DOT for every graph in a `.dot` file. Text, LaTeX and JSON draw none. |
| Layouts | `"default"`: overview, entities, dynamics, schedule, initialization, parameters, outputs. `"odd"`: the ODD protocol; an element commol cannot fill is marked `TO_BE_WRITTEN`. A section with no blocks is left out of every rendering. |

| Alternative | Why not |
|---|---|
| A printer per kind and per format (the former `print_equations`) | Every kind writes every format, and no caller can read or export the description. |
| A free-form dict | The renderers would guess its shape; typed blocks are checked by Pydantic and `ty`. |
| An image (PNG, SVG) of a graph | Needs Graphviz or a browser at run time, a dependency for a view. |
| TikZ in the LaTeX rendering | Needs node coordinates, so a layout algorithm; a PDF made from the DOT file does the same. |

**Guards:** `tests/description/` (`test_equations*.py` check the lines of the
equations through `equation_forms`; `test_rendering.py` checks the snapshots of
`description/snapshots/`, parses the DOT with `pydot` and compiles the LaTeX
with `latexmk` when they are installed), `tests/building/test_engine.py`
(every engine gives a form).

## Reports

`api/reporting.py` returns blocks of data, not documents:
`simulation_report` (observed values, series, accumulators) and
`applied_parameters_report` (calibrated parameters, and the sections of the
model kind: `initial_conditions` for an aggregate model).

- A single run and an ensemble (replicates or a `Predictive`) differ only at
  the leaves (`ReportLeaf`); a predictive's members carry their weights.
- The key order of each entry is part of the contract, because it changes saved
  JSON files.
- Labels and summary figures belong to the caller, because they carry domain
  vocabulary.

**Guard:** `tests/reporting/`.

## Model files are written in full

`Model.to_json` writes every field, nulls included. Dropping nulls is not
round-trip safe: an uncalibrated `Parameter.value` or `BinFraction.fraction`
would lose a required key and fail to reload. `render_json`
(`context/_model/serialization.py`) keeps files readable by writing long numeric
arrays on one line. It is internal on purpose: an output format in the public
API becomes a compatibility promise.

## Public models forbid unknown fields

Every public Pydantic class sets `model_config = ConfigDict(extra="forbid")`.
An unknown field, such as a misspelt or an old name (`window_steps`), raises a
`ValidationError` that names it, instead of being dropped with no error.

**Guard:** `tests/building/test_unknown_fields.py`.
