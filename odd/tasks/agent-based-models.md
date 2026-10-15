# Feature: agent-based models

Goal: a new structure, `Agents`, under which a process written with
`ModelBuilder` (foundation D10) runs as a set of units, each in one bin, with
its own attribute values, that change by the rules of the process and read
their neighbors. The user writes the process once; to run it as agents they
pass `Agents(…)` to `build`. Commol simulates it in Rust, calibrates it, and
writes its description in the structure of the ODD protocol, the standard for
describing agent-based models.
Motivation: the user asked which kinds of modelling commol should add (L1).
Agents are the most expressive kind and the most costly; they come last, on
an interface tested by two kinds (aggregate and network) and on the
stochastic engines.
Branch: `feat/agent-based-models` from `develop`, after
`odd/tasks/network-models.md` is merged.
Engram mirror: topic `odd/agent-based-models/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ty check commol` and `vulture` (from `py-commol/`), `mkdocs build
--strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `901236d`)

- The generic layers use only `ModelStructure` and `SimulationEngine`
  (`docs/codebase/design-decisions.md`, "One interface for every model kind");
  the playbook names agents as an example of a new kind.
- Rate expressions are parsed, checked for security
  (`utils/security.py`) and compiled to machine code with Cranelift, with a
  global cache (`commol-core/src/math_expression/`). The compiled function
  reads a buffer of variables, and each engine resolves names to slots
  (`VarSlot`, `commol-model/src/types.rs`), so an attribute of a unit is one
  more slot.
- `CompartmentalStructure` requires at least one transition
  (`validate_transitions_not_empty`, `context/compartmental.py`).
- After the earlier ODDs: the process and the structure apart, `Aggregate`,
  `neighbors_<bin>` and `neighbors` (foundation D10); `describe`, seeds,
  replicates, calibration with common random numbers (foundation); event
  engines and their oracles (stochastic); `Agents` with a `Graph`
  neighborhood, the variant `Network`, graphs from `networkx` fixed or
  generated per replicate, and per-unit rates (`network-models.md`).

## The model

| Part | Where | Example (abstract) |
|---|---|---|
| Bins | Process (`add_bin`): each unit is in exactly one; one bin is enough | `A`, `B`, `C` |
| Categorical attributes | Process: the stratifications | `group ∈ {g1, g2}` |
| Moves between categories | Process: a transition between categories (`source=["g1"], target=["g2"]`) | `g1 → g2` at `k2` |
| Numeric attributes and their update | Process (`add_attribute`, D7) | `x`, updated to `x + 0.1` each step |
| Transitions | Process: a per-unit rate of the parameters, `t`, the unit's attributes and its neighbors | `k1 * (1 + x) * neighbors_C / neighbors` |
| Python rules | Process (`add_python_rule`, D1, D7) | `PythonRule("module:function")` |
| Units appearing and leaving | Process: a transition with no source (absolute: units created per unit of the axis) or no target (per unit: units removed) (D7) | `∅ → A` at `k3`; `C → ∅` at `k4` |
| Initial units | Structure `Agents`: `total`, `bin_fractions`, `stratification_fractions`, `attribute_values` (distributions), or a table with one row per unit | 10 000 units, `x ~ Uniform(0, 1)` |
| Neighborhood | Structure `Agents`: `neighbors=Complete()`, `Graph(…)`, `Graph.generated(…)` or `Groups("group", weights=…)` (D2), required | a graph of 10 000 nodes |
| Outputs | Counts per compartment; the mean of each numeric attribute per compartment; accumulators and integrals as in the aggregate kind (D4) | `A_g1`, `mean_x_A_g1` |

`neighbors_C` is the number (or the weighted number) of the unit's neighbors
in bin `C`, and `neighbors` the number of its neighbors, in every
neighborhood and in `Aggregate` (D2). A unit is not its own neighbor, so
under `Complete()` it has `N − 1` neighbors where `Aggregate` counts
`total`; one rule gives values that differ by `O(1/N)`, equal in the limit. One process runs under `Aggregate`, `Network` and `Agents`
as long as it uses no piece that only units have (numeric attributes, Python
rules); `build` names any such piece when the structure cannot run it.

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — The process pieces of units

- Fix (D7): `ModelBuilder.add_attribute(id, update=None, conditions=None)`
  declares a numeric attribute of every unit, or only of the units whose
  categories meet `conditions` (the mechanism of conditional stratifications);
  `update` is an expression applied at each step (`StochasticSteps`) or integrated
  as `dx/dt` (`StochasticEvents`, D3). `ModelBuilder.add_python_rule(rule)` adds a
  Python rule (D1). A process with at least one attribute update needs no
  transition: the check `validate_transitions_not_empty` becomes "at least one
  transition or one attribute update".
- Fix (D1): rule expressions use the expression language of rates with the
  unit's attributes, `neighbors_<bin>`, `neighbors` and the parameters as
  variables; `check_model` validates every variable in Rust and passes every
  expression through `validate_expression_security`. The engine resolves
  each name to a slot, as the network engine does (`network-models.md` D4),
  so the compiler is unchanged by the agent kind.
- Fix (D1): the language grows by what unit rules need, each as small as the
  architecture allows:
  - `&&` and `||` inside `if`, in the parser and compiler of `commol-core`,
    for every kind;
  - aggregates over neighbors as reserved variables that the engine fills:
    `neighbors_sum_<attribute>` and `neighbors_mean_<attribute>` (over all
    neighbors) and `neighbors_<bin>_mean_<attribute>` (over neighbors in a
    bin); no compiler change;
  - draws as reserved variables that the engine fills from `commol-random`
    at each evaluation: `draw_uniform` (in `[0, 1)`) and `draw_normal`
    (standard); allowed only where a rate is evaluated per unit (`Network`,
    `Agents`), so `check_model` rejects them elsewhere; no compiler change.
- Fix (D1): a Python rule, `PythonRule("module:function")`, holds logic that
  no expression holds. `StochasticSteps` calls it once per step and per rule, with
  the attributes of every unit as `numpy` arrays (views of the engine's
  structure of arrays, no copy), the neighbor counts, `t`, and a `numpy`
  generator seeded from the replicate's seed; it returns the new values or
  the probabilities of change for every unit. The model stores only the
  reference `"module:function"`, never its code, and loading imports the
  module. `StochasticEvents` rejects a Python rule, since it would call Python at
  each event. A model with a Python rule runs its replicates in a process
  pool instead of threads, since the GIL holds Python calls to one thread.
- Fix (D7): a transition with no target removes the unit; one with no source
  creates units at its rate, with categories by `stratification_fractions`
  and attributes drawn from `attribute_values` of the structure. Under
  `Network`, a transition with no source fails at `build` (a new node has no
  edges).
- Check: a model saved and loaded back is equal; a rule that reads an unknown
  attribute is rejected and named; an unsafe expression is rejected; an
  attribute with `conditions` exists only in the units of those categories,
  and a rule that reads it for other units is rejected; a process with only
  an attribute update and one bin builds and runs;
  `neighbors_mean_x` equals the mean of `x` over the neighbors written by hand
  on a small graph; `draw_uniform` under `Aggregate` is rejected; the mean of
  `draw_uniform` over many evaluations is `0.5` within its tolerance; a Python
  rule and the expression with the same logic give the same distribution of
  outputs; a model with a Python rule saves the reference and loads back
  when the module imports, and raises naming the module when it does not;
  `StochasticEvents` with a Python rule raises; a removal at `k4` per unit gives
  the mean of `StochasticEvents` with the same process; a creation at `k3`
  gives a Poisson number of new units with mean `k3 t`; a process with an
  attribute fails at `build` with `Aggregate` and with `Network`, naming it.

### S2 — The structure

- Fix (D7, D2): Rust `AgentsStructure` (`commol-core/src/types/agents.rs`) and
  Python `Agents` (`context/agents.py`), a `ModelStructure` of kind `agents`:
  the initial units (`total`, `bin_fractions`, `stratification_fractions`,
  `attribute_values` as distributions: `Uniform`, `Normal`, `LogNormal`,
  `Empirical`; or a table, a `pandas.DataFrame` or a CSV path with one row per
  unit) and the neighborhood, required (`Complete()`, `Graph(graph=…)`,
  `Graph.generated(name, **parameters)` as in `Network`, or
  `Groups(id, weights=None)` by the categories of a stratification, with a
  matrix of weights between categories, the identity when none is given;
  `O(categories)` per evaluation, counts per group updated in `O(1)` per
  event). Drawn values use the replicate's seed. A table is stored
  in the model, so the file reproduces runs (as the graph of `Network`).
  `ModelKind::Agents` accepts `initial_condition` (a bin fraction).
- Check: drawn units have the requested fractions and distributions within
  their tolerance; a table round-trips through save and load; one process
  builds with `Aggregate`, `Network` and `Agents` when it has no unit-only
  piece.

### S3 — The engines

- Fix (D3): `StochasticSteps` and `StochasticEvents` run `Agents`, as they run
  `Aggregate` and `Network`: an engine names its method and the structure
  picks the implementation, in `commol-agents`. Under `Agents`,
  `StochasticSteps` evaluates every unit's rules from the state at the start
  of the step, then applies the changes, so the order of units does not
  matter; `StochasticEvents` holds the total rate of each unit, draws the next
  event as under `Network`, and updates only the unit and its neighbors, with
  the sum tree of rates of `commol-random`. The units are
  stored as a structure of arrays (one array per attribute), and every rule
  is compiled once with Cranelift (D1). Replicates run in parallel
  (foundation S6), in threads unless a Python rule needs processes.
- Check (oracles):
  - units with no attributes under `Complete()` and the rate
    `k * neighbors_C / neighbors` give the distribution of `StochasticEvents`
    under `Aggregate` with `k * A * C / (total - 1)` (a unit is not its own
    neighbor);
  - under `Groups("group", weights=W)`, the mean of many `StochasticSteps` runs
    from one state equals one step of the aggregate model with the same
    `Groups`;
  - units on a graph, with no attributes, give the same distribution of
    outputs as `Network` on that graph (two-sample test);
  - a categorical attribute with its own rate gives the stratified aggregate
    model;
  - an update `x + c` gives `x(n) = x(0) + c n` for every unit.

### S4 — No builder of its own

- Fix: no `AgentModelBuilder`. `ModelBuilder` writes the process and
  `build(Agents(…), engine="StochasticSteps")` runs it as units. The guide shows one
  process built with `Aggregate`, `Network` and `Agents`.
- Check: the guide examples run; `build` with `Agents` returns
  `Model[Agents]` (an overload on the structure type).

### S5 — Simulation, calibration, reports and plots

- Fix: no change to the generic layers. `run` returns the outputs; a separate
  `Simulation.snapshot(at)` returns the table of units at a position, for
  inspection, never used by calibration (D4).
- Fix: calibration runs replicates with common random numbers and the
  `replicate_summary` the user chooses (foundation S7). The guide states the
  cost (units × replicates × evaluations) and recommends particle swarm with
  few particles and parallel replicates; likelihood-free inference (ABC) is
  the ODD that follows the foundation's S7.
- Check: the tests of `tests/model_kinds/` copied for the agent kind pass; a
  calibration of one rate on synthetic data recovers it within the tolerance
  of its replicates.

### S6 — Description as an ODD protocol document

- Fix (D5): the foundation gives `describe(layout="odd")`, the ODD protocol
  (Grimm et al. 2020) for every structure (foundation S3). `Agents.describe`
  fills its elements:

  | ODD element | From |
  |---|---|
  | 1. Purpose and patterns | `Model.description` (text the user gives) |
  | 2. Entities, state variables and scales | Bins, attributes, neighborhood, axis and step unit |
  | 3. Process overview and scheduling | The engine's `schedule` and the order of rules |
  | 4. Design concepts | Generated where commol knows them (interaction: the neighborhood; stochasticity: the engine and its draws; observation: the outputs); the others are marked "to be written by the modeller" |
  | 5. Initialization | The initial units of the structure |
  | 6. Input data | Series parameters and the table of units, if any |
  | 7. Submodels | Every rule with its expression, and the bin graph of a unit (foundation S2) |

- Check: `describe(layout="odd")` of an agent model holds the seven elements
  in order, with every rule and attribute, in every format; the "to be
  written" markers appear only in the elements commol cannot fill.

### S7 — Documentation

- Fix: a guide page "Agent-based models" (`docs/guide/agent-based-models.md`),
  its API pages, `mental-model.md`, `repository-map.md`,
  `design-decisions.md` ("Rules use the rate expression language", "Neighbor
  terms"), `advanced/model-structure.md`, the README section "What you can
  model", changelog.
- Check: `pytest -m docs`, `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1, S3 | How are rules written? | In commol's expression language, compiled with Cranelift, with the unit's attributes as variables resolved to slots by the engine; the language grows by `&&` and `\|\|`, aggregates over neighbors and draws (the last two as reserved variables, so the compiler does not change). A Python function, vectorized over all units, is a second way for logic that no expression holds, in `StochasticSteps` only. Where each piece is declared: D7. | Expressions only (first recommended): a rule that needs a loop or a structure has no way in (L2, L3). Python functions per unit (as in Mesa, Kazil et al. 2020): one Python call per unit and step, orders of magnitude slower, and no parallel replicates. Python functions only: every model loses speed, parallel replicates and the security check of expressions. A new rule language: a second parser and compiler. Rules in Rust as plugins: users of a Python library would compile Rust. Cost of the Python way: replicates in processes, not threads; the model stores a reference, so its file needs the module to load; not in `StochasticEvents`; the rule must draw from the generator it receives, or runs do not reproduce. | approved (L3) |
| D2 | S2 | Neighborhoods | Four, also in `Aggregate` (foundation D10): `Complete()`, `Graph(…)` and `Graph.generated(…)` (as in `Network`), and `Groups(id, weights=None)` (a matrix of weights between the categories of a stratification; the identity, within the category, when none is given). Required, with no default. `neighbors_C` is the weighted number of neighbors in `C` and `neighbors` their number; a unit is not its own neighbor, so `Agents` and `Aggregate` differ by `O(1/N)`. | A graph only: a complete or grouped neighborhood would need `O(n²)` edges (5·10⁹ for 10⁵ units). `Groups` within the category only (first proposed): a matrix of interaction between categories, usual in stratified models, could not move from `Aggregate` to `Agents` (L6). `Complete()` by default in `Aggregate`: the library would choose how units interact. `neighbors_C` as a fraction under a complete neighborhood (first written): the same rule would mean a different thing in each neighborhood. The neighborhood in the process: the graph of a network is data of the structure, and the line that changes between structures would grow. Continuous space (units with coordinates and a radius): a spatial index and its own engine; it can follow as a fifth neighborhood. Cost: the frequency-dependent form divides by `neighbors`; `Aggregate` gains a required argument when the process reads `neighbors_`. | approved (L6) |
| D3 | S3 | Which engines? | Synchronous discrete steps and exact continuous-time events, as `StochasticSteps` and `StochasticEvents`: one name per method for every structure, and the structure picks the implementation (`EngineType::kinds()`, foundation S0). | Engines named per kind (`AgentSteps`, `AgentEvents`, with `NetworkSteps` and `NetworkEvents`): moving a process between structures would change the engine too, though the method is the same (L7). Asynchronous random-order steps (common in NetLogo and Mesa): the result depends on the order, which needs one more seed and is harder to test against the aggregate oracles. Discrete only: the continuous limit is what the continuous engines share. Cost: the registry maps an engine to several kinds; `network-models.md` D3 changes its names. | approved (L7) |
| D4 | S5 | Outputs for calibration and plots | Aggregates only: the counts per compartment, accumulators and integrals, as under `Aggregate` and `Network`, plus the mean of each numeric attribute per compartment (`mean_x_A_g1`); the table of units only through `Simulation.snapshot(at)`, for inspection. | Unit tables as outputs: a run of 10⁶ units × 10³ positions does not fit in memory, the losses compare aggregates, and reports and plots read series. Counts only: a numeric attribute could not be observed or calibrated. More statistics per attribute (variance, quantiles): useful, and they can follow without a break; an integral already builds other aggregates. Cost: one unit is followed through several snapshots. | approved (L8) |
| D5 | S6 | How is an agent model described? | With the ODD protocol, as a layout of the foundation's description for every structure: `describe(layout="odd")`, rendered in any format; `Agents` fills its elements. The name is `"odd"`, the standard name of the protocol, and the guide explains it. | An `"odd"` renderer for agents only (first written): the same process under `Aggregate` or `Network` would not have the protocol, so two modellings of one process would read in two layouts (foundation D10); and as a value of `render(format)` it mixed a layout with a format, so the protocol could not come out in LaTeX (L9). `"odd_protocol"` as the name: the user kept the standard name (L10). Only the foundation's sections: readers of agent models expect the protocol, and journals ask for it. A separate document type: a second description, which the generic renderers and `save` would not know. | approved (L10) |
| D6 | — | `Network` as a special case of `Agents`? | One structure for units, `Agents`: a network is `Agents(…, neighbors=Graph(g))` with no attributes, and the engine takes its simple per-node path when units have no attributes. The decisions of `network-models.md` (graph in the model, `networkx`, `neighbors_C`, `Model.graph(seed)`, the `networkx` version) apply to `Agents` with `Graph`; that ODD becomes the first delivery of `Agents`. A shortcut named per PEP 8 builds it from a graph (D8). | Two structures (first recommended): two names for one model, and a network that gains an attribute changes structure, against foundation D10 (L11). A subclass `Network(Agents)`: the registry holds one class per kind, and a model saved as `Network` would load as `Agents`, unequal for pydantic. A `network` kind of its own: the two structures again. A CapWords factory `Network(…)`: against PEP 8 (L12). Cost: users who think of nodes write `Agents` or the shortcut; the guide page "Network models" starts from it. | approved (L12) |
| D7 | S1, S2 | Where does each piece of a unit model go, and what may units do? | What describes behavior goes in the process: numeric attributes with their update (`add_attribute`, with `conditions` for attributes of some categories only), Python rules (`add_python_rule`), and transitions with no source or target, which create or remove units. What sets the start goes in `Agents`: the initial units and their attribute values, and the neighborhood. A process with an attribute update needs no transition. | Attributes and rules inside `Agents(…)`: the transitions of the process would read `x` declared elsewhere, and the update of `x` is behavior, not a start value, so the mix that foundation D10 undid would come back. A builder of its own: a second syntax for one process. Without the three additions (L5): a model of attributes only could not be written (one transition is required today), every unit would carry every attribute, and no unit could appear or leave, which Mesa allows. Cost: a process with a unit-only piece builds only with `Agents`, and `build` names the piece. | approved (L5) |
| D8 | S2 | How does a user build units on a graph in short? | The class `Network(Agents)`, a variant (foundation D11): `Network(g, bin_fractions=…)`, with the total from the number of nodes and no numeric attributes; it loads back as `Network`. | A CapWords function `Network(…)` returning `Agents`: against PEP 8 (L12). A classmethod `Agents.network(…)`: `Network` would not be a concept of the API (L13). A function `network(…)`: beside the classes `Aggregate` and `Agents`, it does not say what it returns. Cost: the registry of structures gains variants (foundation D11). | approved (L13) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | Process pieces: `add_attribute` (with `conditions`), `add_python_rule`; a process with no transition when an attribute updates; validation, save and load | direct | |
| T2 | S2 | `Agents` grows: initial units (tables and distributions), `Complete` and `Groups` (the `Graph` neighborhood and `ModelKind::Agents` come from `network-models.md`) | direct | |
| T3 | S3 | `commol-agents` crate: `StochasticSteps`, `StochasticEvents`, structure of arrays, compiled rules, units created and removed; registry entries, bindings | direct | |
| T3a | S1 | Language: `&&` and `\|\|`; aggregate and draw variables | direct | |
| T3b | S1, S3 | Python rules in `StochasticSteps`; replicates in a process pool | direct | |
| T4 | S4 | Overloads of `build` for `Agents`; guide example of one process under three structures | direct | |
| T5 | S5 | Snapshots; the kind through simulation, calibration, reports and plots; oracles | direct | |
| T6 | S6 | ODD protocol description and renderer | direct | |
| T7 | S7 | Documentation | direct | |
| T8 | all | Final audit: every runner, and a run of 10⁶ units to record its time and memory | direct | |

## Delivery

- Strategy: one PR to `develop` from `feat/agent-based-models`, squash-merged
  (`feat(agents): …`). If the PR grows past review size, T1–T3 and T4–T7 go in
  two PRs, the first with the engine and no guide page.
- Order: T1 → T2 → T3 → T3a → T3b → T4 → T5 → T6 → T7 → T8.
- Version impact: added (a structure); not breaking.

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-14): asked which kinds of modelling commol should add,
  then for one ODD per kind on top of `model-kind-foundation.md`, each with
  how to implement the kind and how to describe it.
- L2 (user, 2026-10-14): asked what Mesa is, and what it would take to have
  its flexibility. Three ways: vectorized Python rules beside expressions, a
  larger expression language, or rules in Rust (not recommended).
- L3 (user, 2026-10-14): approved D1 with both the larger expression language
  and vectorized Python rules in `StochasticSteps`.
- (2026-10-14) Foundation D10 (`model-kind-foundation.md` L12): one process,
  structures apart, neutral terms (`neighbors_<bin>` for `contacts_<state>`).
  The decisions marked "to revisit" are reviewed under it.
- L4 (user, 2026-10-14): under foundation D10, asked whether placing the
  pieces of units in the process and the structure loses flexibility. It
  does not against this ODD; against Mesa three gaps showed, present in the
  first design too: models of attributes only, attributes of some types of
  unit only, and units that appear and leave.
- L5 (user, 2026-10-14): approved D7: behavior in the process, start values
  and neighborhood in `Agents`, with the three additions.
- L6 (user, 2026-10-15): asked whether D2 was the best approach. It was, but
  `Groups` needed a matrix of weights between categories, and "the same in
  every structure" holds up to `O(1/N)`, since a unit is not its own
  neighbor. Approved D2 with both, and with `neighbors` in `Aggregate` with no
  default.
- L7 (user, 2026-10-15): approved D3 with engines named by method,
  `StochasticSteps` and `StochasticEvents`, for every structure; the
  structure picks the implementation, so moving a process between structures
  changes one line.
- L8 (user, 2026-10-15): approved D4 (aggregates only, with the mean of each
  numeric attribute per compartment; tables through `snapshot`).
- L9 (user, 2026-10-15): asked why the layout is named "odd". After the ODD
  protocol (Overview, Design concepts, Details); it clashes with the task
  documents of `odd/`, and as a format of `render` it mixed a layout with a
  format.
- L10 (user, 2026-10-15): kept the name `"odd"`, as the standard, to be
  explained in the documentation. D5 approved as a layout of `describe` for
  every structure, apart from the format.
- L11 (user, 2026-10-15): asked whether `Network` could stay in the API while
  the same construct runs inside. Yes, as a shortcut that returns `Agents`;
  not as a subclass (one class per kind in the registry, and a saved model
  would load as another class) nor as a kind of its own.
- L12 (user, 2026-10-15): approved D6 (one structure, `Agents`, with a
  shortcut from a graph); Python names must always follow PEP 8, so the
  shortcut cannot be a CapWords function (D8).
- L13 (user, 2026-10-15): chose classes for the shortcut: `Network(Agents)`,
  a variant of `Agents` (foundation D11), as `MarkovChain` is of `Aggregate`.
