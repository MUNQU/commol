# Feature: network models

Goal: the first delivery of the structure `Agents` (`agent-based-models.md`
D6): units on a graph, with no attributes, built with the class
`Network(Agents)`, a variant (foundation D11). A process written with
`ModelBuilder` (foundation D10) runs on a graph: each node is in one bin and
moves between bins at the rates of the process, which may read its neighbors
through `neighbors_<bin>`. The user changes only the structure passed to
`build`; commol simulates the process, calibrates it, and describes it with
the same sections as any other kind.
Motivation: the user asked which kinds of modelling commol should add (L1).
A network sits between the aggregate kind (no units, interaction through
totals) and the agent kind (units with attributes and rules): it adds a graph
and no rule, so it is the first real second kind to test the generic layers.
`agent-based-models.md` D6 (L12) joined it into `Agents`; this ODD delivers
`Agents` with a `Graph` neighborhood and the variant `Network`, and the
agent ODD adds attributes, `Complete`, `Groups` and Python rules.
Branch: `feat/network-models` from `develop`, after
`odd/archive/tasks/model-kind-foundation.md`,
`odd/archive/tasks/stochastic-compartmental.md` and `odd/tasks/integral-outputs.md`
are merged.
Engram mirror: topic `odd/network-models/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ty check commol` and `vulture` (from `py-commol/`), `mkdocs build
--strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `901236d`)

- `ModelKind` has one real value, `Compartmental`
  (`commol-core/src/types/engine.rs:13`); `ModelStructure` (Rust,
  `types/structure.rs`) one real variant.
- The checklist "Model kind" (`docs/codebase/maintainer-playbook.md`) lists
  every place a kind touches: `ModelKind` and its `value_types`, a variant of
  `ModelStructure` in Rust, engines in `EngineType`, `EngineValuePlan`, a
  `ModelStructure` subclass in Python, a builder in `api/`, the tests of
  `tests/model_kinds/` copied for the kind.
- `ModelBuilder` writes the process of every kind, and the structure passed
  to `build` sets the kind (foundation D10).
- The foundation ODD gives `describe`, seeds, replicates and calibration with
  common random numbers; the stochastic ODD gives the event engines and their
  test oracles.
- No graph code exists in the repository.

## The model

| Part | From | Example (abstract) |
|---|---|---|
| Bins | The process: each node is in exactly one | `A`, `B`, `C` |
| Stratifications | The process: each node has one category of each, as a compartment does | `group = g1, g2` |
| Transitions, parameters, accumulators, integrals | The process, unchanged | `t_ab` at `k1 + k2 * neighbors_C` |
| Graph | The structure `Network(Agents)`: nodes and edges, directed or not, with an optional weight per edge, fixed or generated per replicate | 1 000 nodes, a list of edges |
| Initial assignment | The structure: fractions (`bin_fractions`, `stratification_fractions`, as in `Aggregate`) drawn per replicate, or the bin and categories of each node | 1 % of the nodes in `B` |

A rate is per node: a formula of the parameters, `t`, and `neighbors_C`, the
number of the node's neighbors in bin `C` or the sum of the weights of their
edges (D4), with `neighbors` the node's degree. A transition whose rate reads
`neighbors_C` is induced by `C`; one that reads no `neighbors_` variable is
spontaneous. The agent kind uses the same variable (`agent-based-models.md`).
A rate that names a bin, `total` or a `total_…` variable is an absolute flow
of the aggregate kind, with no meaning per node, and fails at `build` with
the per-unit form that carries over. Outputs are the number of nodes in each
compartment (bin and categories), and accumulators and integrals work as in
the aggregate kind.

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — The structure

- Fix (D5): Rust `AgentsStructure` (`commol-core/src/types/agents.rs`) and
  Python `Agents` (`context/agents.py`), a `ModelStructure` of kind `agents`
  (foundation D10), with the `Graph` neighborhood only in this ODD; and the
  variant `Network(Agents)` (foundation D11, `"variant": "network"`), whose
  constructor takes the graph first: `Network(g, bin_fractions=…,
  stratification_fractions=…)` with fractions drawn per replicate with its
  seed, or `Network(g, node_bins=…, node_categories=…)` per node; `total` is
  the number of nodes, and the variant rejects numeric attributes. The
  process comes from `ModelBuilder`, unchanged. `ModelKind::Agents` accepts
  the calibration value type `initial_condition` (a bin fraction, as in
  `Aggregate`).
- Fix: `check_model` validates, in Rust, the graph (node ids, edges to known
  nodes, no self-loops unless allowed), the assignment (every node in a bin
  and a category of each stratification it has), and that every rate of the
  process can run per node: no bin, `total` or `total_…` variable, and every
  `neighbors_<bin>` names a bin of the process.
- Check: a model saved and loaded back is equal and is a `Network`; one
  process builds with `Aggregate` and with `Network` and no change; `Network(g,
  …)` and `Agents(total=len(g), …, neighbors=Graph(g))` give equal runs; an edge to an unknown node, a
  node with no bin, a rate that reads `neighbors_X` for an unknown bin `X` and
  a rate that names a bin are rejected at `build` and named, the last with the
  per-unit form.

### S2 — The graph

- Fix (D1, D2): the graph is an edge list held in the model, so the model
  file reproduces runs: the node ids and the edges in columns (`source`,
  `target`, and an optional `weight`), not in pairs, which take 17 % more
  space (L2). `Network(g, …)` and `Graph(g)` take an edge list or a
  `networkx` graph.
  `networkx` is a required dependency, used for everything done once: input,
  generators, statistics and export; commol has no graph generators of its
  own, so a random graph comes from `networkx`
  (`Network(nx.barabasi_albert_graph(n, 10, seed=1), bin_fractions=…)`).
  The engine stores the graph as compressed sparse rows and never calls
  Python.
- Fix (D1): `Model.to_json` and `Model.from_json` (and
  `ModelLoader.from_json`) write and read gzip when the path ends in
  `.json.gz`, for every model kind, with the `gzip` module of the standard
  library; the content is the same JSON.
- Fix (D7): a graph per replicate. `Graph.generated(name, **parameters)`,
  passed as the graph (`Network(Graph.generated("barabasi_albert_graph",
  n=1000, m=3), bin_fractions=…)`), names a `networkx` generator that takes
  `seed` (50 of the 121 generators of
  `networkx` 3.7) and its parameters other than `seed`. The model stores the
  name, the parameters and the `networkx` version installed when it is built
  (`networkx.__version__`); saving it again does not change that version
  (D7). Before the engine runs,
  Python generates the graph of each replicate with a seed derived from the
  replicate's seed (the derivation of `commol-random`, exposed to Python), in
  a process pool, and passes the graphs to the engine; a single `run` uses
  the seed of the run. A calibration with common random numbers (foundation
  D8) generates the graphs once and reuses them in every evaluation.
  `Model.graph(seed)` returns the graph a run with that seed uses (S6).
  With a generator, the bins and categories of the nodes are drawn per
  replicate by fractions; a generated graph takes no per-node assignment.
- Fix: the guide states the cost of a graph per replicate: about 1 s to
  generate a graph of 10⁶ edges in `networkx` (measured, L3), divided by the
  workers of the pool, and about 8 MB per graph in the engine, so 100
  replicates hold about 800 MB.
- Check: a graph from `networkx` and its edge list give equal models; with a
  generator, one seed gives the same graphs and trajectories twice, two
  replicates get different graphs, and the mean degree of the graphs of a
  `gnp_random_graph(n, p)` generator equals `(n − 1) p` within its
  tolerance; a calibration with `r` replicates calls the generator `r` times
  in total; a generator without `seed` raises and says why; on a fixed graph
  with fractions, two replicates start from different assignments with the
  fractions of the model; a run with a `networkx` version other than
  the stored one warns once and names both, and a load does not warn; a model saved to
  `.json.gz` and loaded back is equal, and the file decompresses with `gzip`
  to the JSON that `.json` writes; a graph of 10⁶ edges on 10⁵ nodes saves to
  at most 3 MB as `.json.gz`.

### S3 — The engines

- Fix (D3): the engines `StochasticEvents` and `StochasticSteps`
  (`stochastic-compartmental.md`) run `Network` too: the engine names the
  method and the structure picks the implementation, in `commol-agents`, on
  the per-node path for units with no attributes
  (`agent-based-models.md` D3, L14). Under `Network`, `StochasticEvents` is
  continuous and exact: the direct method over node events, with per-node
  rates updated only for the neighbors of a node that changed, in the sum
  tree of rates of `commol-random` for `O(log n)` selection; `StochasticSteps`
  is discrete and synchronous: each node draws its change from the state at
  the start of the step; a rate is a per-step probability, as under
  `Aggregate`, and the transitions out of one bin are one draw between
  staying and each target; a probability outside
  `[0, 1]`, or a sum above 1, is the error of `stochastic-compartmental.md`
  D5, naming the node and the transition; D4).
- Fix (D4): rates are compiled with Cranelift as in the aggregate kind;
  the network engine resolves `neighbors_C` to a slot of its own that reads a
  vector of counts per node and bin, as the compartmental engine resolves
  compartments, `step` and `t` (`VarSlot`, `commol-model/src/types.rs`). The
  expression language, its parser, its compiler and its security check do not
  change. When a node changes state, the engine updates the counts and the
  rates of its neighbors, `O(degree)` per event. Both use `set_seed` and replicates
  from the foundation, and take one graph per replicate when the model has a
  generator (S2).
- Check: on a complete graph of `n` nodes with the rate
  `k * neighbors_C / (n - 1)`, `StochasticEvents` under `Network` is the same
  Markov chain as under `Aggregate` with `k * A * C / (total - 1)`, so their distributions
  agree (two-sample test); from one state, the mean change of one
  `StochasticSteps` step under `Network` with `1 - (1 - k) ^ neighbors_C` equals one
  `DifferenceEquations` step with `A * (1 - (1 - k) ^ C)` (with
  `k * A * C / total` it would differ by `O(k² m²)`); the forms of the guide (independent per
  neighbor, linear, intensity, threshold) give their closed form for one
  node with `m` neighbors in one step; on a star of `m` leaves in `C` around a center in `A`, the
  time to the induced move of the center at `k * neighbors_C` is exponential
  with mean `1/(k m)` (a closed form, unlike the pair approximation of
  Keeling 1999, which is only approximate); a node with no neighbors never
  moves by a transition whose rate is `k * neighbors_C`; a linear form that
  exceeds 1 at a hub raises and names the node.

### S4 — No builder of its own

- Fix (D5): no `NetworkModelBuilder`. `ModelBuilder` writes the process and
  `build(Network(…), engine="StochasticEvents")` runs it on a graph. The guide
  shows one process built with `Aggregate` and with `Network`, with the two
  lines that change.
- Check: the guide examples run; `build` with a `Network` returns
  `Model[Network]` (an overload on the structure type).

### S5 — Calibration, reports and plots

- Fix: no change to `Simulation`, `Calibrator`, the reports or the plotter:
  the kind implements `outputs_by_source`, `series_label`,
  `uncalibrated_values`, `with_calibrated_values` and `report_sections`
  (`initial_state`). Calibration uses replicates with common random numbers
  (foundation S7).
- Check: the tests of `tests/model_kinds/` copied for the agent kind pass with `Network`; a
  calibration of `k` in `k * neighbors_C` on synthetic data recovers it.

### S6 — Description

- Fix: `Network.describe` gives:
  - `entities`: the bins, the stratifications, and, for a fixed graph, its
    statistics computed with `networkx` (nodes, edges, directed, mean and
    maximum degree, degree distribution as a table, connected components,
    clustering coefficient); for a generator, its name, its parameters and
    the `networkx` version;
  - `dynamics`: a table of rules (from, to, induced by the bins whose
    `neighbors_` the rate reads, rate) and the state graph of a node
    (foundation S2), with induced edges drawn with their inducing state;
  - `initialization`: the fractions or the per-node assignment;
- Fix (D6): commol writes no graph file. `Model.graph(seed=None)` returns, as
  a `networkx` graph, the graph that a run with that seed uses: its edges and
  weights (fixed or generated), and the bin and categories of each node as
  assigned or drawn. `seed` is required when something is drawn (a generator
  or fractions) and may be left out otherwise. The user saves the graph with
  `networkx` (`nx.write_graphml`, `write_gexf`, `write_gml`, `write_pajek`,
  `write_edgelist`, node-link JSON).
- Check: the Markdown rendering lists every rule and the graph statistics;
  `Model.graph(seed)` has the edges, weights and node attributes of the run
  with that seed, and survives `nx.write_graphml` and `nx.read_graphml`;
  `graph()` with no seed on a model that draws raises and says why.

### S7 — Documentation

- Fix: a guide page "Network models" (`docs/guide/network-models.md`), its API
  pages, `mental-model.md` (a second kind), `repository-map.md`,
  `design-decisions.md` ("The graph is part of the model", "Neighbors are a
  variable of the rate"), the forms of an induced rate in the guide (a table
  of independent per neighbor, linear, intensity and threshold, each in
  `StochasticSteps` and `StochasticEvents`),
  `docs/guide/building-models.md` (saving to `.json.gz`),
  `advanced/model-structure.md`, the README section "What you can model",
  changelog.
- Check: `pytest -m docs`, `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1, S2 | Is the graph part of the model? | Yes: the edge list is in the model and its file, in columns; a generator writes its edges there when the model is built. A path that ends in `.json.gz` saves the model compressed with gzip, for every kind. | A graph passed at run time: a saved model would not reproduce a run, against "What the user saves, the engine reloads" (`mental-model.md`). Only the generator and its seed in the model: compact, but a graph from data has no generator, and the file would depend on the generator's code never changing. For the size (L2, 10⁶ edges on 10⁵ nodes): JSON in columns is 11.8 MB, with gzip 2.9 MB (random graph) or 1.1 MB (local graph); xz saves 20–40 % more but is slower and needs a C library or a less mature crate in Rust, and can follow as `.json.xz`; a binary encoding of commol inside the JSON saves 1.3–2.7 times more than gzip but no tool reads it; a separate graph file (npz, Parquet, GraphML) breaks one file per model. | approved (L2) |
| D2 | S2 | `networkx` | A required dependency, for everything done once: input, generators (commol has none of its own), statistics and GraphML. The engine receives compressed sparse rows and never calls Python. | No dependency, reading graphs through the interface of `networkx` (first recommended, then an optional extra): the user has `networkx` anyway, and commol could not use its algorithms. Generators and statistics in Rust: algorithms that `networkx` has tested, written again to save about 1 s per graph of 10⁶ edges (L3); generators of commol were set aside as too complex (L4). Cost: 13 MB installed for every user, no transitive dependency, 77 ms to import (measured, `networkx` 3.7). | approved (L5) |
| D3 | S3 | Which engines? | Exact continuous time and synchronous discrete time: `StochasticEvents` and `StochasticSteps`, the engines of `stochastic-compartmental.md`, implemented for `Network` (L14). | Continuous only: users of discrete models would convert every rate. A deterministic engine (pair or mean-field approximation): an approximation of the network, which `Aggregate` already gives at the mean-field level; it can follow. Engines named per kind (`NetworkEvents`, `NetworkSteps`, approved first, L8): moving a process from `Aggregate` to `Network` would change two lines, the structure and the engine (L14). | approved (L8, L14) |
| D4 | S1, S3 | How a rate reads the neighbors, and what a rate means in `StochasticSteps` under `Network` | A variable of the rate, `neighbors_<bin>` (the weighted number of neighbors in that state), as in the agent kind; no `induced_by`. Under `Network`, `StochasticSteps` reads a rate as a per-step probability, as in every discrete engine of commol. The user writes the form: `1 - (1 - p) * (1 - k) ^ neighbors_C` (independent per neighbor, Reed–Frost), `p + k * neighbors_C` (linear), `1 - exp(-(p + k * neighbors_C))` (intensity) or `if(neighbors_C >= theta, 1, 0)` (threshold). The engine resolves the variable to a slot; the compiler does not change. | A fixed independent form with `induced_by` (first recommended): one form chosen by the library, and no threshold model (L9). A setting that picks one of three forms: a menu of what commol foresaw, still without threshold models. A fixed linear form: at a hub of a Barabási–Albert graph (`k = 0.01`, 1000 neighbors) the probability is 10, so the run fails on the usual graphs. A hazard reading (`1 − e^(−rate)`) of every rate: `StochasticSteps` under `Network` would be the only discrete engine that reads rates as intensities. Cost: the user writes the form, from the table of the guide; a Reed–Frost form with a different weight per edge (`Π(1 − k w)`) cannot be written with a sum of weights, and would need a function of the engine over edges, which leaves the compiler unchanged. | approved (L10) |
| D5 | S1, S4 | How does a user run a process on a graph? | As a structure passed to `build`: `Network(g, bin_fractions=…)`, `Network(g, node_bins=…, node_categories=…)` or `Network(Graph.generated(name, **parameters), bin_fractions=…)`; no builder of its own. Fractions are drawn per replicate. | `NetworkModelBuilder` (approved first, L11): a second syntax for one process, against foundation D10 (L13). A `generator=` argument in `Network(…)`: `graph` and `generator` would exclude each other in one signature; a class constructor per case keeps each signature to what it means. Draw the assignment once at build: every replicate would start the same, and the variability of the start would be lost without the user choosing it. Cost: the guide shows which lines change between structures. | approved (L13) |
| D6 | S6 | How is the graph exported? | No export of commol: `Model.graph(seed=None)` returns the graph a run with that seed uses, as a `networkx` graph with the node attributes, and the user writes it with `networkx` in any of its formats; `seed` is required when the model draws. | `export_graph` to CSV and GraphML (first recommended, before D2): two formats chosen by commol, and code to write what `networkx` already writes (L12). Two methods, `graph()` and `generate_graph(seed)` (approved first, L12): with fractions drawn per replicate (D5), a fixed graph also needs a seed for its node attributes, so one method with the meaning "the graph of the run with this seed" covers both (L13). Only the model JSON: graph tools (Gephi, Cytoscape, igraph) cannot read it. Cost: one more line to save a file. | approved (L13) |
| D7 | S2 | A graph per replicate: what the model stores, and a `networkx` version that differs | The generator's name, its parameters and the `networkx` version installed when the model is built, which a later save keeps. When the graphs are generated (a run, replicates, a calibration) with another version installed, a warning names both versions, says that one seed may give another graph, and says to build the model again to adopt the new version. The reproducibility levels of the foundation (D6) list the version of `networkx`. | A warning at load (first recommended): the graphs are generated at run time, so a load with no run would warn for nothing, and an upgrade after the load would never warn (L7). An error: every model saved before an upgrade would stop running, even when its generator did not change. No check: a run that gives another graph after an upgrade would not say why. Update the stored version on save: the file would name a version the model was never defined with. The graphs of every replicate in the model: the number of replicates is not part of the model, and 100 graphs of 10⁶ edges take 110–290 MB with gzip (L2). A pin of `networkx` in commol: commol states ranges (`pyproject.toml`), and an exact pin would clash with the user's other packages (L6). | approved (L7) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | `ModelKind::Agents`, `Agents` with `Graph` and the variant `Network` in Rust and Python, validation of the process per node, save and load | direct | |
| T2 | S2 | Graph type, edge list and `networkx` input; gzip; generators per replicate (D7) | direct | |
| T3 | S3 | `commol-agents` crate: `StochasticEvents` and `StochasticSteps` under `Agents`, per-node path; registry entries, bindings | direct | |
| T4 | S4 | Overloads of `build` for `Network`; guide example of one process under two structures | direct | |
| T5 | S5 | The kind through simulation, calibration, reports and plots | direct | |
| T6 | S6 | Description; `Model.graph()` | direct | |
| T7 | S7 | Documentation | direct | |
| T8 | all | Final audit: every runner | direct | |

## Delivery

- Strategy: one PR to `develop` from `feat/network-models`, squash-merged
  (`feat(network): …`).
- Order: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8.
- Version impact: added (a model kind); not breaking.

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-14): asked which kinds of modelling commol should add,
  then for one ODD per kind on top of `model-kind-foundation.md`, each with
  how to implement the kind and how to describe it.
- L2 (user, 2026-10-14): approved D1 and asked for file formats that reduce
  its cost. Measured on 10⁶ edges and 10⁵ nodes: JSON pairs 13.8 MB, JSON
  columns 11.8 MB, with gzip 2.9 MB (Erdős–Rényi) and 1.1 MB (Watts–Strogatz),
  with xz 2.3 and 0.6 MB, a delta-coded compressed sparse row with xz 1.8 and
  0.4 MB (the bound of the random graph is about 13.7 bits per edge, 1.7 MB).
  Approved columns and `.json.gz`.
- L3 (user, 2026-10-14): asked why commol would not use `networkx` itself.
  Measured with `networkx` 3.7 on 10⁵ nodes and 10⁶ edges: Barabási–Albert
  0.94 s, Watts–Strogatz 0.73 s, Erdős–Rényi 1.07 s, components 0.04 s,
  clustering 6.7 s, edge list 0.12 s; 13 MB installed, no required
  dependency, 77 ms to import. D2 changed to a required dependency with no
  generators of commol.
- L4 (user, 2026-10-14): set aside generators of commol as too complex, and
  asked to bring a graph per replicate into scope, since it will be used:
  S2 and D7.
- L5 (user, 2026-10-14): approved D2 (`networkx` required, no generators of
  commol).
- L6 (user, 2026-10-14): asked whether a fixed version of commol fixes the
  version of `networkx`. It does not: commol states ranges
  (`numpy (>=2.5.1,<3.0)`), and only `uv.lock`, for development, pins them.
- L7 (user, 2026-10-14): asked how commol knows the version a model was saved
  with: the model stores `networkx.__version__` from the build. The check
  moved from load to graph generation. Approved D7.
- L8 (user, 2026-10-14): approved D3 (`NetworkEvents` and `NetworkSteps`).
- L9 (user, 2026-10-14): asked whether the independent form was the best,
  and whether the forms could be configured. The linear form, dismissed
  first for a wrong reason, is the form of the compartmental discrete
  engines; its real problem is hubs. A variable `neighbors_<bin>` lets the
  user write any form, threshold included.
- L10 (user, 2026-10-14): accepted the versatility only with a sound
  architecture. Checked: the JIT takes a buffer of variables in a fixed
  order, and each engine resolves names to slots (`VarSlot`), so the new
  variable is one slot of the network engine. Approved D4.
- L11 (user, 2026-10-14): approved D5 with the rename of `ModelBuilder` moved
  to the foundation (its D9), since `MarkovChainBuilder` is the first new
  builder and the foundation already breaks the API.
- L12 (user, 2026-10-14): approved D6 as revised after D2: commol returns a
  `networkx` graph and writes no graph file.
- (2026-10-14) Foundation D10 (`model-kind-foundation.md` L12): one process,
  structures apart, neutral terms (`neighbors_<bin>` for `contacts_<state>`).
  The decisions marked "to revisit" are reviewed under it.
- L13 (user, 2026-10-14): approved D5 under foundation D10: `Network` is a
  structure passed to `build`, with `Network.generated` (now `Graph.generated`) for a graph per
  replicate and fractions drawn per replicate; no builder. D6 follows: one
  `Model.graph(seed=None)` in place of `graph()` and `generate_graph(seed)`.
- L14 (user, 2026-10-15): approved, in `agent-based-models.md` D3, that an
  engine names its method and the structure picks the implementation:
  `NetworkEvents` and `NetworkSteps` become `StochasticEvents` and
  `StochasticSteps` under `Network`, so a process moves from `Aggregate` to
  `Network` by changing the structure only. D3 keeps its substance.
- L15 (user, 2026-10-15): with `agent-based-models.md` D6 and D8 and
  foundation D11, this ODD delivers `Agents` with a `Graph` neighborhood and
  the class `Network(Agents)`, a variant; `Network.generated` becomes
  `Graph.generated` passed as the graph.
