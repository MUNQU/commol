# Refactor: describe, randomness and replicates for any model kind

Goal: the code that every model kind shares can describe a model in a
readable, exportable form, run a stochastic engine, and calibrate it, so that
the ODDs of new kinds and engines (`odd/stochastic-compartmental.md`,
`odd/markov-chains.md`, `odd/network-models.md`,
`odd/agent-based-models.md`) only add their own parts.
Motivation: the user noted that `print_equations` only makes sense for a model
with equations; a network or an agent model has rules, not equations (L1).
Reading the code for this ODD found two more gaps that every new kind hits:
the engine contract has no randomness, and the calibration scores one run per
evaluation.
Branch: `refactor/model-kind-foundation` from `develop`, after the 0.3.0
release (`odd/README.md`).
Engram mirror: topic `odd/model-kind-foundation/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ty check commol` and `vulture` (from `py-commol/`), `mkdocs build
--strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `901236d`)

### Describing a model

| What | Where | Kind |
|---|---|---|
| `Model.print_equations(output_file, format)` | `context/model.py:271`, delegates to `ModelStructure.print_equations` | Any; the default raises `NotImplementedError("A {kind} model has no equations to print")` (`context/structure.py:248`) |
| `EquationPrinter`: header, compact form, expanded form, impulses | `context/_model/equations.py` | Compartmental only; the header reads `structure.bins`, `stratifications`, `transitions` |
| Left-hand side of an equation | `EngineType::equation_lhs` (`commol-core/src/types/engine.rs:191`) | Per engine: `[X(t+Dt) - X(t)] / Dt` or `dX/dt` |
| Formats | `PrintEquationsOutputFormat` (`constants.py`): `text`, `latex` | — |
| Model file | `Model.to_json` | Any; exact, but read by a machine |

The description is printed, not returned: no caller can read it as data,
export it as Markdown or JSON, or draw it. The only other readable view of a
model is its JSON file.

### Randomness

- `SimulationEngine` (`commol-core/src/simulation_engine.rs`) has no seed and
  no random source. `reset()` returns to the initial state, so two runs give
  the same result: an invariant of today's engines ("Results are
  reproducible", `docs/codebase/mental-model.md`).
- The only seeds are the optimizer's (`OptimizationConfig::seed`,
  `commol-calibration/src/optimization.rs:276`) and the ensemble's
  (`CalibrationProblem.seed`).
- `Simulation.run(num_steps)` and `run_at(at)` return one `SimulationResults`
  (`api/results.py`).

### Calibration

`CalibrationProblem::cost` (`commol-calibration/src/calibration_problem.rs:812`)
clones the base engine, resets it, writes the values, runs it once at the
observed positions and computes the loss. With a stochastic engine the loss
would change from one evaluation to the next for the same parameters, and
Nelder-Mead and particle swarm would chase noise.

### One description per kind

- `CompartmentalStructure` (`commol-core/src/types/compartmental.rs`, Python
  `context/compartmental.py`) holds two different things: the process
  (`bins`, `stratifications`, `transitions`, `accumulators`) and how much
  starts where (`initial_conditions`: `total` and fractions).
- `ModelBuilder` (`api/model_builder.py`) writes both, and `build(engine)`
  returns a compartmental model. A network or agent kind as first planned
  would need its own builder, and the user would write the same transitions
  again in other words.
- A rate that names a compartment, a bin total, `total` or a `total_…`
  variable is absolute (`plan_transition_flows`, `commol-model/src/flows.rs`);
  every other rate is proportional, a rate per unit of its source.

### What is already generic

- `ModelStructure` (`context/structure.py`): `check_model`,
  `outputs_by_source`, `series_label`, `calibration_value_types`,
  `uncalibrated_values`, `with_calibrated_values`, `report_sections`,
  `check_unit_consistency`.
- `EngineType` names the kind each engine runs, its settings and what a rate
  means (`RateMeaning::PerStep`, `Continuous`). A stochastic engine fits these
  two meanings: a per-step probability (chain binomial) or a hazard
  (continuous time).
- Outputs: `output_names`, `state_outputs` and `output_sums` describe any
  vector of outputs, such as the count of agents in each state.
- The test kind (`tests/support/kinds.py`, `py-commol/src/test_kind.rs`) runs
  through every generic layer.

## Specs

Every spec follows the [documentation definition of done](../docs/codebase/conventions.md#documentation-definition-of-done).

### S0 — One process, structures apart

- Fix (D10): a model is a header (name, description, version, parameters),
  a **process** and a **structure**, run by an engine. The process is
  written once and holds the bins, the stratifications, the transitions and
  the accumulators (and the integrals of `integral-outputs.md`). The structure
  is what changes between kinds of model: `Aggregate` (amounts per
  compartment: `total`, `bin_fractions`, `stratification_fractions`,
  `compartment_fractions`, today's initial conditions); `Network` and
  `Agents` come with their ODDs. Rust: `Model { …, process: Process,
  structure: ModelStructure }`, with `ModelStructure` tagged by kind as today
  and `ModelKind::Aggregate` in place of `Compartmental`; each engine names
  the kind of structure it runs, as today. Python: `Process` (one pydantic
  model) and the subclasses of `ModelStructure`, registered by kind; the
  interface of a kind (`check_model`, `outputs_by_source`, `describe`, …)
  stays on the structure and reads the process.
- Fix (D10): `ModelBuilder` writes the process; `build(structure, engine)`
  joins it with a structure: `builder.build(Aggregate(total=1000,
  bin_fractions={"A": 0.99, "B": 0.01, "C": 0.0}), engine="DifferentialEquations")`.
  `set_initial_conditions` leaves the builder for `Aggregate`. `build` always
  returns `Model`; its static type follows the type of the structure
  argument (overloads), never a value.
- Fix (D10): rates may read `neighbors_<bin>` and `neighbors`, the terms that
  carry a process from one structure to another. `Aggregate` takes
  `neighbors=Complete()` (every unit a neighbor of every other:
  `neighbors_<bin>` is the bin total and `neighbors` is `total`) or
  `neighbors=Groups(id, weights=None)` (by the categories of one
  stratification: in a compartment of category `g`, `neighbors_<bin>` is
  `Σ_g' w[g][g'] × (amount of the bin in g')`, with the identity for no
  weights), as `Agents` does (`agent-based-models.md` D2). `neighbors` is
  required, with no default, when the process reads a `neighbors_` variable,
  and `build` lists the options when it is missing. A rate that reads them
  and no compartment is proportional, so `k1 * neighbors_B / neighbors` from
  `A` gives the flow `k1 * A * B / total` under `Complete()`. `neighbors_`
  names are reserved in `check_names`.
- Fix (D11): a structure may have **variants**: a subclass that keeps the
  kind, the fields and the engines of its structure, narrows the constructor
  and may add checks (`MarkovChain(Aggregate)`, `Network(Agents)`). The JSON
  holds the kind and the variant (`"kind": "aggregate", "variant":
  "markov_chain"`); the Python registry is keyed by `(kind, variant)`, so a
  model loads back as the class it was saved with; Rust reads the variant
  only to run the checks of the variant, and the engines do not change.
- Fix (D10): an engine names a method, not a kind: `EngineType::kind()`
  becomes `EngineType::kinds()`, the kinds of structure it runs, and
  `check_kind` checks membership; `Engine::build` (`py-commol/src/engine.rs`)
  picks the implementation by `(engine, kind)`. `DifferenceEquations` and
  `DifferentialEquations` run `Aggregate`; the stochastic engines run every
  structure that implements them (`agent-based-models.md` D3). A pair with
  no implementation fails at `build` and lists the engines of that
  structure.
- Fix: Python names follow PEP 8 everywhere (L16): ruff selects the `N`
  rules (pep8-naming) in `py-commol/pyproject.toml`; the five pydantic
  validators that `N805` finds in `context/calibration.py` take
  `@classmethod`, as pydantic documents.
- Fix: a process feature that a structure cannot run fails at `build` and
  names it, with the form that would carry over (an absolute rate outside
  `Aggregate`: "write it per unit of its source").
- Check: one process builds with `Aggregate` for each engine of the aggregate
  kind and runs; `k1 * neighbors_B / neighbors` under `Complete()` and
  `k1 * A * B / total` give the same trajectories; under `Groups("group")`
  with a weight matrix, the flow of each compartment equals the one written
  by hand with `total_<cat>`; a process that reads `neighbors_B` with no
  `neighbors` raises and lists `Complete()` and `Groups(…)`; a model saved and loaded back is equal, and its
  JSON has `process` and `structure`; a parameter named `neighbors_A` is
  rejected; a test variant of `Aggregate` saves its variant, loads back as
  its own class, is an instance of `Aggregate`, and runs every engine of
  `Aggregate`.

### S1 — A model description as data

- Fix: `ModelStructure.describe(model)` returns a `ModelDescription` (D1): an
  ordered list of sections, each with a key, a title and typed blocks. The
  generic part (`Model.describe()`) builds the sections every kind shares
  from `Model` (name, description, engine and its settings, parameters with
  value, unit and whether calibration sets them) and adds the sections of the
  kind. The sections follow the "Details" part of the ODD protocol for
  describing models (Grimm et al. 2006, 2020), so the same keys fit a
  compartmental, a network and an agent model (D2):

  | Key | Content | Compartmental today |
  |---|---|---|
  | `overview` | Name, description, kind, engine, axis and step unit | From `Model` |
  | `entities` | What holds the state, and its variables | Bins, accumulators, stratifications and their categories, the expanded compartments |
  | `dynamics` | How the state changes | Equations (compact and expanded), impulses |
  | `schedule` | How the engine advances | From the engine (D4) |
  | `initialization` | The starting state | Total, fractions, overrides, the initial state the engine builds |
  | `parameters` | Every parameter | From `Model` |
  | `outputs` | What a run returns | `outputs_by_source` |

  A section with no blocks is left out of every rendering.

- Fix: the blocks are a closed set of Pydantic models: `Text`, `Table`
  (columns and rows), `Equations` (left and right sides, as text and LaTeX),
  `StateGraph` (S2) and `Matrix` (row and column labels, values). A kind
  builds its sections from them; a renderer never knows the kind.
- Check: the compartmental description holds every bin, transition, flow,
  parameter and initial amount of a stratified model; the test kind returns a
  description with its own `dynamics` section; `Model.describe()` of a kind
  that adds nothing still has `overview`, `parameters` and `outputs`.

### S2 — A state graph for any kind

- Fix: `StateGraph` holds nodes (id, label, group) and edges (source, target,
  label). Every kind that has discrete states gives one: the compartmental
  kind gives one node per bin and one edge per transition (label: its rate),
  and a second, expanded graph with one node per compartment; a network or an
  agent kind gives the states of a node or an agent. Renderers write a graph
  as Mermaid and as Graphviz DOT (D3):

  | Graph | Markdown rendering and Jupyter (Mermaid) | `.dot` file |
  |---|---|---|
  | States of the model (bins and transitions; states of a node or an agent) | Embedded as a ` ```mermaid ` block | Yes |
  | Expanded compartments | Not embedded: the rendering gives their number and points to the `.dot` file | Yes |

  The text, LaTeX and JSON renderings embed no diagram; the JSON holds the
  graphs as data.
- Check: the Mermaid and DOT output of a model with several sources, an
  outflow (`target=[]`), an inflow (`source=[]`) and a category transition
  parse with their reference tools in a test (the Mermaid grammar is checked
  by a snapshot; DOT with `pydot` if it is installed, else a snapshot).

### S3 — Renderers and export

- Fix: `Model.describe()` returns the `ModelDescription`;
  `ModelDescription.render(format)` returns a string for `"text"`,
  `"markdown"`, `"latex"` or `"json"`; `ModelDescription.save(path, format)`
  writes it, and the format comes from the suffix when not given (`.md`,
  `.tex`, `.json`, `.txt`, and `.dot` for the graphs of S2). The renderers
  live in `api/description.py`; the blocks in `context/description.py`.
- Fix (`agent-based-models.md` D5): `Model.describe(layout=…)` chooses which
  sections and in what order, apart from the format of `render`. The default
  layout is the seven sections of S1; `layout="odd"` follows the ODD protocol
  (Overview, Design concepts, Details; Grimm et al. 2006, updated 2010 and
  2020), the standard description of agent-based and individual-based
  models, for every structure: purpose (`Model.description`), entities and
  scales, process overview and scheduling (the engine's `schedule`), design
  concepts, initialization, input data (series parameters, tables) and
  submodels (transitions and the bin graph). Each structure fills what it
  knows (`Aggregate`: compartments and fractions; `Network`: the graph and
  its statistics; `Agents`: attributes, neighborhood and design concepts),
  and an element commol cannot fill is marked "to be written by the
  modeller". `model.describe(layout="odd").render("latex")` gives the
  protocol in LaTeX.
- Fix: `ModelDescription._repr_markdown_` returns the Markdown rendering, so
  `model.describe()` in a notebook shows the description with its state
  graph drawn (JupyterLab draws Mermaid from version 4.1).
- Fix (D5): `Model.print_equations`, `ModelStructure.print_equations`,
  `EquationPrinter` and `PrintEquationsOutputFormat` go; the compartmental
  equations become the `dynamics` section, computed by the same code
  (`transition_flows` from the engine). The text rendering of that section
  keeps today's compact and expanded forms, so the tests of
  `tests/units/test_print_equations*.py` move to `tests/description/` and
  check the same lines.
- Check: the Markdown rendering embeds the state graph of the model and not
  the expanded one; `save("model.dot")` writes both graphs;
  `_repr_markdown_` equals `render("markdown")`; `describe(layout="odd")`
  holds the elements of the protocol in its order for an `Aggregate` model,
  with "to be written" markers only where commol cannot fill them, and renders
  in every format.
- Check: the text rendering of a compartmental model contains today's
  compact and expanded forms line by line; the LaTeX rendering compiles in a
  test that runs `latexmk` when it is installed, and is snapshot-checked
  otherwise; the JSON rendering loads back into an equal `ModelDescription`.

### S4 — The engine says how it advances

- Fix: `EngineType::equation_lhs` becomes `EngineType::dynamics_form`, an
  enum the description reads: `DifferenceEquation`, `DifferentialEquation`,
  and, for the engines of the next ODDs, `StochasticEvents` (an event table:
  each event, its change of state and its propensity) and `Rules`. Each form
  gives the text of the `schedule` section ("each step applies every flow
  once", "the solver integrates…", "events happen one at a time at random
  times…"), and the left-hand side when the form has equations.
- Check: `tests/building/test_engine.py` checks every entry of the list
  gives a form; the compartmental equations keep their left-hand sides.

### S5 — Randomness in the engine contract

- Fix: `EngineType::is_stochastic()`; `SimulationEngine::set_seed(seed: u64)`
  with a default that does nothing, so the deterministic engines do not
  change. A stochastic engine draws every random number from one generator
  seeded there, and `reset()` restores the seed, so a run is reproducible
  from its seed. The generator is `Xoshiro256PlusPlus` of `rand_xoshiro`,
  whose sequence is the same on every platform and version, and the
  samplers (binomial, Poisson, exponential, multinomial) are written in
  commol, not taken from `rand_distr`, whose values may change in a minor
  version (D6).
- Fix: a crate `commol-random` holds what every stochastic engine of every
  kind shares and knows no model: the generator, the derivation of replicate
  seeds (S6), the samplers, and a sum tree of rates that picks the next event
  in `O(log n)`. The stochastic test engine uses it, and so do the engines of
  `stochastic-compartmental.md`, `network-models.md` and
  `agent-based-models.md` (`stochastic-compartmental` D2, L3).
- Fix: the documentation states the three levels of reproducibility: the
  same machine and commol version give the same run; another platform with
  the same commol version gives the same random numbers, but the rate
  functions (`exp`, `sin`, `log`, …) come from the system's libm through the
  JIT (`math_expression/jit/libm.rs`), which can differ in the last bit
  between operating systems, so a stochastic run there is equal in
  distribution and rarely differs draw by draw; another commol version may
  change a run, and the changelog says so.
- Fix: the invariant "Results are reproducible" in `mental-model.md` becomes
  "a run is reproducible from its model and its seed, on one platform and one
  commol version".
- Check: a test engine with randomness (added to the test kind) gives equal
  runs for one seed and different runs for two seeds; the deterministic
  engines ignore the seed; the first numbers of each sampler for a fixed seed
  equal a stored reference, so a change of algorithm fails a test.

### S6 — Replicates in the simulation API

- Fix (D7): each method returns one type.
  - `Simulation.run(num_steps, *, seed=None)` and `run_at(at, *, seed=None)`
    return `SimulationResults`, as today; for a stochastic engine it is one
    realization. Without a seed, commol draws one and keeps it in
    `results.seed`, so every run can be repeated.
  - `Simulation.replicates(n, *, seed=None)` returns an object with the same
    `run` and `run_at`, which return `SimulationReplicates`: a sequence of
    `SimulationResults` with one seed each, plus `mean()`,
    `quantiles(levels)` and `member_statistics` per output, reduced member by member as `ci_percentiles` requires
  (`docs/codebase/design-decisions.md`, "One confidence-interval
  convention"). Replicate seeds come from `seed` through one documented
  derivation (SplitMix64 of `seed + index`), and replicates run in parallel in
  Rust (`rayon`, already a dependency of `commol-calibration`).
- Fix: `replicates(n)` on a deterministic engine raises `ValueError` naming
  the engine: every replicate would be equal.
- Fix: `SimulationPlotter` draws the mean and a band for
  `SimulationReplicates`, as it does for an ensemble today; reports accept
  them as `ensemble_runs` (`simulation_report`), whose layout already holds
  a spread per number.
- Check: the mean of many replicates of the test engine converges to its
  expected value; plots and reports of replicates run.

### S7 — Calibration of a stochastic engine

- Fix (D8): `CalibrationProblem(replicates=…, replicate_summary=…)`. With a
  stochastic engine, each evaluation runs `replicates` runs with common random
  numbers: the same replicate seeds at every evaluation, derived from
  `CalibrationProblem.seed`, so the loss is a deterministic function of the
  parameters and the optimizers keep working. The user chooses what the
  observations are compared with; there is no default:

  | `replicate_summary` | Compared with each observation | Loss |
  |---|---|---|
  | `"mean"` | The mean of the replicates | `loss_function` (`sse`, `rmse`, `mae`) |
  | `"median"` | The median | `loss_function` |
  | `Quantile(q)` | The quantile `q` | `loss_function` |
  | `"crps"` | The empirical distribution of the replicates | The continuous ranked probability score, a proper scoring rule; with one replicate it is the absolute error |

- Fix: a stochastic calibration without `replicate_summary` raises and lists
  the options; without a seed it raises, as it would not be reproducible.
  With a deterministic engine both are unused. The summaries and the CRPS are
  computed in `LossConfig` (`commol-calibration/src/types.rs`), so the
  optimizer, the ensemble central loss and the reports measure the same way
  ("One meaning of an observation weight", `design-decisions.md`); weights
  apply to every option.
- Fix: the probabilistic calibration runs the same objective; its ensemble
  then holds parameter sets, and `SimulationReplicates` give the stochastic
  spread of each member.
- Check: on synthetic data from a stochastic test model, the calibration
  recovers the parameter with each summary within the tolerance its
  replicates allow; two runs with one seed give the same result; the CRPS of
  one replicate equals the absolute error, and the CRPS of a known ensemble
  equals its closed form `E|X − y| − ½ E|X − X'|`.
- Out of scope: likelihood-based inference (particle filters, ABC). ABC is
  in `odd/smc-inference.md`, particle filters in
  `odd/state-space-inference.md` (L18).

### S8 — Documentation and README

- Fix: a guide page "Describing a model" (`docs/guide/describing-models.md`)
  replaces the "Printing equations" part of `units-and-equations.md`, and
  explains the `"odd"` layout: what the ODD protocol is, its elements, its
  references (Grimm et al. 2006, 2010, 2020), and which elements commol fills
  for each structure; `docs/codebase/` notes that the `"odd"` layout is the
  ODD protocol, unrelated to the task documents of `odd/`; the
  README example shows `model.describe()` instead of `print_equations`;
  `design-decisions.md` gets "A description is data, rendered apart" and
  "Stochastic engines and seeds"; `maintainer-playbook.md` adds `describe`
  and `set_seed` to the checklists of a model kind and of an engine;
  `mental-model.md` updates the reproducibility invariant; the changelog maps
  `print_equations` to `describe`.
- Check: `pytest -m docs`, `mkdocs build --strict`.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | What does a kind give: text or data? | Data: a `ModelDescription` of typed blocks; renderers turn it into text, Markdown, LaTeX or JSON. | Each kind prints its own text (today): every kind writes a printer per format, and no caller can read or export the description. A free-form dict: renderers would guess its shape; typed blocks are checked by Pydantic and `ty`. | approved (L2) |
| D2 | S1 | Which sections? | The seven keys of S1, after the "Details" part of the ODD protocol (entities and state variables, process overview and scheduling, initialization, submodels). | Free sections per kind: a reader of two kinds finds two layouts, and the agent ODD would have to map them to the protocol anyway. The full ODD protocol (purpose, design concepts, …): those parts are the modeller's text, not data commol has; `overview` takes the model's `description` instead. | approved (L3) |
| D3 | S2, S3 | Diagram formats | Mermaid and Graphviz DOT, as text. Mermaid is embedded in the Markdown rendering (and so in Jupyter, through `_repr_markdown_`) for the state graph of the model only; DOT is a separate file with every graph, the expanded one included. | An image (PNG, SVG): needs Graphviz or a browser at run time, a dependency for a view. Mermaid only: most Markdown viewers draw it (GitHub, GitLab, MkDocs Material, JupyterLab ≥ 4.1, Obsidian), but journals and LaTeX use Graphviz, and Mermaid stops drawing near 500 edges by default, which an expanded stratified model passes. DOT only: almost no Markdown viewer draws it. TikZ in the LaTeX rendering: needs node coordinates, so a layout algorithm or LuaLaTeX's `graphdrawing`; a PDF made from the DOT file with `\includegraphics` does the same. | approved (L4) |
| D4 | S4 | Where does the form of the dynamics come from? | The engine's list (`EngineType::dynamics_form`), as `equation_lhs` today. | The structure: a compartmental structure runs as equations or as events depending on its engine, so the structure cannot know. Python: breaks the golden rule (the engine is the source of truth). | approved (L5) |
| D5 | S3 | What happens to `print_equations`? | Remove it in the same release; `describe()` gives the equations as a section, and the text rendering keeps their form. Commol is alpha. | Keep it as an alias of `describe().section("dynamics")`: two ways to do one thing, and the name says "equations" for kinds that have none. Keep it only on compartmental models: a method that some models have and others do not, on one `Model` class. | approved (L6) |
| D6 | S5 | Where does randomness live? | In the engine, seeded by `set_seed`; one `Xoshiro256PlusPlus` per engine, and samplers written in commol. | A random source passed to every call: the trait grows an argument that the deterministic engines ignore. Seeds in the model file: a model is a description; the seed belongs to a run, as `CalibrationProblem.seed` does. A cryptographic generator (`StdRng`): slower, and simulation needs no security. `SmallRng` (first recommended): its documentation calls it non-portable, as its algorithm may change between versions of `rand` and with the platform. `rand_distr` samplers: their values may change in a minor version. A pure-Rust libm in the JIT, for bit-identical runs across operating systems: costs speed in every engine for a difference in the last bit; documented instead (L7). | approved (L7) |
| D7 | S6 | How does the API return replicates? | `run` and `run_at` (with `seed`) always return `SimulationResults`; `simulation.replicates(n, seed=…)` returns an object whose `run` and `run_at` always return `SimulationReplicates`. | `run(…, replicates=n)` returning one type or the other (first recommended): the return type depends on a value, so every caller narrows a union (L8). Always a list: breaks every caller of a deterministic model for no gain. Methods `run_replicates` and `run_at_replicates`: four run methods with almost the same arguments. A class `ReplicatedSimulation(model, …)`: repeats the engine build and the model sync of `Simulation`. A generator of runs: statistics need all of them, and Rust runs them in parallel. | approved (L8) |
| D8 | S7 | How does a deterministic optimizer calibrate a stochastic engine? | `replicates` runs with common random numbers (fixed seeds), and a summary the user chooses, with no default: mean, median, a quantile, or the CRPS of the whole distribution. | The mean, fixed by the library (first recommended): what to fit is a modelling choice, and the mean of runs where half go extinct describes no run (L9). One run per evaluation: the loss is noise and Nelder-Mead stops anywhere. New random numbers at each evaluation: the same noise, smaller. A likelihood (particle filter): the right tool for small counts, but a new inference method; it gets its own ODD (S7). | approved (L9) |
| D9 | — | Does `ModelBuilder` take the name of a kind? | No. Under D10 it writes the process of every kind, and the structure passed to `build` sets the kind, so its general name is the right one. | `CompartmentalModelBuilder` (approved first, L11): under D10 it would say the builder only serves aggregate models, the opposite of what it does (L13). `ProcessBuilder`: names what it writes, but `build` returns a `Model` and the user thinks of building a model; it would break every script for no clarity. | approved (L13) |
| D10 | S0 | How does one process run under several kinds of model? | A model is a header, a process written once, a structure that changes between kinds (`Aggregate`, `Network`, `Agents`), and an engine. `ModelBuilder` writes the process; `build(structure, engine)` joins them. Rates carry over through `neighbors_<bin>` and `neighbors`. Every term is neutral: no "population", "size" or "contact" (`design-decisions.md`, "Neutral terms"). | A builder per kind (`NetworkModelBuilder`, `AgentModelBuilder`, first planned): the user writes the same transitions again for each kind, against the goal of commol (L12). One structure per kind that holds its own transitions: the same duplication inside the model files. A setting of the engine that picks the structure: the graph or the agents are data of the model, not settings, and the file would not reproduce a run. Cost: the JSON layout, `build` and `set_initial_conditions` change; commol is alpha. | approved (L12) |
| D11 | S0 | How can a structure have a narrower form with its own class (`MarkovChain`, `Network`)? | Variants: a subclass with the same kind, fields and engines, its own constructor and checks, and a `variant` tag in the JSON; the registry is keyed by `(kind, variant)`. | A subclass with no tag: the registry holds one class per kind, and a model saved as the subclass would load as the base class, unequal for pydantic. A kind per narrower form: the structures that D10 and `agent-based-models.md` D6 joined would split again. A classmethod (`Agents.network(…)`): works, but `Network` would not be a concept of the API, and `MarkovChain` would still need this mechanism. A CapWords function: against PEP 8 (L16). Cost: one more level in the registry, written once. | approved (L17) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T0 | S0 | `Process` and `ModelStructure` apart in Rust and Python; `Aggregate`; `build(structure, engine)`; `neighbors_` variables; migrate tests, examples and guide | direct | `70ca27b`, `7734914` |
| T1 | S1, S2 | Description blocks and `ModelStructure.describe`; the compartmental sections from today's printer and initial state; the test kind's sections | direct | `613e10b` |
| T2 | S3 | Renderers (text, Markdown, LaTeX, JSON) and `save`; layouts (the seven sections, `"odd"`); Mermaid and DOT; remove `print_equations` (D5); move its tests | direct | `613e10b` |
| T3 | S4 | `EngineType::dynamics_form` replaces `equation_lhs` | direct | `613e10b` |
| T4 | S5 | `is_stochastic`, `set_seed`; the `commol-random` crate; a stochastic test engine | direct | `6d05741` |
| T5 | S6 | Replicates in `Simulation`, `SimulationReplicates`, plotter and reports | direct | `6d05741` |
| T6 | S7 | Replicates with common random numbers in the calibration objective | direct | `6d05741` |
| T7 | S8 | Documentation, README, changelog | direct | `6d05741` and the audit commit |
| T8 | all | Final audit: every runner | direct | the audit commit |

## Delivery

- Strategy: one PR to `develop` from `refactor/model-kind-foundation`,
  squash-merged (`refactor(api): …`).
- Order: T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8.
- Version impact: breaking (D10: the model file has `process` and
  `structure`, `build` takes a structure, `set_initial_conditions` moves to
  `Aggregate`; D5: `print_equations` goes; D7: `run` returns one type);
  added
  `describe`, replicates and seeds. The changelog maps the old name.

## Evidence

Verified 2026-10-16 on `refactor/model-kind-foundation`, every runner of
`AGENTS.md`: Rust 312 tests (with `commol-random` and the `test-kind`
feature), `cargo fmt --check`, `clippy -D warnings` (with and without
`--all-features`), `rustdoc -D warnings`; Python 1116 tests (1 skipped:
`pydot` is not installed, so the DOT parse runs only as a snapshot), 71 docs
tests, `ruff`, `ty`, `vulture`; `mkdocs build --strict`. The LaTeX renderings
of both layouts compile with `latexmk` (`tests/description/test_rendering.py`).

| Spec | Evidence |
|---|---|
| S0 | `crates/commol-core/src/types/{process,aggregate,structure,model}.rs`; `context/{process,aggregate,neighbors}.py`; `ModelBuilder.build(structure, engine)`; `resolve_neighbors` (`crates/commol-model/src/flows.rs`); `check_model` reserves `neighbors`/`neighbors_*`; `EngineType::kinds`, `runs`, `engines_of`; `Engine::build` by `(engine, kind)`; variants (`AggregateVariant`, registry by `(kind, variant)`, `UnstratifiedVariant`); ruff `N`. Tests: `tests/transitions/test_neighbors.py` (Complete equals `k1 * A * B / total`; Groups with weights equals the per-category rates by hand; missing neighbors lists `Complete()` and `Groups(`; `neighbors_A` rejected), `tests/model_kinds/test_interface.py::TestVariants`, `tests/building/test_engine.py::TestBuild`, `tests/building/test_save_and_load.py`. |
| S1, S2 | `context/description.py`, `context/_model/{describe,aggregate_description,equations}.py`; `tests/description/test_description.py`. |
| S3 | `api/description.py`; `tests/description/test_rendering.py` (snapshots in `tests/description/snapshots/`), `tests/description/test_equations*.py` (the lines of `print_equations`, through `equation_forms`). |
| S4 | `DynamicsForm` (`crates/commol-core/src/types/engine.rs`); `tests/building/test_engine.py::TestEngineList`. |
| S5 | `crates/commol-random/` (reference values in `tests/reference_values.txt`); `SimulationEngine::set_seed`; `TestStochasticEngine` (`py-commol/src/test_kind.rs`); `py-commol/src/engine.rs` tests. |
| S6 | `Simulation.run/run_at(seed=)`, `Simulation.replicates`, `ReplicateRuns`, `SimulationReplicates`, `run_replicates`; `tests/model_kinds/test_stochastic.py::{TestSeeds,TestReplicates,TestLayers}`. |
| S7 | `ReplicateSummary`, `crps`, `Replicates`, `LossConfig::replicate_loss`, `CalibrationProblem::with_replicates`; `Quantile`; `rust_loss_config`; `replicate_tests` in `crates/commol-calibration/src/types.rs`; `tests/model_kinds/test_stochastic.py::TestCalibration`. |
| S8 | `docs/guide/describing-models.md`; `docs/guide/simulations.md` (seeds and replicates), `docs/guide/calibration.md` (stochastic engines); `design-decisions.md` ("A description is data, rendered apart", "Stochastic engines and seeds"); `mental-model.md`, `maintainer-playbook.md`, `repository-map.md`, `AGENTS.md`, README, changelog. |

## Log

- L1 (user, 2026-10-14): while reviewing the README example, noted that
  printing equations fits only models with equations, and asked how a network
  or an agent model would get an exportable, readable form. Asked for one
  ODD per new kind of modelling, each with how to implement and how to
  describe it, and this ODD first, which they build on.
- L2 (user, 2026-10-14): approved D1 (a description is data, typed blocks,
  renderers apart).
- L3 (user, 2026-10-14): approved D2 (the seven sections); a section with
  no blocks is left out of a rendering.
- L4 (user, 2026-10-14): asked whether Markdown viewers draw Mermaid and DOT.
  Most draw Mermaid and almost none DOT. Approved D3 with two changes: the
  Markdown rendering embeds only the state graph of the model (Mermaid
  draws badly past a few hundred edges; the expanded graph goes to the
  `.dot` file), and `_repr_markdown_` shows the description in a notebook.
- L5 (user, 2026-10-14): approved D4 (`EngineType::dynamics_form`).
- L6 (user, 2026-10-14): approved D5 (remove `print_equations`; the README
  example of PR #110 moves to `describe()` in S8).
- L7 (user, 2026-10-14): asked whether two people with the same seed get the
  same result. Checked: `SmallRng` is documented as non-portable (rand
  0.10.3, `rngs/small.rs`). Approved D6 with `Xoshiro256PlusPlus` and samplers
  of commol's own. A pure-Rust libm in the JIT is not needed, as long as the
  documentation states the three levels of reproducibility.
- L8 (user, 2026-10-14): did not want the return type of `run` to depend on
  an argument. Approved D7 as `run(…, seed)` for one run and
  `simulation.replicates(n, seed).run(…)` for several, each with one return
  type.
- L9 (user, 2026-10-14): the user, not the library, chooses what to fit.
  Approved D8 with `replicate_summary` required for a stochastic engine:
  `"mean"`, `"median"`, `Quantile(q)` or `"crps"`, computed in `LossConfig`.
- L10 (user, 2026-10-14): while reviewing `stochastic-compartmental` D2,
  approved a shared crate `commol-random` (generator, seeds, samplers, sum
  tree of rates), made here because the stochastic test engine needs it.
- L11 (user, 2026-10-14): while reviewing `network-models.md` D5, approved
  the rename of `ModelBuilder` here (D9), since `MarkovChainBuilder` comes
  first and this ODD already breaks the API.
- L12 (user, 2026-10-14): stated the goal of commol: one process modelled with
  difference equations, differential equations, stochastic engines, Markov
  chains, networks or agents, changing as little syntax as possible; the code
  must read naturally and the design must stay abstract, with no field of
  application assumed. Approved D10 with neutral terms (`structure`,
  `Aggregate`, `Network`, `neighbors_<bin>`, `total`; "population", "size"
  and "contacts" rejected). D9 and the builders of the later ODDs are to be
  revisited under it.
- L13 (user, 2026-10-14): approved D9 as revised under D10: `ModelBuilder`
  keeps its name, and S9 (the rename) is gone.
- L14 (user, 2026-10-15): approved, in `agent-based-models.md` D3, engines
  named by method for every structure; S0 maps an engine to several kinds.
- L15 (user, 2026-10-15): approved, in `agent-based-models.md` D5, an
  `"odd"` layout of the description for every structure; the name stays
  `"odd"`, the standard name of the protocol, and the documentation explains
  it.
- L16 (user, 2026-10-15): Python names must always follow PEP 8. `ruff
  check --select N` finds five `N805` (validators without `@classmethod`);
  S0 enables the rules and fixes them.
- L17 (user, 2026-10-15): approved D11, variants of a structure, so
  `MarkovChain` and `Network` are classes that load back as themselves.
- L18 (user, 2026-10-15): asked for the uncertainty and identifiability
  program (`odd/inference-foundation.md`) before the benchmark; the
  out-of-scope note of S7 now points to its ODDs. This ODD runs before that
  program, which builds on its seeds, replicates and calibration objective.
- L19 (implementation, 2026-10-16): `Aggregate` stores its fractions as
  mappings by id, in Python and in the file (`bin_fractions={"A": 0.99}`),
  so the constructor of S0 is the plain Pydantic one; Pydantic calls a custom
  `__init__` when it validates a file, so a list storage with a mapping
  constructor would need two shapes. Rust keeps the order (`IndexMap`).
  `BinFraction`, `StratificationFraction(s)` and `CompartmentFraction` go.
- L20 (implementation): the interface of a kind reads the process from the
  model (`outputs_by_source(model)`, `series_label(model, id)`), and the
  methods of `Aggregate` that need the bins take the process
  (`initial_state(process)`, `subgroup_total(process, …)`); a structure that
  held a reference to its process would keep a stale copy.
- L21 (implementation): `build(structure, engine)` is typed with a `TypeVar`
  of the structure class, which gives what the overloads of S0 asked for;
  `EngineName` (all the engines of a release) replaces
  `CompartmentalEngineType`.
- L22 (implementation): `neighbors_<bin>` under `Groups` resolves to sums of
  compartments found by label, not to `total_<cat>` names, which a category
  shared by two stratifications would make ambiguous. The planner decides
  whether a rate is absolute before it resolves them.
- L23 (implementation): the unit check read each raw rate, so every
  proportional rate (`k2` in `1/day`) failed it, and so would every rate of
  the neighbors. It now reads the flows the engine plans (rate × source for a
  proportional rate); the changelog lists the fix.
- L24 (implementation): the stochastic test engine draws events one at a time
  (an exact Gillespie run with a `SumTree`), so a run does not depend on the
  positions it reports at. `Simulation.replicates` returns `ReplicateRuns`.
  With `"crps"` the loss is the weighted mean score, whatever `loss_function`
  is; a constraint at positions is checked on every replicate. The
  replicates of a calibration run one after the other: the objective is
  generic and the engines need not be `Send`.
