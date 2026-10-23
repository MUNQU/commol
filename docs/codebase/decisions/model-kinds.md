# Model kinds

[Back to Design decisions](../design-decisions.md)

How a model is composed, and what lets the generic layers run any model kind.

## One interface for every model kind

The layers that do not depend on how a model is built (`Simulation`,
`Calibrator`, the uncertainty methods, the reports and the figures)
use only `Model` and the methods of `ModelStructure`
(`py-commol/commol/context/structure.py`), and in Rust only the
`SimulationEngine` trait (`crates/commol-core/src/simulation_engine.rs`). What
only an aggregate model has (compartments, initial amounts) is behind them:
`Aggregate` (`context/aggregate.py`), and in Rust `CompartmentalEngine` and
`InitialConditionPlan` (`crates/commol-model/`). A new kind, such as an agent
network, is a new `ModelStructure` subclass with its engines in the list;
nothing that exists changes.

| Part | Rule | Why, against the alternative |
|---|---|---|
| `ModelStructure` | An abstract base class (Pydantic's metaclass derives from `ABCMeta`). A missing method fails when the object is made and names it; optional methods have a default. | A `typing.Protocol` shows a missing method only under `ty`, or mid-run as an `AttributeError`, and gives no defaults. |
| Engine kinds | An engine names a method, not a kind: each entry of `EngineType` names the kinds it runs (`EngineType::kinds`). `Model` asks the list (`check_kind`) when it is made, and `Engine::build` asks it again and picks the implementation by the pair of the engine and the kind; the message of a mismatch lists the engines that run the kind (`EngineType::engines_of`). | Without the check, a model of one kind reaches the engine of another and fails deep in its compilation. One engine per kind would multiply the names: a stochastic engine runs every structure that implements it. |
| Values calibration sets | `parameter`, `factor` and `noise` are generic. Each kind declares the types of value of its own it accepts (`ModelKind::value_types`, which `ModelStructure.calibration_value_types` asks); the aggregate kind accepts `initial_condition`. The engine resolves and writes them (`SimulationEngine::plan_values`, `apply_values`). | The calibration crate would know compartments, and a new kind could not calibrate values of its own without editing it. |
| Report sections | `ModelStructure.report_sections` adds the sections of a kind to `Model.applied_parameters` (default: none); the aggregate kind adds `initial_amounts` and `category_splits` (`Aggregate.report_sections`). | A check of the kind in `reporting.py` brings back a branch per kind. |
| Expressions in constraints | The engine names the values of an output row: `output_names`, `state_outputs` (their sum is `total`) and `output_sums` (a bin total, for an engine of the aggregate kind). | The calibration would rebuild bin totals from compartment labels. |

The test kind (`tests/support/kinds.py`, engine `TestEngine` in
`py-commol/src/test_kind.rs`) is written only against the interface. The
extension has it only when it is built with the cargo feature `test-kind`, as
the tests and CI build it; a release build does not.

**Guards:** `tests/model_kinds/` (a test model through a simulation, a
calibration, a report and a plot; the kind check both ways; a process the test
kind cannot run),
`crates/commol-calibration/tests/test_model_values.rs` (the calibration with
an engine that is not of the aggregate kind: values through `plan_values` and
`apply_values`, constraints through `output_sums` and `state_outputs`),
`py-commol/src/engine.rs::an_engine_rejects_a_model_of_another_kind`,
`crates/commol-core/src/types/engine.rs::an_engine_rejects_a_model_of_another_kind`.

## A model is a header, a process, a structure and an engine

`Model` (Python `context/model.py`, Rust `crates/commol-core/src/types/model.rs`)
is one class for every kind: `name`, `description`, `version`, `parameters`,
`engine` (`{type, settings}`), `process` and `structure`, and in Python only,
`step_unit` and `origin` (the engines never read them). The
**process** (`Process`: bins, accumulators, stratifications, transitions) is
what changes and how, written once. The **structure** is what changes between
kinds of model: `Aggregate` lays the process out as amounts per compartment;
`Network` and `Agents` come with their proposals ([#144](https://github.com/MUNQU/commol/issues/144), [#145](https://github.com/MUNQU/commol/issues/145)). It is tagged in JSON by `kind`. In
Python `Model` is generic in its structure class: `ModelBuilder.build(structure,
engine)` returns a `Model` of the class of `structure` (a `TypeVar`, never a
value), so `model.structure.bin_amounts` type-checks, and `Model.from_json` reads the
class from the `kind` and `variant` of the file. The engine is part of the
model because it sets what a rate means.

The interface of a kind stays on the structure and reads the process from the
model (`outputs_by_source(model)`, `series_label(model, id)`); the methods of
`Aggregate` that need the bins take the process (`initial_state(process)`).

| Alternative | Why not |
|---|---|
| Inheritance: an abstract `Model` with a `CompartmentalModel` | Direct access (`model.bins`), but two parallel hierarchies, of models and of builders, and the header repeated per kind. With composition, `Simulation` and `Calibrator` take one type. |
| A builder per kind (`NetworkModelBuilder`, …) | The user writes the same transitions again for each kind. |
| One structure per kind that holds its own transitions | The same duplication, inside the model files. |
| A setting of the engine that picks the structure | A graph or a set of agents is data of the model, not a setting; the file would not reproduce a run. |
| A structure that holds a reference to its process | Two copies to keep equal: a model copied or changed would carry a stale process. |

A structure may have **variants**: a subclass that keeps the kind, the fields
and the engines of its structure, narrows the constructor and may add checks
(`MarkovChain(Aggregate)`, `Network(Agents)` in later proposals, [#142](https://github.com/MUNQU/commol/issues/142) and [#144](https://github.com/MUNQU/commol/issues/144)). The JSON holds the
kind and the variant; the Python registry (`ModelStructure._kinds`) is keyed by
`(kind, variant)`, so a model loads back as the class it was saved with. Rust
reads the variant only to run its checks (`AggregateVariant::check`), so the
engines do not change. A subclass with no tag would load back as its base class
and compare unequal; a kind per narrower form would split the structures again.

The amounts and fractions of `Aggregate` are mappings by id
(`bin_amounts={"A": 990}`), in Python and in the file, and Rust keeps their
order (`IndexMap`). A list of `{"bin", "amount"}` records needs a second shape for the constructor, and
Pydantic calls a custom `__init__` when it validates a file too.

**Guards:** `crates/commol-core/src/types/model.rs` (tests),
`tests/building/test_save_and_load.py`,
`tests/model_kinds/test_interface.py::TestStructureInterface`,
`tests/model_kinds/test_interface.py::TestVariants`,
`tests/building/test_engine.py::TestBuild`.

## Neutral terms carry a process across structures

A rate may read `neighbors_<bin>`, the amount of a bin among the neighbors of
a unit, and `neighbors`, their total. They carry a process from one structure
to another: a rate that reads them and no compartment is proportional, a rate
per unit of its source, which a network or an agent model reads per node or
per agent. Under `Aggregate` the structure says who the neighbors are, with no
default:

| `Aggregate.neighbors` | `neighbors_<bin>` from a compartment of category `g` | `neighbors` |
|---|---|---|
| `Complete()` | The bin total | `total` |
| `Groups(stratification, weights)` | `Σ_g' w[g][g'] × (amount of the bin in g')` | `Σ_g' w[g][g'] × (amount of every bin in g')` |

`plan_transition_flows` (`crates/commol-model/src/flows.rs`) decides whether a
rate is absolute **before** `resolve_neighbors` writes the terms as the sums
they are, found by label (`CompartmentLayout::with_category`), so
`k1 * neighbors_B / neighbors` from `A` is the proportional flow
`k1 × A × B / total` under `Complete()`. `check_model`
(`crates/commol-model/src/builder.rs`) reserves the names, rejects a
`neighbors_<id>` that names no bin and a formula parameter that reads them,
and asks for `neighbors` when a rate reads them; `Neighbors::check` checks the
groups and their weights.

Every term is neutral: no "population", "size" or "contact", which assume
individuals and a field. A default of `Complete()` would choose a modelling
assumption for the user.

**Guards:** `tests/transitions/test_neighbors.py`,
`crates/commol-model/src/flows.rs` (tests `complete_neighbors_…`,
`group_neighbors_…`, `a_rate_that_reads_the_neighbors_…`),
`crates/commol-model/src/builder.rs` (tests `a_name_of_the_neighbors_…`,
`a_rate_reads_the_neighbors_of_a_bin_only_when_the_structure_has_them`).
