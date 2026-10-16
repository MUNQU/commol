# Mental Model

[Back to Codebase Guide](index.md)

Commol builds compartment models from a declarative definition and runs them
as difference equations, as ordinary differential equations, or as stochastic
processes of whole units. Python owns the
user-facing API and validation; Rust owns every rule about how a model becomes
numbers. A model has a **process** written once (bins, stratifications,
transitions), a **structure** that lays it out and sets its **kind** (today
`aggregate`: amounts per compartment), and an **engine** that runs it; the
generic layers work with any kind.

## What Commol is

| It is | Evidence |
|---|---|
| A Python library with a fluent builder of the process | `py-commol/commol/api/model_builder.py` |
| A validated, serializable model definition: a header, a process, a structure and an engine | `py-commol/commol/context/` (Pydantic): `model.py`, `process.py`, `structure.py`, `aggregate.py` |
| Generic layers that work with any model kind | `Simulation`, `Calibrator`, reports and plots use only `ModelStructure`; Rust calibration uses only `SimulationEngine` (see [Design decisions](design-decisions.md#one-interface-for-every-model-kind)) |
| One model compiler for every engine: expansion, flows, rates, the change of one event of each flow, the right-hand side | `crates/commol-model/` |
| A Rust difference-equations engine | `crates/commol-difference/` |
| A Rust ODE engine on `diffsol` | `crates/commol-differential/` |
| Two Rust stochastic engines of whole units: events in continuous time (direct method, tau-leaping) and the chain binomial | `crates/commol-stochastic/` |
| Reproducible random numbers for stochastic engines: the generator, replicate seeds, samplers and a sum tree of rates | `crates/commol-random/` |
| A JIT compiler for rate expressions | `crates/commol-core/src/math_expression/jit/` (Cranelift) |
| A calibration toolkit: optimizers, constraints, ensembles | `crates/commol-calibration/`, `py-commol/commol/api/calibrator.py`, `probabilistic_calibrator.py` |

## What Commol is not

| It is not | Boundary |
|---|---|
| A domain library | It knows bins, categories and rates, not diseases or drugs. Domain vocabulary belongs in user scripts. |
| A numerical-methods library | The differential engine delegates the integration to `diffsol`; Commol keeps no ODE method of its own (see [Design decisions](design-decisions.md#the-solver-is-diffsol)). |
| A clock | A model has no physical units: rates are per step and positions and periods are step counts, and the axis is the independent variable of the model, whatever quantity it is. `AxisUnits` converts at the edges (see [Design decisions](design-decisions.md#physical-units-are-not-in-the-model)). |
| A rule engine for transitions | A transition has no condition. A rate expresses a gate, such as `if(A > 0, k, 0)` (see [Mathematical expressions](../guide/mathematical-expressions.md)), and [stratified rates](../guide/transitions.md#stratified-rates) set a rate per category. |

## Model kind and engine

| Term | What it is | Where |
|---|---|---|
| Process | What changes and how, written once for every kind: bins, accumulators, stratifications, transitions. | `Process` (`crates/commol-core/src/types/process.rs`, `context/process.py`); `Model.process` |
| Model kind | How the process is laid out: the kind of its `structure`. `aggregate`: the bins expanded over the stratifications into compartments, each holding an amount. A **variant** is a narrower form of a structure with the same kind and engines. | `ModelKind` (`crates/commol-core/src/types/engine.rs`); `ModelStructure` and its subclasses in Python |
| Engine | A method: how a model advances along the axis, and what a rate means: `DifferenceEquations`, `DifferentialEquations`, `StochasticEvents`, `StochasticSteps`. | `EngineType`, the one list (`engine.rs`); `Model.engine` |

Each engine runs some kinds (`EngineType::kinds`). A model whose engine does
not run its kind is rejected when it is made, and the error lists the engines
of its kind.

## How a model flows through the layers

```text
ModelBuilder ... .build(Aggregate(...), engine)
                                   -> commol.context.Model[Aggregate]
                                      (Pydantic validation; the engine's list
                                      checks model.engine and its kind; the
                                      structure checks itself against the process)
Simulation(model)                  -> engine_state.build_engine(model JSON)
                                   -> commol_rs.core.Model.from_json(...)
                                   -> commol_rs.engine.Engine(...): the implementation of
                                      the engine model.engine.type names (EngineType list)
                                      for the kind of model.structure
                                   every engine of the aggregate kind compiles it with commol-model (CompiledModel):
                                        expand the bins of the process over its stratifications
                                        build the initial state
                                        plan the transition flows, resolve the neighbors, compile the rates
                                        flow_rates and flow_changes (one event of each flow)
                                        derivatives(step, t, X) = Σ changes × flows
Simulation.run(n) / run_at(at)     -> difference: X += derivatives, once per step
                                      differential: diffsol integrates derivatives
                                      stochastic events: one event (or a leap) at a time
                                      stochastic steps: a chain binomial draw per step
                                      (a stochastic engine: set_seed, then reset)
Simulation.replicates(n).run(…)    -> Engine.run_replicates: one seed per replicate,
                                      run in parallel (run_replicates, rayon)
Calibrator(simulation, problem)    -> commol_rs.calibration.calibrate(engine, ...):
                                      parameters by id; values of the kind, such as
                                      initial conditions, through the engine
```

The same model functions also answer questions about a process and its
aggregate structure without a simulation:
`Aggregate.initial_state(process)`, `Model.describe` and the unit checker call
`initial_state`, `transition_flows` and `expanded_outputs` of `commol_rs.model`
through `py-commol/commol/context/_model/engine_state.py`.

## Invariants

These hold for every change. A pull request that breaks one needs a design
discussion first.

- **Rust is the single source of truth.** Python asks the engine how a model
  expands, which state it starts with, and which flows it has. It never
  keeps a second implementation of those rules.
- **Compartments are identified by labels, not by names.** A compartment is a
  bin plus one category per applied stratification (`OutputLabels`,
  `CompartmentLayout`). Code never parses a compartment name or matches it by
  prefix: bin `A` and bin `A_x` must never be confused.
- **A model that cannot be simulated as written is rejected when it is built.**
  The flow planner returns an error instead of dropping a transition.
- **Stratification declaration order is part of the model.** Stratifications
  expand in declaration order, and a conditional stratification may only
  reference stratifications declared before it.
- **A run is reproducible from its model and its seed, on one platform and one
  commol version.** The same formula always runs the same compiled code
  (global JIT cache); a stochastic engine draws every random number from one
  generator seeded by `SimulationEngine::set_seed`, which `reset` restores;
  calibrations accept a `seed`. See
  [Stochastic engines and seeds](design-decisions.md#stochastic-engines-and-seeds).
- **What the user saves, the engine reloads.** `Model.to_json` writes every
  field, and calibrated values written back to a model reproduce the
  state the optimizer scored.

## Checklist

- [ ] Name the user-facing behavior first.
- [ ] Find the layer that owns it in the [Repository map](repository-map.md).
- [ ] If Python needs an engine rule, add or use a binding instead of
      re-implementing it.
- [ ] Read the tests next to that code before you add new patterns.

## Navigation

Previous: [Codebase Guide](index.md) | Next: [Repository map](repository-map.md)
