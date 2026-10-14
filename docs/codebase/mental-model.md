# Mental Model

[Back to Codebase Guide](index.md)

Commol builds compartment models from a declarative definition and runs them
as difference equations or as ordinary differential equations. Python owns the
user-facing API and validation; Rust owns every rule about how a model becomes
numbers. A model has a **kind** (today only `compartmental`) and an **engine**
that runs that kind; the generic layers work with any kind.

## What Commol is

| It is | Evidence |
|---|---|
| A Python library with a fluent builder | `py-commol/commol/api/model_builder.py` |
| A validated, serializable model definition: a header, an engine and the structure of its kind | `py-commol/commol/context/` (Pydantic): `model.py`, `structure.py`, `compartmental.py` |
| Generic layers that work with any model kind | `Simulation`, `Calibrator`, reports and plots use only `ModelStructure`; Rust calibration uses only `SimulationEngine` (see [Design decisions](design-decisions.md#one-interface-for-every-model-kind)) |
| One model compiler for both engines: expansion, flows, rates, the right-hand side | `crates/commol-model/` |
| A Rust difference-equations engine | `crates/commol-difference/` |
| A Rust ODE engine on `diffsol` | `crates/commol-differential/` |
| A JIT compiler for rate expressions | `crates/commol-core/src/math_expression/jit/` (Cranelift) |
| A calibration toolkit: optimizers, constraints, ensembles | `crates/commol-calibration/`, `py-commol/commol/api/calibrator.py`, `probabilistic_calibrator.py` |

## What Commol is not

| It is not | Boundary |
|---|---|
| A domain library | It knows bins, categories and rates, not diseases or drugs. Domain vocabulary belongs in user scripts. |
| A numerical-methods library | The differential engine delegates the integration to `diffsol`; Commol keeps no ODE method of its own (see [Design decisions](design-decisions.md#the-solver-is-diffsol)). |
| A clock | A model has no physical time: rates are per step and periods are step counts, and the axis is the independent variable of the model, whatever quantity it is. `TimeScale` converts at the edges (see [Design decisions](design-decisions.md#physical-time-is-not-in-the-model)). |
| A rule engine for transitions | A transition has no condition. A rate expresses a gate, such as `if(A > 0, k, 0)` (see [Mathematical expressions](../guide/mathematical-expressions.md)), and [stratified rates](../guide/transitions.md#stratified-rates) set a rate per category. |

## Model kind and engine

| Term | What it is | Where |
|---|---|---|
| Model kind | How a model is built: its `structure`. `compartmental`: bins expanded over stratifications, joined by transitions. | `ModelKind` (`crates/commol-core/src/types/engine.rs`); `ModelStructure` and its subclasses in Python |
| Engine | How a model of a kind advances along the axis, and what a rate means: `DifferenceEquations`, `DifferentialEquations`. | `EngineType`, the one list (`engine.rs`); `Model.engine` |

Each engine runs one kind. A model whose engine runs another kind is rejected
when it is made.

## How a model flows through the layers

```text
ModelBuilder ... .build(engine)    -> commol.context.Model[CompartmentalStructure]
                                      (Pydantic validation; the engine's list
                                      checks model.engine and its kind)
Simulation(model)                  -> engine_state.build_engine(model JSON)
                                   -> commol_rs.core.Model.from_json(...)
                                   -> commol_rs.engine.Engine(...): the engine
                                      model.engine.type names (EngineType list),
                                      after it checks the kind of model.structure
                                   every compartmental engine compiles it with commol-model (CompiledModel):
                                        expand bins and stratifications into compartments
                                        build the initial state
                                        plan the transition flows and compile the rates
                                        derivatives(step, t, X) = Σ flows
Simulation.run(n) / run_at(at)     -> difference: X += derivatives, once per step
                                      differential: diffsol integrates derivatives
Calibrator(simulation, problem)    -> commol_rs.calibration.calibrate(engine, ...):
                                      parameters by id; values of the kind, such as
                                      initial conditions, through the engine
```

The same model functions also answer questions about a compartmental structure
without a simulation:
`CompartmentalStructure.initial_state()`, `print_equations` and the unit checker call
`initial_state`, `transition_flows` and `expanded_outputs` of `commol_rs.model`
through `py-commol/commol/context/_model/engine_state.py`.

## Invariants

These hold for every change. A pull request that breaks one needs a design
discussion first.

- **Rust is the single source of truth.** Python asks the engine how a model
  expands, which population it starts with, and which flows it has. It never
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
- **Results are reproducible.** The same formula always runs the same compiled
  code (global JIT cache), and calibrations accept a `seed`.
- **What the user saves, the engine reloads.** `Model.to_json` writes every
  field, and calibrated values written back to a model reproduce the
  population the optimizer scored.

## Checklist

- [ ] Name the user-facing behavior first.
- [ ] Find the layer that owns it in the [Repository map](repository-map.md).
- [ ] If Python needs an engine rule, add or use a binding instead of
      re-implementing it.
- [ ] Read the tests next to that code before you add new patterns.

## Navigation

Previous: [Codebase Guide](index.md) | Next: [Repository map](repository-map.md)
