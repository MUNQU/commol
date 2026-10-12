# Mental Model

[Back to Codebase Guide](index.md)

Commol builds compartment models from a declarative definition and runs them
as difference equations. Python owns the user-facing API and validation; Rust
owns every rule about how a model becomes numbers.

## What Commol is

| It is | Evidence |
|---|---|
| A Python library with a fluent builder | `py-commol/commol/api/model_builder.py` |
| A validated, serializable model definition | `py-commol/commol/context/` (Pydantic) |
| A Rust difference-equations engine | `crates/commol-difference/` |
| A JIT compiler for rate expressions | `crates/commol-core/src/math_expression/jit/` (Cranelift) |
| A calibration toolkit: optimizers, constraints, ensembles | `crates/commol-calibration/`, `py-commol/commol/api/calibrator.py`, `probabilistic_calibrator.py` |

## What Commol is not

| It is not | Boundary |
|---|---|
| A domain library | It knows bins, categories and rates, not diseases or drugs. Domain vocabulary belongs in user scripts. |
| An ODE solver | `ModelTypes::DifferentialEquations` exists so any model can be expanded, but no engine runs it: the difference engine rejects that typology. |
| A clock | A model has no physical time: rates are per step and periods are step counts. `TimeScale` converts at the edges (see [Design decisions](design-decisions.md#physical-time-is-not-in-the-model)). |
| A rule engine for transitions | `Transition.condition` is stored and serialized, but the engine does not evaluate it. |

## How a model flows through the layers

```text
ModelBuilder ... .build()          -> commol.context.Model (Pydantic validation)
Simulation(model)                  -> Model.model_dump_json()
                                   -> commol_rs.core.Model.from_json(...)
                                   -> commol_rs.difference.DifferenceEquations(...)
                                        expand bins and stratifications into compartments
                                        build the initial state
                                        plan the transition flows
Simulation.run(n)                  -> DifferenceEquations.run(n) -> rows of outputs
Calibrator(simulation, problem)    -> commol_rs.calibration.calibrate(engine, ...)
```

The same engine functions also answer questions without a simulation:
`Model.initial_state()`, `print_equations` and the unit checker call
`initial_state`, `transition_flows` and `expanded_outputs` through
`py-commol/commol/context/_model/engine_state.py`.

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
