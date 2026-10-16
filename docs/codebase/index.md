# Codebase Guide

[Back to Home](../index.md)

This guide helps you find the right code before you change Commol. It is an
index for maintainers and AI agents, not an API reference: for the public API,
see the [User Guide](../guide/core-concepts.md) and the
[API Reference](../api/model-builder.md).

!!! note "These pages describe `main`"
    The `dev` version of this site is built from `main` and may describe
    unreleased code. The `latest` version is built from the last release tag.

## Who this is for

| Reader | Use this guide to |
|---|---|
| New maintainer | Build the mental model before you touch files. |
| Contributor | Find the module that owns a behavior. |
| Reviewer | Check that a pull request changes the right layer. |
| AI agent | Start from [`AGENTS.md`](https://github.com/MUNQU/commol/blob/main/AGENTS.md), then read these pages by path. |

## 90-second mental model

Commol has three layers ([Architecture](architecture.md) draws them). The user writes a model in Python. Python validates
it and serializes it to JSON. The Rust engine builds and runs it.

```text
User code
  |
  v
py-commol/commol/api/        ModelBuilder, Simulation, Calibrator, reports
  |
  v
py-commol/commol/context/    Pydantic model: validation and serialization
  |
  |  Model.model_dump_json()
  v
py-commol/src/               PyO3 bindings (module commol.commol_rs._commol_rs)
  |
  v
crates/commol-core/          model types, expressions (Cranelift JIT), SimulationEngine trait
crates/commol-model/         expansion, initial state, flow planning, rates, derivatives
crates/commol-difference/    the discrete step: X += derivatives
crates/commol-differential/  the ODE engine: diffsol integrates derivatives
crates/commol-calibration/   the problem in layers, optimizers (argmin fork), starts
crates/commol-inference/     parameter sample, predictive, scores, quantiles
```

**Golden rule: the Rust engine is the single source of truth.** When Python
needs to know how the engine expands, initializes, plans or simulates a model,
it asks the engine through a binding. It never re-implements the rule.

## Guide pages

| Page | Job |
|---|---|
| [Mental model](mental-model.md) | What Commol is, what it is not, and the invariants every change must keep. |
| [Architecture](architecture.md) | Diagrams of the two halves, the Rust crates, and the path of a simulation and a calibration. |
| [Repository map](repository-map.md) | Which module owns which behavior, and where new code goes. |
| [Design decisions](design-decisions.md) | The rules that are not obvious from the code, and the tests that guard them. |
| [Conventions](conventions.md) | How code, comments, docs and commits are written. |
| [Maintainer playbook](maintainer-playbook.md) | Checklists by type of change. |

## Recommended reading path

1. [Mental model](mental-model.md).
2. [Architecture](architecture.md) for the picture of the code.
3. [Repository map](repository-map.md) before you edit code.
4. The [Design decisions](design-decisions.md) of the area you change.
5. [Maintainer playbook](maintainer-playbook.md) before you open or review a
   pull request.

## Related pages

| Page | Job |
|---|---|
| [Development workflow](../development/workflow.md) | Setup, branches, CI, and merge rules. |
| [Contributing](../development/contributing.md) | Bug reports, pull requests, code review. |
| [Release process](../development/release.md) | How a version reaches PyPI. |
| [Model structure](../advanced/model-structure.md) | Reference of the model data types. |
