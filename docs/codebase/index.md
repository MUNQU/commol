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

## The golden rule

Rust decides how a model becomes numbers, and Python asks it. The
[Mental model](mental-model.md#the-golden-rule) states the rule and draws the
layers.

## Guide pages

| Page | Job |
|---|---|
| [Mental model](mental-model.md) | What Commol is, what it is not, and the invariants every change must keep. |
| [Architecture](architecture.md) | Diagrams of the two halves, the Rust crates, and the path of a simulation and a calibration. |
| [Repository map](repository-map.md) | Which module owns which behavior, and where new code goes. |
| [Design decisions](design-decisions.md) | The rules that are not obvious from the code, one page per area, with the tests that guard them. |
| [Conventions](conventions.md) | How code, comments, docs and commits are written. |
| [Design principles](design-principles.md) | The structure, the rules of each unit and of the interface, and the proof that every part of the code follows. |
| [Tests](tests.md) | The layout, the models, the oracles and the tolerances of the tests; what a test must not check, duplicates, skips and the time budget. |
| [Coverage and mutation testing](coverage.md) | The coverage of Python and Rust, its thresholds, the lines left out and why, and mutation testing. |
| [Maintainer playbook](maintainer-playbook.md) | Checklists by type of change. |

## Recommended reading path

1. [Mental model](mental-model.md).
2. [Architecture](architecture.md) for the picture of the code.
3. [Repository map](repository-map.md) before you edit code.
4. The page of [Design decisions](design-decisions.md) of the area you change.
5. [Maintainer playbook](maintainer-playbook.md) before you open or review a
   pull request.

## Related pages

| Page | Job |
|---|---|
| [Development workflow](../development/workflow.md) | Setup, branches, CI, and merge rules. |
| [Contributing](../development/contributing.md) | Bug reports, pull requests, code review. |
| [Release process](../development/release.md) | How a version reaches PyPI. |
| [Model structure](../advanced/model-structure.md) | Reference of the model data types. |
