---
name: Proposal
about: A change that needs a specification - a feature, a refactor or a breaking change
title: "<type>(<scope>): <outcome>"
labels: proposal
---

## Goal

<!-- One paragraph: what becomes possible when this is done, in the present
tense. Keep the whole issue short and scoped: split a large proposal into
sub-issues, one per group of requirements. -->

## Why

<!-- Which problem it solves, and the evidence: an issue, a discussion, a
benchmark, a paper. -->

## Current state

<!-- Verified on `develop` at `<sha>`, YYYY-MM-DD. What the code does today,
with the paths and symbols an agent must read first:
`crates/<crate>/src/<file>.rs` (`function`), `py-commol/commol/<module>.py`
(`Class.method`). No overviews of the repository. -->

## Out of scope

- <!-- What this proposal does not do, and the issue that does it, if any. -->

## Requirements

<!-- One block per behavior. The ids (R1, R2, …) are stable: the decisions,
the tasks, the tests and the pull request name them. Each scenario names the
test that checks it; the test is written before the code. -->

### R1 - <name>

The <component> SHALL <behavior>.

- **WHEN** <condition> **THEN** <observable result>.
  Test: `py-commol/tests/<package>/test_<topic>.py::test_<name>`
- **WHEN** <invalid input> **THEN** <error that names …>.
  Test: `crates/<crate>/tests/<file>.rs::<name>`

## Decisions

<!-- To settle when the implementation starts. Each row
says why the recommendation beats each alternative. Status: proposed, or
approved with a link to the comment that approves it. -->

| ID | Req | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | R1 | | | | proposed |

## Tasks

<!-- In order. The tests of a requirement come before its code. -->

- [ ] T1 (R1): write the tests of R1; they fail
- [ ] T2 (R1): implement R1; its tests pass
- [ ] T3: documentation (definition of done: guide, design decisions, changelog)
- [ ] T4: final audit, every runner of `AGENTS.md`

## Delivery

- Branch: `<type>/<name>` from `develop`
- Breaking: yes | no
- Release: the milestone of the issue

<!-- Log: each decision or implementation finding is a comment,
"L<n> (decision|implementation): …". Evidence: the body of the pull request,
with each requirement and the tests that pass. -->
