---
name: Audit
about: The check of what one milestone implements against its articles, which blocks its release
title: "chore(methods): what X.Y.Z implements is audited against its articles"
labels: proposal, audit
---

<!-- Put this issue in the milestone of the release: it blocks the release and
waits for the other proposals and changes of the milestone
(`python .github/scripts/parallel.py --releases` writes the links). It never
moves to another milestone. The rules are in docs/development/workflow.md,
"Audits". -->

## Goal

Release X.Y.Z ships only when each method it implements has been checked against the articles it is based on: <!-- the methods, with their issues -->. Each article was obtained and read, or a decision accepts to go on without it (D4); each method agrees with its articles and has a test against a reference value, and no defect of the milestone is open.

## Why

<!-- What the milestone implements that no audit has checked. A milestone that
implements no method says so here: its record is one comment `L1`. -->

## Current state

<!-- Verified on `main` at `<sha>`, YYYY-MM-DD. What the milestone implements,
by group of issues; the rules of the repository with no published reference. -->

### Articles

<!-- Use: A, commol cites it; B, commol cites nothing or not the origin: a
proposal of reference; C, the proposal cites it; D, a review or an
alternative. Access: to obtain, open version (it may differ from the article),
open. Check each DOI. The copies go to `papers/`, which git ignores. -->

| Use | Publication | DOI | For | Access |
|---|---|---|---|---|
| | | | | |

## Out of scope

- The methods of other milestones: the audit of each one.
- The verdict of use: the task T0 of each proposal.
- The quality of the code, the tests and the documentation, which each change meets on its own.

## Touches

- Changes: none; each fix lists its paths in its own issue.
- Adds: none

## Requirements

### R1 - Each method agrees with its articles

The audit SHALL check each method of the milestone: its algorithm, formulas and thresholds agree with the article it is based on, which the auditor has read, and the guide cites it; a test checks it against a reference value (closed form, a reference implementation, a published example) or a coverage study; the guide states its assumptions, and the code checks them or warns when they fail.

- **WHEN** the audit ends **THEN** each method has a verdict, with its articles (by DOI), the test that checks it, its level of evidence (D3), and its findings or "no finding".
  Test: none; the record of the audit (D1) lists them.
- **WHEN** no source of D3 was read for a method **THEN** its verdict is "not verified", which is a finding, and the audit does not close (D4).
  Test: none; the record of the audit (D1) lists them.

### R2 - Compositions, and degenerate and extreme inputs

The audit SHALL check what one method against one article does not reach: each rule of the repository that combines methods or the results of one method over several series, parameters or blocks, and each method on degenerate and extreme inputs, inside and outside the range its tests cover.

- **WHEN** a rule combines methods or results **THEN** the record gives its reference, or a derivation or a simulation over what it accepts; a rule with none is a finding.
  Test: none; the record of the audit (D1) lists them.
- **WHEN** a method accepts a degenerate or extreme input **THEN** the record says what it does and the test that checks it: a correct result, a refusal or a warning that names the cause; a wrong result with no warning, or an input with no test, is a finding.
  Test: none; the record of the audit (D1) lists them.

### R3 - The audit closes last and blocks the release

Each finding SHALL be an open issue, from a template, with a milestone. A finding about a method of the milestone has this milestone, so it blocks the release and this issue; a finding about a method of an earlier release goes to the nearest release it fits.

- **WHEN** a finding is fixed in this milestone **THEN** the audit checks the fix against the article before this issue closes.
  Test: none; the record of the audit (D1) lists them.
- **WHEN** a change since the last release touches a method, with an issue of the milestone or with none (an update of a solver) **THEN** it is in the record.
  Test: none; T4 compares the changes with the record.

## Decisions

| ID | Req | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | R1-R3 | Where the record lives | One comment `L<n>` on this issue per family of methods | Against a page of `docs/`: the record describes `main` at one time and goes out of date | the rule of every audit ([#328 L2](https://github.com/MUNQU/commol/issues/328#issuecomment-6342602744)) |
| D2 | R1, R2 | Who audits | A panel of three judges per family: statistics, numerics, the modeller who uses the method | Against one reviewer: one lens misses what the others see | the rule of every audit ([#310 L3](https://github.com/MUNQU/commol/issues/310#issuecomment-6333760078)) |
| D3 | R1 | What counts as evidence | A reference implementation with values fixed in a test; the published article; an open version, noted as one that may differ; or "not verified" | Against the memory of the reviewer: a formula recalled wrong is the defect R1 looks for | the rule of every audit ([#310 L4](https://github.com/MUNQU/commol/issues/310#issuecomment-6334271950)) |
| D4 | R1 | What a method whose article could not be obtained does to the release | It stays "not verified" and the release waits, unless the user accepts, in an `L<n>` decision that names the method, to go on without the article | Against a note in the record only: a release could ship a method that nobody checked. Against no exception: one article out of reach would stop a release for good | the rule of every audit ([#328 L6](https://github.com/MUNQU/commol/issues/328#issuecomment-7129141609)) |

## Tasks

- [ ] T1 (R1): complete the list of methods with their articles, and obtain the articles, in `papers/`; it needs no branch, so it is done before the blockers close, without `start.sh`
- [ ] T2 (R1): audit each method against its articles; record the verdicts (D1)
- [ ] T3 (R2): audit the compositions and the degenerate and extreme inputs; record them
- [ ] T4 (R3): open an issue for each finding; compare the closed issues of the milestone, and the changes since the last release (`git log --first-parent`), with the record; check each fix
- [ ] T5: documentation (definition of done: changelog, and the references the guide is missing, by DOI)
- [ ] T6: final audit, every runner of `AGENTS.md`

## Delivery

- Branch: `chore/audit-X.Y.Z` from `main`
- Breaking: no; a fix that breaks the API says so in its own issue
- Release: the milestone of the issue
