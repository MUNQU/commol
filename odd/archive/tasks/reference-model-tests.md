# Feature: Reference model tests

Goal: add one test for each problem with a stated result in the local reference
catalogue, to show that commol can express each model and that the engine
computes it correctly. A problem that commol cannot express, or that it
computes wrong, becomes a finding (F1, F2, …) for a later ODD.
Branch: `test/reference-models` from `develop`. Both prerequisites are merged:
the test refactor (PR #100, `c3cc346`) and the ODE engine (PR #101,
`3c13102`).
Engram mirror: topic `odd/reference-model-tests/tasks` (project `commol`).
Runners: `maturin develop --release` then `pytest` and `pytest -m docs` (from
`py-commol/`), `ruff check . && ruff format --check . && ty check commol &&
vulture commol .vulture_allowlist.py --min-confidence 80`, `mkdocs build
--strict` (from the root), and the leak check of Confidentiality.

## Current state (verified 2026-10-13 on `develop` at `0672c1c`)

- Confidentiality: the catalogue directory is in `.git/info/exclude` and
  untracked; `.git/info/forbidden-terms` holds 8 terms. No tracked file, commit
  message (all branches), branch name, or PR title or body (last 30 PRs)
  contains one. `test-map.md` does not exist yet (T1).
- Catalogue: 89 problems with a stated result: 33 solved examples of the
  appendices (linear algebra, first-order ODEs, systems of ODEs) and 56 project
  questions with a given result. 69 open project questions are out of scope.
- Engine: `DifferentialEquations` models run with `diffsol` (`tsit45`, `bdf`);
  outputs at any position (`Simulation.run_at`); impulses (`impulse_at`,
  `impulse_every`); accumulators integrate their flows. Not supported: delays,
  state-dependent events, stochastic models, rates that read an accumulator,
  and partial or integral equations (continuous age).
- `tests/support/` already has what class C needs: `build_ode` and `TIGHT`
  (`rtol=1e-10`, `atol=1e-8`) in `models.py`, `assert_close` and `ENGINE_REL`
  (`1e-9`) in `assertions.py`, `reference_run` (difference engine in plain
  Python) in `reference.py`.
- Initial conditions are fractions of an integer `population_size`
  (`InitialConditions`, `u64` in Rust). A problem with a non-integer total and
  a rate that is not homogeneous in the population (mass action without `/N`)
  cannot be rescaled; it is a candidate gap (class G).
- An impulse test with synthetic values exists
  (`tests/transitions/test_impulses.py::TestJump::test_pulse_vaccination_reaches_the_periodic_orbit`).
  A reference test uses the values and the stated result of its source, so it
  checks a different claim and both stay.

## Confidentiality (hard rule)

The reference catalogue is private material. Its directory under
`py-commol/tests/` is listed in `.git/info/exclude` (local, not tracked), not in
`.gitignore` (tracked), so its name never reaches git or GitHub.

- No tracked file, commit message, branch name, PR title or PR body names the
  source, its authors, its title, its year, or its chapter, section, question or
  page numbers. This ODD follows the same rule.
- Tests name the model by its epidemiology (`test_discrete_sir_final_size_relation`),
  never by its place in the source.
- The map from each test to its catalogue entry lives in the excluded
  directory (`test-map.md`), so it is local only.
- Leak check before every commit and before the PR:
  `git diff develop... | grep -i -F -f .git/info/forbidden-terms` and
  `git log develop.. --format=%B | grep -i -F -f .git/info/forbidden-terms`
  print nothing. `.git/info/forbidden-terms` is local and lists the terms.
- Do not copy prose from the catalogue. A test holds the model, the parameter
  values and the stated result as code, with a one-line description in our words.

### Attribution (approved, L8)

Each test cites the primary sources of its model, never the catalogue:

- Every module of `reference_models/` has a `References` section in its
  docstring: the original papers of the model and of its parameter values
  (authors, year, title, journal, DOI), in the NumPy docstring format.
- A parameter value or a stated result taken from a paper cites that paper in
  a comment next to the value.
- The local `test-map.md` keeps, for each test, both its catalogue entry and
  the primary sources, so the origin of every problem can be shown on request.
- If no primary source exists for a model (it appears only in the catalogue),
  the test is not added to the repository; it is listed in `test-map.md` with
  the reason and counted in Evidence.
- A paper by an author of the catalogue is cited as any other paper; the
  catalogue itself is never cited. So the terms in `.git/info/forbidden-terms`
  name the catalogue (title, directory, author with year), not its authors,
  and the leak check also runs on the `References` sections.
- Check: every module in `reference_models/` has a `References` section with at
  least one DOI or a full citation; review in the PR.

## Specs

### S1 — Triage of every catalogue entry

The catalogue has 89 problems with a stated result. Each one gets one class,
recorded in the local `test-map.md` and counted (without names) in Evidence:

| Class | Meaning | Test |
|---|---|---|
| E (exact) | A discrete-time model: commol runs the same equations | The stated result holds to a relative tolerance of 1e-9 |
| C (continuous) | An ODE model: commol runs it with the ODE engine (D1) | The stated result holds to a relative tolerance of 1e-6 with `TIGHT` solver settings (S3) |
| N (not a model) | Pure linear algebra with no model behind it | No test; counted only (D3) |
| G (gap) | commol cannot express the model, or computes it wrong | `xfail(strict=True)` with the finding id (D2) |

### S2 — Model families to cover

Grouped by the commol feature they exercise, not by their place in the source:

- Discrete-time SIR and staged-treatment models (rates as `1 - exp(...)`
  probabilities): final-size relation, limits of the sequences.
- SIR with births and deaths: threshold, endemic equilibrium, herd-immunity
  fraction; pulse vaccination (`impulse_every`) and its periodic orbit.
- Two competing strains: coexistence and exclusion conditions.
- Logistic population growth with an epidemic.
- Direct plus environmental (indirect) transmission; quarantine and isolation.
- Heterogeneous mixing and co-infection of two diseases: reproduction numbers.
- Vector–host models (different host and vector totals), one and two patches.
- Latent-stage models with one and two strains; partial progression.
- Sub-exponential growth and movement restrictions between patches.
- Age classes (stratifications as age groups). Continuous age of infection
  is an integral equation: class G unless the source gives a staged
  approximation.
- Three-patch spatial model with movement.
- Scalar ODE examples (exponential growth and decay, separable equations,
  stability of equilibria) and planar linear systems.

### S3 — The oracle is independent of commol

- Fix: the expected value comes from the stated result, computed in the test
  with `math`/`numpy` (closed form, root of the final-size equation, eigenvalue
  sign), never from another commol call.
- A stated quantity that commol does not output (a reproduction number, a
  threshold) is checked through what it predicts about a run: the epidemic
  invades just above the threshold and dies out just below it, the early
  growth rate equals `(R0 − 1)·γ`, or the run reaches the equilibrium the
  formula gives.
- An asymptotic result (equilibrium, limit of a sequence, periodic orbit) is
  read after a horizon long enough that the distance to the limit is below the
  tolerance: the test states the horizon, and doubling it changes the value
  by less than the tolerance.
- A test is self-contained: it never reads the catalogue, which CI does not
  have.
- Check: review; no test compares commol with commol.

### S4 — Findings are recorded, not fixed here

- Fix (D2): a G problem gets a finding F-n in Evidence with the smallest model
  that shows it, the observed and the expected result, and the suspected code
  (file and function). The test is `xfail(strict=True, reason="F-n: …")`, so a
  later fix makes it pass and forces the marker out.
- Check: every `xfail` reason names a finding that exists in Evidence.

### S5 — Documentation

Follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done):
the "Tests" section of `docs/codebase/conventions.md` gains a row for
`reference_models/` (the one package that does not follow a guide page: it
checks stated results across features), the attribution rule and how to add a
test; `docs/codebase/repository-map.md` lists `reference_models/`. No changelog
entry: nothing visible to users changes.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | How to compare an ODE result with a discrete engine? | (first draft) Scale every rate by a step `h` and check convergence when `h` halves. | The user chose a Rust ODE engine instead, delivered as its own ODD and PR before this one. The draft said an ODE solver goes against the golden rule; that was wrong: the rule forbids a Python re-implementation, not a new Rust engine. `ModelTypes::DifferentialEquations` already existed and the difference engine rejected it (`docs/codebase/mental-model.md` L24, before PR #101). | approved (L4): ODE engine, own ODD (`ode-engine.md`) |
| D2 | S4 | What to do when a model fails? | `xfail(strict=True)` + finding; fixes go to a later ODD, as the documentation overhaul did for code findings. | Fix in the same PR: unbounded scope, and an engine change hides in a test PR. `skip`: a later fix goes unnoticed. Delete the test: the gap is lost. | approved (L5) |
| D3 | S1 | Test the pure linear-algebra examples? | No: class N, counted only. | commol has no matrix API to test; such a test would check `numpy`, not commol. Where a matrix decides stability of a model, the model test (class C) already checks the stability by simulation. | approved (L6): class N, recorded in `test-map.md` and counted in Evidence |
| D4 | S2 | Test names? | Describe the model and the checked result (`test_pulse_vaccination_reaches_periodic_orbit`); the local `test-map.md` links each test to its entry. | Opaque ids (`test_ref_017`): safe, but a failing test says nothing to the reader. Names with the source's numbering: break the confidentiality rule. | approved (L7): one module per model family, one test per checked result |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1 | Triage of the 89 entries into E/C/N/G in the local `test-map.md`, with the primary sources of each model (Attribution); counts in Evidence | direct | `f2c4e24` |
| T2 | S2, S3 | `reference_models/` helpers on top of `tests/support/` (which already has `build_ode` and `TIGHT`): final-size root finder, a builder from absolute initial values, a threshold check (S3) | direct | `f2c4e24` |
| T3 | S2, S3, S4 | Class E tests | direct | `f2c4e24` |
| T4 | S2, S3, S4 | Class C tests, one module per model family | direct | `f2c4e24` |
| T5 | S4, S5 | Findings in Evidence; docs; leak check; merge | direct | `f2c4e24` |

## Delivery

- Strategy: one squash commit on `develop` from `test/reference-models`
  (`test(reference): …`). Delivered as `f2c4e24`, squash-merged locally
  without a PR at the user's request (L12).
- Order: T1 → T5. Both prerequisites are merged (PR #100, PR #101).
- Version impact: none (tests and contributor docs only).
- Leak check (Confidentiality) before every commit and before the PR.

## Evidence

Measured 2026-10-13 on `test/reference-models`.

### Triage (T1)

Every entry has a row in the local `test-map.md`, with its class, its test and
its primary sources.

| Class | Entries | With a test in the repository |
|---|---:|---:|
| E | 13 | 13 |
| C | 58 | 53; 5 not added (no primary source: pure linear systems and an oscillator with no model behind them) |
| N | 9 | 0 (8 pure linear algebra, 1 interpolation formula for census data) |
| G | 9 | 2 (F1, `xfail`); 7 (F3) without a test |
| Total | 89 | 68 |

Policy applied in the triage:

- A result stated for an arbitrary stage distribution is tested on an instance
  that commol states exactly when the source itself gives a staged or gamma
  version: a bounded discrete distribution as a chain of one-step stages, a
  gamma distribution of integer shape as a chain of exponential stages. With
  no staged version in the source, the entry is G (F3), as S2 says for
  continuous age.
- 16 entries have a slip in the stated result or in the printed equations (a
  dropped factor, a swapped index, a missing coefficient). The test checks the
  result that follows from the model and its primary source; `test-map.md`
  records each slip.

### Tests (T2–T4)

- 23 modules in `py-commol/tests/reference_models/`, 63 test functions: 88 runs
  pass and 4 are `xfail` (2 for F1, 2 for F2). The whole suite: 895 passed,
  4 xfailed.
- Helpers in `tests/support/`: `flows.py` (`Flow`, `flow_model` from absolute
  initial values, `increments` for the right-hand side at one state, the
  `FINE` solver settings) and `oracles.py` (`bisect`, `final_size`,
  `growth_ratio`, `assert_threshold`, `settled`); `assertions.py` gains
  `ODE_REL` and compares mappings of single values.
- Every module has a `References` section with at least one full citation; the
  37 distinct DOIs cited were checked against Crossref.

### Findings

| ID | Smallest model | Observed | Expected | Suspected code | Test |
|---|---|---|---|---|---|
| F1 | One bin `Y` with `y0 = 0.5` (or `0`) and the flow `Y * Y` out of the model | `ValidationError`: `population_size` must be an integer (and positive) | The model runs; an initial state is a list of values, not shares of an integer count | `InitialConditions.population_size` (`py-commol/commol/context/initial_conditions.py`), `u64` in Rust (`commol-model` `initial_state`) | `test_quadratic_decay.py` (2 `xfail`) |
| F2 | A rate that names a parameter or bin `e1`, `E2` or `c1star` | `ValidationError`: "scientific notation without base" or "digits followed by letters" | An identifier is accepted wherever it is valid; only a malformed number is rejected | The numeric-format patterns of `utils/security.py` (`(?<!\w)e\d+` with `IGNORECASE`, `\d+[a-df-zA-DF-Z]+`) scan identifiers too | `test_discrete_epidemic.py::test_staged_treatment_with_ids_that_hold_digits` (2 `xfail`) |
| F3 | SIS with a fixed infectious period, or any model with a density in infection age | Cannot be stated: no delays, integral or partial differential equations | A way to state an arbitrary stage duration or an age density | Model kind, not a code defect | None: a strict `xfail` needs a call that a fix makes pass, and no API states such a model yet |

F2 does not come from a catalogue entry: it appeared when the chains of stages
were named `E1`, `E2`, … The tests name those stages `L1`, `L2`, … meanwhile.

## Log

- L1 (user, 2026-10-13): asked for an ODD to add the problems of a private
  reference catalogue as tests (and to refactor the test suite, now in
  `test-suite-refactor.md`), with no trace of the source in git or GitHub.
- L2: catalogue directory added to `.git/info/exclude`; `.git/info/forbidden-terms`
  created; leak check on every tracked file and every commit message prints
  nothing.
- L3 (user, 2026-10-13): refactor first, as its own PR (D1 of
  `test-suite-refactor.md`).
- L4 (user, 2026-10-13): D1 → implement the ODE in commol, in its own ODD and
  PR before this one.
- L5 (user, 2026-10-13): D2 → "xfail estricto + hallazgo".
- L6 (user, 2026-10-13): D3 → "No probarlos, solo registrarlos como clase N"
  (each one has a row in the local `test-map.md`; Evidence gives only the
  totals per class).
- L7 (user, 2026-10-13): D4 → "Modelo + resultado" (e.g.
  `reference_models/test_pulse_vaccination.py::test_reaches_periodic_orbit`).
- L8 (user, 2026-10-13): asked for a rule that cites the origin of the
  problems, to prevent a plagiarism claim. Citing the catalogue conflicts with
  the confidentiality rule; chose "Citar las fuentes primarias": tests cite the
  original papers, the catalogue stays in the local `test-map.md`.
- L9 (user, 2026-10-13): split from the test-suite refactor into its own ODD;
  specs, decisions, tasks and log entries renumbered (old S6–S10 → S1–S5,
  D3–D6 → D1–D4, T6–T10 → T1–T5).
- L10 (agent, 2026-10-13): reviewed after PR #101. Added "Current state";
  class C tolerance fixed at 1e-6 with `TIGHT`; S3 gained the rules for
  quantities commol does not output, for asymptotic results and for
  self-contained tests; T2 reuses `build_ode` and `TIGHT`; the integer
  `population_size` and continuous age of infection are named as candidate
  gaps; Delivery no longer waits.
- L11 (agent, 2026-10-13): implemented T1–T5. Decisions taken during the
  work, for review in the PR: the arbitrary-distribution policy of Evidence;
  F3 without a test (S4 asks for an `xfail`, but no API exists to call);
  textbook examples with no model and no primary source not added (L8);
  `flow_model` solves with `FINE` because `TIGHT` (`atol = 1e-8`) misses
  `1e-6` on values below one; no `scipy` oracle (`increments` reads the
  right-hand side through a difference step, roots use `bisect`).
- L12 (user, 2026-10-13): "archiva, haz merge, push y borra la rama".
  Squash-merged locally into `develop` as `f2c4e24` (no PR), pushed, branch
  deleted, ODD archived.
