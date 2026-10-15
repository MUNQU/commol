# Feature: Benchmark against other compartment-model libraries

Goal: measure, on one stratified model taken from the group's applied work,
what commol does better or worse than the closest libraries, so that the work
after this ODD is chosen by evidence. The result also serves as material for a
software paper.
Branch: `chore/library-benchmark` from `develop`, after the release that
follows the uncertainty and identifiability program
(`odd/tasks/inference-foundation.md` to `odd/tasks/emulation-discrepancy.md`;
`odd/execution-order.md`, L7).
Engram mirror: topic `odd/library-benchmark/tasks` (project `commol`).
Runners: `pytest` (from `py-commol/`), the benchmark command of S5,
`mkdocs build --strict` (from the root).

## Context (2026-10-13)

The user's goal for commol is both a tool for the group's projects and a public
library for other researchers. The closest libraries:

| Library | Language | Model | Calibration | Notes |
|---|---|---|---|---|
| epydemix | Python | Stochastic simulation; age structure from population data and contact matrices | Approximate Bayesian computation (rejection, ABC-SMC) | GPL-3.0; PLOS Comput Biol (2025) |
| odin2 + dust2 + monty | R (DSL compiled to C++) | Difference equations and ODE, deterministic or stochastic | Particle filter, MCMC | MRC-IDE |

Commol today: deterministic only; optimization (Nelder-Mead, PSO) and ensemble
calibration, no likelihood-based Bayesian inference. When this ODD runs (L7),
commol will have stochastic engines (`stochastic-compartmental.md`), ABC-SMC
(`smc-inference.md`), particle filters and PMMH (`state-space-inference.md`),
and no ensemble; D6 revisits D3 and D4 for that state.

## Specs

### S1 — One model, written well in each library

- Fix: the model of D2 is written in each library of D1 by its documented,
  recommended API (no deliberately slow code), with the versions pinned and
  recorded.
- Check: each implementation reproduces the same reference run (S3) within the
  tolerance of S3; a reviewer who knows each library can read its file.

### S2 — Fair comparison of a deterministic and a stochastic engine (D3)

- Fix: commol's deterministic run is compared with the mean of many stochastic
  runs of the other library, at a population size where the mean is close to
  the deterministic solution. The number of runs and the gap are reported.
- Check: the gap between the stochastic mean and commol falls when the
  population grows (shown for three sizes).
- odin2 runs the model deterministically too (difference equations, and ODE
  once commol has its ODE engine), so commol is compared with it run against
  run, without the stochastic mean. Its stochastic mode is reported only as
  extra information.

### S3 — Metrics

- Simulation: wall time per run (median of repeats after warm-up), memory,
  scaling with the number of compartments (three stratification sizes).
- Model definition: lines of code, and what one library cannot express that
  the other can.
- Calibration (D4): synthetic data from known parameters; time to a fit and
  parameter recovery error.
- Check: every number comes from the benchmark command (S5), with machine,
  versions and seeds recorded next to it.

### S4 — Data stays private

- Fix: the benchmark uses the structure of the project model (bins,
  stratifications, rates) and synthetic data generated from known
  parameters, never the project's data.
- Check: no file in the benchmark reads data from outside the benchmark
  directory.
- The structure comes from the group's flu vaccination project (sibling
  directory, not part of this repository). Before the PR the user confirms
  that the structure may be public; if not, the benchmark uses a renamed,
  equivalent structure (same sizes and rate forms).

### S5 — Where it lives and how it runs (D5)

- Fix (D5): `benchmarks/` at the repository root, outside the package and
  the wheel. Python dependencies (epydemix, GPL-3.0) in their own dependency
  group; R dependencies (odin2, dust2, monty) pinned with `renv`. The scripts
  import epydemix but nothing of it ships with commol, so the package stays
  MIT. The summary goes to a page under `docs/about/`.
- Check: one command runs the whole benchmark and writes a results table;
  `uv build` (or `maturin build`) produces a wheel with no file from
  `benchmarks/`.

### S6 — Decision record

- Fix: the results go to Evidence with a short reading: where commol wins,
  where it loses, and which next ODD that suggests (for example stochastic
  models or Bayesian inference).
- Check: the reading names the measurement behind each claim.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S1 | Which libraries? | epydemix only. | It is the direct competitor (Python, stratified models, calibration) for the users commol wants. Adding odin2: a stronger engine reference, but it needs R and its DSL, which doubles the work, and Python users would not switch to R. Adding Julia tools: a different audience. | approved (L2): epydemix and odin2 |
| D2 | S1, S4 | Which model? | The most stratified model of the group's current projects (structure only, synthetic data). | A textbook model (SIR by age): easy, but it shows nothing the user needs. Several models: more evidence, but more work before any decision. | approved (L3): the flu vaccination model (age × vaccination × comorbidity, 12 transitions, hospital-admission accumulator), scaled with more age groups |
| D3 | S2 | How to compare deterministic with stochastic? | Commol's run against the stochastic mean at large population, with the gap and the number of runs reported. | Compare single runs: one stochastic run is noise. Ask commol for a stochastic mode: does not exist. Skip accuracy, time only: a fast wrong answer looks good. | approved (L4) |
| D4 | S3 | How to compare calibration? | Parameter recovery on synthetic data, and wall time to reach it; the posterior of ABC is reported, not scored. | Score the posterior against commol's ensemble: they answer different questions (Bayesian posterior against an ensemble of good fits). Skip calibration: it is half of what commol offers. | approved (L5); the posteriors of ABC (epydemix) and MCMC (odin2) are shown, not scored |
| D5 | S5 | Where does the benchmark live? | A `benchmarks/` directory in this repository, outside the package, with its own dependency group (not shipped in the wheel); results summarized in `docs/about/`. | A separate repository: cleaner for the GPL-3.0 dependency, but it drifts from commol's API. Inside `tests/`: it would run in CI, which is slow and depends on another library. | approved (L6) |
| D6 | S2, S3 | With stochastic engines and Bayesian inference in commol (L7), how do D3 and D4 change? | Compare like with like: commol's stochastic engine against epydemix's and odin2's stochastic runs; commol's ABC-SMC against epydemix's ABC-SMC and commol's PMMH against monty's, on the same synthetic data, scoring each posterior by coverage of the true parameters and CRPS (`commol.scores`), not only showing it. Keep D3's deterministic comparison as one row. | Keep D3 and D4 as approved: compares commol's deterministic run with stochastic means and leaves posteriors unscored, when like-for-like runs exist. Drop the deterministic row: loses the run-against-run comparison with odin2's deterministic mode that S2 keeps. | proposed |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1, S4 | Model structure from D2 and synthetic data generator | direct | |
| T2 | S1 | Commol implementation | direct | |
| T3 | S1, S2 | epydemix implementation; agreement of the stochastic mean (S2) | direct | |
| T3b | S1 | odin2 implementation (R toolchain pinned with `renv`); run-against-run agreement in both engines | direct | |
| T4 | S3, S5 | Benchmark command: simulation, scaling, calibration; results table | direct | |
| T5 | S6 | Run on one machine; Evidence and reading; docs summary | direct | |

## Delivery

- Strategy: one PR to `develop`, squash-merged (`chore(benchmark): …`).
- Order: after the release that follows `emulation-discrepancy.md`
  (`odd/execution-order.md`). T1 → T2 → T3 → T3b → T4 → T5.
- Version impact: none (nothing in the package changes).
- The leak check of `odd/archive/tasks/reference-model-tests.md` (Confidentiality) also runs
  here.

## Evidence

(Filled during implementation.)

## Log

- L1 (user, 2026-10-13): asked whether commol adds value; goal is both a
  group tool and a public library. Asked for this ODD, to run after the
  planned ODDs.
- L2 (user, 2026-10-13): D1 → "epydemix y odin2".
- L3 (user, 2026-10-13): D2 → "El modelo de gripe".
- L4 (user, 2026-10-13): D3 → "Media estocástica, población grande".
- L5 (user, 2026-10-13): D4 → "Recuperar parámetros conocidos".
- L6 (user, 2026-10-13): D5 → "benchmarks/ en este repo".
- L7 (user, 2026-10-15): asked to run the benchmark after the uncertainty
  and identifiability program (`odd/tasks/inference-foundation.md` L5). That
  program needs the stochastic engine for its state-space methods, so the
  benchmark also follows `model-kind-foundation.md` and
  `stochastic-compartmental.md`; the other modelling ODDs (Markov chains,
  integral outputs, networks, agents) stay after it. D6 proposed.
