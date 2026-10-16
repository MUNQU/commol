# Task documents (ODDs)

An ODD is the specification of one piece of work: its goal, current state,
specs, the decisions of the user with their alternatives, its tasks, the
evidence of its delivery and a log. One ODD is one branch and one pull request
to `develop`. The file is the specification; GitHub holds the state.

The name is unrelated to the `"odd"` layout of `Model.describe`, which follows
the ODD protocol for describing models (Grimm et al. 2006, 2010, 2020).

| What | Where |
|---|---|
| The specification | `odd/<name>.md`. A file never moves, so links to it keep working. |
| Status, order and dependencies | One issue per ODD, label `odd`, with its blockers ("blocked by"). The issue links to the file and the file links to the issue. |
| Releases | One issue per release, label `release`, with the checklist of `docs/development/release.md`, blocked by the issues it ships; a milestone per release groups them. |
| Delivery | The pull request says `Closes #N`; when it merges into `develop`, `.github/workflows/close-issues.yaml` closes the issue. |
| Why the order | This page. |

## Read the state

```bash
gh issue list --label odd --state open                       # pending ODDs
gh issue list --milestone 0.5.0                              # what a release ships
gh api repos/MUNQU/commol/issues/131/dependencies/blocked_by --jq '.[].number'
```

An issue is ready when every issue that blocks it is closed.

## Lifecycle of an ODD

1. Write `odd/<name>.md`, and open its issue: title the first line of the file,
   body a link to the file and its goal, label `odd` (and `breaking` when it
   breaks the public API), the milestone of its release, and its blockers.
   Add the line `Issue: [#N](…)` under the title of the file.
2. Settle its decisions when its implementation starts (see below).
3. Work on a branch from `develop`. Fill the Evidence and the Log of the file
   in the same branch.
4. Open the pull request with `Closes #N`. The merge closes the issue; the
   file stays where it is.

## Why this order

- **`inference-foundation` (#125) first of the program.** It splits the
  calibration problem, removes the ensemble, and adds the sample, the
  predictive, the scores and the entry point that the thirteen later ODDs
  use. No release between it and `resampling-uncertainty` (#127; its D2), so
  no release lacks an uncertainty method.
- **Identifiability (#126) and resampling (#127) next.** `identifiability`
  gives the sensitivity port that the Gibbs scale, profiles, Laplace and
  gradients read; `resampling-uncertainty` gives the blocks and the replicate
  runner that conformal prediction, the Gibbs scale, profiles and the sandwich
  use. Both need no noise model, the user's first concern
  (`inference-foundation` L3).
- **Methods without a noise model before methods with one.** Conformal
  prediction (#129) and the SMC core (#130: Gibbs posterior, ABC) come before
  `observation-models` (#131), which then adds the likelihood target to the
  same core.
- **Likelihood methods in dependency order.** Profiles (#132) and local
  approximations (#133) need the likelihood; MCMC (#134) reuses the SMC
  kernels and the unconstrained transform of local approximations; gradients
  (#135) need the MCMC runner for NUTS and PSIS for ADVI; state-space methods
  (#136) need the likelihood, the chains and the stochastic engines.
- **Analysis and comparison last of the program.** Sensitivity analysis
  (#137) needs only the distributions and the predictive; model comparison
  (#138) needs the evidence, pointwise likelihood, PSIS and kernels; emulation
  (#139) needs the quantities of interest and the chains.
- **One release per break.** The program breaks the API once more, with the
  removal of the ensemble (#125). Release 0.5.0 (#128) ships that break with
  the first uncertainty methods (#125–#127); release 0.6.0 (#140) ships the
  rest of the program (#129–#139). 0.4.0 shipped the foundation and the
  stochastic engines before the program started, because `inference-foundation`
  D2 allows no release from #125 until #127 merges. Cost: one changelog and
  migration guide per release; commol is alpha, so a release costs little.
- **The benchmark (#141) after 0.6.0.** It runs on the release that closes
  the program, as the user asked (`library-benchmark` L7).
- **Benchmark before the other modelling ODDs.** The user kept the benchmark
  before new kinds of modelling (Markov chains, networks, agents); only the
  stochastic engines moved ahead of it, because the inference program needs
  them.
- **Markov chains (#142), then integral outputs (#143).** Most of a chain fits
  the aggregate kind already; its builder, matrix and analysis (S1–S4) need
  only the foundation. Rewards are the main use of Markov chains, so integral
  outputs follow with no gap (Markov D4), before networks and agents, so those
  kinds add integrals of their counts with the declaration already in place.
- **Networks (#144) before agents (#145).** Both deliver one structure,
  `Agents` (`agent-based-models.md` D6). The network ODD delivers it first,
  with a graph and no attributes, and the variant `Network`: the first real
  second kind, which tests the generic layers with less to build. The agent
  ODD then adds attributes, `Complete`, `Groups` and Python rules on the same
  structure and engines.

## When decisions are made

The decisions of an ODD of the uncertainty program (milestones 0.5.0 and
0.6.0) are settled when its implementation starts, not before (user,
2026-10-15). A decision that changes what an earlier ODD built, or the
architecture of `inference-foundation.md` (its D1, D3, D4, D7), updates the
later ODDs that depend on it in the same PR, with a line in their Log.

## Not in parallel

The ODDs run one after the other, each from `develop` after the previous PR
merges: each builds on the interfaces the one before adds. Inside the
program, `identifiability` (#126) and `resampling-uncertainty` (#127), and
`sensitivity-analysis` (#137) and `state-space-inference` (#136), touch
different modules and could run in parallel on two branches; the plan keeps
one order so each ODD is written against the merged code of the one before.
