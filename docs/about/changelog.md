# Changelog

[Back to Home](../index.md)

All notable changes to Commol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `GibbsResult` and `ConformalResult` give `block_length` and
  `block_length_estimates`, as the bootstraps, the jackknife and the sandwich
  do: the length a method ran with, and with `block_length="auto"` the
  estimate of each series, by label. `GibbsResult` gives them for
  `InformationMatchingEta` and `BootstrapCoverageEta`, and `None` for a
  number `η`; `ConformalResult` for `JackknifePlus` and `EnbPI`, and `None`
  for the variants without blocks. See
  [Sequential Monte Carlo](../guide/uncertainty/smc.md#what-means) and
  [Conformal Prediction Intervals](../guide/uncertainty/conformal.md#results-plots-and-reports).

### Changed

- A formula has at most 100 000 nodes (numbers, variables, operators and
  calls), next to its limit of 100 levels of nesting: a longer one raises
  `ValueError` when its object is built, before it compiles. A message of
  the parser or of the compiler quotes the formula cut after its first 80
  characters. See
  [Limits of the parser](../guide/mathematical-expressions.md#limits-of-the-parser).
- A rate gives the same description with or without spaces: the description
  and its LaTeX are written from the tree that the parser of the engine
  reads, so `k1*A*B/total` and `k1 * A * B / total` give one description. A
  formula shows one space around each operator, only the parentheses that
  its grouping needs, `^` for a power and each number as its value (`1e-5`
  shows as `1e-05`, `2.0` as `2`). In LaTeX, a power raises its exponent
  (`A^{2}`), `%` is `\bmod` and a comparison has its symbol (`\leq`). See
  [The equations](../guide/describing-models.md#the-equations).
- The unit check reads the tree of the engine: it reads `!` in a condition
  and a variable named like a word of Python, such as `lambda`, and a formula
  outside the language gives the message of the parser.

### Removed

- **Breaking:** `commol.utils.security` is removed, with `SecurityConfig`,
  `SecurityError`, `validate_expression_security` and the `config=` argument
  of `AxisPattern.combine` and `AxisPattern.from_formula`. The parser of the
  engine is the only check of a formula: a rate, a formula parameter, a
  calibration constraint or an axis pattern that it rejects raises
  `ValueError` with its message, and no setting changes its limits. Remove
  the `config=` argument, and catch `ValueError` instead of `SecurityError`.
  A formula of the language that the old text guards rejected is now valid:
  a name with letters outside ASCII (`α * β`), a name such as `__import__`,
  `os` or `my_key`, a number such as `1e400`, `nan` as the name of a
  variable, and a formula of more than 1000 nodes, 10 000 characters or 10
  levels of nesting, within the limits of the parser. The
  build still rejects a variable that the model does not define. See
  [Limits of the parser](../guide/mathematical-expressions.md#limits-of-the-parser).

### Fixed

- The flow of a proportional rate is the whole rate times its source, in the
  description and in the unit check: a rate `k1 + k2` from `B` is
  `(k1 + k2) * B`; before, both read `k1 + k2 * B`, and the unit check of
  such a model failed. In LaTeX, a division followed by a product is a
  fraction times the product: the flow of a proportional rate `a / b` from
  `B` is `\frac{a}{b} \cdot B`, no longer `\frac{a}{b \cdot B}`.

### Documentation

- [Design principles](../codebase/design-principles.md): the structure,
  the rules of each unit and of the interface, and the proof that every part
  of the code follows. «Before you change code» of `AGENTS.md` links them.
  `tests/suite/test_architecture.py` fails when a crate of `crates/`
  depends on PyO3 or on the bridge.
- A numbered list of the documentation keeps its first number, and a line
  inside a paragraph that starts with a number and a period, such as `512.`,
  stays in its paragraph: the site reads lists with `sane_lists`.

## [0.10.2] - 2026-10-24

This release makes the engine the one source of every rule: the range of
each setting, the values of a model, the windows, the checks of a model and
a calibration, and the language of formulas, which commol now reads with a
parser of its own. A value that was valid stays valid, and an invalid one
still raises `ValueError` when its object is built, but the messages are
the engine's. Read the changes before you upgrade: seeded results change
value, as every draw comes from one seed scheme in which neighbouring seeds
share no replicate; a chain of powers groups from the right, so
`2 ** 3 ** 2` is 512; `BootstrapCoverageEta` gives a slightly different η;
a stratification fraction outside 0 to 1 is rejected; and two results are
equal when their values, a NaN included, and their positions, seed and
model are equal, so a run is no longer equal to a plain `dict`. See
[Building models](../guide/building-models.md) and
[Uncertainty](../guide/uncertainty/index.md).

### Changed

- The engine holds the range rule of every setting: a count, a share, a
  length, a scale, a tolerance, a bound of an optimizer, an observation, an
  observation model, an axis pattern, a distribution, a quantity of interest
  and the setting of an uncertainty, identifiability or sensitivity method.
  A value outside its range still raises `ValueError` when the object is
  built, and the same values are valid; the message is the engine's, and
  names the setting and its range: `replicates is a whole number of at least
  1, got 0`, `sd is a finite number greater than 0, got 0`. A count gives
  one message for every value outside its range, a negative one included,
  also when a binding is called with no Python object in between, and for
  `Simulation.replicates` (`count is a whole number of at least 1, got 0`)
  and `Calibrator.run` (`starts is a whole number of at least 1, got 0`).
  A seed outside its range gives `seed is a whole number from 0 to
  18446744073709551615, got -1` in every function that takes one:
  `Simulation.run`, `Simulation.run_at`, `Simulation.replicates`,
  `CalibrationProblem`, `SensitivityAnalysis` and each binding.
- The engine holds the rule of every value of a model and of every window.
  An initial amount, an impulse and the points of a series fail when their
  object (`Aggregate`, `Transition`, `SeriesValue`) is built or an amount is
  updated, as before, and the same values are valid; the message is the one
  the model compiler gives, which names the bin, the compartment or the
  transition: `Bin 'B': initial amount is a finite number of at least 0, got
  -1`, `Transition 't': impulse period is a finite number greater than 0, got
  0`, `a series has one point per position, got two at 2`. The windows of
  `windowed_totals`, of reports, of figures and of `AxisUnits.window_start`,
  `window_end`, `window_index` and `window_ends` come from the engine, with
  its messages (`window is a finite number greater than 0, got 0`, `index is
  a whole number of at least 0, got -1`), and an observation or a quantity
  whose window starts before 0 gives `the window 4 that ends at 3 starts
  before 0; an end is at least its window`. A `tick_every` of a figure that
  is not a finite number greater than 0 raises `ValueError`.
- A calibration, a sensitivity analysis or any other method that sets a
  model parameter raises `ValueError` when it is built if the parameter is
  a formula or a series parameter. The engine computes such a parameter at
  each step, so a value set on it was lost and the method fitted nothing.
  The message is the engine's, for this case and for a parameter the model
  does not have: `Parameter 'k9' is not in the model; the parameters that
  can be set are k1, k2`. Python keeps no copy of this check.
- The four engines of the aggregate kind name themselves in their messages
  by the name of the engine: a position an engine does not report at gives
  `The DifferenceEquations engine reports only at integer steps; 2.5 is not
  one`, and an impulse or a series point between two steps names `the
  DifferenceEquations engine` or `the StochasticSteps engine`.
- `Predictive.at_observations` and every method that reads a predictive at
  other observations raise `ValueError` for a position the engine rejects,
  with the message of the engine, as a calibration does. The bands, the
  automatic block lengths and their estimates keep their values: one Rust
  function of each computes them for every method.
- The engine decides the practical verdicts of `Collinearity` and
  `SampleIdentifiability` and the two notes of a profile on its refits, and
  results read a position with the rule of an observation: the verdicts,
  reasons, notes and values stay. `windowed_totals` reads each window as a
  calibration reads a windowed observation, and it gives floats. A figure asked for a series that is neither an
  output nor a bin or accumulator raises `KeyError` with the message of
  `Simulation.outputs_for`, and a sum of outputs that the results do not hold
  gives one message, `Missing output series in results`, in a simulation, a
  report and a figure.
- The engine checks a model, a calibration and the engine settings, and
  Python repeats none of its checks, so each mistake gets the message of the
  engine. A parameter named after an engine variable, a bin, a compartment
  or a subgroup total names each conflict and asks to rename the parameter.
  An observation of an output the model does not have names the output, the
  observation and the outputs: `Invalid observed output name 'X' for
  observation 'series1' (available: A, B, C)`. A factor with a bound of 0 or
  below fails with `Factor parameter 'f' must have positive bounds (got
  min=0, max=2)`. `Collinearity` checks `max_subset_size` when it runs, with
  `max_subset_size must be from 2 to the number of parameters (3), got 9`.
  The needs of a stochastic engine (`replicates`, `seed` and
  `replicate_summary` of a calibration; `replicates` and `seed` of a
  sensitivity analysis) keep their messages, which the engine now gives.
- `ODESolver` and `StochasticEventsSettings` take their defaults from the
  engine and check themselves with the engine when they are built:
  `StochasticEventsSettings(epsilon=1.5)` gives `epsilon is strictly between
  0 and 1, got 1.5`. Every other setting strictly between 0 and 1
  (`target_cess`, `stay_probability`, `min_acceptance`, the `quantile` of
  `ABCRejection`, `correlation_threshold`) gives `<name> is strictly between
  0 and 1, got <value>`, `Quantile(1.5)` gives `A quantile level is between 0
  and 1, got 1.5`, and a Sobol analysis of the Rust API with fewer than 2
  draws says `draws`, the name of the setting.
- A negative count that reaches the engine, such as `Simulation.run(-1)`, the
  `replicates` of a `SensitivityAnalysis` or the `max_subset_size` of
  `Collinearity`, raises `ValueError` that names the setting,
  `num_steps is a whole number from 0, got -1`, in place of an
  `OverflowError` with no name. Every count of the bindings reads this way.
- The engine draws a seed that its user does not set, in one place, and the
  result gives the seed it ran with, as before: `results.seed`,
  `SimulationReplicates.seed`, `Calibrator.seed` and
  `SensitivityAnalysis.seed`. A calibrator with no seed logs the seed it drew
  when it is built, not when it first runs.
- Ruff checks that every signature of the package has annotations and that
  none is `typing.Any`; the raw input of a validator is a `JsonValue`. A
  keyword that
  `Quantile`, `Groups`, `Level` or `Increments` does not take raises
  `TypeError` that names it, such as `Quantile(0.5, unknown=1)`, in place of
  the `ValidationError` `Extra inputs are not permitted`. A `ParameterSample`
  whose `names`, `bounds` or `values` are not valid gets the pydantic message
  of that field before the engine checks the log weights.
- A calibration and every fit of a replicate of a bootstrap, the jackknife,
  a profile threshold or a conformal method run their starts with one rule:
  the engine chooses the best start, the first of lowest loss, and the seed
  gives the Latin hypercube and the seed of each start from two streams of
  its own. So a calibration with a seed, its structural rank and its fits of
  replicates give other values than in 0.10.1; a run stays reproducible
  from its seed on one version. When every start fails, the message lists
  the reason of each start, also in the message of a replicate that fails.
- A particle swarm draws its numbers with the generator of commol from the
  seed of its start, which a calibration always gives: a verbose calibration
  with a swarm is the calibration without `verbose`, and
  `Calibrator.run_with_history` is start 0 of `Calibrator.run`.
- The summaries of `PropagationResult`, of the scenarios and of
  `member_statistics`, the effective size of a propagation and the densities
  of observation models and priors keep their values: one Rust function
  computes each, and Python calls the summary as
  `commol_rs.inference.weighted_summary`. The error of a `Laplace` fit on its
  bound points to `ProfileLikelihood` and cites no issue.
- `BootstrapCoverageEta` takes `η` with the one quantile of commol, the one
  of `weighted_quantiles`, on the logs of the thresholds, so `η` changes
  once. It needs at least `1/(2(1 − c))` replicates times parameters, where
  it needed `1/(1 − c) − 1`.

### Fixed

- A fraction of a stratification outside 0 to 1 raises `ValueError` when its
  `Aggregate` is built, with the message of the engine: `Stratification 'g',
  category 'g1': fraction is a number from 0 to 1, got 1.5`. Before, only the
  sum was checked, so fractions such as `{"g1": 1.5, "g2": -0.5}` built a
  model whose compartment started at a negative amount. The sum of the
  fractions of a stratification, 1 within 1e-6, is checked at the same time
  (`Stratification 'g': the fractions of its categories sum to 1 within 1e-6,
  got 0.9`), where it was checked by `build`, and
  `update_stratification_fractions` and the model compiler give the same
  messages.
- Two results of commol with the same values are equal, also when they hold
  a NaN: a NaN equals a NaN, whatever the objects that hold it. Before, `==`
  compared the identity of the items of a list first, so two runs with one
  seed gave results that were not equal when one held a NaN, such as a PRCC
  coefficient of an input that the other inputs explain entirely. A frozen
  result has a hash that agrees with `==`, also when it holds a list, a dict
  or a NaN. `ParameterSample` compares as every other result. A result no
  longer compares its private attributes
  ([Uncertainty](../guide/uncertainty/index.md#comparing-results)).
- `==` of `SimulationResults` and `SimulationReplicates` now also compares
  their positions, seed and model, and a NaN equals a NaN. Before, it read
  only the values, so two runs at other positions or with other seeds and
  the same values were equal, and two runs with one seed that held a NaN
  were not. A plain `dict` or `list` with the same values is no longer equal
  to a result: compare `dict(results)` to read the values only
  ([Running Simulations](../guide/simulations.md#results)).
- A profile reads a parameter with one finite bound, such as `[0, +∞)`, on
  its bound within `1e-8` of `max(1, |bound|)`, the rule of `Laplace`.
  Before, every value of such a parameter was on its bound, so a profile
  never saw the ridge leave the box through it, and the structure of a flat
  profile gave `undetermined="on_bound"` with that parameter in place of a
  reading.

- A window whose start rounds to just below 0, such as a window of
  `1000 + 1e-7` that ends at 1000, starts at 0 in an observation, a
  calibration and a quantity, as `windowed_totals` reads it. Before, a
  calibration rejected it while the same window that ends at 2000 was
  accepted. One rule of the engine gives the start of a window.
- Two neighbouring seeds share no replicate: before, replicate `i + 1` of
  `simulation.replicates(n, seed=42)` was replicate `i` of `seed=43`, and so
  were the starts and the swarms of a calibration with neighbouring seeds.
  Two methods that take one seed no longer draw from one sequence either:
  every random method draws from a stream of its own of the seed, and with
  one seed, replicate `i + 1` of a bootstrap no longer has the generator of
  particle `i` of an SMC sampler, nor the draws of `Laplace` the design of a
  sensitivity analysis. Every method draws with the samplers of the one
  generator of commol. So with one seed, the draws of the random methods and
  the replicates of a stochastic engine differ from 0.10.1; a run stays
  reproducible from its seed on one version.
- One check of the engine defines the language of formulas, the one
  [Mathematical expressions](../guide/mathematical-expressions.md)
  documents, for rates, formula parameters, calibration constraints and the
  unit check. A formula outside it fails when it is added, with a message
  that names the formula and its first part outside the language. Before,
  `k1 * (step > 30)` was accepted as a formula parameter, and `step > 1` as
  a rate became a parameter named `step > 1` that failed only when the model
  ran. A formula that calls a function the guide does not list, such as
  `erf`, `fma` or `trunc`, fails as a rate always did; a formula parameter
  accepted it before.
- A rate or a formula parameter that reads a variable the model does not
  define fails when the model is built. Before, only a transition rate was
  checked, and a formula parameter failed when the model ran. A category id
  is not a variable: before, a rate that read one passed the build and
  failed when the model ran.
- An error of a rate when the model runs names its transition, not the
  indices of its compartments.
- `SecurityConfig.max_complexity_score` limits the number of nodes of the
  tree of a formula (numbers, variables, operators and calls), as the guide
  documents. Before, it limited nothing.
- The unit check reads `^` as a power, as the engine does, and `cbrt` takes
  the cube root of a unit.
- A formula with a long chain of terms, such as a sum of 20 000 terms, no
  longer stops the Python process with a segmentation fault. The engine reads
  formulas with a parser of its own: a sum or a product of any length is one
  level, and a formula that nests deeper than 100 levels of parentheses,
  calls, signs and powers fails with a message that names the limit. Before,
  the parse used one step of the stack for each term, and the process
  crashed near 20 000 terms, or near 2 000 on a thread of the parallel work.
- A chain of powers groups from the right, as in Python: `2 ** 3 ** 2` is
  512. Before, it was 64, `(2 ** 3) ** 2`.

- `ParticleSwarmConfig.inertia("constant")` and
  `.acceleration("constant")` take the weight and the coefficients of
  canonical particle swarm, `1/(2 ln 2)` and `0.5 + ln 2`, which a swarm with
  no inertia or acceleration strategy uses. Before, their defaults were the
  rounded `0.721` and `1.193`, so a swarm that named the constant strategy
  moved a little differently from one that did not.
- Three messages of a calibration with observation models no longer hold a
  run of spaces in the middle of a sentence.

## [0.10.1] - 2026-10-23

This release ships the fixes of the audit at 0.9.0 that change no API. In the
`StochasticEvents` engine, a parameter or a seed changed between two runs acts
from the position where the first run stopped. The documentation reads as one
account of the library and names no application field, and commol no longer
installs `cffi`. CI runs with read rights and pinned tools, and the test suite
checks its own layout. Nothing breaks: code that runs on 0.10.0 runs on
0.10.1. See [Stochastic engines](../guide/stochastic.md) and the
[Quickstart](../getting-started/quickstart.md).

### Removed

- `cffi` is no longer a dependency of commol: no code of commol uses it, so
  `pip install commol` no longer installs it. Code that imports `cffi`
  installs it itself.

### Fixed

- In the `StochasticEvents` engine, a parameter or a seed changed between
  two runs acts from the position where the first run stopped. The pending
  wait is drawn again from there, and a pending leap keeps only its events up
  to there, by binomial thinning. Before, the new rate acted back from the
  last event, and copies of one state with new seeds shared their next event.
  In Python this changes only `simulation.engine.set_seed` between two calls
  of `simulation.engine.step()`; `Simulation.run`, replicates and calibration
  reset the engine after the seed, so their results do not change.

- The documentation and the docstrings name no application field, as
  [Conventions](../codebase/conventions.md) asks: the dispersion of
  `NegativeBinomialObservations` is "the dispersion `k` (`size` in R)", and
  bins of different units are "a count and a concentration". A test checks
  the words of the user pages and of the source of `commol`.

### Documentation

- The documentation reads as one account of the library: the home page, the
  quickstart and the examples show one process run by several engines, then
  calibration, uncertainty, the comparison of runs and parallel work, and the
  guide follows that order. Each rule lives in one page and the others link
  to it; a test checks that every page is in the navigation and that no text
  repeats across pages. The API reference names every name of
  `commol.__all__` and the module of each name that the root does not export.
  The maintainer pages name the paths, symbols and tests of the code, and the
  design note of per-compartment initial conditions is part of
  [Initial conditions](../codebase/decisions/initial-conditions.md).

## [0.10.0] - 2026-10-23

This release makes equivalent things the same across commol, after an audit
of the code, the repository and the documentation at 0.9.0. A setting that
means the same thing has one name, every method takes its seed from its
source and fits from its starts, equivalent results hold their failed fits in
one place, equivalent methods raise one error type, a replicate summary takes
the one quantile of commol, and commol adds no log handler. Before you
upgrade, read each **Breaking** entry: most code changes only the names of
settings and the type of an exception. See
[Uncertainty](../guide/uncertainty/index.md) and
[Log messages](../getting-started/quickstart.md#log-messages).

### Added

- `Laplace`, `Sandwich`, `InformationMatchingEta` and `BootstrapCoverageEta`
  take `starts`, at least 1 with the default 1: their fit is the best of
  `starts` starts, `Calibrator.run(starts)`, as in every bootstrap, jackknife,
  conformal method and profile. Code that does not pass it runs as before.
  See [Local Approximations](../guide/uncertainty/local.md#laplace-approximation-and-the-delta-method)
  and [What `η` means](../guide/uncertainty/smc.md#what-means).

### Changed

- **Breaking:** every uncertainty and identifiability method takes its random
  numbers from the seed of its source, `Calibrator.seed` (`problem.seed`) or
  `SensitivityAnalysis.seed`, and has no seed of its own. Remove `seed=` from
  `StructuralRank` and from `scores.pit`, and set `problem.seed` (or the seed
  of the analysis) to choose the random numbers. The PIT randomizes ties from
  a stream of the seed of its draws, so its values at tied draws change for
  the same seed. See [Scores](../guide/uncertainty/index.md#scores).
- `LaplaceMixture.starts` defaults to 1.

- **Breaking:** every uncertainty, identifiability and sensitivity method
  raises one exception for one kind of error: `ValueError` when a setting, an
  input, the data or the model does not fit the method, and `RuntimeError`
  when a computation that could start fails (an engine run, an optimizer,
  every replicate, a sampler that stops before its target). An input error of
  an SMC method, a bootstrap, the jackknife or a profile that raised
  `RuntimeError`, such as a loss of 0 at the fit or a fit outside the
  support for `BootstrapCoverageEta`, now raises `ValueError`. Catch `ValueError` for these errors where your code caught
  `RuntimeError`. See [Design decisions](../codebase/decisions/uncertainty.md#methods-raise-one-error-type).

- **Breaking:** a setting that means the same thing has one name in every
  method. The level of an interval is `confidence_level`, a count is a plain
  plural noun, and the rule that combines the replicates of a ratio is
  `replicate_rule`. An old name raises a `ValidationError` that names it as
  not permitted. Rename each old name in your code:

  | Where | Before | Now |
  |---|---|---|
  | `SplitConformal`, `RollingConformal`, `JackknifePlus`, `EnbPI`, `AdaptiveConformal` | `level=` | `confidence_level=` |
  | `ConformalResult.intervals` (`PredictionIntervals`), and its report block | `.level`, `"level"` | `.confidence_level`, `"confidence_level"` |
  | `Threshold`, `Threshold.chi_square`, `Threshold.beale`, `Threshold.bootstrap_calibrated` | `level` | `confidence_level` |
  | `ProfileSection` (`IdentifiabilityReport.profile`) | `.level` | `.confidence_level` |
  | `ParticleSwarmConfig` | `num_particles=` | `particles=` |
  | `Sobol` | `base_draws=`, `bootstrap=` | `draws=`, `replicates=` |
  | `Ratio` | `replicates=` | `replicate_rule=` |
  | `Simulation.replicates`, `ReplicateRuns` | `n` | `count` |

  See [A setting has one name and one check](../codebase/decisions/results.md#a-setting-has-one-name-and-one-check).

- One check, in Rust, rejects a confidence level that is not strictly between
  0 and 1, with one message in every method:
  `confidence_level is strictly between 0 and 1, got 1.5; use 0.95 for a 95%
  interval`. Every other setting strictly between 0 and 1 (`target_cess`,
  `stay_probability`, `min_acceptance`, the `quantile` of `ABCRejection`,
  `correlation_threshold`) says `<name> is strictly between 0 and 1, got
  <value>`.

- **Breaking:** a calibration with `replicate_summary=Quantile(q)` compares
  the observations with the one quantile of commol, the one of
  `weighted_quantiles`, `SimulationReplicates.quantiles` and every interval
  (NumPy's `method="hazen"` with equal weights), instead of NumPy's default
  `linear` quantile. `Quantile(0.25)` of the replicates 1, 2, 3, 4, 5 is now
  1.75, not 2.0. So for `q` other than 0.5 the loss and the fitted values
  change slightly; `"median"` and `Quantile(0.5)` do not change. Your code
  does not change; run again a calibration with `Quantile(q)` whose result
  you stored. See
  [One confidence-interval convention](../codebase/decisions/results.md#one-confidence-interval-convention).

- **Breaking:** results that hold the same thing hold it in the same place,
  under the same name. Every result of a method that fits several times lists
  each fit that failed in `failed`, `"<Start|Replicate|Block|Origin> <index>
  failed: <reason>"`, logged as a warning, and counts the fits kept although
  their optimizer did not converge in `not_converged`. Change your code:

  | Before | Now |
  |---|---|
  | `CalibrationResult.failed_runs` | `CalibrationResult.failed` |
  | `BootstrapResult.diagnostics.failed`, `.diagnostics.not_converged` | `BootstrapResult.failed`, `.not_converged` |
  | `JackknifeResult.diagnostics.failed`, `.diagnostics.not_converged` | `JackknifeResult.failed`, `.not_converged`; an entry names the block by its index from 0, `"Block 2 failed: …"`, not by its span |
  | `ProfileResult.failed_replicates` | `ProfileResult.failed`, and the new `ProfileResult.not_converged` |
  | The failed refits of `RollingConformal`, `JackknifePlus` and `EnbPI`, only logged as `"Refit i failed: …"` | `ConformalResult.failed` and `ConformalResult.not_converged` |
  | The failed replicates of `BootstrapCoverageEta` in `GibbsResult.diagnostics.notes` | `GibbsResult.failed` |
  | `ProfileResult.report()` | `ProfileResult.identifiability()`; its `report()` gives the data |
  | `sample.effective_size()` | `sample.effective_size`, a property, as on every other object |

  When a method has fewer successful fits than it needs, its `RuntimeError`
  says how many of how many fitted, how many it needs, and the first
  reasons, and names the unit: `"Only 1 of 5 blocks fitted, and the method
  needs 2; …"`. It says `"Every origin failed: …"` (or replicate, block) only
  when none fitted. The jackknife and jackknife+ no longer say that every
  replicate failed when one fit succeeded. See
  [Resampling](../guide/uncertainty/resampling.md#the-result).

### Removed

- **Breaking:** `commol.add_stderr_logger` is removed: commol configures no
  logging, and adds only a `NullHandler` to the logger `commol`. Call
  `logging.basicConfig(level=logging.INFO)` instead, or add a handler of your
  own to `logging.getLogger("commol")`. See
  [Log messages](../getting-started/quickstart.md#log-messages).

### Fixed

- `IdentifiabilityReport.report()` gives the `profile` section, the
  evidence of each profile likelihood; before, the report data had no such
  section.

## [0.9.0] - 2026-10-22

This release gives control over the work of a calibration and over what a
result records. `commol.threads(n, levels)` sets how many threads the parallel
methods use and which levels run in parallel, a particle swarm can stop when
it has converged, a polynomial chaos takes a truncated basis for many inputs,
and `Ratio` gives a sensitivity analysis the ratio of two quantities. A
calibration result records its seed, and `commol.build_info()` names the
version and the commit of the package. Before you upgrade, read the
**Breaking** entry: `CalibrationResult` requires `seed`. See
[Parallelism](../guide/parallelism.md) and
[Sensitivity Analysis](../guide/uncertainty/sensitivity.md).

### Added

- `PolynomialChaos(q_norm=..., max_interaction=...)`: a truncated basis for
  many inputs. `max_interaction` keeps the terms of at most that many inputs,
  and `q_norm`, in `(0, 1]`, the terms whose degrees have
  `Σ α_i^q ≤ degree^q`; the defaults keep the total-degree basis. 20 inputs
  at degree 4 with `max_interaction=2` have 1,221 terms instead of 10,626,
  and the number is known before any run. `ChaosResult` records both. See
  [Bases for many inputs](../guide/uncertainty/sensitivity.md#bases-for-many-inputs).

- `ParticleSwarmConfig.improvement_stop(window=..., tol_abs=..., tol_rel=...)`
  and `ParticleSwarmConfig.radius_stop(tol)`: a particle swarm stops when its
  best loss improved by at most `tol_abs + tol_rel·|best loss|` over the last
  `window` iterations, or when every particle is within `tol` of the best
  position, as a fraction of the size of the search box. Both are off by
  default and have no default values. A run that stops by a rule reports
  `converged=True`, so a profile that holds a prediction with a swarm also
  converges. See
  [Stop rules](../guide/calibration-optimizers.md#stop-rules).

- `commol.set_threads(n)`, the context manager `commol.threads(n, levels)`,
  `commol.get_threads()` and `commol.get_levels()` choose the number of
  threads of every parallel method and the levels that run in parallel. The
  setting holds per Python thread (`contextvars`); without it, commol uses
  one thread per core. A particle swarm scores the particles of each
  iteration in parallel, in the order of the particles, with
  `levels="all"`, the default; `levels="outer"` keeps them in order. A seed
  gives the same result, history included, with any number of threads and
  either level. See [Threads and Parallelism](../guide/parallelism.md).

- A calibration result records the seed it used, `CalibrationResult.seed`
  (`Calibrator.seed`: `problem.seed`, or the seed the calibrator drew), and
  the sample of every uncertainty method and of `Calibrator.point_sample`
  names it in `method.settings["seed"]`. Set as `problem.seed`, it repeats
  the run, also from a result read back from JSON. See
  [Reproducibility](../guide/calibration-optimizers.md#reproducibility).

- `commol.__version__` gives the installed version, and `commol.build_info()`
  gives a `BuildInfo` with the version and the full commit the extension was
  built from (`None` when the source has no git metadata, such as a source
  distribution). A build that is not a release, such as an install from a
  git commit, names its commit as a PEP 440 local version:
  `0.1.0+g<first 7 characters of the commit>`. See
  [Check the installation](../getting-started/installation.md#check-the-installation).

- `Ratio(numerator, denominator, replicates)` is a quantity of interest: one
  `ValueAt`, `Gain` or `Peak` over another, in each run, so PRCC, Morris,
  Sobol and a polynomial chaos read a ratio whose denominator depends on the
  inputs. `replicates` is required: with a stochastic engine, a point of a
  design holds the mean of the ratios (`"within"`), the mean of the ratios
  of the replicates with a positive denominator (`"within_positive"`) or the
  ratio of the means (`"pooled"`). A denominator of 0 makes the ratio of a
  run NaN, and every method follows the NaN rule. See
  [A ratio of two quantities](../guide/uncertainty/sensitivity.md#a-ratio-of-two-quantities).

### Changed

- The tolerance of a profile under a particle swarm with the improvement stop
  rule is the improvement the rule allows at the fit,
  `tol_abs + tol_rel·|objective at the fit|`; without that rule it stays
  `1e-6·max(1, |objective at the fit|)`. See
  [Verdicts](../guide/uncertainty/profiles.md#verdicts-and-the-identifiability-report).

- The parallel methods run in a thread pool of commol, not in the global pool
  of `rayon`: the environment variable `RAYON_NUM_THREADS` no longer limits
  them; use `commol.set_threads`. A particle swarm, which scored its particles
  one after the other, now scores them in parallel by default.

- **Breaking:** `CalibrationResult` requires `seed`; a result saved by an
  earlier version, or built by hand, needs one to load.

### Fixed

- A particle swarm without `problem.seed` takes the seed the calibrator
  draws and logs, so that seed repeats it; before, it drew one of its own
  that no result or log kept.

- A profile that holds a prediction with a particle swarm scores its
  particles in parallel only when the levels of `commol.threads` are
  `"all"`, and stops with the error of the first failing particle, as every
  other swarm does; before, it scored them in parallel at any level and
  reported the error of any failing particle.

## [0.8.0] - 2026-10-22

This release compares runs. `Scenarios` runs every draw of a sample under
several sets of parameter values and compares them in pairs, a quantity of
interest reads a series and reduces it (the peak of the gains over windows,
for example), and `results.quantities` reads the quantities of one run.
Figures are methods of the results they draw: `plot.series`,
`plot.observations` and `plot_compared`, which draws several runs or samples
on the same panels. Before you upgrade, read the **Breaking** entries:
`SimulationPlotter` and seaborn are removed and matplotlib is the optional
extra `commol[plot]`, the quantities take a `series`, and the runs of the
draws of a stochastic sample follow one tree of seeds, so their numbers change.
See [Sensitivity Analysis](../guide/uncertainty/sensitivity.md) and
[Simulations](../guide/simulations.md).

### Added

- `Scenarios(source, sample, scenarios, positions)` runs every draw of a
  sample under each scenario, a set of parameter values that override those
  of the model, and is a mapping from the name of a scenario to its
  `Predictive`. Every scenario runs with the seed of the source, so run `k`
  of each scenario is the same draw with the same seed (common random
  numbers). `compare(name, baseline, quantities, confidence_level)` takes
  the difference of each quantity of interest within each run and reduces
  the differences with the weights of the runs; `summary` reduces one
  scenario. Both give a `ScenarioSummary` per label, with no interval when
  there is one run. `Predictive.quantities(quantities)` gives the quantities
  of every draw, `Calibrator.point_sample(result)` the point estimate of a fit
  as a sample of one draw, and `Calibrator.with_model` and
  `SensitivityAnalysis.with_model` the same source on another model with the
  same seed. See [Scenarios](../guide/uncertainty/index.md#scenarios).

- `plot.observations(observed_data, …)` of `SimulationResults`,
  `SimulationReplicates` and `Predictive` draws one panel per observed series
  (one `label`), with the model read through the observation operator of each
  observation: the sum of its `outputs`, its `window` and its factor. A
  series that sums some outputs, such as `B_g2 + B_g3`, gets its own panel. A
  value with a window is a step over its window, and the band of replicates
  or of a predictive (`at_observations(…, NoNoise())`) a rectangle over it;
  its legend says that it is a band of the model, not a prediction interval.
  See [Plotting](../guide/simulations.md#a-panel-per-observed-series).

- `SimulationResults.quantities(quantities)` reduces one run to its
  quantities of interest, as `dict[str, float]`, and
  `SimulationReplicates.quantities(quantities)` gives one such dict per
  replicate, with the Rust code of `Propagation`. `quantity_schedule` gives
  the positions to run at; a run that lacks one raises `ValueError` that lists
  each missing position with its quantity. The guide shows how to take a ratio
  of two quantities row by row. See
  [The quantities of one run](../guide/uncertainty/sensitivity.md#the-quantities-of-one-run).

- `plot_compared({label: result}, …)` draws several runs, replicates or
  predictives of the same model on the same panels: one line per result,
  labelled with its key and in the same color in every panel, the band of
  each set of replicates and of each predictive, and the observations once.
  `window=` draws the gain of each series over each window. A result without
  the outputs of the first raises `ValueError` that names its key. See
  [Plotting](../guide/simulations.md#compare-results-on-the-same-panels).

- Every figure takes `axes=`, the `Figure` of an earlier figure of commol, and
  draws each series on the panel of that series: each panel has the id of its
  series as the label of its `Axes`. A series with no panel raises
  `ValueError` that names it. See
  [Plotting](../guide/simulations.md#several-results-on-one-figure).

- A quantity of interest reads a series and reduces it. The series is a name
  or a list of names, their `Level`, or `Increments(outputs, window)`: the
  gain of their sum over each window that ends at a position. So
  `Peak(series=Increments("acc", window=7), positions=ends)` is the largest
  gain of `acc` over a window of 7, and `PeakPosition` with the same arguments
  the end of that window, with no class of their own. `commol` exports `Level`
  and `Increments`. See
  [Quantities of interest](../guide/uncertainty/sensitivity.md#quantities-of-interest).

- `PropagationResult.where(mask)` keeps the runs where `mask` is true and
  renormalises their weights to sum to 1, and
  `PropagationResult.effective_size` gives Kish's effective size of the
  weights. A filter conditions the result on the condition of the mask.

- `plot_compared(…, panels="observations")` draws one panel per observed
  series (one `label` of `observed_data`), with each result read through the
  observation operator of each observation, as `plot.observations` reads one
  result: the steps, markers and band of each result in the color of its key,
  and the observations once. The default, `panels="series"`, draws one panel
  per series. See
  [Plotting](../guide/simulations.md#compare-results-on-the-same-panels).

### Changed

- **Breaking:** each figure is a method of the results it draws, and
  `SimulationPlotter` is removed. Each figure returns the matplotlib `Figure`.
  A predictive gives its own figures, so no `Simulation` is built only to draw
  it. See [Plotting](../guide/simulations.md#plotting).

  | Before | Now |
  |---|---|
  | `SimulationPlotter(simulation, results).plot_series(…)`, `.plot_cumulative(…)` | `results.plot.series(…)`, `results.plot.cumulative(…)` |
  | `SimulationPlotter(simulation, replicates, confidence_level=c).plot_series(…)` | `replicates.plot.series(…, confidence_level=c)` |
  | `SimulationPlotter(simulation, run, confidence_level=c).plot_series(calibration_result=predictive, …)`, and `plot_cumulative` | `predictive.plot.series(…, confidence_level=c)`, `predictive.plot.cumulative(…)` |
  | `SimulationPlotter.plot_prcc(result)`, `plot_morris`, `plot_sobol`, `plot_profiles` | `result.plot()` of `PRCCResult`, `MorrisResult`, `SobolResult`, `ChaosResult` and `ProfileResult` |

- **Breaking:** matplotlib draws the figures, and is the optional extra
  `commol[plot]` (`pip install 'commol[plot]'`); seaborn is no longer a
  dependency, and `import commol` loads no plotting library. Without
  matplotlib, a figure raises `ImportError` with the install line. The other
  keyword arguments of a figure go to `Axes.plot`, not to `seaborn.lineplot`.

- **Breaking:** `PlotConfig.style` takes a matplotlib style or a list of
  styles, and `PlotConfig` has no `palette` and no `context`. The style sheets
  of matplotlib give the former values: `style="whitegrid", context="talk",
  palette="colorblind"` is `style=["seaborn-v0_8-whitegrid",
  "seaborn-v0_8-talk", "seaborn-v0_8-colorblind"]`. The presets
  (`presentation`, `publication`, `notebook`) use these sheets.

- **Breaking:** `ValueAt`, `Peak`, `PeakPosition` and `Trajectory` take
  `series` in place of `outputs`: `ValueAt(outputs="B", at=10)` is
  `ValueAt(series="B", at=10)`. `Gain(outputs, start, end)` keeps `outputs`,
  and the outputs of a quantity read back as a list. The quantities live in
  `commol.context.inference.quantities`, and cross to Rust as the JSON of
  their classes.

- Every reduction across runs raises `ValueError` on a quantity that is NaN:
  `PropagationResult.summary()`, `member_statistics`, `weighted_quantiles`, and
  Morris, Sobol and the polynomial chaos before they estimate, as PRCC did.
  The message of a method names the quantity, the points where it is NaN, and
  the point, inputs, replicate and seed of the first such run.
  `PropagationResult.values` keeps the NaN of each run.

- **Breaking:** the runs of the draws of a sample follow one tree of seeds.
  Draw `j` of a stochastic `Predictive`, `Propagation`, `at_observations` or
  score takes a seed of its own, and its replicate `i` takes the seed that
  replicate `i` of `Simulation.replicates(n, seed=…)` takes from it. Before,
  the seed of a replicate depended on the number of replicates. So the
  numbers of these methods with a stochastic engine change from 0.7.0. The
  runs of a draw no longer change with the number of replicates or of
  draws, and `Predictive.draw_seeds` and `PropagationResult.draw_seeds` give
  the seed of each draw: `Simulation.replicates(n, seed=draw_seeds[j])` of
  the model at the values of draw `j` gives its runs. See
  [The predictive](../guide/uncertainty/index.md#the-predictive).

### Fixed

- A NaN in the values a `Peak` or a `PeakPosition` reads makes it NaN. The
  peak skipped the NaN, and the position of the peak was the first position
  when the value there was NaN.

- A weighted quantile of values with a NaN raises `ValueError`, where it
  sorted the NaN by its sign bit and gave a wrong number.

## [0.7.0] - 2026-10-22

This release reads a run through the classes of commol. The axis of a model
gives the window grid (`AxisUnits.window_end` and the other methods) and, from
a calendar origin, the date of each position. `Simulation.sample`,
`total_series` and `windowed_totals` read bins, accumulators and output names,
and each report is a method of the object it describes. Before you upgrade,
read the two **Breaking** entries: the module functions of the window grid
and the report functions are removed, and the table under Changed gives the
method that replaces each one. See [Reports](../guide/reports.md) and
[Simulations](../guide/simulations.md).

### Added

- `commol.sample(series, positions, at)` reads a series at given positions,
  and `Simulation.sample(results, ids, at)` reads bins and accumulators by id
  at given positions, next to `Simulation.windowed_totals`: for example, the
  amount in a bin at the end of each window. An id stands for all its outputs,
  summed; an output name stands for itself.

- An axis of time can start at a calendar date: `ModelBuilder(origin=...)`
  takes a `date` or a `datetime` with no time zone, and `Model.axis_units()`
  gives `position_of`, `moment_of`, `day_ends` and `week_ends` (weeks that
  start on a weekday of the caller). `SimulationPlotter.plot_series` labels the
  ticks of such a model with dates when it has no `position_label`. An origin
  needs a step unit of time of a whole number of days or a whole fraction of a
  day: `month` and `year` raise `ValueError`.

- `Simulation.total_series` and `Simulation.windowed_totals` take output names
  as well as bin and accumulator ids. An output name stands for itself, so
  `simulation.windowed_totals(results, ["moved_g2", "moved_g3"], 2)` sums two
  strata of the accumulator `moved`; a bin or accumulator id still stands for
  all its outputs. An id that is neither raises `KeyError` with the bins,
  accumulators and outputs of the model.

- The window grid on the axis: `AxisUnits.window_start(index, span, unit)`,
  `window_end(index, span, unit)`, `window_index(at, span, unit)` and
  `window_ends(span, unit, end)`, for a window of a span in a unit, such as
  one week. The width must be a whole number of steps.

- `SimulationResults.model` and `SimulationReplicates.model`: the results of
  `Simulation.run`, `run_at` and `replicates` hold a reference to their
  `Model`, and the runs of `Predictive.runs()` the model of its source. A
  report reads the outputs of each bin and accumulator from it; results with
  their model still pickle. See [Reports](../guide/reports.md).

### Changed

- **Breaking:** the module functions `window_start`, `window_end`,
  `window_index` and `window_ends` are removed from `commol`. Use the methods
  of `AxisUnits` (`Model.axis_units()`), with the width as a span and a unit:
  `window_end(2, 168)` on an axis of hours is
  `units.window_end(2, 1, "week")`, and a width of 7 steps is `(7, "day")`,
  or `(7, "step")` when the model declares no step unit.

- **Breaking:** each report is a method of the object whose data it
  describes, and `commol` exports no report function. The private functions of
  `api/reporting.py` build the blocks, with the same keys in the same order.
  See [Reports](../guide/reports.md).

  | Before | Now |
  |---|---|
  | `simulation_report(simulation, results, at, window, num_windows, …)` | `results.report(positions, window, windows, …)` |
  | `simulation_report(…, ensemble_runs=replicates, confidence_level=c)` | `replicates.report(c, positions, window, windows, …)` |
  | `simulation_report(…, ensemble_runs=predictive, confidence_level=c)` | `predictive.report(c, positions, window, windows, …)`, with no run of a point estimate |
  | `applied_parameters_report(model, result, problem, …)` | `model.applied_parameters(result, problem, …)` |
  | `identifiability_report(report)` | `report.report()` |
  | `commol.api.reporting.observed_report`, `series_report`, `accumulators_report` | The blocks `observed`, `series` and `accumulators` of a report |

## [0.6.1] - 2026-10-22

This release fixes bugs of 0.6.0 and changes no API. A series point or an
impulse that the engine cannot run now fails at `ModelBuilder.build` and
`Model.update_parameters`, not at `Simulation`. Errors of a `Predictive` draw
say what was not feasible. A PRCC analysis with a NaN value raises a
`ValueError` and no longer runs forever. The quantities of a sensitivity
analysis read a gain at its start and give different labels to close
positions. A windowed series is plotted at the position of each observation,
and a figure no longer changes the global style of matplotlib.

### Fixed

- The default label of a quantity of interest wrote each position with 6
  significant digits, so two positions close together gave one label and the
  analysis raised `ValueError`: `Trajectory(outputs="A", positions=[1000000,
  1000001])` gave `"A at 1e+06"` twice. The label now writes each position in
  full (`"A at 1000000"`, `"A at 1000001"`).
- `ModelBuilder.build` and `Model.update_parameters` accepted a series point
  or an impulse position that the engine cannot run, such as a point or an
  `impulse_at` of 1.5 under `DifferenceEquations` or `StochasticSteps`, or a
  `"pulse"` point at 1.5 under any engine; the error came only at
  `Simulation(model)`, as `RuntimeError`. Both now raise `ValueError` with the
  message of the engine, and a rejected update leaves the model unchanged.
- The `InfeasibleStateError` of a draw of a `Predictive` said only
  `InfeasibleRun`. It now gives the message of the run after the index of the
  parameter set: the compartment, the step and the flows, as
  `Simulation.run` reports them.
- The `InfeasibleStateError` of a draw of a `Predictive` whose initial
  amounts leave the remainder below zero said only
  `InfeasibleStart { excess: ... }`. It now gives the message a calibration
  gives for the same values after the index of the parameter set: what the
  amounts exceed, the remainder and by how much.
- A `Gain(start, end)` quantity read its start at `end - (end - start)`,
  which can round away from `start`: `Gain(start=0.1, end=0.4)` read it at
  `0.09999999999999998`, so on the `DifferentialEquations` engine the gain
  did not start where `ValueAt(0.1)` reads. A gain now reads its start at
  exactly `start`.
- `SimulationPlotter.plot_series` read the model of a windowed series with
  the first window it found, at the window ends of a grid from 0, so with
  observations at 7.5 and 17.5 no model value sat on an observation and the
  windows of the other observations were ignored. When the observations of a
  series have a window, or differ in outputs or factor, the model line, the
  band of the replicates and the band of a `Predictive` now read each
  observation through its own operator at its own position, as the loss
  reads it. Observations with several factors no longer leave the line
  unscaled or raise `ValueError` with a `Predictive`.
- A figure of `SimulationPlotter` with a `PlotConfig` `style`, `palette` or
  `context` changed the global style of matplotlib, so every figure drawn with
  matplotlib after it had the style of commol. The style now applies to the
  figure of that plot only, and `matplotlib.rcParams` is as it was before the
  call.
- `PRCC` never returned when a quantity was NaN at a draw. It now raises
  `ValueError` that gives the index of the quantity and the draws where it
  is NaN, before it ranks any value.

## [0.6.0] - 2026-10-22

This release adds the first uncertainty methods with a stated model of the
data: observation models and their likelihood, the Bayesian posterior and the
Gibbs posterior by sequential Monte Carlo, approximate Bayesian computation,
priors, local approximations (Laplace, the sandwich and the Laplace mixture),
conformal intervals, the profile likelihood and sensitivity analysis with no
data. Before you upgrade, read the **Breaking** entries: an `Aggregate` states
the initial amount of each bin in the unit of that bin, a
`CalibrationProblem` needs a `loss_function`, a `Predictive` takes a
`Calibrator`, and public class names write every letter of an acronym in
capitals. See [Uncertainty](../guide/uncertainty/index.md).

### Added

- Local approximations through `Calibrator.uncertainty`, on the unbounded
  scale of the parameters, with a fit on a bound refused: `Laplace(draws)`,
  the normal `N(θ̂, H⁻¹)` of the observation models, and
  `Calibrator.delta_band(result, positions, level)`, the band of the outputs
  by the delta method; `Sandwich(block_length, draws)`, the covariance
  `H⁻¹ B H⁻¹` with the scores summed over blocks (CR1) and a Student t with
  `G − 1` degrees of freedom, which needs no noise model and refuses
  `loss_function="mae"` and too few blocks; `LaplaceMixture(starts, draws,
  target)`, a Laplace approximation at each optimum of a multi-start fit as
  the proposal of Pareto-smoothed importance sampling of a
  `LikelihoodTarget(priors)` or `GibbsTarget(priors, eta)`, which reports
  `k_hat` and marks a sample with `k_hat > 0.7` as unreliable. `curvature`
  is `"gauss_newton"` (the expected Fisher information of each observation
  model, by default) or `"hessian"`. See
  [Local Approximations](../guide/uncertainty/local.md).

- `GibbsPosterior`: the Gibbs posterior `prior(θ) · exp(−η · loss(θ))` of the
  problem's loss, sampled by sequential Monte Carlo from the priors to the
  posterior; it returns a `GibbsResult` with the `η` it ran with. `eta` is
  required: a number above 0, `InformationMatchingEta(block_length)` (the
  curvature of the sum of squares or of observation models against the
  variance of the scores of its terms summed over blocks, with a
  deterministic engine) or `BootstrapCoverageEta(replicates, block_length,
  confidence_level)` (any loss), joined in the union `Eta`. See
  [Sequential Monte Carlo](../guide/uncertainty/smc.md#gibbs-posterior).
- `ABCSMC` and `ABCRejection`: approximate Bayesian computation on a required
  distance, `LossDistance()` or `SeriesDistance(metric)`. With a stochastic
  engine each distance reads one fresh replicate of its particle, so the
  problem's `replicates` and `replicate_summary` do not apply, and the result
  says so. See [ABC](../guide/uncertainty/smc.md#abc).
- `SMCResult`: the sample, the level of each step, the log evidence, the
  value each particle keeps and the diagnostics of each step (`SMCDiagnostics`:
  effective size, resampling, acceptance rate and move count).
- Priors: `Uniform()`, `LogUniform()`, `Beta(a, b)`, `Normal(mean, sd)`,
  `LogNormal(mu, sigma)` and `Gamma(shape, rate)`, each truncated to the
  bounds of its parameter, and `Priors` (a mapping of parameter to
  distribution, or `Priors.uniform_on_bounds()`). Every sampling method takes
  `priors` with no default, and a prior whose bounds do not fit raises a
  `ValueError` that names the distributions that fit them. Their draws are
  the same on every platform. See [Priors](../guide/uncertainty/smc.md#priors).
- `SampleIdentifiability(reference="prior", priors=...)`: the contraction of
  each parameter against its prior. See
  [Identifiability](../guide/uncertainty/identifiability.md#a-sample).
- Conformal prediction intervals through `Calibrator.uncertainty`: intervals
  for observations with a coverage guarantee that needs no noise model, no
  prior and no correct model. `SplitConformal` scores held-out observations;
  `RollingConformal` gives intervals ahead of the data, one quantile per
  horizon (the `k`-th observation after an origin); `JackknifePlus` refits once
  per block left out, with the CV+ bound for blocks of more than one
  observation; `EnbPI` draws its own replicates of blocks for series with
  dependent errors; `AdaptiveConformal` walks a stream of new observations
  with a working level that moves with each miss. Each takes a required
  `score` (`"absolute"`, `"normalized"` or `"spread"`), `by` (`"series"` or
  `"all"`) and `level`, and returns a `ConformalResult` (`PredictionIntervals`,
  the scores and the guarantee). See
  [Conformal Prediction Intervals](../guide/uncertainty/conformal.md).
- `SimulationPlotter.plot_series(prediction_intervals=...)` draws the band of
  conformal intervals, and `simulation_report(..., prediction_intervals=...)`
  reports them.
- Observation models: `CalibrationProblem(observation_models={series: model})`
  takes one model per series label, `NormalObservations(sd)`,
  `LogNormalObservations(sigma, center)` (`center="mean"` or `"median"`, no
  default), `StudentTObservations(df, scale)`, `PoissonObservations()`,
  `NegativeBinomialObservations(overdispersion)` (variance `μ + φμ²`) or
  `BinomialObservations(trials)` (or the new `ObservedDataPoint.trials` of each
  observation), and the problem fits by maximum likelihood. A noise argument
  is a number or a calibration parameter of the new type `"noise"`, which never
  reaches the engine. Observed values outside a model's support raise a
  `ValueError` that names the observation. A stochastic engine and
  `normalize_observations` need a loss. See
  [Observation Models](../guide/uncertainty/observation-models.md).
- `BayesianPosterior`: the posterior of the observation models and its log
  evidence, on the sequential Monte Carlo sampler. `GibbsPosterior` on
  observation models is the fractional posterior `prior · L^η`.
- `DistributionNoise()`: prediction intervals that draw from the observation
  models; `commol.scores.log_score`, the log density of held-out observations
  under the predictive, which needs `DistributionNoise()`.
- `ParametricBootstrap(replicates)`: fits to data sets drawn from the
  observation models at the fit. `ResamplingDiagnostics.block_length` is `None`
  for it.
- A model can hold bins of different units, such as a count and a
  concentration: each bin starts at an amount in its own unit, a model whose
  bins all start at `0` builds and runs, and the
  `Calibrator` rejects an initial-condition remainder whose unit is not the
  unit of a calibrated amount. See
  [Bins of different units](../guide/building-models.md#bins-of-different-units).

- Sensitivity analysis with no data: `SensitivityAnalysis(simulation,
  inputs, replicates=None, seed=None).run(method)`, the third entry point of
  the registry, with methods that subclass `SensitivityMethod[R]`. What varies
  is an `InputSpace` of calibration parameters (model parameters or initial
  conditions): independent distributions (`InputSpace.independent`,
  `InputSpace.uniform_on_bounds`) or a weighted sample
  (`InputSpace.from_sample`). Each run is reduced in Rust to quantities of
  interest: `ValueAt`, `Gain` (as an observation with a window), `Peak`,
  `PeakPosition` and `Trajectory`. With a stochastic engine `replicates` and
  `seed` are required; a point of a design holds the mean of its replicates
  on common seeds, and a propagation keeps every replicate. See
  [Sensitivity Analysis](../guide/uncertainty/sensitivity.md).
- `Propagation`: the draws of the inputs and the quantities of each run
  (`PropagationResult`, with `summary(confidence_level)`); its sample gives
  the bands of a `Predictive`, which runs the same replicates.
- `PRCC`: partial rank correlation coefficients and their p-values
  (`PRCCResult`), weighted for a sample, with Kish's effective size in the
  test.
- `Morris`: elementary effects, `mu`, `mu_star` and `sigma` (`MorrisResult`),
  on levels at the centres of strata of equal probability.
- `Sobol`: first-order and total indices on the design of Saltelli, with
  bootstrap intervals (`SobolResult`).
- `PolynomialChaos`: a Legendre expansion of `F(x)`, its leave-one-out error,
  its Sobol indices and `predict` as a surrogate (`ChaosResult`); the number
  of terms is checked before any run.
- `SimulationPlotter.plot_prcc`, `plot_morris` and `plot_sobol`.
- `ProfileLikelihood(parameters, threshold, max_steps)`: the profile of each
  parameter, refitted with the problem's own optimizer at values that step
  away from the fit by the adaptive step of Raue et al. (2009), its pointwise
  interval and its verdict (`identifiable`, `practically_non_identifiable`,
  `structurally_non_identifiable`), in a `ProfileResult`;
  `ProfileResult.report()` gives the verdicts as an `IdentifiabilityReport`
  with a `profile` section. A profile is read by resetting the other
  parameters to their fitted values (`ProfileResult.structure`), against the
  floor of two evaluations (`ProfileResult.evaluation_floor`): it gives the
  structural verdict `in_combination` or `non_identifiable`, and names the
  parameters that compensate it. A crossing made by the bound of another
  parameter counts as an open side (`Profile.box_crossings`), and
  `Profile.refit_error` measures the error of a refit at the fit. See
  [Profile Likelihood](../guide/uncertainty/profiles.md).
- `PredictionProfile(specs, threshold)`: the interval of each prediction
  (`PredictionSpec`), held at each value as an equality constraint by an
  augmented Lagrangian. See
  [Prediction profiles](../guide/uncertainty/profiles.md#prediction-profiles).
- `Threshold.chi_square(level)` (observation models),
  `Threshold.beale(level)` (a sum of squares of independent Gaussian errors)
  and `Threshold.bootstrap_calibrated(bootstrap, level)` (any loss, calibrated
  by the replicates of a bootstrap).
- `SimulationPlotter.plot_profiles`.

### Changed

- Nelder-Mead with an infinite bound fails with the message `Nelder-Mead
  needs finite bounds`, instead of a rate that is not a number. The sine
  transform of the bounds and the root finder of the quantiles come from the
  argmin fork; calibrations, profiles, quantiles and draws give the same
  results.
- **Breaking:** `InformationMatchingEta(block_length)` takes a
  required `block_length`, an `int` or `"auto"`: the variance of the scores
  is summed over blocks of positions, as the meat of `Sandwich`, so it holds
  under dependence; `block_length=1` gives the independent version. It also
  takes a problem with observation models, where a correct model gives
  `η ≈ 1`.
- **Breaking:** `Predictive(source, sample, positions)` takes a `Calibrator`
  or a `SensitivityAnalysis` (any `PredictiveSource`), and its attribute
  `calibrator` is now `source`. `Predictive.at_observations` of an analysis
  takes only `NoNoise()`, and `scores.log_score` refuses its predictive.
- With a stochastic engine, every replicate of every draw of a `Predictive`,
  of `Predictive.at_observations` and of `scores.log_score` has a seed of its
  own, derived from the calibrator's seed; before, every draw ran the same
  replicates of the problem, so a predictive with few replicates showed the
  same few noise paths for every draw. The bands and scores of a stochastic
  predictive change; a seed still repeats them.
- **Breaking:** `ModelStructure.check_calibration_values(model, engine,
  parameters, remainder)` takes the values and the remainder instead of a
  calibration problem; a model kind that implements it changes its
  signature.
- A prior whose family does not fit its bounds is named "The distribution
  ... of" in its message, instead of "The prior ... of".
- The starts of `Calibrator.run(starts=n)` come from the Latin hypercube of
  `commol-random`; a seed gives the same starts as before.
- **Breaking:** an `Aggregate` states the initial amount of each bin in the
  unit of the bin, not fractions of one total. A structure that states
  `total`, `bin_fractions` or `compartment_fractions` is rejected with the
  fields to use.

  | Before | Now |
  |---|---|
  | `Aggregate(total=1000, bin_fractions={"S": 0.99, "I": 0.01})` | `Aggregate(bin_amounts={"S": 990, "I": 10})` |
  | `compartment_fractions={"I_g1": 0.005}` (of the total) | `compartment_amounts={"I_g1": 5}` (in the unit of its bin) |
  | `model.structure.total` | No declared total: `sum(model.structure.bin_amounts.values())` for bins of one unit |
  | `update_bin_fractions({...})`, which kept the sum at 1 | `update_bin_amounts({...})`; the other bins keep their amounts |
  | `update_compartment_fractions(process, {...})` | `update_compartment_amounts(process, {...})` |
  | `stratification_fractions` | Unchanged: the split of the amount of every bin a stratification applies to |

- **Breaking:** a calibrated initial condition of a bin, a bin prefix or a
  compartment is an amount in the unit of its bin, not a fraction of the
  total; its bounds are amounts too. A category value is still a split. The
  remainder takes the sum of the amounts the model states less every other
  compartment, so a bin that no value covers keeps its amount; with a bin left
  `None`, the remainder holds the sum (`{"S": 1000, "I": None}`). Without a
  remainder no initial state is infeasible. A remainder below zero is measured
  as a share of the amounts the values set. The write-back updates the amount
  of a bin only when the values and the remainder set every compartment of it.
- **Breaking:** `applied_parameters_report` gives an aggregate model the
  sections `initial_amounts` (`amount`, `unit`, `calibrated`) and
  `category_splits` (`group`, a list of categories, `fraction`, `calibrated`)
  in place of `initial_conditions`, with no `total` row. A category split is
  reported as the fraction of its group, with no rescaling of its interval.
- **Breaking:** a stochastic engine needs a whole amount in each bin, not a
  whole total, and `largest_remainder` and `random` round within each bin, so
  a unit never moves to another bin.
- **Breaking:** the description of an aggregate model shows the bin amounts and
  their units, and the initial state with the unit of each compartment.
- **Breaking (Rust):** `InitialConditions` has `bin_amounts` and
  `compartment_amounts` in place of `total`, `bin_fractions` and
  `compartment_fractions`; `CompartmentalEngine::total` is the sum of the
  amounts the model states; `check_model_values` returns the plan it resolves,
  and `InitialConditionPlan::coverage` gives the compartments each value sets.
- **Breaking:** `CalibrationProblem.loss_function` has no default: a problem needs
  `loss_function` or `observation_models`, exactly one, and raises a
  `ValueError` with both or neither. Write `loss_function="sse"` where a
  problem relied on the default.
- **Breaking:** public class names write every letter of an acronym in
  capitals, as PEP 8 asks: `OdeSolver` is `ODESolver`. `EnbPI` keeps the
  spelling of its authors. See
  [Conventions](../codebase/conventions.md#python).

### Fixed

- A `DifferentialEquations` run with `method="tsit45"` whose solution blows
  up returned NaN with no error. A step that leaves the state not finite now
  raises `RuntimeError` that names the last position the solver reached,
  with either method.
- The finite-difference step of a parameter with an infinite bound was
  infinite, so its sensitivities, identifiability and curvature failed: the
  width of its bounds now counts as 1 in the step.
- The quantile of a distribution hundreds of orders of magnitude below 1,
  such as `Beta(0.01, 1)` at `1e-3`, was far from it, so a prior or a draw
  of such a shape was wrong: when its Newton and halving steps run out, the
  quantile now bisects the floating-point numbers between the two ends.
- A particle swarm whose first particle had a NaN cost kept no best particle
  and returned its start point: a NaN cost now counts as the worst.
- A transition whose rate holds `$compartment` dropped `per_compartment`,
  `accumulators` and the `absolute` of its stratified rates: each transition it
  expands to now keeps them, so with `per_compartment=True` `$compartment`
  reads the compartment that flows, and its accumulators count the flow.
- A `Calibrator` on a `Simulation` that had run shared a calibrated bin, bin
  prefix or remainder across its compartments in the proportions of the last
  state of the run, so its fit, its uncertainty and its `Predictive` started
  from a wrong split: they now share it in the proportions of the initial
  state.
- `Model.update_parameters` wrote each value with no check, so a list of
  points for a series parameter left a model that `Simulation` could not
  read, with an error that named no parameter. It now takes a list of points
  for a series parameter and keeps its mode, or a `SeriesValue` (now in
  `commol`) with a new mode, and it runs the checks of a new model: a value
  it cannot take raises `ValueError` at the call, with the id of the
  parameter, and leaves the model unchanged.
- A `Predictive` at positions out of order failed inside a draw with a
  `RuntimeError` that named a parameter set: it now raises `ValueError`
  before any draw runs, with the first pair out of order, as it does for a
  position the engine rejects.

## [0.5.0] - 2026-10-19

This release replaces the ensemble calibration with uncertainty methods that
have a stated meaning. `Calibrator.uncertainty` resamples the data (bootstraps
and the jackknife) and returns a `ParameterSample`; a `Predictive` runs its
draws, and `commol.scores` scores its bands. `Calibrator.identifiability` tells
which parameters the model and the data determine, and `Calibrator.run(starts=n)`
fits from several start points. Before you upgrade, read the **Breaking**
entries: the ensemble calibration is removed (use `BayesianBootstrap` in its
place), and every interval uses one quantile, so the intervals of
`member_statistics` and `SimulationReplicates.quantiles` move slightly. See
[Uncertainty](../guide/uncertainty/index.md) and
[Resampling](../guide/uncertainty/resampling.md).

### Added

- `Calibrator.run(starts=n)` runs the optimizer from `n` start points in
  parallel and returns the best: the initial guess, then rows of a Latin
  hypercube over the bounds drawn from the problem's `seed`.
  `CalibrationResult.starts` lists every start (`StartSummary`: start, end,
  loss, converged); distinct ends with equal losses show that the data do not
  identify the parameters. See
  [Several starts](../guide/calibration-optimizers.md#several-starts).
- `CalibrationResult.failed_runs`: why each failed start failed, also logged as
  a warning.
- `ParameterSample`: weighted draws of the parameters of a problem, with their
  invariants checked at construction and weighted statistics computed in Rust
  (`mean`, `covariance`, `correlation`, `quantiles`, `interval`,
  `effective_size`, `draws`, `resample`, `summary`). It saves to JSON and loads
  back equal.
- `Predictive(calibrator, sample, positions)`: the run of every draw, in
  parallel, its band (`band`, `series_band`) and every draw's value of a set
  of observations through the observation operator of the loss, with an
  observation noise that has no default (`at_observations(observed_data,
  noise)`: `NoNoise()` for a band of the model, `ResidualNoise(...)` for a
  prediction interval). `SimulationPlotter` and `simulation_report` take a
  predictive.
- `commol.scores`: empirical coverage, interval score, weighted interval score,
  the CRPS of a weighted sample and PIT values of a predictive, per
  observation and per series. Each takes the observation noise, so a band of
  the model is never scored as a prediction interval by accident.
- `Calibrator.uncertainty(method)`: one entry point for every uncertainty
  method. A method's configuration subclasses `UncertaintyMethod[R]` and
  registers its runner; the result type follows the configuration's type.
- `Calibrator.identifiability(method)`: which parameters the model and the
  data determine, and which combinations they determine in their place.
  `StructuralRank` (the rank of the sensitivities at random points, on the
  design of the observations or a dense one), `Collinearity` (Brun's indices
  at a fit or over a sample), `SampleIdentifiability` (contraction,
  correlations and modes of a sample) and `StartsIdentifiability` (starts of
  equal loss that end apart). Each returns an `IdentifiabilityReport` with a
  structural and a practical verdict per parameter; `merge` joins reports and
  `identifiability_report` gives one as plain data. See
  [Identifiability](../guide/uncertainty/identifiability.md).
- `InferenceMethod[R]`, the common base of `UncertaintyMethod[R]` and
  `IdentifiabilityMethod[R]`; each entry point refuses the other kind with a
  `TypeError`.
- `weighted_quantiles`, and a `weights` argument for `member_statistics`.
- `CalibrationHistory` and `CalibrationEvaluation`, returned by
  `Calibrator.run_with_history()`.
- The crate `commol-inference`: the parameter sample, the one quantile, the
  predictive and the scores. See [Uncertainty](../guide/uncertainty/index.md).
- Resampling methods with no noise model, through `Calibrator.uncertainty`:
  `BayesianBootstrap`, `ResidualBootstrap` and `WildBootstrap` return a
  `BootstrapResult` (the fit, one draw per replicate, diagnostics, and the
  Monte Carlo error of each interval bound, `interval_error`);
  `BlockJackknife` returns a `JackknifeResult` (estimate, bias, standard
  error, pseudo-values and the influence of each block). Blocks of consecutive
  positions span every series; `block_length` is an integer or `"auto"`
  (Politis and White). See [Resampling](../guide/uncertainty/resampling.md).
- `ResidualNoise`: observation noise made of blocks of residuals of a fit,
  which gives a prediction interval with no noise model; `NoNoise` gives a
  band of the model.
- `CalibrationProblem::with_data` (Rust): a copy of a problem with other
  observed values and weights.

### Changed

- The `latest` version of the documentation site is the documentation of the
  last release, published from its tag; the `dev` version follows `main`.
- **Breaking:** every interval uses one quantile: the weighted inverse of the
  empirical distribution between the midpoints of the cumulative weights,
  NumPy's `method="hazen"` with equal weights. `member_statistics` and
  `SimulationReplicates.quantiles` used NumPy's `linear`; their intervals move
  slightly.
- **Breaking:** `CalibrationResult.best_parameters` keeps the order of
  `CalibrationProblem.parameters`.
- **Breaking:** `Calibrator.run_with_history()` returns a `CalibrationHistory`
  instead of the extension's object.
- **Breaking:** `applied_parameters_report(..., sample=, confidence_level=)`
  reads the statistics of a `ParameterSample`; `parameter_statistics` is gone.
- **Breaking (Rust):** `CalibrationProblem` is split into layers
  (`ForwardModel`, `ObservationOperator`, `Feasibility`, `Discrepancy`,
  `OptimizationObjective`); `optimize` takes the problem by reference, and the
  optimizer configurations have no `verbose` field. `CalibrationEvaluation` has
  no `predictions`.
- A verbose optimizer writes its table to Python's `sys.stdout` only, and needs
  one start; the library never writes to the standard error.
- A rate, a formula parameter or a constraint that the expression compiler
  cannot compile, such as `typeof(k)`, is rejected when the model or the
  calibration problem is built, with the reason; it no longer fails when it is
  evaluated.
- The functions `erf`, `erfc`, `tgamma`, `lgamma`, `j0`, `j1`, `y0`, `y1`,
  `fdim` and `remainder` of a rate come from the `libm` crate, so they give the
  same values on every platform; the other functions still come from the
  system's math library.
- `AxisPattern` is abstract: `AxisPattern()` raises `TypeError`. Build a
  pattern with one of its class methods.
- Error messages that list ids (the stratifications of a transition, the
  missing and extra keys of the fractions of an `Aggregate`) list them in
  sorted order.

### Fixed

- Several starts of Nelder-Mead began at the same point: each start now comes
  from the run seed.
- A parameter value could land on another parameter when results were read by
  name from an unordered mapping.
- A failed run of a multi-start optimization was written to the standard error
  and dropped; it is now part of the result.
- An error of the model at a vertex of the initial Nelder-Mead simplex made
  the optimizer panic; it now stops the calibration with its message (the
  argmin fork returns the error).
- A `to` condition on a transition without a source was ignored by its flow
  but applied by its accumulators; `build` now rejects it, and a `category`
  condition chooses the compartment the flow enters.
- `Model.from_json` read a file in the encoding of the system; it now reads
  UTF-8, the encoding of JSON.
- `Model.check_unit_consistency` checked every flow of a category transition
  in the unit of the first bin of the model; each flow is now checked in the
  unit of its own bin. In a model of bins of different units, a rate that
  reads `total`, a subgroup total, a category or `neighbors` fails the check:
  these add amounts of different units.
- The LaTeX description wrote a part of a rate in parentheses, such as
  `(I / N)` or the argument of `sin(t / T)`, as upright text, and wrote a
  unit inside the name of another variable (`A_{g1(\text{item})}`); only units
  are text now, each after its variable. The expanded form merges `k * p` and `k * (1 - p)` also when a
  third term shares another factor, and a merged term keeps the place of its
  first term.
- `ModelBuilder` and `Simulation` logged to the root logger of the
  application; they log to the `commol` logger, as the rest of the library
  does, so `add_stderr_logger` shows their messages.
- `AxisPattern.combine(mode="max")` raised "Unknown combine mode": a mode may
  be given by its value.
- A quantile level outside `[0, 1]`, or NaN, gave the smallest or the largest
  value; `ParameterSample.quantiles` and `weighted_quantiles` now raise
  `ValueError`.
- `AxisPattern.from_formula` and `AxisPattern.combine` with a `config` that
  allows formulas longer than the default rejected them anyway; the
  configuration now applies.
- A rate of `StochasticSteps` that is not a number raised
  `InfeasibleStateError` and said that a rate cannot be negative; it now raises
  `RuntimeError`, as `StochasticEvents` does, and says that the rate is not a
  number.
- A rate that is not a number gave amounts that are not numbers in
  `DifferenceEquations` and `DifferentialEquations`, and a calibration only
  penalized the run; the run now stops with `RuntimeError` that names the flow
  and the position, as in the stochastic engines.
- A rate that read an id with digits, such as `e1`, `E2` or `c1star`, was
  rejected as a malformed number; and the exponent of a number such as `1e1`
  was read as the undefined variable `e1`. An identifier now starts only after
  a character that is not a letter, a digit or `_`, so the digits of an id
  belong to its name and an exponent belongs to its number.

### Removed

- `commol.api.model_loader.ModelLoader`: a copy of `Model.from_json` that no
  page documented and nothing called.
- **Breaking:** the ensemble calibration: `Calibrator.run_probabilistic`,
  `ProbabilisticCalibrator`, `CalibrationProblem.probabilistic_config`,
  `ProbabilisticCalibrationConfig`, `ProbabilisticCalibrationResult`,
  `EnsembleSolution`, `EnsembleSelectionSummary`, `ParameterSetStatistics` and
  their settings classes. Its intervals had no statistical meaning: its members
  were points the optimizer visited, and the selection maximized the coverage
  of the data it was fitted to. The closest replacement with no noise model is
  `Calibrator.uncertainty(BayesianBootstrap(replicates=..., block_length=...))`:
  the same parallel fits, each on randomly weighted data, with a stated
  meaning.
- The dependency on `scikit-learn`.

## [0.4.0] - 2026-10-16

### Added

- `Model.describe(layout="default")` returns a `ModelDescription`: the model
  as data, sections of tables, equations, state graphs and text, for every
  kind of model. `render(format)` gives text, Markdown (with the state graph
  as a Mermaid diagram), a LaTeX document, JSON or Graphviz DOT, and
  `save(path)` writes it in the format of the suffix (`.txt`, `.md`, `.tex`,
  `.json`, `.dot`). In a notebook it shows as Markdown. See
  [Describing a Model](../guide/describing-models.md).
- `layout="odd"`: the description follows the ODD protocol for describing
  individual- and agent-based models (Grimm et al. 2006, 2010, 2020), and
  marks what commol cannot fill "To be written by the modeller.".
- `EngineType.dynamics_form` and `EngineType.schedule`: how an engine advances
  (`difference_equation`, `differential_equation`, `stochastic_events`,
  `rules`), as a description shows it.
- The process of a model apart from its structure: `Model.process`
  (`Process`: bins, accumulators, stratifications, transitions) is written
  once, and `Model.structure` lays it out. `Aggregate` is the structure of
  amounts per compartment, the kind `aggregate`.
- `neighbors_<bin>` and `neighbors` in a transition rate: the amount of a bin
  among the neighbors of a unit and their total. A rate that reads them and no
  compartment is proportional. `Aggregate(neighbors=Complete())` makes every
  unit a neighbor of every other; `Groups(stratification, weights)` makes units
  neighbors by the categories of a stratification. A rate that reads them
  needs `neighbors`, with no default.
- Variants of a structure: a subclass with the same kind, fields and engines,
  saved with its `variant` and loaded back as its own class.
- Seeds and replicates for stochastic engines: `Simulation.run(…, seed=)` and
  `run_at(…, seed=)` (`SimulationResults.seed` keeps the seed, drawn when none
  is given), `Simulation.replicates(n, seed=)` with `run` and `run_at` that
  return `SimulationReplicates` (runs in parallel, one seed each; `mean`,
  `quantiles`, `member_statistics`), `Simulation.is_stochastic`, and
  `EngineType.is_stochastic`. `SimulationPlotter` draws the mean and a band of
  replicates, and `simulation_report` takes them as `ensemble_runs`.
- `CalibrationProblem(replicates=…, replicate_summary=…)` calibrates a
  stochastic engine with common random numbers: `"mean"`, `"median"`,
  `Quantile(q)` or `"crps"` (the continuous ranked probability score). A
  stochastic engine needs them and a `seed`; there is no default summary.
- The crate `commol-random`: a portable generator (`Xoshiro256PlusPlus`), the
  derivation of replicate seeds, samplers and a sum tree of rates, for every
  stochastic engine.
- Two stochastic engines of an aggregate model, in the crate
  `commol-stochastic`, that run the process of any model as whole units moved
  at random, with no change to it: `"StochasticEvents"` (continuous time:
  Gillespie's direct method, exact also when a rate reads `t`, or
  `method="tau_leaping"`) and `"StochasticSteps"` (the chain binomial, with
  per-step probabilities). Their settings are `StochasticEventsSettings` and
  `StochasticStepsSettings`. A total must be whole, and initial amounts that
  are not whole need `rounding="largest_remainder"` or `"random"`, with no
  default. A per-step probability outside `[0, 1]`, or probabilities out of
  one compartment that sum to more than 1, raise `InfeasibleStateError`. See
  [Stochastic Simulation](../guide/stochastic.md).
- The `dynamics` section of a stochastic model is an event table: each flow,
  the change of one event and its propensity or its draw; the `schedule`
  section lists the settings of the engine.

### Changed

- **Breaking:** a model is built from a process and a structure.
  `ModelBuilder` writes the process, and `build(structure, engine,
  settings=None)` joins it with a structure; the model is a `Model` of the
  class of the structure.

  | Before | Now |
  |---|---|
  | `builder.set_initial_conditions(total=…, bin_fractions=[{"bin": "A", "fraction": 0.9}, …], stratification_fractions=[…], compartment_fractions=[…])` then `builder.build(engine=…)` | `builder.build(Aggregate(total=…, bin_fractions={"A": 0.9, …}, stratification_fractions={"group": {"g1": 0.5, …}}, compartment_fractions={"A_g1": 0.0}), engine=…)` |
  | `CompartmentalStructure`, `Model[CompartmentalStructure]` | `Aggregate`, `Model[Aggregate]` |
  | `model.structure.bins`, `stratifications`, `transitions`, `accumulators` | `model.process.bins`, … |
  | `model.structure.initial_conditions.total`, `bin_fractions`, … | `model.structure.total`, `bin_fractions`, … (mappings by id) |
  | `model.structure.initial_state()`, `subgroup_total(…)`, `get_conditioning_categories(…)`, `update_compartment_fractions(…)` | The same methods, with the process first: `model.structure.initial_state(model.process)`, … |
  | `model.structure.update_initial_conditions(…)` | `model.structure.update_bin_fractions(…)` |
  | `model.structure.outputs_by_source()`, `series_label(id)` | `outputs_by_source(model)`, `series_label(model, id)` |
  | `CompartmentFraction`, `BinFraction`, `StratificationFraction(s)` | Mappings in `Aggregate` |
  | `CompartmentalEngineType` | `EngineName` |

- **Breaking:** the model file has `process` and `structure`, the structure has
  `kind: "aggregate"` and `variant`, and the fractions are objects keyed by
  id. A file of 0.3 does not load; rebuild the model and save it again.
- **Breaking:** an engine names a method, not a kind: `EngineType.kinds`
  replaces `EngineType.kind`, and the error of an engine that does not run a
  kind lists the engines of that kind.
- **Breaking:** `neighbors` and the names that start with `neighbors_` are
  reserved: no parameter, bin or accumulator may take them.
- `EngineType.equation_lhs` comes from the form of the dynamics of the engine
  (`DynamicsForm` in Rust), and returns `None` for an engine without
  equations.
- Python names follow PEP 8: ruff checks them (`N` rules).

### Fixed

- The unit check reads the flows the engine plans, so a proportional rate is
  checked as its flow, the rate times its source: `k2` in `1/day` from a bin in
  `item` passes as `item/day` instead of failing as `1/day`.

### Removed

- **Breaking:** `Model.print_equations`, `ModelStructure.print_equations` and
  `PrintEquationsOutputFormat`. `model.describe().render("text")` shows the
  same compact and expanded forms, and `render("latex")` writes them in LaTeX;
  a description with some units missing shows none instead of raising.

## [0.3.0] - 2026-10-15

### Added

- An ODE engine: a model built with `engine="DifferentialEquations"` runs in
  continuous time, `dX/dt = Σ flows`, integrated by `diffsol`
  (`commol-differential`). It has the same flows,
  outputs, accumulators, calibration, ensembles, reports and plots as a
  difference model. On `[n, n + 1)` a rate sees `step = n` and the continuous
  `t`. The engine restarts its integration at each integer when a rate reads
  `step`, and where `t` reaches a value a rate compares it with
  (`t >= t_on`), so the solver cannot step over a jump.
- `OdeSolver` (`method` `"tsit45"` or `"bdf"`, `rtol`, `atol`), the settings
  of the `DifferentialEquations` engine, stored in the model as
  `Model.engine.settings` and set with
  `build(engine="DifferentialEquations", settings=...)`.
- One list of engines in the Rust engine (`EngineType`): each entry names the
  model kind it runs, validates its settings, and says the form of its
  equations and what a rate means (`per_step` or `continuous`). Python asks it
  through `commol_rs.engine.EngineType` instead of branching on engine names.
- `Simulation.run_at(at)`: the outputs at the given sorted positions. A
  differential model takes any position, such as `0.5`.
- `SimulationResults`: what `Simulation.run` and `run_at` return. A `dict`
  from output name to values, as before, with the positions in `positions`
  and the rows in `rows()`.
- The window grid as module functions: `window_start`, `window_end`,
  `window_index` and `window_ends`, beside `windowed_totals`.
- A public Pydantic class rejects an unknown field and names it
  (`extra="forbid"` on every class of `commol.__all__`), so a misspelt or an
  old name is an error instead of being dropped.
- Impulses: `add_transition(..., impulse_at=[...])` or
  `impulse_every=(period, offset)` moves the rate (a fraction or an amount) at
  once at those steps, in both engines, before the flows of the step.
  `print_equations` lists impulses apart.
- `ModelBuilder(..., step_unit=...)` (`Model.step_unit`): the unit of one step
  for the unit check, of any single dimension: the axis is the independent
  variable of the model, which need not be time (for example `"meter"`).
  Without it, the step unit is `step`, the default increment of any axis.
- `AxisUnits(step_unit, engine)` and `Model.axis_units()`: `steps(value, unit)`
  converts a value in any unit of the step dimension to steps on `pint`
  (`3 day` with a step of `hour` is `72.0`; `2 km` with a step of `m` is
  `2000.0`), and raises, naming both units, for a unit of another dimension.
  `rate_from_mean_span(span, unit)` and `rate_from_probability(probability,
  period, unit)` ask the engine what a rate means: the per-step rate for
  `DifferenceEquations` (`1 - exp(-1/n)`, `1 - (1 - p)^(1/n)`) and the
  continuous rate for `DifferentialEquations` (`1/n`, `-ln(1 - p)/n`).
- `EngineType.rate_from_mean_span` and `rate_from_probability` in the engine's
  list (`RateMeaning` in Rust), which hold those formulas.
- The unit check reads `if(condition, a, b)`: both branches share a unit.
- One interface for every kind of model: `ModelStructure`, an abstract base
  class that the structure of each kind inherits. `Simulation`, `Calibrator`,
  `ProbabilisticCalibrator`, the reports and `SimulationPlotter` use only it,
  so another kind (for example an agent network) is a new subclass with its
  engines in the engine's list. `CompartmentalStructure` is the structure of a
  compartmental model. A subclass that misses a method cannot be made, and the
  error names the method.
- An engine runs one model kind, and a model whose engine runs another kind is
  rejected when it is made, with an error that names the engine and both kinds.
- Each kind declares the values calibration may set besides `parameter` and
  `factor` (`ModelStructure.calibration_value_types`); a compartmental model
  accepts `initial_condition`. A type the kind does not accept raises an error
  that names the kind.

### Changed

- **Breaking:** a `DifferenceEquations` step that leaves a compartment below
  zero (by more than `10⁻¹²` of the total) raises `InfeasibleStateError`, a
  `ValueError`. The message names the compartment, the step, its amount before
  the step and the flows out of it. Before, the run went on with negative
  amounts. A calibration gives such a run a penalty and goes on; any other
  error of a run still stops it. See
  [Amounts below zero](../guide/transitions.md#amounts-below-zero).
- **Breaking (Rust):** `SimulationEngine::run_at_positions_into_buffer`,
  `run_at`, `run` and `step` return `SimulationError` (`Infeasible` or
  `Failed`) instead of `String`.
- A position on the axis is a `float` everywhere, and the engine alone decides
  which positions its model type accepts: a difference model integer steps, a
  differential model any non-negative position. Its error names the
  observation, constraint, impulse transition or series parameter that holds
  a rejected position. The engine trait runs only at positions
  (`run_at_positions_into_buffer`); `run(n)` returns the grid it ran on.
- The engine of a model and its settings are one field,
  `Model.engine = {type, settings}` (`ModelEngine`), which replaces
  `Model.dynamics.typology`. The
  builder takes the engine where it builds, so one builder builds the model
  for each engine with its own settings. Settings of one engine given to
  another raise an error from the engine that names both. A model file
  saved before has `dynamics.typology` and does not load: move it to
  `"engine": {"type": …, "settings": null}`.

  | Before | Now |
  |---|---|
  | `builder.build(typology="DifferenceEquations")` | `builder.build(engine="DifferenceEquations")` |
  | `model.dynamics.typology` | `model.engine.type` |
  | `commol_rs.difference.DifferenceEquations(model)`, `commol_rs.differential.DifferentialEquations(model)` | `commol_rs.engine.Engine(model)`, the engine the model names |
- A model is a header, an engine and a `structure`: `Model.structure`, tagged
  in JSON by `kind`, replaces `Model.population` and `Model.dynamics`, and the
  transitions are held once. `Model` is generic in its structure class:
  `ModelBuilder.build` returns `Model[CompartmentalStructure]`, and
  `Model.from_json` reads the class from the `kind`. The methods that only a
  compartmental model has moved to its structure. A model file saved before
  has `population` and `dynamics` and does not load: move the fields of
  `population` and `dynamics.transitions` into
  `"structure": {"kind": "compartmental", …}`.

  | Before | Now |
  |---|---|
  | `model.population.bins`, `.accumulators`, `.stratifications`, `.initial_conditions` | `model.structure.bins`, … |
  | `model.dynamics.transitions` (and `model.population.transitions`) | `model.structure.transitions` |
  | `model.initial_state()` | `model.structure.initial_state()` |
  | `model.update_initial_conditions(…)`, `update_stratification_fractions(…)`, `update_compartment_fractions(…)` | `model.structure.update_initial_conditions(…)`, … |
  | `model.subgroup_population(…)`, `model.get_conditioning_categories(…)` | `model.structure.subgroup_total(…)`, `model.structure.get_conditioning_categories(…)` |
  | `model.get_outputs_by_source()` | `model.structure.outputs_by_source()` |
  | `model.get_uncalibrated_initial_conditions()` | `model.structure.uncalibrated_values()` |
  | `commol.context.dynamics` (`Transition`, `StratifiedRate`) | `commol.context.transition` |
  | `commol.context.engine.EngineType` | `CompartmentalEngineType`; `ModelEngine.type` is a `str` that the engine's list checks |
  | `commol_rs.core.Model(population=…, dynamics=…)` | `commol_rs.core.Model(structure=commol_rs.core.CompartmentalStructure(…))` |
  | `commol_rs.model.initial_state(model)`, `expanded_outputs`, `transition_flows` | they take a `commol_rs.core.CompartmentalStructure` |
  | `commol_rs.calibration.check_initial_conditions` | `check_model_values` |
  | `commol_rs.calibration.calibrate(…, initial_population_size, …)` and the other calibration functions | no `initial_population_size`: the engine knows it |

- `applied_parameters_report` returns `parameters` and the sections of the
  kind of the model; a compartmental model adds `initial_conditions`, with the
  same keys as before.
- The Rust engine trait (`SimulationEngine`) is the same for every model
  kind: `compartments`, `state` (was `population`), `set_initial_condition` and
  `output_labels` moved to `CompartmentalEngine` (`commol-model`), and the
  engine resolves and writes the values of its kind (`plan_values`,
  `apply_values`). The initial-condition plan moved from `commol-calibration`
  to `commol-model`. The crates are not published.
- Names of positions, lengths and counts, and names with more than one
  meaning. No old name is kept; an old keyword raises an error that names it.

  | Before | Now |
  |---|---|
  | `ObservedDataPoint(step=…)` | `ObservedDataPoint(at=…)` |
  | `ObservedDataPoint(window_steps=…)` | `ObservedDataPoint(window=…)` |
  | `ObservedDataPoint(compartment="B")` | `ObservedDataPoint(output="B")` |
  | `ObservedDataPoint(compartment="label", compartments=[…])` | `ObservedDataPoint(label="label", outputs=[…])` |
  | `ObservedDataPoint(scale_id=…)` | `ObservedDataPoint(factor_id=…)` |
  | `CalibrationParameter(parameter_type="scale")`, `CalibrationParameterType.SCALE` | `"factor"`, `CalibrationParameterType.FACTOR` |
  | `CalibrationConstraint(time_steps=…)` | `CalibrationConstraint(at=…)` |
  | `NelderMeadConfig(initial_step=…)` | `NelderMeadConfig(initial_simplex_size=…)` |
  | `Simulation.run_at(steps)` | `Simulation.run_at(at)` |
  | `Simulation.windowed_totals(results, source_ids, window_steps, at_steps)` | `Simulation.windowed_totals(results, ids, window, at)` |
  | `Simulation.total_series(results, source_ids)`, `group_outputs(source_ids)`, `outputs_for(source_id)` | `ids`, `id` |
  | `windowed_totals(series, window_steps, at_steps)` | `windowed_totals(series, positions, window, at)` |
  | `window_end_steps(window_steps, num_steps)` | `window_ends(window, end)` |
  | `TimeScale.window_start`, `window_end`, `window_index` | `commol.window_start`, `window_end`, `window_index` |
  | `simulation_report(…, steps, window_steps, …)`, `series_report`, `accumulators_report`, `observed_report` | `at`, `window` |
  | `SimulationPlotter.plot_series(step_to_label=…)` | `position_label=…`, which takes a `float` |
  | `TimePattern` (`commol.api.time_patterns`) | `AxisPattern` (`commol.api.patterns`) |
  | `TimePattern.window(start, end, amount)` | `AxisPattern.interval(start, end, value)` |
  | `TimeSeriesValue`, "time-series parameter" | `SeriesValue`, "series parameter" |
  | `EnsembleSolution.windowed_prediction_steps` | `windowed_prediction_positions` |
  | `engine.current_step()` | `engine.current_position()` |
  | `simulation.run(n, output_format="list_of_lists")` | `simulation.run(n).rows()` |

- Reports, windows, plots and ensemble statistics read a value at a position
  of the results, not at a list index, so a differential model windows and
  reports at non-integer positions once its results hold them. The report key
  `scale_id` is `factor_id`, and `compartments` is `outputs`.
- `EnsembleSolution` has `prediction_positions`; its prediction series hold the
  unit grid and every observation position and window start.
- Series parameters take `(position, value)` points with `float` positions.
  `"linear"` interpolates in `t` and `"step_function"` holds from any
  position; a differential engine restarts its integration at their points,
  so a `"linear"` series is a ramp, not a staircase. `"pulse"` keeps integer
  positions. A difference model gives the same results and rejects a point
  between integers.
- `AxisPattern.interval` and `linear_ramp` read `t` and take `float` bounds. A
  difference model gives the same results; in a differential model an
  interval opens at any position and a ramp is continuous.
- A pattern parameter that the builder generates is named `_series_<n>`
  (was `_ts_<n>`).
- The model compiler moved from `commol-difference` to the new crate
  `commol-model`, which both engines use; `commol_rs.difference` no longer has
  `initial_state`, `expanded_outputs`, `transition_flows` and `check_names`,
  which are in `commol_rs.model`. No public Python name changes.
- `SimulationPlotter.plot_series` and `plot_cumulative` take `series` instead
  of `bins`. An entry is an output name or a bin or accumulator id; an id plots
  the sum of its outputs, so a bin of a stratified model can be plotted. There
  is no `bins` alias: rename the argument.
- Every loss uses the observation weights the same way: `"sse"` is
  `Σ w·r²`, `"rmse"` is `√(Σ w·r² / Σ w)` and `"mae"` is `Σ w·|r| / Σ w`.
  Before, `"sse"` squared the weight and `"rmse"` and `"mae"` ignored it. With
  every weight at 1 nothing changes. The ensemble central loss follows the same
  rule.
- `Simulation.run` starts from the initial state at every call, so two calls
  give the same result. Before, a second `run` continued the first one. To
  continue a run, call `simulation.engine.step()`.
- `Model.check_unit_consistency` infers one step unit from the parameter
  units (`day`, `week`, `step`, …; `step` when there is none) and requires
  every flow to be exactly `<bin unit>/<step unit>`. A model in `"1/step"` now
  passes. Rates in two units of time, such as `"1/day"` and `"1/week"`, are an
  error that names each parameter, because the engine converts nothing. Before,
  the check accepted any convertible unit and defaulted to `day`.

- Nelder-Mead searches inside the bounds: it optimizes an unbounded
  coordinate and scores `x = lo + (hi − lo)·(sin u + 1)/2`. Its first simplex
  is set by `NelderMeadConfig.initial_simplex_size` (a fraction of each bound width,
  default 0.1), which replaces `simplex_perturbation`; there is no alias. Every
  Nelder-Mead result can change.

- The library names what the compartments hold without assuming individuals:
  the **state** (the amount of every compartment), the **amount** (the value
  of one compartment), the **total** (their sum, which the initial fractions
  are taken of) and a **subgroup total** (the sum of the compartments with some
  categories). The total is a `float`, finite and greater than 0, so a model of
  a mass or a concentration can be stated; before, it was an `int`. The rate
  variable `N` is `total` and `N_<cat>…` is `total_<cat>…`, so `N` is a free id
  (a bin may be named `N`). `AxisPattern` takes a `value`. No old name is
  kept: a model file with `population_size` does not load, and the error names
  the field; rename it to `total`, and `N` to `total` in every rate, formula
  and constraint.

  | Before | Now |
  |---|---|
  | `set_initial_conditions(population_size=…)`, `InitialConditions.population_size: int`, JSON key `population_size` | `set_initial_conditions(total=…)`, `InitialConditions.total: float`, JSON key `total` |
  | Rate, formula and constraint variables `N`, `N_<cat>…` (`N_g1`, `N_g1_t1`) | `total`, `total_<cat>…` (`total_g1`, `total_g1_t1`) |
  | `RESERVED_VARIABLE_NAMES = ("N", …)` | `("total", "step", "t", "pi", "e")` |
  | `InitialConditions.subgroup_population(…)`, `CompartmentalStructure.subgroup_population(…)` | `subgroup_total(…)` |
  | `InitialCompartment.population` | `InitialCompartment.amount` |
  | `applied_parameters_report`: row `"N"`, group `"population"` | row `"total"`, group `"total"` |
  | `AxisPattern.pulse(at, amount)`, `pulses(at, amount)`, `periodic(period, amount, offset)`, `interval(start, end, amount)` | `value` instead of `amount` |
  | `get_predefined_variable_units(stratifications, bin_unit)` | `get_predefined_variable_units(subgroup_totals, bin_unit)`, with the names the engine gives |
  | `commol_rs.engine.Engine.population` | `Engine.state` |
  | `commol_rs.core.InitialConditions(population_size=…)` | `InitialConditions(total=…)` |
  | `commol_rs.calibration.calibrated_initial_population` | `calibrated_initial_state` |
  | — | `commol_rs.model.subgroup_total_names(structure)` |
  | Rust: `CompartmentalEngine::population`, `population_size`; `DifferenceEquations::population`, `DifferentialEquations::population` | `state`, `total` |

- The library and its documentation assume no application and no axis: text
  says "along the axis" or "position", not "over time", and examples use
  abstract names. A rate that names no compartment is **proportional** (was
  "per-capita"), because a source can hold a mass or a concentration. The
  errors of a constraint at positions say "Constraint '…' evaluation failed at
  position …" and "Position … not found …" (were "Time-dependent constraint"
  and "Time step").

  | Before | Now |
  |---|---|
  | `AxisPattern.seasonal(amplitude, period, phase, baseline)` | `AxisPattern.sinusoid(amplitude, period, phase, baseline)` |
  | `TransitionFlow.per_capita_of` (`commol.context._model.engine_state`) | `TransitionFlow.proportional_to` |
  | Guide anchor `#absolute-and-per-capita-rates` | `#absolute-and-proportional-rates` |
  | Guide anchor `#time-dependent-rates` | `#rates-that-change-along-the-axis` |

- The unit check no longer infers the step unit from the parameter units: it
  is the declared `step_unit` or else `step`. A model with rates in `1/day`
  and no `step_unit` fails the check, and the message says to declare it:
  add `step_unit="day"` to `ModelBuilder`.
- A unit means what `pint` defines. The list of time aliases (`decade`,
  `quarter`, `semester`, `mo`, `wk`, …) is gone: `pint` reads `quarter` as a
  mass and `decade` as a number, so `1/quarter` was a mass without error. A
  name that `pint` does not define is a unit of its own dimension.
- In the unit check, `t` has the step unit (it is a position on the axis), and
  a number written beside it in a sum, a difference or a comparison is read in
  that unit. `step` stays dimensionless. A rate such as `sin(2 * pi * t / 365)`
  now fails the check; write `sin(2 * pi * t / period)` with `period` in the
  step unit.
- Guides and docstrings say what a rate means for each engine
  ([What a rate means](../guide/core-concepts.md#what-a-rate-means)) instead
  of "every flow applies once per step".

### Removed

- `TimeScale` (`commol.api.time_scale`). Its rate conversions gave per-step
  rates only, which a differential model reads as too small.

  | Before | Now |
  |---|---|
  | `TimeScale(step_seconds=3600)`, `TimeScale.from_step_unit("hour")` | `AxisUnits(step_unit="hour")`, or `model.axis_units()` |
  | `steps_from_seconds/minutes/hours/days/weeks(x)` | `steps(x, "second")`, `steps(x, "minute")`, … |
  | `steps_per_hour`, `steps_per_day`, `steps_per_week` | `steps(1, "hour")`, `steps(1, "day")`, `steps(1, "week")` |
  | `rate_from_mean_duration(seconds)` | `rate_from_mean_span(span, unit)`, which asks the engine |
  | `rate_from_probability(p, period_steps)` | `rate_from_probability(p, period, unit)`, which asks the engine |
  | `window_start`, `window_end`, `window_index` | `commol.window_start`, `window_end`, `window_index` |

  `steps` returns a `float` and does not raise for a fraction of a step; the
  engine rejects a position it does not accept where the position is used.

- Transition conditions: `Transition.condition`, the `condition` argument of
  `ModelBuilder.add_transition`, `ModelBuilder.create_condition`, the
  `Condition` and `Rule` types, and the `LogicOperators` and
  `VariablePrefixes` enums, in Python and in Rust. The engine never evaluated
  them, so a condition that was never true still moved population. Write the
  condition in the rate (`"if(step > 30, k, 0)"`), or use an `AxisPattern` or
  a stratified rate. A model file with a `condition` key raises an error that
  names it.
- `Simulation.run(..., output_format=...)` and `run_at(..., output_format=...)`:
  both return `SimulationResults`; use `rows()` for one row per position.
- `commol_core::integer_steps`, `SimulationEngine::run_into_buffer` and
  `run_at_steps_into_buffer`; the Python checks that repeated the engine's
  rules on positions (`check_steps_for_typology`, the impulse check of
  `Dynamics`, the order check of `Simulation.run_at`).
- The `"weighted_sse"` loss and `LossConfig.weighted_sse()`. Use `"sse"`,
  which uses the weights.
- `commol.constants.ModelTypes`. The names of the compartmental engines are
  the `Literal` `commol.context.engine.CompartmentalEngineType`.
- `Population` and `Dynamics`, in Python (`commol.context.population`,
  `commol.context.dynamics`) and in Rust, and their PyO3 classes; the
  structure of a model is `CompartmentalStructure`.
- `Model.get_uncalibrated_initial_conditions`, `Model.initial_state`, and the
  other methods of a compartmental model on `Model`: they are on
  `model.structure` (see Changed).

### Fixed

- Examples 6 to 9 of the guide (`examples.md`) no longer go below zero.
  Example 6 writes the rate it describes, `k2_max * B * max(0, 1 - C /
  capacity)`; Examples 7, 8 and 9 set `per_compartment=True` on `flow_AB`, so
  each compartment of `A` loses an amount computed from its own amount.
- A differential model: on a piece of integration that ends where a rate
  jumps (a value `t` is compared with, a series point), the right-hand side
  sees `t` below the end. Before, the last solver step of the piece already
  saw the rate after the jump, so a rate of 0.5 that starts at 2.5 moved
  `3·10⁻⁸` of the state before 2.5.

- `CalibrationResult.converged` is `True` only when the optimizer met its
  convergence test or `target_cost`. Before, every finished run reported
  `True`, also one that stopped at `max_iterations`.
- Nelder-Mead from a start value of 0 builds a real simplex, and no longer
  stops at a bound where the clamped loss is flat. Before, the simplex was
  degenerate at 0, and could collapse on a bound and report `converged=True`.
- The verbose Nelder-Mead run uses the configured simplex. Before, it always
  used a 10% perturbation.
- `Model.check_unit_consistency` names every bin without a unit. Before, it
  failed with "'None' is not defined in the unit registry".
- A `Simulation` follows its model: `run` and `engine` rebuild the engine when
  the model changed after the `Simulation` was created, for example with
  `update_parameters` or `apply_calibration_parameters`. Before, it used the
  old values, or failed with "Parameter index … not found" for a parameter
  that was `None`.
- `pi` and `e` work in every transition rate, such as `"pi * k * A"`. Before,
  outside a function argument, the simulation failed with "Variable 'pi' not
  found".
- `%` (remainder) works as the only operator of a rate, as in `"step % 7"`.
  Before, the rate was read as a parameter name.
- A `total_…` variable that names no subgroup total, such as `total_t1_g1`
  when `group` is declared before `type`, is a build error that lists the valid
  names. Before, a formula parameter or the engine gave it the value 0.
- A bin or an accumulator id cannot be `total`, `step`, `t`, `pi` or `e`, and
  a bin cannot expand into a name that is also a `total_<categories>` subgroup
  total. Before, the engine variable hid such a bin, and the two sums shared
  one value.
- A parameter id cannot be `total`, `step`, `t`, `pi`, `e`, an expanded
  compartment or a subgroup total. Before, only bin ids were rejected, and
  the parameter and the engine variable shared one value.
- `ModelBuilder.reset` removes the accumulators, and `ModelBuilder.clone`
  copies them and the `bin_unit`. Before, `reset` kept the accumulators and a
  clone lost them.
- An `initial_condition` id is resolved by the engine when the `Calibrator`
  or the `ProbabilisticCalibrator` is created. A category of a non-binary
  stratification now gives "needs a binary stratification" instead of "not
  found", and bin prefixes (`"B_g1"`) and qualified categories (`"group:g1"`)
  are accepted, as documented. Before, both were rejected as "not found".

### Documentation

- The guide is rewritten and split into pages with one job each (building
  models, transitions, expressions, time patterns, units, simulations, and six
  calibration pages). Every example runs in CI, and every claim was checked
  against the engine. Behaviors that were described wrongly are now documented
  as they are, among them: a violated constraint replaces the loss with a
  penalty.
- The API reference documents every public name, and `Model`, plotting and
  reports have their own pages.
- The site publishes `llms.txt` and `llms-full.txt` for AI agents, and marks
  the pages of the `dev` version.
- `README.md` presents Commol as a library for declarative dynamical
  modelling: the idea behind it, what it can model and its limits, an example
  that derives, calibrates and simulates a model, and links to the guides. The
  citation title is "Commol: High-Performance Declarative Dynamical Modelling
  in Python".
- The package metadata on PyPI has the same description, and keywords and
  classifiers (Python versions, audience, topic, development status).

## [0.2.0] - 2026-10-12

### Added

- Per-compartment initial conditions: `InitialConditions.compartment_fractions`
  (Python and Rust) accepts overrides keyed by expanded compartment name,
  expressed as fractions of the total population and applied after the uniform
  expansion. Includes `ModelBuilder.set_initial_conditions(compartment_fractions=...)`,
  `Model.update_compartment_fractions`, and `Model.initial_state()`, which
  returns each expanded compartment with its bin, categories and initial
  population as the engine builds it. The Rust engine is the single source of
  truth for the expansion and validates the overrides (unknown or repeated
  name, fraction outside `0.0..=1.0`, total above `population_size`).
- Rust `commol_difference::initial_state` and
  `commol_calibration::calibrated_initial_population`, exposed to Python. The
  calibration objective, the calibrated predictions and the model write-back
  share one implementation of how calibration values reach the engine.

### Changed

- **Breaking:** calibrated initial conditions no longer auto-calculate the
  last population-fraction value as the remainder of the population. The new
  `CalibrationProblem.initial_condition_remainder` names the compartment or bin
  that absorbs the difference; it is not calibrated and keeps its own split.
  Without it, values are applied as given and only an initial population above
  `population_size` is penalized. A single exact compartment, such as a
  composite state, can now be calibrated on its own.
- **Breaking:** calibrated initial conditions are applied in fixed layers —
  bins, category splits, exact compartments, remainder — instead of in
  parameter order.
- **Breaking:** a stratification category may be named with a qualified id
  (`sex:female`). A bare name that more than one stratification declares is
  rejected as ambiguous, in calibration and in
  `update_stratification_fractions`, and the report qualifies its rows.
  Category pairs come from the engine's compartment labels instead of parsing
  compartment names.
- **Breaking:** `get_category_fraction`, `subgroup_population` and
  `get_conditioning_categories` resolve category ids like calibration: a
  qualified id (`sex:female`) selects its stratification and a bare name that
  more than one stratification declares raises instead of silently using the
  first one. `get_conditioning_categories` returns qualified ids for shared
  names.
- Bin and bin-prefix initial-condition ids, and the remainder, resolve from the
  engine's compartment labels instead of name prefixes, so a bin `A` no longer
  covers the compartments of a bin `A_x`.
- The Rust engine expands bins and accumulators in one function; the equation
  printer and unit checker use it (`expanded_outputs`) instead of their own
  Python expansion. The Rust `ModelTypes` gains `DifferentialEquations`, so any
  model definition can be expanded; the difference engine rejects it.
- **Breaking:** `update_bin_fractions` / `update_initial_conditions` apply the
  same rule as construction: the bin fractions must sum to 1.0 after the
  update.
- **Breaking (Rust):** `SimulationEngine` requires `compartment_labels()`.
  `CalibrationProblem::fix_auto_calculated_parameters` and
  `get_parameter_fix_info` are removed; `applied_parameter_values` returns the
  clamped values the objective scored, which is what results now report.

- **Breaking:** transitions are planned by the Rust engine from compartment
  labels (bin and categories) instead of compartment names, in one planner
  shared by the simulation and `print_equations`. The printed equations are
  therefore the simulated ones: a per-capita rate is shown multiplied by its
  source (`gamma * I`), a transition without flows shows `(no flows)`, and the
  compact form lists the flows of each transition.
- **Breaking:** building a model rejects a stratified rate whose `category`
  conditions match no source compartment (use `to` to route into a
  stratification the source does not carry), a transition with several
  sources and a per-capita rate, a category transition without exactly one
  source and one target, and a flow that ends in categories no compartment has.
  These used to be skipped without notice.
- Constraint aggregates (a bin or bin prefix in a constraint expression) are
  built from output labels, so a bin `A` no longer sums the outputs of a bin
  `A_x`.

### Fixed

- A transition between bins no longer moves bins whose id starts with the
  source bin id: `A -> B` moved `AB -> BB` as well.
- Transitions between the categories of a stratification (`young -> old`) are
  now simulated; the engine used to ignore them while `print_equations` showed
  them.
- A transition with several sources or targets (`A + B -> 2 C`) now applies its
  flow to every source and target, as documented; the engine used to move only
  the first source to the first target.

- The applied-parameters report now shows the effective head count for bin,
  stratification-category, and subgroup rows — the uniform expansion plus the
  declared `compartment_fractions` overrides — and reports each `fraction` as
  that head count over the declared `population_size`, so every row shares one
  denominator with `N`. An aggregate row written back as overrides no longer reports the stale
  pre-calibration value as `calibrated`, and a bin left uncalibrated keeps its
  `null` marker even when overrides touch some of its compartments.
- A conditional-stratification category is calibrated as a fraction of its
  conditioning subgroup but reported as a fraction of the whole population, so
  the report rescales its `interval` by the subgroup's share of the population;
  the interval now brackets the reported `fraction`.
- `apply_calibration_parameters` writes initial-condition values back so the
  model rebuilds the initial population the engine scored for them. Bin and
  category values update the fractions they declare, as the engine applied them;
  every compartment those fractions do not reproduce is pinned as a
  per-compartment override. The call computes the result before it changes the
  model, so a rejected value leaves the model unchanged.
- `population_size` must be greater than zero, reported as a domain `ValueError`
  rather than a later `ZeroDivisionError`. `update_bin_fractions` re-applies the
  sum-to-1/conservation rule.
- Calibrating a bare category name that two stratifications declare no longer
  builds an initial population larger than `population_size`: the engine used
  to split the overlapping pairs of both stratifications from one snapshot.
- The PyO3 `InitialConditions` binding exposes `population_size`, `bin_fractions`,
  `stratification_fractions` and `compartment_fractions` with getters, and the
  `_commol_rs` stub declares only those real attributes (correcting the
  `bin_fractions` field name).
- An initial-condition override that names an unknown compartment raises a
  `ValueError` when the model is built or updated, and a `RuntimeError` from the
  `DifferenceEquations` engine constructor.
- The greedy ensemble search of probabilistic calibration takes its random
  starts from the shuffled part of the sample (argmin fork). The starts were
  biased towards low indices, so the same seed can now select a different
  ensemble.

### Dependencies

- cranelift 0.126, rand 0.10 (with the argmin fork migrated to it),
  thiserror 2, evalexpr 13 and pyo3 0.29.

## [0.1.0] to [0.1.3] - 2026

See the [GitHub Releases](https://github.com/MUNQU/commol/releases) page for
the details of these versions.

### Added

- GitHub Actions CI/CD pipelines for code quality, building, and PyPI releases
- MkDocs Material documentation site
- Comprehensive development workflow documentation
- Contributing guidelines and release process documentation

## [0.0.1] - 2025

### Added

- Initial release of Commol
- ModelBuilder fluent API for constructing compartment models
- Support for difference equations models
- Compartment states (bins) and stratifications
- Mathematical expression support in transition rates
- Rust-powered simulation engine
- Python API with Pydantic validation
- Basic compartment model support (difference equations)
- Initial conditions and parameter management
- Simulation runner with multiple output formats

### Security

- Expression validation to prevent code injection
- Input sanitization for mathematical formulas

---

## Version History

- **Unreleased** - Current development version
- **0.10.2** - The engine as the one source of rules: settings, model values, windows and formulas; one seed scheme; results equal by value
- **0.10.1** - Fixes of the audit: stochastic changes from the position, one account in the documentation, read-only CI and pinned tools
- **0.10.0** - One way for equivalent things: setting names, seeds and starts, result shapes, error types, the quantile and logging
- **0.9.0** - Control over the work and the record: threads and levels, swarm stop rules, truncated chaos bases, ratios, seeds and build info
- **0.8.0** - Comparing runs: scenarios, quantities of a series, figures as methods and runs on the same panels
- **0.7.0** - Reading a run through objects: the window grid and dates on the axis, series by output name and reports as methods
- **0.6.1** - Fixes of 0.6.0: checks at build, messages of infeasible draws, PRCC with NaN, quantity labels and starts, plots
- **0.6.0** - Uncertainty methods with a model of the data: likelihood, posteriors, local approximations, conformal intervals, profiles and sensitivity analysis
- **0.5.0** - Uncertainty by resampling the data, identifiability of parameters and the removal of the ensemble calibration
- **0.4.0** - One process under several structures and engines, stochastic engines and model descriptions
- **0.3.0** - ODE engine, positions on the axis and one interface for every model kind
- **0.2.0** - Per-compartment initial conditions and engine-planned transitions
- **0.1.0 to 0.1.3** - First 0.1 versions
- **0.0.1** - Initial release with core functionality

For detailed information about each release, see the [GitHub Releases](https://github.com/MUNQU/commol/releases) page.
