# Changelog

[Back to Home](../index.md)

All notable changes to Commol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
  is set by `NelderMeadConfig.initial_step` (a fraction of each bound width,
  default 0.1), which replaces `simplex_perturbation`; there is no alias. Every
  Nelder-Mead result can change.

### Removed

- Transition conditions: `Transition.condition`, the `condition` argument of
  `ModelBuilder.add_transition`, `ModelBuilder.create_condition`, the
  `Condition` and `Rule` types, and the `LogicOperators` and
  `VariablePrefixes` enums, in Python and in Rust. The engine never evaluated
  them, so a condition that was never true still moved population. Write the
  condition in the rate (`"if(step > 30, k, 0)"`), or use a `TimePattern` or a
  stratified rate. A model file with a `condition` key still loads; the key is
  ignored.
- The `"weighted_sse"` loss and `LossConfig.weighted_sse()`. Use `"sse"`,
  which uses the weights.

### Fixed

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
- An `N_…` variable that names no subpopulation, such as `N_t1_g1` when
  `group` is declared before `type`, is a build error that lists the valid
  names. Before, a formula parameter or the engine gave it the value 0.
- A parameter id cannot be `N`, `step`, `t`, `pi`, `e`, an expanded
  compartment or a subpopulation total. Before, only bin ids were rejected, and
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
- `README.md` is a short landing page with links to the documentation.

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
- **0.2.0** - Per-compartment initial conditions and engine-planned transitions
- **0.1.0 to 0.1.3** - First 0.1 versions
- **0.0.1** - Initial release with core functionality

For detailed information about each release, see the [GitHub Releases](https://github.com/MUNQU/commol/releases) page.
