# Changelog

All notable changes to Commol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-compartment initial conditions: `InitialConditions.compartment_fractions`
  (Python and Rust) accepts overrides keyed by expanded compartment name,
  expressed as fractions of the total population and applied after the uniform
  expansion. Includes `ModelBuilder.set_initial_conditions(compartment_fractions=...)`,
  a Python compartment-name generator with expansion-parity tests, engine-build
  validation (unknown compartment is an error), and calibration write-back for
  exact-compartment initial-condition parameters via
  `Model.update_compartment_fractions`.
- GitHub Actions CI/CD pipelines for code quality, building, and PyPI releases
- MkDocs Material documentation site
- Comprehensive development workflow documentation
- Contributing guidelines and release process documentation

### Fixed

- The applied-parameters report now shows the effective head count for bin,
  stratification-category, and subgroup rows — the uniform expansion plus the
  declared `compartment_fractions` overrides — and reports each `fraction` as
  that head count over the declared `population_size`, so every row shares one
  denominator with `N`. A diverted aggregate row no longer reports the stale
  pre-calibration value as `calibrated`, and a bin left uncalibrated keeps its
  `null` marker even when overrides touch some of its compartments.
- A conditional-stratification category is calibrated as a fraction of its
  conditioning subgroup but reported as a fraction of the whole population, so
  the report rescales its `interval` by the subgroup's share of the population;
  the interval now brackets the reported `fraction`.
- `apply_calibration_parameters` is atomic: a validation failure after the first
  mutation restores the prior population and bin/stratification/compartment
  fractions. Diverted overrides are persisted in sorted order, so the model JSON
  is deterministic across `PYTHONHASHSEED` values. The write-back is a pure
  function of the current definition and the values and matches the engine's
  sequential replay, including pre-existing overrides: bin and unique-category
  applies are idempotent, while a category declared by two stratifications
  re-scores on re-apply exactly like a fresh engine problem. A bin value whose
  update would break the declared bin-fraction mapping is diverted to
  per-compartment overrides instead of relaxing the bin guard, and a
  non-conserving compartment override the engine scores is persisted and
  reloads through `model_validate_json` and `Model.from_json`.
- `population_size` must be greater than zero, reported as a domain `ValueError`
  rather than a later `ZeroDivisionError`. `update_bin_fractions` re-applies the
  sum-to-1/conservation rule.
- `initialize_population` rejects compartment overrides with a fraction outside
  `0.0..=1.0` or a duplicate name. It does not check the override total against
  `population_size`; the Python `ModelBuilder` and the public `update_*` methods
  do. The `InitialConditions` construction total guard is withdrawn so a reload
  of the scored state is accepted, and calibration write-back may persist a
  non-conserving compartment override the engine accepts (matching
  `set_initial_condition`). A non-conserving bin value is not persisted: it is
  diverted to per-compartment overrides, so the declared bin fractions keep
  their conservation rule while the replayed head counts are preserved.
- The PyO3 `InitialConditions` binding exposes `population_size`, `bin_fractions`,
  `stratification_fractions` and `compartment_fractions` with getters, and the
  `_commol_rs` stub declares only those real attributes (correcting the
  `bin_fractions` field name).
- An initial-condition override that names an unknown compartment now raises a
  `ValueError` from the Python construction and update paths and a
  `RuntimeError` from the `DifferenceEquations` engine constructor. The
  `update_compartment_fractions` checks validate the post-upsert compartment
  total before storing, so a rejected call leaves the model unchanged.
- Calibration write-back replays aggregate and stratification-category values
  in problem parameter order over the live population, so a reload reproduces
  scores that combine bin, category, and per-compartment overrides. The
  category replay snapshots pair totals once per parameter like the Rust engine,
  and a category shared by two stratifications is diverted to overrides replayed
  from the live population, so a repeated apply re-scores exactly like the
  engine instead of converging to a fixed point.
- The applied-parameters report memoizes per-category compartment pairs and the
  duplicate-compartment check is linear, removing the cubic and quadratic
  blow-ups on models with many compartments and overrides.

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

- **0.0.1** - Initial release with core functionality
- **Unreleased** - Current development version

For detailed information about each release, see the [GitHub Releases](https://github.com/MUNQU/commol/releases) page.
