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

- An initial-condition override that names an unknown compartment now raises a
  `ValueError` from the Python construction and update paths and a
  `RuntimeError` from the `DifferenceEquations` engine constructor. The
  `update_compartment_fractions` checks validate the post-upsert compartment
  total before storing, so a rejected call leaves the model unchanged.
- Calibration write-back replays aggregate and stratification-category values
  in problem parameter order over the live population, so a reload reproduces
  scores that combine bin, category, and per-compartment overrides.

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
