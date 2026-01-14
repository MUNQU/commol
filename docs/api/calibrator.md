# Calibrator API

The `Calibrator` class provides a unified interface for both standard and probabilistic calibration.

## Methods

- `run()` - Runs standard single-solution calibration using the `optimization_config` from the problem
- `run_probabilistic()` - Runs probabilistic calibration using the `probabilistic_config` from the problem

::: commol.api.calibrator.Calibrator
options:
show_root_heading: true
show_source: true
heading_level: 2
show_docstring_attributes: false

## Related Classes

### CalibrationProblem

::: commol.context.calibration.CalibrationProblem
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### CalibrationResult

::: commol.context.calibration.CalibrationResult
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### CalibrationParameter

::: commol.context.calibration.CalibrationParameter
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### ObservedDataPoint

::: commol.context.calibration.ObservedDataPoint
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### OptimizationConfig

!!! info "Type Alias"
`OptimizationConfig` is a type alias for `NelderMeadConfig | ParticleSwarmConfig`.

### NelderMeadConfig

::: commol.context.calibration.NelderMeadConfig
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### ParticleSwarmConfig

::: commol.context.calibration.ParticleSwarmConfig
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

## Enumerations

### LossFunction

::: commol.context.calibration.LossFunction
options:
show_root_heading: true
show_source: false
heading_level: 3
members: true

### OptimizationAlgorithm

::: commol.context.calibration.OptimizationAlgorithm
options:
show_root_heading: true
show_source: false
heading_level: 3
members: true
