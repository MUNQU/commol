# Probabilistic Calibration API

!!! info "Unified API"
Probabilistic calibration is accessed through the `Calibrator` class using the `run_probabilistic()` method. See the [Calibrator API](calibrator.md) for the complete API reference.

## Quick Reference

```python
from commol import Calibrator, CalibrationProblem, ProbabilisticCalibrationConfig

# Configure probabilistic calibration in the problem
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_function="sse",
    optimization_config=pso_config,
    probabilistic_config=ProbabilisticCalibrationConfig(n_runs=20),
)

# Use the unified Calibrator with run_probabilistic()
calibrator = Calibrator(simulation, problem)
result = calibrator.run_probabilistic()
```

## Related Classes

### ProbabilisticCalibrationConfig

::: commol.context.probabilistic_calibration.ProbabilisticCalibrationConfig
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### ProbabilisticCalibrationResult

::: commol.context.probabilistic_calibration.ProbabilisticCalibrationResult
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### ParetoSolution

::: commol.context.probabilistic_calibration.ParetoSolution
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### ParameterSetStatistics

::: commol.context.probabilistic_calibration.ParameterSetStatistics
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### CalibrationEvaluation

::: commol.context.probabilistic_calibration.CalibrationEvaluation
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

## Visualization

The `SimulationPlotter` class supports visualization of probabilistic calibration results with confidence intervals.

### Plotting with ProbabilisticCalibrationResult

When a `ProbabilisticCalibrationResult` is passed to `plot_series` or `plot_cumulative`, the plotter automatically displays:

- **Median predictions** as the main line
- **Confidence interval bands** as shaded regions
- **Observed data** as scatter points

```python
from commol import SimulationPlotter

# Create plotter with median predictions from selected ensemble
plotter = SimulationPlotter(simulation, result.selected_ensemble.prediction_median)

# Plot with confidence intervals
fig = plotter.plot_series(
    observed_data=observed_data,
    calibration_result=result,
    output_file="probabilistic_fit.png",
)

# Access the selected ensemble solution
selected = result.selected_ensemble
print(f"Ensemble size: {selected.ensemble_size}")
print(f"Coverage: {selected.coverage_percentage:.2f}%")
print(f"Parameter estimates: {selected.parameter_statistics}")

# Explore alternative solutions on the Pareto front
for i, solution in enumerate(result.pareto_front):
    print(f"Solution {i}: size={solution.ensemble_size}, "
          f"coverage={solution.coverage_percentage:.2f}%, "
          f"CI width={solution.average_ci_width:.4f}")
```

### SimulationPlotter.plot_series

::: commol.api.plotter.SimulationPlotter.plot_series
options:
show_root_heading: false
show_source: false
heading_level: 4

### SimulationPlotter.plot_cumulative

::: commol.api.plotter.SimulationPlotter.plot_cumulative
options:
show_root_heading: false
show_source: false
heading_level: 4

## Helper Classes

The probabilistic calibration workflow is orchestrated using focused helper classes:

### CalibrationRunner

Runs multiple calibrations in parallel with different random seeds.

::: commol.api.probabilistic.calibration_runner.CalibrationRunner
options:
show_root_heading: true
show_source: false
heading_level: 3
members: - run_multiple

### EvaluationProcessor

Handles deduplication, filtering, and clustering of calibration evaluations.

::: commol.api.probabilistic.evaluation_processor.EvaluationProcessor
options:
show_root_heading: true
show_source: false
heading_level: 3
members: - collect_evaluations - deduplicate - filter_by_loss_percentile - find_optimal_k - cluster_evaluations - select_representatives

### EnsembleSelector

Runs NSGA-II multi-objective optimization for ensemble selection.

::: commol.api.probabilistic.ensemble_selector.EnsembleSelector
options:
show_root_heading: true
show_source: false
heading_level: 3
members: - select_ensemble

### StatisticsCalculator

Computes ensemble statistics and predictions with confidence intervals.

::: commol.api.probabilistic.statistics_calculator.StatisticsCalculator
options:
show_root_heading: true
show_source: false
heading_level: 3
members: - calculate_parameter_statistics - generate_ensemble_predictions - calculate_prediction_intervals - calculate_coverage_metrics
