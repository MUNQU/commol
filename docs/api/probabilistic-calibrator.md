# Probabilistic Calibration API

Probabilistic calibration is invoked through `Calibrator.run_probabilistic()`.
The default ensemble selector is NSGA-II; the fit-gated greedy local search is
selected by passing `ProbGreedyLocalSearchConfig`.

Any `loss_function` (`sse`, `weighted_sse`, `rmse`, `mae`) is supported. The
central-fit gate scores the ensemble median with the same loss the members were
fit with, so both sides of the gate are always comparable. Set
`normalize_observations=True` on the problem to keep series of different
magnitudes comparable within that loss.

```python
from commol import Calibrator, CalibrationProblem
from commol.context.probabilistic_calibration import (
    ProbClusteringConfig,
    ProbEvaluationFilterConfig,
    ProbGreedyLocalSearchConfig,
    ProbNsga2Config,
    ProbabilisticCalibrationConfig,
)

problem.probabilistic_config = ProbabilisticCalibrationConfig(
    n_runs=50,
    evaluation_processing=ProbEvaluationFilterConfig(
        evaluation_retention="top_k_per_run",
        top_k_per_run=100,
        max_loss_ratio=1.25,
        tail_max_loss_ratio=2.0,
        tail_max_representatives=50,
    ),
    clustering=ProbClusteringConfig(feature_space="observed_predictions"),
    ensemble_selection=ProbNsga2Config(
        ensemble_size_mode="bounded",
        ensemble_size_min=15,
        ensemble_size_max=30,
        population_size=100,
        generations=100,
        pareto_preference=0.5,
    ),
)
result = Calibrator(simulation, problem).run_probabilistic()
```

Use `ProbGreedyLocalSearchConfig` instead when the central-fit gate and beam
search controls are required:

```python
ensemble_selection=ProbGreedyLocalSearchConfig(
    ensemble_size_mode="bounded",
    ensemble_size_min=15,
    ensemble_size_max=30,
    central_fit_max_loss_ratio=1.25,
    search_beam_width=32,
)
```

`max_loss_ratio` defines the core near-optimal pool. When
`tail_max_loss_ratio` and `tail_max_representatives` are set, candidates outside
the core pool but inside the wider loss band can be added only if they increase
prediction-space diversity. `feature_space="observed_predictions"` clusters the
actual calibrated quantities, including observation windows, aggregates, and
member-specific scales. The greedy selector uses a bounded beam search,
retaining temporary bridge subsets so complementary interval tails can be
evaluated together. It maximizes minimum per-series coverage and breaks ties by
interval width. It never accepts an ensemble whose memberwise-median loss
exceeds `central_fit_max_loss_ratio` times the best member's loss. NSGA-II
returns compact Pareto summaries balancing interval width and observed-data
coverage; `pareto_preference` chooses the reported solution.

Use `result.selected_ensemble.point_parameters` when a single parameterized
model is needed. These parameters belong to a real, lowest-loss ensemble
member. Parameter-wise medians are descriptive statistics only and must not be
combined into a new model.

`observation_diagnostics` reports coverage and average interval width separately
for each observed series. This should be inspected alongside aggregate coverage.
`selection_diagnostics` reports the candidate and subset-search limits that led
to the final ensemble.

`result.selection_algorithm` identifies the backend used. For NSGA-II,
`result.pareto_front` contains lightweight summaries and
`result.selected_pareto_index` identifies the selected summary. Greedy selection
leaves the Pareto summary fields unset.

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

### ProbNsga2Config

::: commol.context.probabilistic_calibration.ProbNsga2Config
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### ProbGreedyLocalSearchConfig

::: commol.context.probabilistic_calibration.ProbGreedyLocalSearchConfig
options:
show_root_heading: true
show_source: false
heading_level: 3
show_docstring_attributes: true

### EnsembleSolution

::: commol.context.probabilistic_calibration.EnsembleSolution
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
