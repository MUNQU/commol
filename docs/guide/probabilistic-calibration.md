# Probabilistic Calibration

Probabilistic calibration estimates uncertainty from an ensemble of calibrated
parameter sets. Both selection backends use compact, observation-aware
predictions that include the configured aggregates, windows, and scales.

Any configured `loss_function` (`sse`, `weighted_sse`, `rmse`, or `mae`) works
end-to-end. The fit-gated selection scores the ensemble median prediction with
the same loss the members were optimized with — the central loss and the
member losses it is compared against are always measured the same way. As with
the optimizer, `sse`/`weighted_sse` apply observation weights while `rmse`/`mae`
do not. Use `normalize_observations=True` on the problem to keep series of very
different magnitudes comparable within whichever loss you choose.

## Workflow

1. Run multiple independent calibrations.
2. Deduplicate optimizer evaluations and retain a core near-optimal loss pool.
   When configured, keep a controlled wider-loss tail for candidates that add
   new observed-prediction shapes.
3. Cluster candidates in parameter space or in transformed observed-prediction
   space.
4. Select diverse representatives from each cluster.
5. Select the final ensemble. NSGA-II is the default and returns a Pareto front
   balancing interval width and observed-data coverage. The
   `greedy_local_search` backend uses a fit-gated beam search with temporary
   bridges and preserves the central fit under the configured loss.
6. Calculate member-consistent prediction intervals. Aggregates, observation
   windows, and scale parameters are applied per member before percentiles are
   calculated.

## Recommended configuration

```python
from commol.context.probabilistic_calibration import (
    ProbClusteringConfig,
    ProbEvaluationFilterConfig,
    ProbGreedyLocalSearchConfig,
    ProbNsga2Config,
    ProbabilisticCalibrationConfig,
    ProbRepresentativeConfig,
)

problem.probabilistic_config = ProbabilisticCalibrationConfig(
    n_runs=75,
    evaluation_processing=ProbEvaluationFilterConfig(
        evaluation_retention="top_k_per_run",
        top_k_per_run=100,
        max_loss_ratio=1.25,
        tail_max_loss_ratio=2.0,
        tail_max_representatives=50,
    ),
    clustering=ProbClusteringConfig(
        feature_space="observed_predictions",
    ),
    representative_selection=ProbRepresentativeConfig(
        max_representatives=1000,
        percentage_elite_cluster_selection=0.5,
        cluster_selection_method="maximin_distance",
    ),
    ensemble_selection=ProbNsga2Config(
        ensemble_size_mode="bounded",
        ensemble_size_min=15,
        ensemble_size_max=30,
        population_size=100,
        generations=100,
        pareto_preference=0.5,
    ),
)
```

The default is `ProbNsga2Config`. To use the fit-gated greedy backend, replace
it with `ProbGreedyLocalSearchConfig` and configure
`central_fit_max_loss_ratio` and `search_beam_width`:

```python
ensemble_selection=ProbGreedyLocalSearchConfig(
    ensemble_size_mode="bounded",
    ensemble_size_min=15,
    ensemble_size_max=30,
    central_fit_max_loss_ratio=1.25,
    search_beam_width=32,
)
```

Increase `n_runs`, particles, and optimizer iterations before relaxing the loss
gates. If the final intervals are narrow because all near-optimal candidates
have almost identical predictions, prefer a small prediction-novel tail over
raising `max_loss_ratio` globally. A broader interval is only useful when it is
produced by candidates that still fit the data. With the greedy backend, if the
requested ensemble size cannot satisfy the central fit gate, reduce the size or
improve the calibration runs; increasing the gate is an explicit decision to
permit a worse central fit.

## Reading the result

```python
result = Calibrator(simulation, problem).run_probabilistic()
ensemble = result.selected_ensemble

print(ensemble.point_loss)
print(ensemble.central_loss)
print(ensemble.observation_diagnostics)
print(ensemble.selection_diagnostics)
```

`point_parameters` is the only parameter dictionary suitable for building a
single model. It comes from the lowest-loss selected member. The contents of
`parameter_statistics` describe the ensemble and must not be combined into a
parameter-wise median model, because that combination may never have been
calibrated.

Use `observation_diagnostics` to check every observed series. Aggregate coverage
can hide a poor fit for a small but important series.

Use `selection_algorithm`, `pareto_front`, and `selected_pareto_index` to inspect
NSGA-II's compact trade-off summaries. For greedy selection,
`selection_diagnostics` distinguishes a weak candidate pool from a tight
central-fit gate; it reports the candidate count, evaluated subsets, maximum
feasible size, rejected singleton additions, and the best coverage encountered
during the beam search.

## Computing your own ensemble statistics

`ci_percentiles` maps a confidence level to the pair of percentile points used
for every interval in the result:

```python
from commol import ci_percentiles, member_statistics

lower_point, upper_point = ci_percentiles(0.95)  # (2.5, 97.5)
```

`member_statistics` reduces one quantity across the ensemble members and returns
`mean`, `median`, `ci_lower`, `ci_upper`, `min` and `max`:

```python
final_values = [run["A"][-1] for run in ensemble_runs]
stats = member_statistics(final_values, result.confidence_level)
```

Members may be scalars, as above, or equal-length series, in which case every
statistic is a series reduced across members position by position.

!!! warning "Reduce each member first"

    Compute the quantity you care about for each member, then pass those values
    to `member_statistics`. Taking percentile bands of the raw trajectories and
    reducing the bands afterwards gives a difference of percentiles rather than
    a percentile of members: an interval that no member of the ensemble
    realizes, and generally a wider one.

    Windowed quantities follow the same rule and are available directly as
    `windowed_prediction_median`, `windowed_prediction_ci_lower` and
    `windowed_prediction_ci_upper`. Each value is taken at the step of a
    windowed observation, using that observation's own `window_steps`;
    `windowed_prediction_steps` lists those steps.
