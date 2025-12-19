# Probabilistic Calibration

Probabilistic calibration extends standard calibration by finding an _ensemble_ of parameter sets rather than a single best-fit solution. This approach quantifies parameter uncertainty and produces predictions with confidence intervals, providing a more complete picture of model uncertainty.

## Overview

Probabilistic calibration is a robust approach to parameter estimation that quantifies uncertainty by finding an ensemble of plausible parameter sets rather than a single "best" solution. This method acknowledges that multiple parameter combinations may fit the observed data equally well, providing a more complete picture of model uncertainty.

### The Workflow

The probabilistic calibration process follows a systematic workflow designed to explore the parameter space thoroughly and select a diverse, high-quality ensemble:

#### 1. Multiple Independent Calibration Runs

Execute many independent calibration runs, each with a different random initialization. This explores different regions of the parameter space and helps escape local optima. Each run uses your chosen optimization algorithm to minimize the loss function.

#### 2. Evaluation Collection

Collect all parameter evaluations from the optimization history of each run, not just the final solutions. This captures the entire search trajectory, including intermediate solutions that may offer valuable diversity.

#### 3. Deduplication and Quality Filtering

Remove duplicate parameter sets (within a tolerance) and optionally filter by loss, keeping only the top percentage high-quality solutions.

#### 4. Clustering in Parameter Space

Group similar parameter sets using K-means clustering. Each cluster represents a distinct region of parameter space where solutions behave similarly.

#### 5. Representative Selection from Clusters

Select diverse representatives from each cluster through a two-stage process:

**Stage 1: Elite Selection** - First, include a fraction of the best solutions by loss from each cluster. This ensures high-quality solutions are always represented in the ensemble, regardless of diversity considerations.

**Stage 2: Diversity Selection** - Then, select additional representatives using one of these specialized strategies:

- **Crowding distance**: NSGA-II style selection that explores parameter space boundaries, favoring solutions in sparse regions
- **Maximin distance**: Ensures uniform coverage with no boundary bias, maximizing minimum distances between selected points
- **Latin hypercube**: Stratified space-filling selection for optimal coverage, dividing parameter space into strata

The two-stage approach balances solution quality (elite selection) with diversity (spatial selection), ensuring the final representatives capture both performance and parameter space coverage.

#### 6. NSGA-II Ensemble Optimization

Use NSGA-II multi-objective optimization to select the final parameters set ensemble. NSGA-II treats ensemble selection as a combinatorial optimization problem, where each candidate solution represents a subset of representative parameter sets. The algorithm simultaneously optimizes two competing objectives by evaluating the population statistics that would result from each ensemble:

- **Minimize confidence interval width**: Select ensembles whose population statistics produce narrow confidence intervals, indicating precise predictions
- **Maximize data coverage**: Select ensembles whose prediction intervals span and capture the observed data points, ensuring the uncertainty estimates are realistic and reliable

The Pareto front contains ensembles that optimally balance these trade-offs. You can control the preference: favor narrow intervals (precision), high coverage (reliability), or a balanced trade-off.

#### 7. Statistical Analysis and Predictions

Calculate comprehensive statistics for each parameter (mean, median, standard deviation, percentiles) and generate predictions with confidence intervals across all time steps. Coverage metrics quantify how well the ensemble captures observed data.

Use probabilistic calibration when you need:

    - Uncertainty quantification for model predictions
    - Confidence intervals around parameter estimates
    - Coverage metrics for observed data
    - A more robust understanding of model behavior

## Basic Example

Here's a complete example of probabilistic calibration for an SIR model:

```python
from commol import (
    ModelBuilder,
    Simulation,
    ProbabilisticCalibrator,
    ProbabilisticCalibrationConfig,
    CalibrationProblem,
    CalibrationParameter,
    CalibrationParameterType,
    ObservedDataPoint,
    LossConfig,
    LossFunction,
    OptimizationConfig,
    OptimizationAlgorithm,
    ParticleSwarmConfig,
)
from commol.constants import ModelTypes

# Build model with parameters to be calibrated set to None
model = (
    ModelBuilder(name="SIR Model", version="1.0")
    .add_bin(id="S", name="Susceptible")
    .add_bin(id="I", name="Infected")
    .add_bin(id="R", name="Recovered")
    .add_parameter(id="beta", value=None)   # To be calibrated
    .add_parameter(id="gamma", value=None)  # To be calibrated
    .add_transition(
        id="infection",
        source=["S"],
        target=["I"],
        rate="beta * S * I / N"
    )
    .add_transition(
        id="recovery",
        source=["I"],
        target=["R"],
        rate="gamma * I"
    )
    .set_initial_conditions(
        population_size=1000,
        bin_fractions=[
            {"bin": "S", "fraction": 0.99},
            {"bin": "I", "fraction": 0.01},
            {"bin": "R", "fraction": 0.0}
        ]
    )
    .build(typology=ModelTypes.DIFFERENCE_EQUATIONS)
)

# Define observed data
observed_data = [
    ObservedDataPoint(step=0, compartment="I", value=10.0),
    ObservedDataPoint(step=10, compartment="I", value=45.2),
    ObservedDataPoint(step=20, compartment="I", value=78.5),
    ObservedDataPoint(step=30, compartment="I", value=62.3),
    ObservedDataPoint(step=40, compartment="I", value=38.1),
    ObservedDataPoint(step=50, compartment="I", value=18.7),
]

simulation = Simulation(model)

# Define parameters to calibrate
parameters = [
    CalibrationParameter(
        id="beta",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.3
    ),
    CalibrationParameter(
        id="gamma",
        parameter_type=CalibrationParameterType.PARAMETER,
        min_bound=0.0,
        max_bound=1.0,
        initial_guess=0.1
    ),
]

# Configure the calibration problem with probabilistic config
problem = CalibrationProblem(
    observed_data=observed_data,
    parameters=parameters,
    loss_config=LossConfig(function=LossFunction.SSE),
    optimization_config=OptimizationConfig(
        algorithm=OptimizationAlgorithm.PARTICLE_SWARM,
        config=ParticleSwarmConfig(
            num_particles=30,
            max_iterations=500,
        ),
    ),
    probabilistic_config=ProbabilisticCalibrationConfig(),  # Use defaults
    seed=42,  # For reproducibility across all stochastic processes
)

# Run probabilistic calibration
calibrator = ProbabilisticCalibrator(simulation, problem)
result = calibrator.run()
```

## Results

The `ProbabilisticCalibrationResult` contains the selected ensemble solution and the full Pareto front:

### Selected Ensemble

The optimal ensemble solution chosen based on your `pareto_preference` setting:

```python
# Access the selected ensemble
selected = result.selected_ensemble

print(f"Ensemble size: {selected.ensemble_size}")
print(f"Coverage: {selected.coverage_percentage:.1f}%")
print(f"Average CI width: {selected.average_ci_width:.4f}")

# Access individual parameter sets
for i, params in enumerate(selected.ensemble_parameters):
    print(f"Set {i}: beta={params['beta']:.4f}, gamma={params['gamma']:.4f}")
```

### Parameter Statistics

Statistics for each parameter across the selected ensemble:

```python
for param_name, stats in selected.parameter_statistics.items():
    print(f"{param_name}:")
    print(f"  Mean: {stats.mean:.4f}")
    print(f"  Std: {stats.std:.4f}")
    print(f"  Median: {stats.median:.4f}")
    print(f"  Range: [{stats.min:.4f}, {stats.max:.4f}]")
    print(f"  95% CI: [{stats.percentile_lower:.4f}, {stats.percentile_upper:.4f}]")
```

### Prediction Intervals

Median predictions and confidence interval bounds for each compartment:

```python
# Access predictions for the Infected compartment
median_I = selected.prediction_median["I"]
lower_I = selected.prediction_ci_lower["I"]
upper_I = selected.prediction_ci_upper["I"]

print(f"At step 30:")
print(f"  Median: {median_I[30]:.1f}")
print(f"  95% CI: [{lower_I[30]:.1f}, {upper_I[30]:.1f}]")
```

### Exploring the Pareto Front

The result contains all Pareto-optimal solutions, allowing you to explore different trade-offs:

```python
# Examine all solutions on the Pareto front
for i, solution in enumerate(result.pareto_front):
    print(f"Solution {i}:")
    print(f"  Ensemble size: {solution.ensemble_size}")
    print(f"  Coverage: {solution.coverage_percentage:.2f}%")
    print(f"  CI width: {solution.average_ci_width:.4f}")
    print(f"  Normalized objectives: CI={solution.ci_width:.4f}, Cov={solution.coverage:.4f}")

# The selected solution is one of them
print(f"\nSelected solution index: {result.selected_pareto_index}")
assert result.selected_ensemble == result.pareto_front[result.selected_pareto_index]
```

### Metadata

Information about the calibration process:

```python
print(f"Calibration runs performed: {result.n_runs_performed}")
print(f"Unique evaluations found: {result.n_unique_evaluations}")
print(f"Clusters identified: {result.n_clusters_used}")
print(f"Confidence level: {result.confidence_level}")
```

## Visualizing Results

The `SimulationPlotter` provides visualization support for probabilistic calibration results, displaying predictions with confidence interval bands.

### Plotting with Confidence Intervals

```python
from commol import SimulationPlotter

# Use predictions from the selected ensemble
plotter = SimulationPlotter(simulation, result.selected_ensemble.prediction_median)

# Plot with confidence intervals
plotter.plot_series(
    observed_data=observed_data,
    calibration_result=result,
    output_file="probabilistic_fit.png",
)
```

This creates a plot showing:

- **Median prediction**: Solid line representing the ensemble median
- **95% confidence interval**: Shaded band around the median
- **Observed data**: Red scatter points

### Cumulative Plots

For cumulative quantities (e.g., total infections over time):

```python
plotter.plot_cumulative(
    observed_data=observed_data,
    calibration_result=result,
    output_file="probabilistic_cumulative.png",
)
```

### Customizing Plots

You can customize the appearance using Seaborn styles:

```python
from commol import PlotConfig, SeabornStyle, SeabornContext

config = PlotConfig(
    figsize=(12, 8),
    dpi=150,
)

plotter.plot_series(
    observed_data=observed_data,
    calibration_result=result,
    config=config,
    seaborn_style=SeabornStyle.WHITEGRID,
    context=SeabornContext.TALK,
)
```

### Plotting Specific Compartments

```python
# Plot only specific compartments
plotter.plot_series(
    observed_data=observed_data,
    calibration_result=result,
    bins=["I", "R"],  # Only Infected and Recovered
)
```

## Configuration Options

Probabilistic calibration is configured through the `CalibrationProblem.probabilistic_config` field using a `ProbabilisticCalibrationConfig` object. The configuration includes:

- **n_runs**: Number of independent calibration runs
- **evaluation_processing**: Deduplication and filtering settings
- **clustering**: Clustering parameters
- **representative_selection**: Selecting diverse solutions from clusters
- **ensemble_selection**: NSGA-II ensemble optimization
- **confidence_level**: Confidence interval level (e.g., 0.95 for 95% CI)

The random seed is set at the **CalibrationProblem level** (`problem.seed`), not within the configuration.

### Core Parameters

#### n_runs (int, default: 10)

Number of independent calibration runs to perform. Each run explores the parameter space from a different random initialization.

- **Lower values 1-5)**: Faster execution, less thorough exploration, may miss solution regions
- **Medium values (6-20)**: Balanced trade-off, good for most applications
- **Higher values (20+)**: More thorough exploration, better uncertainty quantification, longer runtime

**When to adjust**: Increase for complex models with many local optima or when initial results show poor coverage. Decrease for rapid prototyping or simple models.

#### confidence_level (float, default: 0.95)

Confidence level for prediction and parameter intervals (e.g., 0.95 = 95% confidence intervals).

- **Lower values (0.80-0.90)**: Narrower intervals, less conservative uncertainty estimates
- **Standard value (0.95)**: Commonly used in scientific practice, balances precision and reliability
- **Higher values (0.99)**: Wider intervals, more conservative for critical decisions

**When to adjust**: Use higher values (0.99) for safety-critical applications or regulatory compliance. Use lower values (0.90) when you need tighter bounds and can tolerate more uncertainty.

### Evaluation Processing Parameters (ProbEvaluationFilterConfig)

**deduplication_tolerance** (float, default: 1e-6)

Absolute tolerance for identifying duplicate parameter sets. Two parameter sets are considered identical if all parameters differ by less than this value.

- **Smaller values (1e-8 to 1e-10)**: Stricter deduplication, keeps more near-duplicate solutions
- **Default value (1e-6)**: Good balance for most numerical precision scenarios
- **Larger values (1e-4 to 1e-3)**: Aggressive deduplication, may remove legitimately different solutions

**When to adjust**: Decrease if you're getting "too few evaluations" errors. Increase if you suspect many near-duplicates are wasting computational resources.

**loss_percentile_filter** (float, default: 1.0, range: (0.0, 1.0])

Fraction of best solutions by loss to retain before clustering. For example, 0.3 keeps only the best 30% of evaluations.

- **Low values (0.1-0.3)**: Aggressive filtering, mostly elite solutions
- **Medium values (0.5-0.7)**: Removes clearly poor solutions while keeping diversity
- **High values (0.9-1.0)**: Minimal filtering, maximum diversity

**When to adjust**: Decrease to tighten confidence intervals at the cost of coverage. Increase if clustering struggles or coverage is poor.

**min_evaluations_required** (int, default: 5)

Minimum number of unique evaluations required after deduplication. Calibration fails if fewer remain.

**When to adjust**: Decrease for quick tests with few runs. Increase to ensure statistical validity for production use.

### Clustering Parameters (ProbClusteringConfig)

**n_clusters** (int | None, default: None)

Number of clusters to use. If `None`, optimal number is determined automatically using silhouette analysis.

- **None (automatic)**: Finds optimal number based on silhouette scores, adapts to data structure
- **Fixed value**: Forces specific number of clusters, useful when you know the solution structure

**When to adjust**: Set explicitly if you have prior knowledge about distinct solution regions. Use automatic for exploratory analysis.

**min_evaluations_for_clustering** (int, default: 100)

Minimum evaluations needed to perform clustering. Below this threshold, all evaluations go into a single cluster.

**silhouette_threshold** (float, default: 0.2, range: [-1.0, 1.0])

Minimum silhouette score for beneficial clustering. Scores near 0 indicate overlapping clusters; higher scores indicate well-separated clusters.

**silhouette_excellent_threshold** (float, default: 0.8)

Early stopping threshold for silhouette search. If a cluster number achieves this score, search stops.

**identical_solutions_atol** (float, default: 1e-10)

Absolute tolerance for detecting identical solutions. Used to detect when there's no variance in parameter space.

**kmeans_max_iter** (int, default: 100)

Maximum iterations for K-means algorithm.

**kmeans_algorithm** ("lloyd" | "elkan" | "auto" | "full", default: "elkan")

K-means algorithm variant. "elkan" is typically faster for dense data.

### Representative Selection Parameters (ProbRepresentativeConfig)

**max_representatives** (int, default: 1000)

Maximum total representatives across all clusters to select as candidates for NSGA-II ensemble selection.

- **Lower values (500-800)**: Faster NSGA-II execution, may miss diversity
- **Default value (1000)**: Good balance for most problems
- **Higher values (1500+)**: More thorough ensemble optimization, slower

**When to adjust**: Decrease if NSGA-II is too slow. Increase for very complex parameter spaces with many clusters.

**percentage_elite_cluster_selection** (float, default: 0.1, range: [0.0, 1.0])

Fraction of best solutions by loss to include from each cluster before diversity selection (elite selection stage).

- **0.0**: Pure diversity selection, no quality bias
- **0.1-0.3**: Balanced, ensures some high-quality solutions
- **0.5-1.0**: Strong quality bias, diversity may suffer

**cluster_representative_strategy** ("proportional" | "equal", default: "proportional")

How to distribute representatives across clusters:

- **"proportional"**: Larger clusters get more representatives (proportional to size)
- **"equal"**: All clusters get equal representatives (ensures small clusters aren't ignored)

**cluster_selection_method** ("crowding_distance" | "maximin_distance" | "latin_hypercube", default: "latin_hypercube")

Strategy for selecting diverse representatives within each cluster:

- **"latin_hypercube"**: Stratified sampling, divides space into strata for optimal coverage
- **"crowding_distance"**: NSGA-II style, favors boundary solutions and sparse regions
- **"maximin_distance"**: Maximizes minimum distance between points, uniform coverage

**quality_temperature** (float, default: 1.0) - _Only for "maximin_distance"_

Controls quality vs diversity trade-off in maximin selection:

- **Low values (0.1-0.5)**: Strong preference for low-loss solutions
- **Medium values (1.0-2.0)**: Balanced trade-off
- **High values (5.0+)**: Prioritize diversity over quality

**k_neighbors_min** (int, default: 5)

Minimum k for k-nearest neighbors in density estimation.

**k_neighbors_max** (int, default: 10)

Maximum k for k-nearest neighbors in density estimation.

**sparsity_weight** (float, default: 2.0)

Exponential weight for sparsity bonus in maximin selection. Higher values give stronger preference for solutions in sparse regions.

**stratum_fit_weight** (float, default: 10.0)

Weight for stratum fit vs quality in latin_hypercube selection. Higher values prioritize space-filling over quality.

### Ensemble Selection Parameters (ProbEnsembleConfig)

**ensemble_size_mode** ("automatic" | "fixed" | "bounded", default: "automatic")

How to determine ensemble size:

- **"automatic"**: NSGA-II optimizes size within min/max bounds
- **"fixed"**: Use exact `ensemble_size` value
- **"bounded"**: NSGA-II optimizes within specified range

**ensemble_size** (int | None, required for "fixed" mode)

Fixed ensemble size when mode="fixed". Must be ≥ 2.

**ensemble_size_min** (int | None, required for "bounded" mode)

Minimum ensemble size when mode="bounded". Must be ≥ 2. Not used for "automatic" or "fixed" modes.

**ensemble_size_max** (int | None, required for "bounded" mode)

Maximum ensemble size when mode="bounded". Must be ≥ ensemble_size_min. Not used for "automatic" or "fixed" modes.

**nsga_population_size** (int, default: 100, must be > 3)

NSGA-II population size. More individuals explore more ensemble combinations but increase runtime.

**nsga_generations** (int, default: 100)

Number of NSGA-II generations. More generations improve solution quality but increase runtime.

**nsga_crossover_probability** (float, default: 0.9, range: [0.0, 1.0])

Probability of crossover in NSGA-II. Higher values encourage more exploration.

**pareto_preference** (float, default: 0.5, range: [0.0, 1.0])

Preference for selecting from Pareto front:

- **0.0**: Minimize CI width (narrow intervals, precise but may underestimate uncertainty)
- **0.5**: Balanced trade-off (default, suitable for most applications)
- **1.0**: Maximize coverage (wide intervals, reliable but less precise)

**When to adjust**: Use lower values (0.2) for applications requiring precision. Use higher values (0.8) for safety-critical applications requiring conservative uncertainty estimates.

**ci_margin_factor** (float, default: 0.1, range: [0.0, 1.0])

Safety margin for CI width bounds normalization. Adds a buffer (e.g., 10%) to min/max CI width bounds to avoid numerical edge cases during optimization. Higher values provide more conservative bounds.

**ci_sample_sizes** (list[int], default: [10, 20, 50, 100])

Sample sizes for CI width estimation. Used to explore how CI width varies with ensemble size when computing maximum CI width bounds. The algorithm tests random ensembles of these sizes to find the widest possible CI, ensuring proper normalization.

## Usage Example

```python
from commol import (
    ProbabilisticCalibrationConfig,
    ProbEvaluationFilterConfig,
    ProbClusteringConfig,
    ProbRepresentativeConfig,
    ProbEnsembleConfig,
)

problem.probabilistic_config = ProbabilisticCalibrationConfig(
    n_runs=100,
    confidence_level=0.95,
    evaluation_processing=ProbEvaluationFilterConfig(
        loss_percentile_filter=0.5,
        deduplication_tolerance=1e-6,
    ),
    clustering=ProbClusteringConfig(
        n_clusters=None,  # Automatic
    ),
    representative_selection=ProbRepresentativeConfig(
        max_representatives=1000,
        cluster_selection_method="crowding_distance",
    ),
    ensemble_selection=ProbEnsembleConfig(
        ensemble_size_mode="automatic",
        pareto_preference=0.5,
    ),
)
problem.seed = 42
```

## Comparison with Standard Calibration

| Aspect      | Standard Calibration      | Probabilistic Calibration  |
| ----------- | ------------------------- | -------------------------- |
| Output      | Single best parameter set | Ensemble of parameter sets |
| Uncertainty | None                      | Confidence intervals       |
| Predictions | Point estimates           | Distributions with CIs     |
| Coverage    | Not applicable            | Quantified                 |
| Computation | Fast                      | More intensive             |
| Use case    | Point estimates           | Uncertainty quantification |

## See Also

- [Model Calibration](calibration.md) - Standard single-solution calibration
- [API Reference](../api/probabilistic-calibrator.md) - Complete ProbabilisticCalibrator API
- [Examples](examples.md) - Additional examples
