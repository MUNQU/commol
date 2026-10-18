# Uncertainty

[Back to Home](../../index.md)

A calibration gives one set of parameters. This section shows how far you can
trust it: which parameters the data identify, how the inputs move the outputs,
and whether the intervals you report cover what they claim. This page defines
what every method returns and reads: a weighted **parameter sample**, the
**predictive** of a sample, and the **scores** that check it against data.
Each method has its own page, listed in the table below.

## The three questions

| Question | What answers it |
|---|---|
| Which parameters fit the data? | The [identifiability](identifiability.md) methods; a sample of the parameters, or a profile of the loss; several starts that end apart with equal losses show that the data do not identify the parameters |
| How do the inputs move the outputs? | The predictive of a sample: the run of every draw, and its band |
| Are the intervals calibrated? | Scores of the predictive against observations the method did not see, and coverage studies |

## A band of the model, or a prediction interval

A **band of the model** is the spread of the model's output across the draws of
the parameters. It says where the model curve lies, not where the next
observation will fall: an observation also carries measurement noise. A
**prediction interval** adds that noise to every draw before it reduces them
(the `ObservationNoise` port of `commol-inference`). Every reader of
observations takes the noise as an argument with no default: `NoNoise()` gives
a band of the model, and `ResidualNoise(...)` adds residuals of a fit, which
gives a prediction interval with no noise model
([Resampling](resampling.md#prediction-intervals)). Comparing a band of the
model with data underestimates the spread, so its coverage of the data is below
its level.

## Which method answers which question

| Method | Question | Needs a noise model | Page |
|---|---|---|---|
| Several starts, `Calibrator.run(starts=n)` | Which parameters fit the data: distinct ends with equal losses | No | [Optimizers](../calibration-optimizers.md#several-starts) |
| `StructuralRank` | Which parameters any data of this design can determine, and which combinations | No | [Identifiability](identifiability.md#structural-rank) |
| `Collinearity` | Which parameters can be fitted together | No | [Identifiability](identifiability.md#collinearity) |
| `SampleIdentifiability` | Which parameters a sample determines: contraction, correlations, modes | No | [Identifiability](identifiability.md#a-sample) |
| `StartsIdentifiability` | Which parameters differ between starts of equal loss | No | [Identifiability](identifiability.md#starts) |
| `BayesianBootstrap` | How far the parameters move when the data are reweighted by blocks | No | [Resampling](resampling.md#bayesian-bootstrap) |
| `ResidualBootstrap` | How far the parameters move when the residuals of the fit are resampled by blocks | No | [Resampling](resampling.md#residual-block-bootstrap) |
| `WildBootstrap` | The same, when the size of the noise changes along the series | No | [Resampling](resampling.md#wild-block-bootstrap) |
| `BlockJackknife` | The standard error and bias of each parameter, and which block of data moves it most | No | [Resampling](resampling.md#block-jackknife) |
| `ResidualNoise` | Where the next observations fall: a prediction interval | No | [Resampling](resampling.md#prediction-intervals) |
| `GibbsPosterior` | A posterior of the parameters from the loss and priors, with a scale `η` of the loss | No | [Sequential Monte Carlo](smc.md#gibbs-posterior) |
| `AbcSmc` | A posterior of the parameters whose runs come within a tolerance of the data, with priors | No | [Sequential Monte Carlo](smc.md#abc) |
| `AbcRejection` | The same, from the closest draws of the priors | No | [Sequential Monte Carlo](smc.md#abc) |

Each uncertainty method adds its row here when it lands.
## Quick path

```python
from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    NoNoise,
    ObservedDataPoint,
    ParameterSample,
    Predictive,
    SampleMethod,
    Simulation,
    scores,
)


def basic_model(k1: float | None, k2: float | None):
    return (
        ModelBuilder(name="Basic Model")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="k2", value=k2)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A * B / total")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(total=1000, bin_fractions={"A": 0.99, "B": 0.01, "C": 0.0}),
            engine="DifferenceEquations",
        )
    )


truth = Simulation(basic_model(0.4, 0.1)).run(num_steps=60)
observations = [
    ObservedDataPoint(at=step, output="B", value=truth["B"][step])
    for step in range(5, 61, 5)
]
# Fit the first 40 steps; hold out the rest to check the predictive.
fitted = [point for point in observations if point.at <= 40]
held_out = [point for point in observations if point.at > 40]

problem = CalibrationProblem(
    observed_data=fitted,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=42,
)
calibrator = Calibrator(Simulation(basic_model(None, None)), problem)

# A sample of the parameters. A method returns one; here it is written by hand.
sample = ParameterSample.equally_weighted(
    names=["k1", "k2"],
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    values=[[0.38, 0.09], [0.40, 0.10], [0.42, 0.11]],
    method=SampleMethod(name="by hand"),
)
print(sample.interval(0.9)["k1"])  # (0.38, 0.42)

# The predictive: every draw run to step 60, and its band of the model.
predictive = Predictive(calibrator, sample, positions=list(range(61)))
band = predictive.band(confidence_level=0.9)
print(band.lower["B"][50] <= truth["B"][50] <= band.upper["B"][50])  # True

# Scores of the band of the model on the held-out observations.
print(scores.coverage(predictive, held_out, NoNoise(), confidence_level=0.9).mean)
print(scores.crps(predictive, held_out, NoNoise()).per_series)
```

## The parameter sample

A `ParameterSample` holds draws of the parameters of one problem, each with a
weight. Every method with draws returns one; the plots, the reports, the
predictive and the scores read it, never the result of a method.

| Field | Meaning |
|---|---|
| `names` | One name per column, in the order of the problem's parameters |
| `bounds` | The bounds of each parameter; every value lies inside them |
| `values` | One row per draw |
| `log_weights` | The log weight of each draw, normalized so the weights sum to 1; `-inf` for a weight of zero |
| `groups` | The chain, replicate or wave of each draw, or `None` |
| `method` | The name and settings of the method that made it (`SampleMethod`) |

Every statistic is weighted and computed in Rust. With equal weights it is the
unweighted statistic.

| Member | Returns |
|---|---|
| `mean()` | The weighted mean of each parameter |
| `covariance()`, `correlation()` | Weighted matrices, in the order of `names`; with equal weights, NumPy's `cov` and `corrcoef` |
| `quantiles(levels)`, `interval(confidence_level)` | Weighted quantiles by [the one convention](#the-one-quantile) |
| `effective_size()` | Kish's effective sample size, `1 / Σ wᵢ²` |
| `draws()` | One `{name: value}` mapping per draw |
| `resample(count, seed)` | A sample of equal weights, drawn by weight |
| `summary(confidence_level)` | A `ParameterSummary` per parameter, for a report |

```python
print(round(sample.effective_size(), 6))  # 3.0
print(next(sample.draws()))  # {'k1': 0.38, 'k2': 0.09}
```

A sample has no single set of parameters, so
`Model.apply_calibration_parameters` rejects it with a `TypeError`; apply one of
`sample.draws()`. A sample saves to JSON and loads back equal
(`model_dump_json`, `model_validate_json`).

### The one quantile

Every interval of commol is a weighted quantile: the inverse of the weighted
empirical distribution, interpolated between the midpoints of the cumulative
weights. With equal weights it equals `numpy.percentile(..., method="hazen")`.
A draw of weight zero does not count. `weighted_quantiles` computes it for any
values, and `member_statistics` reduces ensemble members with it.

```python
from commol import weighted_quantiles

print(weighted_quantiles([0.0, 1.0], levels=[0.5], weights=[1.0, 3.0]))  # [0.75]
```

## The predictive

`Predictive(calibrator, sample, positions)` runs every draw at `positions`, in
parallel in Rust. With a stochastic engine every replicate of a draw is a draw
of the predictive, with an equal share of its weight.

| Member | Returns |
|---|---|
| `band(confidence_level)` | The weighted mean, median and central interval of every output at every position (`PredictiveBand`) |
| `series_band(outputs, confidence_level, cumulative=False)` | The band of the sum of some outputs, or of its running total; each draw is reduced first |
| `at_observations(observed_data, noise)` | Every draw's value of each observation, read through the observation operator the loss uses (sums, windows and factors), with `noise` drawn from the calibrator's seed: `NoNoise()` or `ResidualNoise(...)` (`ObservationDraws`) |
| `runs()`, `weights` | The trajectory and weight of every draw |

`SimulationPlotter.plot_series` and `plot_cumulative` draw the band of a
predictive when it is the `calibration_result`, and `simulation_report` reduces
its draws with their weights when it is the `ensemble_runs`.
`applied_parameters_report` takes the sample itself (`sample=`, with a
`confidence_level`).

## Scores

The functions of `commol.scores` compare a predictive with observations, and
return a `Score`: the value of each observation, the mean of each series and
the overall mean. Which observations to score and the observation noise are
arguments with no default: score data the method did not see, such as the
held-out positions above, and say whether you score a band of the model
(`NoNoise()`) or a prediction interval (`ResidualNoise(...)`).

| Function | Score | Better |
|---|---|---|
| `coverage(predictive, observations, noise, confidence_level)` | 1 inside the central interval, 0 outside; the mean is the empirical coverage | Close to the level |
| `interval_score(predictive, observations, noise, confidence_level)` | `(u − l) + (2/α)(l − y)₊ + (2/α)(y − u)₊`, `α = 1 − c` | Lower |
| `weighted_interval_score(predictive, observations, noise, confidence_levels)` | The weighted interval score of Bracher et al. (2021) | Lower |
| `crps(predictive, observations, noise)` | `Σ wᵢ|xᵢ − y| − ½ Σᵢⱼ wᵢwⱼ|xᵢ − xⱼ|`, the CRPS of a stochastic calibration's loss | Lower |
| `pit(predictive, observations, noise, seed)` | `F(y)`, randomized where draws tie with `y` | Uniform |

## The entry point

Every method runs through `Calibrator.uncertainty(method)`. A method is a
configuration, a subclass of `UncertaintyMethod[R]` whose type parameter `R` is
its result; the call returns an `R`, which a static type checker infers from
the configuration's type. A configuration with no registered runner raises a
`ValueError` that names the registered ones.

The identifiability methods run through their own entry point,
`Calibrator.identifiability(method)`, with a subclass of
`IdentifiabilityMethod[R]`: they answer another question
([Identifiability](identifiability.md)). Both kinds subclass
`InferenceMethod[R]` and share one registry; each entry point refuses the
other kind with a `TypeError` that names the right one.

```python
from commol import BayesianBootstrap

result = calibrator.uncertainty(BayesianBootstrap(replicates=20, block_length=1))
print(type(result).__name__, len(result.sample))  # BootstrapResult 20
```

## Checklist

- [ ] You report a band of the model as a band of the model, not as a prediction interval.
- [ ] You score observations the method did not see.
- [ ] The effective size of a weighted sample is large enough for the intervals you read.

## Next steps

| To learn | Read |
|---|---|
| Bootstraps, the jackknife and prediction intervals with no noise model | [Resampling](resampling.md) |
| Posteriors with no noise model: the Gibbs posterior and ABC | [Sequential Monte Carlo](smc.md) |
| Several starts of an optimizer | [Optimizers](../calibration-optimizers.md#several-starts) |
| Which parameters the data determine | [Identifiability](identifiability.md) |
| Reports of a sample | [Reports](../reports.md) |
| Plots of a predictive | [Running Simulations](../simulations.md#plotting) |
