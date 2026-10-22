# Uncertainty

[Back to Home](https://munqu.github.io/commol/latest/index.md)

A calibration gives one set of parameters. This section shows how far you can trust it: which parameters the data identify, how the inputs move the outputs, and whether the intervals you report cover what they claim. This page defines what every method returns and reads: a weighted **parameter sample**, the **predictive** of a sample, and the **scores** that check it against data. Each method has its own page, listed in the table below.

## The three questions

| Question                            | What answers it                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Which parameters fit the data?      | The [identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/index.md) methods; a sample of the parameters, or a [profile](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/index.md) of the likelihood or the loss; several starts that end apart with equal losses show that the data do not identify the parameters |
| How do the inputs move the outputs? | The predictive of a sample: the run of every draw, and its band; with no data, a [sensitivity analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/index.md), which propagates the inputs and ranks them                                                                                                                                       |
| Are the intervals calibrated?       | Scores of the predictive against observations the method did not see, and coverage studies                                                                                                                                                                                                                                                                              |

## A band of the model, or a prediction interval

A **band of the model** is the spread of the model's output across the draws of the parameters. It says where the model curve lies, not where the next observation will fall: an observation also carries measurement noise. A **prediction interval** adds that noise to every draw before it reduces them (the `ObservationNoise` port of `commol-inference`). Every reader of observations takes the noise as an argument with no default: `NoNoise()` gives a band of the model, `ResidualNoise(...)` adds residuals of a fit, which gives a prediction interval with no noise model ([Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#prediction-intervals)), and `DistributionNoise()` draws from the [observation models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md) of the problem. Comparing a band of the model with data underestimates the spread, so its coverage of the data is below its level. A [conformal](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/index.md) interval is a prediction interval with a coverage guarantee, from the scores of observations the fit did not use.

## Which method answers which question

| Method                                       | Question                                                                                                                | Needs a noise model                                                             | Page                                                                                                                                       |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Several starts, `Calibrator.run(starts=n)`   | Which parameters fit the data: distinct ends with equal losses                                                          | No                                                                              | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#several-starts)                                           |
| `StructuralRank`                             | Which parameters any data of this design can determine, and which combinations                                          | No                                                                              | [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/#structural-rank)                                |
| `Collinearity`                               | Which parameters can be fitted together                                                                                 | No                                                                              | [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/#collinearity)                                   |
| `SampleIdentifiability`                      | Which parameters a sample determines: contraction, correlations, modes                                                  | No                                                                              | [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/#a-sample)                                       |
| `StartsIdentifiability`                      | Which parameters differ between starts of equal loss                                                                    | No                                                                              | [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/#starts)                                         |
| `BayesianBootstrap`                          | How far the parameters move when the data are reweighted by blocks                                                      | No                                                                              | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#bayesian-bootstrap)                                       |
| `ResidualBootstrap`                          | How far the parameters move when the residuals of the fit are resampled by blocks                                       | No                                                                              | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#residual-block-bootstrap)                                 |
| `WildBootstrap`                              | The same, when the size of the noise changes along the series                                                           | No                                                                              | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#wild-block-bootstrap)                                     |
| `BlockJackknife`                             | The standard error and bias of each parameter, and which block of data moves it most                                    | No                                                                              | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#block-jackknife)                                          |
| `ResidualNoise`                              | Where the next observations fall: a prediction interval                                                                 | No                                                                              | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#prediction-intervals)                                     |
| `GibbsPosterior`                             | A posterior of the parameters from the loss and priors, with a scale `η` of the loss                                    | No                                                                              | [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/#gibbs-posterior)                                     |
| `ABCSMC`                                     | A posterior of the parameters whose runs come within a tolerance of the data, with priors                               | No                                                                              | [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/#abc)                                                 |
| `ABCRejection`                               | The same, from the closest draws of the priors                                                                          | No                                                                              | [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/#abc)                                                 |
| `SplitConformal`                             | Where new observations fall, with a guaranteed coverage, from held-out observations                                     | No                                                                              | [Conformal](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/#split-conformal)                                            |
| `RollingConformal`                           | Where the next observations fall at each horizon ahead of the data                                                      | No                                                                              | [Conformal](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/#rolling-origins)                                            |
| `JackknifePlus`                              | Where new observations fall, with a guaranteed coverage and no held-out data                                            | No                                                                              | [Conformal](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/#jackknife)                                                  |
| `EnbPI`                                      | Where the next observations of a series with dependent errors fall                                                      | No                                                                              | [Conformal](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/#enbpi)                                                      |
| `AdaptiveConformal`                          | Intervals for observations as they arrive, whose share of misses tends to its level                                     | No                                                                              | [Conformal](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/#adaptive-conformal-inference)                               |
| `observation_models` with `Calibrator.run()` | The maximum-likelihood fit of the parameters and the noise                                                              | Yes                                                                             | [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/#maximum-likelihood-against-the-posterior) |
| `BayesianPosterior`                          | The posterior of the parameters from the likelihood and priors, with its evidence                                       | Yes                                                                             | [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/#maximum-likelihood-against-the-posterior) |
| `ParametricBootstrap`                        | How far the parameters move in data sets drawn from the models at the fit                                               | Yes                                                                             | [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/#maximum-likelihood-against-the-posterior) |
| `DistributionNoise`                          | Where the next observations fall: a prediction interval of the models                                                   | Yes                                                                             | [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/#prediction-with-noise)                    |
| `Laplace`                                    | A normal around the fit from the curvature of `−log L`, and the band of the outputs by the delta method                 | Yes                                                                             | [Local Approximations](https://munqu.github.io/commol/latest/guide/uncertainty/local/#laplace-approximation-and-the-delta-method)          |
| `Sandwich`                                   | A covariance around the fit that holds when the noise model is wrong or the noise depends on its neighbours             | No                                                                              | [Local Approximations](https://munqu.github.io/commol/latest/guide/uncertainty/local/#sandwich)                                            |
| `LaplaceMixture`                             | A posterior from Laplace approximations at the optima of several starts, corrected by importance sampling, with `k_hat` | With `LikelihoodTarget`                                                         | [Local Approximations](https://munqu.github.io/commol/latest/guide/uncertainty/local/#laplace-mixture-and-importance-sampling)             |
| `Propagation`                                | How uncertainty in the inputs moves the outputs, with no data                                                           | No                                                                              | [Sensitivity Analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/#propagation)                                   |
| `PRCC`                                       | How steadily each output rises or falls with each input, of independent inputs or a sample                              | No                                                                              | [Sensitivity Analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/#prcc)                                          |
| `Morris`                                     | Which inputs matter, in few runs                                                                                        | No                                                                              | [Sensitivity Analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/#morris-screening)                              |
| `Sobol`                                      | The share of the variance of an output each input explains, alone and with interactions                                 | No                                                                              | [Sensitivity Analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/#sobol-indices)                                 |
| `PolynomialChaos`                            | A polynomial surrogate of an output and its Sobol indices, in few runs                                                  | No                                                                              | [Sensitivity Analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/#polynomial-chaos)                              |
| `ProfileLikelihood`                          | The interval of each parameter and whether the data determine it, from refits with the parameter held                   | No: `Threshold.chi_square` needs one, `beale` and `bootstrap_calibrated` do not | [Profile Likelihood](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/index.md)                                            |
| `PredictionProfile`                          | The interval of a prediction: the values the data do not reject                                                         | No, as `ProfileLikelihood`                                                      | [Profile Likelihood](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/#prediction-profiles)                                |

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
            Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
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

A `ParameterSample` holds draws of the parameters of one problem, each with a weight. Every method with draws returns one; the plots, the reports, the predictive and the scores read it, never the result of a method.

| Field         | Meaning                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------- |
| `names`       | One name per column, in the order of the problem's parameters                                |
| `bounds`      | The bounds of each parameter; every value lies inside them                                   |
| `values`      | One row per draw                                                                             |
| `log_weights` | The log weight of each draw, normalized so the weights sum to 1; `-inf` for a weight of zero |
| `groups`      | The chain, replicate or wave of each draw, or `None`                                         |
| `method`      | The name and settings of the method that made it (`SampleMethod`)                            |

Every statistic is weighted and computed in Rust. With equal weights it is the unweighted statistic.

| Member                                            | Returns                                                                                      |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `mean()`                                          | The weighted mean of each parameter                                                          |
| `covariance()`, `correlation()`                   | Weighted matrices, in the order of `names`; with equal weights, NumPy's `cov` and `corrcoef` |
| `quantiles(levels)`, `interval(confidence_level)` | Weighted quantiles by [the one convention](#the-one-quantile)                                |
| `effective_size()`                                | Kish's effective sample size, `1 / Σ wᵢ²`                                                    |
| `draws()`                                         | One `{name: value}` mapping per draw                                                         |
| `resample(count, seed)`                           | A sample of equal weights, drawn by weight                                                   |
| `summary(confidence_level)`                       | A `ParameterSummary` per parameter, for a report                                             |

```python
print(round(sample.effective_size(), 6))  # 3.0
print(next(sample.draws()))  # {'k1': 0.38, 'k2': 0.09}
```

A sample has no single set of parameters, so `Model.apply_calibration_parameters` rejects it with a `TypeError`; apply one of `sample.draws()`. A sample saves to JSON and loads back equal (`model_dump_json`, `model_validate_json`).

### The one quantile

Every interval of commol is a weighted quantile: the inverse of the weighted empirical distribution, interpolated between the midpoints of the cumulative weights. With equal weights it equals `numpy.percentile(..., method="hazen")`. A draw of weight zero does not count. `weighted_quantiles` computes it for any values, and `member_statistics` reduces ensemble members with it.

```python
from commol import weighted_quantiles

print(weighted_quantiles([0.0, 1.0], levels=[0.5], weights=[1.0, 3.0]))  # [0.75]
```

## The predictive

`Predictive(source, sample, positions)` runs every draw at `positions`, in parallel in Rust, through the forward model of its source: a `Calibrator`, or a `SensitivityAnalysis` of the inputs of the sample. With a stochastic engine every replicate of a draw is a draw of the predictive, with an equal share of its weight. The runs follow one tree of seeds: draw `j` takes the seed `draw_seeds[j]`, derived from the seed of the source, and its replicate `i` takes the seed that replicate `i` of `Simulation.replicates(n, seed=draw_seeds[j])` takes. So every draw shows fresh noise, the predictive repeats from the seed of the source, and the runs of a draw change neither with the number of replicates nor with the number of draws. The positions are sorted and unique, and each one is a position the engine reports at; otherwise `Predictive` raises `ValueError` before any draw runs. A draw whose run leaves the values its model allows raises `InfeasibleStateError` with the index of its parameter set and the message of the run: the compartment, the step and the flows, as `Simulation.run` reports them ([Amounts below zero](https://munqu.github.io/commol/latest/guide/transitions/#amounts-below-zero)). A draw whose initial amounts leave the remainder below zero raises `InfeasibleStateError` with the index of its parameter set and the message a calibration gives for the same values: the remainder and the excess ([Calibrating initial conditions](https://munqu.github.io/commol/latest/guide/calibration-initial-conditions/index.md)).

| Member                                                                                    | Returns                                                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `band(confidence_level)`                                                                  | The weighted mean, median and central interval of every output at every position (`PredictiveBand`)                                                                                                                                                                           |
| `series_band(outputs, confidence_level, cumulative=False)`                                | The band of the sum of some outputs, or of its running total; each draw is reduced first                                                                                                                                                                                      |
| `at_observations(observed_data, noise)`                                                   | Every draw's value of each observation, read through the observation operator the loss uses (sums, windows and factors), with `noise` drawn from the calibrator's seed: `NoNoise()`, `ResidualNoise(...)` or `DistributionNoise()` (`ObservationDraws`)                       |
| `runs()`, `weights`                                                                       | The trajectory and weight of every draw                                                                                                                                                                                                                                       |
| `draw_seeds`                                                                              | The seed of each draw of the sample: `Simulation.replicates(n, seed=draw_seeds[j])` of the model at the values of draw `j` gives its runs                                                                                                                                     |
| `report(confidence_level, positions, window, windows, observed_data=(), accumulators=())` | The report of a run, with each number reduced across the draws with their weights ([Reports](https://munqu.github.io/commol/latest/guide/reports/#the-report-of-an-ensemble))                                                                                                 |
| `plot.series(…)`, `plot.cumulative(…)`, `plot.observations(…)`                            | The weighted median of the draws of each series and their band, those of its running total, or those of each observed series, read through `at_observations(…, NoNoise())` ([Plotting](https://munqu.github.io/commol/latest/guide/simulations/#a-panel-per-observed-series)) |

`predictive.plot.series()` and `predictive.plot.cumulative()` draw the band of the predictive, each draw read through the observation operator of its observations, and `Predictive.report` reduces its draws with their weights. `Model.applied_parameters` takes the sample itself (`sample=`, with a `confidence_level`).

## Scores

The functions of `commol.scores` compare a predictive with observations, and return a `Score`: the value of each observation, the mean of each series and the overall mean. Which observations to score and the observation noise are arguments with no default: score data the method did not see, such as the held-out positions above, and say whether you score a band of the model (`NoNoise()`) or a prediction interval (`ResidualNoise(...)`, `DistributionNoise()`).

| Function                                                                      | Score                                                                        | Better                                                                                                                      |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `coverage(predictive, observations, noise, confidence_level)`                 | 1 inside the central interval, 0 outside; the mean is the empirical coverage | Close to the level                                                                                                          |
| `interval_score(predictive, observations, noise, confidence_level)`           | `(u − l) + (2/α)(l − y)₊ + (2/α)(y − u)₊`, `α = 1 − c`                       | Lower                                                                                                                       |
| `weighted_interval_score(predictive, observations, noise, confidence_levels)` | The weighted interval score of Bracher et al. (2021)                         | Lower                                                                                                                       |
| `crps(predictive, observations, noise)`                                       | \`Σ wᵢ                                                                       | xᵢ − y                                                                                                                      |
| `pit(predictive, observations, noise, seed)`                                  | `F(y)`, randomized where draws tie with `y`                                  | Uniform                                                                                                                     |
| `log_score(predictive, observations, DistributionNoise())`                    | \`log Σ_d w_d · p(y                                                          | μ_d)`, the log density of `y`under the mixture of the observation models over the draws; another noise raises a`TypeError\` |

## The entry point

Every method runs through `Calibrator.uncertainty(method)`. A method is a configuration, a subclass of `UncertaintyMethod[R]` whose type parameter `R` is its result; the call returns an `R`, which a static type checker infers from the configuration's type. A configuration with no registered runner raises a `ValueError` that names the registered ones.

The identifiability methods run through their own entry point, `Calibrator.identifiability(method)`, with a subclass of `IdentifiabilityMethod[R]`: they answer another question ([Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/index.md)). Both kinds subclass `InferenceMethod[R]` and share one registry; each entry point refuses the other kind with a `TypeError` that names the right one. The sensitivity methods, subclasses of `SensitivityMethod[R]`, run with no data through `SensitivityAnalysis(simulation, inputs).run(method)`, the third entry point of the registry ([Sensitivity Analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/index.md)).

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

| To learn                                                                         | Read                                                                                                         |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Bootstraps, the jackknife and prediction intervals with no noise model           | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/index.md)                    |
| Posteriors with no noise model: the Gibbs posterior and ABC                      | [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/index.md)               |
| Maximum likelihood, the posterior and prediction intervals of observation models | [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md)    |
| Prediction intervals with a coverage guarantee                                   | [Conformal Prediction Intervals](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/index.md) |
| How the inputs move the outputs, with no data                                    | [Sensitivity Analysis](https://munqu.github.io/commol/latest/guide/uncertainty/sensitivity/index.md)         |
| Several starts of an optimizer                                                   | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#several-starts)             |
| Which parameters the data determine                                              | [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/index.md)          |
| Reports of a sample                                                              | [Reports](https://munqu.github.io/commol/latest/guide/reports/index.md)                                      |
| Plots of a predictive                                                            | [Running Simulations](https://munqu.github.io/commol/latest/guide/simulations/#plotting)                     |
