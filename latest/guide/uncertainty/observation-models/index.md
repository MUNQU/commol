# Observation Models

[Back to Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)

An observation model says how an observed value scatters around the model: a normal error, a count, a proportion, an outlier. With one model per series, a calibration fits by maximum likelihood instead of a loss, and the same likelihood gives a Bayesian posterior with its evidence, prediction intervals that hold the noise, the log score and the parametric bootstrap. This page is for you when you know how your data were measured and want methods that use it; the methods of [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/index.md), [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/index.md) and [Conformal](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/index.md) need no noise model.

## Quick path

```python
import numpy as np

from commol import (
    Aggregate,
    BayesianPosterior,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    DistributionNoise,
    ModelBuilder,
    NegativeBinomialObservations,
    NelderMeadConfig,
    ObservedDataPoint,
    ParametricBootstrap,
    Predictive,
    Priors,
    Simulation,
    scores,
)


def decay_model(k1: float | None):
    return (
        ModelBuilder(name="Decay")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_parameter(id="k1", value=k1)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * A")
        .build(
            Aggregate(bin_amounts={"A": 1000, "B": 0}),
            engine="DifferenceEquations",
        )
    )


# Counts of the units that reach B in each step, scattered more than a
# Poisson: a negative binomial with overdispersion 0.1 around the model.
truth = Simulation(decay_model(0.05)).run(num_steps=40)
rng = np.random.default_rng(3)
counts = []
for step in range(1, 41):
    mean = truth["B"][step] - truth["B"][step - 1]
    size = 1 / 0.1
    counts.append(float(rng.negative_binomial(size, size / (size + mean))))
observations = [
    ObservedDataPoint(at=step, output="B", value=value, window=1)
    for step, value in zip(range(1, 41), counts)
]
fitted, held_out = observations[:30], observations[30:]

problem = CalibrationProblem(
    observed_data=fitted,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=0.5),
        CalibrationParameter(id="phi", parameter_type="noise", min_bound=0.0, max_bound=2.0),
    ],
    observation_models={"B": NegativeBinomialObservations(overdispersion="phi")},
    optimization_config=NelderMeadConfig(max_iterations=2000),
    seed=42,
)
calibrator = Calibrator(Simulation(decay_model(None)), problem)

# Maximum likelihood: the loss is the negative log-likelihood.
fit = calibrator.run()
print({name: round(value, 2) for name, value in fit.best_parameters.items()})
# {'k1': 0.05, 'phi': 0.11}

# The Bayesian posterior and its log evidence.
posterior = calibrator.uncertainty(
    BayesianPosterior(
        priors=Priors.uniform_on_bounds(),
        particles=500,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.05,
    )
)
print([round(bound, 3) for bound in posterior.sample.interval(0.9)["k1"]])
# [0.044, 0.072]
print(round(posterior.log_evidence, 1))  # -113.2

# Prediction intervals of the held-out counts, with the noise of the model.
predictive = Predictive(calibrator, posterior.sample, positions=list(range(41)))
print(scores.coverage(predictive, held_out, DistributionNoise(), confidence_level=0.9).mean)
# 1.0
print(round(scores.log_score(predictive, held_out, DistributionNoise()).mean, 2))
# -2.83
```

## Choosing a model

The mean of every model is the prediction of the observation (its sums, window and factor), except the median of a `LogNormalObservations` with `center="median"`. A model only adds the scatter around it.

| Your data                                            | Model                                          | Variance                               | Notes                                                                                             |
| ---------------------------------------------------- | ---------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Continuous values with errors of one size            | `NormalObservations(sd)`                       | `sd²`                                  | With a known `sd` the fit is the least-squares fit                                                |
| Positive values with errors that grow with the value | `LogNormalObservations(sigma, center)`         | `(e^{σ²} − 1)·μ²` with `center="mean"` | `center` is required: `"mean"` makes `E[y] = μ`; `"median"` is the proportional error `y = μ·e^ε` |
| Continuous values with outliers                      | `StudentTObservations(df, scale)`              | `scale²·df/(df − 2)` when `df > 2`     | Heavier tails: an outlier pulls the fit less                                                      |
| Counts                                               | `PoissonObservations()`                        | `μ`                                    |                                                                                                   |
| Counts that scatter more than a Poisson              | `NegativeBinomialObservations(overdispersion)` | `μ + φμ²`                              | `φ = 0` is the Poisson; the dispersion `k` (`size` in R) is `1/φ`                                 |
| Successes out of known trials                        | `BinomialObservations(trials)`                 | `μ(1 − μ/n)`                           | The prediction is the expected count `n·p`, not a proportion                                      |

A `BinomialObservations` takes its trials from `trials=n`, for every observation of the series, or with no argument from `ObservedDataPoint.trials` of each observation, for trials that change from one observation to the next. A series has one source; both or neither raise a `ValueError` that names the observation.

## Noise parameters

A noise argument is a number, or the id of a calibration parameter of type `"noise"`. A noise parameter is fitted, sampled and reported with the others, and never reaches the engine; `Model.apply_calibration_parameters` skips it, as it skips a factor. Its bounds must lie in the domain of the argument: above 0 for `sd`, `sigma`, `scale` and `df`, at least 0 for `overdispersion`.

| Rule                                                                                                        | Error                                                |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| A problem has `loss_function` or `observation_models`, exactly one, with no default                         | `ValueError` from `CalibrationProblem`               |
| Every observed series has a model, and every model names an observed series                                 | `ValueError` from `Calibrator` that names the series |
| Every noise parameter is an argument of a model                                                             | `ValueError` that names the parameter                |
| Counts are whole numbers, at least 0; a binomial count is at most its trials; a log-normal value is above 0 | `ValueError` that names the observation              |
| A stochastic engine needs a loss                                                                            | `ValueError` from `Calibrator`                       |
| `normalize_observations` needs a loss: the models set the scale of each series                              | `ValueError` from `CalibrationProblem`               |

`Likelihood::new` in `crates/commol-calibration/src/likelihood/mod.rs` checks the models and values; `CalibrationProblem.validate_loss_or_models` checks the first and the last rules.

A stochastic engine is refused because the likelihood of a stochastic model integrates its hidden path, which the observations of one run do not do. Fit a stochastic model with a loss and its replicates ([Stochastic engines](https://munqu.github.io/commol/latest/guide/calibration/#stochastic-engines)).

## The likelihood

The log-likelihood is `Σᵢ wᵢ · log p(yᵢ | μᵢ)`, where `μᵢ` is the prediction and `wᵢ` the observation weight. The weight raises the density of the observation to its power: an observation of weight 2 counts twice, as in every loss. The log densities are written in commol (`Noise::log_density` in `crates/commol-calibration/src/likelihood/density.rs`), and `crates/commol-calibration/tests/test_likelihood.rs` checks them against reference values of SciPy.

A prediction outside the means a model allows, such as a negative count or a binomial mean above its trials, leaves the point outside the support: the optimizer gets a penalty, and a sampler a density of 0.

## Maximum likelihood against the posterior

| Method                            | Gives                                                                                                                                                               | Needs                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `Calibrator.run()`                | The maximum-likelihood fit; `final_loss` is `−log L`                                                                                                                | Nothing more                                                                                                        |
| `BayesianPosterior(priors, …)`    | The posterior `prior(θ) · L(θ)` and its log evidence, on the core of [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/index.md) | A prior for every parameter, the noise parameters too                                                               |
| `GibbsPosterior(eta=…)`           | The fractional posterior `prior(θ) · L(θ)^η`; `η = 1` is the posterior                                                                                              | `η` as a number, `InformationMatchingEta(block_length)` (`η ≈ 1` for correct models) or `BootstrapCoverageEta(...)` |
| `ParametricBootstrap(replicates)` | Fits to data sets drawn from the models at the fit                                                                                                                  | Models that hold: the observations of a replicate are independent given the fit                                     |

A posterior answers "what do the data and the priors say about the parameters"; the fit and the parametric bootstrap answer "what would another data set of this design give". With little data or a parameter near a bound, they differ; with much data and flat priors they agree.

```python
bootstrap = calibrator.uncertainty(ParametricBootstrap(replicates=40))
low, high = bootstrap.sample.interval(0.9)["k1"]
print(low < 0.05 < high)  # True
```

`LossDistance()` of ABC needs a loss; `SeriesDistance(...)` works with models.

## Prediction with noise

`DistributionNoise()` draws every observation from the model of its series around each draw's prediction, with the draw's noise parameters, so `Predictive.at_observations(observations, DistributionNoise())` gives a prediction interval and every score of `commol.scores` scores one. It needs a problem with `observation_models`, and serves the series they name, at any position.

`scores.log_score(predictive, observations, DistributionNoise())` scores the observed values by the densities of these models ([Scores](https://munqu.github.io/commol/latest/guide/uncertainty/#scores)).

## Checklist

- [ ] Every observed series has a model that fits how its data were measured.
- [ ] Every noise parameter has bounds in its domain, and a prior for a posterior.
- [ ] You use `DistributionNoise()` for prediction intervals of these models.
- [ ] You score observations the fit did not see.

## Next steps

| To learn                                       | Read                                                                                                         |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| The sampler, its settings and its diagnostics  | [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/index.md)               |
| Bootstraps with no noise model                 | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/index.md)                    |
| Prediction intervals with a coverage guarantee | [Conformal Prediction Intervals](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/index.md) |
| The problem, its observations and weights      | [Model Calibration](https://munqu.github.io/commol/latest/guide/calibration/index.md)                        |
