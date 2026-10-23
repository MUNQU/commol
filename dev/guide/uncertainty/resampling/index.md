# Resampling: Bootstraps and the Jackknife

[Back to Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)

These methods give the uncertainty of a calibration with no noise model and no prior. They fit the model again to reweighted or resampled data, starting from the fit, and read the spread of the new fits. Use them when you can say how the data were collected but not how the observations scatter around the model. Each runs through `Calibrator.uncertainty`, in parallel in Rust (`crates/commol-inference/src/resampling/`).

## Quick path

```python
import numpy as np

from commol import (
    Aggregate,
    BayesianBootstrap,
    BlockJackknife,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    NoNoise,
    ObservedDataPoint,
    Predictive,
    ResidualBootstrap,
    ResidualNoise,
    Simulation,
    WildBootstrap,
    scores,
)


def ab_model(k: float | None):
    return (
        ModelBuilder(name="AB")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="k", value=k)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k")
        .build(
            Aggregate(bin_amounts={"A": 1000, "B": 0}),
            engine="DifferenceEquations",
        )
    )


# Data with a multiplicative noise of 5 %: fit steps 1 to 60, hold out the rest.
truth = Simulation(ab_model(0.05)).run(num_steps=80)
rng = np.random.default_rng(1)
observations = [
    ObservedDataPoint(
        at=step,
        output="B",
        value=truth["B"][step] * float(np.exp(rng.normal(0.0, 0.05))),
    )
    for step in range(1, 81)
]
fitted = [point for point in observations if point.at <= 60]
held_out = [point for point in observations if point.at > 60]

problem = CalibrationProblem(
    observed_data=fitted,
    parameters=[
        CalibrationParameter(id="k", parameter_type="parameter", min_bound=0.0, max_bound=0.5)
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=500),
    seed=42,
)
calibrator = Calibrator(Simulation(ab_model(None)), problem)

result = calibrator.uncertainty(BayesianBootstrap(replicates=100, block_length="auto"))
lower, upper = result.sample.interval(0.9)["k"]
print(round(lower, 4), round(upper, 4))  # 0.0486 0.0508
print(result.diagnostics.block_length)  # 1
```

`result.fit` is the fit, `result.sample` holds one draw per replicate that fitted, `result.failed` lists the replicates that failed, and `result.diagnostics` gives the block length.

## Which method to use

| Method              | Changes in each replicate                                                                  | Assumes                                                                           | Answers                                                                |
| ------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `BayesianBootstrap` | The weight of each block of observations                                                   | The blocks are exchangeable                                                       | How far the parameters move when the data are reweighted               |
| `ResidualBootstrap` | The observed values: fitted values plus blocks of residuals drawn with replacement         | The model is right; the residuals of each block are exchangeable across positions | How far the parameters move under new noise like the fit's             |
| `WildBootstrap`     | The observed values: each residual times a random sign or Mammen multiplier, one per block | The model is right; the noise has mean zero at each position                      | The same, when the size of the noise changes along the series          |
| `BlockJackknife`    | One block left out (weight 0) per fit                                                      | The blocks are exchangeable                                                       | A standard error and a bias, and which block moves each parameter most |

None of them needs a noise model. The Bayesian bootstrap never changes the observed values, so it also works when the model is wrong; the residual and wild bootstraps make new data from the model and its residuals. With [observation models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md), `ParametricBootstrap(replicates)` draws new data from the models at the fit; it draws no blocks, so its `diagnostics.block_length` is `None`.

## Blocks and their length

A block holds every observation, of every series, whose position falls in one window of `block_length` consecutive distinct positions (`BlockScheme`, `resampling/blocks.rs`). The dependence of the data across series and along the positions stays inside blocks, so the methods hold for noise that is correlated in time or between series.

`block_length` has no default:

| Value      | Use it when                                                                                                                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`        | The noise of different positions is independent                                                                                                                                                                  |
| An integer | You know how far the dependence of the noise reaches                                                                                                                                                             |
| `"auto"`   | You do not know: the rule of Politis and White (2004), with the correction of Patton, Politis and White (2009), estimates a length for each series on the residuals of the fit, and the method takes the largest |

With `"auto"`, `diagnostics.block_length_estimates` gives the estimate of each series and `diagnostics.block_length` the length used. The estimate depends on how a method draws its blocks: moving blocks get a length about 15 % longer than stationary or non-overlapping ones.

## Bayesian bootstrap

`BayesianBootstrap(replicates, block_length, starts=1)` draws, for each replicate, one Dirichlet(1, …, 1) weight per block, scaled to mean 1, and multiplies the weight of every observation of the block by it. With a loss it is the loss-likelihood bootstrap of Lyddon, Holmes and Walker (2019); with `block_length=1`, the Bayesian bootstrap of Rubin (1981). Its variance matches the sandwich variance of least squares (`tests/uncertainty/test_resampling.py::TestBayesianBootstrap`).

## Residual block bootstrap

`ResidualBootstrap(replicates, block_length, residual_scale, scheme, starts=1)` takes the residuals of the fit, joins blocks of them drawn with replacement, and fits the new data.

| Setting          | Values                                                                                                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `residual_scale` | `"additive"`: `r = y − ŷ`, new data `ŷ + r*`. `"multiplicative"`: `r = log(y / ŷ)`, new data `ŷ · exp(r*)`; it needs positive observed and fitted values, and keeps the new data positive       |
| `scheme`         | `"moving"`: blocks of `block_length` consecutive positions (Künsch 1989). `"stationary"`: blocks of geometric length with mean `block_length`, wrapped around the end (Politis and Romano 1994) |

Use the multiplicative scale for counts or amounts that span orders of magnitude, where the noise grows with the value. On the additive scale a new value can fall below zero: the method keeps it, and counts it in `diagnostics.negative_values`. Many of them point to the multiplicative scale.

```python
residual = calibrator.uncertainty(
    ResidualBootstrap(
        replicates=100,
        block_length="auto",
        residual_scale="multiplicative",
        scheme="stationary",
    )
)
print(residual.diagnostics.negative_values)  # 0
```

When a series has no observation at a drawn position, it takes the residual of the nearest position where it has one.

## Wild block bootstrap

`WildBootstrap(replicates, block_length, multipliers, starts=1)` multiplies the additive residuals of each block by one draw of `multipliers` and fits `ŷ + r · v`. Every residual stays at its own position, so the method holds when the size of the noise changes along the series, where the residual bootstrap mixes small and large residuals.

| `multipliers`  | Distribution                                                                                             |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| `"rademacher"` | `±1` with probability 1/2                                                                                |
| `"mammen"`     | Two values with mean 0, variance 1 and third moment 1 (Mammen 1993); keeps the skewness of the residuals |

```python
wild = calibrator.uncertainty(
    WildBootstrap(replicates=100, block_length=5, multipliers="mammen")
)
print(len(wild.sample))  # 100
```

## Block jackknife

`BlockJackknife(block_length, starts=1)` fits once per block, with that block left out, and returns a `JackknifeResult`. With `B` blocks it needs `B` fits, usually far fewer than a bootstrap.

| Field            | Meaning                                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `estimate`       | The bias-corrected estimate of each parameter                                                                                               |
| `bias`           | `(B − 1)(θ̄ − θ̂)`, with `θ̄` the mean of the fits with one block left out                                                                     |
| `standard_error` | `√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)`                                                                                                             |
| `blocks`         | The first and last position of each block that fitted                                                                                       |
| `pseudo_values`  | `B θ̂ − (B − 1) θ̂₍₋ᵦ₎` per block                                                                                                             |
| `influence`      | How far leaving each block out moves each parameter, in standard errors                                                                     |
| `failed`         | Each block whose fit failed, `"Block <index> failed: <reason>"`, with blocks numbered from 0 in order of position; also logged as a warning |
| `not_converged`  | Fits kept although their optimizer did not converge                                                                                         |
| `diagnostics`    | The block length, as for a bootstrap                                                                                                        |

The pseudo-values are not draws: their percentiles are not an interval, so the result has no `ParameterSample`. A block with a large influence holds data the fit depends on: check it.

```python
jackknife = calibrator.uncertainty(BlockJackknife(block_length=10))
print(jackknife.blocks[:2])  # [(1.0, 10.0), (11.0, 20.0)]
print(len(jackknife.influence["k"]))  # 6
```

## The result

A bootstrap returns a `BootstrapResult(fit, sample, failed, not_converged, diagnostics)`. The sample has equal weights, and the group of each draw is its replicate.

| Field                                                            | Meaning                                                                                                                    |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `failed`                                                         | Each replicate that failed, `"Replicate <index> failed: <reason>"`; it is not in the sample, and it is logged as a warning |
| `not_converged`                                                  | Replicates kept although their optimizer did not converge                                                                  |
| `diagnostics.block_length`, `diagnostics.block_length_estimates` | The length used, and the estimate of each series with `"auto"`                                                             |
| `diagnostics.negative_values`                                    | New observed values below zero, over every replicate                                                                       |

Every result of a method that fits several times lists the fits that failed in `failed` and counts the fits kept without convergence in `not_converged`, in the same way: a bootstrap, the jackknife, a profile with a bootstrap threshold and the conformal methods that refit. When too few fits succeed, the method raises a `RuntimeError` that says how many of how many fitted, how many it needs, and the first reasons.

`result.interval_error(confidence_level)` gives the Monte Carlo error of each bound of the central interval: an order-statistic interval that holds the bound of infinitely many replicates with about 95 % probability. When it is wide against the interval, run more replicates.

```python
error = result.interval_error(0.9)["k"]
print(error.lower[0] <= lower <= error.lower[1])  # True
```

## How replicates run

Each replicate starts from the fit. With `starts > 1` it also starts from the [other starts](https://munqu.github.io/commol/latest/guide/calibration-optimizers/#several-starts) of the optimizer, and keeps the end of lowest loss, so a second basin of the loss shows up instead of adding optimizer noise to the data's uncertainty. Replicate `b` draws its data and its starts from `replicate_seed(seed, b)` of the problem's seed: two runs with one seed give one result. A replicate changes only the observed values or the weights (`CalibrationProblem::with_data`); with `normalize_observations` its normalization comes from its own values, as the fit's does.

## Prediction intervals

The draws of a bootstrap give a band of the model. `ResidualNoise` adds moving blocks of residuals of the fit to every draw's prediction of each observation, which gives a prediction interval with no noise model. It serves the series the fit has residuals of, at any position, and rejects an observation of any other series.

```python
predictive = Predictive(calibrator, result.sample, positions=list(range(81)))
noise = ResidualNoise(
    fit=result.fit,
    block_length=result.diagnostics.block_length,
    residual_scale="multiplicative",
)

print(scores.coverage(predictive, held_out, NoNoise(), confidence_level=0.9).mean)  # 0.0
print(scores.coverage(predictive, held_out, noise, confidence_level=0.9).mean)  # 0.9
```

The band of the model is narrow and covers none of the held-out observations; the prediction interval covers them at its level.

## Short series

A block bootstrap reads the dependence of the noise from few blocks. On a short series its variance is too small by a term of order `1 / block_length`, and its interval covers less than its level. In the coverage study of `tests/uncertainty/test_resampling_coverage.py`, a 90 % interval of a series with AR(1) noise (`φ = 0.5`) covers about 0.8 with 200 positions and about 0.89 with 400. With few positions, prefer `block_length=1` only when the noise is independent, and report that the interval may be narrow.

## Checklist

- [ ] You chose `block_length`, or read the estimates of `"auto"`.
- [ ] You chose the residual scale for the data: multiplicative for values that span orders of magnitude.
- [ ] `failed` is empty, or you know why replicates failed.
- [ ] `not_converged` is 0, or you raised `max_iterations`.
- [ ] `interval_error` is small against the interval.
- [ ] You report a band of the model as a band of the model, and use `ResidualNoise` for a prediction interval.

## Next steps

| To learn                                            | Read                                                                                      |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| The parameter sample, the predictive and the scores | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)           |
| The settings of the optimizer each replicate runs   | [Optimizers](https://munqu.github.io/commol/latest/guide/calibration-optimizers/index.md) |
