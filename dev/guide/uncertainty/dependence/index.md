# Dependence

[Back to Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/index.md)

Many methods of commol read each observation as one independent piece of information: a likelihood sums the terms of its observations, `block_length=1` resamples them one by one, and a Laplace approximation reads the curvature of their sum. When the noise of neighbouring positions moves together, or two series share a shock, the data hold less information than their count says, and those intervals are too narrow. This page shows how to measure that dependence, in the residuals of a fit or in any values, and what to do with the answer. It is for anyone who reads intervals from data along an axis.

The analyses run in Rust (`crates/commol-inference/src/dependence/`, with the estimators in `crates/commol-statistics/src/dependence.rs`) and live in `commol.dependence`.

## Quick path

The residuals of a constant flow fitted to data whose noise is an AR(1) of 0.6: the test rejects independence, and `τ` says the data count each observation about 2.6 times.

```python
import numpy as np

from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    ModelBuilder,
    NelderMeadConfig,
    ObservedDataPoint,
    Simulation,
)
from commol.dependence import Refitted, residual_dependence, series_dependence
from commol.observations import NoNoise
from commol.uncertainty import Predictive


def flow_model():
    """A constant flow c from A to B each step."""
    return (
        ModelBuilder(name="Flow")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="c", value=None)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="c * A / A")
        .build(Aggregate(bin_amounts={"A": 1e6, "B": 0}), engine="DifferenceEquations")
    )


# The flow of each of 60 steps: 10 plus an AR(1) noise of 0.6.
rng = np.random.default_rng(1)
noise = np.zeros(60)
for t in range(1, 60):
    noise[t] = 0.6 * noise[t - 1] + rng.normal()
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=step, output="B", value=float(10 + value), window=1)
        for step, value in enumerate(noise, start=1)
    ],
    parameters=[
        CalibrationParameter(id="c", parameter_type="parameter", min_bound=0.0, max_bound=30.0)
    ],
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=500),
    seed=3,
)
calibrator = Calibrator(Simulation(flow_model()), problem)
fit = calibrator.run()

# The residuals of the fit, against refits of data built from the fit and
# its residuals with signs ±1.
at_fit = Predictive(calibrator, calibrator.point_sample(fit), positions=[1])
result = residual_dependence(
    at_fit,
    problem.observed_data,
    NoNoise(),
    max_lag="auto",
    randomizations=1,
    reference=Refitted(replicates=99),
    confidence_level=0.9,
)
(series,) = result.series
print(series.max_lag)  # 10
print([round(r, 2) for r in series.correlations[:3]])  # [0.52, 0.17, 0.1]
print(series.p_value)  # 0.02
print(round(series.tau, 2), [round(b, 2) for b in series.tau_band])  # 2.58 [0.38, 1.97]
```

## The statistics of a series

A series is the values of one label in position order. Its statistics read them on the grid of its smallest spacing: positions 7, 14, 21 and 35 are a series of step 7 with a gap at 28, and lag 1 is 7 positions. Each lag reads the pairs of values that exist at its distance (the amplitude-modulated estimator of Parzen 1963 and Stoffer and Toloi 1992), so a gap costs pairs, never pairs values at different distances. `pairs` gives the pairs of each lag.

| Statistic      | What it says                                                                                                                                                                                                                                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `correlations` | The autocorrelation at lags `1 … max_lag`; `None` at a lag with no pair                                                                                                                                                                                                                                                                                            |
| `portmanteau`  | The robust portmanteau `Q̃ = Σ_k (Σ e_t e_{t−k})² / Σ e_t² e_{t−k}²` of the values centered per series, the diagonal form of the statistic of Dalla, Giraitis and Phillips (2022): `χ²` of `degrees_of_freedom` lags under independence, also when the size of the noise varies along the series; `None`, with the `portmanteau_reason`, when no lag enters its sum |
| `few_pairs`    | Each lag of fewer than 4 pairs, as `(lag, pairs)`: it keeps its correlation and is left out of `Q̃` and of `degrees_of_freedom`                                                                                                                                                                                                                                     |
| `p_value`      | The p-value of `Q̃` under the reference                                                                                                                                                                                                                                                                                                                             |
| `tau`          | The integrated autocorrelation time `τ = 1 + 2 Σ r(k)`, by the initial monotone sequence of Geyer (1992): how many times the series counts each value                                                                                                                                                                                                              |
| `tau_band`     | The central interval of `τ` at `confidence_level` under independence, with one draw; `None` when no replicate has a `τ`, as for a constant series                                                                                                                                                                                                                  |
| `cut_lag`      | The largest lag in the sum of `τ`: the sequence `Γ_k = r(2k) + r(2k+1)`, made monotone, stops at its first `Γ_k ≤ 0` or at the first lag with no pair; over several sets, the largest of theirs                                                                                                                                                                    |

`max_lag` has no default: an integer, in steps of the grid of each series, or `"auto"`, `min(10, n/5)` for a series of `n` values, at least 1. `τ` does not read it: its sum stops where the sequence of Geyer stops.

A series has no statistics, with the `reason`, when two of its observations share one position (give one of them a label of its own), a position is off its grid, such as 0, 2, 3 and 4.5, or its grid has more than 10 points per value, as when one position lies close to another. `τ` is undefined, with the `tau_reason`, for fewer than 4 values, constant values, or an estimate not above 0, which comes from a negative correlation at the first lags; an undefined `τ` reads as below its band.

A lag of few pairs adds a term bounded by its pairs, of mean near 1 but with less spread than the `χ²(1)` it counts as: a lag of one pair adds 1 whatever the values. So `Q̃` sums only the lags of 4 pairs or more, and `few_pairs` names the others: a `max_lag` near the length of the series leaves out its last lags, and a gap the lags it breaks. Over 3000 sets of independent values each, with `max_lag` at the last lag of 12 positions, of 19 positions on a grid of 35 and of 14 on a grid of 24, `Q̃` rejects at 5 % in 4.3 % of them, and in 3.1 % with every lag in its sum (`tests/uncertainty/test_dependence_coverage.py`).

A lag of 4 pairs or more still pulls the rate down, less the more pairs it reads: one lag of 4 independent pairs alone rejects at 5 % in 0.2 % of the sets, of 6 in 1.2 %, of 10 in 3.1 % and of 20 in 4.4 %. So a short series, or one whose gaps leave few pairs at every lag, stays conservative: `"auto"` gives 5 values one lag of 4 pairs (0.2 %), 6 values 1.0 % and 8 values 2.0 %. Many lags make the test liberal: 6.2 % at `max_lag=11` and 7.5 % at 17 with 35 values, 8.1 % at 33 with 100; `"auto"` reads at most 10 lags and a fifth of the values. A series whose every lag reads fewer than 4 pairs, such as one of 4 values, has no `Q̃`: `portmanteau` and `p_value` are `None`, and the `portmanteau_reason` says so.

## The series together

Two series can share a shock at the same position, which no series shows alone. Over the series standardized by their own spread, `z`, the `problem` of the result gives:

| Statistic            | What it says                                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `variance_factor`    | `Var(Σz) / ΣVar(z)`, `Σz` summed at each position of any series: 1 for series independent of each other                                               |
| `tau`                | `τ` of `Σz`: the dependence of the shared shock along the positions                                                                                   |
| `information_factor` | Their product, `LRV(Σz) / ΣVar(z)`: how many times the observations count their information; 1 for independent observations, 2 when each counts twice |
| `p_value`            | The test of the information factor against the reference                                                                                              |

`correlations` gives the correlation at lag 0 of every two series, over the positions they share.

## Any values

`series_dependence(values, observations, max_lag, replicates, confidence_level, seed)` reads one finite value per observation: a residual of held-out data, a score, any number. `Q̃` is read against `χ²`, and the band of `τ` and the test of the factor come from `replicates` sets of the values centered per series with independent signs ±1 (a wild band), from the `seed`. A band from independent normal series would leave 12 to 39 % of heteroscedastic values outside it; the wild band keeps the size of each value. In a study of 1000 sets of 35 independent values whose spread grows 30 times along the series, `Q̃` at `max_lag=7` rejects at 5 % in 5.1 % of them, and an AR(1) of 0.5 on 100 positions in 98 %.

Two series with a shared AR(1) shock of 0.7: each series alone is near independent, and the factor of the problem says the observations count about twice.

```python
rng = np.random.default_rng(2)
shock = np.zeros(50)
for t in range(1, 50):
    shock[t] = 0.7 * shock[t - 1] + rng.normal()
values = [*(shock + rng.normal(size=50)), *(shock + rng.normal(size=50))]
observations = [
    ObservedDataPoint(at=t, output=output, value=0.0)
    for output in ("north", "south")
    for t in range(1, 51)
]
together = series_dependence(
    values, observations, max_lag=5, replicates=199, confidence_level=0.9, seed=7
)
print([round(s.tau, 2) for s in together.series])  # [1.34, 1.2]
print(round(together.correlations[0].correlation, 2))  # 0.54
print(round(together.problem.information_factor, 2), together.problem.p_value)  # 2.1 0.035
```

`values` is one set: a list of lists raises a `TypeError` that points to `residual_dependence`, which reads the residuals of several draws or randomizations with a valid law.

## The residuals of a predictive

`residual_dependence(predictive, observations, noise, max_lag, randomizations, reference, confidence_level)` reads the residuals of a predictive of a `Calibrator`. Every random number comes from the seed of the calibrator, so the same predictive gives the same result.

| `noise`               | Residuals                                                                                                                                                                                                                                                                                            | `randomizations`                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `DistributionNoise()` | The [quantile residuals](https://munqu.github.io/commol/latest/guide/uncertainty/#quantile-residuals) of the observation models of each draw. A count model randomizes them, so the statistics are averaged over the randomizations, common to every draw: the result does not depend on one of them | At least 1; more for small counts |
| `NoNoise()`           | The additive residuals `y − ŷ` of a loss or a stochastic engine, `ŷ` the prediction the loss scores                                                                                                                                                                                                  | 1                                 |
| `ResidualNoise(...)`  | A `TypeError`: it adds the residuals of a fit as noise                                                                                                                                                                                                                                               | —                                 |

With one draw, such as `Calibrator.point_sample(fit)`, the result has its tests. With many, it gives the statistics averaged with the weights of the draws (`τ` as the ratio of the mean long-run variance to the mean variance), and, for `Q̃` and the factor, `draw_p_values` in place of `p_value` and `tau_band`: the median and the weighted share at or below `1 − confidence_level` of the p-values of the draws. That is no test: a p-value of one draw holds its level only under an exact posterior and moves with the seed. Test the fit with `Calibrator.point_sample(fit)`.

## The reference

`reference` has no default, since what holds depends on the data the residuals come from.

| Reference                        | What it draws                                                                                                                                                                                                                                                               | Use it for                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `AtParameters(replicates)`       | With `DistributionNoise()`, `replicates` data sets from the observation models at the predictions of the draw, with no refit and no run of the model; with `NoNoise()`, `Q̃` against `χ²` and the wild band                                                                  | Held-out data, or data drawn at the parameters |
| `Refitted(replicates, starts=1)` | `replicates` data sets drawn from the observation models at the fit (`DistributionNoise()`), or the fit plus its residuals with signs ±1 (`NoNoise()`), each fitted again by the optimizer of the problem from `starts` starts; the p-value is `(1 + #{Q_b ≥ Q}) / (B + 1)` | The data of the fit                            |

A fit leaves its residuals correlated with each other, the more so the more parameters it has, and `Refitted` holds that in its law. In a study of 400 data sets of independent Poisson counts at 35 positions, fitted by a smooth flow of 3 or 10 parameters with a Poisson model or with the sum of squares, `Refitted(replicates=39)` rejects at 5 % in 4.8 to 5.2 % of them with 3 parameters and 6.5 to 7.2 % with 10; `AtParameters` in 7.5 to 8.5 % and 24 to 26 %. `Refitted` needs a sample of one point and the observations of the problem, and a `ValueError` says so otherwise. Its refits run in parallel, log a record of each, and the result gives `failed` and `not_converged`.

The power is that of a short series. Normal AR(1) noise of 0.5 at 100 positions, fitted with 3 parameters, is rejected in 86 % of data sets; at 35 positions with 10 parameters, in about 7 to 15 %. Not rejecting a short series is no evidence of independence: read `τ` and the autocorrelations, and the bias table below.

## The bias of `τ`

`τ` of a short series is biased: the median, the 10–90 % range and the share undefined of 500 series of each kind, true `τ` in brackets, the fits by least squares on Chebyshev polynomials.

| Series (true `τ`)                                 | 20 positions           | 35                     | 100                    | 1000                   |
| ------------------------------------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| Independent (1)                                   | 0.98 (0.46–1.67), 0 %  | 0.98 (0.56–1.57), 0 %  | 1.04 (0.76–1.45), 0 %  | 1.03 (0.93–1.18), 0 %  |
| AR(1) of 0.5 (3)                                  | 1.95 (1.16–3.56), 0 %  | 2.30 (1.52–4.14), 0 %  | 2.70 (1.86–4.13), 0 %  | 3.01 (2.57–3.83), 0 %  |
| AR(1) of 0.8 (9)                                  | 3.26 (1.64–5.62), 0 %  | 4.30 (2.47–8.14), 0 %  | 6.33 (4.06–11.18), 0 % | 8.70 (6.96–12.05), 0 % |
| AR(1) of 0.5 plus as much white noise (2)         | 1.46 (0.82–2.64), 0 %  | 1.58 (1.00–2.99), 0 %  | 1.74 (1.23–2.73), 0 %  | 2.00 (1.72–2.41), 0 %  |
| AR(1) of −0.5 (1/3)                               | 0.42 (0.12–0.89), 20 % | 0.34 (0.09–0.78), 11 % | 0.31 (0.11–0.60), 4 %  | 0.33 (0.25–0.43), 0 %  |
| AR(2) of 1 and −0.5, oscillating (5/3)            | 2.25 (1.87–3.26), 0 %  | 2.32 (2.04–3.26), 0 %  | 2.33 (2.18–2.91), 0 %  | 2.34 (2.30–2.54), 0 %  |
| Residuals of a fit of 3 to independent noise (1)  | 0.70 (0.25–1.25), 2 %  | 0.83 (0.44–1.27), 0 %  | 0.98 (0.71–1.28), 0 %  | 1.02 (0.92–1.16), 0 %  |
| Residuals of a fit of 10 to independent noise (1) | 0.24 (0.05–0.54), 29 % | 0.40 (0.13–0.77), 4 %  | 0.78 (0.56–1.04), 0 %  | 0.99 (0.90–1.10), 0 %  |
| Residuals of a fit of 10 to an AR(1) of 0.5 (3)   | 0.32 (0.07–0.66), 19 % | 0.88 (0.43–1.33), 0 %  | 1.67 (1.39–2.11), 0 %  | 2.72 (2.40–3.23), 0 %  |

- **`τ` measures the initial positive stretch of the correlation.** An oscillating correlation, as the AR(2), stops the sequence at its first negative pair, so `τ` tends to `1 + 2 r(1)`, 7/3, above the 5/3 of the process. A cycle at a long lag, such as a weekly pattern in daily data, does not reach `τ` at all: the autocorrelations and `Q̃` with `max_lag` at least that lag show it.
- **A fit absorbs dependence.** Its residuals are less dependent than its noise, the more so with many parameters and few positions: the band of `Refitted` reads them against refits that absorbed the same.

## What dependence means

One realization cannot tell noise that depends on its neighbours, such as the noise of the dynamics, from a model that misses a feature of the data: both leave runs of residuals of one sign. A dependent series says that the observations count less than their number, or that the model misses something; a better model or a method that holds for dependent noise answers it, and the guide pages below say which.

Several series give several tests, one per series, so some reject by chance. Holm's correction keeps the chance of any false rejection at the level: sort the `m` p-values, multiply the `i`-th smallest by `m − i + 1`, and keep each at least the one before.

```python
def holm(p_values: dict[str, float]) -> dict[str, float]:
    """The p-values of Holm's correction."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for i, (label, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[label] = running
    return adjusted


print(holm({s.label: s.p_value for s in together.series}))  # {'south': 1.0, 'north': 1.0}
```

## The methods that read dependence

| Method                                           | How it reads dependence                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The bootstraps and the jackknife                 | `block_length`: dependence inside a block is kept ([Blocks and their length](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#blocks-and-their-length))                                                                                                                                                                                |
| `Sandwich`                                       | Scores summed over blocks ([Sandwich](https://munqu.github.io/commol/latest/guide/uncertainty/local/#sandwich))                                                                                                                                                                                                                                           |
| `InformationMatchingEta`, `BootstrapCoverageEta` | The scale `η` of a Gibbs posterior from blocks of scores or of data. Information matching corrects `η` for blocks that share the curvature unevenly and for the degrees of freedom they read; with few blocks its intervals still cover less than their level ([What `η` means](https://munqu.github.io/commol/latest/guide/uncertainty/smc/#what-means)) |
| `EnbPI`                                          | Prediction intervals of a series with dependent errors ([EnbPI](https://munqu.github.io/commol/latest/guide/uncertainty/conformal/#enbpi))                                                                                                                                                                                                                |
| `Laplace`, `BayesianPosterior`, `block_length=1` | Each observation independent: an information factor above 1 says their intervals are too narrow                                                                                                                                                                                                                                                           |

## Errors

| Situation                                                                                    | What happens                                                    |
| -------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `values` is a list of lists                                                                  | `TypeError` that points to `residual_dependence`                |
| `values` is not one finite value per observation                                             | `ValueError`                                                    |
| `max_lag`, `replicates` or `randomizations` below 1, or `max_lag` a text other than `"auto"` | `ValueError` that names the rule                                |
| `ResidualNoise(...)`                                                                         | `TypeError`                                                     |
| `NoNoise()` with more than one randomization                                                 | `ValueError`                                                    |
| `DistributionNoise()` with a loss, or an observed value outside the support of its model     | `ValueError`                                                    |
| `Refitted` with a sample of many draws, or observations other than the problem's             | `ValueError` that points to `point_sample` or to `AtParameters` |
| A predictive of a `SensitivityAnalysis`                                                      | `ValueError`: an analysis has no data                           |
| Every refit fails                                                                            | `RuntimeError`                                                  |

## Checklist

- [ ] You test the residuals of a fit with `Refitted` and those of held-out data with `AtParameters`.
- [ ] A test of many draws reads `Calibrator.point_sample(fit)`.
- [ ] Several series are corrected for several tests.
- [ ] A short series that does not reject is read with `τ` and its bias, not as independent.

## Next steps

| To learn                                                | Read                                                                                                      |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Quantile residuals and the other scores of a predictive | [Uncertainty](https://munqu.github.io/commol/latest/guide/uncertainty/#scores)                            |
| Methods for dependent noise: blocks                     | [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/#blocks-and-their-length) |
| A covariance that holds for dependent noise             | [Local Approximations](https://munqu.github.io/commol/latest/guide/uncertainty/local/#sandwich)           |
| The scale of a Gibbs posterior                          | [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/#gibbs-posterior)    |

## References

- Dalla, V., Giraitis, L., & Phillips, P. C. B. (2022). Robust tests for white noise and cross-correlation. *Econometric Theory*, 38(5), 913–941.
- Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236–244.
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science*, 7(4), 473–483.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Parzen, E. (1963). On spectral analysis with missing observations and amplitude modulation. *Sankhyā A*, 25, 383–392.
- Stoffer, D. S., & Toloi, C. M. C. (1992). A note on the Ljung–Box–Pierce portmanteau statistic with missing data. *Statistics & Probability Letters*, 13(5), 391–396.
