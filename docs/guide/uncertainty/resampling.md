# Resampling: Bootstraps and the Jackknife

[Back to Uncertainty](index.md)

These methods give the uncertainty of a calibration with no noise model and no
prior. They fit the model again to reweighted or resampled data, starting from
the fit, and read the spread of the new fits. Use them when you can say how the
data were collected but not how the observations scatter around the model. Each
runs through `Calibrator.uncertainty`, in parallel in Rust
(`crates/commol-inference/src/resampling/`).

## Quick path

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
    scores,
)
from commol.resampling import (
    BayesianBootstrap,
    BlockJackknife,
    ResidualBootstrap,
    WildBootstrap,
)
from commol.observations import NoNoise, ResidualNoise
from commol.uncertainty import Predictive


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

result = calibrator.uncertainty(BayesianBootstrap(replicates=100, block_length=1))
lower, upper = result.sample.interval(0.9)["k"]
print(round(lower, 4), round(upper, 4))  # 0.0483 0.0512
```

`result.fit` is the fit, `result.sample` holds one draw per replicate that
fitted, `result.failed` lists the replicates that failed, and
`result.diagnostics` gives the block length.

## Which method to use

| Method | Changes in each replicate | Assumes | Answers |
|---|---|---|---|
| `BayesianBootstrap` | The weight of each block of observations | The blocks are exchangeable | How far the parameters move when the data are reweighted |
| `ResidualBootstrap` | The observed values: fitted values plus blocks of residuals drawn with replacement | The model is right; the residuals of each block are exchangeable across positions | How far the parameters move under new noise like the fit's |
| `WildBootstrap` | The observed values: each residual times a random sign or Mammen multiplier, one per block | The model is right; the noise has mean zero at each position | The same, when the size of the noise changes along the series |
| `BlockJackknife` | One block left out (weight 0) per fit | The blocks are exchangeable | A standard error and a bias, and which block moves each parameter most |

None of them needs a noise model. The Bayesian bootstrap never changes the
observed values, so it also works when the model is wrong; the residual and
wild bootstraps make new data from the model and its residuals. With
[observation models](observation-models.md), `ParametricBootstrap(replicates)`
draws new data from the models at the fit; it draws no blocks, so its
`diagnostics.block_length` is `None`.

## Blocks and their length

A block holds every observation, of every series, whose position falls in one
window of `block_length` consecutive distinct positions
(`BlockScheme`, `resampling/blocks.rs`). The dependence of the data across
series and along the positions stays inside blocks, so the methods hold for
noise that is correlated in time or between series.

`block_length` is a whole number of at least 1, with no default: it is your
decision, about your data. `1` states that the noise of different positions is
independent. A longer block states that the noise depends on its neighbours no
further than the block holds. Every method that reads blocks, here and on the
other pages, takes it so.

commol does not choose it. The best length changes with the method and with
the number of blocks, and one number has no single answer for several series of
different dependence.

### The length counts positions

- **Of all the series together.** A block holds consecutive distinct positions
  of the data, whichever series has an observation there. With a weekly series
  beside a daily one, the positions are days: a length of 7 holds one value of
  the weekly series, and 4 of its values need 28.
- **Also where no observation carries information.** A series that the model
  holds at exactly 0 until a process starts says nothing about the parameters
  while it is observed at 0: those scores are 0 for the count models
  (Poisson, negative binomial, binomial), for a normal or Student t model
  whose `sd`, `scale` and `df` are numbers, and for a loss without
  `normalize_observations`. A model with a fitted `sd`, `scale` or `df` reads
  them, and so does the normalization of a series: there they change the
  result at any position. Where the scores are 0, at a position that another
  series also observes they change nothing in the `Sandwich` or in
  `InformationMatchingEta`. At a position that only such observations hold, it
  still counts: it takes a place in a window, so it shifts the windows after
  it, and a window of such positions alone is one more block.

### Choose the length

1. **Read the dependence of the residuals of the fit** with
   [`residual_dependence`](dependence.md): for each series its
   autocorrelations, the integrated autocorrelation time `τ` with its band, and
   `cut_lag`, the largest lag in the sum of `τ`. They count steps of the grid
   of that series, so a lag of 3 of the weekly series is 21 days. A fit absorbs
   dependence and a short series hides it ([The bias of `τ`](dependence.md#the-bias-of)):
   a small `τ` does not show that the noise is independent.
2. **Use what you know of the process**: a weekly pattern of the reports, an
   event that moves several positions, the time that a disturbance lasts. A
   cycle at a long lag does not reach `τ`.
3. **Compare the result at several lengths**, with the number of blocks each
   leaves, for the method and the parameters you report.

The same model with a noise that depends on its neighbours, an AR(1) of 0.7:

```python
from commol.dependence import Refitted, residual_dependence

rng = np.random.default_rng(1)
shock = np.zeros(61)
for t in range(1, 61):
    shock[t] = 0.7 * shock[t - 1] + rng.normal(0.0, 0.05)
dependent = [
    ObservedDataPoint(at=t, output="B", value=truth["B"][t] * float(np.exp(shock[t])))
    for t in range(1, 61)
]
dependent_calibrator = Calibrator(
    Simulation(ab_model(None)), problem.model_copy(update={"observed_data": dependent})
)
fit = dependent_calibrator.run()

(series,) = residual_dependence(
    Predictive(dependent_calibrator, dependent_calibrator.point_sample(fit), positions=[1]),
    dependent,
    NoNoise(),
    max_lag=10,
    randomizations=1,
    reference=Refitted(replicates=99),
    confidence_level=0.9,
).series
print([round(r, 2) for r in series.correlations[:4]])  # [0.6, 0.28, 0.16, 0.03]
print(series.p_value, series.cut_lag)  # 0.01 3
print(round(series.tau, 2), [round(b, 2) for b in series.tau_band])  # 3.09 [0.52, 2.05]

for length in (1, 3, 6, 10, 15):
    compared = dependent_calibrator.uncertainty(
        BayesianBootstrap(replicates=300, block_length=length)
    )
    low, high = compared.sample.interval(0.9)["k"]
    blocks = -(-len(dependent) // length)
    print(length, blocks, round(high - low, 4))
# 1 60 0.003
# 3 20 0.0049
# 6 10 0.0058
# 10 6 0.0049
# 15 4 0.003
```

The residuals depend on their neighbours: the test rejects independence, `τ`
is above its band, and the autocorrelations fall to about 0 at lag 4. Blocks
of one position give the narrowest interval, since they ignore that
dependence. From 3 to 10 positions the interval keeps about one width, and
blocks of 15 leave only 4, whose spread says little: the interval narrows
again although the data did not change.

A range of lengths with one width is not a range with the coverage of the
level. Over 2000 data sets of this example, the 90 % interval of `k` covers its
value in this share of them (300 replicates, or 4000 draws of the sandwich):

| `block_length` | 1 | 3 | 6 | 10 | 15 |
|---|---|---|---|---|---|
| Blocks | 60 | 20 | 10 | 6 | 4 |
| `BayesianBootstrap` | 46 % | 61 % | 66 % | 64 % | 62 % |
| `Sandwich` | 48 % | 67 % | 78 % | 84 % | 87 % |

No length brings the bootstrap near 90 %: 60 values with this dependence leave
few independent blocks, and its spread does not allow for that. The sandwich
does, with a Student t of one degree of freedom less than its blocks, and its
interval is also wider with few of them. Comparing lengths shows how far the
choice moves the result, not which length is right: report the length with the
result, say when the result changes with it, and, when the series is short,
that the interval can cover less than its level
([Short series](#short-series)).

A method also sets how few blocks it takes:

| Method | Blocks it needs |
|---|---|
| `BayesianBootstrap`, `WildBootstrap`, `ResidualBootstrap` | None: they take one block, with no note |
| `BootstrapCoverageEta` | None, but with one block its replicates are equal and tell it nothing |
| `BlockJackknife`, `JackknifePlus`, `InformationMatchingEta` | At least 2 |
| `Sandwich` | More than the parameters |
| `EnbPI` | Each left out by at least one replicate |

[Information matching](smc.md#what-means) gives the degrees of freedom that
its blocks read, and warns when they are few; the
[sandwich](local.md#sandwich) reads a Student t with one degree of freedom less
than its blocks.

With a length of all the positions, one block, the bootstraps draw nothing:
each replicate of `BayesianBootstrap`, and of `ResidualBootstrap` with
`"moving"`, has the data of the fit, so its interval has width 0; that of
`WildBootstrap` is about 0 in this example, and `"stationary"` gives its
narrowest. No note says so: read the number of blocks.

A series with two observations at one position is refused by
`ResidualBootstrap`, which draws one residual per position, whatever the
length; the other block methods accept it
([One position, two observations](../calibration-observations.md#one-position-two-observations)).

## Bayesian bootstrap

`BayesianBootstrap(replicates, block_length, starts=1)` draws, for each
replicate, one Dirichlet(1, …, 1) weight per block, scaled to mean 1, and
multiplies the weight of every observation of the block by it. With a loss it
is the loss-likelihood bootstrap of Lyddon, Holmes and Walker (2019); with
`block_length=1`, the Bayesian bootstrap of Rubin (1981). Its variance matches
the sandwich variance of least squares
(`tests/uncertainty/test_resampling.py::TestBayesianBootstrap`).

## Residual block bootstrap

`ResidualBootstrap(replicates, block_length, residual_scale, scheme, starts=1)`
takes the residuals of the fit, joins blocks of them drawn with replacement,
and fits the new data.

| Setting | Values |
|---|---|
| `residual_scale` | `"additive"`: `r = y − ŷ`, new data `ŷ + r*`. `"multiplicative"`: `r = log(y / ŷ)`, new data `ŷ · exp(r*)`; it needs positive observed and fitted values, and keeps the new data positive |
| `scheme` | `"moving"`: blocks of `block_length` consecutive positions (Künsch 1989). `"stationary"`: blocks of geometric length with mean `block_length`, wrapped around the end (Politis and Romano 1994) |

Use the multiplicative scale for counts or amounts that span orders of
magnitude, where the noise grows with the value. On the additive scale a new
value can fall below zero: the method keeps it, and counts it in
`diagnostics.negative_values`. Many of them point to the multiplicative scale.

```python
residual = calibrator.uncertainty(
    ResidualBootstrap(
        replicates=100,
        block_length=1,
        residual_scale="multiplicative",
        scheme="stationary",
    )
)
print(residual.diagnostics.negative_values)  # 0
```

When a series has no observation at a drawn position, it takes the residual of
the nearest position where it has one.

## Wild block bootstrap

`WildBootstrap(replicates, block_length, multipliers, starts=1)` multiplies the
additive residuals of each block by one draw of `multipliers` and fits
`ŷ + r · v`. Every residual stays at its own position, so the method holds when
the size of the noise changes along the series, where the residual bootstrap
mixes small and large residuals.

| `multipliers` | Distribution |
|---|---|
| `"rademacher"` | `±1` with probability 1/2 |
| `"mammen"` | Two values with mean 0, variance 1 and third moment 1 (Mammen 1993); keeps the skewness of the residuals |

```python
wild = calibrator.uncertainty(
    WildBootstrap(replicates=100, block_length=5, multipliers="mammen")
)
print(len(wild.sample))  # 100
```

## Block jackknife

`BlockJackknife(block_length, starts=1)` fits once per block, with that block
left out, and returns a `JackknifeResult`. With `B` blocks it needs `B` fits,
usually far fewer than a bootstrap.

| Field | Meaning |
|---|---|
| `estimate` | The bias-corrected estimate of each parameter |
| `bias` | `(B − 1)(θ̄ − θ̂)`, with `θ̄` the mean of the fits with one block left out |
| `standard_error` | `√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)` |
| `blocks` | The first and last position of each block that fitted |
| `pseudo_values` | `B θ̂ − (B − 1) θ̂₍₋ᵦ₎` per block |
| `influence` | How far leaving each block out moves each parameter, in standard errors |
| `failed` | Each block whose fit failed, `"Block <index> failed: <reason>"`, with blocks numbered from 0 in order of position; also logged as a warning |
| `not_converged` | Fits kept although their optimizer did not converge |
| `diagnostics` | The block length, as for a bootstrap |

The pseudo-values are not draws: their percentiles are not an interval, so the
result has no `ParameterSample`. A block with a large influence holds data the
fit depends on: check it.

```python
jackknife = calibrator.uncertainty(BlockJackknife(block_length=10))
print(jackknife.blocks[:2])  # [(1.0, 10.0), (11.0, 20.0)]
print(len(jackknife.influence["k"]))  # 6
```

## The result

A bootstrap returns a `BootstrapResult(fit, sample, failed, not_converged,
diagnostics)`. The sample has equal weights, and the group of each draw is its
replicate.

| Field | Meaning |
|---|---|
| `failed` | Each replicate that failed, `"Replicate <index> failed: <reason>"`; it is not in the sample, and it is logged as a warning |
| `not_converged` | Replicates kept although their optimizer did not converge |
| `diagnostics.block_length` | The block length the method ran with; `None` for `ParametricBootstrap` |
| `diagnostics.negative_values` | New observed values below zero, over every replicate |

Every result of a method that fits several times lists the fits that failed in
`failed` and counts the fits kept without convergence in `not_converged`, in
the same way: a bootstrap, the jackknife, a profile with a bootstrap threshold
and the conformal methods that refit. When too few fits succeed, the method
raises a `RuntimeError` that says how many of how many fitted, how many it
needs, and the first reasons.

`result.interval_error(confidence_level)` gives the Monte Carlo error of each
bound of the central interval: an order-statistic interval that holds the bound
of infinitely many replicates with about 95 % probability. When it is wide
against the interval, run more replicates.

```python
error = result.interval_error(0.9)["k"]
print(error.lower[0] <= lower <= error.lower[1])  # True
```

## How replicates run

Each replicate starts from the fit. With `starts > 1` it also starts from the
[other starts](../calibration-optimizers.md#several-starts) of the optimizer,
and keeps the end of lowest loss, so a
second basin of the loss shows up instead of adding optimizer noise to the
data's uncertainty. Replicate `b` draws its data and its starts from
streams of the problem's seed, one for the data and one for the fits, at
index `b`: two runs with one seed give one result. A replicate changes only the observed values or the weights
(`CalibrationProblem::with_data`); with `normalize_observations` its
normalization comes from its own values, as the fit's does.

## Prediction intervals

The draws of a bootstrap give a band of the model. `ResidualNoise` adds moving
blocks of residuals of the fit to every draw's prediction of each observation,
which gives a prediction interval with no noise model. It serves the series the
fit has residuals of, at any position, and rejects an observation of any other
series. It and `ResidualBootstrap` draw one residual per position of a series,
so they refuse a series with two observations at one position, in the fit or
in the observations they serve
([One position, two observations](../calibration-observations.md#one-position-two-observations)).

```python
predictive = Predictive(calibrator, result.sample, positions=list(range(81)))
noise = ResidualNoise(
    fit=result.fit,
    block_length=result.diagnostics.block_length,
    residual_scale="multiplicative",
)

print(scores.coverage(predictive, held_out, NoNoise(), confidence_level=0.9).mean)  # 0.05
print(scores.coverage(predictive, held_out, noise, confidence_level=0.9).mean)  # 0.85
```

The band of the model is narrow and covers 1 of the 20 held-out observations;
the prediction interval covers 17, near its level.

## Short series

A block bootstrap reads the dependence of the noise from few blocks. On a short
series its variance is too small by a term of order `1 / block_length`, and its
interval covers less than its level. In the coverage study of
`tests/uncertainty/test_resampling_coverage.py`, a 90 % interval of a series
with AR(1) noise (`φ = 0.5`) covers about 0.8 with 200 positions and about 0.89
with 400. With few positions, prefer `block_length=1` only when the noise is
independent, and report that the interval may be narrow.

## Checklist

- [ ] You chose `block_length` from the dependence of the residuals and the scale of the process, read the result at several lengths, and report the length.
- [ ] You chose the residual scale for the data: multiplicative for values that span orders of magnitude.
- [ ] `failed` is empty, or you know why replicates failed.
- [ ] `not_converged` is 0, or you raised `max_iterations`.
- [ ] `interval_error` is small against the interval.
- [ ] You report a band of the model as a band of the model, and use `ResidualNoise` for a prediction interval.

## Next steps

| To learn | Read |
|---|---|
| The parameter sample, the predictive and the scores | [Uncertainty](index.md) |
| The settings of the optimizer each replicate runs | [Optimizers](../calibration-optimizers.md) |
