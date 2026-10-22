# Local Approximations: Laplace, Sandwich and Importance Sampling

[Back to Uncertainty](index.md)

These methods give fast uncertainty around a fit from the curvature of the
objective there, with a few runs per parameter instead of one fit per
replicate. The **Laplace approximation** is a normal centred on the fit; the
**delta method** gives the band of the outputs from it with no run per draw;
the **sandwich** holds when the noise model is wrong or the noise is
dependent, and needs no noise model; the **Laplace mixture** puts a Laplace
approximation at each optimum of a multi-start fit and corrects it towards
the posterior by importance sampling, with a check of when that fails. They
run in Rust (`crates/commol-inference/src/local/`) through
`Calibrator.uncertainty`.

## Quick path

```python
import numpy as np

from commol import (
    Aggregate,
    CalibrationParameter,
    CalibrationProblem,
    Calibrator,
    Laplace,
    LaplaceMixture,
    LikelihoodTarget,
    ModelBuilder,
    NelderMeadConfig,
    NormalObservations,
    ObservedDataPoint,
    Priors,
    Sandwich,
    Simulation,
)


def flow_model(rate: str):
    """A flow from A to B each step, of the rate `rate`."""
    return (
        ModelBuilder(name="Flow")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="c", value=None)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate=rate)
        .build(
            Aggregate(bin_amounts={"A": 1000, "B": 0}),
            engine="DifferenceEquations",
        )
    )


# The flow of each of 30 steps, 10 plus a normal noise of sd 1.
rng = np.random.default_rng(1)
flows = 10.0 + rng.normal(0.0, 1.0, 30)
problem = CalibrationProblem(
    observed_data=[
        ObservedDataPoint(at=step, output="B", value=float(value), window=1)
        for step, value in enumerate(flows, start=1)
    ],
    parameters=[
        CalibrationParameter(id="c", parameter_type="parameter", min_bound=0.0, max_bound=30.0)
    ],
    observation_models={"B": NormalObservations(sd=1.0)},
    optimization_config=NelderMeadConfig(max_iterations=500),
    seed=3,
)
calibrator = Calibrator(Simulation(flow_model("c * A / A")), problem)

laplace = calibrator.uncertainty(Laplace(draws=2000))
lower, upper = laplace.sample.interval(0.9)["c"]
print(round(lower, 2), round(upper, 2))  # 9.64 10.25

band = calibrator.delta_band(laplace, [10, 20, 30], 0.9)
print([round(value, 1) for value in band.center["B"]])  # [99.3, 198.7, 298.0]

sandwich = calibrator.uncertainty(Sandwich(block_length=1, draws=2000))
print(sandwich.blocks)  # 30
# The noise of these data is smaller than sd = 1: the sandwich reads it.
print(round(sandwich.covariance[0][0] / laplace.covariance[0][0], 2))  # 0.71
```

## The curvature and the unbounded scale

Every method of this page reads the curvature `H` of the objective at the
fit: `−log L` of the [observation models](observation-models.md), or the
loss. `curvature` chooses it:

| `curvature` | What it is | Runs | Use it when |
|---|---|---|---|
| `"gauss_newton"` (default) | The expected curvature: `Jᵀ W J`, with `J` the derivatives of the predictions and `W` the Fisher information of each observation, plus the information of each noise parameter | `4p + 1` (finite differences) | The model fits and the noise model is right or close |
| `"hessian"` | The observed curvature: central differences of the objective | `4p² + 1` | The residuals are large, the mean is wrong, or the fit sits on a constraint |

The Fisher information of each observation model (`Noise::information` in
`crates/commol-calibration/src/likelihood/density.rs`):

| Model | Mean `μ` | Noise arguments |
|---|---|---|
| `NormalObservations` | `1/σ²` | `sd`: `2/σ²` |
| `LogNormalObservations` | `1/(μ²σ²)` | `sigma`: `2/σ²`; with `center="mean"` also `1` and `−1/(μσ)` with `μ` |
| `StudentTObservations` | `(ν+1)/((ν+3)s²)` | `df` and `scale`, and their cross term |
| `PoissonObservations` | `1/μ` | |
| `NegativeBinomialObservations` | `1/(μ + φμ²)` | `overdispersion`: the variance of its score, summed over the counts |
| `BinomialObservations` | `1/(n p (1 − p))`, `p = μ/n` | |

The Hessian steps are `δ^{1/4} · dθ/du` for a parameter with a bound and
`δ^{1/4} · max(|θ|, 1)` for one without, where `δ` is the accuracy of the
engine (machine precision, or the `rtol` of the ODE engine): the step that
balances the truncation error of the differences with the error of the runs,
and keeps every run inside the bounds. Each entry of either curvature has an
estimated error. An entry whose error is above 1 % of `√(H_jj · H_kk)` is
listed in `notes` and logged as a warning: the error on the scale of a
correlation, so an off-diagonal entry near 0 is judged against its row and
column, not against itself. A note changes no result.

Every distribution lives on an **unbounded scale**: the logit of a parameter
between two bounds, `log(θ − a)` with a lower bound only, `log(b − θ)` with an
upper bound only, the parameter itself with none. The curvature there is
`H_u = D · H · D` with `D = dθ/du` at the fit, and `covariance` is on that
scale. Every draw maps back inside the bounds, so a fit near a bound gives a
skewed sample, not draws outside it. A fit **on** a bound of its parameter
(within `1e-8` of the width of its bounds) raises a `ValueError`: its
variance on the unbounded scale is infinite there, and a profile likelihood
describes it (MUNQU/commol#132).

## Laplace approximation and the delta method

`Laplace(draws, starts=1)` fits the problem (`Calibrator.run(starts)`) and
returns a `LaplaceResult`: the fit, the covariance `Σ = H⁻¹` of the normal on the
unbounded scale, the draws mapped back (`sample`, equal weights) and the
notes. It needs `observation_models`: a loss has no noise scale, so its
curvature is no inverse variance, and `Laplace` with a loss raises a
`ValueError`. To get a Laplace approximation, give a model with a noise
parameter, such as `NormalObservations(sd="sd")` with `sd` calibrated, or
use `Sandwich`, which needs none.

Every local approximation sits at the best of its `starts`: the initial guess,
then rows of a Latin hypercube over the bounds, as
[several starts](../calibration-optimizers.md#several-starts) of
`Calibrator.run`. One start (the default) can end at a local minimum; more
starts find the best one of the optima they reach. `Sandwich` takes `starts`
the same way.

With a noise parameter the fit estimates the noise by maximum likelihood:
for `NormalObservations(sd="sd")`, `σ̂² = SSE/n`, not `SSE/(n − p)`. The
Laplace intervals are then narrower than the classical ones by
`√(n/(n − p))`, which is negligible when `n ≫ p`.

`calibrator.delta_band(result, positions, confidence_level)` gives the band of
every
output by the delta method: the prediction at the fit plus and minus the
quantile of the normal times `√(J Σ Jᵀ)`, with `J` the derivatives of the
output on the unbounded scale. It runs the model a few times per parameter,
not once per draw, and for a model linear in its parameters it equals the
band of the draws
(`Predictive(calibrator, result.sample, positions).band(confidence_level)`).
It reads only the fields of the result, so a result read back from JSON
works.

```python
widths = [round(high - low, 2) for low, high in zip(band.lower["B"], band.upper["B"])]
print(widths)  # [6.01, 12.01, 18.02]
```

## Sandwich

`Sandwich(block_length, draws, starts=1)` gives the covariance `H⁻¹ B H⁻¹`, where `B`
sums the outer products of the **scores** of blocks of observations:
`B = G/(G − 1) · Σ_b s_b s_bᵀ`, with `s_b` the sum of the gradients of the
terms of block `b` (the clustered estimator CR1, Cameron and Miller 2015).
The blocks hold every series in a window of `block_length` positions, as the
[bootstraps](resampling.md#blocks-and-their-length) do; `"auto"` chooses the length of
Politis and White (2004) from the residuals of the fit. It returns a
`SandwichResult`: a `LaplaceResult` with `blocks` (`G`), `block_length` and
the estimates of `"auto"`.

What the sandwich protects against depends on `H`:

| `curvature` | It holds when |
|---|---|
| `"gauss_newton"` | The noise model is wrong (its size changes, it is not normal, it is dependent inside a block) but the mean is right |
| `"hessian"` | Also when the mean is wrong: the fit estimates the parameters that minimize the expected objective |

It needs no noise model: with a loss, `H` and the scores are those of the
loss (the sum of squares and the root mean squared error share them, since
rescaling the objective does not change the sandwich).

- **A Student t with `G − 1` degrees of freedom.** With few blocks, `B` is a
  noisy estimate; the t widens the intervals by the right amount, instead of
  a warning at an arbitrary number of blocks. Intervals of the draws and
  `Calibrator.delta_band` both read it.
- **`G ≤ p` is refused.** `B` sums `G` outer products whose sum has rank at
  most `G − 1` after the fit, so with no more blocks than parameters some
  directions would get zero variance. The `ValueError` suggests shorter
  blocks.
- **`loss_function="mae"` is refused.** The absolute error has no curvature:
  its sandwich needs a density of the residuals at 0 and a bandwidth, a
  choice the library would hide. Use a [bootstrap](resampling.md).

## Laplace mixture and importance sampling

A multi-start fit can end at several optima. `LaplaceMixture(draws, target,
starts=1)` runs `Calibrator.run(starts)`, keeps the distinct ends (an end
closer than one Mahalanobis unit to a kept one merges into it), puts a
Laplace approximation at each, weights each by its Laplace evidence, draws
from the mixture, and weights every draw by the target over the mixture on
the unbounded scale. The largest weights are smoothed by Pareto-smoothed
importance sampling (PSIS, Vehtari et al. 2024).

| `target` | Density |
|---|---|
| `LikelihoodTarget(priors)` | `prior(θ) · L(θ)`: the posterior of the observation models |
| `GibbsTarget(priors, eta)` | `prior(θ) · exp(−η · loss(θ))`: the [Gibbs posterior](smc.md#gibbs-posterior) of the loss |

```python
# A flow c² has the two optima ±√ȳ.
square = problem.model_copy(
    update={
        "parameters": [
            CalibrationParameter(
                id="c", parameter_type="parameter", min_bound=-5.0, max_bound=5.0,
                initial_guess=1.0,
            )
        ],
        "observed_data": [
            ObservedDataPoint(at=step, output="B", value=float(value), window=1)
            for step, value in enumerate(4.0 + rng.normal(0.0, 1.0, 10), start=1)
        ],
    }
)
mixture = Calibrator(Simulation(flow_model("c * c * A / A")), square).uncertainty(
    LaplaceMixture(
        starts=8, draws=2000, target=LikelihoodTarget(priors=Priors.uniform_on_bounds())
    )
)
print([round(component.center["c"], 2) for component in mixture.components])  # [2.05, -2.05]
print(round(mixture.k_hat, 2), mixture.reliable)  # 0.24 True
```

It returns an `ImportanceResult`: the fit, the weighted `sample`, the
`components` (centre, weight, covariance, log evidence), `k_hat`, the
`effective_size` and `reliable`.

### Reading `k_hat`

`k_hat` is the estimated shape of the tail of the importance weights: how
far the mixture misses the target.

| `k_hat` | Meaning |
|---|---|
| Up to 0.5 | The weights have a finite variance: the estimates converge at the usual rate |
| 0.5 to 0.7 | They converge, more slowly: use more draws |
| Above 0.7 | The mixture misses mass of the target, as for a curved (banana) posterior: `reliable` is false, `notes` and a warning say so, and the sample should not be used |

A start that ends on a bound gives no component, and `notes` says so. When
its objective is at least as good as that of the worst kept component, the
mixture leaves out an optimum the data support, whose mass `k_hat` cannot
see: `reliable` is false too. An end on a bound that fits worse than every
component only adds the note.

An unreliable sample calls for a method that samples the posterior itself:
`BayesianPosterior` or `GibbsPosterior` ([Sequential Monte Carlo](smc.md)).

## When a local approximation is good enough

| Situation | Good enough? |
|---|---|
| A model near linear in its parameters around the fit, a correct noise model | Laplace, with `Calibrator.delta_band` for the outputs |
| A wrong or unknown noise model, dependent residuals, a right mean | Sandwich with `"gauss_newton"` and blocks longer than the dependence |
| A mean that may be wrong, large residuals | Sandwich with `"hessian"` |
| Several optima of equal or similar loss | Laplace mixture; check `k_hat` |
| A curved posterior, a parameter whose data barely bound it, a fit on a bound | No: sample the posterior ([Sequential Monte Carlo](smc.md)), or a bootstrap ([Resampling](resampling.md)) |

## Checklist

- [ ] The fit is a minimum inside the bounds; no `ValueError` names a parameter on its bound.
- [ ] `notes` is empty, or you know why an entry of the curvature is uncertain.
- [ ] A Laplace approximation has a noise model you trust; otherwise use the sandwich.
- [ ] A sandwich has more blocks than parameters, of a length that holds the dependence.
- [ ] A Laplace mixture has `k_hat` at most 0.7.

## Next steps

| To learn | Read |
|---|---|
| What every method returns and reads | [Uncertainty](index.md) |
| The observation models and their likelihood | [Observation Models](observation-models.md) |
| Posteriors that do not rely on a local shape | [Sequential Monte Carlo](smc.md) |
| Uncertainty by fitting again: bootstraps and the jackknife | [Resampling](resampling.md) |
| Which parameters the data determine | [Identifiability](identifiability.md) |
