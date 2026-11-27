# Sequential Monte Carlo: Gibbs Posteriors and ABC

[Back to Uncertainty](index.md)

These methods give a posterior of the parameters with no noise model. The
**Gibbs posterior** turns the loss of the calibration into a posterior; **ABC**
(approximate Bayesian computation) keeps the parameters whose runs come close
to the data. Use them when you can state priors but not how the observations
scatter around the model. One sequential Monte Carlo (SMC) sampler runs both,
in parallel in Rust (`crates/commol-inference/src/smc/`), through
`Calibrator.uncertainty`. The same sampler runs `BayesianPosterior`, the
posterior of [observation models](observation-models.md), when you can state
how the observations scatter.

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
)
from commol.smc import ABCSMC, GibbsPosterior, InformationMatchingEta, LossDistance
from commol.distributions import Priors


def flow_model(engine: str):
    """A flow c from A to B each step: absolute, since the rate names A."""
    return (
        ModelBuilder(name="Flow")
        .add_bin(id="A", name="A")
        .add_bin(id="B", name="B")
        .add_parameter(id="c", value=None)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="c * A / A")
        .build(
            Aggregate(bin_amounts={"A": 1000, "B": 0}),
            engine=engine,
        )
    )


# The flow of each of 30 steps, 10 plus a noise of sd 1.
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
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=500),
    seed=3,
)
calibrator = Calibrator(Simulation(flow_model("DifferenceEquations")), problem)

gibbs = calibrator.uncertainty(
    GibbsPosterior(
        priors=Priors.uniform_on_bounds(),
        eta=InformationMatchingEta(block_length=1),
        particles=500,
        target_cess=0.5,
        resample_threshold=0.5,
        stay_probability=0.05,
    )
)
print(round(gibbs.eta, 2))  # 0.69: median(χ²₂₉)/(2 Σ r²), about 1/(2σ̂²)
lower, upper = gibbs.sample.interval(0.9)["c"]
print(lower < float(np.mean(flows)) < upper)  # True

# ABC on the same data, from the chain binomial: each run draws its own flows.
stochastic = problem.model_copy(update={"replicates": 1, "replicate_summary": "mean"})
abc = Calibrator(Simulation(flow_model("StochasticSteps")), stochastic).uncertainty(
    ABCSMC(
        priors=Priors.uniform_on_bounds(),
        distance=LossDistance(),
        particles=300,
        target_cess=0.5,
        min_acceptance=0.05,
        stay_probability=0.05,
    )
)
print(abc.levels[0] > abc.levels[-1])  # True: the tolerance falls
print(abc.diagnostics.notes[0][:44])  # Each distance reads one fresh replicate of t
```

## Priors

Every method of this page takes its `priors` with no default: what you know
about each parameter before the data is a modelling choice. Give a mapping of
parameter to distribution, or `Priors.uniform_on_bounds()`. A parameter
without a prior, or a prior of a parameter that is not calibrated, raises a
`ValueError` that names it.

Each distribution is truncated to the bounds of its parameter:

| Distribution | Density | Bounds it needs |
|---|---|---|
| `Uniform()` | Uniform on the bounds | Finite |
| `LogUniform()` | `∝ 1/x` on the bounds: `log x` is uniform | Finite, lower above 0 |
| `Beta(a=…, b=…)` | Beta, scaled to the bounds | Finite |
| `Normal(mean=…, sd=…)` | Normal, truncated | Any |
| `LogNormal(mu=…, sigma=…)` | `log x` normal, truncated to `(0, ∞)` too | Upper above 0 |
| `Gamma(shape=…, rate=…)` | Gamma of mean `shape / rate`, truncated to `(0, ∞)` too | Upper above 0 |

A prior whose bounds do not fit raises a `ValueError` that names the
parameter, its bounds and the distributions that fit them: a parameter in
`[0, inf]` takes `Normal`, `LogNormal` or `Gamma`, not `Uniform`. Each
distribution has `log_density(x, bounds)`, `cdf(x, bounds)`,
`quantile(p, bounds)` and `sample(count, bounds, seed)`. A draw inverts the
distribution function of a number of the commol generator, so a seed gives
the same draws on every platform.

```python
from commol.distributions import Gamma, Normal

bounds = (0.0, 30.0)
print(round(Normal(mean=10.0, sd=2.0).cdf(10.0, bounds), 3))  # 0.5
draws = Gamma(shape=4.0, rate=0.4).sample(3, bounds, seed=1)
print(all(0.0 <= draw <= 30.0 for draw in draws))  # True
priors = Priors(distributions={"c": Normal(mean=10.0, sd=2.0)})
```

## How the sampler works

The sampler draws `particles` particles from the priors, runs the model at
each, and keeps one value per particle: the loss for a Gibbs posterior, the
distance for ABC. Each step then:

1. **Chooses the next level.** The level is a temperature from 0 (the priors)
   to 1 (the posterior), or an ABC tolerance that falls. The next one is the
   farthest whose conditional effective sample size (Zhou, Johansen and Aston
   2016) is `target_cess` of the particles, by bisection.
2. **Reweights** each particle from the value it keeps. No model runs.
3. **Resamples** (systematic) when the effective size falls below
   `resample_threshold` of the particles.
4. **Moves** every particle by random-walk Metropolis, in parallel, each from
   its own stream of the generator. The walk proposes on an unbounded scale:
   the logit of the position between two finite bounds, `log(θ − a)` with a
   lower bound `a` only, `log(b − θ)` with an upper bound only, the parameter
   itself with none; so no proposal leaves the bounds. Its covariance is that
   of the particles on that scale, times `2.38² / d`. If the first move accepts
   a share `a` of its proposals, the step moves every particle
   `⌈log c / log(1 − a)⌉` times (Drovandi and Pettitt 2011), with `c` the
   `stay_probability`: each particle moves at least once with probability
   `1 − c`.

| Setting | Meaning | Larger values |
|---|---|---|
| `particles` | The size of the sample | Less Monte Carlo error, more runs |
| `target_cess` | The share of the particles each step keeps | More, smaller steps |
| `resample_threshold` | Resample below this share of effective size (`GibbsPosterior` and `BayesianPosterior`) | More resampling |
| `stay_probability` | The chance that a particle does not move in a step | Fewer moves per step |

None of them has a default: each sets the cost and the accuracy of the run.

## Gibbs posterior

`GibbsPosterior` samples `π(θ) ∝ prior(θ) · exp(−η · loss(θ))`, where `loss`
is the problem's loss with its weights and normalization. It returns a
`GibbsResult`: the sample at temperature 1, the temperatures, the log of
`∫ prior(θ) exp(−η loss(θ)) dθ`, the diagnostics, the `η` it ran with, the fit
a calibration of `η` read (`fit`), the coverage of a bootstrap calibration
(`coverage`), each replicate of a bootstrap calibration whose sampler
failed, `"Replicate <index> failed: <reason>"`, also logged as a warning
(`failed`), the block length of a calibration (`block_length`), and the
degrees of freedom of information matching (`degrees_of_freedom`).

### What `η` means

`η` weighs the data against the priors. The loss has no scale of its own, so
`η` has no default: with `η` too large the posterior is too narrow, with `η`
too small too wide. For the sum of squares of a model with Gaussian noise of
variance `σ²`, `η = 1/(2σ²)` makes the Gibbs posterior the Bayesian posterior.

| `eta` | How it is chosen | Needs | Cost |
|---|---|---|---|
| A number above 0 | By you | Nothing | One sampler |
| `InformationMatchingEta(block_length, starts=1)` | `η = p / tr(J⁻¹ I)` at the fit, for `p` parameters: the Gauss–Newton curvature `J` of the objective against the variance `I` of the scores of its terms, summed over blocks of `block_length` positions (Pauli, Racugno and Ventura 2011), corrected for the share of `J` in each block and for the degrees of freedom of the blocks | `loss_function="sse"` or observation models, a deterministic engine, at least two blocks | A fit, finite differences and one sampler |
| `BootstrapCoverageEta(replicates, block_length, confidence_level, starts=1)` | The posteriors of Bayesian-bootstrap data sets hold the fit at `confidence_level` (Syring and Martin 2019) | Any loss and engine | A fit and one sampler per replicate, then one |

Both calibrations read the fit `Calibrator.run(starts)`, the best of
[several starts](../calibration-optimizers.md#several-starts).

Both take a `block_length` that [you choose](resampling.md#choose-the-length),
and `GibbsResult.block_length` gives it; it is `None` for a number `η`.

Information matching is cheap and local; it needs a sum of smooth terms, so
`"rmse"` (a square root of a sum), `"mae"` (no curvature) and a stochastic
engine raise a `ValueError` that points to the other choices. `J` and `I` are
the curvature and the meat of the [sandwich](local.md#sandwich): `I` sums the
scores over blocks, so it holds when the noise depends on its neighbours
inside a block, and `block_length` has no default: `block_length=1` gives
the independent version, every observation its own block. `I` enters only
through a trace, so `η` needs no inverse of `I`: it takes two blocks or more,
also fewer blocks than parameters, and blocks can be long enough to hold the
dependence. `J` is inverted, so a `J` that the data
[do not determine](local.md#the-curvature-and-the-unbounded-scale) raises a
`ValueError`. `η` is the same in any unit of the parameters. `J` is read on
the scale of the parameters, not on the unbounded scale of the
[local approximations](local.md#the-curvature-and-the-unbounded-scale), so a
fit near a bound or on it gives `η` as any other, with no note. The scores of
a fit that a bound cuts, one whose optimum lies beyond the bound, do not sum
to 0 there, so `η` comes out too small: with 24 values of sd 1 in blocks of 3,
where 1 is right, its median over data sets is 0.86, 0.62, 0.41 and 0.20 with
the optimum 1, 2, 3 and 5 sd past the bound.

The blocks read at most `p(G − 1)` degrees of freedom `ν` about the scale,
for `G` blocks, and fewer when they share the curvature `J` unevenly: a
slope, whose curvature sits at the ends of the series, a parameter that the
data read in one part of the series only, or a short last block. Information
matching corrects for both, with `J_b` the curvature of block `b`: the trace
by `p / (p − Σ_b tr((J⁻¹ J_b)²))`, the leverage of the blocks, which is
`G/(G − 1)` when each block holds `1/G` of `J`; and `η` by
`median(χ²_ν)/ν`, with `ν` the Satterthwaite degrees of freedom of the trace,
because `η` is the reciprocal of the trace and comes out too large more often
than too small. Its median over data sets is then its value: with noise of
sd 1, exactly `1/(2σ²)` for one constant at 24 positions in 2 blocks
(`ν = 1`) and 1.02 times it for a line at 24 positions in 3 blocks
(`ν ≈ 1.5`), where the trace alone gives 2.2 and 3.7 times.

With few degrees of freedom one data set still gives an `η` far from its
value, and the intervals of a high level cover less than it.
`GibbsResult.degrees_of_freedom` gives `ν`, `None` for a number `η` and for
bootstrap coverage. Below `ν = 10` a note in `diagnostics.notes`, also logged
as a warning, gives `ν` and the coverage to expect of 90 % intervals. The
loss grows with the level:

| Level | `ν = 1` | 2 | 3 | 5 | 7 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % | 50 % |
| 90 % | 75 % | 81 % | 84 % | 86 % | 87 % | 88 % | 89 % |
| 95 % | 79 % | 86 % | 89 % | 91 % | 92 % | 93 % | 94 % |

The expected coverage is `P(|t_ν| ≤ z √(ν / median(χ²_ν)))`, with `t_ν` a
Student t and `z` the normal quantile of the level. It is exact for flat
priors away from the bounds, a model linear in its parameters, independent
Gaussian noise of one scale, and blocks that share the curvature evenly.
Blocks that share it unevenly cover more at the same `ν`: the line above
covers about 83 % in 3 blocks of 8 (`ν ≈ 1.5`, 79 % expected) and 87 % in 8
blocks of 3 (`ν ≈ 5`). An informative prior or a bound near the fit also
changes it.

Shorter blocks usually give more degrees of freedom, but must still hold the
dependence of the residuals: blocks too short for it undercover with any
`ν`, and no note. More positions give more only when they add curvature to
more blocks, which the later positions of a fast growth do not.

`ν` says how well the data fix one scale, with the scores of the blocks and
of the parameters counted as independent. Many degrees of freedom, and no
note, do not mean that the intervals cover. They do not when:

- **The series share their noise**, as when one misfit shows in all of them:
  `ν` counts each parameter apart. With 15 constant flows whose noise shares
  90 % of its variance, 35 positions in 3 blocks read `ν = 30` and 90 %
  intervals cover about 80 %; in 9 blocks, 86 %.
- **Each series has a noise of another scale**, or an observation model wrong
  by another factor: one `η` leaves the intervals of some parameters too
  narrow and of others too wide. With 15 constant flows, one of them of
  three times the sd, its intervals cover about 50 %.
- **The blocks are shorter than the dependence**, or the noise is far from
  Gaussian on few observations.

Bootstrap coverage runs one tempered sampler per replicate to a large `η`, reads the
posterior of every smaller `η` by reweighting, so its search runs no model,
and finds for each replicate and parameter the `η` where the interval stops
holding the fit. `η` is the one quantile of commol of the log thresholds at
`1 − c` (with equal weights, as
[`weighted_quantiles`](index.md#the-one-quantile) reads them), which interpolates between two
of them; it needs at least `1/(2(1 − c))` replicates times parameters, so
that the level is not below the smallest threshold.

With a stochastic engine the loss reads the problem's `replicates` with
common random numbers, as the optimizers do, so it is a function of the
parameters.

With [observation models](observation-models.md) the loss is the negative
log-likelihood, so the Gibbs posterior is the fractional posterior
`prior(θ) · L(θ)^η`, and `η = 1` is the posterior of `BayesianPosterior`.
With a correct observation model `I ≈ J`, so information matching gives
`η ≈ 1`; an `η` below 1 says the data are noisier, or more dependent, than
the models say.

## ABC

`ABCSMC` keeps a particle when the distance between its run and the data is
at most a tolerance `ε`, and lowers `ε` step by step (Del Moral, Doucet and
Jasra 2012). With weights of 0 or 1, the share of the particles that a
tolerance keeps is its conditional effective sample size, so each next
tolerance is the weighted quantile of the current distances at `target_cess`:
the smallest distance whose share of the weight at or below it reaches it.
The run stops when the first move of a step accepts less than
`min_acceptance`, and keeps the sample of the tolerance before; it also stops
when no lower tolerance keeps `target_cess` of the particles, as with whole
numbers that tie. `diagnostics.stop_reason` says which.

| `distance` | Value |
|---|---|
| `LossDistance()` | The problem's `loss_function`, with its weights and `normalize_observations`; a problem with observation models has none, so it raises a `ValueError` |
| `SeriesDistance(metric="rmse")` or `"mae"` | `metric` of each series on its residuals divided by the root mean square of its observed values, then the largest over the series |

`SeriesDistance` compares series in different units on one scale: scaling a
series and its prediction by a constant leaves its distance unchanged, so the
series of large values does not decide alone.

With a stochastic engine each distance runs one fresh replicate, seeded from
its particle's stream, and the particle keeps that distance until a move
replaces it: that is what makes ABC sample its target. The problem's
`replicates` and `replicate_summary` do not apply, and
`diagnostics.notes` says so. The problem still needs them, since every
calibration of a stochastic engine does.

`ABCRejection(priors, distance, draws, quantile)` is one step of the same
code with no move: it draws from the priors and keeps the closest share
`quantile` of the draws.

```python
from commol.smc import ABCRejection

rejection = calibrator.uncertainty(
    ABCRejection(
        priors=Priors.uniform_on_bounds(), distance=LossDistance(), draws=400, quantile=0.1
    )
)
kept = [weight > float("-inf") for weight in rejection.sample.log_weights]
print(sum(kept))  # 40
```

## Reading the diagnostics

| Field | One entry per | What to look for |
|---|---|---|
| `levels` | Step | A Gibbs posterior ends at 1; an ABC run lists its falling tolerances |
| `diagnostics.effective_sizes` | Step | Kish's effective size after reweighting; near `target_cess` of the particles |
| `diagnostics.resampled` | Step | Whether the step resampled |
| `diagnostics.acceptance_rates` | Step | The share of the first move's proposals accepted; a small one means many moves |
| `diagnostics.move_counts` | Step | How many times each particle moved; it grows as the acceptance falls |
| `diagnostics.evaluations` | Run | The number of model runs |
| `diagnostics.stop_reason` | Run | Why the run stopped before its last level, or None |
| `diagnostics.notes` | Run | Settings that do not apply, and the coverage to expect when `η` rests on few degrees of freedom |
| `log_evidence` | Run | The log of the product of the mean incremental weights |
| `kept` | Particle | The loss or distance each particle keeps |

```python
print(gibbs.levels[-1])  # 1.0
print(len(gibbs.diagnostics.move_counts) == len(gibbs.levels))  # True
```

A step whose acceptance rate is small moves each particle many times, and the
run costs more; more particles or a larger `target_cess` make the steps
smaller. A sample whose effective size is small gives intervals with a large
Monte Carlo error.

## Checklist

- [ ] Each calibrated parameter has a prior you can defend, on bounds that fit it.
- [ ] `η` is a number you can justify, or one of its calibrations.
- [ ] Information matching has `degrees_of_freedom` of about 10 or more, blocks that hold the dependence of the residuals, and series that neither share their noise nor need another scale each.
- [ ] A Gibbs posterior ends at temperature 1; an ABC run says why it stopped.
- [ ] With a stochastic engine, ABC reads one replicate per particle, not the problem's summary.

## Next steps

| To learn | Read |
|---|---|
| What every method returns and reads | [Uncertainty](index.md) |
| The posterior of observation models | [Observation Models](observation-models.md) |
| Uncertainty with no prior: bootstraps and the jackknife | [Resampling](resampling.md) |
| How far a sample contracts each prior | [Identifiability](identifiability.md#a-sample) |
| The stochastic engines and their replicates | [Stochastic engines](../stochastic.md) |
