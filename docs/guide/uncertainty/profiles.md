# Profile Likelihood

[Back to Uncertainty](index.md)

A profile holds one parameter, or one prediction, at values that step away
from the fit, and fits the other parameters again at each value. How fast the
fit gets worse tells how well the data determine that quantity: the profile
gives a confidence interval of each parameter and each prediction, and a
verdict on its identifiability (Raue et al. 2009; Kreutz et al. 2012). Use it
when the model is not linear in its parameters, where the intervals of a
local approximation fail. Each profile refits with the problem's own
optimizer, in parallel in Rust (`crates/commol-inference/src/profiles/`).

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
from commol.observations import NormalObservations
from commol.profiles import (
    PredictionProfile,
    PredictionSpec,
    ProfileLikelihood,
    Threshold,
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


# B every 4 steps up to step 40, with normal noise of sd 10.
truth = Simulation(basic_model(0.4, 0.1)).run(num_steps=60)
rng = np.random.default_rng(3)
observations = [
    ObservedDataPoint(
        at=step,
        output="B",
        value=max(truth["B"][step] + rng.normal(0.0, 10.0), 0.0),
    )
    for step in range(4, 41, 4)
]
parameters = [
    CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
]
problem = CalibrationProblem(
    observed_data=observations,
    parameters=parameters,
    observation_models={"B": NormalObservations(sd=10.0)},
    optimization_config=NelderMeadConfig(max_iterations=2000, sd_tolerance=1e-10),
    seed=42,
)
calibrator = Calibrator(Simulation(basic_model(None, None)), problem)

# The profile of each parameter, and its 95 % interval.
result = calibrator.uncertainty(
    ProfileLikelihood(parameters=["k1", "k2"], threshold=Threshold.chi_square(0.95))
)
for name in ("k1", "k2"):
    interval = result.intervals[name]
    print(name, round(interval.lower, 4), round(interval.upper, 4), result.verdicts[name])
# k1 0.3929 0.4055 identifiable
# k2 0.0984 0.104 identifiable

# The profile of B at step 20, inside the data, and at step 50, after them.
bands = calibrator.uncertainty(
    PredictionProfile(
        specs=[PredictionSpec(at=step, output="B", label=f"B at {step}") for step in (20, 50)],
        threshold=Threshold.chi_square(0.95),
    )
)
for label, interval in bands.intervals.items():
    print(label, round(interval.lower, 1), round(interval.upper, 1))
# B at 20 407.7 430.1
# B at 50 37.1 44.0
```

## How to read a profile

The **statistic** of a point is the objective of its refit less the objective
of the fit: `−log L` with observation models, the loss otherwise. It is 0 at
the fit and rises as the quantity moves away from it. The **interval** of the
quantity is where the statistic stays below the **threshold**; each interval
is pointwise, for one quantity at a time.

| Field of `ProfileResult` | Holds |
|---|---|
| `profiles[name]` | A `Profile`: the `values`, the `statistics`, every parameter of each refit, the `threshold`, the `ends` of each side and the `evaluations` of the objective |
| `intervals[name]` | A `ProfileInterval`: `lower` and `upper`, `None` on a side that does not cross the threshold |
| `verdicts[name]` | What the shape says ([Verdicts](#verdicts-and-the-identifiability-report)) |
| `fit` | The fit every profile starts from |
| `tolerance` | The numerical tolerance of the statistic, `max(τ, ν)` ([Verdicts](#verdicts-and-the-identifiability-report)) |
| `structure` | What resetting the other parameters says of each flat profile of a parameter (`ProfileStructure`) |
| `notes` | A threshold below the tolerance, or a refit below the fit |
| `failed` | With `Threshold.bootstrap_calibrated`, each replicate that failed, `"Replicate <index> failed: <reason>"`, also logged as a warning; empty with any other threshold |
| `not_converged` | With `Threshold.bootstrap_calibrated`, the replicates kept although their optimizer did not converge |

| Shape | Means |
|---|---|
| A valley that crosses the threshold on both sides | The data determine the quantity; the interval is closed |
| A valley open on one side, up to a bound | The data bound the quantity on one side only |
| Flat over the whole range | The data cannot tell the values apart: another parameter makes up for this one |

A refit below the fit (a negative statistic) means that the fit is not the
optimum; `notes` says so, and a fit with more starts (`starts=n`) fixes it.

```python
profile = result.profiles["k1"]
print(profile.ends, round(profile.threshold, 4))
# ('crossed', 'crossed') 1.9207
```

## Thresholds

The threshold is a required argument: it states what you assume about the
noise.

| Threshold | Statistic | Assumes |
|---|---|---|
| `Threshold.chi_square(confidence_level)` | `−log L` of observation models | The models hold, and `2 × statistic` is near `χ²₁` (Wilks); it gives `½ χ²₁(confidence_level)`. It needs observation models |
| `Threshold.beale(confidence_level)` | A sum of squares, the loss `"sse"` | Independent Gaussian errors of one variance: `S(θ̂) · F(1, n − p; confidence_level) / (n − p)`, exact for a model linear in its parameters |
| `Threshold.bootstrap_calibrated(bootstrap, confidence_level)` | Any loss | That the bootstrap mimics how the data vary |

The bootstrap threshold calibrates the statistic itself (prepivoting, Beran
1988). Each replicate of `bootstrap` is the replicate
`Calibrator.uncertainty(bootstrap)` makes from the problem's seed. Each
replicate has two fits: a free one (`θ̂_b`), and one that holds the quantity
at the value of the fit of the original data. The rise of the replicate is
`PL_b(θ̂ⱼ) − L_b(θ̂_b)`. The threshold of the quantity is the quantile
`confidence_level` of the rises, so each parameter gets its own. It costs one fit and one refit per quantity for each
replicate. Any bootstrap of [Resampling](resampling.md) works, and the
[parametric bootstrap](observation-models.md#maximum-likelihood-against-the-posterior)
with observation models.

```python
from commol.resampling import ResidualBootstrap

loss_problem = CalibrationProblem(
    observed_data=observations,
    parameters=parameters,
    loss_function="sse",
    optimization_config=NelderMeadConfig(max_iterations=2000, sd_tolerance=1e-10),
    seed=42,
)
loss_calibrator = Calibrator(Simulation(basic_model(None, None)), loss_problem)
bootstrap = ResidualBootstrap(
    replicates=100, block_length=1, residual_scale="additive", scheme="moving"
)
for threshold in (
    Threshold.beale(0.95),
    Threshold.bootstrap_calibrated(bootstrap, 0.95),
):
    profiled = loss_calibrator.uncertainty(
        ProfileLikelihood(parameters=["k1"], threshold=threshold)
    )
    interval = profiled.intervals["k1"]
    print(threshold.kind, round(interval.lower, 4), round(interval.upper, 4))
# beale 0.3858 0.4129
# bootstrap_calibrated 0.3886 0.41
```

`Threshold.chi_square` on a problem with a loss raises a `ValueError` that
names the other two; `Threshold.beale` on another loss, or on observation
models, raises one too.

## How a profile is traced

| Step | Rule |
|---|---|
| Start | The fit of `Calibrator.run(starts)` |
| Step | The adaptive step of Raue et al. (2009): it starts at 1 % of the width of the bounds (of the value at the fit, for a prediction), doubles while a step rises by less than half of 10 % of the threshold, and halves when a step rises by more than twice it |
| Start of each refit | Every parameter extrapolated along the line through the two points before, as in Data2Dynamics |
| End of a side | Past the threshold (`"crossed"`), at a bound (`"bound"`), after `max_steps` points (`"max_steps"`), or at a prediction no values within the bounds give (`"unreachable"`) |

Every side of every quantity runs in parallel. A held parameter leaves the
values the optimizer searches, so Nelder-Mead searches one dimension less. A
held prediction `g(θ) = c` is met as an equality constraint, by an augmented
Lagrangian whose inner solver is the problem's optimizer. Its final
multiplier is minus the slope of the profile (`Profile.multipliers`). With a
particle swarm, each refit draws from a seed of its own, derived from the
problem's seed, so a profile repeats; a swarm runs each outer iteration of a
held prediction to its `max_iterations`, or until one of its
[stop rules](../calibration-optimizers.md#stop-rules) holds.

## Prediction profiles

`PredictionProfile(specs, threshold)` profiles the value of each prediction: a
`PredictionSpec` reads an output, or a sum of outputs, at a position, with a
`window`, a `factor_id` and a known `scale`, as an observation does. Its
interval is the set of values the data do not reject (Kreutz et al. 2012);
specs at several positions give a band. A prediction past the data has a wider
interval, as `B at 50` shows in the [quick path](#quick-path). The intervals
are pointwise: a band of them is not a joint region.

## Verdicts and the identifiability report

| Verdict | The profile | In the report |
|---|---|---|
| `identifiable` | Crosses the threshold on both sides | Practical `determined` |
| `practically_non_identifiable` | Stays below on at least one side, up to a bound or `max_steps` | Practical `poorly_determined` |
| `structurally_non_identifiable` | Flat within `tolerance` over its whole range inside the box of the parameters, or proven flat by the resets below | Structural `in_combination` or `non_identifiable` from the resets; with no structural verdict, practical `poorly_determined` |

`tolerance` is `ε = max(τ, ν)`. `τ` is the `sd_tolerance` of Nelder-Mead, an
absolute amount: argmin stops when the costs of its simplex spread by less.
For a particle swarm with the improvement stop rule, `τ` is the improvement
the rule allows at the fit, `tol_abs + tol_rel · |objective at the fit|`: the
swarm stops when its best value improves by less. A swarm without that rule
has no tolerance on the objective, and `τ` is
`1e-6 · max(1, |objective at the fit|)`. A swarm with a
[polish](../calibration-optimizers.md#polish-with-nelder-mead) ends with
Nelder-Mead, so `τ` is the `sd_tolerance` of the polish. `ν`, the noise floor, is the error of one
evaluation of the objective at the fit, from the accuracy of a run:
`Σ |∂ℓᵢ/∂mᵢ| · eᵢ`, with `eᵢ = rtol · |mᵢ| + k · atol` the error of a run on
an observation that sums `k` outputs, its prediction `mᵢ`, and the accuracy
of the engine (machine epsilon for a difference model, the `rtol` and `atol`
of the ODE solver). A solver holds its tolerances at each step, so `ν` is an
estimate, not a bound. A threshold at or below `ε` gives no verdict: the
profile cannot tell a rise from numerical error, and `notes` says so.

The box of the parameters is the space inside their bounds. A free parameter
of a refit is on a bound when it is within `1e-8` of the width of its bounds,
or of the larger of 1 and the size of its one finite bound, the rule of the
[local approximations](local.md).
When it is on a bound that it is not on at the fit, the refit has left the
box: the bound makes its rise, not the data. The flatness reads only the
points inside the box, and a note names the parameter and the value where
the ridge leaves it. When the crossing point of a side has such a parameter,
and the parameter is not on that bound over the whole side, the side crosses
because of the box. It counts as open, and its interval end stays, with a
note. A parameter on its bound at every point from the fit to the crossing
does not count: the verdict then holds within the box.

A profile is flat for one of two reasons: other parameters compensate the
parameter along it, or nothing the data read depends on it. Raue et al.
(2009) read such relations from the paths of the profiles. To tell them
apart with no optimizer, each *reset* puts the parameter at one value of its
profile inside the box, or at one end of its box, and every other parameter
at its fitted value. Each reset evaluates the objective once and compares it
with the objective at the fit, `f̂`. The evaluation floor
`2(ν₁ + ν₂) + 2γₙ Σ|ℓᵢ|` (`ProfileResult.evaluation_floor`) is how
far two evaluations can differ:
`ν₁` is `ν`, `ν₂ = Σ ½ |ℓᵢ''| · eᵢ²` uses the expected curvature of each
term (an estimate for a model that is not normal), and `γₙ · Σ |ℓᵢ|` bounds
the rounding of the sum of the `n` terms, with the constants of `−log L`.

| Resets | Structural verdict |
|---|---|
| Every reset feasible and within the floor of `f̂` | `non_identifiable`, unless another parameter sits on its bound at the fit; the profile then reads `structurally_non_identifiable` |
| A flat profile with a feasible reset inside the box above `f̂ + T` | `in_combination` |
| Otherwise, a profile below the fit by more than the floor, no floor (a loss with no term derivatives), or a stochastic engine | None: practical `poorly_determined` for a flat profile, and a note |

A reset whose evaluation fails counts as infeasible. The ends of the box only
rule out `non_identifiable`. A parameter is named in the reason when resetting
it alone, where it moves most along the profile, raises the objective of that
point above `T`; the parameters not named are then reset together, and a rise
above the floor says that the list may be incomplete. Noise parameters of
observation models are never named. `combinations` stays empty for a
profile: it holds the algebraic combinations of the structural rank. Every
reason says that the verdict holds with the other parameters at their fitted
values. `Profile.refit_error` is the distance of a refit at the fit from
`f̂`, a diagnostic of the optimizer, with a note when it is above `ε`.

What the settings give: with Nelder-Mead at its default `sd_tolerance`
(`1e-6`), the resets prove a parameter with no effect `non_identifiable` on
a difference model and on an ODE model at the default solver tolerances. A
ridge needs its profile flat within `ε`: on an ODE model at the default
tolerances `ν` is near `1e-4` and the profiles of the ridge below read
`in_combination`; on a difference model `ν` is near `1e-13`, a refit lands
about `1e-6` from its minimum, and the profile reads practically
non-identifiable. Its `in_combination` needs an `sd_tolerance` near `ν`, as
the example below uses.

`ProfileIdentifiability(result=profiles)` (`commol.identifiability`), run
through `Calibrator.identifiability`, turns the verdicts into an
[identifiability report](identifiability.md#the-report) with a `profile`
section, which `merge` joins with the other methods. A rate that reads
`k1 * s` cannot tell `k1` from `s`: their profiles are flat on the ridge
`k1 · s = constant`, resetting `s` breaks it, and the profile of `k1` reads
`in_combination`, compensated by `s`, as the structural rank does. The two still disagree in some cases, and the
merged verdict is then `conflicting`:

- the rank reads the rank most points of a Latin hypercube across the whole
  box have, a profile one ridge through the fit;
- the rank calls a parameter non-identifiable when its projection on the null
  space is above 0.95, and identifiable below 0.05: in a sum `k1 + k2` with
  `k1 ≫ k2`, the rank finds `k2` non-identifiable while its profile, which
  `k1` compensates, reads `in_combination`;
- a parameter with no effect while the others sit at their fit, but an
  effect elsewhere in the box, is `non_identifiable` for the profile, and may
  be identifiable for the rank.

```python
from commol.identifiability import ProfileIdentifiability, StructuralRank


def product_model(k1: float | None, s: float | None, k2: float | None):
    return (
        ModelBuilder(name="Product")
        .add_bin(id="A", name="State A")
        .add_bin(id="B", name="State B")
        .add_bin(id="C", name="State C")
        .add_parameter(id="k1", value=k1)
        .add_parameter(id="s", value=s)
        .add_parameter(id="k2", value=k2)
        .add_transition(id="t_ab", source=["A"], target=["B"], rate="k1 * s * A * B / total")
        .add_transition(id="t_bc", source=["B"], target=["C"], rate="k2 * B")
        .build(
            Aggregate(bin_amounts={"A": 990, "B": 10, "C": 0}),
            engine="DifferenceEquations",
        )
    )


product_problem = CalibrationProblem(
    observed_data=observations,
    parameters=[
        CalibrationParameter(id="k1", parameter_type="parameter", min_bound=0.2, max_bound=0.8),
        CalibrationParameter(id="s", parameter_type="parameter", min_bound=0.5, max_bound=2.0),
        CalibrationParameter(id="k2", parameter_type="parameter", min_bound=0.0, max_bound=1.0),
    ],
    observation_models={"B": NormalObservations(sd=10.0)},
    optimization_config=NelderMeadConfig(max_iterations=4000, sd_tolerance=1e-15),
    seed=42,
)
product = Calibrator(Simulation(product_model(None, None, None)), product_problem)
profiles = product.uncertainty(
    ProfileLikelihood(parameters=["k1"], threshold=Threshold.chi_square(0.95))
)
print(profiles.verdicts)
# {'k1': 'structurally_non_identifiable'}
print(profiles.notes[0])
# 'k1': the ridge leaves the box at 0.8 above the fit, where 's' reaches its
# bound; the profile is flat inside the box, and its rise past it comes from
# the bounds, not the data.
print(profiles.structure["k1"].compensated_by)  # ['s']
report = product.identifiability(StructuralRank(points=5, design="observations"))
merged = report.merge(
    product.identifiability(ProfileIdentifiability(result=profiles))
)
print(merged.verdicts["k1"].structural.verdict, merged.verdicts["s"].structural.verdict)
# in_combination in_combination
```

## Plots

`result.plot()` draws one panel per profile: the statistic of each point, the
threshold as a dashed line and the ends of the interval as dotted lines. Like
every figure of commol, it returns the matplotlib `Figure` and takes
`output_file`, `config` and `axes` ([Plotting](../simulations.md#plotting)).

```python
figure = result.plot()
```

## Checklist

- [ ] The threshold states your noise: `chi_square` with observation models,
      `beale` for a sum of squares of independent Gaussian errors,
      `bootstrap_calibrated` otherwise.
- [ ] No note says that a refit went below the fit; if one does, fit with
      more starts.
- [ ] Every verdict is set; a threshold below the tolerance needs data with
      noise, or a tighter `sd_tolerance`.
- [ ] For `in_combination` on a difference model, `sd_tolerance` is near
      the noise floor `ν` (`notes` says why a flat profile has no structural
      verdict, and when a refit is less accurate than the tolerance).
- [ ] A side that ends at `max_steps` is open only because it stopped:
      raise `max_steps`.
- [ ] Read an open or flat profile with the [structural rank](identifiability.md#structural-rank)
      before you call a parameter non-identifiable.

## Next steps

| To learn | Read |
|---|---|
| The other evidence of identifiability, and the report | [Identifiability](identifiability.md) |
| Intervals from refits to resampled data | [Resampling](resampling.md) |
| Observation models and the likelihood | [Observation Models](observation-models.md) |
| The parameter sample, the predictive and its scores | [Uncertainty](index.md) |
