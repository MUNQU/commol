# Profiles

[Back to Design decisions](../design-decisions.md)

How a profile steps away from the fit, which threshold reads its statistic,
and when its shape gives a verdict.

## A profile steps with the adaptive step of Raue

`ProfileTracer` (`crates/commol-inference/src/profiles/tracer.rs`) walks each
side of each quantity from the fit, all in parallel. It knows no optimizer and
no threshold rule: a `Subproblem` holds the quantity, and the caller gives the
threshold. The first step is 1 % of the width of the bounds (of `|value at the
fit|` for a prediction); a step that rises by more than `2 q T` is rejected and
halved, one that rises by less than `q T / 2` doubles the next (`q = 0.1`,
`T` the threshold, at least the tolerance). Each refit starts from every
parameter, and the multiplier of a held prediction, extrapolated along the
line through the two points before. A side ends past the threshold, at a
bound, after `max_steps` points, or at the fifth miss of a held prediction
(20 outer iterations of the augmented Lagrangian that leave `|g − c|` above
a thousandth of the step).
The end of the interval is the root of the quadratic through the last three
points of the side, or of the line through the last two.

| Alternative | Why not |
|---|---|
| A fixed grid | Too coarse near the fit or too slow far from it, and blind to jumps. |
| Integration of the profile path (Chen and Jennrich 2002) | It needs exact derivatives (#135). |
| Only the ends of the interval (Venzon and Moolgavkar 1988) | No trace for the verdicts and the plots. |
| The previous point as the start | More iterations on a curved ridge, for no saving. |
| The line through the last two points as the end | Its error on a parabola is a share of the step; the quadratic is exact there. |

**Guards:** `crates/commol-inference/tests/test_profiles.rs::profile_of_a_linear_model_is_a_parabola`,
`tests/uncertainty/test_profiles.py::test_linear_normal_profiles_are_exact`.

## A threshold states the noise

`Threshold` has no default. `chi_square` gives `½ χ²₁(confidence_level)` on
`−log L` and refuses a loss; `beale` gives
`S(θ̂) · F(1, n − p; confidence_level) / (n − p)` and needs
the loss `"sse"`; `bootstrap_calibrated` calibrates the statistic itself
(prepivoting, Beran 1988): replicate `b` of the bootstrap, rebuilt from
`replicate_seed(seed, b)` as `ReplicateRunner::perturbations` builds it, is fitted and
then fitted again with the quantity held at its value at the original fit, and
the threshold is the quantile `confidence_level` of the rises
(`bootstrap_thresholds`, `threshold.rs`).

| Alternative | Why not |
|---|---|
| The profile of the original data at the values of the replicates | Not a pivot: it gives the coverage of the percentile bootstrap, and a threshold near 0 on a flat profile. |
| One threshold from `L(θ̂_b) − L(θ̂)` | It calibrates a joint region, not a pointwise interval. |
| Only `chi_square` | It drops every problem with a loss. |
| Only `beale` | Exact only with independent Gaussian errors. |

**Guards:** `tests/uncertainty/test_profiles.py` (`TestThresholds`),
`tests/uncertainty/test_profiles_coverage.py`.

## A verdict needs a threshold above the tolerance

`Profile::verdict` (`verdict.rs`) reads the ends and the statistics of a
profile, at the tolerance `ε = max(τ, ν)` (`ProfileTracer::tolerance`). `τ`
is the tolerance of the optimizer: the `sd_tolerance` of Nelder-Mead as an
absolute amount, since argmin stops when the sample standard deviation of the
costs of the simplex falls below it; for a swarm with the improvement stop
rule, `tol_abs + tol_rel · |f̂|`, the improvement the rule allows at the fit,
since the swarm stops when its best cost improves by less; for a swarm
without it, `1e-6 · max(1, |f̂|)`, since the swarm has no tolerance on the
cost (the radius rule reads positions). `ν` is the noise floor: the error
of one evaluation of the objective at the fit, propagated from the accuracy
of a run, `Σ |∂ℓᵢ/∂mᵢ| · eᵢ` with `eᵢ = rtol · |mᵢ| + k · atol` for an
observation that sums `k` outputs, the term derivatives of
`CalibrationProblem::term_derivatives` and the `output_accuracy` and
`output_absolute_accuracy` of the engine (`ProfileTracer::noise_floor`);
solvers hold their tolerances per step, so `ν` is an estimate, not a bound. A
point whose refit has a free parameter on a bound, within `ON_BOUND = 1e-8`
of the width of its bounds (the tolerance of #133), that it is not on at the
fit has left the box of the parameters (`ProfilePoint::on_bound`): the bound,
not the data, makes its rise. `structurally_non_identifiable`: each side is
flat within `ε` over its points inside the box, has one past the fit, and
ends at a bound, an unreachable prediction or a point outside the box; the
result notes where the ridge leaves the box (`Profile::box_exits`).
`identifiable`: otherwise, both sides cross. `practically_non_identifiable`:
the rest. A crossing whose point has a free parameter on a bound it is not
on over the whole side from the fit comes from the box
(`Profile::box_crossings`): the side counts as a bound end, and the interval
end stays with a note. A parameter that stays on its bound from the fit to
the crossing is not seen, so the verdict holds within the box. A threshold at or below `ε` gives no verdict, and the result says
why. With a stochastic engine a flat profile reads
`practically_non_identifiable` (`ProfileTracer::verdict`).

| Alternative | Why not |
|---|---|
| A verdict from a threshold of 0 | A rise of numerical error would read as information. |
| `τ` scaled by `max(1, |f̂|)` | argmin compares the spread of the costs, an absolute amount; a `−log L` far from 0 would hide real rises. |
| Every point, outside the box too | A ridge `k1 · s = c` leaves a box unless its bounds match `c` exactly, so a flat ridge would read as practically non-identifiable. |
| Any parameter on a bound | A parameter whose fit sits on its bound would put every point outside the box, and any profile would read as flat. |
| A box exit before the crossing demotes it | A ridge that leaves the box and then rises through the data would lose a correct `identifiable`. |

**Guards:** `verdict::tests`, `crates/commol-inference/tests/test_profile_structure.rs`
(`the_noise_floor_propagates_the_accuracy_of_a_run`,
`a_threshold_at_or_below_the_tolerance_reads_no_structure`),
`crates/commol-inference/tests/test_profiles.rs::the_tolerance_follows_the_optimizer`,
`tests/uncertainty/test_profile_verdicts.py` (`TestShapes`).

## A flat profile is read by resetting the others

A profile is flat because other parameters compensate the parameter along
it, or because nothing the data read depends on it; both satisfy "not
determined alone", and Raue et al. (2009) read relations from the paths of
the profiles. `ProfileTracer::read_structure` (`resets.rs`) tells them
apart with no optimizer: the parameter is set to each value of its profile
inside the box and to both ends of its box, the others to their fitted
values, and the objective, evaluated once, is compared with the objective at
the fit `f̂`, feasibility from `Score.feasible`; an evaluation that fails is
an infeasible reset. Two evaluations differ by up to the evaluation floor
`2 · (ν₁ + ν₂) + 2 · γₙ · Σ |ℓᵢ|` (`ProfileTracer::evaluation_floor`): `ν₁`
is `ν`, `ν₂ = Σ ½ |ℓᵢ''| eᵢ²` from the expected curvature of
`TermDerivatives` (an estimate for a model that is not normal), and
`γₙ = n·u / (1 − n·u)` bounds the rounding of the sum of the `n` terms,
constants of `−log L` included. With no floor, as for the mean absolute error
or the root mean squared error of a perfect fit, there is no structural
verdict, with a note.

Every reset feasible and within the floor of `f̂` gives `non_identifiable`,
unless a free parameter sits on its bound at the fit; since the minimum of
the profile is at most each reset, this proves the profile flat from above,
and its own verdict reads `structurally_non_identifiable`. A profile flat
within `ε` with a feasible reset inside the box above `f̂ + T` gives
`in_combination`; the ends of the box only rule out `non_identifiable`. Below
the fit by more than the floor, with a stochastic engine, or otherwise, a
flat profile gets no structural verdict, the report gives the practical
`poorly_determined`, and a note says why. A parameter is named in the reason
when resetting it alone, at its largest displacement on the scale of
`Transform`, raises the objective of that point above `T`; the parameters not
named, noise parameters included, are then reset together on each side, and
a rise above the floor (two evaluations) makes the list "may be incomplete".
`combinations` keeps the algebraic combinations of the structural rank. Every
verdict holds with the other parameters at their fitted values. The error of
a refit at the fit, `|δ|` (`ProfileTracer::refit_error`), is a diagnostic
only, with a note when it is above `ε`: the flatness keeps `ε`, and a too
small `ε` falls back to the practical verdict.

Conflicts with `StructuralRank` remain, and the merge shows them as
`conflicting`: the rank reads the generic rank over a Latin hypercube across
the box, a profile one ridge through the fit; the rank cuts the projection on
the null space at 0.95 and 0.05, so in `k1 + k2` with `k1 ≫ k2` it calls `k2`
non-identifiable while its profile reads `in_combination`; and a parameter
with no effect at the fit of the others but an effect elsewhere reads
`non_identifiable` for the profile.

| Alternative | Why not |
|---|---|
| A flat profile is always `non_identifiable` | It conflicts with the rank on every product of parameters. |
| The parameters that move along the profile | A parameter nothing reads moves by chance, and its noise reads as a combination. |
| A reset at every point for every parameter | It multiplies the cost of a profile by the parameters. |
| Each reset against the profile at the same point | The profile carries the error of the optimizer; `f̂` and the reset are two evaluations, whose difference the floor bounds. |
| `|δ|` as the tolerance | Two samples of the error of the optimizer are no bound on it. |

**Guards:** `crates/commol-inference/tests/test_profile_structure.rs`,
`tests/uncertainty/test_profile_verdicts.py` (`TestStructure`, `TestReport`).
