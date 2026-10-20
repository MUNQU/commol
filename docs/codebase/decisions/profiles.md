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

`Threshold` has no default. `chi_square` gives `½ χ²₁(level)` on `−log L` and
refuses a loss; `beale` gives `S(θ̂) · F(1, n − p; level) / (n − p)` and needs
the loss `"sse"`; `bootstrap_calibrated` calibrates the statistic itself
(prepivoting, Beran 1988): replicate `b` of the bootstrap, rebuilt from
`replicate_seed(seed, b)` as `ReplicateRunner::run` builds it, is fitted and
then fitted again with the quantity held at its value at the original fit, and
the threshold is the quantile `level` of the rises
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
costs of the simplex falls below it; for a swarm, `1e-6 · max(1, |f̂|)`,
provisional until it has stop rules (#230). `ν` is the noise floor: the error
of one evaluation of the objective at the fit, propagated from the accuracy
of a run, `Σ |∂ℓᵢ/∂mᵢ| · (rtol · |mᵢ| + atol)` with the term derivatives of
`CalibrationProblem::term_derivatives` and the `output_accuracy` and
`output_absolute_accuracy` of the engine (`ProfileTracer::noise_floor`). A
point whose refit has a free parameter on a bound, within `ON_BOUND = 1e-8`
of the width of its bounds (the tolerance of #133), that it is not on at the
fit has left the box of the parameters (`ProfilePoint::on_bound`): the bound,
not the data, makes its rise. `structurally_non_identifiable`: each side is
flat within `ε` over its points inside the box, has one past the fit, and
ends at a bound, an unreachable prediction or a point outside the box; the
result notes where the ridge leaves the box (`Profile::box_exits`).
`identifiable`: otherwise, both sides cross. `practically_non_identifiable`:
the rest. A threshold at or below `ε` gives no verdict, and the result says
why. With a stochastic engine a flat profile reads
`practically_non_identifiable` (`ProfileTracer::verdict`).

| Alternative | Why not |
|---|---|
| A verdict from a threshold of 0 | A rise of numerical error would read as information. |
| `τ` scaled by `max(1, |f̂|)` | argmin compares the spread of the costs, an absolute amount; a `−log L` far from 0 would hide real rises. |
| Every point, outside the box too | A ridge `k1 · s = c` leaves a box unless its bounds match `c` exactly, so a flat ridge would read as practically non-identifiable. |
| Any parameter on a bound | A parameter whose fit sits on its bound would put every point outside the box, and any profile would read as flat. |

**Guards:** `verdict::tests`, `crates/commol-inference/tests/test_profile_structure.rs`
(`the_noise_floor_propagates_the_accuracy_of_a_run`,
`a_threshold_at_or_below_the_tolerance_reads_no_structure`),
`tests/uncertainty/test_profile_verdicts.py` (`TestShapes`).

## A flat profile is read by resetting the others

A profile is flat because other parameters compensate the parameter along
it, or because nothing the data read depends on it; both satisfy "not
determined alone", and Raue et al. (2009) read relations from the paths of
the profiles. `ProfileTracer::read_structure` (`resets.rs`) tells them
apart: at each point inside the box, and at each end of the box where the
profile has a point, the other parameters go back to their fitted values and
the objective is evaluated once; the rise is measured against the profile at
the same point, feasibility from `Score.feasible`. A feasible rise above the
threshold `T` gives the structural `in_combination`; every reset feasible
and within `ε` gives `non_identifiable`, unless a free parameter sits on its
bound at the fit. Otherwise, below the fit by more than `ε`, or with a
stochastic engine, there is no structural verdict, the report gives the
practical `poorly_determined`, and a note says why. A parameter is named in
the reason when resetting it alone, at its largest displacement on the scale
of `Transform`, rises above `T`; the parameters not named, noise parameters
included, are then reset together on each side, and a rise above `ε` makes
the list "may be incomplete". `combinations` keeps the algebraic combinations
of the structural rank. Every verdict holds with the other parameters at
their fitted values.

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

**Guards:** `crates/commol-inference/tests/test_profile_structure.rs`,
`tests/uncertainty/test_profile_verdicts.py` (`TestStructure`, `TestReport`).
