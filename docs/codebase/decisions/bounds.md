# Fits near a bound

[Back to Design decisions](../design-decisions.md)

What the local approximations and information matching do with a fit close to
a bound of a parameter. A fit on a bound is refused by the rule of
[Curvature](curvature.md#the-curvature-lives-on-the-unbounded-scale); the
methods are on [Local approximations](local-approximations.md).

## A bound within 3 sd of the fit is a note

`Gaussian::near_bounds` (`crates/commol-inference/src/local/gaussian.rs`)
gives each bound whose distance from the centre is below `NEAR_BOUND = 3`
standard deviations, read on the scale of the parameters, as a `NearBound`.
One formatter, `NearBound::head`, words the parameter, the distance and the
bound. `Gaussian::bound_notes` adds what the draws of `Laplace` and `Sandwich`
show and the pointer to a profile likelihood, and the binding appends them to
the notes of the curvature (`PyLocalRun::of`, `py-commol/src/local.rs`).
`laplace_mixture` adds the end of the component and its own consequence: the
importance weights correct the skew as far as `k̂` says, and with a bound at
0.49 sd its weighted 90 % interval is `[9.638, 10.080]` against the exact
`[9.628, 10.080]` (0.05 sd: `[9.613, 9.997]` against `[9.604, 9.997]`), so
it points to no profile. The runners of Python log every note as a warning.

The scale is `(dθ/du)_j √Σ_jj` with `Σ` the covariance that the draws read:
`H_u⁻¹` for Laplace, `(s·H_u)⁻¹` for a component of the mixture, and the
sandwich for the sandwich, whose grouped t it scales. With `σ = √Σ_jj` and the
centre `u` on the unbounded scale, the distance `d` in that scale is
`(1 + eᵘ)/σ` from the lower bound of a logit, `(1 + e⁻ᵘ)/σ` from its upper
bound and `1/σ` from the bound of a log: no product with `dθ/du`, which
rounds near a bound. A parameter with no bound has none. A normal reads `d`;
a Student t of `ν_j` degrees of freedom reads `z = Φ⁻¹(1 − P(T_ν > d))`, at
most `d`, where a normal leaves the mass that the t leaves past the bound, and an
infinite `ν` is a normal. So the rule is the mass past the bound, 0.135 % at
3 sd, whatever the tails. The words give the distance with two decimals, never
rounded up to 3, or two digits below 0.01, and say when it is read from a t.
Only a fit within `ON_BOUND` of its bound is refused.

The covariance and `delta_band` describe the fit at any distance from a bound;
the draws do not. A flat prior and a normal likelihood cut by the bound give a
90 % interval of the draws 1.05, 1.18, 1.94 and 6.0 times the width of the
exact one at 3, 2, 1 and 0.5 sd, whose far end sits at 2.19, 2.55, 4.18 and
12.9 sd, where the exact one sits at 1.65 to 1.82. A natural bound, a Poisson
rate at 0, gives 1.01, 1.02 and 1.14 times at 3, 2 and 1 sd. Of the 33 fits of `Laplace` and
`Sandwich` in the tests and the guide, 8 have a bound within 3 sd, and none
within 1.

| Alternative | Why not |
|---|---|
| A refusal below some number of sd | The covariance and the band hold, and a natural bound skews the draws little at 1 sd: the reader judges from the note. |
| A threshold of 1 or 2 sd | At 2 sd the interval is already 1.18 times as wide. |
| A share of the width of the bounds, as `ON_BOUND` | It says where an optimizer that the bound stops ends: a fit at `1e-6` of the width has draws over the whole width. |
| `√(H_θ⁻¹)_jj` for the sandwich | With a loss it is no variance, and the draws read the sandwich. |
| The scale `√Σ_jj` of the t as its sd | It under-notes with few degrees of freedom: at `ν = 1` a bound 3.27 units away has 9.4 % past it, and at `ν = 2` one 4.46 units away 2.3 %, with no note, while the 99 % interval of the draws spans the whole width. |
| The variance of the t, `σ² ν/(ν − 2)` | It has no value for `ν ≤ 2`, jumps near 2, and still under-notes. |
| One wording for every method | The weighted sample of the mixture is near the exact one: its draws do not spread far, and a profile is no remedy it needs. |
| The check in `Curvature` | The curvature does not know the covariance that is drawn, which the sandwich and the curvature scale of a component change. |
| Truncated or reflected draws | They change the distribution that the covariance describes. |

**Guards:** `crates/commol-inference/tests/test_local.rs::a_fit_within_a_few_sd_of_a_bound_is_noted`,
`tests/uncertainty/test_fit_near_a_bound.py::TestLaplace`,
`tests/uncertainty/test_fit_near_a_bound.py::TestSandwich`,
`tests/uncertainty/test_fit_near_a_bound.py::TestLaplaceMixture`,
`crates/commol-inference/src/local/gaussian.rs::a_distance_in_sd_never_reads_as_the_threshold`,
`tests/uncertainty/test_local.py::TestCurvature::test_a_fit_on_its_bound_points_to_a_profile`.

## Information matching reads the curvature on the scale of the parameters

`information_matching_eta` (`smc/gibbs.rs`) takes its curvature from
`Curvature::gauss_newton_on_the_parameter_scale`: the Gauss–Newton curvature
with the identity transform, whose scales are powers of two and which refuses
no fit on a bound. It reads no covariance on the unbounded scale, only traces
of matrices in their units, and those do not read the transform: inside the
bounds `η` is the one that the curvature of `Curvature::gauss_newton` gives,
and on a bound it is a number. There is no note. When the optimum lies past
the bound, the scores of the fit do not sum to 0, so the trace is too large
and `η` too small: with 24 values of sd 1 in blocks of 3 and 4000 data sets,
the median of `η` is 1.03, 0.86, 0.62, 0.41 and 0.20, where 1 is right, with
the optimum 0, 1, 2, 3 and 5 sd past the bound. The guide says so.

| Alternative | Why not |
|---|---|
| Refuse a fit on a bound, as the local approximations do | `tr(J⁻¹ I)` does not depend on the scale of a parameter: the refusal reads a variance that `η` does not use. |
| Skip the check and keep the unbounded scale | A scale of 0 on the bound, which every reader of `Curvature::scales` would have to guard. |
| A note of `η` near a bound | `η` holds near a bound that does not cut the optimum, and the fit alone does not say whether it cuts it. |

**Guards:** `crates/commol-inference/tests/test_smc.rs::information_matching_gives_eta_at_a_fit_near_or_on_its_bound`,
`tests/uncertainty/test_fit_near_a_bound.py::TestInformationMatching`.
