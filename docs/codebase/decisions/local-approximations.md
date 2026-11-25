# Local approximations

[Back to Design decisions](../design-decisions.md)

How the curvature, the Laplace approximation, the sandwich and the Laplace
mixture read a fit, and where PSIS lives. Their scores, Fisher information and
distribution functions are on [Scores and distribution functions](scores.md).

## The curvature lives on the unbounded scale

`Curvature` (`crates/commol-inference/src/local/curvature.rs`) holds
`H_u = D · H_θ · D`, with `D = dθ/du` at the fit on the scale of
`Transform` (`transform.rs`: a logit with two bounds, a log with one, the
identity with none), centred on the fit mapped to that scale, with no
curvature of the log Jacobian. Every draw maps back with `from_unbounded`,
so it lies inside the bounds; every density on that scale, as the importance
weights of the mixture, adds `log_jacobian`. A fit on a bound is refused with
a message that names the parameter and points to a profile likelihood: its
variance on that scale is infinite. The one rule of a value on a bound,
`on_bound` in `transform.rs`, which the profiles also read, is `1e-8` of the
width of two finite bounds, or of `max(1, |bound|)` with one. The tolerance
comes from the optimizer: Nelder-Mead searches on a sine (`SineTransform`) of the unbounded
coordinate, which approaches a bound quadratically in `u` and ends about
`1e-15` of the width from it; `1e-8` of the width is a step `δu ≈ 2e-4`, far
below any posterior spread. The mixture skips such an end and notes it; when
the end fits at least as well as the worst kept component (its objective is
not above), the mixture misses an optimum the data support, and the sample
is marked unreliable, since `k̂` cannot see mass the proposal never
reaches.

| Alternative | Why not |
|---|---|
| The scale of the parameters | A normal there puts mass outside the bounds near them. |
| A truncated normal on the scale of the parameters | It biases the mean and breaks the delta method. |
| Centring on the mode of the density on the unbounded scale, Jacobian included | It moves the centre away from the fit for no gain: importance sampling corrects the difference. |
| A fit on a bound by exact equality | An optimizer that the bound stops ends a few roundings from it, where the variance on the unbounded scale is huge but finite. |
| Every skipped end marks the sample unreliable | An end on a bound that fits worse than every component holds little mass; the note is enough. |

**Guards:** `crates/commol-inference/tests/test_local.rs::draws_mapped_back_stay_inside_the_bounds`,
`crates/commol-inference/tests/test_local.rs::an_end_on_a_bound_better_than_a_component_marks_the_sample_unreliable`,
`crates/commol-inference/tests/test_local.rs::an_end_on_a_bound_worse_than_every_component_is_only_noted`,
`tests/uncertainty/test_local.py::TestCurvature::test_a_fit_on_a_bound_is_refused`,
`tests/uncertainty/test_local.py::TestResults::test_an_end_on_a_bound_gives_no_component`.

## Gauss–Newton by default, the Hessian on request

`Curvature::gauss_newton` reads the expected curvature of each observation's
term over its arguments, the mean and the noise arguments of its model
(`Noise::information`), through the Jacobian of `SensitivitySource` and the
identity of each noise parameter: `Jᵀ W J` on the model parameters, with the
cross terms where they do not vanish (a log-normal of a `Mean` center, the
two arguments of a Student t). `Curvature::hessian` takes the central
Hessian of the objective with given steps of the `finitediff` crate of the
argmin fork, with steps `δ^{1/4} · dθ/du` (`δ^{1/4} · max(|θ|, 1)` with no
bound), `δ` the output accuracy of the engine, so `θ ± 2h` stays inside the
bounds. Each entry of either carries an estimated error; an entry above 1 %
of `√(H_jj H_kk)` is a note. The threshold measures the error on the scale
of a correlation, so an off-diagonal entry near 0 is judged against its row
and column; a note changes no result.

| Alternative | Why not |
|---|---|
| The Hessian always | `4p² + 1` runs, and indefinite away from the optimum. |
| Gauss–Newton always | Wrong when the residuals are large or the mean is wrong, where the observed and expected curvature differ. |
| The fixed step `√(2ε)` of `finitediff` | With an ODE solved to `rtol`, an error of about `accuracy / h²`. |
| Differences written in commol | The fork keeps the numerical methods; commol chooses the steps from its engine. |

**Guards:** `crates/commol-inference/tests/test_local.rs::curvatures_of_a_linear_model_equal_the_information`,
`crates/commol-inference/tests/test_local.rs::the_hessian_of_a_linear_model_equals_the_information`,
`crates/commol-calibration/tests/test_likelihood.rs::fisher_information_of_each_model_has_its_closed_form`.

## The curvature refuses a prediction or a derivative that is not finite

A `SensitivitySource` ([one port](identifiability.md#sensitivities-come-from-one-port))
may give entries that are not finite, as a difference of finite runs that
overflows, so `Curvature::gauss_newton` finds the first
(`Sensitivities::first_not_finite`) before it reads any term, and refuses it
with a `MethodError::Failed` that names the observation, its part and, for a
derivative, the parameter (`check_finite`). The curvature does not show an
entry that it does not read: a derivative in a term of no curvature (a weight
of 0, a noise whose information underflows) and the prediction of a loss,
whose curvature is finite with any weight. The gradient of the term is then
not finite, in the meat of the sandwich and in information matching. The rule
does not read the weights, as a run that is not finite is refused whatever
the weights (`ForwardModel::run_into`), and the predictions and derivatives
come from the same runs. It therefore also refuses a Laplace approximation
whose matrix is right without the entry: one of the Hessian, or one with the
entry in a term of weight 0. A loss with no curvature is refused before the
derivatives (`Discrepancy::check_curvature`), since no other units give it
one. A curvature that overflows with finite derivatives, or that has an
infinite information, is refused after it is summed.

| Alternative | Why not |
|---|---|
| Check in `FiniteDifferences` | Another source of `SensitivitySource`, a public trait, could bring them back to the curvature. |
| Check that the gradients are finite | They do not say which part is not finite, and a Laplace approximation reads none of them. With a weight of 0, the gradient of an observation model is 0 at any prediction, so it does not show one that is not finite. |
| Skip the terms of weight 0 | An observation would be refused or not by its weight, when a run that is not finite never is. |
| Skip what the method at hand does not read | The curvature is one object that each method reads in part, so each would need its own rule. |

**Guards:** `crates/commol-inference/tests/test_local.rs::a_jacobian_that_is_not_finite_is_refused`,
`crates/commol-inference/tests/test_local.rs::a_prediction_that_is_not_finite_is_refused`,
`crates/commol-inference/tests/test_local.rs::a_difference_of_finite_runs_that_overflows_is_refused`,
`crates/commol-inference/tests/test_local.rs::a_derivative_too_large_for_its_square_has_no_finite_curvature`,
`crates/commol-inference/tests/test_local.rs::a_loss_with_no_curvature_is_refused_before_a_derivative_that_is_not_finite`,
`crates/commol-inference/src/sensitivity/mod.rs::the_first_entry_that_is_not_finite_names_its_place`,
`tests/uncertainty/test_local.py::TestSandwich::test_sandwich_refuses_derivatives_that_are_not_finite`.

## The sandwich sums scores over blocks

`sandwich` (`local/sandwich.rs`) gives `H⁻¹ B H⁻¹` with
`B = G/(G − 1) · Σ_b s_b s_bᵀ` over the non-overlapping blocks of
`BlockScheme` (CR1), and its draws and intervals read a Student t with
`G − 1` degrees of freedom. `G ≤ p` is refused, since the block scores of a
fit sum to zero and `B` would have rank below `p`.
[The user gives the length of a block](dependence.md#the-user-gives-the-length-of-a-block).
Information matching (`smc/gibbs.rs`) reads the same `H` as `J` and the same
block scores for `I`, so it takes a likelihood and holds under dependence; it
reads them only through a trace, so it needs two blocks, not more than `p`
([below](#information-matching-reads-the-meat-through-a-trace)).

| Alternative | Why not |
|---|---|
| The independent sandwich (White) | Too narrow when the residuals are autocorrelated. |
| Newey–West | It needs lags across series at irregular positions, and does not reuse the blocks. |
| A warning below a fixed number of blocks | An arbitrary threshold; the t widens the intervals by the right amount. |
| CR2 with Bell–McCaffrey degrees of freedom | Linear theory and the leverage of each block; kept for when a coverage study needs it. Information matching reads the leverage of its blocks ([below](#information-matching-reads-the-meat-through-a-trace)). |

**Guards:** `tests/uncertainty/test_local.py::TestSandwich`,
`tests/uncertainty/test_coverage.py::TestSandwich`,
`tests/uncertainty/test_information_matching.py::TestInformationMatching`.

## Information matching reads the meat through a trace

`information_matching_eta` (`smc/gibbs.rs`) gives `η = p / tr(J⁻¹ I)`, the
magnitude adjustment of a composite likelihood (Pauli, Racugno and Ventura
2011; Ribatet, Cooley and Davison 2012), with `J` the Gauss–Newton `H` and
`I` the variance of the block scores `s_b`, both on the unbounded scale,
where the trace has the value it has on the scale of the parameters. It
equals `tr(J I⁻¹ J) / tr(J)` when `I = J/η`. It refuses one block, whose
score of a fit is zero, a trace of 0, when every score is 0, a trace below
`10⁻¹⁰` times that of one block per observation, when each parameter's
scores fall inside one block, and blocks that each hold all the curvature
of their parameters.

The trace reads at most `p(G − 1)` degrees of freedom for `G` blocks, fewer
when the blocks share the curvature unevenly (a slope, a parameter read in
one part of the series, a short last block), and `η` is its reciprocal.
With `A_b = J⁻¹ J_b`, `J_b` the Gauss–Newton curvature of block `b`
(`Curvature::block_matrix`), and `L = Σ_b tr(A_b²)`, the trace is
`Σ_b s_bᵀ J⁻¹ s_b · p / (p − L)`, the leverage of the blocks in place of the
`G/(G − 1)` of CR1, which it equals when every `J_b = J/G`; with independent
scores of the variance `J/η` its mean is `p/η`. `η` is then scaled by
`m(ν) = median(χ²_ν)/ν`, with Satterthwaite's degrees of freedom of the
trace `ν = (p − L)² / (L − 2 Σ_b tr(A_b³) + tr((Σ_b A_b²)²))`, `p(G − 1)`
for equal blocks, so that its median over data sets is its value. A numpy
simulation of the estimator (sum of squares, noise of sd 1, 10⁵ data sets;
the first row exact) gives the median of `η` over its value:

| Design | `ν` | CR1 | Leverage | CR2 | CR3 | Leverage and `m(ν)` |
|---|---|---|---|---|---|---|
| One constant, 24 positions, blocks of 12 | 1 | 2.20 | 2.20 | 2.20 | 1.10 | 1.00 |
| A line, 24 positions, blocks of 8 | 1.5 | 3.74 | 1.70 | 1.79 | 0.43 | 1.02 |
| A line, 26 positions, blocks of 8 (last of 2) | 2.4 | 2.34 | 1.38 | 1.42 | 0.54 | 1.02 |
| A line, 60 positions of AR(1) noise of 0.3, blocks of 12 | 3.0 | 2.01 | 1.36 | 1.41 | 0.74 | 1.07 |
| 10 constants, 35 positions, blocks of 18 | 10 | 1.08 | 1.08 | 1.08 | 0.54 | 1.01 |

With few degrees of freedom `η` still varies, which the posterior does not
hold: 90 % intervals of the line cover about 87 % in 8 blocks of 3 (`ν ≈ 5`;
CR1 81 %) and 83 % in 3 blocks of 8 (CR1 62 %). The result gives `ν`
(`GibbsResult.degrees_of_freedom`), and below `FEW_DEGREES_OF_FREEDOM = 10`,
within a relative rounding of `1e-9`, `InformationCalibration::few_coverage`
gives the coverage to expect of 90 % intervals (`FEW_COVERAGE_LEVEL`), which
the runner words in a note of `diagnostics.notes` and logs as a warning. The
trace is about `(p − L)/η · χ²_ν/ν`, independent of the fit, so the coverage
is `P(|t_ν| ≤ z √(ν / median(χ²_ν)))`: 75 % at `ν = 1`, 86 % at 5, 88 % at 10
and 89 % at 20. At 10 it is 2 points below the level. The numpy simulation of
one constant agrees within 1 point at `ν` of 1, 2, 3, 5 and 10, and the
lines, whose blocks share the curvature unevenly, cover up to 4 points more.
`ν` counts the scores of the parameters and the blocks as independent, so it
is no guarantee: noise shared between series, a scale per series, or blocks
shorter than the dependence undercover at a large `ν`, which the guide says.

| Alternative | Why not |
|---|---|
| `tr(J I⁻¹ J) / tr(J)` (Holmes and Walker 2017; Lyddon, Holmes and Walker 2019) | It inverts `I`: the inverse of an estimate from `G` block scores of `p` parameters, `G − 1` degrees of freedom after the fit, has the mean `(G − 1)/(G − p − 2)` times `I⁻¹`, none for `G ≤ p + 2`, so 12 blocks for 10 parameters give an `η` about 16 times too large, and `G ≤ p` has no inverse at all. |
| Correct the inverse by `(G − p − 2)/(G − 1)` | It still needs `G > p + 2`, so blocks short enough for that are too short for the dependence. |
| The leverage alone, or CR2, `(I − A_b)^{-1/2}` on each block score (Bell and McCaffrey 2002) | The trace has its mean (0.94 of it under the AR(1) noise), but `η`, its reciprocal, a median 1.4 to 2.2 times its value below about 3 degrees of freedom; CR2 also needs a matrix root per block and is infinite for a block that holds all the curvature of a parameter. |
| CR3, `(I − A_b)^{-1}`, the jackknife | Too large a trace: a median `η` of 0.43 to 0.77 times its value for a line in few blocks. |
| A warning below a number of degrees of freedom in place of the correction | It leaves `η` too large. The warning is on top of it. |
| A warning by the number of blocks | With independent series the coverage follows `ν`, not the count: 2 blocks of 10 constants read 10. With noise shared between series few blocks do undercover (15 constants that share 90 % of the variance: 80 % in 3 blocks, 86 % in 9), but `G` does not say whether they share it. |
| Another threshold, or none | A threshold of 5 is silent at 86 %; above 10 the loss is below 2 points, inside the error of the formula with uneven blocks. `ν` is in the result whatever its size. |
| A setting of the level, or the level asked for later | The note is written before any interval is asked for; the guide gives 50 %, 90 % and 95 %, and the formula. |
| The threshold and the wording in Python | The engine holds the rule and the numbers; the runner words them, as for the size of a sample (`FEW_DRAWS_PER_PARAMETER`). |
| An error below a number of degrees of freedom | The posterior is usable with its coverage known, and a problem of few positions may have no more to read. |
| `η` scaled so the intervals of a level hold `t_ν` | `η` would depend on the level of the intervals asked for later, and its median would be far below its value. |

**Guards:** `crates/commol-inference/tests/test_smc.rs::information_matching_gives_eta_near_its_value_with_few_blocks_per_parameter`, `crates/commol-inference/tests/test_smc.rs::information_matching_gives_eta_near_its_value_with_blocks_of_uneven_curvature`, `crates/commol-inference/tests/test_smc.rs::information_matching_reads_the_degrees_of_freedom_of_its_blocks`,
`crates/commol-inference/tests/test_smc.rs::information_matching_gives_its_degrees_of_freedom`,
`crates/commol-inference/tests/test_smc.rs::information_matching_needs_two_blocks_and_scores_that_are_not_all_0`, `tests/uncertainty/test_information_matching.py::TestInformationMatching`, `tests/uncertainty/test_coverage.py::TestInformationMatching`.

## The Laplace approximation needs observation models

`laplace_approximation` (`local/approximation.rs`) refuses a loss: a loss has
no noise scale, so its curvature is no inverse variance. The message names
`NormalObservations(sd="sd")` with a noise parameter, whose fit estimates
the scale by maximum likelihood (`SSE/n`, narrower than `SSE/(n − p)` by
`√(n/(n − p))`), and the sandwich, which needs no noise model. A result
holds only data: `LocalResult.delta_band(calibrator, positions, confidence_level)`
reads the problem from the calibrator it is given, through `CalibrationPort`
(`_calibration/method.py`), so a result read back from JSON gives the same
band, and the calibrator knows no method of `commol.local`.

| Alternative | Why not |
|---|---|
| A variance scale `SSE / (n − p)` | It assumes normal, independent noise of one size, which a loss does not state. |
| `LaplaceResult.delta_band` with a private reference to the problem | A result read back from JSON has none. |
| `Calibrator.delta_band(result, …)` | The calibrator would import `commol.local`, a package above it, and grow with each method that gives a band. |

**Guards:** `tests/uncertainty/test_local.py::TestLaplace::test_laplace_refuses_a_loss`,
`tests/uncertainty/test_local.py::TestLaplace::test_a_result_read_from_json_gives_the_same_delta_band`,
`tests/uncertainty/test_local.py::test_a_delta_band_comes_from_the_local_module`.

## PSIS lives in commol-inference

`smooth` (`crates/commol-inference/src/psis.rs`) follows Vehtari et al.
(2024) as ArviZ does: the tail of `⌊3√S⌋` or `⌊S/5⌋` weights, the
generalized Pareto fit of Zhang and Stephens (2009) with a prior on `k`, the
quantiles at `(r + ½)/M` kept at most the largest weight. Its test compares
`k̂` and the smoothed weights with values computed once with ArviZ and
stored in the test. `importance_sample` (`importance.rs`) takes any
`Proposal` and any `Target`; the mixture is one proposal.

| Alternative | Why not |
|---|---|
| ArviZ | A large Python dependency for one function, and the weights would leave Rust and come back. |
| The argmin fork | PSIS is a statistical estimator, not optimization. |
| A crate of its own | Every reader is in `commol-inference`; no Rust crate implements it. |

**Guards:** `crates/commol-inference/tests/test_local.rs::psis_matches_the_reference_values`,
`crates/commol-inference/tests/test_local.rs::banana_target_gives_a_large_k_hat`,
`crates/commol-inference/tests/test_local.rs::mixture_finds_both_modes_in_proportion`.
