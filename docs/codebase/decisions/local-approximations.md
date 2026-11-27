# Local approximations

[Back to Design decisions](../design-decisions.md)

How the Laplace approximation, the sandwich and the Laplace mixture read a fit,
and where PSIS lives. The curvature they read and its inverse are on
[Curvature](curvature.md); their scores, Fisher information and distribution
functions are on [Scores and distribution functions](scores.md).

## The sandwich sums scores over blocks

`sandwich` (`local/sandwich.rs`) gives
`√(f_j f_k) · (H⁻¹ Σ_b s_b s_bᵀ H⁻¹)_jk` over the non-overlapping blocks of
`BlockScheme`, and its draws and intervals read a Student t of `ν_j` degrees
of freedom for each parameter.
[The user gives the length of a block](dependence.md#the-user-gives-the-length-of-a-block).
The factor and the degrees of freedom are those of Bell and McCaffrey (2002)
for the share of each block in the Gauss–Newton curvature
(`local/shares.rs`): with `J_b` the curvature of block `b`, `J = Σ_b J_b`
and `v_bc = cᵀ J⁻¹ (J_b δ_bc − J_b J⁻¹ J_c) J⁻¹ c` for a direction `c`,
`f_j = (J⁻¹)_jj / Σ_b v_bb` and `ν = (Σ_b v_bb)² / Σ_bc v_bc²`, at least 1.
When every `J_b = J/G`, `f_j = G/(G − 1)` (CR1) and `ν = G − 1`; a parameter
read in few blocks gets a larger factor and fewer degrees of freedom, and
the blocks whose `J_b` is 0 drop out of both sums. The shares are those of Gauss–Newton also
when the bread is the Hessian: they describe where the data read each
parameter, and need no second derivative of the model. The sums are read
with no `G × G` matrix and no difference, so a block that holds nearly all
the curvature of a direction keeps its share. The draws are a grouped t
(Daul et al. 2003): one `χ²` level, read through the quantile of each `ν_j`,
so its margins are the t of each parameter and equal `ν` give the
multivariate t. `delta_band` reads the `ν` of the direction of each output
at each position from the curvature of the blocks at the fit, so a result
read back from JSON gives the same band.

It refuses no more blocks with curvature than parameters, since the block
scores of a fit sum to zero: a block has curvature when its share
`tr(J⁻¹ J_b) / p` is above `ROUNDED_SHARE = 10⁻¹⁰`, the threshold of
information matching, while every block with a `J_b` that is not 0 stays in
`J` and in the sums of `f_j` and `ν_j`, which are continuous in it. It
refuses block scores of rank below `p`: the covariance on a diagonal of 1 is
[singular](curvature.md#one-rule-says-what-the-data-do-not-determine), which
the rounding of a Cholesky factor can hide. Each block's scores are read
after one Gauss–Newton step to the zero of their sum `g`,
`s̃_b = s_b − J_b J⁻¹ g` (`BlockShares::corrected_scores`), so they sum to 0
by construction, as at an exact minimum, where `V_bc` is their covariance:
the gradient that an optimizer leaves does not fill the missing rank, and the
covariance does not follow the error of the fit. And it refuses blocks that each
hold all the curvature of a parameter, where `Σ_b v_bb` is at most
`10⁻¹⁰ (J⁻¹)_jj` and the scores of the parameter sum to zero in each block.
With a Hessian bread, a singular Gauss–Newton `Σ_b J_b` has a message of its
own. It is computed as `√(f_j f_k) · Σ_b a_bj a_bk`, with the
influence `a_b = H⁻¹ s̃_b` of each block on the fit
(`InverseCurvature::sandwich`): neither the square of a score nor the
product of two inverses is formed, so the covariance is not refused for the
range of its products.
Information matching (`smc/gibbs.rs`) reads the same `H` as `J` and the same
block scores for `I`, so it takes a likelihood and holds under dependence; it
reads them only through a trace, so it needs two blocks, not more than `p`
([below](#information-matching-reads-the-meat-through-a-trace)).

| Alternative | Why not |
|---|---|
| The independent sandwich (White) | Too narrow when the residuals are autocorrelated. |
| Newey–West | It needs lags across series at irregular positions, and does not reuse the blocks. |
| A warning below a fixed number of blocks | An arbitrary threshold; the t widens the intervals by the right amount. |
| The product `H⁻¹ · B · H⁻¹` | The square of a score leaves the range where the covariance is a number like any other: a squared error on data of `1e78` has scores of about `1e156`. The product also cancels: with a curvature whose smallest eigenvalue is `1e-8` of the largest, its rounding makes a covariance that is not positive definite. |
| A rank threshold of `G·p·ε` | The floor of the smallest eigenvalue is the optimizer's, from `1e-15` to `1e-5`, not the rounding of a sum, and the refusal would depend on the number of blocks. |
| A rank threshold from the size of the terms | On a unit diagonal it is `p·ε` again. |
| An exact 0 test of each block | A block of weight `1e-300` counts; two blocks with curvature for two parameters then give correlation −1 and `ν = 1`. |
| CR1 with `G − 1` degrees of freedom | It counts a block with no curvature as one that reads the parameters, and gives a parameter read in two blocks the `ν` of all of them: its intervals cover about 61 % where they claim 90 %. |
| The leverage `h_jb` of each block alone | It gives the factor but not the degrees of freedom, which need the covariance of the blocks after the fit. |
| CR2, with a matrix root per block | A root of `I − H_bb` for each block, and a block that holds all its curvature has none; the factor of each parameter keeps the meat of the blocks, with no root. |
| One `ν` for every parameter, from a trace | A parameter read in few blocks would share the `ν` of those read in all. |
| `G` as the sum of the weights of the blocks | It has no account of the uneven shares of the blocks that read a parameter. |
| The smallest `ν` for every parameter | Too wide for the parameters that every block reads. |
| A Gaussian copula with t margins | Its draws are not the multivariate t when every `ν` is equal, and its margins have no common `χ²`. |

**Guards:** `tests/uncertainty/test_local.py::TestSandwich`,
`tests/uncertainty/test_sandwich_blocks.py`,
`tests/uncertainty/test_coverage.py::TestSandwich`,
`crates/commol-inference/tests/test_local.rs::equal_blocks_give_cr1_and_one_degree_of_freedom_fewer_than_the_blocks`,
`crates/commol-inference/tests/test_local.rs::a_sandwich_reads_the_share_of_the_curvature_of_each_block`,
`crates/commol-inference/tests/test_local.rs::a_block_with_no_curvature_changes_nothing`,
`crates/commol-inference/tests/test_local.rs::a_sandwich_needs_more_blocks_with_curvature_than_parameters`,
`crates/commol-inference/tests/test_local.rs::blocks_that_each_hold_all_the_curvature_of_a_parameter_are_refused`,
`crates/commol-inference/tests/test_local.rs::the_shares_of_a_hessian_sandwich_are_those_of_gauss_newton`,
`crates/commol-inference/tests/test_local.rs::block_scores_that_move_together_in_every_block_are_refused`,
`crates/commol-inference/tests/test_local.rs::parameters_read_in_fewer_blocks_than_themselves_are_refused`,
`crates/commol-inference/tests/test_local.rs::the_sandwich_of_a_fit_off_its_optimum_is_that_of_the_optimum`,
`crates/commol-inference/tests/test_local.rs::a_hessian_sandwich_whose_gauss_newton_curvature_is_singular_says_so`,
`crates/commol-inference/tests/test_local.rs::a_band_reads_the_degrees_of_freedom_of_a_direction_beyond_the_range_of_a_quotient`,
`crates/commol-inference/tests/test_local.rs::a_sandwich_reads_scores_whose_squares_underflow`,
`tests/uncertainty/test_local.py::TestSandwich::test_a_sandwich_reads_scores_whose_squares_overflow`,
`tests/uncertainty/test_singular_curvature.py::test_a_sandwich_does_not_depend_on_the_scale_of_a_parameter`,
`tests/uncertainty/test_singular_curvature.py::test_parameters_told_apart_by_little_are_not_refused`,
`tests/uncertainty/test_information_matching.py::TestInformationMatching`.

## Information matching reads the meat through a trace

`information_matching_eta` (`smc/gibbs.rs`) gives `η = p / tr(J⁻¹ I)`, the
magnitude adjustment of a composite likelihood (Pauli, Racugno and Ventura
2011; Ribatet, Cooley and Davison 2012), with `J` the Gauss–Newton `H` and
`I` the variance of the block scores `s_b`, both in the
[units of the curvature](curvature.md#the-curvature-is-kept-in-units-of-its-parameters),
where the trace has the value it has on any scale of the parameters; it gives
`η` [on a bound](bounds.md#information-matching-reads-the-curvature-on-the-scale-of-the-parameters)
too. It equals `tr(J I⁻¹ J) / tr(J)` when `I = J/η`. It refuses a `J` that
[the data do not determine](curvature.md#one-rule-says-what-the-data-do-not-determine), one
block, whose score of a fit is zero, a trace of 0, when every score is 0, a trace below
`10⁻¹⁰` times that of one block per observation, when each parameter's
scores fall inside one block, and blocks that each hold all the curvature
of their parameters. One message refuses scores that are all 0 and scores so
small against the curvature that their squares are below the smallest number
or `η` above the largest: a number is the only `η` that a posterior takes.

The trace reads at most `p(G − 1)` degrees of freedom for `G` blocks, fewer
when the blocks share the curvature unevenly (a slope, a parameter read in
one part of the series, a short last block), and `η` is its reciprocal.
With `A_b = J⁻¹ J_b`, `J_b` the Gauss–Newton curvature of block `b`
(`Curvature::scaled_block_matrix`), and `L = Σ_b tr(A_b²)`, the trace is
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
`crates/commol-inference/tests/test_smc.rs::information_matching_needs_two_blocks_and_scores_that_are_not_all_0`, `crates/commol-inference/tests/test_smc.rs::information_matching_refuses_scores_too_small_for_eta_to_be_a_number`, `crates/commol-inference/tests/test_smc.rs::information_matching_does_not_depend_on_the_unit_of_a_parameter`, `tests/uncertainty/test_information_matching.py::TestInformationMatching`, `tests/uncertainty/test_coverage.py::TestInformationMatching`.

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

## An end that gives no component is noted

`laplace_mixture` (`local/mixture.rs`) makes one component of each end of the
starts that no kept component holds. An end on a bound
([Curvature](curvature.md#the-curvature-lives-on-the-unbounded-scale)), one
whose curvature the data do not determine and one whose covariance is refused
give none, and a note of that end says why. Each goes into one list with its
objective. When one fits at least as well as the worst kept component (its
objective is not above), the mixture misses an optimum: its mass is missing
from the sample, which `k̂` cannot see, so the sample is marked unreliable and
a note names the ends. When no end gives a component, the mixture is refused
with the notes of every end.

| Alternative | Why not |
|---|---|
| Every end that gives no component marks the sample unreliable | One that fits worse than every component holds little mass; the note is enough. |
| Only an end on a bound marks it | An end refused for its curvature or its covariance leaves out the same mass. |
| An error for any end that gives no component | The other ends still give a mixture, and the notes say what it lacks. |

**Guards:** `crates/commol-inference/tests/test_local.rs::an_end_on_a_bound_better_than_a_component_marks_the_sample_unreliable`,
`crates/commol-inference/tests/test_local.rs::an_end_on_a_bound_worse_than_every_component_is_only_noted`,
`crates/commol-inference/tests/test_local.rs::a_refused_end_that_fits_best_marks_the_sample_unreliable`,
`tests/uncertainty/test_local.py::TestResults::test_an_end_on_a_bound_gives_no_component`,
`tests/uncertainty/test_singular_curvature.py::test_a_parameter_the_data_do_not_read_is_refused`.

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
