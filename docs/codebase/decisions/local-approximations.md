# Local approximations

[Back to Design decisions](../design-decisions.md)

How the Laplace approximation, the sandwich and the Laplace mixture read a fit,
and where PSIS lives. The curvature they read and its inverse are on
[Curvature](curvature.md); their scores, Fisher information and distribution
functions are on [Scores and distribution functions](scores.md).

## The sandwich sums scores over blocks

`sandwich` (`local/sandwich.rs`) gives `H⁻¹ B H⁻¹` with
`B = G/(G − 1) · Σ_b s_b s_bᵀ` over the non-overlapping blocks of
`BlockScheme` (CR1), and its draws and intervals read a Student t with
`G − 1` degrees of freedom. `G ≤ p` is refused, since the block scores of a
fit sum to zero and `B` would have rank below `p`.
[The user gives the length of a block](dependence.md#the-user-gives-the-length-of-a-block).
It is computed as `G/(G − 1) · Σ_b a_b a_bᵀ`, with the influence
`a_b = H⁻¹ s_b` of each block on the fit (`InverseCurvature::sandwich`):
neither the square of a score nor the product of two inverses is formed, so
the covariance is not refused for the range of its products.
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
| CR2 with Bell–McCaffrey degrees of freedom | Linear theory and the leverage of each block; kept for when a coverage study needs it. Information matching reads the leverage of its blocks ([below](#information-matching-reads-the-meat-through-a-trace)). |

**Guards:** `tests/uncertainty/test_local.py::TestSandwich`,
`tests/uncertainty/test_coverage.py::TestSandwich`,
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
where the trace has the value it has on any scale of the parameters. It
equals `tr(J I⁻¹ J) / tr(J)` when `I = J/η`. It refuses a `J` that
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
