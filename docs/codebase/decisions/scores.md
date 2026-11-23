# Scores and distribution functions

[Back to Design decisions](../design-decisions.md)

Where the scores, the Fisher information and the distribution functions of
the observation models live, and what they give at the ends of a support. The
methods that read them are on [Local approximations](local-approximations.md).

## Scores and information live in commol-calibration

`Discrepancy::term_derivatives` (`crates/commol-calibration/src/discrepancy.rs`)
gives each observation's gradient and expected curvature over its arguments:
`−2 w n² r` and `2 w n²` for the sum of squares, the same times
`c = 1 / (2 √(W · S))` for the root mean squared error (its derivative per
unit of `S`), `−w` times `Noise::scores` and `w` times `Noise::information`
for an observation model. A term of several parts reads the density of its
reading (`sum_scores` and `sum_information` of `likelihood/sums.rs`): its
scores over the mean of each part, which the curvature chains with the block
of the parts, and its expected information; for `"exact"` both come from the
recursion and from its derivative in `φ`, the information summed over the
counts from 0 until the terms above the mean are negligible, with the recursion tilted towards `M` plus 10 standard deviations. The information of the moments
under `"exact"` is off by up to −15 %/+27 % and gives `η` of 0.90–0.96 for a
correct model. The mean absolute error is refused: its sandwich
needs a density of the residuals at 0 and a bandwidth. The sandwich,
information matching and later gradient methods read them.

A count at an end of the means of its model has the limit of its score,
the one-sided derivative of its log density (0 there): `−1` for a count of 0
at a mean of 0 (Poisson, negative binomial with a `φ` score of 0, binomial),
`1` for a binomial count of `n` at a mean of `n`; a sum of parts of mean 0
reads the same through its total or the exact recursion. Its information
stays infinite for the mean. When the model holds the mean there (a series
that stays 0 for a while), its Jacobian is 0, so the term adds nothing to the
gradients, the curvature (`add_term`) or the floors of a profile (a part with
no prediction error is skipped). The sandwich and information matching then
give the same result with and without it for a given `block_length` at
positions other series share; it still counts in the blocks and in the
residuals of `"auto"` (#517). `Likelihood::term_derivatives` refuses a score
that is not finite, as it refuses a mean outside its model, naming the
observation (`MethodError::Invalid`): no probability at the fit, outside the
support, or an overflow next to the edge. No reader of the terms sums a NaN.
Information matching reads an inverse of the curvature that is not finite as
singular.

| Alternative | Why not |
|---|---|
| Derivatives in each method | Every method would repeat the models and their noise arguments. |
| A kernel density for the absolute error | It hides a bandwidth choice; the bootstrap needs none. |
| Drop the observations at the end of their means before the scores | The density and the information already read them; a second rule would let the readers disagree. |
| Skip a Jacobian row of 0 in the gradient of a term | The score stays NaN, apart from its density, and an infinite score of a value with no probability would pass unreported. |
| Refuse a score that is not finite in each reader of the gradients | One rule in several methods, and the floors of a profile read the derivatives without a curvature. |

**Guards:** `crates/commol-calibration/src/likelihood/density.rs::the_poisson_score_of_a_count_of_zero_at_a_mean_of_zero_is_its_limit`, `crates/commol-calibration/src/likelihood/density.rs::the_negative_binomial_scores_of_a_count_of_zero_at_a_mean_of_zero_are_their_limits`, `crates/commol-calibration/src/likelihood/density.rs::the_binomial_scores_at_the_ends_of_its_trials_are_their_limits`,
`crates/commol-calibration/tests/test_likelihood.rs::sums_of_parts_of_mean_zero_have_the_limit_of_their_scores`, `crates/commol-calibration/tests/test_likelihood.rs::terms_name_a_score_that_is_not_finite`,
`crates/commol-inference/tests/test_local.rs::a_count_of_zero_at_a_mean_of_zero_adds_no_score`, `crates/commol-inference/tests/test_local.rs::a_count_with_no_probability_at_its_mean_has_no_curvature`,
`tests/uncertainty/test_smc.py::test_information_matching_ignores_observations_of_mean_and_value_zero`, `tests/uncertainty/test_smc.py::test_information_matching_refuses_a_curvature_singular_to_working_precision`, `tests/uncertainty/test_profiles.py::test_a_count_of_zero_at_a_mean_of_zero_keeps_the_floors`,
`crates/commol-calibration/tests/test_likelihood.rs::scores_equal_the_differences_of_the_terms`,
`crates/commol-inference/tests/test_local.rs::the_gradient_of_a_sum_is_that_of_its_density`,
`crates/commol-inference/tests/test_local.rs::the_information_of_a_sum_is_that_of_its_density`,
`tests/uncertainty/test_information_matching.py::TestInformationMatching::test_a_correct_model_with_sums_gives_one`.

## Distribution functions live in commol-calibration

`Noise::tails` (`likelihood/density.rs`) gives `P(Y < y)`, `p(y)` and
`P(Y > y)` of every model, `Noise::sum_tails` (`likelihood/sums.rs`) those of
a sum of parts by its reading (the cumulative sums of the recursion for
`"exact"`), and `Likelihood::tails` those of each observation, at the nearest
mean its model allows. Both tails come back, each accurate in its own tail:
`Φ(8.3)` is 1 in `f64`, so a quantile residual read from `F` alone would be
infinite in the upper tail. The special functions live whole in
`crates/commol-statistics/src/special.rs`, one copy for the distributions of
the priors and for the models: the factor of the incomplete gamma and beta
reads Stirling's series from a shape of 10, the incomplete beta takes `x` and
`1 − x`, and `p(y)` of a count model is that factor, so a tail of a count of
`10⁴` keeps 10 digits. The series and continued fraction of the incomplete
gamma take up to `16 √a` terms, so a Poisson of mean `10⁸` keeps them too. A
negative binomial of `φ ≤ 1e-8` has the tails of the Poisson. Two limits
remain, from the conditioning of the continued fraction of the beta at a
large shape: a negative binomial of size `1/φ` keeps about `10⁻¹⁷/φ`, and a
Student t about `10⁻¹⁶ ν`, as its log density. An `"exact"` sum reads
`P(Y > y)` as a complement down to 0.01, and sums the counts above `y`
beyond that, `O(n²)` in the last count. The predictive mixes the tails of its draws (`ObservationBounds`,
`crates/commol-inference/src/scores.rs`), the one primitive of the PIT, the
quantile residuals and every later reader.

| Alternative | Why not |
|---|---|
| A distribution function next to `draw_observation` in commol-inference | The calibration crate could not read it for the sums of parts. |
| `F` alone | Its upper tail rounds to 1, and a residual there is infinite. |
| The PIT from the draws of the noise | It carries the Monte Carlo error of the draws, and a sample of one point gives 0 or 1. |

**Guards:** `crates/commol-calibration/tests/test_likelihood.rs::the_tails_of_each_model_match_reference_values`,
`crates/commol-calibration/tests/test_likelihood.rs::the_tails_at_the_edges_of_each_model`,
`crates/commol-calibration/tests/test_parts.rs::the_tails_of_a_sum`,
`crates/commol-inference/tests/test_predictive.rs::quantile_residuals_of_unseen_data_are_standard_normal`.
