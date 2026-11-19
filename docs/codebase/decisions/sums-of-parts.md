# Sums of parts

[Back to Design decisions](../design-decisions.md)

How an observation model reads the sum of the parts of a window: the density
of each model's sum, and the cap of the exact sum of negative binomials. What
the parts of a window are, and which readers read them, is on
[Observations](observations.md#the-parts-of-a-window-are-observations-of-every-reader).

## A model reads the sum of the parts

`Noise::sum_log_density`, `sum_scores` and `sum_information`
(`crates/commol-calibration/src/likelihood/sums.rs`) read a term from the
means of its parts; with one part they are the functions of one mean. A
Poisson reads `Poisson(M)` and a normal `N(M, K · sd²)`, both exact. A
negative binomial series with parts needs `sum` (`Sum::Exact` or
`Sum::Moments`), and one without parts refuses it (`Likelihood::new`).
`"exact"` is the recursion of Panjer over the parts, in `O(K · y + y²)`, with
`log p(0)` kept apart and the terms scaled down past `1e150`; `"moments"` is
`NB(M, φ · Σμₜ²/M²)`. Every other model refuses parts. The draws read the sum
as the density does (`draw_sum` of `predictive.rs`): each part, summed, or
the matched negative binomial for `"moments"`.

**Rule:** a term of several parts reads the density of their sum, and its
draws, scores and information read that same density; a model with no closed
form for the sum refuses parts.

| Alternative | Why not |
|---|---|
| Moments only | They drift by up to 56 % of an interval with few effective parts, and bias `φ`. |
| Exact only | 1.3·10⁹ operations per evaluation at a total of 50 000. |
| A default `sum`, or a switch by the size of the total | A judgment hidden in the library. |
| Approximate sums of the log-normal and the Student t | No caller ([L8](https://github.com/MUNQU/commol/issues/498#issuecomment-6983148460)). |
| The Poisson below a small `φ` | A jump in the likelihood at the threshold; the recursion, with `log1p(φμ)/(φμ)` and no division by `φ`, reaches the Poisson at `φ = 0` within 6e-14 ([L9](https://github.com/MUNQU/commol/issues/498#issuecomment-6983149187)). |
| Truncating the recursion where `q_max^j < 1e-17` | It saves 10–100× at small `φ` and costs up to 7.6e-3 in `log p` far in the upper tail; only with a declared tolerance and a test of that tail ([L10](https://github.com/MUNQU/commol/issues/498#issuecomment-6983149799)). |

**Guards:** `crates/commol-calibration/tests/test_parts.rs::the_exact_sum_is_the_convolution_of_its_parts`,
`crates/commol-calibration/tests/test_parts.rs::the_moments_of_a_sum`,
`crates/commol-inference/tests/test_predictive.rs::draws_of_a_sum_have_the_variance_of_their_parts`,
`tests/uncertainty/test_observation_sums.py::TestSums`,
`tests/uncertainty/test_part_sums_coverage.py::TestPartSums`.

## An exact sum ends at a fixed count

`"exact"` computes the counts below `MAX_COUNTS = 100 000` (`likelihood/sums.rs`).
`Recursion::extend_to` refuses a count at the cap before computing it. Before
the upper tail and the information start, `ends_within_counts` fails them at
once when `p(MAX_COUNTS − 1) ≥ maxₜ pₜ(n) Π_{s≠t} pₛ(0)` is above `NEGLIGIBLE`.
Between, as at `M = 10⁴, φ = 3`, they run to the cap, then fail. `UnendedSum`
becomes a message that names the observation, its total, `M`, `φ` and the
counts, and points to `sum="moments"`: a `RuntimeError` in every method. Seven
equal parts need 13 902 counts at `M = 10⁴, φ = 0.01`, 81 971 at `φ = 1`, 54 257 at
`M = 10³, φ = 10` (`Noise::sum_information`, release); the tests, at most 2 062.

**Rule:** the exact reading ends within `MAX_COUNTS` or fails with the reason; it never switches to `"moments"` ([D2](https://github.com/MUNQU/commol/issues/498#issuecomment-6979416382)).

| Alternative | Why not |
|---|---|
| No cap | At `φ = 10⁶` the tail spans about `40 φμ` counts: a fit hangs with no message. |
| A cap relative to `M` or the sd | The counts needed grow from 10 to 45 sd as `φ` goes from 0.01 to 10: no multiple bounds the time. |
| Only the check at the cap | 10⁵ counts take 4 s for a density, 12 s for an information, before it fails at `φ = 10⁶`. |

**Guards:** `crates/commol-calibration/tests/test_parts.rs::an_exact_sum_that_does_not_end_fails`, `tests/uncertainty/test_observation_sums.py::TestSums::test_an_exact_sum_that_does_not_end_says_why`.
