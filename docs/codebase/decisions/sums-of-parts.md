# Sums of parts

[Back to Design decisions](../design-decisions.md)

How an observation model reads the sum of the parts of a window: the density
of each model's sum, and the tails and the cap of the exact sum of negative
binomials. What the parts of a window are, and which readers read them, is on
[Observations](observations.md#the-parts-of-a-window-are-observations-of-every-reader).

## A model reads the sum of the parts

`Noise::sum_log_density`, `sum_scores` and `sum_information`
(`crates/commol-calibration/src/likelihood/sums.rs`) read a term from the
means of its parts; with one part they are the functions of one mean. A
Poisson reads `Poisson(M)` and a normal `N(M, K · sd²)`, both exact. A
negative binomial series with parts needs `sum` (`Sum::Exact` or
`Sum::Moments`), and one without parts refuses it (`Likelihood::new`).
`"exact"` is the recursion of Panjer over the parts, with `log p(0)` kept
apart and the terms scaled down past `1e150`. The jumps of a part are
geometric, so each part keeps its share of the sum over the jumps as a state
updated once per count: `O(K)` per count, `O(K · y)` up to a total `y`, with
no jump cut and only sums of positive terms for the density. The terms are
tilted to the observed total, so its log density holds where its density is no
number; `tilt` searches between 1 and `y/M`, at any `φ`, and its doc comment says
from which means the terms underflow. `"moments"` is `NB(M, φ · Σμₜ²/M²)`. Every
other model refuses parts. The draws read the sum as the density does (`draw_sum`
of `predictive.rs`): each part, summed, or the matched negative binomial for
`"moments"`.

**Rule:** a term of several parts reads the density of their sum, and its
draws, scores and information read that same density; a model with no closed
form for the sum refuses parts.

| Alternative | Why not |
|---|---|
| Moments only | They drift by up to 56 % of an interval with few effective parts, and bias `φ`. |
| Exact only | It refuses a total of 10⁶ or more, and a tail or an information that does not end within the cap, as at `φ = 10⁶`; `"moments"` reads them. |
| A default `sum`, or a switch by the size of the total | A judgment hidden in the library. |
| Approximate sums of the log-normal and the Student t | No caller ([L8](https://github.com/MUNQU/commol/issues/498#issuecomment-6983148460)). |
| The Poisson below a small `φ` | A jump in the likelihood at the threshold; the recursion, with `log1p(φμ)/(φμ)` and no division by `φ`, reaches the Poisson at `φ = 0` within 6e-14 ([L9](https://github.com/MUNQU/commol/issues/498#issuecomment-6983149187)). |
| Truncating the recursion where `q_max^j < 1e-17` | It saves 10–100× at small `φ` and costs up to 7.6e-3 in `log p` far in the upper tail; only with a declared tolerance and a test of that tail ([L10](https://github.com/MUNQU/commol/issues/498#issuecomment-6983149799)). |
| A cut of the jumps once their remaining sum is below `1e-17`, `O(y · J)` | `J ≈ 40 / (1 − q_max)` jumps against `K` states: 865 at `M = 10⁴, φ = 0.01` with a largest part of `0.2 M`, and no cut once `φ μ_max ≳ y / 40`, as from `φ = 0.1` at `M = 10³` (measured for [#513](https://github.com/MUNQU/commol/issues/513)). |
| The sum over every earlier count at each count, `O(K · y + y²)` | 4 s for a density and 9 s for the scores at a total of 10⁵, 100 s for the four readings of seven parts of mean 10⁵ (`test_parts.rs::an_exact_sum_of_a_large_total_is_cheap`), against 0.07 s by the states. Against 40 digits on a grid of seven parts, totals 10² to 10⁵ and `φ` 0 to 10⁶, both err at most 3e-12 in `log p`, 3.3e-11 in a tail and 2.7e-9 of the largest score, where `log p(0)` is large (measured for [#513](https://github.com/MUNQU/commol/issues/513)). |
| A convolution of the parts by FFT, `O(K · y · log y)` | Its error is about `1e-16` of the largest probability, not of each: relative 1e-11 at `p = 5·10⁻⁸`, and a negative `p` at `3·10⁻⁶¹`, so neither the tails nor the log density of a far count hold; it also needs the probabilities of each part, and their derivatives for the scores (measured for [#513](https://github.com/MUNQU/commol/issues/513)). |

**Guards:** `crates/commol-calibration/tests/test_parts.rs::the_exact_sum_is_the_convolution_of_its_parts`,
`crates/commol-calibration/tests/test_parts.rs::an_exact_sum_of_a_large_total_is_cheap`,
`crates/commol-calibration/tests/test_parts.rs::the_moments_of_a_sum`,
`crates/commol-inference/tests/test_predictive.rs::draws_of_a_sum_have_the_variance_of_their_parts`,
`tests/uncertainty/test_observation_sums.py::TestSums`,
`tests/uncertainty/test_part_sums_coverage.py::TestPartSums`.

## The tails of an exact sum split its mass

`exact_tails` (`likelihood/sums.rs`) reads `P(Y < y)`, `p(y)` and `P(Y > y)` from
the recursion tilted to `y`. While `P(Y > y)` is at least
`COMPLEMENT_TAIL = 0.01` it is the complement of the counts up to `y`. Below, it
is the sum of the counts above `y`, and the larger of the other two is the
complement: the three sum to 1 and none leaves `[0, 1]`. Far above the mean the
terms near `M` underflow, so the counts below `y` are not summed once the
Chernoff bound at the tilt (`Recursion::log_tail_bound`) shows
`P(Y ≥ y) < COMPLEMENT_TAIL`. Where it does not, those that hold the mass cannot
underflow:

- With `θ = ln ρ` and `K` the cumulant function of the sum, the tilted mass of a
  count `n` is `p(n) e^(θn − K(θ))`, and no term is below it.
- Up to `M`, `θ ≤ 0` and `θn − K(θ) ≥ θy − K(θ) ≥ 0` for `n ≤ y`.
- Above `M`, `K′` is convex and rises from `M` to `y`, so `K(θ) − θM` is at most
  `θy − K(θ)`, at most `−ln COMPLEMENT_TAIL ≈ 4.605`, and `K″ ≥ σ²` gives
  `θσ ≤ 3.1`. From `8.6σ` below `M`, under which `1e-16` of the mass lies, the
  terms are at least `p(n) e^−32`.

Errors measured for [#524](https://github.com/MUNQU/commol/issues/524), not
bounds. Against 45 digits (3 940 cases: `M` from 0.01 to 10⁴, `φ` from 0 to 10³,
totals from 0 to 100 sd above `M`, equal, unequal and zero parts) the tails err
by 1e-14 in the median. The worst are the complements near `COMPLEMENT_TAIL`,
which lose digits with `|log p(0)|`. Against 25 digits or more (2 000
cases: `M` from 10² to 3·10⁵, `φ` from 0 to 0.05, or 0.01 at 3·10⁵, 2 or 7 equal
parts, totals from 1.9 to 2.4 sd above `M`) they err by up to about 5e-10 for `M`
up to 10⁴, 1e-8 at 10⁵ and 1e-7 at 3·10⁵.

**Rule:** the tails of a total, however far from the mean, are a partition of the mass, each on its side.

| Alternative | Why not |
|---|---|
| The sum of the terms below `y` at every `y` | Their underflow gives an extreme upper outlier `P(Y < y) ≈ 0`, as at `M = 100, φ = 0, y = 1500`, and a part of the mass in between, as at `M = 1000, φ = 0, y = 2790` in two parts, where they sum to 0.459. |
| A bound of `p(y)` below which the counts below `y` are not summed | A constant with no meaning of its own; the Chernoff bound asks the question of `COMPLEMENT_TAIL`, gives the same tails in 23 940 cases, and spares the sum below sooner. |
| The sum above `y` at every `y > M` | With a heavy tail it reaches far beyond `y`, or past the cap, for totals near the mean, where the complement is as accurate. |
| A recursion with no tilt for the tails | Past the mean its terms fall into the subnormal numbers, which lose the digits of a tail below about `1e-290`, and stay there: 0.4 s instead of 0.02 s at `M = 70, φ = 0.5, y = 999 999`. |
| A second recursion, with no tilt, for the counts below `y` | Twice the cost for a mass that is the complement. |

**Guards:** `crates/commol-calibration/tests/test_parts.rs::an_exact_sum_far_above_its_mean_is_in_its_upper_tail`,
`crates/commol-calibration/tests/test_parts.rs::the_tails_of_an_exact_sum_split_its_mass`,
`tests/uncertainty/test_observation_sums.py::TestSums::test_an_outlier_of_an_exact_sum_has_a_pit_near_one`.

## An exact sum ends at a fixed count

`"exact"` computes the counts below `MAX_COUNTS = 1 000 000` (`likelihood/sums.rs`).
`Recursion::extend_to` refuses a count at the cap before computing it. Before
the upper tail and the information start, `ends_within_counts` fails them at
once when `p(MAX_COUNTS − 1) ≥ maxₜ pₜ(n) Π_{s≠t} pₛ(0)` is above `NEGLIGIBLE`;
below `φ = 1e-8`, where `pₜ(n)` sums `n` terms, the sum is about `Poisson(M)`
and takes no bound. Between, as at `M = 10⁵, φ = 3`, they run to the cap in
0.2 s, then fail. `UnendedSum` becomes a message that names the observation,
its total, `M`, `φ` and the counts, and points to `sum="moments"`: a
`RuntimeError` in every method. Seven equal parts need 13 902 counts at
`M = 10⁴, φ = 0.01`, 81 971 at `φ = 1`, 507 453 at `φ = 10`, 54 257 at
`M = 10³, φ = 10`, 237 757 at `M = 10⁵, φ = 0.1` and 780 366 at `φ = 1`
(`Noise::sum_information`, release, 0.16 s); the tests, at most 237 757.

**Rule:** the exact reading ends within `MAX_COUNTS` or fails with the reason; it never switches to `"moments"` ([D2](https://github.com/MUNQU/commol/issues/498#issuecomment-6979416382)).

| Alternative | Why not |
|---|---|
| No cap | At `φ = 10⁶` the tail spans about `40 φμ` counts: a fit hangs with no message. |
| A cap relative to `M` or the sd | The counts needed grow from 10 to 45 sd as `φ` goes from 0.01 to 10: no multiple bounds the time. |
| Only the check at the cap | 10⁶ counts take 0.2 s for an information before it fails at `φ = 10⁶`, in every term of every evaluation. |
| A cap of 10⁵ counts | It refuses a weekly total of 10⁵, and the information of seven parts of mean 10⁴ at `φ = 3` (185 239 counts). |
| A cap of 10⁷ counts | 160 MB of terms, and 2 s for an information that runs to it. |

**Guards:** `crates/commol-calibration/tests/test_parts.rs::an_exact_sum_that_does_not_end_fails`, `tests/uncertainty/test_observation_sums.py::TestSums::test_an_exact_sum_that_does_not_end_says_why`.
