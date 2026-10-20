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
bound, after `max_steps` points, or after four misses of a held prediction.
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
profile: `identifiable` when both sides cross, `structurally_non_identifiable`
when both sides reach a bound (or an unreachable prediction) and every
statistic is within the tolerance, `practically_non_identifiable` otherwise.
The tolerance is the `sd_tolerance` of Nelder-Mead (at least `1e-9`), or
`1e-6` for a swarm, times `max(1, |objective at the fit|)`
(`ProfileTracer::tolerance`). A threshold below it gives no verdict, and the
result says why. `ProfileResult.report()` maps `identifiable` to the practical
`determined`, `practically_non_identifiable` to `poorly_determined` and
`structurally_non_identifiable` to the structural `non_identifiable`.

| Alternative | Why not |
|---|---|
| A verdict from a threshold of 0 | A rise of numerical error would read as information. |
| The structural verdict from every flat stretch | Flat is what the structural rank answers; the profile adds only what holds over the whole range. |

**Guards:** `verdict::tests`,
`tests/uncertainty/test_profiles.py` (`TestVerdicts`).
