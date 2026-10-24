# Local approximations

[Back to Design decisions](../design-decisions.md)

How the curvature, the Laplace approximation, the sandwich and the Laplace
mixture read a fit, and where their scores, their Fisher information and PSIS
live.

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

## Scores and information live in commol-calibration

`Discrepancy::term_derivatives` (`crates/commol-calibration/src/discrepancy.rs`)
gives each observation's gradient and expected curvature over its arguments:
`−2 w n² r` and `2 w n²` for the sum of squares, the same times
`c = 1 / (2 √(W · S))` for the root mean squared error (its derivative per
unit of `S`), `−w` times `Noise::scores` and `w` times `Noise::information`
for an observation model. The mean absolute error is refused: its sandwich
needs a density of the residuals at 0 and a bandwidth. The sandwich,
information matching and later gradient methods read them.

| Alternative | Why not |
|---|---|
| Derivatives in each method | Every method would repeat the models and their noise arguments. |
| A kernel density for the absolute error | It hides a bandwidth choice; the bootstrap needs none. |

**Guards:** `crates/commol-calibration/tests/test_likelihood.rs::scores_equal_the_differences_of_the_terms`.

## The sandwich sums scores over blocks

`sandwich` (`local/sandwich.rs`) gives `H⁻¹ B H⁻¹` with
`B = G/(G − 1) · Σ_b s_b s_bᵀ` over the non-overlapping blocks of
`BlockScheme` (CR1), and its draws and intervals read a Student t with
`G − 1` degrees of freedom. `G ≤ p` is refused, since the block scores of a
fit sum to zero and `B` would have rank below `p`. `block_length` has no
default; `"auto"` is the length of non-overlapping blocks of the residuals.
Information matching (`smc/gibbs.rs`) reads the same `H` as `J` and the same
`B` as `I`, so it takes a likelihood and holds under dependence.

| Alternative | Why not |
|---|---|
| The independent sandwich (White) | Too narrow when the residuals are autocorrelated. |
| Newey–West | It needs lags across series at irregular positions, and does not reuse the blocks. |
| A warning below a fixed number of blocks | An arbitrary threshold; the t widens the intervals by the right amount. |
| CR2 with Bell–McCaffrey degrees of freedom | Linear theory and the leverage of each block; kept for when a coverage study needs it. |

**Guards:** `tests/uncertainty/test_local.py::TestSandwich`,
`tests/uncertainty/test_coverage.py::TestSandwich`,
`tests/uncertainty/test_local.py::TestInformationMatching`.

## The Laplace approximation needs observation models

`laplace_approximation` (`local/approximation.rs`) refuses a loss: a loss has
no noise scale, so its curvature is no inverse variance. The message names
`NormalObservations(sd="sd")` with a noise parameter, whose fit estimates
the scale by maximum likelihood (`SSE/n`, narrower than `SSE/(n − p)` by
`√(n/(n − p))`), and the sandwich, which needs no noise model. A result
holds only data: `Calibrator.delta_band(result, positions, confidence_level)` reads the
problem from the calibrator, so a result read back from JSON gives the same
band.

| Alternative | Why not |
|---|---|
| A variance scale `SSE / (n − p)` | It assumes normal, independent noise of one size, which a loss does not state. |
| `LaplaceResult.delta_band` with a private reference to the problem | A result read back from JSON has none. |

**Guards:** `tests/uncertainty/test_local.py::TestLaplace::test_laplace_refuses_a_loss`,
`tests/uncertainty/test_local.py::TestLaplace::test_a_result_read_from_json_gives_the_same_delta_band`.

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
