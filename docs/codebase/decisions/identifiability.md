# Identifiability

[Back to Design decisions](../design-decisions.md)

Where the derivatives of the identifiability methods come from, and which
verdicts a report gives each parameter.

## Sensitivities come from one port

Every reader of derivatives of the predicted observations asks
`SensitivitySource` (`crates/commol-inference/src/sensitivity/`); only a source
knows how its derivatives are made. `FiniteDifferences` is the first: central
differences, one-sided of second order at a bound, with the step
`h = η^{1/3} · max(|θ|, w)` (at most `w / 8`), where `η` is the relative
accuracy of the engine (`SimulationEngine::accuracy`: machine epsilon,
or the `rtol` of the ODE engine) and `w` the width of the bounds (1 when infinite). Each entry
carries an estimated error, from the steps `h` and `2h` and the accuracy of
the runs; the structural rank counts a singular value only above the norm of
that error (Weyl's inequality). The scaled sensitivities
`S_ij = J_ij · |θ_j| / s_i` do not depend on units. A stochastic engine is
refused: its runs change in jumps with the parameters. The derivatives of the
means of the parts of each observation
([parts of a window](observations.md#the-parts-of-a-window-are-observations-of-every-reader))
come from the same runs, in a block of their own (`Sensitivities.parts`):
the curvature and the noise floor of a profile read it, the structural rank
and the collinearity read the observations.

| Alternative | Why not |
|---|---|
| Each method differentiates on its own | The step, the error and the scaling would differ between methods. |
| Exact derivatives first | Symbolic differentiation of expressions and sensitivity equations in two engines; a later source behind the same port. |
| Machine epsilon as the accuracy of every engine | With `rtol = 1e-6` the solver's error over the step is a fifth of the derivative, and the rank reads noise. |
| A fixed rank threshold such as `1e-10` | Below the error of finite differences, so every model looks identifiable. |

**Guards:** `crates/commol-inference/tests/test_sensitivity.rs`,
`crates/commol-inference/tests/test_identifiability.rs`,
`tests/uncertainty/test_identifiability.py`.

## A rank lost to the accuracy gives no verdict

The error of the runs of an engine that integrates with a tolerance enters
the estimated error of the sensitivities, so it can hide a singular value
that a tighter tolerance resolves. `structural_rank` asks the source for the
accuracy of its runs (`SensitivitySource::accuracy`, an `Accuracy` of
`commol-core`), for the settings that set it
(`SensitivitySource::accuracy_settings`, from
`SimulationEngine::accuracy_settings`: `ODESolver.rtol` and `ODESolver.atol`)
and for the same source at a tighter accuracy
(`SensitivitySource::at_accuracy`, which `FiniteDifferences` builds from
`ForwardModel::at_accuracy` and `SimulationEngine::at_accuracy`). It tightens
only when both are `Some`, and `StructuralAnalysis::tighter` holds the
tighter accuracy with its settings (`Tightening`), so a verdict left out at
the tighter accuracy always has settings for its reason to name. A point of
rank below the number of parameters differentiated runs once more, at each
accuracy divided by `TIGHTENING` (100), no lower than `TIGHTEST` (1e-14); the
verdicts are computed again with the ranks and null spaces there. A parameter
whose verdict changes has no verdict (`None`), as a profile whose structure
is undetermined. When a point cannot run at the tighter accuracy, every
parameter that is not identifiable has no verdict, and `unchecked` says why.
The section keeps the generic rank, the projections and the combinations at
the accuracy of the engine, and the generic rank at the tighter one; a
combination that holds a parameter with no verdict goes.

| Alternative | Why not |
|---|---|
| Differentiate at one fixed tighter tolerance | `1e-9` still lost ranks in a model of 15 parameters where `1e-12` resolved them: no fixed value is enough, and the user would not see that the result depends on it. |
| A warning only | The user cannot tell which verdicts to trust. |
| Search until the rank is stable | Its cost has no bound, and a stiff model may not integrate at the tightest tolerances. |
| A factor of 1,000 or more | Each run costs more as the tolerance tightens; the reason already tells the user to go further. |
| A verdict name such as `unresolved` | A second way to say what a profile says with no verdict, and one more case in `merge` and the reports. |
| Rebuild the model in Python at the tighter tolerances | The sensitivities run in Rust, and Python would need to know which engines integrate with a tolerance. |
| The settings and the tighter source as two separate answers | A source could give one and not the other, and the reason of a verdict left out would have no settings to name. |

**Guards:** `crates/commol-inference/tests/test_identifiability.rs`
(`a_rank_lost_to_the_accuracy_gives_no_verdict`,
`a_point_that_fails_at_the_tighter_accuracy_leaves_the_rank_unchecked`,
`an_exact_engine_is_not_evaluated_again`),
`tests/uncertainty/test_identifiability_tolerance.py`.

## Noise parameters are left out of the sensitivities

A noise parameter of an observation model (`CalibrationParameterType::Noise`)
changes the likelihood and never a prediction, so its column of the
sensitivities is zero: it would lower the rank and make every subset that
holds it collinear. `structural_rank` and `collinearity` differentiate only
the other parameters (`SensitivitySource::scaled_jacobian` with their
columns; `FiniteDifferences::jacobian_of` runs no difference of the
others). `StructuralAnalysis::parameters` and `Collinearity::parameters` name
them by index; the projections, the verdicts, the sensitivity measures and
the pair indices follow that order, and the combinations, the subsets and
`collinear_with` name parameters by their index in the problem. A point is
evaluated again at a tighter accuracy when its rank is below the number of
parameters differentiated. The sections name the parameters left out in
`left_out`; a noise parameter has the structural verdict `None`, with the
reason, and no practical verdict from collinearity. A problem with no other
parameter is an error, and collinearity needs at least 2 of them.

| Alternative | Why not |
|---|---|
| Keep the noise parameters with a verdict that says the method does not apply | The zero column still lowers the rank and gives infinite pair indices, so every count and subset needs a special case. |
| Judge them by the sensitivities of the likelihood (the Fisher information) | A different method, with its own threshold. |
| `FiniteDifferences` leaves them out | The local approximations read the same source, and their curvature needs the noise parameters. |
| A verdict name such as `not_applicable` | A third way to say that a method gives no verdict, beside `None` here and in a profile. |

**Guards:** `crates/commol-inference/tests/test_identifiability.rs`
(`a_noise_parameter_is_left_out_of_the_structural_rank`,
`a_noise_parameter_is_left_out_of_the_collinearity`,
`a_combination_names_the_parameters_by_their_index_in_the_problem`,
`a_rank_with_only_noise_parameters_is_an_error`,
`a_subset_larger_than_the_differentiated_parameters_is_an_error`,
`collinearity_needs_two_parameters_that_are_not_noise_parameters`),
`tests/uncertainty/test_identifiability_noise.py`,
`tests/uncertainty/test_identifiability_report.py::TestReport::test_a_noise_parameter_leaves_the_verdicts_of_the_others_alone`.

## Two verdicts per parameter

An `IdentifiabilityReport` gives each parameter a structural verdict (what
any data of the design of the observations can determine) and a practical
verdict (what these data determine). Only `StructuralRank` on the design of
the observations gives the structural verdict; a dense design keeps its
verdicts in its section. Merging reports keeps a verdict the reports agree
on, and makes one they disagree on `conflicting`, with every reason.
Thresholds and tolerances of the methods are required arguments: they are
modelling judgments. Rust decides each verdict and gives its causes
(`StructuralAnalysis::verdicts`; `Collinearity::verdicts`: insensitive, the
parameters it is collinear with; `SampleIdentifiability::verdicts`: a contraction below the threshold,
the parameters it is correlated with, its number of modes, and the bounds
that set a side, which do not set the verdict;
`StartsIdentifiability::flagged`). The runners of
`identifiability/runners.py` write the reasons from the causes and
decide nothing.

| Alternative | Why not |
|---|---|
| One verdict | A parameter structurally identifiable and poorly determined by the data, the usual case, would look like a conflict. |
| Defaults for the thresholds | A judgment hidden in the library. |

**Guards:** `tests/uncertainty/test_identifiability_report.py` (`TestReport`),
`crates/commol-inference/tests/test_identifiability.rs::collinearity_gives_the_verdict_of_each_parameter`,
`crates/commol-inference/tests/test_identifiability.rs::a_sample_gives_the_verdict_of_each_parameter`.

## A sample is judged against its reference

`sample_identifiability` (`crates/commol-inference/src/identifiability/sample.rs`)
reads the distinct draws of a sample, each with the summed weight of its
copies (`ParameterSample::distinct`), and measures each parameter against its
reference: the uniform distribution on the bounds, or its prior
(`Distribution`). The contraction, `1 − Var / Var_ref`, reads the variance
of the weighted draws, `Σ wᵢ (xᵢ − x̄)²` with no correction for the size:
a sample and its distinct draws give the same, and no sample on the bounds
spreads more than the widest distribution on them. The contraction enters
the verdict. The shape measures
read `θ`, or `ln θ` for `log_parameters`; the masses at each side and the
overlap read the distribution function of the reference.

- **Modes.** A weighted Gaussian kernel density with Silverman's bandwidth
  `h`, reflected at each finite bound, on a grid over the draws ± 3`h` cut
  at the bounds, of step at most `h/4` (`sample/marginal.rs`); the grid
  keeps only the points a kernel reaches, and one point of density 0 in
  each gap, so the step holds on a range of any width. The standard error
  at `x` is `√(Σ wᵢ² (Σ K_h(x − cᵢ))²)`, the inner sum over the kernels of
  draw `i` and its reflections `cᵢ`: one draw of a large weight is one draw,
  not a mode. With equal weights it is `√(f R(K) / (n h))`, times
  `√(1 + e^{−d²/h²})` at a distance `d` from a reflected bound. The lower peak of the
  shallowest dip goes while `f(peak) − f(valley) − 3·hypot(se_peak,
  se_valley)` is not positive; the ends of the grid are candidates.
- **A bound that sets a side** (`at_bound`): the density at the bound is not
  more than three standard errors below its highest point up to the nearest
  mode, and exceeds the reference smoothed by the same reflected kernel
  (`Distribution::smoothed_at_bound`) by more than three standard errors.
  It is a cause apart: commol cannot tell a natural bound from a chosen one.
- **Directions.** The eigenvectors of `D⁻¹ Σ D⁻¹`, `D` the standard
  deviations of the reference on the scale of each parameter
  (`Distribution::log_variance` for `ln θ`), so they do not depend on units.
  The count of hidden ridges reads the determined parameters only, with the
  contraction of each direction corrected by a Bayesian bootstrap of the
  weights, `2ĉ − mean ĉ*`, from `Stream::SampleDirections` of the seed of the
  problem; it is a warning of the section, not a verdict.

The numbers of the estimators are constants of the code, not settings, as
the step of a finite difference: three standard errors
(`STANDARD_ERRORS`), Rice's `⌈2 n_eff^{1/3}⌉` bins of the overlap,
`BOOTSTRAP_REPLICATES` (100) replicates of the count, a kernel cut at eight
bandwidths (`KERNEL_REACH`), 64 Gauss–Legendre nodes for each quadrature
over the quantile (below `10⁻²⁸⁰`, where the quantile of a small shape
underflows, `ln Q` follows the power law of the distribution function near
0). Three standard errors hold per dip, not jointly: a flat marginal shows a
false second mode in about 0.5 % of samples of 2000 draws and 2 % of 20 000
(1000 seeds each), and in under 1 % of
the marginals of the guard over four references and three sizes.

| Alternative | Why not |
|---|---|
| A mode above a tenth of the highest density, or a wider bandwidth | The noise bumps of a flat marginal are as high as the highest, and more smoothing erases real modes. |
| Hartigan's dip test | No count nor place of modes, blind to a pile at a bound, no weighted version. |
| The density on the scale of the distribution function of the reference | It reads the ratio of the sample to a non-uniform prior: `N(1, 0.5²)` under `N(0, 1)` gets a mode at 1.33. |
| Merging equal values of one marginal | An atom of 30 % at a bound drops the size to 11 and hides a mode. |
| A bound flagged by the mass beyond the quantile `ε` of the reference | It depends on `ε` and on the width of the range, and flags narrow modes near a bound as `n` grows. |
| `at_bound` in the verdict | A value at a natural bound, such as a rate of 0, is determined. |
| One scale for the whole sample | A dispersion with a lower bound of 0 and a draw there would forbid the log scale for all. |
| Directions as a cause of each parameter | It needs a threshold of loading. |
| No correction of the eigenvalues | Ten determined parameters show a false direction below the threshold in 90 % of samples of 300 draws. |
| A threshold of Marchenko–Pastur | It hides real directions: a direction of 0.45 is found in 2 % of samples. |
| A direction as a `Combination` | A combination is a product of powers that the rank of the sensitivities determines exactly; a direction has a contraction and mixes natural and log scales. |
| The thresholds of the estimators as settings | No user can choose them better than the estimator. |

**Guards:** `crates/commol-inference/tests/test_identifiability.rs`
(`a_sample_of_its_reference_is_poorly_determined_by_its_contraction`,
`a_flat_or_smooth_marginal_has_one_mode`, `real_modes_are_kept`,
`a_bound_that_sets_a_side`, `an_infinite_side_has_mass_and_no_bound`,
`the_overlap_of_a_sample_with_its_reference`,
`log_parameters_set_the_scale_of_every_shape_measure`,
`directions_and_their_contraction`, `the_count_of_hidden_ridges`,
`the_directions_do_not_depend_on_units`),
`tests/uncertainty/test_identifiability_sample.py`.
