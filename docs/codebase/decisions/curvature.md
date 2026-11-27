# Curvature

[Back to Design decisions](../design-decisions.md)

How the curvature of the objective at a fit is read, and inverted, for the
methods of [Local approximations](local-approximations.md) and for information
matching.

## The curvature lives on the unbounded scale

`Curvature` (`crates/commol-inference/src/local/curvature.rs`) describes
`H_u = D · H_θ · D`, with `D = dθ/du` at the fit on the scale of `Transform`
(`transform.rs`: a logit with two bounds, a log with one, the identity with
none), centred on the fit mapped to that scale, with no curvature of the log
Jacobian; information matching reads it on the [scale of the parameters](bounds.md#information-matching-reads-the-curvature-on-the-scale-of-the-parameters). It holds `H_u` in units of its parameters
([below](#the-curvature-is-kept-in-units-of-its-parameters)). Every draw maps back with `from_unbounded`,
so it lies inside the bounds; every density on that scale, as the importance
weights of the mixture, adds `log_jacobian`. A fit on a bound is refused with
a message that names the parameter and points to a profile likelihood: its
variance on that scale is infinite. The one rule of a value on a bound,
`on_bound` in `transform.rs`, which the profiles also read, is `1e-8` of the
width of two finite bounds, or of `max(1, |bound|)` with one. The tolerance
comes from the optimizer: Nelder-Mead searches on a sine (`SineTransform`) of the unbounded
coordinate, which approaches a bound quadratically in `u` and ends about
`1e-15` of the width from it; `1e-8` of the width is a step `δu ≈ 2e-4`, far
below any posterior spread. The mixture skips such an end
([Local approximations](local-approximations.md#an-end-that-gives-no-component-is-noted)), and a bound [near a fit](bounds.md) is a note.

| Alternative | Why not |
|---|---|
| The scale of the parameters, for a distribution that is drawn | A normal there puts mass outside the bounds near them; information matching draws nothing. |
| A truncated normal on the scale of the parameters | It biases the mean and breaks the delta method. |
| Centring on the mode of the density on the unbounded scale, Jacobian included | It moves the centre away from the fit for no gain: importance sampling corrects the difference. |
| A fit on a bound by exact equality | An optimizer that the bound stops ends a few roundings from it, where the variance on the unbounded scale is huge but finite. |

**Guards:** `crates/commol-inference/tests/test_local.rs::draws_mapped_back_stay_inside_the_bounds`,
`tests/uncertainty/test_local.py::TestCurvature::test_a_fit_on_its_bound_points_to_a_profile`.

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
bounds ([shorter](#the-steps-of-the-hessian-keep-the-objective-a-number) for
a large curvature). Each entry of either carries an estimated error; an entry
above 1 % of `√(H_jj H_kk)` is a note. The threshold measures the error on the
scale of a correlation, so an off-diagonal entry near 0 is judged against its
row and column; a note changes no result. The note gives the share only, read in
the [units of the parameters](#the-curvature-is-kept-in-units-of-its-parameters),
since an entry on the unbounded scale can be below or above any number, and
words a share above 100 % as that.

| Alternative | Why not |
|---|---|
| The Hessian always | `4p² + 1` runs, and indefinite away from the optimum. |
| Gauss–Newton always | Wrong when the residuals are large or the mean is wrong, where the observed and expected curvature differ. |
| The fixed step `√(2ε)` of `finitediff` | With an ODE solved to `rtol`, an error of about `accuracy / h²`. |
| Differences written in commol | The fork keeps the numerical methods; commol chooses the steps from its engine. |

**Guards:** `crates/commol-inference/tests/test_local.rs::curvatures_of_a_linear_model_equal_the_information`,
`crates/commol-inference/tests/test_local.rs::the_hessian_of_a_linear_model_equals_the_information`,
`crates/commol-inference/tests/test_local.rs::a_hessian_keeps_the_scores_and_the_blocks_of_gauss_newton`,
`crates/commol-calibration/tests/test_likelihood.rs::fisher_information_of_each_model_has_its_closed_form`,
`tests/uncertainty/test_local.py::TestCurvature::test_an_uncertain_curvature_is_noted`,
`tests/uncertainty/test_local.py::TestCurvature::test_an_error_above_the_scale_of_its_entry_is_noted_as_such`.

## The steps of the Hessian keep the objective a number

A step `δ^{1/4}` (unbounded scale) puts the objective at `θ ± 2h` above the
largest number from `H_u ≈ 6e315` (`δ = ε`); `1/H_u` is a number to `5e-324`.
From `H_u ≈ 4e314` for that step (`6e309` at `δ = 1e-6`; sooner for the step
`δ^{1/4}|θ|` of no bounds) the step is the power of two at or below the one
whose `H_jj h_j²`, by the Gauss–Newton curvature in its units, is `MAX / 32`
(`OBJECTIVE_ROOM`). By `|H_jk| ≤ √(H_jj H_kk)` each point of the steps `2h`
moves the objective by at most a quarter of the largest number if the observed
curvature is at most that one and the objective grows like its quadratic (not a
Student t of small `df`); a point still not finite keeps the infeasible-point
error. A `MethodError::Failed` that names the rounding refuses the step unless
`θ ± h`, `θ ± 2h` are exact (the differences divide by the step), and unless
the rounding `e` of the runs (`run_rounding`) is at most the error of the step
`δ^{1/4}`: `Σ |C| |A| e ≤ δ^{1/2} h Ĥ_jj` by Gauss–Newton (`rounding_of_runs`).

| Alternative | Why not |
|---|---|
| Steps that move the objective by `δ^{1/2}` by the Gauss–Newton curvature | They change every Hessian, and `δ^{1/4}/√H_u` is below a rounding of `θ` long before the objective overflows. |
| The steps of `δ^{1/4}`, refused with words of the curvature when a point is not finite | It refuses a covariance that is a number, which Gauss–Newton gives. |
| Halve the steps until every point is finite | Up to `4p² + 1` runs for each halving, when the bound is known before any run. |
| A floor of `δ^{1/2}` times the size of `θ` on the step | It reads the origin of `θ`: it refuses a parameter far from 0 whose runs are those of one near it, and accepts a large constant in the runs. |
| Exact points only | The rounding of the runs then gives wrong values with no note. |
| A note in place of the refusal | When the rounding makes the Hessian indefinite, the error still says that the data do not determine the parameters. |

**Guards:** `crates/commol-inference/tests/test_local.rs::a_hessian_steps_where_its_objective_is_a_number`, `crates/commol-inference/tests/test_local.rs::a_hessian_that_cannot_step_says_that_the_curvature_is_too_large`, `crates/commol-inference/tests/test_local.rs::a_hessian_reads_no_step_within_the_rounding_of_its_parameter_or_its_runs`, `tests/uncertainty/test_hessian_steps.py::test_a_hessian_reads_a_curvature_whose_differences_would_overflow`, `tests/uncertainty/test_local.py::TestCurvature::test_a_curvature_that_overflows_on_the_unbounded_scale_says_so`, `tests/uncertainty/test_hessian_steps.py::test_a_hessian_refuses_a_step_that_the_rounding_of_the_runs_hides`.

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
one. A curvature that is not finite in its units is refused after it is
summed: with finite derivatives only an infinite information gives one, a mean
on the edge of the means of its model.

| Alternative | Why not |
|---|---|
| Check in `FiniteDifferences` | Another source of `SensitivitySource`, a public trait, could bring them back to the curvature. |
| Check that the gradients are finite | They do not say which part is not finite, and a Laplace approximation reads none of them. With a weight of 0, the gradient of an observation model is 0 at any prediction, so it does not show one that is not finite. |
| Skip the terms of weight 0 | An observation would be refused or not by its weight, when a run that is not finite never is. |
| Skip what the method at hand does not read | The curvature is one object that each method reads in part, so each would need its own rule. |

**Guards:** `crates/commol-inference/tests/test_local.rs::a_jacobian_that_is_not_finite_is_refused`,
`crates/commol-inference/tests/test_local.rs::a_prediction_that_is_not_finite_is_refused`,
`crates/commol-inference/tests/test_local.rs::a_difference_of_finite_runs_that_overflows_is_refused`,
`crates/commol-inference/tests/test_local.rs::a_mean_on_the_edge_of_its_model_has_no_finite_curvature`,
`crates/commol-inference/tests/test_local.rs::a_loss_with_no_curvature_is_refused_before_a_derivative_that_is_not_finite`,
`crates/commol-inference/src/sensitivity/mod.rs::the_first_entry_that_is_not_finite_names_its_place`,
`tests/uncertainty/test_local.py::TestSandwich::test_sandwich_refuses_derivatives_that_are_not_finite`.

## The curvature is kept in units of its parameters

The units give `H_u = S Ĥ S`: `Ĥ`, the gradients of the terms and the
curvature of a block (the `scaled` accessors) are in a unit of each parameter,
and `S = diag(s)` (`Curvature::scales`) is `dθ/du` times a power of two, so a
change of unit rounds nothing. For Gauss–Newton the power is the one at the largest `√c |A|`
of the parameter over the arguments of the terms (`units`), kept as its
exponent, the sum of those of `√c` and `|A|`: the product of a finite root and
a finite derivative can be above the largest number. For the Hessian it
is `1/q_j`, with `q_j` the power of two of the step `h_j`: the differences are
those of `v ↦ f(v · q)` at `fit / q` with steps `h / q`, the same runs, and
`s_j` is the quotient `(dθ/du)_j / q_j`, since the product with `1/q_j`
overflows for a small step. The gradients of the terms of the Hessian are those
of Gauss–Newton times `s_GN / s_H`. No method forms `H_u`: the Laplace
approximation and the mixture read `S⁻¹ Ĥ⁻¹ S⁻¹`, the sandwich the influences
`S⁻¹ Ĥ⁻¹ ŝ_b`, and information matching only traces, which are the same in any
units. With bounds that move with a unit that is a power of two, the Laplace
approximation and the sandwich of either curvature, `η` and `ν` are the same,
bit for bit, at the units the guards read, from `2⁻¹⁰⁰⁰` to `2¹⁰⁰⁰`. A covariance that leaves the units can still
be too large or too small for a number
([below](#one-rule-says-what-the-data-do-not-determine)).

| Alternative | Why not |
|---|---|
| `H_u` as summed, scaled to a diagonal of 1 before its inverse | What the sum has lost does not come back: a curvature below the smallest normal number has fewer digits the smaller it is, one below any number is 0, and the squares of the scores that a sandwich sums leave the range first. |
| The unit `√H_jj` in place of a power of two | Known only after the sum, and a division by it rounds. |
| For the Hessian, the power of two at the root of its diagonal | Known only after the differences, whose points already read the step; the unit of the step is known before them. |
| The logarithm or the hypotenuse of each sum | A second arithmetic for a range that a change of units gives. |
| The unit as a number, the power of two of the product | The product overflows when the root of the information times a finite derivative is above the largest number, and the unit falls back to 1 with a curvature that is not finite. |

**Guards:** `crates/commol-inference/tests/test_smc.rs::information_matching_does_not_depend_on_the_unit_of_a_parameter`,
`crates/commol-inference/tests/test_local.rs::a_covariance_does_not_depend_on_the_unit_of_a_parameter`,
`crates/commol-inference/tests/test_local.rs::a_unit_holds_a_derivative_whose_product_with_the_root_of_its_information_overflows`,
`crates/commol-inference/tests/test_local.rs::a_scale_is_a_power_of_two_times_the_derivative_of_its_parameter`,
`crates/commol-inference/tests/test_local.rs::a_sandwich_reads_scores_whose_squares_underflow`,
`crates/commol-inference/tests/test_local.rs::a_term_of_no_weight_adds_nothing_whatever_its_derivative`,
`crates/commol-inference/src/transform.rs::the_derivative_does_not_depend_on_the_unit_of_the_bounds`,
`tests/uncertainty/test_information_matching.py::TestInformationMatching::test_information_matching_does_not_depend_on_the_scale_of_a_parameter`,
`tests/uncertainty/test_singular_curvature.py::test_a_sandwich_does_not_depend_on_the_scale_of_a_parameter`,
`tests/uncertainty/test_local.py::TestSandwich::test_a_sandwich_reads_scores_whose_squares_overflow`.

## One rule says what the data do not determine

`Curvature::inverse` (`InverseCurvature`, `local/inverse.rs`) inverts the
curvature for the Laplace approximation, the sandwich, the mixture and
information matching, on `symmetric_inverse` of `commol_statistics`: the matrix
is scaled to a diagonal of 1, and it is singular when its smallest eigenvalue is
at most `p·ε` times the largest (`singular`, which `collinearity_index` also
reads), the size of the rounding of an eigenvalue of 0. A singular curvature is
a `MethodError::Invalid` with one message, and the mixture words it in the note
of the end that gives no component. `BlockShares::new` inverts the Gauss–Newton
curvature of the blocks by it, with its own message. The rule reads neither the
scale of a parameter nor the estimated error of the derivatives.

Two parameters that the model reads only through their product have an
eigenvalue of 0 but for the rounding of the sums, which grows with the number
of observations. With exact derivatives of `a·b·(1 + t)`, a Poisson model and
`a`, `b` uniform in `(0.1, 2)`, a plain sum leaves it above `p·ε` for 5 of 1000
pairs at 200 observations and for 38 at 5000, and the compensated sums of
`add_term` (`CompensatedSum`, `commol_statistics::numeric`) for none. A
diagonal entry that is at most `ε` times the sum of the sizes of its terms
(`CompensatedSum::magnitude`) is the rounding of terms that cancel, and
`symmetric` sets it to 0: the parts of a window can read a parameter that the
total of the window does not, as in `l·(t + a(1 − cos 2πt)/2π)`, and the
rounding of their sum would otherwise pass the rule. The error of finite
differences is not a rounding: with a long series it can hide a product, which
the rule does not read (the guide says so). The inverse is `L⁻ᵀ L⁻¹` of the
Cholesky factor of the scaled matrix, and none when an entry of it is not
finite. It keeps the digits of a coupling far below the others, which the
sandwich of a parameter that the data read weakly needs.

With a curvature that the data determine, a covariance that leaves the units
can still be too large or too small for a number. An entry above the largest
number is refused by `Gaussian::new` (`MethodError::Invalid`). A variance of 0
is one below the smallest number, since `Ĥ⁻¹_jj` is above 0 once the rule
passes: `InverseCurvature::covariance`, which also takes the curvature scale
of the target of a mixture, refuses it, and `InverseCurvature::sandwich`
refuses it when a scaled influence on the parameter is not 0. The error is a
`MethodError::Failed` that names the parameter and says that its curvature is
too large for the covariance to be a number. A sandwich whose block scores are
all 0 gives a covariance of 0, which `Gaussian::new` refuses. A subnormal variance is given, and information matching reads no
covariance. The check is in `InverseCurvature`, where every covariance leaves
the units.

| Alternative | Why not |
|---|---|
| The sign of a pivot (a Cholesky factor, an LU inverse that is finite) | The pivot of a direction that the data do not read is a rounding of either sign: the Laplace approximation gives a sd of `1e5`, and information matching blames the blocks. |
| A rule that reads the estimated error of the derivatives | It is an estimate, and a coarse solver would turn a curvature that the data determine into a singular one; the note says how uncertain an entry is. |
| A larger limit, or one that grows with the observations | It refuses a direction that the data determine, the more so the more data there are. |
| A diagonal entry kept as summed | The rounding of parts that cancel is of either sign and size: the Laplace approximation of a window in 3, 4 or 12 parts accepts 77 to 93 of 200 random fits, with sd of `a` on the unbounded scale from `9e7` to `8e9`. |
| The singular values of `√W A` | A root of the information of each term, and the Hessian has none. |
| The inverse from the eigenvectors, which the rule has at hand | Their entries carry an absolute rounding of `ε`, far above a coupling of `1e-30` (`the_inverse_keeps_the_digits_of_a_weak_coupling`). |
| A variance of 0 refused in each method, or any 0 on the diagonal of a sandwich | Three checks for one condition; a sandwich whose block scores are all 0 would be blamed on its curvature. |
| A message that the data do not determine the parameter | They do: the curvature is too large for the range of a number, as a unit too small for it. |

**Guards:** `crates/commol-statistics/src/linear_algebra.rs::a_matrix_singular_to_working_precision_has_no_inverse`,
`crates/commol-statistics/src/linear_algebra.rs::a_matrix_that_is_not_positive_definite_has_no_inverse`,
`crates/commol-statistics/src/linear_algebra.rs::the_inverse_of_equal_correlations_is_symmetric_and_has_its_closed_form`,
`crates/commol-statistics/src/linear_algebra.rs::the_inverse_does_not_depend_on_the_scales_of_the_rows`,
`crates/commol-statistics/src/linear_algebra.rs::the_inverse_keeps_the_digits_of_a_weak_coupling`,
`crates/commol-statistics/src/linear_algebra.rs::an_inverse_above_the_largest_number_is_none`,
`crates/commol-statistics/src/linear_algebra.rs::eigenvalues_that_are_not_numbers_are_singular`,
`crates/commol-statistics/tests/test_numeric.rs::a_compensated_sum_keeps_what_each_addition_rounds`,
`crates/commol-statistics/tests/test_numeric.rs::a_compensated_sum_holds_the_size_of_its_terms`,
`crates/commol-inference/tests/test_local.rs::parameters_read_through_their_product_are_refused_by_every_method`,
`crates/commol-inference/tests/test_local.rs::a_parameter_read_only_by_parts_that_cancel_is_refused_by_every_method`,
`crates/commol-inference/tests/test_local.rs::a_covariance_is_given_until_the_curvature_is_too_large_for_it`,
`crates/commol-inference/tests/test_local.rs::a_sandwich_whose_scores_are_all_0_is_refused_by_its_covariance`,
`crates/commol-inference/tests/test_local.rs::a_component_reads_the_curvature_of_the_target`,
`crates/commol-inference/src/local/gaussian.rs::a_covariance_that_is_not_finite_is_refused`,
`crates/commol-inference/tests/test_smc.rs::information_matching_refuses_a_parameter_the_data_do_not_read`,
`tests/uncertainty/test_singular_curvature.py::test_a_parameter_the_data_do_not_read_is_refused`,
`tests/uncertainty/test_singular_curvature.py::test_parameters_read_through_their_product_are_refused`,
`tests/uncertainty/test_singular_curvature.py::test_a_variance_above_the_largest_number_is_refused`.
