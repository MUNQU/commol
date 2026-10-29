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
refused: its runs change in jumps with the parameters.

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
parameter is an error.

| Alternative | Why not |
|---|---|
| Keep the noise parameters with a verdict that says the method does not apply | The zero column still lowers the rank and gives infinite pair indices, so every count and subset needs a special case. |
| Judge them by the sensitivities of the likelihood (the Fisher information) | A different method, with its own threshold. |
| `FiniteDifferences` leaves them out | The local approximations read the same source, and their curvature needs the noise parameters. |
| A verdict name such as `not_applicable` | A third way to say that a method gives no verdict, beside `None` here and in a profile. |

**Guards:** `crates/commol-inference/tests/test_identifiability.rs`
(`a_noise_parameter_is_left_out_of_the_structural_rank`,
`a_noise_parameter_is_left_out_of_the_collinearity`,
`a_rank_with_only_noise_parameters_is_an_error`),
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
parameters it is collinear with; `SampleIdentifiability::verdicts`: the
parameters it is correlated with, its number of modes;
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
