# Identifiability

[Back to Design decisions](../design-decisions.md)

Where the derivatives of the identifiability methods come from, and which
verdicts a report gives each parameter.

## Sensitivities come from one port

Every reader of derivatives of the predicted observations asks
`SensitivitySource` (`crates/commol-inference/src/sensitivity/`); only a source
knows how its derivatives are made. `FiniteDifferences` is the first: central
differences, one-sided of second order at a bound, with the step
`h = η^{1/3} · max(|θ|, w)` (at most `w / 8`), where `η` is the output
accuracy of the engine (`SimulationEngine::output_accuracy`: machine epsilon,
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
`api/inference/identifiability.py` write the reasons from the causes and
decide nothing.

| Alternative | Why not |
|---|---|
| One verdict | A parameter structurally identifiable and poorly determined by the data, the usual case, would look like a conflict. |
| Defaults for the thresholds | A judgment hidden in the library. |

**Guards:** `tests/uncertainty/test_identifiability_report.py` (`TestReport`),
`crates/commol-inference/tests/test_identifiability.rs::collinearity_gives_the_verdict_of_each_parameter`,
`crates/commol-inference/tests/test_identifiability.rs::a_sample_gives_the_verdict_of_each_parameter`.
