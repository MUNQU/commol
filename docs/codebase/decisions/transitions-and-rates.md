# Transitions and rates

[Back to Design decisions](../design-decisions.md)

What a rate means, which stratified rate applies, and how expressions are evaluated.

## Absolute and proportional rates

The flow planner (`plan_transition_flows` in
`crates/commol-model/src/flows.rs`) decides for each flow whether the rate
is absolute or proportional; the cases are in the table of
[Absolute and proportional rates](../../guide/transitions.md#absolute-and-proportional-rates).
The closure `is_absolute` reads only the compartment names and the names of
`subgroup_total_names` (`builder.rs`) that the rate reads outside its matrix
sums (`variables_outside_matrix_sums` in `crates/commol-model/src/matrices.rs`):
`total` is not one of them, so a rate that reads `total` and no compartment
stays proportional, and so does `k1 * matrix_sum(M, B / total)`, whose `B` is
the amount of other categories.

A transition with several sources and a proportional rate is rejected, because
it has no single source to multiply. `Model.describe` shows the planned flows,
so a proportional rate prints as `k1 * A`.

The name says what the flow is: proportional to its source. The name
"per-capita" counts heads, which assumes the source holds individuals; a
source can hold a mass or a concentration. "Relative" would clash with the
relative tolerance `rtol` of the ODE solver, and "per-unit" with units of
measure.

**Guard:** `tests/transitions/test_flows.py`.

## The most specific stratified rate wins

For each compartment, the stratified rate whose conditions match it with the
largest number of conditions wins (`helpers.rs`). On a tie, the first one in
the list wins. If none matches, the transition's default `rate` applies.
Building a model fails if a stratified rate matches no compartment that the
transition starts from.

**Guard:** `crates/commol-model/tests/flows.rs::the_most_specific_stratified_rate_wins_and_the_first_on_a_tie`.

## Formula parameters are evaluated every step, in order

Parameters with a formula are evaluated again at each step, in the order they
were declared (`prepare_context` in `crates/commol-model/src/system.rs`). A
formula can use the value of an earlier formula from the same step. Declare
parameters in dependency order.

**Guard:** `crates/commol-model/tests/rates.rs::a_formula_reads_an_earlier_formula_of_the_same_step`.

## Compiled expressions are cached globally

`JIT_CACHE` in `crates/commol-core/src/math_expression/mod.rs` maps each
formula to one compiled function for the life of the process. The
same formula always runs the same machine code, so results are reproducible.

**Rule:** do not invalidate or bypass the cache.

**Guard:** `crates/commol-core/src/math_expression/mod.rs::one_formula_compiles_to_one_function`.

## A matrix sum is written out as the sum it is

A matrix between the categories of a stratification (`CategoryMatrix`,
`crates/commol-core/src/types/process.rs`) is a data object of the process,
not a parameter: its values are fixed, and a parameter, constant or
calibrated, may scale the row and the column of each category
(`row_scales`, `column_scales`), so the model reads
`diag(a) · values · diag(b)`. A rate reads it through `matrix_sum(M, e)`,
a node of the tree (`Expr::MatrixSum`) whose first argument names the matrix
and is not a variable. The JIT never compiles that node:
`MatrixSums::resolve` (`crates/commol-model/src/matrices.rs`) writes it, for a
compartment of category `i`, as `Σ_j <M>_<i>_<j> * (e_j)` while
`plan_transition_flows` plans the flows. In `e_j` a bin reads the subgroup
total of exactly its compartments in `j` (the explicit sum of them when no
subgroup total sums that set alone), `total` reads `total_<j>`, and every
other variable is unchanged. The terms of a zero entry are dropped; a row of
zeros is `if(<M>_<i>_<j> != 0, <M>_<i>_<j> * (e_j), 0)` of its first column,
which keeps the unit of the matrix for the unit check and is 0 even where
`e_j` is not a number (the JIT compiles `if` as a select).

Each entry is the variable `<M>_<i>_<j>`, a parameter of the engine
(`matrix_variables`), not a number written into the rate. So the planned
flow keeps the unit of the matrix for the unit check and `describe` names the
entry. The variable holds the scaled entry `a_i · values[i][j] · b_j`, and
`<M>_radius` the spectral radius of the scaled matrix (`spectral_radius`,
`crates/commol-statistics/src/linear_algebra.rs`, the real Schur
decomposition of `nalgebra`). The engine keeps each matrix with the values of
its scales (`ScaledMatrix`, `crates/commol-core/src/types/process.rs`) and
writes its variables at the build and again when a caller sets a scale
(`CompiledModel::set_parameter`, `variable_values`): once per parameter
vector, never during a run. A matrix with a scale to calibrate has no values
until the scale is set. So `matrix_sum`, `<M>_<i>_<j>`, `<M>_radius` and a
calibration constraint
([A constraint reads a matrix at its point](calibration.md#a-constraint-reads-a-matrix-at-its-point))
read one matrix. A scale is a constant or a parameter to calibrate
(`check_matrices`): a formula or a series parameter changes during a run. The
unit check needs it dimensionless, so the entries keep the unit of the
matrix (`UnitChecker._scale_unit_errors`).

Whether the rate is absolute is decided on the variables outside the matrix
sums (`variables_outside_matrix_sums`), as for the neighbors of a unit: what
`e` reads is a quantity of the other categories, not of the source.

| Alternative | Why not |
|---|---|
| A variable per category, `<M>_<bin>_<i>`, made by the builder | It fixes the expression to one form, a bin over a total. |
| A new kind of transition | The coupling is part of a rate, not a flow of its own. |
| The matrix sum compiled by the JIT | The JIT has no compartment, so no category `i`; each engine would need the layout at run time. |
| The entries written as numbers into the rate | The unit check and the description lose the matrix, and a scale by parameters needs a rewrite of every rate. |
| The scales written into the rate, `a_i * <M>_<i>_<j> * b_j`, with the entries unscaled | `<M>_<i>_<j>` would read another matrix than `matrix_sum` and `<M>_radius`. |
| Any expression of parameters per entry | `n²` expressions to check and evaluate at each point; the common case is a scale of the rows or the columns of a fixed matrix. |
| Each entry a parameter | `n²` parameters for a matrix of fixed values, the common case; one entry can be calibrated today as a parameter. |
| A function `radius(M)` | The value does not change during a run, so a variable computed once per parameter vector is enough. |
| Power iteration for the radius | It converges slowly when the two largest moduli are close, and fails when the largest is a complex pair; a matrix may hold any finite values. |

**Guard:** `crates/commol-model/tests/matrices.rs` (with
`a_scale_multiplies_the_entries_and_the_radius_follows_it`),
`crates/commol-model/tests/rates.rs::a_rate_reads_the_largest_eigenvalue_of_a_matrix`,
`py-commol/tests/rates/test_matrices.py`,
`py-commol/tests/units/test_matrices.py::test_a_scale_of_a_matrix_has_no_unit`,
`py-commol/tests/units/test_matrices.py::test_a_matrix_sum_has_the_unit_of_the_matrix_times_the_expression`.
