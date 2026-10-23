# Transitions and rates

[Back to Design decisions](../design-decisions.md)

What a rate means, which stratified rate applies, and how expressions are evaluated.

## Absolute and proportional rates

The flow planner (`plan_transition_flows` in
`crates/commol-model/src/flows.rs`) decides for each flow whether the rate
is absolute or proportional:

| Case | Rate is | Flow per step |
|---|---|---|
| A stratified rate sets `absolute` | As set | — |
| The rate references a compartment, a bin total, `total` or a `total_…` variable | Absolute | `rate` |
| The transition has no source | Absolute | `rate` |
| Otherwise | Proportional | `rate × source` |

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

## Formula parameters are evaluated every step, in order

Parameters with a formula are evaluated again at each step, in the order they
were declared (`prepare_context` in `crates/commol-model/src/system.rs`). A
formula can use the value of an earlier formula from the same step. Declare
parameters in dependency order.

**Guard:** `crates/commol-model/tests/rates.rs::a_formula_reads_an_earlier_formula_of_the_same_step`.

## Compiled expressions are cached globally

`JIT_CACHE` in `crates/commol-core/src/math_expression/mod.rs` maps each
preprocessed formula to one compiled function for the life of the process. The
same formula always runs the same machine code, so results are reproducible.

**Rule:** do not invalidate or bypass the cache.

**Guard:** `crates/commol-core/src/math_expression/mod.rs::one_formula_compiles_to_one_function`.
