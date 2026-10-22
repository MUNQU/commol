# Sensitivity Analysis

[Back to Design decisions](../design-decisions.md)

How a sensitivity analysis runs a model with no data, what a quantity of
interest reads from a run, and what a NaN does to a quantity and to a method.

## An analysis needs no data

`ForwardAnalysis` runs every method on `ForwardModel::unobserved` (values
written as a calibration writes them). Quantities read the observation
operator. A design point is the mean of its replicates on common seeds; a
predictive or a propagation gives each replicate of each draw its own seed
(`run_sets`), and each purpose has a stream (`stream_seed`). Morris, Sobol
and chaos refuse a sample; PRCC reads its weights.

| Alternative | Why not |
|---|---|
| A problem with fake observations | Invented data in a type that means a fit. |
| Python callables as quantities | Every run would return to Python, one at a time. |
| Resampling a sample to equal weights | Repeated draws that the test of PRCC counts as independent. |
| Hermite polynomials for a normal input | Every distribution is truncated, so they are not orthogonal. |

**Guards:** `crates/commol-inference/tests/test_global_sensitivity.rs`,
`tests/uncertainty/test_sensitivity.py`.

## A quantity is a series and a reduction

A quantity reads a `Series` of a run, the `Level` of some outputs or their
`Increments` over windows of one width, and reduces it: the value at a
position, the peak or its first position, or each value
(`crates/commol-inference/src/quantity.rs`). Every value is an observation of
the one observation operator, so a level is an observation and an increment
an observation with a window. A `Gain` passes its start to
`ObservationOperator::with_starts`. A quantity crosses from Python as the JSON
of its pydantic class, read into a Rust enum tagged by `kind`
(`py-commol/src/quantity.rs`); pydantic checks it first, and Rust checks it
again.

| Alternative | Why not |
|---|---|
| One class for each series and reduction (`PeakGain`) | Each new question adds classes, and the choice between level and increments, a modelling choice, is hidden in a name. |
| Composable wrappers or operators | They allow meaningless nesting, and the shape of the result depends on the operands. |
| A tuple whose fields depend on a kind string | The meaning of a position depends on the kind, and a new kind needs a parser by hand. |
| One pyclass per variant | A quantity holds a series, and nested conversions need code by hand. |

**Guards:** `crates/commol-inference/tests/test_quantities.rs`,
`tests/uncertainty/test_quantities.py`.

## NaN propagates in a run and stops a reduction across runs

Inside a run, a NaN in a value a quantity reads makes the quantity NaN: a peak
and the position of a peak are NaN, not the largest of the other values.
`PropagationResult.values` keeps the NaN rows. Every reduction across runs
raises `PredictiveError::NotANumber`, a `ValueError` in Python, with the
quantity and the points where it is NaN: `BatchRunner::means`, through which
PRCC, Morris, Sobol and the polynomial chaos read their design, adds the
first run, its replicate, seed and inputs; `prcc()`, `morris()`, `sobol()`
and `ChaosExpansion::fit` check before they estimate; `statistics::quantiles`
returns an error, so `summary()`, `member_statistics` and every band raise.
`PropagationResult.where(mask)` keeps the runs a condition chooses and
renormalises their weights; the user writes the condition.

| Alternative | Why not |
|---|---|
| A warning and a value | A quantile sorts a NaN by its sign bit, part of the Morris effects are NaN, and a warning is easy to lose. |
| Skip the runs that give NaN | It conditions the result on the runs that give a value, a choice no one made, and Sobol and Morris cannot drop a point of their design. |
| Raise at the run | The rows, and the pattern of NaN against the inputs, are lost. |
| `dropna()` | Dropping NaN looks neutral; it conditions the result as any filter does. |

**Guards:** `crates/commol-inference/tests/test_quantities.rs`
(`a_nan_makes_the_quantity_nan`, `every_estimator_rejects_a_nan`,
`a_nan_run_is_named_by_its_point_and_inputs`),
`tests/uncertainty/test_quantities.py` (`TestRules`, `TestFilter`).
