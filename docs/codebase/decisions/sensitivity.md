# Sensitivity Analysis

[Back to Design decisions](../design-decisions.md)

How a sensitivity analysis runs a model with no data, what a quantity of
interest reads from a run, and what a NaN does to a quantity and to a method.

## An analysis needs no data

`ForwardAnalysis` runs every method on `ForwardModel::unobserved` (values
written as a calibration writes them). Quantities read the observation
operator. A design point is the mean of its replicates on common seeds; a
predictive or a propagation gives each draw its own seed and each of its
replicates a seed of that one (`run_sets`, `draw_seeds`; see
[Stochastic engines and seeds](engines.md#stochastic-engines-and-seeds)), and
each purpose has a stream (`stream_seed`). Morris, Sobol
and chaos refuse a sample; PRCC reads its weights.

| Alternative | Why not |
|---|---|
| A problem with fake observations | Invented data in a type that means a fit. |
| Python callables as quantities | Every run would return to Python, one at a time. |
| Resampling a sample to equal weights | Repeated draws that the test of PRCC counts as independent. |
| Hermite polynomials for a normal input | Every distribution is truncated, so they are not orthogonal. |
| Replicate `i` of draw `j` of a sample at the flat index `j · n + i` of one stream | The seed of a draw depends on `n`, and no public seed gives the runs of a draw. |
| The seed of the source for a sample of one draw, the tree for more | Two rules that depend on the size of the sample. |

**Guards:** `crates/commol-inference/tests/test_global_sensitivity.rs`,
`crates/commol-inference/tests/test_predictive.rs::a_draw_keeps_its_runs_when_the_replicates_change`,
`tests/uncertainty/test_sensitivity.py`,
`tests/simulation/test_quantities.py::test_the_replicates_of_a_draw_are_those_of_its_seed`.

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

## The quantities of one run take the path of a propagation

`SimulationResults.quantities` and `SimulationReplicates.quantities` reduce
the rows the run holds, with no rerun: `Quantities::evaluate_run` finds each
position of the schedule among the positions of the run with the tolerance of
the axis (`index_at` of `commol-calibration`), then calls the
`Quantities::evaluate` of `Propagation`. The binding (`quantities_of_runs` in
`py-commol/src/quantity.rs`) is a pure function of the quantities and the
rows. A position the run lacks is an error that lists every such position
with its quantity; `quantity_schedule` gives the positions to run at. A ratio
of two quantities is a division of the rows of `PropagationResult.values`,
before any reduction, not a quantity.

| Alternative | Why not |
|---|---|
| `Simulation.quantities(results, …)` | The producer with the data as an argument; the accessors of a result live on the result. |
| A rerun at the schedule | Numbers from a run the user does not plot: another run with a stochastic engine, and no run of the point estimate of a fit. |
| Exact matching of positions | A position computed as `end − window` misses the row of the run. |
| A `Ratio` quantity | NaN would become a normal value of a quantity, and the methods read a ratio row by row correctly already; a derived output cannot divide values at two positions. |

**Guards:** `crates/commol-inference/tests/test_quantities.rs`
(`a_done_run_is_read_at_the_schedule_within_the_tolerance_of_the_axis`,
`a_done_run_lists_every_position_it_lacks_with_its_quantity`),
`tests/simulation/test_quantities.py`,
`tests/uncertainty/test_sensitivity.py::TestPropagation::test_propagation_values_have_one_row_per_replicate_of_each_draw`.

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
