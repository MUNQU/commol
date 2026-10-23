# Sensitivity Analysis

[Back to Design decisions](../design-decisions.md)

How a sensitivity analysis runs a model with no data, what a quantity of
interest reads from a run, what a NaN does to a quantity and to a method, and
how the replicates of a ratio combine.

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
`tests/uncertainty/test_sensitivity.py`, `tests/uncertainty/test_propagation.py`,
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
with its quantity; `quantity_schedule` gives the positions to run at.

| Alternative | Why not |
|---|---|
| `Simulation.quantities(results, …)` | The producer with the data as an argument; the accessors of a result live on the result. |
| A rerun at the schedule | Numbers from a run the user does not plot: another run with a stochastic engine, and no run of the point estimate of a fit. |
| Exact matching of positions | A position computed as `end − window` misses the row of the run. |

**Guards:** `crates/commol-inference/tests/test_quantities.rs`
(`a_done_run_is_read_at_the_schedule_within_the_tolerance_of_the_axis`,
`a_done_run_lists_every_position_it_lacks_with_its_quantity`),
`tests/simulation/test_quantities.py`,
`tests/uncertainty/test_propagation.py::TestPropagation::test_propagation_values_have_one_row_per_replicate_of_each_draw`.

## The modeller truncates a polynomial chaos basis

`Truncation` (`global_sensitivity/chaos.rs`) keeps the terms of a basis of
`degree` whose degrees have `Σ α_i^q ≤ degree^q` and that hold at most
`max_interaction` inputs; `PolynomialChaos` passes `q_norm` and
`max_interaction`, whose defaults, 1 and no limit, keep the total-degree
basis. `ChaosExpansion::terms` counts the terms with no basis built, the sum
over the interaction orders `r` of `C(inputs, r)` times the ordered tuples
of `r` positive degrees the truncation keeps, so `ForwardAnalysis::chaos`
refuses too few draws before any run. The basis is built by the recursion of
the total degree, which stops a branch that cannot hold the degrees left, so
its terms keep the order of the total-degree basis. A `ChaosResult` holds
the truncation, and `predict` rebuilds the terms from it.

| Alternative | Why not |
|---|---|
| A fixed budget of terms | A hidden default that refuses valid analyses. |
| An automatic choice (least-angle regression, adaptive bases) | It needs a criterion, which is a modelling choice too; a later proposal can choose terms from the data. |
| One class per truncation, joined in a union | The two bounds are independent and combine. |
| Filter the total-degree basis | The basis of many inputs that the truncation exists to avoid is built anyway. |

**Guards:** `crates/commol-inference/src/global_sensitivity/chaos.rs`
(tests module),
`crates/commol-inference/tests/test_global_sensitivity.rs::a_truncated_chaos_basis_fits_many_inputs`,
`tests/uncertainty/test_sensitivity.py::TestPolynomialChaos`.

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

## A ratio combines its replicates as the user chooses

`Quantity::Ratio` divides two quantities of one value each
(`crates/commol-inference/src/quantity.rs`). `Quantities::terms` gives each
scalar of a run as a numerator and a denominator (1 for a scalar that is not
a ratio), and `Quantities::combine` reduces the terms of the replicates of a
point by the `Replicates` rule of the scalar: `Within` (E[N/D], the mean that
every other scalar takes), `WithinPositive` (E[N/D | D > 0]) or `Pooled`
(E[N]/E[D]). `BatchRunner::means` reads `combine`, so PRCC, Morris, Sobol and
the polynomial chaos see one value per point; `Quantities::evaluate` gives
`Term::value`, N/D, to a propagation and to the quantities of one run. A
denominator of 0 makes the ratio of a run NaN, whatever the numerator. A point
with no value is `Undefined::Replicate` (the first replicate that is NaN) or
`Undefined::Point` (no replicate with D > 0, or a pooled denominator of 0),
and `NanRun::replicate` is `None` for the second. The rule has no default:
the Python class requires `replicates`.

| Alternative | Why not |
|---|---|
| One estimand, or a default | E[N/D] and E[N]/E[D] differ when D is small or 0 in some replicates; the choice is the modeller's. |
| `"within"` alone, with no conditional rule | A design of 10^5 replicates almost surely holds a replicate with D = 0 when D can reach 0, and the only way out would be another estimand. |
| Skip a replicate with D = 0 under `"within"` | It conditions on D > 0 with no choice made; `"within_positive"` names that estimand. |
| 0/0 = 0, or N/0 = ±inf | 0 changes the estimand unseen; an infinity passes a check for NaN and breaks the indices. |
| A ratio of a ratio, a `PeakPosition` or a `Trajectory` | A ratio of positions has no use, and a trajectory gives several values; a third operation would go to an expression on the Rust expression engine, not to more classes. |

**Guards:** `crates/commol-inference/tests/test_quantities.rs`
(`a_ratio_combines_replicates_as_the_user_chooses`,
`a_zero_denominator_is_undefined_unless_the_user_conditions`,
`a_draw_with_no_positive_denominator_raises`,
`a_zero_denominator_follows_the_nan_rule_in_every_method`),
`tests/uncertainty/test_quantities.py` (`TestRatio`).
