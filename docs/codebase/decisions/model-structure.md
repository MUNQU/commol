# Model structure

[Back to Design decisions](../design-decisions.md)

How compartments are named, what they hold, and how outputs and variables find them.

## Compartments are labels with a name

A compartment is a bin plus one category for each stratification that applies
to it, in declaration order. Its name, `{bin}_{category}_{category}…`, is for
display only. Its identity is its label: `OutputLabels` in
`crates/commol-model/src/labels.rs` and `CompartmentLayout` in
`crates/commol-model/src/layout.rs`.

**Rule:** never parse a compartment name or match it by prefix. Bin `A` and bin
`A_x` both produce names that start with `A_`, and a conditional
stratification gives some compartments fewer suffixes than others. Use
`CompartmentLayout::find` or `of_bin` in Rust, and the labels that the engine
queries return in Python.

**Guard:** `crates/commol-model/tests/flows.rs::a_bin_whose_id_starts_with_another_finds_only_its_compartments`.

## State, amount, total and subgroup total

The library names what the compartments hold without assuming individuals:

| Term | Meaning | Where |
|---|---|---|
| **State** | The vector of the amount of every compartment | `DifferenceEquations::state` and the `state` of the other engines, `Engine.state` |
| **Amount** | The value of one compartment, in the unit of its bin | `InitialCompartment.amount`, `InitialConditions.bin_amounts` |
| **Total** | The sum of every compartment, accumulators excluded | the rate variable `total` (`SPECIAL_VAR_TOTAL`) |
| **Subgroup total** | The sum of the compartments with some categories | `subgroup_total`, the rate variables `total_<cat>…` (`SUBGROUP_TOTAL_PREFIX`) |

**Rule:** use these four terms in code, docs and tests. `AxisPattern` takes a
`value`, so "amount" means only the value of a compartment.

**Why, against the alternatives:**

| Alternative | Why not |
|---|---|
| "population", "head count", `N` | They assume individuals and an epidemiological convention (`N = S + I + R`). A reserved `N` takes a letter that other fields use for a bin, such as a nutrient. |
| "mass" | A unit, not a quantity of any unit. |
| "stock" | An economic term, new to the code. |
| "content" | Ambiguous in software. |
| "size" | Reads as a count. |
| A function `total(g1, t1)` | No reserved names, but the expression language has no text arguments (`jit/ast.rs`), so the parser, the JIT and the check of the language would change. |

**Guards:** `tests/building/test_reserved_names.py::test_a_bin_named_n_builds_and_runs`.

## The initial state is an amount per bin

An aggregate structure states the initial amount of each bin in the unit of
that bin (`bin_amounts`), the split of each bin among the categories
(`stratification_fractions`) and the amount of single compartments
(`compartment_amounts`). There is no declared total: a model of a count and of
a concentration has none, because a total over bins of different units has
no unit, and the [unit check](../../guide/units-and-equations.md) rejects a
rate that reads one. An amount is a real number, finite and not negative, so a
model of a count, a mass or a concentration can be stated, and a model whose
bins all start at `0` builds and runs (`initial_state` in
`crates/commol-model/src/builder.rs`, `InitialConditions.validate_bin_amounts`).
A structure that states `total`, `bin_fractions` or `compartment_fractions` is
rejected with the fields to use (`InitialConditions.reject_fractions_of_a_total`).

**Why, against the alternatives:**

| Alternative | Why not |
|---|---|
| Fractions of one total when every bin shares a unit, amounts otherwise | Two forms of one thing and two paths in the engine, the write-back and the report; a calibrated value would be a fraction in one model and an amount in another. |
| A total per unit and fractions within it | The unit of a bin is optional free text, and Rust has no units, so the engine could not group the bins. |
| Fractions of one total | It has no meaning over bins of different units, and a total of 0 makes every fraction 0. |

The cost: a model of one unit writes the amount of each bin, such as
`{"A": 990, "B": 10}`, not a total and fractions.

**Guards:** `tests/initial_conditions/test_bin_amounts.py`,
`tests/reference_models/test_quadratic_decay.py::test_zero_is_the_solution_from_zero`,
`builder.rs` tests `bins_that_all_start_at_zero_build` and
`a_bin_amount_that_is_not_finite_or_is_negative_is_a_build_error`.

## Stratification order is part of the model

`expand` in `crates/commol-model/src/builder.rs` applies the
stratifications in declaration order. A stratification with `conditions` only
expands the compartments whose earlier categories satisfy all of them, so the
result can be a non-uniform product. A condition may only reference a
stratification declared before it (`ModelBuilder.add_stratification` checks
this).

**Rule:** changing the order of `add_stratification` calls changes which
compartments exist and what they are named.

**Guard:** `tests/building/test_stratifications.py::test_the_order_of_the_stratifications_sets_the_compartments`.

## Outputs are resolved by label

`Aggregate.outputs_by_source(model)` maps each bin or accumulator id to its
output names in engine order. It uses the engine expansion (`expanded_outputs`), so
`Simulation.outputs_for`, `group_outputs` and `total_series` group outputs by
label, not by name. `total_series` and `windowed_totals` also take output
names: an id that is a source stands for all its outputs, and any other id
must be an output name, which stands for itself.

**Guards:** `tests/simulation/test_outputs.py::TestOutputsBySource::test_expansion_matches_the_engine_output_names`
(order), `TestSourceIdSharingAPrefix` (a bin and an accumulator that share
a prefix), `TestTotalSeries::test_an_output_name_stands_for_itself` and
`tests/simulation/test_windows.py::test_windowed_totals_take_output_names`
(output names).

## The engine owns the variable names

The engine sets one variable per expanded compartment and per subgroup total
(`total_<categories>`, bin totals and partial bin totals).
`check_model` in `crates/commol-model/src/builder.rs` rejects a parameter
id that is one of those names, a parameter, bin or accumulator id that is an
engine variable (`ENGINE_VARIABLES`: `total`, `step`, `t`, `pi`, `e`) or a
name of the neighbors (`neighbors`, `neighbors_…`), a bin
whose compartments or bin totals read as a `total_<categories>` name (two sums
would share one slot), and a `total_…` variable in a rate or a formula
parameter that names no subgroup total. `Aggregate.check_model` calls it at
`build`, and `CompiledModel::try_from_model` calls it again.
`subgroup_total_names` lists the names; Python reads them through the binding
`commol_rs.model.subgroup_total_names` (`engine_state.subgroup_total_names`)
to check rates and give units.

**Rule:** never list these names in Python. Ask the engine.

**Guards:** `builder.rs` tests
`parameter_id_cannot_be_a_compartment_or_a_subgroup_total`,
`subgroup_total_out_of_declaration_order_is_a_build_error` and
`subgroup_total_names_follow_the_declaration_order`,
`an_engine_variable_is_not_a_parameter_bin_or_accumulator_id` and
`a_bin_name_that_reads_as_a_subgroup_total_is_rejected`;
`tests/building/test_reserved_names.py`;
`tests/rates/test_expressions.py::TestSubgroupTotals::test_every_name_the_engine_provides_is_accepted_in_a_rate`.
