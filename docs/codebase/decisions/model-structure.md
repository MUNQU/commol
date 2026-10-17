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

## State, amount, total and subgroup total

The library names what the compartments hold without assuming individuals:

| Term | Meaning | Where |
|---|---|---|
| **State** | The vector of the amount of every compartment | `CompartmentalEngine::state`, `Engine.state` |
| **Amount** | The value of one compartment | `InitialCompartment.amount` |
| **Total** | The sum of every compartment, accumulators excluded; the initial fractions are taken of it | `InitialConditions.total`, the rate variable `total` (`SPECIAL_VAR_TOTAL`) |
| **Subgroup total** | The sum of the compartments with some categories | `subgroup_total`, the rate variables `total_<cat>…` (`SUBGROUP_TOTAL_PREFIX`) |

The total is a real number, finite and greater than 0, so a model of a count, a
mass or a concentration can be stated. `0` is rejected: every fraction of it is
zero, and the report divides by it (`initial_state` in
`crates/commol-model/src/builder.rs`, `InitialConditions.validate_total`).

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
| A function `total(g1, t1)` | No reserved names, but the expression language has no text arguments (`jit/ast.rs`), so the parser, the JIT and the security check would change. |

**Guards:** `builder.rs` tests
`a_real_total_starts_with_amounts_that_sum_to_it` and
`a_total_that_is_not_finite_and_positive_is_a_build_error`;
`tests/building/test_reserved_names.py::test_a_bin_named_n_builds_and_runs`;
`tests/initial_conditions/test_bin_fractions.py::TestTotal`.

## Stratification order is part of the model

`expand` in `crates/commol-model/src/builder.rs` applies the
stratifications in declaration order. A stratification with `conditions` only
expands the compartments whose earlier categories satisfy all of them, so the
result can be a non-uniform product. A condition may only reference a
stratification declared before it (`ModelBuilder.add_stratification` checks
this).

**Rule:** changing the order of `add_stratification` calls changes which
compartments exist and what they are named.

## Outputs are resolved by label

`Aggregate.outputs_by_source(model)` maps each bin or accumulator id to its
output names in engine order. It uses the engine expansion (`expanded_outputs`), so
`Simulation.outputs_for`, `group_outputs` and `total_series` group outputs by
label, not by name.

**Guards:** `tests/simulation/test_outputs.py::TestOutputsBySource::test_expansion_matches_the_engine_output_names`
(order) and `TestSourceIdSharingAPrefix` (a bin and an accumulator that share
a prefix).

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
