# Names

[Back to Design decisions](../design-decisions.md)

How the engine names the compartments, the totals and the variables of a
model, and why each name names one thing. What a compartment is, and how
outputs are found by label, is in [Model structure](model-structure.md).

## The engine owns the variable names

The engine sets one variable per expanded compartment and per subgroup total
(`total_<categories>`, bin totals and partial bin totals).
The variables of a matrix between categories (`<M>_<i>_<j>` and
`<M>_radius`) are engine variables too: `check_matrices`
(`crates/commol-model/src/matrices.rs`) rejects a parameter, bin,
accumulator, compartment, subgroup total or engine variable that has the
id or a variable name of a matrix, and two matrices that give one variable.
`check_model` in `crates/commol-model/src/builder.rs` rejects a parameter
id that is one of those names, a parameter, bin or accumulator id that is an
engine variable (`ENGINE_VARIABLES`: `total`, `step`, `t`, `pi`, `e`) or a
name of the neighbors (`neighbors`, `neighbors_…`), a name that two things
build ([A name names one thing](#a-name-names-one-thing)), and a `total_…`
variable in a rate or a formula parameter that names no subgroup total.
`Aggregate.check_model` calls it at `build`, and
`CompiledModel::try_from_model` calls it again.
`subgroup_total_names` lists the names; Python reads them through the binding
`commol_rs.model.subgroup_total_names` (`engine_state.subgroup_total_names`)
to check rates and give units.

**Rule:** never list these names in Python. Ask the engine.

**Guards:** `builder.rs` tests
`parameter_id_cannot_be_a_compartment_or_a_subgroup_total`,
`subgroup_total_out_of_declaration_order_is_a_build_error`,
`subgroup_total_names_follow_the_declaration_order` and
`an_engine_variable_is_not_a_parameter_bin_or_accumulator_id`;
`tests/building/test_reserved_names.py`;
`tests/rates/test_expressions.py::TestSubgroupTotals::test_every_name_the_engine_provides_is_accepted_in_a_rate`.

## A name names one thing

A name joins ids with `_`, and ids have no rule on their characters, so two
things can build one name: bin `A` in category `g1` and a bin `A_g1` both
build `A_g1`. `check_names` in `crates/commol-model/src/labels.rs`, from
`check_model`, maps each name to what it names: each bin and each
accumulator, its prefixes and its outputs (`OutputLabels::prefixes`), and
each bin total, partial bin total and `total_<categories>`
(`subgroup_total_entries` in `crates/commol-model/src/builder.rs`). The
owner of a name is a bin, an accumulator, or the totals over categories. A
category id is not one of these names; the plan of initial conditions reads
it next to them, and refuses an id that is both
([An initial-condition id names one target](calibrated-initial-conditions.md#an-initial-condition-id-names-one-target)).

**Rule:** the build refuses a name of two owners, such as two bins, a bin
and an accumulator, or a bin and a total over categories, and a name of one
owner from two lists of category ids, such as two compartments `A_a_b` when
`group = [a, a_b]` and `type = [b]` applies only in `a`. It refuses the model
even when nothing reads the name. So a name names one thing in a rate, an
initial condition and a result. The one exception is a category id that two
stratifications share: one list of ids in two stratifications is one owner's
name, which `total_c1` and a partial total `A_c1` read in both categories
([#475](https://github.com/MUNQU/commol/issues/475)); an initial condition
names each with its qualified id.

The message gives each conflict once, at its root: a conflict is left out
when, without the last category of each meaning, or without a category that
every meaning takes, its meanings still conflict at one shorter name
(`follows_from_a_shorter`). The ids it gives to rename are the owners, when
they differ, and every category that not every meaning takes.

| Alternative | Why not |
|---|---|
| Refuse an id when the plan of initial conditions resolves it (`resolve_target`, `with_remainder`) | DRY: a rate, a sum of outputs and a result read the same names, and would still take one thing for another with no error. No id names either thing apart, so a rename is needed anyway, and the build asks for it before any run. |
| Refuse across bins only | S, DRY: one rule in one check. Two lists of one bin give two compartments `A_a_b`, a clash of the same kind. |
| Refuse only a bin id that is a compartment of another bin | DRY: a partial total (`A_t1` next to a bin `A_t1`), an accumulator and a total over categories (a bin `total_y`) are the same defect. |
| Resolve the bin before the compartment | One idea, one name: an initial condition would disagree with a rate, which resolves the compartment first (`resolve_var_slot` in `crates/commol-model/src/builder.rs`), and the other thing would still be lost with no error. |
| A syntax that names a bin apart, such as `bin:A_g1` | One idea, one name: a second spelling of an id, for a model that a rename makes clear. |
| A character rule on ids, or a separator that ids cannot contain | KISS: it refuses ordinary ids, such as `k_1` or `c_0_4`, in every model, for a rare clash. |
| Refuse a category id that two stratifications share | Leave the code better (15): it is a documented feature, which a qualified id names apart; the double count of its total is a change of its own, [#475](https://github.com/MUNQU/commol/issues/475). |

**Guards:**
`crates/commol-model/tests/model_rules.rs::a_name_of_two_owners_is_refused`,
`crates/commol-model/tests/model_rules.rs::a_name_of_one_owner_from_two_lists_of_category_ids_is_refused`,
`crates/commol-model/tests/model_rules.rs::the_message_gives_only_the_root_of_each_conflict`,
`crates/commol-model/tests/model_rules.rs::a_category_id_that_two_stratifications_share_builds`,
`crates/commol-model/src/builder.rs::a_bin_name_that_reads_as_a_subgroup_total_is_rejected`,
`tests/building/test_reserved_names.py::test_a_name_of_two_owners_is_refused`,
`tests/building/test_reserved_names.py::test_a_category_id_that_two_stratifications_share_builds`,
`tests/calibration/test_initial_conditions.py::test_an_id_of_a_compartment_and_of_another_bin_is_refused`.
