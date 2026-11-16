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
`tests/rates/test_expressions.py::TestSubgroupTotals::test_every_subgroup_total_of_one_meaning_is_accepted_in_a_rate`.

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
initial condition and a result. One owner may build a name from one list of
ids in two stratifications, when they share a category id: `total_c1` is
then the total of `group:c1` and that of `type:c1`. Such a model builds, and
no formula reads that name
([A formula reads a name of one meaning](#a-formula-reads-a-name-of-one-meaning)).

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
| Refuse a category id that two stratifications share | KISS: it refuses a documented feature in every model, also in one that reads no name of two meanings and runs right; a qualified id names each category apart, and no formula reads a name of two meanings. |

**Guards:**
`crates/commol-model/tests/model_rules.rs::a_name_of_two_owners_is_refused`,
`crates/commol-model/tests/model_rules.rs::a_name_of_one_owner_from_two_lists_of_category_ids_is_refused`,
`crates/commol-model/tests/model_rules.rs::the_message_gives_only_the_root_of_each_conflict`,
`crates/commol-model/tests/model_rules.rs::a_category_id_that_two_stratifications_share_builds`,
`crates/commol-model/src/builder.rs::a_bin_name_that_reads_as_a_subgroup_total_is_rejected`,
`tests/building/test_reserved_names.py::test_a_name_of_two_owners_is_refused`,
`tests/building/test_reserved_names.py::test_a_category_id_that_two_stratifications_share_builds`,
`tests/calibration/test_initial_conditions.py::test_an_id_of_a_compartment_and_of_another_bin_is_refused`.

## A formula reads a name of one meaning

A category id that two stratifications share gives a name a meaning in
each. With `group = [a, b]` and `type = [a, c]`, `total_a` is the total of
`group:a` and the total of `type:a`, and `A_a` is the compartments of bin `A`
in `group:a` and those in `type:a`. A compartment can be such a name: when
`type` splits only `group:b`, `A_a` is the compartment of `group:a` and the
partial total of `type:a`. `Names` in `crates/commol-model/src/labels.rs`,
which `check_names` returns, holds the meanings of each name, and `shared`
finds these: one owner, one list of ids, two stratifications.

**Rule:** the build refuses a transition rate, a stratified rate or a
formula parameter that reads a name of two meanings (`Names::check_reads`,
from `check_model`, after the check of the parameter ids). The message names
each formula and what each name names, with the categories to rename, and
gives each meaning as a sum of names of one meaning when one is
(`Names::sum_of`: each compartment once, its own name when that has one
meaning). A meaning that no such sum reads, such as that compartment `A_a`,
is read only after a rename, and the message says so. A model whose formulas
read no such name builds and runs. The engine sums no such name either:
`build_subgroup_total_mappings` in `crates/commol-model/src/builder.rs` gives
it no slot (`Names::shared_names`, which `check_aggregate` hands to
`CompiledModel::try_from_model`), so a matrix sum writes the sum of the
compartments in its place (`subgroup_totals_by_compartments`). The name
stays reserved: `subgroup_total_names` lists it, and no parameter takes it.

A matrix sum still writes a compartment by its name, which the engine reads
first (`resolve_var_slot`), also when that name has two meanings: in the
case above, `Model.describe` can show `B_a / (A_a + B_a)` for `matrix_sum(M,
B / total)` in `a`, a rate whose names a user may not write.

The cost: some models that ran right are refused. A `total_x` of two
stratifications with disjoint conditions summed each compartment once, and a
rate that read that compartment `A_a` read the compartment. A sum of names
of one meaning (`total_a_x + total_b_x`) or a rename reads them.

An initial condition, an observation, a constraint and a result need no such
check. They read an output or a prefix of its leading categories
(`prefix_indices` in `crates/commol-model/src/initial_conditions.rs`,
`output_sums` in `labels.rs`), and such a name has one list of categories,
because the categories before a position decide which stratification is at
it (`expand` in `builder.rs`). There, `A_a` is `group:a`, as in
the results; a total of `type:a` is not one of their names. An initial
condition names a category by its qualified id (`resolve_category`).

| Alternative | Why not |
|---|---|
| Refuse the read when the flows are compiled (`resolve_var_slot` in `builder.rs`) | A model that cannot run as written is refused when it is built ([Invariants](../mental-model.md#invariants)), and the compilation comes later, at `Simulation`. DRY: `Names` holds the meanings. |
| Refuse the read in Python | The golden rule: the engine holds every rule of a model, so a model in JSON meets it too. |
| A qualified form in the grammar, such as `total_group:a` | KISS, YAGNI: the parser, the JIT, the formula tree, the unit check and the equations of Python would change for a rare model. A sum of names of one meaning reads each meaning (`total_a_a + total_a_c` for `group:a`), and a rename gives it a name. |
| A function `total(group, a)` | KISS: the expression language has no text arguments, so its parser, its check and the JIT would change ([Model structure](model-structure.md#state-amount-total-and-subgroup-total)). |
| Read the first stratification, as a prefix does | One idea, one name: the rate would hide a choice that its reader cannot see, and an initial condition refuses a bare shared category. |
| Keep the slot of such a name, and refuse only the formulas that read it | DRY: the engine writes rates too. A matrix sum read the slot whose set of compartments it needed, and counted a compartment twice with no error (`subgroup_totals_by_compartments`). |
| Sum each compartment once | One idea, one name: the union is a third meaning that neither stratification has. A matrix sum that wrote the name would show a rate whose reader cannot tell it. |
| Refuse the name in an initial condition, an observation and a constraint too | KISS: there it has one meaning, the leading one, which the results also give it. |

**Guards:**
`crates/commol-model/tests/model_rules.rs::a_formula_that_reads_a_name_of_a_shared_category_is_refused`,
`crates/commol-model/tests/model_rules.rs::a_meaning_that_no_name_of_one_meaning_reads_needs_a_rename`,
`crates/commol-model/tests/model_rules.rs::stratifications_of_disjoint_conditions_that_share_an_id_read_each_meaning_by_a_total`,
`crates/commol-model/tests/model_rules.rs::a_shared_category_builds_when_no_formula_reads_a_name_it_gives_two_meanings`,
`crates/commol-model/tests/model_rules.rs::a_name_of_two_meanings_stays_reserved`,
`crates/commol-model/tests/model_rules.rs::the_valid_subgroup_totals_leave_out_the_names_of_two_meanings`,
`crates/commol-model/src/labels.rs::a_compartment_whose_own_name_has_two_meanings_reads_as_another_name_of_it`,
`crates/commol-model/src/labels.rs::a_name_that_no_sum_reads_in_any_meaning_is_read_only_after_a_rename`,
`crates/commol-model/tests/matrices.rs::a_matrix_sum_never_reads_a_name_that_a_shared_category_gives_two_meanings`,
`tests/building/test_stratifications.py::test_a_rate_that_reads_a_name_of_a_shared_category_is_refused`,
`tests/building/test_stratifications.py::test_a_rate_reads_a_shared_category_through_names_of_one_meaning`.
