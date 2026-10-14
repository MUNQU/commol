# Refactor: neutral names for what the compartments hold

Goal: the library names what the compartments hold without assuming
individuals. "Population" goes from the API, the engine, the bindings, the
docs and the tests: the vector of every compartment is the **state**, the
value of one compartment is its **amount**, the total that the initial
fractions are taken of is the **total**, and the total of the compartments
with some categories is a **subgroup total**. The total becomes a real
number, so a model of a mass or a concentration can be stated. The rate
variable `N` and its subgroup totals `N_<cat>…` become `total` and
`total_<cat>…` (D6).
Motivation: `odd/tasks/domain-agnostic-audit.md`, findings C1 and D1, and its
section "Decisions" (agreed 2026-10-14).
Branch: `refactor/neutral-state-terms` from `develop`.
Engram mirror: topic `odd/neutral-state-terms/tasks` (project `commol`).
Runners: the Rust tests, clippy and rustdoc (from the root), `maturin develop
--release --features test-kind`, then `pytest`, `pytest -m docs`, `ruff check
.`, `ruff format --check .`, `ty check commol` and `vulture` (from
`py-commol/`), `mkdocs build --strict` (from the root).

## Current state (verified 2026-10-14 on `develop` at `77aa9fd`)

Outside `odd/` and the NSGA-II optimizer, "popul…" is on about 900 lines in
109 files; `population_size` on 211 lines in 67 files; "subpopulation" and
`subgroup_population` on 125 lines in 24 files.

| Layer | Names today |
|---|---|
| Model JSON and Rust core | `InitialConditions.population_size: u64` (`crates/commol-core/src/types/compartmental.rs:84`) |
| Rust model compiler | `CompartmentalEngine::population`, `population_size` (`crates/commol-model/src/engine.rs:24, 27`); `InitialCompartment.population` (`builder.rs:494`); `SubpopulationMapping`, `SubpopulationLayout`, `subpopulation_mappings`, `subpopulation_layout`, `subpopulation_names`, `subpopulation_param_names`, `build_subpopulation_mappings` (`builder.rs`, `types.rs`, `layout.rs`); `covered_population`, `population_indices`, `POPULATION_TOLERANCE` (`initial_conditions.rs`, `builder.rs`); `total_population` (`builder.rs`, `system.rs`) |
| Rust engines and calibration | `DifferenceEquations::population`, `DifferentialEquations::population`; `calibrated_initial_population`, `CalibratedInitialPopulation` (`commol-calibration`) |
| Bindings | `Engine.population` (`py-commol/src/engine.rs:305`), `InitialConditions.population_size` (`py-commol/src/core.rs:267`), `calibrated_initial_population` (`py-commol/src/calibration.rs:1402`), and their `_commol_rs.pyi` entries |
| Python API | `ModelBuilder.set_initial_conditions(population_size, …)`, `InitialConditions.population_size: int`, `subgroup_population` (`context/compartmental.py:490`, `context/initial_conditions.py:349`), `InitialCompartment.population` (`context/_model/engine_state.py:38`), the report group `"population"` (`context/_model/report.py:14`, `WHOLE_POPULATION_GROUP`), `_subpopulation_n_vars`, `_bin_subpopulation_vars` |
| `AxisPattern` | The argument `amount` of `pulse`, `pulses`, `periodic` and `interval`, the fields of their classes and `_validate_amount_string` (`api/patterns.py`, 50 lines) |
| Rate variables | `N` and `N_<cat>…`, spelled in eight places: `crates/commol-core/src/math_expression/preprocessing.rs:9` (`SPECIAL_VAR_N`), `math_expression/mod.rs:276`, `jit/compiler.rs:91, 97`, `crates/commol-model/src/system.rs:13` (`PARAM_N`), `builder.rs:156, 168, 707, 810, 822`, `crates/commol-calibration/src/calibration_problem.rs:363`; in Python `constants.py:5` (`RESERVED_VARIABLE_NAMES`), `context/compartmental.py:594-603, 662` and `utils/equations.py:256-277`, which build the `N_` names again instead of asking the engine; the report row `"N"` (`context/_model/report.py:129`). `\bN\b` or `N_…` is on 354 lines in 71 files, plus 122 lines in `tests/reference_models/` |
| Tests | `assert_population_conserved`, `initial_population`, `population_size=` in `tests/support/` and most modules; `population_size` in the Rust test models |

Names that stay: the NSGA-II `population_size` and `initialize_population`
(`crates/commol-calibration/src/probabilistic/ensemble/nsga2.rs`), argmin's
`PopulationState` (`optimization.rs`), and "population" in the prose of
`tests/reference_models/`, which states published models in their own words.

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).
A name not in the tables below is renamed after what it holds, with the same
four terms.

### S1 — The engine says state, amount, total and subgroup total

| Today | New |
|---|---|
| `InitialConditions.population_size: u64` (JSON key `population_size`) | `total: f64`, finite and greater than 0 (D1, D2) |
| `CompartmentalEngine::population()` | `state()` |
| `CompartmentalEngine::population_size()` | `total()` |
| `DifferenceEquations::population`, `DifferentialEquations::population` | `state` |
| `InitialCompartment.population` | `amount` |
| `calibrated_initial_population`, `CalibratedInitialPopulation` | `calibrated_initial_state`, `CalibratedInitialState` |
| `SubpopulationMapping`, `SubpopulationLayout`, `subpopulation_*`, `build_subpopulation_mappings` | `SubgroupTotalMapping`, `SubgroupTotalLayout`, `subgroup_total_*`, `build_subgroup_total_mappings` |
| `POPULATION_TOLERANCE`, `total_population` | `TOTAL_TOLERANCE`, `total` |
| Error text "exceed population_size", "a subpopulation total" | "exceed the total", "a subgroup total" |

- Check: the Rust tests, renamed; a test that a model with a total of `2.5`
  builds and starts with amounts that sum to `2.5`; a test that `0`, a
  negative total and a non-finite total are rejected.

### S2 — The bindings and the stubs follow

- Fix: `Engine.state`, `InitialConditions.total`, `calibrated_initial_state`,
  and their entries in `_commol_rs.pyi`.
- Check: `ty check commol`; the Python tests that call them.

### S3 — The Python API says state, amount, total and subgroup total

| Today | New |
|---|---|
| `ModelBuilder.set_initial_conditions(population_size: int, …)` | `set_initial_conditions(total: float, …)` |
| `InitialConditions.population_size: int` | `total: float`, finite and greater than 0 |
| `subgroup_population(categories)` | `subgroup_total(categories)` |
| `InitialCompartment.population` | `amount` |
| Report group `"population"` (`WHOLE_POPULATION_GROUP`) | `"total"` (`WHOLE_GROUP`) (D3) |
| `_subpopulation_n_vars`, `_bin_subpopulation_vars` | `_subgroup_total_vars`, `_partial_bin_total_vars` |

- A saved model with `population_size` does not load; the error names the
  unknown field (D4).
- Check: the tests of `building/`, `initial_conditions/`, `calibration/`
  (write-back of a calibrated total) and `reporting/`, renamed; a model saved
  with `total` reloads with the same initial state.

### S4 — `AxisPattern` takes a `value`

- Fix: `amount` becomes `value` in `pulse`, `pulses`, `periodic` and
  `interval`, in their classes and in `_validate_value_string`, so that
  "amount" means only the value of a compartment.
- Check: the tests of `axis_patterns/`, renamed; `docs/guide/axis-patterns.md`.

### S5 — The docs use the four terms

- Fix: every guide, API page, codebase page, `README.md`,
  `py-commol/README.md` and `docs/index.md`. "Population" goes, and so does
  "head count" (`context/initial_conditions.py:163`, `docs/guide/reports.md:123`).
  The changelog `[Unreleased]` gets a Changed entry with a
  mapping table (old name → new name). A design decision records the four
  terms and why "mass", "stock", "content" and "size" were not chosen.
- Check: `mkdocs build --strict`, `pytest -m docs`, and the search of T5.

### S6 — The tests use the four terms

- Fix: `tests/support/` first (`assert_population_conserved` →
  `assert_total_conserved`, `initial_population` → `initial_state`, the
  factories and their docstrings), then every module, the Rust test models and
  the calls of `tests/reference_models/`. Test names that say "population"
  are renamed after what they check.
- Check: `pytest` and the Rust tests pass with the same counts as before,
  apart from the new tests of S1 and S3.

### S7 — The rate variable `N` becomes `total` (D6)

| Today | New |
|---|---|
| `N` | `total`: the sum of every compartment at the current position, accumulators excluded |
| `N_<cat>…` (`N_g1`, `N_g1_t1`) | `total_<cat>…` (`total_g1`, `total_g1_t1`), same suffix rule |
| `SPECIAL_VAR_N`, `PARAM_N` | One constant in `commol-core` for the name and one for the prefix, used by every Rust place of Current state |
| `RESERVED_VARIABLE_NAMES = ("N", …)` | `("total", "step", "t", "pi", "e")` |
| Report row `"N"` | `"total"` |

- An id that starts with `total_` stays valid when its suffix names no
  categories, as an id that starts with `N_` is today
  (`test_an_n_prefix_that_names_no_subpopulation_is_a_valid_id`).
- Python asks the engine for the subgroup total names (a binding of
  `subgroup_total_names`) instead of building them in
  `context/compartmental.py` and `utils/equations.py` (golden rule).
- `N` is then a free id, so a model can name a bin `N` (a nutrient in an NPZ
  model, nitrogen).
- Check: the rate tests read `total` and `total_<cat>`; a test that a bin
  named `N` builds and runs; a test that a parameter named `total` is rejected
  as reserved; a test that the Python names equal the engine's.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D0 | all | Which terms replace "population"? | state, amount, total, subgroup total; `AxisPattern` takes `value` | See `odd/tasks/domain-agnostic-audit.md`, "Decisions": "mass" is a unit, "stock" is economic and new to the code, "content" is ambiguous in software, "size" clashes with NSGA-II. | approved (L1) |
| D1 | S1, S3 | Name of the field and of the builder argument | `total`, in JSON, in `InitialConditions` and in `set_initial_conditions(total=…)` | `initial_total`: longer, and the field already lives in `initial_conditions`. `size`: reads as a count and clashes with NSGA-II. `n` or `N`: an epidemiological convention (N = S + I + R) and a reserved letter (D6). With D6 the field is the value of the rate variable `total` at the start; `initial_conditions` tells the two apart. (The first reason given against `N`, that the variable and the field differ, was weak for this reason; corrected in L3.) | approved (L2) |
| D2 | S1, S3 | Type of the total | `f64` / `float`, finite and greater than 0 | `u64` / `int` (today): a model of a mass or a concentration cannot be stated (audit D1). Allow `0`: every fraction of a zero total is zero, and the report divides by the total. | approved (audit "Decisions") |
| D3 | S3 | Label of the whole-model group in reports | `"total"` | With D6 the group names the total a fraction is taken of, as a subgroup's label names its categories (`total_<cat>`): the row of `total` is `1.0` of `total`. `"all"`: says which compartments, not what the value is. `"N"`: the name D6 removes. Keep `"population"`: the term this task removes. | approved (L2) |
| D4 | S3 | A saved model with `population_size` | Rejected with the validation error, which names the field; the changelog mapping says the new name | An alias that reads the old key: commol is alpha and recommends the best design without backward compatibility (`feedback_alpha_no_backward_compat`); an alias keeps the old word in the code. A migration function: no released user needs it. | approved (L2) |
| D5 | — | Place in `odd/execution-order.md` | Before `axis-scale-refactor`, so that it is written against the final names, and in the same 0.3.0 release | After `axis-scale-refactor`: that ODD adds `steps(amount, unit)` (its D2), a third meaning of "amount", and its tests migrate twice. After 0.3.0: one more release with a break, and a second changelog mapping. | approved (L2) |
| D6 | S7 | Keep the rate variable `N`? | Rename `N` to `total` and `N_<cat>…` to `total_<cat>…` | Keep `N`: an epidemiological convention, it reserves a letter that other fields use for a bin (nutrient, nitrogen), and after this task the docs would explain it as "the total", so the name no longer explains itself. A function `total()`, `total(g1, t1)`: no reserved names, no dependence on declaration order, and qualified ids (`sex:y`); but the expression language has no text arguments (`jit/ast.rs:9`: constants, variables, operators, functions), so the parser, the JIT and the security check change: a task of its own, not justified without a case that needs it. | approved (L3) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S1, S2, S7 | Rust core, model compiler, engines, calibration and bindings; the `.pyi`; the `total` variable and the binding of its names | direct | `bc606e9` |
| T2 | S3, S4, S7 | Python API, reports and `AxisPattern`; reserved names and the subgroup total names from the engine | direct | `2e610e8` |
| T3 | S6, S7 | `tests/support/`, test modules, Rust test models, `reference_models/` calls and rates (`N` → `total`) | direct | `5814677` |
| T4 | S5 | Guides, API pages, codebase pages, READMEs, changelog, design decision | direct | `2faf2eb` |
| T5 | all | Final audit: every runner; a search for `popul` and `subgroup_population` that finds only the names that stay (see Current state), and for `\bN\b` and `N_` in rates, code and docs that finds none | direct | `9f7f891` |

## Delivery

- Strategy: one PR to `develop` from `refactor/neutral-state-terms`,
  squash-merged (`refactor(api): …`).
- Order: T1 → T2 → T3 → T4 → T5. T1 and T2 leave the tests red until T3.
- Version impact: breaking (JSON key, builder argument, public methods, the
  rate variable `N` in every saved model and script). It
  ships in 0.3.0 with the other breaks.
- Out of scope: "per-capita" (audit C2) and the other findings of the audit.

## Evidence

Runners on `refactor/neutral-state-terms` at `9f7f891` (2026-10-14):

| Runner | Result |
|---|---|
| Rust tests (six crates) | 255 passed, 0 failed; `develop` had 252 (+3: `a_real_total_starts_with_amounts_that_sum_to_it`, `a_total_that_is_not_finite_and_positive_is_a_build_error`, `subgroup_total_names_follow_the_declaration_order`) |
| `cargo test -p commol --features test-kind` | 4 passed |
| `cargo fmt --check`, `clippy -D warnings`, `cargo doc` with `-D warnings` | clean |
| `pytest` | 986 passed, 3 xfailed (989 collected); `develop` had 982 (+7: `TestTotal` ×5 replacing 2, `test_a_bin_named_n_builds_and_runs`, `test_every_name_the_engine_provides_is_accepted_in_a_rate`, `test_a_file_with_population_size_names_it`, `test_a_real_total_reloads_with_the_same_initial_state`) |
| `pytest -m docs` | 62 passed |
| `ruff check`, `ruff format --check`, `ty check commol`, `vulture` | clean |
| `mkdocs build --strict` | built, exit 0 |

The search of T5 (`git grep`, outside `odd/`) finds "popul" only in the names
that stay (NSGA-II `population_size`, `PopulationState`), the prose and
references of `tests/reference_models/`, the released and the "Before" columns
of `docs/about/changelog.md`, the dated record
`docs/development/design-per-compartment-initial-conditions.md` (a header line
gives the new names), the design decision that says why the word goes, and the
old file key `population` of `test_a_file_with_population_and_dynamics_names_them`.
`\bN\b` and `N_` are left only in "O(log N)", "N calibration runs", the design
decision and the paper notation of `tests/reference_models/` (docstrings and the
constants `N` that the oracles use); every rate there reads `total`.

Found on the way:

- `test_quadratic_decay.py::test_every_initial_value_gives_one_over_t_plus_c[0.5]`
  was a strict xfail (F1, an integer `population_size`); a real total makes it
  pass, so its mark is gone. `y0 = 0` stays an xfail: D2 rejects a total of 0.
- A bin or an accumulator could take an engine variable as id (`total`, `t`,
  `step`; `N` before), and the variable hid it; and a bin such as `total_y`
  over `age = [y, o]` and `sex = [y, o]` expanded into `total_y_y`, which the
  subgroup totals also use, so two sums shared one slot. Fixed in L5.
- `axis-scale-refactor` D2 named its argument `steps(amount, unit)`, against D5
  of this ODD. Fixed in L5 (`axis-scale-refactor` L11).

## Log

- L1 (user, 2026-10-14): approved the four terms and `value` for
  `AxisPattern` (D0); asked for this task before any change, and to wait for
  confirmation before running it.
- L2 (user, 2026-10-14): approved D1 (`total`), D3 (`"total"`), D4 (no alias)
  and D5 (before `axis-scale-refactor`, in 0.3.0).
- L3 (user, 2026-10-14): approved D6 (`N` → `total`, `N_<cat>` →
  `total_<cat>`; option B of three). The agent corrected the reasoning of D1
  against `N` and updated D3's reason, which D6 changes.
- L4 (agent, 2026-10-14): ran T1 → T5 on `refactor/neutral-state-terms`. The
  subgroup total names come from the engine (`subgroup_total_names`, binding
  `commol_rs.model.subgroup_total_names`), which also gives the units of
  `get_predefined_variable_units`, so its argument is now the names instead of
  the stratifications. The paper notation `N` of `tests/reference_models/`
  stays, as its prose "population" does; their rates read `total`.
- L5 (user, 2026-10-14): asked to fix the two findings of Evidence.
  `check_names` (`crates/commol-model/src/builder.rs`) rejects a parameter, bin
  or accumulator id in `ENGINE_VARIABLES` (one constant in `commol-core`, also
  read by the calibration constraints), and a bin whose compartments, bin total
  or partial bin totals read as a `total_<categories>` name
  (`subgroup_total_entries` says where each name comes from). Tests:
  `an_engine_variable_is_not_a_parameter_bin_or_accumulator_id`,
  `a_bin_name_that_reads_as_a_subgroup_total_is_rejected`,
  `a_bin_that_starts_with_the_prefix_but_names_no_subgroup_builds`, and in
  `tests/building/test_reserved_names.py` a bin id per engine variable, an
  accumulator `total` and a bin `total_y`. `axis-scale-refactor` D2 takes
  `steps(value, unit)`. Runners after it: Rust 258 passed (+3), test-kind 4,
  `pytest` 993 passed and 3 xfailed (+7), `pytest -m docs` 62; clippy, fmt,
  rustdoc, ruff, ty, vulture and `mkdocs build --strict` clean.
