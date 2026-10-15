# Refactor: no application and no unit in the library and its docs

Goal: remove every place where the library, its documentation, its examples
or its tests assume an application (epidemiology, demography) or a unit
(time, persons), except the tests copied from the literature.

The audit below was verified 2026-10-14 on `refactor/model-kind-interface` at
7664ee5. Line numbers refer to that commit.

## What is done or owned elsewhere

| Findings | Owner |
|---|---|
| C1, D1 | Done: `odd/archive/tasks/neutral-state-terms.md` (PR #108). |
| A5, D3, D4, D5 | `odd/archive/tasks/axis-scale-refactor.md`: S3 (E2, "once per step"), S4 and D6 (time is a unit like any other, no time aliases), D4 (`TimeScale` goes). |
| Everything else | This ODD. |

## The rule

| Source | Rule |
|---|---|
| `docs/codebase/mental-model.md:28` | "It knows bins, categories and rates, not diseases or drugs. Domain vocabulary belongs in user scripts." |
| `docs/codebase/mental-model.md:29` and `docs/codebase/design-decisions.md:208` | The axis is the independent variable of the model, whatever quantity it is; time is one case. |
| `docs/codebase/conventions.md:102-104` | Examples use abstract names only (`A`, `group1`, `k1`). No names from epidemiology, chemistry or any other field. |

## Scope and method

- Searched: `py-commol/commol/` (Python, `.pyi`), `py-commol/src/` and
  `crates/*/src/` (Rust, with rustdoc and unit tests), `docs/`, `README.md`,
  `py-commol/README.md`.
- Also searched: `py-commol/tests/`, the 78 tracked modules outside
  `tests/reference_models/` (section F). `tests/reference_models/` (24
  modules) is domain vocabulary on purpose (`conventions.md:235-237`) and is
  only summarized.
- Not searched: `odd/`, and `py-commol/tests/brauer-2019-mathematical-models-epidemiology/`
  (a textbook, untracked: excluded in `.git/info/exclude:16`).
- Second pass (inventory): every tracked text file in scope, including
  `crates/*/tests/`, `.github/`, `Cargo.toml`, `pyproject.toml` and
  `mkdocs.yaml`, split into words. Identifiers are split at `_` and at
  case changes (`PSOTimeVaryingAcceleration` → `pso`, `time`, `varying`,
  `acceleration`). The 4,406 distinct words were read one by one. This
  covers names of classes, functions, variables and fields, not only prose.
  Limit: one-letter names (`S`, `I`, `R`) are not in the inventory; they are
  covered where they appear with `beta`, `gamma` or `N`.
- Search terms of the first pass: time words (`time`, `time-varying`, `time step`, `over time`,
  `day`, `week`, `month`, `year`, `daily`, `weekly`, `season`, `annual`,
  `365`) and domain words (`population`, `per-capita`, `person`, `people`,
  `infect*`, `death`, `vaccin*`, `hospital`, `age`, `young`, `SIR`, `beta`,
  `gamma`, `R0`, `transmission`, `recovery`, and more).

## Not violations

| Where | Why |
|---|---|
| `TimeScale`: `py-commol/commol/api/time_scale.py`, `docs/api/time-scale.md`, `docs/guide/core-concepts.md` ("Steps and physical time") | A helper for the case where the axis is time, applied at the edges. Documented in `design-decisions.md:208-219`. See D3 for the design question. |
| PSO "time-varying acceleration" (TVAC): `PSOTimeVaryingAcceleration`, `PSOAccelerationType.TIME_VARYING`, `crates/commol-calibration/src/optimization.rs:185`, `py-commol/src/calibration.rs:382` | Name of the algorithm in the literature (Ratnaweera et al. 2004). "Time" means optimizer iterations. Optional: `py-commol/commol/context/calibration.py:499` says "over time"; "over iterations" is clearer. |
| PSO `velocity`, `personal best`; NSGA-II `population_size`; Nelder-Mead `gamma` | Optimizer vocabulary, not model vocabulary. |
| `docs/guide/units-and-equations.md:70-72` | States that time is one case among others. |
| Greek letter list, `py-commol/commol/context/_model/equations.py:337` | LaTeX rendering of any identifier. |
| `docs/about/license.md`, `docs/about/changelog.md` | Legal text and history. |
| `docs/guide/calibration-optimizers.md:168` ("Symptom") | Troubleshooting column header. |

## A. Text that states the axis is time

These are statements, not examples. Each one should say "axis", "position",
"step" or "along the axis".

| # | Where | Text | Note |
|---|---|---|---|
| A1 | `docs/api/axis-patterns.md:5` | "Time-varying rates and schedules per subgroup." | API reference page summary. |
| A2 | `py-commol/commol/api/patterns.py:2` | "Time-varying rate patterns for use in model transitions." | Module docstring. |
| A3 | `py-commol/commol/api/patterns.py:4-5` | "time-dependent rate schedules … seasonal curves" | Module docstring. |
| A4 | `py-commol/commol/api/patterns.py:264` | "A time-varying expression usable as (or within) a transition rate." | `AxisPattern` class docstring; rendered in the API reference. |
| A5 | `py-commol/commol/api/model_builder.py:384` | "The rate is evaluated ONCE per time step" | `add_transition` docstring. |
| A6 | `py-commol/commol/api/model_builder.py:480` | "fully specifies the time-varying behaviour" | Code comment. |
| A7 | `py-commol/commol/context/model.py:347` | "Time step variables (t, step) are dimensionless" | Docstring. |
| A8 | `py-commol/commol/api/plotter.py:86` | "the evolution of one series over time" | `plot_series` docstring. |
| A9 | `py-commol/commol/api/plotter.py:180` | "the running sum of each series over time" | `plot_cumulative` docstring. |
| A10 | `py-commol/commol/api/calibrator.py:417` | "negative time steps" | `Raises` section. |
| A11 | `py-commol/commol/api/calibrator.py:440`, `py-commol/commol/api/probabilistic_calibrator.py:183` | "time-dependent constraints" | Docstrings. The public name is "constraints at positions" (`crates/commol-calibration/src/types.rs:291`). |
| A12 | `crates/commol-calibration/src/calibration_problem.rs:789` | Error: "Time-dependent constraint '{}' evaluation failed at step {}" | **User-facing error message.** |
| A13 | `crates/commol-calibration/src/calibration_problem.rs:796` | Error: "Time step {} not found in simulation results for constraint '{}'" | **User-facing error message.** |
| A14 | `crates/commol-calibration/src/calibration_problem.rs:57, 66, 107, 148, 249, 341, 381, 390, 393, 468, 716, 719, 724, 745, 755, 871, 874, 876` | Field `time_dependent_constraint_indices`, method `evaluate_time_dependent_constraints`, and their comments | Internal names. |
| A15 | `crates/commol-calibration/src/calibration_problem.rs:409, 417, 428-430, 757-760, 773, 778, 790, 797` | Variable `time_step`, "time step" in comments and `expect` messages | Internal names. |
| A16 | `crates/commol-calibration/src/types.rs:292` | "Evaluated at specified time steps during simulation" | Rustdoc of `CalibrationConstraint`. |
| A17 | `crates/commol-calibration/src/types.rs:460-461` | "Model predictions at each time step", `predictions[time_step][compartment_idx]` | Rustdoc. |
| A18 | `crates/commol-calibration/src/probabilistic/calibration.rs:280, 358` | `expect("metric point time step must be present")` | Panic messages. |
| A19 | `docs/guide/transitions.md:80` | Heading "Time-dependent rates" | Anchor may be linked from other pages. |
| A20 | `docs/guide/transitions.md:450`, `docs/guide/core-concepts.md:473` | "Rates that change over time" | "See also" tables. |
| A21 | `docs/guide/mathematical-expressions.md:157` | "For rates that change over time" | Prose. |
| A22 | `docs/guide/mathematical-expressions.md:209` | "Ready-made time-varying rates" | "See also" table. |
| A23 | `docs/guide/examples.md:634` | "Time-varying rate helpers" | "See also" list, not an example. |
| A24 | `docs/codebase/repository-map.md:23` | "Time stepping." | Repository map. |
| A25 | `README.md:19`, `py-commol/README.md:19`, `docs/index.md:80`, `docs/guide/axis-patterns.md:6`, `docs/guide/mathematical-expressions.md:158` | "seasons" in the list of patterns | Follows from the name `seasonal` (D2). |

## B. Text that states an application

| # | Where | Text | Note |
|---|---|---|---|
| B1 | `py-commol/commol/api/model_builder.py:394-395` | "ideal for processes like per-capita death rates, where the rate should be proportional to each compartment's population" | `add_transition` docstring. |
| B2 | `py-commol/commol/api/probabilistic/ensemble_selector.py:100` | "They contain weekly differences, aggregates, …" | Docstring. Also wrong: the library computes no weekly differences. |
| B3 | `py-commol/commol/context/initial_conditions.py:163` | "every fraction-to-head-count conversion" | Docstring. See D1. |
| B4 | `docs/guide/calibration-observations.md:158-159` | "such as new arrivals per week" | Prose illustration with a time unit. |
| B5 | `docs/development/design-per-compartment-initial-conditions.md:12, 32, 55, 78, 165` | "the hospital pathway only exists for the `oe60` age stratum", `IH_y60`, `S_oe60_v_c` | A design document in the published site. |
| B6 | `docs/codebase/conventions.md:237` | "A module holds one model family and is named after its epidemiology" | Conflicts with `conventions.md:102-104` and `mental-model.md:28`. Say "named after the published model" instead. |
| B7 | `docs/codebase/conventions.md:210, 212` | "A death rate `d`", "an infection grows at a reproduction number" | Describes test oracles with domain words. |
| B8 | `crates/commol-core/src/math_expression/mod.rs:445, 450` | "common in SIR models", "normalized infection rate" | Test comments. |
| B9 | `crates/commol-model/src/builder.rs:1731` | "Compartment 0 is A_g1_s1 with 250 people." | Test comment. |
| B10 | `crates/commol-difference/src/lib.rs:22` | "Access compartment names and final populations" | Crate rustdoc. See C1. |
| B11 | `docs/guide/transitions.md:110-111`, `docs/guide/core-concepts.md:361-362` | "moves **one** flow, like a reaction" | Chemistry analogy in the prose. The examples below them also use the id `reaction` (`transitions.md:114-131`, `core-concepts.md:365-371`). |

## C. Vocabulary in the API itself

These are not single lines. They are names of the public API and of the
concepts in the docs. A change needs a decision first.

| # | Term | Size | Where it is public | Neutral term |
|---|---|---|---|---|
| C1 | "population" (amount in the compartments) | about 500 lines in 72 files | `population_size` (`ModelBuilder.set_initial_conditions`, `InitialConditions`, `crates/commol-core/src/types/compartmental.rs:84`), `subgroup_population` (`py-commol/commol/context/compartmental.py:490`, `initial_conditions.py:349`), the trait methods `CompartmentalEngine::population` and `population_size` (`crates/commol-model/src/engine.rs:24, 27`) and their bindings (`py-commol/src/engine.rs:305`, `py-commol/src/core.rs:267`, `_commol_rs.pyi:97`), `calibrated_initial_population` (`py-commol/src/calibration.rs:1402`, `_commol_rs.pyi:429`), "subpopulation" (`N_…` totals; `subpopulation_names`, `SubpopulationMapping` in `crates/commol-model/src/`), `README.md:4`, `docs/index.md:3` | "total", "amount", "state" |
| C2 | "per-capita" (rate multiplied by its source) | 59 lines in 25 files | Concept name in `docs/guide/transitions.md:58` ("Absolute and per-capita rates"), `core-concepts.md:62`, `design-decisions.md:131`, the `absolute` field docs, error text in `crates/commol-model/src/flows.rs:216` | "proportional", "relative" |

## D. Design that assumes an application or a unit

| # | Where | Finding | Consequence |
|---|---|---|---|
| D1 | `py-commol/commol/context/initial_conditions.py:134` (`population_size: int`), `py-commol/commol/api/model_builder.py:751`, `crates/commol-core/src/types/compartmental.rs:84` (`population_size: u64`) | The total amount at the start must be a whole positive count. | A model of a mass, a concentration or a fraction (total `0.5`, `2.75`) cannot be stated. The type assumes discrete individuals. |
| D2 | `AxisPattern.seasonal` (`py-commol/commol/api/patterns.py:451`, class `_SeasonalAxisPattern` at `:656`; `docs/guide/axis-patterns.md:52, 58, 267`) | A sine pattern has a name from time (seasons). | The public name assumes a time axis. Neutral names: `sinusoid`, `sine`, `harmonic`. |
| D3 | `py-commol/commol/context/_model/units.py:14` (`TIME_DIMENSIONS = ("[time]", "[step]")`), `:169`; `design-decisions.md:224-234`; `docs/guide/units-and-equations.md:77-87` | The step unit is inferred only from units of time. Any other axis must declare `step_unit`. | Time is the default axis of the unit check. Documented and guarded on purpose (a `1/meter` parameter may be a density). Decide whether to keep it. |
| D4 | `TimeScale` (`py-commol/commol/api/time_scale.py`) | A public, time-only helper (`steps_per_week`, `steps_from_hours`). | Acceptable as an edge helper (see "Not violations"). An axis-neutral `AxisScale` would make the API symmetric. `odd/archive/tasks/axis-scale-refactor.md` already plans to replace `TimeScale` with axis-neutral pieces. |
| D5 | `py-commol/commol/context/_model/helpers.py:290-321` (`known_time_aliases`) | The unit registration knows aliases only for time (`fortnight`, `semester`, `yr`, `hrs`, …). Every other unknown unit gets a new dimension of its own. | A second place where time is special in the unit check (found by the tests search). Same decision as D3. |

## E. Examples with application or unit names

Out of scope by request: these are examples. Most of them break
`docs/codebase/conventions.md:102-104`, which forbids domain names in examples.

| Where | Names |
|---|---|
| `py-commol/commol/context/calibration.py:236-247` (doctest) | `beta`, `gamma`, "Basic reproduction number R0", "Transmission rate >= recovery rate" |
| `py-commol/commol/context/transition.py:107-110, 131-132` | `beta * S * I / N`, `sin(2 * pi * t / 365)` "(time-dependent formula)" |
| `py-commol/commol/api/model_builder.py:266, 274, 416, 771, 784-788` | `N_young`, `"1/day"`, `"person"`, `death__S`, `S`, `age_group`, `young`/`adult`/`elderly` |
| `py-commol/commol/context/parameter.py:73, 93`, `py-commol/commol/context/model.py:326` | `"1/day"`, `"person"`, `person/day` |
| `py-commol/commol/utils/equations.py:33-38, 50` | `beta * S * I / N`, `{"beta": "1/day", "S": "person"}`, `person/week`, `person/day` |
| `py-commol/commol/utils/security.py:863` | `beta * S * I / N` |
| `py-commol/commol/context/stratification_condition.py:19`, `py-commol/commol/context/model.py:346`, `py-commol/commol/utils/equations.py:234, 264` | `young`, `urban`, `N_young`, `N_young_urban` |
| `py-commol/src/engine.rs:309-310`, `py-commol/src/lib.rs:22` | `["S", "I_young", "R_old"]`, `sir_model.json` |
| `crates/commol-difference/src/lib.rs:14, 19`, `crates/commol-calibration/src/lib.rs:29, 41-42`, `crates/commol-calibration/src/calibration_problem.rs:35-36` | `sir_model.json`, `create_sir_model`, `beta`, `gamma`, "100 time steps" |
| `crates/commol-calibration/src/types.rs:8, 300-309, 321-323, 354-357` | `beta`, `gamma`, `R0`, "Infected compartment", `max_infected` |
| `crates/commol-core/src/simulation_engine.rs:202-226`, `crates/commol-core/src/math_expression/mod.rs:153-156`, `crates/commol-core/src/math_expression/jit/mod.rs:20` | `beta`, `gamma`, `S` |
| `crates/commol-model/src/types.rs:278`, `crates/commol-model/src/builder.rs:663-665` | `N_young`, `N_young_urban`, `S_young` |
| Rust unit tests: `crates/commol-differential/src/tests.rs:187-222, 520-533`; `crates/commol-model/src/flows.rs:412, 438-553`; `crates/commol-core/src/math_expression/{mod.rs,preprocessing.rs}`, `jit/{tests.rs,parser.rs,compiler.rs,ast.rs}` | SIR model, `infection`, `recovery`, `vaccination`, `birth`, `death`, `aging`, `young`/`old`, `beta`, `gamma`, "the epidemic is over" |
| `docs/guide/simulations.md:321` | `if(t >= t_on, beta, 0)` |
| `docs/guide/units-and-equations.md`, `docs/getting-started/quickstart.md:75-76` | `"1/day"`, `"1/week"`, `person/week` (illustrations of the unit check) |
| `docs/guide/examples.md:87, 104, 119, 551-578`; `docs/guide/transitions.md:85-94`; `docs/guide/mathematical-expressions.md:20-21, 167-179`; `docs/guide/calibration-observations.md:170-177` | `365`, "annual cycle", "monthly", "seasonal", `k1_seasonal`, `weekly` |
| `docs/development/workflow.md:87`, `docs/codebase/conventions.md:276` | Sample commit message "add time-varying rates" |
| `py-commol/commol/context/compartmental.py:469, 505`, `py-commol/commol/context/initial_conditions.py:9, 276, 329, 384, 453`, `py-commol/commol/context/_model/report.py:60`, `crates/commol-model/src/initial_conditions.rs:21` | `sex:female` as the example of a qualified category id |
| `crates/commol-model/src/builder.rs:372, 485`, `py-commol/commol/context/initial_conditions.py:89`, `py-commol/commol/context/_model/engine_state.py:45, 69`, `py-commol/commol/api/model_builder.py:800` | `IH_y60`, `S_oe60_home`, `S_oe60_v_c` as examples of expanded compartment names |
| `docs/guide/mathematical-expressions.md:164` | "mass action" |
| Rust unit tests: `crates/commol-model/src/flows.rs:510-556`, `crates/commol-model/src/initial_conditions.rs:469-526` | Stratifications `care` (`home`, `hosp`), `age`, `sex` (`m`, `f`) |
| Rust integration tests: `crates/commol-calibration/tests/test_constraints.rs:8-120` | `beta`, `gamma`, id `r0`, "Beta must be >= gamma" |

## F. Tests (`py-commol/tests/`)

52 of the 78 tracked modules outside `reference_models/` use application or
time words. The cause is in `tests/support/` and in the convention that
prescribes it: almost every test builds an epidemic model.

### F1. The shared models and oracles are epidemiological

| # | Where | Text |
|---|---|---|
| F1.1 | `docs/codebase/conventions.md:189, 197` | The convention prescribes `sir_builder`, `seir_builder` and the fixtures `sir_model`, `sir_simulation` as the default test models. |
| F1.2 | `tests/support/models.py:125-172` (`sir_builder`) | Bins "Susceptible", "Infected", "Recovered"; transitions `infection`, `recovery`; `beta`, `gamma`; "1000 people". |
| F1.3 | `tests/support/models.py:175-225` (`seir_builder`) | "Exposed", `exposure`, `progression`, `sigma`; "1000 people". |
| F1.4 | `tests/support/models.py:248` (`a_to_b_builder`) | "the 1000 people start in A". |
| F1.5 | `tests/support/models.py:291-319` (`shared_category_builder`) | Stratifications `age` and `sex`. |
| F1.6 | `tests/support/models.py:322-357` (`conditional_pathway_builder`) | Bin "Hospitalized", age strata `y60`/`oe60`, condition `care`. |
| F1.7 | `tests/support/reference.py:62-74` | `sir_flows(infection, recovery)`, `SIR_START`. |
| F1.8 | `tests/support/oracles.py:1-111` | "The susceptible fraction left by an epidemic", `final_size(reproduction_number, …)`, "initial pathogen", `growth_ratio(…, infected, …)`, `assert_threshold` ("the infection fell by"). These oracles are general (a final-size relation, a threshold of growth) but are written in epidemic words. |
| F1.9 | `tests/support/flows.py:32` | `"gamma * I"`. |
| F1.10 | `tests/conftest.py:11-19`, `tests/calibration/conftest.py:1-13` | Fixtures `sir_model`, `sir_simulation`; "SIR with `beta` and `gamma`, the values to recover". |

### F2. Test modules with application words

Counts of matched words, per module. Most come from F1 (`beta`, `gamma`,
`S`, `I`, `R`) and from stratifications named after age or place.

| Module | Main words |
|---|---|
| `units/test_print_equations_stratified.py` | `young` (155), `urban` (77), `aging` (77), `rural` (53), `age` (31), `old` (26), `vaccination` (3), SIR |
| `units/test_model_units.py` | `person` (18), `day` (42), `infection` (21), `vaccination` (5), `urban`/`rural`, `adult`, `seasonal`, `365` |
| `units/test_unit_checks.py` | `person` (69), `day` (30), `365` (5), `seasonal` |
| `units/test_print_equations.py`, `units/test_print_equations_units.py` | SIR, SEIR, `infection`, `recovery`, `person` (20), `day` |
| `rates/test_formulas.py` | `infection` (13), `recovery` (18), `transmission`, `seasonal`, `365` (6) |
| `rates/test_variables.py` | `age` (12), `young` (10), `infected`, `susceptible` |
| `rates/test_security.py` | `vaccination`, `exposed`, `susceptible`, `365` |
| `transitions/test_compartment_placeholder.py` | `death` (14), `treatment` (8), `young`, `recovery`, `infection` |
| `transitions/test_stratified_rates.py` | `young` (15), `age` (10), `infected`, `vaccination` |
| `transitions/test_flows.py` | `young`, `old`, `aging` |
| `transitions/test_impulses.py` | `mu`, `vaccination`, `birth`, `death` |
| `building/test_stratifications.py` | `age` (18), `vaccination` (17), `urban`, `rural` |
| `building/test_parameters.py` | `young`, `old`, `age`, `infection`, "people" |
| `initial_conditions/test_initial_state.py` | `age` (10), `vaccination` (7) |
| `calibration/test_constraints.py` | `infections` (5), `infected`, `recovered` |
| `calibration/test_initial_conditions.py` | `infected` (6), `age` |
| `calibration/test_write_back.py`, `initial_conditions/test_stratification_fractions.py`, `reporting/test_applied_parameters.py` | `age` |
| `axis_patterns/test_patterns.py` | `elderly` (11), `age` (7) |
| `plotting/test_plotter.py` | `infected`, `susceptible`, `recovered` |
| `simulation/test_runs.py`, `building/test_builder.py`, `building/test_reserved_names.py`, `calibration/test_observations.py`, `calibration/test_optimizers.py` | `infection`, `recovery` |

Words found by the vocabulary inventory (see Method), in addition to the
table above:

| Module | Words |
|---|---|
| `building/test_stratifications.py:10, 70-74`, `initial_conditions/test_initial_state.py:72-76` | `comorbidities` |
| `building/test_parameters.py:25-34` | `contact_multiplier` |
| `building/test_expansion.py:36-37`, `initial_conditions/test_stratification_fractions.py:178-186`, `transitions/test_stratified_rates.py:155-168` | `care` (`home`, `hosp`) |
| `calibration/test_write_back.py`, `initial_conditions/test_stratification_fractions.py:155-186`, `reporting/test_applied_parameters.py:155-162`, `support/models.py:294-315` | `sex` |
| `probabilistic/test_selection.py:30, 59, 267, 322, 329` | output `admissions` |
| `rates/test_formulas.py:102, 137-141` | `mass-action`, `intervention_time` |
| `units/test_print_equations_units.py:17-168` | `_reinfection`, "Chemical Reaction", "Reactant", `mol` |
| `units/test_model_units.py` | `svir`, `vax`, `vaccinated`, `senior`, `cow` |
| `transitions/test_flows.py` | `stoichiometric`, `rec` |
| `support/oracles.py:92` | "the threshold of invasion" |
| `crates/commol-differential/src/tests.rs` | SIS model |

### F3. Tests that state the axis is time

| # | Where | Text |
|---|---|---|
| F3.1 | `tests/axis_patterns/test_pattern_runs.py:83, 113` | "A pattern holds for one time unit", "four time units". |
| F3.2 | `tests/axis_patterns/test_pattern_runs.py:117-118` | `TestSeasonalIntegration.test_seasonal_time_average_near_baseline`. |
| F3.3 | `tests/axis_patterns/test_pattern_runs.py:294` | `test_generated_time_series_parameter_collision_raises`: "time series parameter" is the old name of "series parameter" (`docs/about/changelog.md:142`). |
| F3.4 | `tests/simulation/test_differential.py:1` | "the ODE engine, `run_at` and the time rule". |
| F3.5 | `tests/support/oracles.py:102` | "A time long enough for the dominant growth rate". |
| F3.6 | `tests/units/test_unit_checks.py:21-26` | Defines `person` as the dimension `[population]` for the unit tests. |
| F3.7 | `rates/test_formulas.py:114-177`, `rates/test_security.py:31, 46`, `units/test_model_units.py:113-122, 259-272`, `units/test_unit_checks.py:63-200` | Seasonal forcing with a period of `365`. |

### F4. Not violations

| Where | Why |
|---|---|
| `tests/simulation/test_time_scale.py` | Tests `TimeScale`, a time helper (D4). |
| `tests/units/` tests of time units (`1/day` vs `1/week`, inference) | They test D3 and D5, which are about units of time. `person` could still be `item`. |
| `calibration/test_optimizers.py:51-57`, `calibration/test_initial_conditions.py:120` | PSO `time_varying` acceleration. |
| `building/test_unknown_fields.py:31-32` | Checks that the removed field `time_steps` is rejected. |
| `reporting/test_applied_parameters.py:3, 388-390` | Wall-clock time of a performance check. |
| `units/test_model_units.py:426` | States that the axis need not be time. |
| `tests/reference_models/` (24 modules) | Published models, domain on purpose (`conventions.md:235-237`). See B6. |

## Specs

Every spec follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done).

### S1 — Text says "axis", not "time"

- Fix: A1–A4 and A6–A25. Statements say "axis", "position", "step" or "along
  the axis". The user-facing errors A12 and A13 say "Constraint '…' evaluation
  failed at position …" and "Position … not found in simulation results for
  constraint '…'". The internal names of A14 and A15 become
  `position_constraint_indices`, `evaluate_position_constraints` and
  `position`. The heading of A19 becomes "Rates that change along the axis",
  and every link to its anchor follows.
- Fix: F3.1–F3.5 in the tests that are not copied from the literature.
- Check: `git grep -i -E "time.(varying|dependent|step)|over time"` outside
  `odd/`, `tests/reference_models/`, the changelog and the "Not violations"
  finds only the names that stay.

### S2 — Text names no application

- Fix: B1–B11. B6 says "named after the published model". B5 keeps its record
  and uses abstract names. B11 drops the analogy with a reaction, and its
  examples use the id `t1`.
- Check: the audit search terms find no match outside the excepted places.

### S3 — "Proportional" replaces "per-capita" (D1)

- Fix: "Absolute and proportional rates" in the guides, the design decision
  and the error of `crates/commol-model/src/flows.rs`; `per_capita_of` becomes
  `proportional_to` in Rust, the binding, the stub and
  `context/_model/engine_state.py`. Every anchor `#absolute-and-per-capita-rates`
  follows.
- Check: `git grep -i "per.capita"` finds only the changelog of released
  versions and the "Before" column.

### S4 — `AxisPattern.sinusoid` replaces `seasonal` (D2)

- Fix: `AxisPattern.sinusoid`, class `_SinusoidAxisPattern`, the guide, the
  API page, the tests and a changelog "Changed" row.
- Check: `git grep -i seasonal` finds only the changelog of released versions
  and the "Before" column.

### S5 — Examples use abstract names (D3)

- Fix: the places of E. Bins `A`, `B`, `C`; parameters `k1`, `k2`;
  categories `group1`, `cat1`; the amount unit `item`. Time units (`1/day`,
  `365`) may stay, because time is the most common axis (`axis-positions`
  L8), but not names of seasons or of an application.
- Check: `pytest -m docs`; the audit search terms in docstrings, guides and
  Rust docs.

### S6 — Tests use abstract names, except the literature (D4)

- Fix: `conventions.md` (F1.1, B6, B7) prescribes abstract models. The
  factories of `tests/support/models.py` are named after their structure
  (`sir_builder` → `abc_builder`, `seir_builder` → `abcd_builder`), with bins
  `A`, `B`, `C`, `D`, parameters `k1`, `k2`, `k3` and categories `group1`. The
  fixtures, `reference.py` (`sir_flows`, `SIR_START`) and the modules of F2
  and F3 follow, and the Rust unit and integration tests of E.
- Not changed: `tests/reference_models/` and the support code only they use
  (`support/oracles.py`, `support/flows.py`), copied from the literature.
- Check: every runner; the audit search terms in `py-commol/tests/` find only
  the excepted places.

## Decisions (the user's to make)

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | S3 | Term for "per-capita" | "proportional"; Rust `per_capita_of` → `proportional_to` | "relative": clashes with `rtol` ("relative tolerance", `crates/commol-core/src/types/engine.rs:392`) and does not say relative to what. "per-unit": clashes with units of measure (`step_unit`, the unit check). "specific", "first-order": jargon of chemistry or kinetics. | approved (L1) |
| D2 | S4 | Name of `AxisPattern.seasonal` | `sinusoid` | `sine`: next to the function `sin` of the rate expressions. `harmonic`: suggests multiples of a base frequency. `periodic`: taken (a train of pulses). `sinusoid` names a shape, as `gaussian_pulse` and `linear_ramp` do. | approved (L1) |
| D3 | S5 | What may stay in examples | Abstract ids; time units stay; the unit `person` becomes `item` | `axis-positions` L8 allows time in examples, the most common axis. `person` is a unit of one application. `item` needs no registration: an unknown unit gets its own dimension. | approved (L1) |
| D4 | S6 | Change the tests too | Yes, except the tests copied from the literature (`tests/reference_models/`, `support/oracles.py`, `support/flows.py`) | Keep them: cheaper, but the test convention would keep prescribing what `mental-model.md:28` forbids, and every new test would copy SIR. Change the literature tests too: they are concrete examples copied from references, and their names must match the source. | approved (L1) |
| D5 | — | Order and delivery | One ODD, one PR, before `axis-scale-refactor` | After it: that ODD writes "per-capita" (S4) and rewrites tests of `units/`, which would migrate twice. Two PRs: the ODD is archived in the PR that implements it; a split gains nothing. | approved (L1) |

## Tasks

| ID | Specs | Task | Route | Status |
|---|---|---|---|---|
| T1 | S1, S2, S5 | Text and examples of the library, the Rust crates and the docs (A, B, E) | direct | done |
| T2 | S3 | "proportional" | direct | done |
| T3 | S4 | `AxisPattern.sinusoid` | direct | done |
| T4 | S6 | Test convention, `tests/support/`, test modules, Rust tests (F) | direct | done |
| T5 | all | Final audit: every runner and the searches of S1–S6 | direct | done |

T1 and T5 of the first plan (text, examples) touch the same files, so they ran
as one task.

## Delivery

- Strategy: one PR to `develop` from `refactor/domain-agnostic`, squash-merged
  (`refactor(api): …`). The PR archives this ODD.
- Order: T1 → T6.
- Version impact: breaking (`AxisPattern.seasonal`); a changelog "Changed"
  row. It ships in 0.3.0 with the other breaks.

## Evidence

Runners on `refactor/domain-agnostic` (2026-10-14), before commit:

| Runner | Result |
|---|---|
| Rust tests (six crates) | all pass; no `#[test]` added or removed, only renamed |
| `cargo test -p commol --features test-kind` | 4 passed |
| `cargo fmt --check`, `clippy -D warnings`, `cargo doc` with `-D warnings` | clean |
| `pytest` | 994 passed, 3 xfailed; `develop` had 993 (12 tests renamed, +1 `test_a_greek_letter_id_prints_as_its_latex_command`) |
| `pytest -m docs` | 62 passed |
| `ruff check`, `ruff format --check`, `ty check commol`, `vulture` | clean |
| `mkdocs build --strict` | built, exit 0 |

The searches of S1–S6 (`git grep`, outside `odd/`, the changelog,
`tests/reference_models/`, `support/oracles.py` and `support/flows.py`) find
only:

- "time-varying" of the PSO acceleration (TVAC) and the removed field
  `time_steps` that `test_unknown_fields.py` checks is rejected ("Not
  violations").
- "per-capita" in the design decision that says why the word goes.
- The license, "child" of a parse tree, "old" in the sense of "earlier", the
  Greek letter `beta` of the LaTeX list, `beta` of a release tag, and NSGA-II
  `population_size`.
- Units of time (`1/day`, `365`, `TimeScale`, the time aliases): time stays in
  examples (D3), and the unit check and `TimeScale` are
  `axis-scale-refactor` S4 and D4.

Found on the way:

- The LaTeX tests of `print_equations` were the only check that a Greek letter
  id prints as its LaTeX command (`beta` → `\beta`). With `k1` they lost it, so
  `test_a_greek_letter_id_prints_as_its_latex_command` keeps it.
- `print_equations` lists the sources of a transition in name order, so the
  header of `A + B → 2B` reads `A, B -> B` where `I + S → 2I` read `I, S -> I`.
- `abc_builder` names its bins `State A`, `State B`, `State C`, so the plotter
  test still tells a bin name from its id.
- "proportional" is also a value of `cluster_representative_strategy`, in its
  plain sense (proportional to the cluster size). The two uses do not clash.

## Log

- L1 (user, 2026-10-14): approved D1, D2, D3 and D5. For D4: the tests that
  come from the literature keep their names, because they are concrete
  examples copied from references. Only `tests/reference_models/` and the
  support code only they use (`oracles.py`, `flows.py`) are such tests.
- L2 (agent, 2026-10-14): ran T1 → T5 on `refactor/domain-agnostic`. The SEIR
  model of the tests is `abcd_builder` (A → B → C → D), so `B` is the stage
  between `A` and `C`; the conditional and shared-category factories use
  `group` (`g1`, `g2`), `type` (`t1`, `t2`) and `level` (`c1`, `c2`).
