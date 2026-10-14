# Audit: application and unit references in the library and its docs

Goal: list every place where the library or its documentation assumes an
application (epidemiology, demography) or a unit (time, persons), outside
examples and use cases.
Verified 2026-10-14 on `refactor/model-kind-interface` at 7664ee5.

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
- Not searched: `py-commol/tests/` (`tests/reference_models/` is domain
  vocabulary on purpose), `odd/`.
- Search terms: time words (`time`, `time-varying`, `time step`, `over time`,
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

## C. Vocabulary in the API itself

These are not single lines. They are names of the public API and of the
concepts in the docs. A change needs a decision first.

| # | Term | Size | Where it is public | Neutral term |
|---|---|---|---|---|
| C1 | "population" (amount in the compartments) | about 500 lines in 72 files | `population_size` (`ModelBuilder.set_initial_conditions`, `InitialConditions`, `crates/commol-core/src/types/compartmental.rs:84`), `subgroup_population` (`py-commol/commol/context/compartmental.py:490`, `initial_conditions.py:349`), `engine.population()`, "subpopulation" (`N_…` totals), `calibrated_initial_population`, `README.md:4`, `docs/index.md:3` | "total", "amount", "state" |
| C2 | "per-capita" (rate multiplied by its source) | 59 lines in 25 files | Concept name in `docs/guide/transitions.md:58` ("Absolute and per-capita rates"), `core-concepts.md:62`, `design-decisions.md:131`, the `absolute` field docs, error text in `crates/commol-model/src/flows.rs:216` | "proportional", "relative" |

## D. Design that assumes an application or a unit

| # | Where | Finding | Consequence |
|---|---|---|---|
| D1 | `py-commol/commol/context/initial_conditions.py:134` (`population_size: int`), `py-commol/commol/api/model_builder.py:751`, `crates/commol-core/src/types/compartmental.rs:84` (`population_size: u64`) | The total amount at the start must be a whole positive count. | A model of a mass, a concentration or a fraction (total `0.5`, `2.75`) cannot be stated. The type assumes discrete individuals. |
| D2 | `AxisPattern.seasonal` (`py-commol/commol/api/patterns.py:451`, class `_SeasonalAxisPattern` at `:656`; `docs/guide/axis-patterns.md:52, 58, 267`) | A sine pattern has a name from time (seasons). | The public name assumes a time axis. Neutral names: `sinusoid`, `sine`, `harmonic`. |
| D3 | `py-commol/commol/context/_model/units.py:14` (`TIME_DIMENSIONS = ("[time]", "[step]")`), `:169`; `design-decisions.md:224-234`; `docs/guide/units-and-equations.md:77-87` | The step unit is inferred only from units of time. Any other axis must declare `step_unit`. | Time is the default axis of the unit check. Documented and guarded on purpose (a `1/meter` parameter may be a density). Decide whether to keep it. |
| D4 | `TimeScale` (`py-commol/commol/api/time_scale.py`) | A public, time-only helper (`steps_per_week`, `steps_from_hours`). | Acceptable as an edge helper (see "Not violations"). An axis-neutral `AxisScale` would make the API symmetric. `odd/tasks/axis-scale-refactor.md` already plans to replace `TimeScale` with axis-neutral pieces. |

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

## Recommended order

1. A12, A13: user-facing error messages. Smallest change, most visible.
2. A1-A11, A16-A25, B1-B10: text edits only, no API change.
3. D1: `population_size` as a positive `float`. This is an API break, which
   the alpha status allows.
4. D2, C1, C2: renames of public names and concepts. Decide the neutral terms
   first.
5. E: rename example identifiers to `A`, `B`, `k1`, `group1` as
   `conventions.md:102-104` requires.
6. D3, D4: decide whether time keeps a special place in the unit check and in
   `TimeScale`.
