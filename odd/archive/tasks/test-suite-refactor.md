# Feature: Refactor of the Python test suite

Goal: refactor `py-commol/tests/` so that a reader finds the tests of a feature
in one place, every test checks a result, and the new tests of
`odd/tasks/reference-model-tests.md` have a structure and shared helpers to
build on. Tests and contributor docs only; no change to `commol/` or `crates/`.
Branch: `test/suite-layout` from `develop`, after PR #99 merges.
Engram mirror: topic `odd/test-suite-refactor/tasks` (project `commol`).
Runners: `maturin develop --release` then `pytest`, `pytest -m docs` and
`pytest --cov=commol` (from `py-commol/`), `ruff check . && ruff format --check .
&& ty check commol`, `mkdocs build --strict` (from the root).

Baseline (2026-10-13, `fix/code-findings` at a887f81): 719 tests pass in 20 s;
26 test files, 14,729 lines.

## Specs

### S1 — One package per feature, as the user guide

- Cause: files mix three criteria. Some follow an API object (`test_model.py`,
  2,087 lines, one `TestModel` class of 1,900 lines), some a feature
  (`test_time_patterns.py`), and two directories nest one file each
  (`test_models_execution/test_difference_equations/test_difference_equations_sir.py`,
  `test_transition_rate/`). Stratification tests live in the SIR execution file.
- Fix (D2): one package per page of `docs/guide/`, so "where is the test of X"
  has the same answer as "where is the guide of X".
- Check: every test file sits in the package of its feature; no test module is
  over ~600 lines; `docs/codebase/repository-map.md` (L54) describes the new
  layout.

### S2 — Shared model factories and assertions

- Cause: there is no `conftest.py`. Tests call `ModelBuilder(` more than 140
  times, and more than 40 private helpers (`_model`, `_problem`,
  `_stratified_model`, …) repeat the same small models in different files.
- Fix: `tests/support/` holds model factories (SIR, SEIR, stratified SIR, A→B)
  with keyword overrides, a run helper, and assertions
  (`assert_population_conserved`, `assert_close` with a stated tolerance).
  `tests/conftest.py` exposes them as fixtures where a fixture helps.
- Check: no two test files define the same model by hand; `rg "def _model"`
  finds only models that are specific to one test.

### S3 — Every test checks a result

- Cause: some tests check only that a call does not fail
  (`TestStratifiedSIRSimulation::test_stratified_sir_runs_successfully`).
  A wrong engine passes them.
- Fix: each such test gets an oracle: conservation of population, an analytic
  value, or a value computed by hand in the test.
- Check: review list in Evidence; no test whose only assertion is "no
  exception" unless an exception is the subject of the test.

### S4 — No duplicate tests and no deprecated pytest use

- Cause: parameter-name validation is tested in
  `test_difference_equations_sir.py::TestParameterNameValidation` and in
  `test_reserved_names.py`. A class-scoped fixture defined as an instance
  method raises `PytestRemovedIn10Warning`; `pyproject.toml` allows
  `pytest<10`, so it fails in pytest 10.
- Fix: one home per behavior; fixtures follow pytest 9 rules.
- Check: `pytest -W error::pytest.PytestRemovedIn10Warning` passes; the
  duplicate pairs and their kept test are listed in Evidence.

### S5 — The refactor does not lose coverage

- Fix: this ODD changes tests only, never `commol/` or `crates/`.
- Check: line coverage of `commol/` (`pytest --cov=commol`) is not lower than
  the baseline; every removed test names the kept test that covers it
  (Evidence).

### S6 — Documentation

Follows the [documentation definition of done](../../docs/codebase/conventions.md#documentation-definition-of-done):
`docs/codebase/repository-map.md` (test layout), a "Tests" section in
`docs/codebase/conventions.md` (packages, factories, the oracle rule,
tolerances), and `AGENTS.md` if a command changes. No changelog entry:
nothing visible to users changes.

## Decisions (the user's to make)

Each row says why the recommendation beats each alternative.

| ID | Spec | Question | Recommendation | Why, against each alternative | Status |
|---|---|---|---|---|---|
| D1 | all | Order and number of PRs? | The refactor is its own PR, first; the ODE engine (`ode-engine.md`) and the reference tests (`reference-model-tests.md`) follow. Start after PR #99 merges. | One PR with the reference tests: a reviewer cannot check that the refactor lost no test when many new tests arrive with it. Reference tests first: they would use the old helpers and move again in the refactor. Start before #99: #99 edits many test files, so the refactor would conflict. | approved (L2) |
| D2 | S1 | Layout of `tests/`? | One package per guide page (`building/`, `transitions/`, `rates/`, `time_patterns/`, `initial_conditions/`, `simulation/`, `units/`, `calibration/`, `probabilistic/`, `reporting/`, `plotting/`, `docs/`) plus `support/`; `reference_models/` comes with `reference-model-tests.md`. | Mirror the source modules (`api/`, `context/`): behavior crosses modules and the engine owns it (golden rule), so one feature would spread over several packages. Keep flat files: the 2,000-line files stay and the reader still searches. | approved (L3) |

## Tasks

| ID | Specs | Task | Route | Commit |
|---|---|---|---|---|
| T1 | S5 | Baseline: test count, `pytest --cov=commol` report, warnings; record in Evidence | direct | #100 (`c3cc346`) |
| T2 | S2 | `tests/support/` factories and assertions, `conftest.py` | direct | #100 (`c3cc346`) |
| T3 | S1, S2 | Move and split files into the D2 packages; replace hand-built models with factories | direct | #100 (`c3cc346`) |
| T4 | S3, S4 | Oracles for smoke tests; merge duplicates; fix deprecated fixtures | direct | #100 (`c3cc346`) |
| T5 | S5, S6 | Coverage and count check against T1; `repository-map.md`, "Tests" section in `conventions.md`; PR | direct | #100 (`c3cc346`) |

## Delivery

- Strategy: one PR to `develop` from `test/suite-layout`, squash-merged
  (`test(suite): …` commits).
- Order: after PR #99 merges; T1 → T5. Then `ode-engine.md`, then
  `reference-model-tests.md`, then `library-benchmark.md`.
- Version impact: none (tests and contributor docs only).

## Evidence

### T1 — Baseline

Measured on `test/suite-layout` at 9819489 (`develop` with PR #99), after
`maturin develop --release`:

| Measure | Value |
|---|---|
| `pytest` | 719 passed in 16 s; 26 test files, 14,729 lines |
| `pytest --cov=commol` | 4,405 statements, 436 missed: 90 % |
| `pytest -m docs` | 55 passed |
| Warnings | 5 `PytestRemovedIn10Warning`: class-scoped fixtures written as instance methods (`TestCalibrator.model`, `TestCalibrator.model_with_accumulator`, `TestSimulationPlotter.sir_model`, `TestSIR.sir_model`, `TestStratifiedSIRSimulation.age_stratified_sir_model`). With `-W error::pytest.PytestRemovedIn10Warning`, every test of those classes errors. |

Three hidden dependencies between tests, removed by the refactor:

- `test_units.py` defined `person`, `time` and `semester` in the global pint
  registry in `setup_class`; the custom-unit test passed only because an
  earlier class had run. Models now register their own units
  (`helpers.py:319` is now covered); the function-level checks define
  `person` and `time` in a module fixture (`tests/units/test_unit_checks.py`).
- `test_probabilistic_calibrator.py` shared a module-scoped model, and
  `test_windowed_median_matches_a_member_wise_reduction` wrote calibrated
  values into it (`Simulation` does not copy its model). The fixture is now
  the function-scoped `sir_model`.
- `TestCalibrator.model_with_accumulator` was class-scoped and changed by
  `update_parameters`. The test now builds its own model.

### T2–T4 — Result

| Measure | Before | After |
|---|---|---|
| Tests | 719 | 749 (parametrized cases replace loops; no test removed except the duplicates below) |
| Test modules | 26 files, 14,729 lines | 43 files in 12 packages, 9,363 lines; largest `units/test_print_equations_stratified.py`, 558 lines |
| Shared code | none | `tests/support/` (models, reference solver, assertions, problems, run helpers) and three `conftest.py` |
| `ModelBuilder(` outside `support/` | more than 140 | 22: models one test needs (bins sharing a prefix, a reaction, A → B → C, SVIR) and error paths of the builder |
| `pytest -W error::pytest.PytestRemovedIn10Warning` | errors | passes; now permanent through `filterwarnings` in `pyproject.toml` (pytest `>=9.0`) |

S1: every test module sits in the package of its guide page
(`docs/codebase/conventions.md#tests`); no module is over 600 lines.

S2: the remaining `def _…model` helpers are one-line calls of a factory with
the keywords of their file, or a model only that file needs.

### S3 — Tests that checked no result, and their oracle

| Test (old name) | Checked before | Oracle now |
|---|---|---|
| `test_transition_rate_sir_models.py`: 10 rate families (`test_constant_rate_sir_model` … `test_trigonometric_rate_combinations`) | nothing (`_ = simulation.run(10)`, a `TODO`) | Whole trajectory against `reference_run`, `rel=1e-9`, and conservation (`rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes`) |
| `TestMathematicalFunctionComprehensive`: 5 tests, 21 functions | result length, non-negative values | Trajectory against the function evaluated in Python, one case per function (`test_a_function_gives_the_value_it_computes`) |
| `test_very_small_rates`, `test_max_min_functions_in_rates`, `test_floor_ceil_round_functions`, `test_division_by_zero_scenarios`, `test_zero_rate_scenarios` | length, conservation or "R grows" | Trajectory against `reference_run` |
| `test_undefined_parameters_in_formulas` | a `try` that passed whatever happened | `ValueError` "Undefined variables in transition" at build |
| `TestStratifiedSIRSimulation::test_stratified_sir_runs_successfully` | names, conservation, "S falls" | Trajectory of the six compartments against `reference_run` (`rates/test_variables.py`) |
| `test_stratified_sir_with_default_and_specific_rates` | young falls faster than old | Trajectory, and step 1 by hand: 495 − 3.96 and 495 − 1.98 |
| `test_death_transition_simulation` | total does not grow | `N(t) = 1000 · 0.99^t` for 200 steps |
| `TestEndToEndOverride::test_zero_override_keeps_the_other_stratum_empty` | IH grows, S falls | Step 1 by hand: `beta * S` moves 9 from each oe60 compartment |
| `test_full_calibration_workflow` | output has 101 values | The rerun of the fitted model matches the truth, `rel=1e-3` |
| `test_parameter_with_formula_referencing_*` | the formula is stored | The run equals the run with the formula's value |
| `test_plotter.py`: 8 tests (`test_plot_series_basic`, `…_with_observed_data`, `…_subset_of_bins`, `…_with_seaborn_config`, `test_plot_cumulative_basic`, `…_with_observed_data`, both `…_with_kwargs`) | `fig is not None` | The drawn lines equal the results (or their running sums), the markers equal the observations, titles, line width, alpha and style |
| `TestScheduleIntegration::test_default_covers_unmatched_groups`, `test_differential_schedules`, `test_cross_category_routing` | "grows", "more events" | Exact flows (250, 100, 500) and event counts (9 and 3) |
| `TestFullModelUnitFailures` and similar | raises `UnitConsistencyError` | Also names the faulty expression in the message |
| `test_zero_amount_is_legal`, `test_large_step_count_accepted` | the formula is not empty | The exact formula; 50 `step ==` terms |
| `test_probabilistic_config_accepts_any_loss_family` | the config is not `None` | The problem keeps the loss and the config |

Tests whose subject is an exception or its absence keep that check: the
security validators (`rates/test_security.py`, valid expressions), a model
that passes `check_unit_consistency` (`units/test_model_units.py::TestConsistentModels`),
and the accepted ids of `calibration/test_initial_conditions.py::TestIds`.

### S4 — Duplicates

| Removed test | Kept test |
|---|---|
| `test_difference_equations_sir.py::TestParameterNameValidation::test_parameter_name_conflicts_with_bin_id_raises_error` | `building/test_reserved_names.py::test_a_bin_id_is_not_a_parameter_id` |
| `…::test_parameter_name_conflicts_with_bin_id_raises_error_with_stratification` | `building/test_reserved_names.py::test_a_bin_id_is_not_a_parameter_id` (its model is stratified) |
| `test_units.py::TestModelUnitConsistency::test_sir_model_with_inconsistent_units` | `units/test_model_units.py::TestInconsistentModels::test_a_rate_in_the_wrong_unit_names_its_transition` |
| `test_units.py::TestFullModelUnitFailures::test_constant_parameter_without_unit_raises_error` | `units/test_model_units.py::TestMissingUnits::test_one_parameter_without_a_unit_is_named` |
| `test_difference_equations_sir.py::TestConditionalStratification::test_no_full_cartesian_compartments` | `building/test_stratifications.py::TestConditionalStratification::test_vaccination_splits_only_the_oe60_compartments` (asserts the exact set) |

`TestParameterNameValidation::test_multiple_conflicting_parameters_all_reported`
was not a duplicate: it moved to `building/test_reserved_names.py` and now
checks both names (`['I', 'S']`). The second assertion that the default
ensemble backend is NSGA-II was removed from
`probabilistic/test_selection.py::test_ensemble_selection_configs_expose_both_algorithms`;
`probabilistic/test_calibrator.py::test_ensemble_configuration_defaults_to_nsga2`
keeps it.

### S5 — Coverage

| Measure | Baseline | After |
|---|---|---|
| `pytest --cov=commol` | 4,405 statements, 436 missed | 4,405 statements, 435 missed |
| Lines that lost coverage (per file, against the baseline JSON) | — | 0 |
| Lines that gained coverage | — | `context/_model/helpers.py:319` |

Every old test is in the new suite under its old name, or in the table of S4,
or in this map (checked by script: every target exists):

<details>
<summary>143 tests renamed, moved or parametrized</summary>

| Old test (old file) | New test |
|---|---|
| `test_calibrate_initial_condition` (test_calibrator.py) | `calibration/test_initial_conditions.py::TestFitting::test_an_initial_fraction_is_recovered` |
| `test_calibrate_parameter_and_initial_condition_together` (test_calibrator.py) | `calibration/test_initial_conditions.py::TestFitting::test_a_parameter_and_an_initial_fraction_together` |
| `test_calibration_with_compartment_sum_constraint` (test_calibrator.py) | `calibration/test_constraints.py::TestOutputConstraints::test_a_bound_on_a_sum_of_outputs` |
| `test_calibration_with_constraint` (test_calibrator.py) | `calibration/test_constraints.py::TestParameterConstraints::test_a_ratio_bound` |
| `test_calibration_with_linear_constraint` (test_calibrator.py) | `calibration/test_constraints.py::TestParameterConstraints::test_a_linear_sum_bound` |
| `test_calibration_with_multiple_constraints` (test_calibrator.py) | `calibration/test_constraints.py::TestParameterConstraints::test_several_constraints_hold_together` |
| `test_calibration_with_ordering_constraint` (test_calibrator.py) | `calibration/test_constraints.py::TestParameterConstraints::test_an_ordering` |
| `test_calibration_with_time_dependent_constraint` (test_calibrator.py) | `calibration/test_constraints.py::TestOutputConstraints::test_an_output_bound_at_given_steps` |
| `test_constraint_expression_security_validation` (test_calibrator.py) | `calibration/test_constraints.py::TestExpressionSecurity::test_a_dangerous_expression_is_rejected[eval(beta)-None]` |
| `test_constraint_n_excludes_accumulators` (test_calibrator.py) | `calibration/test_constraints.py::TestOutputConstraints::test_n_is_the_population_without_accumulators` |
| `test_full_calibration_workflow` (test_calibrator.py) | `calibration/test_problem.py::TestWorkflow::test_calibrate_write_back_and_simulate` |
| `test_get_uncalibrated_parameters` (test_calibrator.py) | `building/test_parameters.py::TestUncalibratedParameters::test_the_model_lists_the_parameters_without_a_value` |
| `test_invalid_bin_id_for_initial_condition_raises_error` (test_calibrator.py) | `calibration/test_initial_conditions.py::TestIds::test_an_unknown_bin_of_an_unstratified_model_is_rejected` |
| `test_model_calibration_nelder_mead` (test_calibrator.py) | `calibration/test_optimizers.py::TestRecovery::test_nelder_mead` |
| `test_model_calibration_particle_swarm` (test_calibrator.py) | `calibration/test_optimizers.py::TestRecovery::test_particle_swarm` |
| `test_multiple_scales_for_different_compartments` (test_calibrator.py) | `calibration/test_observations.py::TestScaleFactors::test_each_output_has_its_own_scale` |
| `test_parameter_is_calibrated_method` (test_calibrator.py) | `building/test_parameters.py::TestUncalibratedParameters::test_a_parameter_with_a_value_is_calibrated` |
| `test_parameter_with_none_value` (test_calibrator.py) | `building/test_parameters.py::TestUncalibratedParameters::test_a_parameter_may_have_no_value` |
| `test_particle_swarm_with_advanced_features` (test_calibrator.py) | `calibration/test_optimizers.py::TestRecovery::test_particle_swarm_with_time_varying_acceleration_and_mutation` |
| `test_per_compartment_initial_condition_is_written_back_not_silently_skipped` (test_calibrator.py) | `calibration/test_write_back.py::TestApplyCalibrationParameters::test_per_compartment_initial_condition_is_written_back` |
| `test_scale_and_parameter_calibration_combined` (test_calibrator.py) | `calibration/test_observations.py::TestScaleFactors::test_a_scale_and_a_parameter_are_recovered_together` |
| `test_scale_parameter_calibration` (test_calibrator.py) | `calibration/test_observations.py::TestScaleFactors::test_a_scale_alone_is_recovered` |
| `test_simulation_fails_with_uncalibrated_parameters` (test_calibrator.py) | `simulation/test_runs.py::TestOutputFormats::test_a_run_with_an_uncalibrated_parameter_is_rejected` |
| `test_update_parameters` (test_calibrator.py) | `building/test_parameters.py::TestUncalibratedParameters::test_update_parameters_sets_the_values` |
| `test_update_parameters_with_invalid_id` (test_calibrator.py) | `building/test_parameters.py::TestUncalibratedParameters::test_update_parameters_rejects_an_unknown_id` |
| `test_building_a_simulation_raises_a_catchable_error` (test_compartment_initial_conditions.py) | `initial_conditions/test_compartment_fractions.py::TestCompartmentFractionValidation::test_an_unknown_override_fails_when_the_simulation_is_built` |
| `test_every_declared_initial_conditions_attribute_is_readable` (test_compartment_initial_conditions.py) | `initial_conditions/test_initial_state.py::TestInitialConditionsBinding::test_every_declared_attribute_is_readable` |
| `test_initial_state_binding_reports_bins_and_categories` (test_compartment_initial_conditions.py) | `initial_conditions/test_initial_state.py::TestInitialState::test_the_binding_reports_bins_and_categories` |
| `test_low_level_binding_exposes_and_applies_overrides` (test_compartment_initial_conditions.py) | `initial_conditions/test_initial_state.py::TestInitialConditionsBinding::test_overrides_reach_the_engine` |
| `test_negative_population_size_is_a_domain_error` (test_compartment_initial_conditions.py) | `initial_conditions/test_bin_fractions.py::TestPopulationSize::test_a_non_positive_population_size_is_rejected[-10]` |
| `test_remainder_bin_with_a_longer_id_is_not_an_overlap` (test_compartment_initial_conditions.py) | `calibration/test_write_back.py::TestWriteBackReadability::test_a_remainder_bin_with_a_longer_id_is_not_an_overlap` |
| `test_uniform_expansion_is_preserved_without_overrides` (test_compartment_initial_conditions.py) | `initial_conditions/test_initial_state.py::TestExpansion::test_fractions_multiply_without_overrides` |
| `test_update_rejects_a_bare_ambiguous_category` (test_compartment_initial_conditions.py) | `initial_conditions/test_stratification_fractions.py::TestUpdateStratificationFractions::test_a_bare_shared_category_is_rejected_as_ambiguous` |
| `test_zero_population_size_is_a_domain_error` (test_compartment_initial_conditions.py) | `initial_conditions/test_bin_fractions.py::TestPopulationSize::test_a_non_positive_population_size_is_rejected[0]` |
| `test_base_compartment_equals_sum_of_stratified` (test_difference_equations_sir.py) | `rates/test_variables.py::TestBinTotals::test_a_bin_id_equals_the_explicit_sum` |
| `test_compartment_names_single_conditional` (test_difference_equations_sir.py) | `building/test_stratifications.py::TestConditionalStratification::test_vaccination_splits_only_the_oe60_compartments` |
| `test_forward_reference_in_conditions_raises_error` (test_difference_equations_sir.py) | `building/test_stratifications.py::TestConditionalStratification::test_a_condition_on_an_undeclared_stratification_is_rejected` |
| `test_initial_population_distribution` (test_difference_equations_sir.py) | `initial_conditions/test_initial_state.py::TestExpansion::test_a_conditional_stratification_splits_only_its_category` |
| `test_multi_stratified_sir_with_base_compartments` (test_difference_equations_sir.py) | `building/test_stratifications.py::TestFullProduct::test_a_run_of_a_two_stratification_model_conserves_the_population` |
| `test_multiple_conflicting_parameters_all_reported` (test_difference_equations_sir.py) | `building/test_reserved_names.py::test_every_conflicting_parameter_id_is_reported` |
| `test_no_conditions_produces_full_cartesian_product` (test_difference_equations_sir.py) | `building/test_stratifications.py::TestFullProduct::test_two_stratifications_give_every_combination` |
| `test_simulation_conserves_population` (test_difference_equations_sir.py) | `building/test_stratifications.py::TestConditionalStratification::test_a_run_conserves_the_population` |
| `test_sir_dict_of_lists_output` (test_difference_equations_sir.py) | `simulation/test_runs.py::TestOutputFormats::test_dict_of_lists_is_the_default` |
| `test_sir_list_of_lists_output` (test_difference_equations_sir.py) | `simulation/test_runs.py::TestOutputFormats::test_list_of_lists_has_one_row_per_step` |
| `test_stratification_condition_without_category_raises_error` (test_difference_equations_sir.py) | `building/test_stratifications.py::TestConditionalStratification::test_a_condition_without_a_category_is_rejected` |
| `test_stratified_rates_match_conditional_compartments` (test_difference_equations_sir.py) | `transitions/test_stratified_rates.py::TestConditionalCompartments::test_each_conditional_group_takes_its_own_rate` |
| `test_stratified_sir_runs_successfully` (test_difference_equations_sir.py) | `rates/test_variables.py::TestBinTotals::test_a_bin_id_in_a_rate_sums_its_compartments` |
| `test_stratified_sir_with_default_and_specific_rates` (test_difference_equations_sir.py) | `transitions/test_stratified_rates.py::TestFallbackRate::test_a_group_without_a_stratified_rate_takes_the_rate` |
| `test_three_conditional_stratifications_compartment_names` (test_difference_equations_sir.py) | `building/test_stratifications.py::TestConditionalStratification::test_two_conditional_stratifications_on_one_category` |
| `test_three_conditional_stratifications_initial_population` (test_difference_equations_sir.py) | `initial_conditions/test_initial_state.py::TestExpansion::test_two_conditional_stratifications_split_their_category_twice` |
| `test_parameter_with_formula_referencing_N` (test_model.py) | `building/test_parameters.py::TestFormulaParameters::test_a_formula_may_use_subpopulation_totals` |
| `test_parameter_with_formula_referencing_other_parameters` (test_model.py) | `building/test_parameters.py::TestFormulaParameters::test_a_formula_of_other_parameters_runs_as_its_value` |
| `test_comparison_with_manual_expansion` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestRuns::test_the_expansion_runs_like_transitions_written_by_hand` |
| `test_compartment_in_complex_formula` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestExpansion::test_every_occurrence_in_a_formula_is_replaced` |
| `test_death_transition_simulation` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestRuns::test_a_death_rate_on_every_compartment_shrinks_the_total_geometrically` |
| `test_error_multiple_targets` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestErrors::test_several_sources_and_several_targets_are_rejected` |
| `test_error_single_source` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestErrors::test_a_single_source_is_rejected` |
| `test_expansion_with_empty_target` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestExpansion::test_an_empty_target_removes_from_each_source` |
| `test_expansion_with_single_target` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestExpansion::test_a_single_target_receives_from_each_source` |
| `test_with_stratified_rates` (test_model_builder.py) | `transitions/test_compartment_placeholder.py::TestExpansion::test_stratified_rates_are_expanded_too` |
| `test_plot_cumulative_basic` (test_plotter.py) | `plotting/test_plotter.py::TestSimulationPlotter::test_plot_cumulative_draws_the_running_sums` |
| `test_plot_cumulative_with_kwargs` (test_plotter.py) | `plotting/test_plotter.py::TestSimulationPlotter::test_plot_cumulative_passes_keywords_to_the_lines` |
| `test_plot_cumulative_with_observed_data` (test_plotter.py) | `plotting/test_plotter.py::TestSimulationPlotter::test_plot_cumulative_draws_the_running_sum_of_observations` |
| `test_plot_series_basic` (test_plotter.py) | `plotting/test_plotter.py::TestSimulationPlotter::test_plot_series_draws_every_bin` |
| `test_plot_series_with_kwargs` (test_plotter.py) | `plotting/test_plotter.py::TestSimulationPlotter::test_plot_series_passes_keywords_to_the_lines` |
| `test_plot_series_with_observed_data` (test_plotter.py) | `plotting/test_plotter.py::TestSimulationPlotter::test_plot_series_draws_the_observations` |
| `test_bin_rows_report_the_effective_head_count_with_overrides` (test_reporting.py) | `reporting/test_applied_parameters.py::TestCompartmentOverrides::test_bin_rows_report_the_effective_head_count` |
| `test_category_rows_report_the_effective_head_count_with_overrides` (test_reporting.py) | `reporting/test_applied_parameters.py::TestCompartmentOverrides::test_category_rows_report_the_effective_head_count` |
| `test_overrides_are_folded_into_multi_compartment_subgroups` (test_reporting.py) | `simulation/test_outputs.py::TestSubgroupPopulation::test_overrides_are_folded_into_each_subgroup` |
| `test_single_overridden_compartment_sets_the_subgroup_population` (test_reporting.py) | `simulation/test_outputs.py::TestSubgroupPopulation::test_a_single_overridden_compartment_sets_its_subgroup` |
| `test_constant_rate_sir_model` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[constant-numbers-are-per-capita]` |
| `test_division_by_zero_scenarios` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[guarded-division]` |
| `test_exponential_and_logarithmic_functions` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_function_gives_the_value_it_computes[exp]` |
| `test_exponential_decay_transmission` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[exponential-decay]` |
| `test_floor_ceil_round_functions` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[floor-of-step]` |
| `test_frequency_dependent_transmission` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[frequency-dependent]` |
| `test_hyperbolic_functions` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_function_gives_the_value_it_computes[sinh]` |
| `test_invalid_mathematical_expressions` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::TestInvalidFormulas::test_an_incomplete_expression_is_rejected[S * I /]` |
| `test_inverse_trigonometric_functions` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_function_gives_the_value_it_computes[asin]` |
| `test_logistic_transmission_rate` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[logistic]` |
| `test_mass_action_kinetics` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[mass-action]` |
| `test_max_min_functions_in_rates` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[min-and-max]` |
| `test_parameter_reference_rates` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[a-parameter-alone-is-per-capita]` |
| `test_power_law_recovery_rate` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[power-law-recovery]` |
| `test_seasonal_transmission` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[seasonal-sin-of-step]` |
| `test_step_function_intervention` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[sigmoid-step-of-t]` |
| `test_trigonometric_functions` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_function_gives_the_value_it_computes[sin]` |
| `test_trigonometric_rate_combinations` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[sin-and-cos]` |
| `test_undefined_parameters_in_formulas` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::TestInvalidFormulas::test_an_undefined_parameter_is_rejected_at_build` |
| `test_utility_mathematical_functions` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_function_gives_the_value_it_computes[abs]` |
| `test_very_small_rates` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_a_rate_formula_gives_the_flow_it_computes[very-small-rates]` |
| `test_zero_rate_scenarios` (test_transition_rate_sir_models.py) | `rates/test_formulas.py::test_without_a_recovery_transition_nobody_recovers` |
| `test_abs_function` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[abs]` |
| `test_abstract_time_unit` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[abstract-time]` |
| `test_age_structured_with_floor_dimensional_error` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_floor_of_a_population_in_a_stratified_rate_fails` |
| `test_atan2_with_mismatched_units_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[atan2-of-different-units]` |
| `test_check_equation_units_mismatch` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[mismatch]` |
| `test_check_equation_units_simple` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[mass-action]` |
| `test_combined_math_functions` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[combined-functions]` |
| `test_complex_intervention_model_with_min_max_errors` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_a_function_of_a_wrong_unit_fails[min-of-a-rate-and-persons]` |
| `test_complex_model_seasonal_wrong_sin_argument` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_sin_of_steps_over_days_fails` |
| `test_convertible_time_units_are_not_equal` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[convertible-time-units-are-not-equal]` |
| `test_custom_bin_unit_auto_registration` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_a_custom_bin_unit_and_a_time_alias_are_registered` |
| `test_dimensionless_parameters` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_a_dimensionless_parameter` |
| `test_division_in_math_function_correct_dimensionality` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[division-inside-a-function]` |
| `test_exp_with_dimensional_argument_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[dimensional-exp]` |
| `test_exponential_decay` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[exponential-decay]` |
| `test_floor_with_dimensional_argument_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[dimensional-floor]` |
| `test_formula_parameter_chained_dependencies` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_chained_formula_parameters` |
| `test_formula_parameter_missing_dependency_units` (test_units.py) | `units/test_model_units.py::TestMissingUnits::test_a_formula_of_an_unknown_variable_has_no_unit` |
| `test_formula_parameter_unit_inference` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_the_unit_of_a_formula_parameter_is_inferred` |
| `test_formula_parameter_with_N_reference` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_a_formula_of_population_totals_is_dimensionless` |
| `test_get_predefined_variable_units` (test_units.py) | `units/test_unit_checks.py::TestHelpers::test_population_totals_have_the_bin_unit` |
| `test_get_predefined_variable_units_multiple_stratifications` (test_units.py) | `units/test_unit_checks.py::TestHelpers::test_totals_of_category_combinations_have_the_bin_unit` |
| `test_log_with_dimensional_argument_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[dimensional-log]` |
| `test_logarithmic_rate` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[log]` |
| `test_math_function_result_unit_mismatch` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[result-of-a-function-has-the-wrong-unit]` |
| `test_min_max_functions` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[min-max]` |
| `test_min_max_with_incompatible_units_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[max-of-different-units]` |
| `test_mixed_units_some_parameters_without_units` (test_units.py) | `units/test_model_units.py::TestMissingUnits::test_one_parameter_without_a_unit_is_named` |
| `test_multi_stratified_model_with_nested_math_errors` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_log_of_persons_in_one_of_four_combinations_fails` |
| `test_nested_math_functions_with_wrong_units_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[nested-functions]` |
| `test_pow_function` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[pow]` |
| `test_pow_with_dimensional_exponent_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[dimensional-exponent]` |
| `test_seasonal_forcing_cos` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[seasonal-cos]` |
| `test_seasonal_forcing_sin` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[seasonal-sin]` |
| `test_seir_with_exponential_wrong_units` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_exp_of_days_fails` |
| `test_simple_sir_wrong_beta_units` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_a_rate_in_the_wrong_unit_names_its_transition` |
| `test_sin_with_dimensional_argument_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[dimensional-sin]` |
| `test_sir_model_with_consistent_units` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_sir` |
| `test_sir_model_with_dimensional_sin_argument_fails` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_a_function_of_a_wrong_unit_fails[sin-of-days]` |
| `test_sir_model_with_exponential_decay` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_exponential_decay_of_a_rate` |
| `test_sir_model_with_seasonal_forcing` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_seasonal_forcing` |
| `test_sir_model_without_units_raises_error` (test_units.py) | `units/test_model_units.py::TestMissingUnits::test_parameters_without_units_are_named` |
| `test_sir_with_log_of_population_fails` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_a_function_of_a_wrong_unit_fails[log-of-a-population]` |
| `test_sirs_model_with_atan2_wrong_units` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_a_function_of_a_wrong_unit_fails[atan2-of-persons-and-days]` |
| `test_sqrt_function` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_a_consistent_equation_passes[sqrt]` |
| `test_sqrt_with_dimensional_argument_fails` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_an_inconsistent_equation_is_reported[dimensional-sqrt]` |
| `test_stratified_model_with_consistent_units` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_rates_divided_by_population_totals` |
| `test_stratified_model_with_sqrt_wrong_units` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_sqrt_of_a_population_in_a_stratified_rate_fails` |
| `test_vaccination_model_with_pow_wrong_exponent` (test_units.py) | `units/test_model_units.py::TestInconsistentModels::test_pow_with_an_exponent_in_persons_fails` |
| `test_vaccination_rate_with_time_dependent_coverage` (test_units.py) | `units/test_model_units.py::TestConsistentModels::test_a_vaccination_rate` |
| `test_various_time_units` (test_units.py) | `units/test_unit_checks.py::TestCheckEquationUnits::test_every_unit_of_time_is_known[day]` |

</details>

### T5 — Checks

| Check | Result |
|---|---|
| `pytest` | 749 passed |
| `pytest -m docs` | 55 passed |
| `ruff check .`, `ruff format --check .`, `ty check commol` | pass |
| `mkdocs build --strict` | pass |
| `cargo test -p commol-core -p commol-difference -p commol-calibration` | pass |

## Log

- L1 (user, 2026-10-13): asked for a refactor of the test suite, together with
  the reference-model tests.
- L2 (user, 2026-10-13): D1 → "Dos PRs, refactor primero".
- L3 (user, 2026-10-13): D2 → "Paquetes de la guía", after asking why not one
  package per source module (answer: most tests check engine behavior with no
  Python module to mirror; a feature crosses several modules; modules couple
  tests to internals). Pure-Python helpers (`security.py`, `time_scale.py`,
  `api/probabilistic/`) go in the package of their feature.
- L4 (user, 2026-10-13): the refactor and the reference tests are separate
  deliveries with the ODE engine between them, so they are separate ODDs.
  This ODD holds the former Part 1 of `reference-model-tests.md`.
- L5 (user, 2026-10-13): "implementa" the ODD. Implemented T1–T5 on
  `test/suite-layout`. Outside the plan: `pytest>=9.0` and
  `filterwarnings` make the pytest 10 deprecation an error in every run; the
  calibration tests got one file per calibration guide page, and the
  write-back of calibrated initial conditions lives in
  `calibration/test_write_back.py`, next to its guide page.
- L6 (user, 2026-10-13): PR #100 squash-merged into `develop` as `c3cc346`
  (branch commit `b9e1f9d`); branch `test/suite-layout` deleted in local and
  remote; this ODD archived.
