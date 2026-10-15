//! How a compartmental engine resolves and writes the initial conditions that
//! calibration sets (`InitialConditionPlan`), through `plan_values` and
//! `apply_values` of `SimulationEngine`.

use std::collections::HashMap;

use commol_core::{ModelValue, ModelValueType, SimulationEngine, SimulationError, ValuePlan};
use commol_model::{CompartmentalEngine, InitialConditionPlan, OutputLabels};

/// A compartmental engine that holds its state and does not move it.
/// Compartment names are `<bin>` or `<bin>_<category>` of one stratification
/// `group`, and the total is 1000.
#[derive(Clone)]
struct FrozenEngine {
    compartments: Vec<String>,
    initial_state: Vec<f64>,
    state: Vec<f64>,
    parameters: HashMap<String, f64>,
}

impl FrozenEngine {
    fn with(compartments: &[&str], state: &[f64]) -> Self {
        Self {
            compartments: compartments.iter().map(|name| name.to_string()).collect(),
            initial_state: state.to_vec(),
            state: state.to_vec(),
            parameters: HashMap::new(),
        }
    }

    /// Bins A and B split by `cat0` and `cat1`: 300, 200, 400, 100.
    fn two_bins() -> Self {
        Self::with(
            &["A_cat0", "A_cat1", "B_cat0", "B_cat1"],
            &[300.0, 200.0, 400.0, 100.0],
        )
    }

    /// The two bins and an unstratified bin C of 100.
    fn with_fixed_c() -> Self {
        Self::with(
            &["A_cat0", "A_cat1", "C", "B_cat0", "B_cat1"],
            &[300.0, 200.0, 100.0, 300.0, 100.0],
        )
    }

    /// Bins A and B split by `seg0` and `seg1`.
    fn with_other_categories() -> Self {
        Self::with(
            &["A_seg0", "A_seg1", "B_seg0", "B_seg1"],
            &[250.0, 250.0, 100.0, 400.0],
        )
    }
}

impl SimulationEngine for FrozenEngine {
    type ValuePlan = InitialConditionPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        buffer.clear();
        buffer.extend(positions.iter().map(|_| self.state.clone()));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        self.compartments.clone()
    }

    fn state_outputs(&self) -> usize {
        self.compartments.len()
    }

    fn reset(&mut self) {
        self.state.clone_from(&self.initial_state);
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(
        &self,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<InitialConditionPlan, String> {
        InitialConditionPlan::new(self, values, remainder)
    }

    fn apply_values(&mut self, plan: &InitialConditionPlan, values: &[f64]) -> Result<f64, String> {
        plan.apply(self, values)
    }
}

impl CompartmentalEngine for FrozenEngine {
    fn compartments(&self) -> Vec<String> {
        self.compartments.clone()
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        self.compartments
            .iter()
            .map(|name| match name.split_once('_') {
                Some((bin, category)) => OutputLabels {
                    source: bin.to_string(),
                    categories: vec![("group".to_string(), category.to_string())],
                },
                None => OutputLabels {
                    source: name.clone(),
                    categories: vec![],
                },
            })
            .collect()
    }

    fn state(&self) -> Vec<f64> {
        self.state.clone()
    }

    fn total(&self) -> f64 {
        1000.0
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        self.initial_state[compartment_index] = value;
        self.state[compartment_index] = value;
        Ok(())
    }
}

fn initial_conditions(ids: &[&str]) -> Vec<ModelValue> {
    ids.iter()
        .map(|id| ModelValue {
            value_type: ModelValueType::InitialCondition,
            id: id.to_string(),
        })
        .collect()
}

/// The state after the engine writes the values, or the reason it
/// rejects them; an infeasible state is an error with the plan's reason.
fn written_state(
    mut engine: FrozenEngine,
    ids: &[&str],
    remainder: Option<&str>,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    let plan = engine.plan_values(&initial_conditions(ids), remainder)?;
    let excess = engine.apply_values(&plan, values)?;
    if excess > 0.0 {
        return Err(plan.infeasibility(excess));
    }
    Ok(engine.state())
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} != {expected:?}");
    }
}

#[test]
fn a_bin_value_keeps_the_split_of_its_compartments() {
    // A = 0.65 of 1000 keeps its 300:200 split: 390 and 260.
    let state = written_state(FrozenEngine::two_bins(), &["A", "B"], None, &[0.65, 0.35]).unwrap();

    assert_close(&state, &[390.0, 260.0, 280.0, 70.0]);
}

#[test]
fn remainder_absorbs_the_difference_and_keeps_its_split() {
    // A = 0.65 keeps its 300:200 split; B takes the remaining 350 at 400:100.
    let state = written_state(FrozenEngine::two_bins(), &["A"], Some("B"), &[0.65]).unwrap();

    assert_close(&state, &[390.0, 260.0, 280.0, 70.0]);
}

#[test]
fn remainder_leaves_fixed_compartments_alone() {
    // C stays at 100, so B takes 1000 - 300 - 100 = 600 at 300:100.
    let state = written_state(FrozenEngine::with_fixed_c(), &["A"], Some("B"), &[0.3]).unwrap();

    assert_close(&state, &[180.0, 120.0, 100.0, 450.0, 150.0]);
}

#[test]
fn single_exact_value_is_free_with_a_remainder() {
    let state = written_state(FrozenEngine::two_bins(), &["A_cat1"], Some("B"), &[0.1]).unwrap();

    assert_close(&state, &[300.0, 100.0, 480.0, 120.0]);
}

#[test]
fn without_remainder_values_are_applied_as_given() {
    let state = written_state(FrozenEngine::two_bins(), &["A_cat0"], None, &[0.1]).unwrap();

    assert_close(&state, &[100.0, 200.0, 400.0, 100.0]);
}

#[test]
fn without_remainder_an_excess_over_the_total_is_infeasible() {
    let error = written_state(FrozenEngine::two_bins(), &["A_cat0"], None, &[0.5]).unwrap_err();

    assert!(error.contains("exceed the total"), "{error}");
}

#[test]
fn negative_remainder_is_infeasible() {
    let error =
        written_state(FrozenEngine::with_fixed_c(), &["A"], Some("B"), &[0.95]).unwrap_err();

    assert!(error.contains("remainder 'B'"), "{error}");
}

#[test]
fn remainder_overlapping_a_calibrated_value_is_rejected() {
    let error =
        written_state(FrozenEngine::two_bins(), &["A"], Some("A_cat0"), &[0.5]).unwrap_err();

    assert!(error.contains("overlaps"), "{error}");
}

#[test]
fn category_splits_apply_after_bin_values_whatever_the_order() {
    // A = 0.6 first (360 / 240), then cat1 = 0.1 splits every pair, and the
    // remainder B takes 400 at the new 450:50 split.
    let state = written_state(
        FrozenEngine::two_bins(),
        &["cat1", "A"],
        Some("B"),
        &[0.1, 0.6],
    )
    .unwrap();

    assert_close(&state, &[540.0, 60.0, 360.0, 40.0]);
}

#[test]
fn a_category_value_keeps_the_pair_totals() {
    // A_cat0 and A_cat1 total 500; cat1 = 0.1 makes them 450 and 50.
    let state = written_state(FrozenEngine::two_bins(), &["cat1"], None, &[0.1]).unwrap();

    assert_close(&state, &[450.0, 50.0, 450.0, 50.0]);
}

#[test]
fn either_category_of_a_binary_stratification_is_a_target() {
    let state = written_state(FrozenEngine::two_bins(), &["cat0"], None, &[0.2]).unwrap();

    assert_close(&state, &[100.0, 400.0, 100.0, 400.0]);
}

#[test]
fn category_names_come_from_the_labels() {
    let state = written_state(
        FrozenEngine::with_other_categories(),
        &["seg0"],
        None,
        &[0.8],
    )
    .unwrap();

    assert_close(&state, &[400.0, 100.0, 400.0, 100.0]);
}

#[test]
fn an_expanded_compartment_is_a_target() {
    let state = written_state(FrozenEngine::two_bins(), &["A_cat0"], None, &[0.2]).unwrap();

    assert_close(&state, &[200.0, 200.0, 400.0, 100.0]);
}

#[test]
fn an_unknown_id_names_the_compartments() {
    let error = written_state(FrozenEngine::two_bins(), &["X"], None, &[0.2]).unwrap_err();

    assert!(error.contains("'X'") && error.contains("A_cat0"), "{error}");
}

#[test]
fn the_plan_covers_what_its_values_and_remainder_set() {
    let engine = FrozenEngine::with_fixed_c();
    let plan = engine
        .plan_values(&initial_conditions(&["A", "cat1"]), Some("B"))
        .unwrap();

    // A covers 0 and 1, the remainder B covers 3 and 4; a category split and
    // the fixed C cover nothing.
    assert_eq!(plan.covered_compartments(), vec![0, 1, 3, 4]);
}
