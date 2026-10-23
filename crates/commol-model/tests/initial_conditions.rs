//! How a compartmental engine resolves and writes the initial conditions that
//! calibration sets (`InitialConditionPlan`), through `plan_values` and
//! `apply_values` of `SimulationEngine`.

use commol_core::{
    Model, ModelValue, ModelValueType, SimulationEngine, SimulationError, ValuePlan,
};
use commol_model::{CompartmentalEngine, CompiledModel, OutputLabels};

/// A compartmental engine that holds its state and does not move it: one bin
/// of the compiled model per compartment. Compartment names are `<bin>` or
/// `<bin>_<category>` of one stratification `group`, which its labels read,
/// and the total is the sum of the state it starts with.
#[derive(Clone)]
struct FrozenEngine {
    model: CompiledModel,
    state: Vec<f64>,
}

impl FrozenEngine {
    fn with(compartments: &[&str], state: &[f64]) -> Self {
        let bins: Vec<String> = compartments
            .iter()
            .map(|id| format!(r#"{{"id":"{id}","name":"{id}"}}"#))
            .collect();
        let amounts: Vec<String> = compartments
            .iter()
            .zip(state)
            .map(|(id, amount)| format!(r#""{id}":{amount:?}"#))
            .collect();
        let json = format!(
            r#"{{"name":"frozen","parameters":[],"engine":{{"type":"DifferenceEquations"}},
            "process":{{"bins":[{}],"accumulators":[],"stratifications":[],"transitions":[]}},
            "structure":{{"kind":"aggregate","bin_amounts":{{{}}}}}}}"#,
            bins.join(","),
            amounts.join(",")
        );
        let model: Model = serde_json::from_str(&json).expect("valid model JSON");
        Self {
            model: CompiledModel::try_from_model(&model).unwrap(),
            state: state.to_vec(),
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
    commol_model::delegate_to_compiled_model!();

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

    fn reset(&mut self) {
        self.state = self.model.initial_state().to_vec();
    }

    fn current_position(&self) -> f64 {
        0.0
    }
}

impl CompartmentalEngine for FrozenEngine {
    fn model(&self) -> &CompiledModel {
        &self.model
    }

    fn model_mut(&mut self) -> &mut CompiledModel {
        &mut self.model
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        self.compartments()
            .into_iter()
            .map(|name| match name.split_once('_') {
                Some((bin, category)) => OutputLabels {
                    source: bin.to_string(),
                    categories: vec![("group".to_string(), category.to_string())],
                },
                None => OutputLabels {
                    source: name,
                    categories: vec![],
                },
            })
            .collect()
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        self.model.set_initial_condition(compartment_index, value)?;
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
    Ok(engine.state)
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} != {expected:?}");
    }
}

#[test]
fn a_bin_value_keeps_the_split_of_its_compartments() {
    // A = 650 keeps its 300:200 split: 390 and 260.
    let state =
        written_state(FrozenEngine::two_bins(), &["A", "B"], None, &[650.0, 350.0]).unwrap();

    assert_close(&state, &[390.0, 260.0, 280.0, 70.0]);
}

#[test]
fn a_bin_and_the_remainder_keep_the_split_of_the_initial_state_after_a_run() {
    // The run moved A to 50:450 and B to 100:400; A = 100 keeps the 300:200
    // it starts with, and B takes the remaining 900 at its 400:100.
    let mut engine = FrozenEngine::two_bins();
    engine.state = vec![50.0, 450.0, 100.0, 400.0];
    let plan = engine
        .plan_values(&initial_conditions(&["A"]), Some("B"))
        .unwrap();

    assert_eq!(engine.apply_values(&plan, &[100.0]).unwrap(), 0.0);

    assert_close(&engine.initial_state(), &[60.0, 40.0, 720.0, 180.0]);
}

#[test]
fn remainder_absorbs_the_difference_and_keeps_its_split() {
    // A = 650 keeps its 300:200 split; B takes the remaining 350 at 400:100.
    let state = written_state(FrozenEngine::two_bins(), &["A"], Some("B"), &[650.0]).unwrap();

    assert_close(&state, &[390.0, 260.0, 280.0, 70.0]);
}

#[test]
fn remainder_leaves_fixed_compartments_alone() {
    // C stays at 100, so B takes 1000 - 300 - 100 = 600 at 300:100.
    let state = written_state(FrozenEngine::with_fixed_c(), &["A"], Some("B"), &[300.0]).unwrap();

    assert_close(&state, &[180.0, 120.0, 100.0, 450.0, 150.0]);
}

#[test]
fn single_exact_value_is_free_with_a_remainder() {
    let state = written_state(FrozenEngine::two_bins(), &["A_cat1"], Some("B"), &[100.0]).unwrap();

    assert_close(&state, &[300.0, 100.0, 480.0, 120.0]);
}

#[test]
fn without_remainder_values_are_applied_as_given() {
    let state = written_state(FrozenEngine::two_bins(), &["A_cat0"], None, &[100.0]).unwrap();

    assert_close(&state, &[100.0, 200.0, 400.0, 100.0]);
}

#[test]
fn without_remainder_any_amount_is_feasible() {
    let state = written_state(FrozenEngine::two_bins(), &["A_cat0"], None, &[5000.0]).unwrap();

    assert_close(&state, &[5000.0, 200.0, 400.0, 100.0]);
}

#[test]
fn negative_remainder_is_infeasible() {
    let error =
        written_state(FrozenEngine::with_fixed_c(), &["A"], Some("B"), &[950.0]).unwrap_err();

    assert!(error.contains("remainder 'B'"), "{error}");
}

#[test]
fn an_excess_over_the_remainder_is_its_share_of_the_amounts_set() {
    // A and B keep 900, so A = 950 leaves B at −50, 50 / 950 of A.
    let error =
        written_state(FrozenEngine::with_fixed_c(), &["A"], Some("B"), &[950.0]).unwrap_err();

    assert!(error.contains("'B' keeps by 0.052632 of them"), "{error}");
}

#[test]
fn the_remainder_keeps_the_declared_sum_of_what_it_covers() {
    // A and B hold a count and C a concentration: B = 12 takes 2 from A,
    // and C, which nothing covers, keeps its amount.
    let engine = FrozenEngine::with(&["A", "B", "C"], &[990.0, 10.0, 1e5]);
    let state = written_state(engine, &["B"], Some("A"), &[12.0]).unwrap();

    assert_close(&state, &[988.0, 12.0, 1e5]);
}

#[test]
fn a_remainder_of_bins_that_start_at_zero_cannot_give_an_amount() {
    let engine = FrozenEngine::with(&["A", "B", "C"], &[0.0, 0.0, 5.0]);
    let error = written_state(engine, &["B"], Some("A"), &[3.0]).unwrap_err();

    assert!(error.contains("'A' keeps by 1.000000 of them"), "{error}");
}

#[test]
fn a_bin_of_empty_compartments_splits_its_value_evenly() {
    let engine = FrozenEngine::with(
        &["A_cat0", "A_cat1", "B_cat0", "B_cat1"],
        &[0.0, 0.0, 400.0, 300.0],
    );
    let state = written_state(engine, &["A"], None, &[300.0]).unwrap();

    assert_close(&state, &[150.0, 150.0, 400.0, 300.0]);
}

#[test]
fn a_remainder_of_empty_compartments_splits_evenly() {
    let engine = FrozenEngine::with(
        &["A_cat0", "A_cat1", "B_cat0", "B_cat1"],
        &[500.0, 500.0, 0.0, 0.0],
    );
    let state = written_state(engine, &["A"], Some("B"), &[600.0]).unwrap();

    assert_close(&state, &[300.0, 300.0, 200.0, 200.0]);
}

#[test]
fn remainder_overlapping_a_calibrated_value_is_rejected() {
    let error =
        written_state(FrozenEngine::two_bins(), &["A"], Some("A_cat0"), &[500.0]).unwrap_err();

    assert!(error.contains("overlaps"), "{error}");
}

#[test]
fn category_splits_apply_after_bin_values_whatever_the_order() {
    // A = 600 first (360 / 240), then cat1 = 0.1 splits every pair, and the
    // remainder B takes 400 at the new 450:50 split.
    let state = written_state(
        FrozenEngine::two_bins(),
        &["cat1", "A"],
        Some("B"),
        &[0.1, 600.0],
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
    let state = written_state(FrozenEngine::two_bins(), &["A_cat0"], None, &[200.0]).unwrap();

    assert_close(&state, &[200.0, 200.0, 400.0, 100.0]);
}

#[test]
fn an_unknown_id_names_the_compartments() {
    let error = written_state(FrozenEngine::two_bins(), &["X"], None, &[200.0]).unwrap_err();

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

#[test]
fn a_remainder_is_a_compartment_or_a_bin() {
    let error = written_state(FrozenEngine::two_bins(), &["A"], Some("X"), &[500.0]).unwrap_err();

    assert_eq!(
        error,
        "Initial-condition remainder 'X' is not a compartment or a bin \
         (available: A_cat0, A_cat1, B_cat0, B_cat1)"
    );
}

#[test]
fn a_plan_takes_one_value_per_initial_condition() {
    let error = written_state(FrozenEngine::two_bins(), &["A"], None, &[500.0, 200.0]).unwrap_err();

    assert_eq!(error, "Expected 1 initial-condition values, got 2");
}

#[test]
fn a_qualified_category_that_no_compartment_has_is_rejected() {
    let error = written_state(FrozenEngine::two_bins(), &["group:cat9"], None, &[0.5]).unwrap_err();

    assert_eq!(
        error,
        "Initial condition category 'group:cat9' matches no compartment"
    );
}
