//! How a compartmental engine resolves and writes the initial conditions that
//! calibration sets (`InitialConditionPlan`), through `plan_values` and
//! `apply_values` of `SimulationEngine`.

use commol_core::{
    Model, ModelValue, ModelValueType, SimulationEngine, SimulationError, ValuePlan,
};
use commol_model::{CompartmentalEngine, CompiledModel, OutputLabels};

/// A compartmental engine that holds its state and does not move it: one bin
/// of the compiled model per compartment, or the compartments of a
/// stratified model ([`engine_of`]), which keep the labels of the model. The
/// labels of a model of bins read a compartment name as `<bin>` or
/// `<bin>_<category>` of one stratification `group`, and the total is the sum
/// of the state it starts with.
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
        let labels = self.model.output_labels();
        if labels.iter().any(|label| !label.categories.is_empty()) {
            return labels.to_vec();
        }
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

/// Bins A and B over `group` (`g1`, `g2`) and `type` (`t1`, `t2`), and the
/// stratifications of `more`, with the bin amounts `amounts` and the
/// stratification fractions `fractions`, as JSON.
fn stratified_model(more: &str, amounts: &str, fractions: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[],"engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "stratifications":[{{"id":"group","categories":["g1","g2"]}},
        {{"id":"type","categories":["t1","t2"]}}{more}],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"0.1"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{amounts},
        "stratification_fractions":{fractions}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

/// `type` split by `group`: `g1` 0.25 / 0.75 and `g2` 0.5 / 0.5.
const TYPE_BY_GROUP: &str = r#"{"by":"group","g1":{"t1":0.25,"t2":0.75},"g2":{"t1":0.5,"t2":0.5}}"#;

/// The fractions of [`stratified_model`]: `group` 0.4 / 0.6, and `type`.
fn fractions_with(type_fractions: &str) -> String {
    format!(r#"{{"group":{{"g1":0.4,"g2":0.6}},"type":{type_fractions}}}"#)
}

fn initial_state_error(model: &Model) -> String {
    let error =
        commol_model::initial_state(&model.process, model.aggregate().unwrap()).unwrap_err();
    assert_eq!(commol_model::check_model(model).unwrap_err(), error);
    error
}

/// An engine of a stratified model that starts from its initial state.
fn engine_of(model: &Model) -> FrozenEngine {
    let model = CompiledModel::try_from_model(model).unwrap();
    let state = model.initial_state()[..model.num_compartments()].to_vec();
    FrozenEngine { model, state }
}

#[test]
fn a_stratification_splits_each_category_of_an_earlier_one() {
    let model = stratified_model(
        "",
        r#"{"A":1000.0,"B":0.0}"#,
        &fractions_with(TYPE_BY_GROUP),
    );

    let state = commol_model::initial_state(&model.process, model.aggregate().unwrap()).unwrap();

    let amounts: Vec<f64> = state.iter().map(|c| c.amount).collect();
    assert_close(&amounts, &[100.0, 300.0, 300.0, 300.0, 0.0, 0.0, 0.0, 0.0]);
    let fractions: Vec<f64> = state.iter().flat_map(|c| c.fractions.clone()).collect();
    assert_close(&fractions[8..], &[0.4, 0.25, 0.4, 0.75, 0.6, 0.5, 0.6, 0.5]);
}

#[test]
fn a_split_by_category_names_an_earlier_stratification() {
    let conditional = r#",{"id":"level","categories":["l1","l2"],
        "conditions":[{"stratification":"group","category":"g1"}]}"#;
    let cases = [
        (
            "",
            fractions_with(r#"{"by":"level","g1":{"t1":1.0,"t2":0.0}}"#),
            "Stratification 'type': its fractions differ by 'level', which is not a \
             stratification declared before it. Declared before it: [\"group\"]",
        ),
        (
            "",
            r#"{"group":{"by":"type","t1":{"g1":1.0,"g2":0.0},"t2":{"g1":1.0,"g2":0.0}},
            "type":{"t1":0.5,"t2":0.5}}"#
                .to_string(),
            "Stratification 'group': its fractions differ by 'type', which is not a \
             stratification declared before it. Declared before it: []",
        ),
        (
            "",
            fractions_with(r#"{"by":"group","g1":{"t1":0.5,"t2":0.5},"g3":{"t1":1.0,"t2":0.0}}"#),
            "Stratification 'type': its fractions by 'group' must name exactly the \
             categories of 'group' that its compartments have. Missing categories: \
             [\"g2\"], Extra categories: [\"g3\"].",
        ),
        (
            "",
            fractions_with(r#"{"by":"group","g1":{"t1":1.0},"g2":{"t1":0.5,"t2":0.5}}"#),
            "The fractions of stratification 'type' within category 'g1' of 'group' must \
             name exactly its categories. Missing categories: [\"t2\"], Extra categories: [].",
        ),
        (
            conditional,
            r#"{"group":{"g1":0.4,"g2":0.6},"type":{"t1":0.5,"t2":0.5},
            "level":{"by":"type","t1":{"l1":0.5,"l2":0.5},"t2":{"l1":1.0,"l2":0.0}}}"#
                .to_string(),
            "",
        ),
        (
            r#",{"id":"level","categories":["l1","l2"]}"#,
            r#"{"group":{"g1":0.4,"g2":0.6},"type":{"t1":0.5,"t2":0.5},
            "level":{"l1":0.5,"l2":0.5}}"#
                .to_string(),
            "",
        ),
    ];
    for (more, fractions, message) in cases {
        let model = stratified_model(more, r#"{"A":1000.0,"B":0.0}"#, &fractions);
        if message.is_empty() {
            assert!(commol_model::check_model(&model).is_ok());
        } else {
            assert_eq!(initial_state_error(&model), message);
        }
    }
}

#[test]
fn a_split_by_category_needs_its_category_in_every_compartment() {
    // `type` applies only to `g1`, so the compartments of `g2` have no `type`;
    // `level` applies to every compartment and cannot differ by `type`.
    let model = stratified_model(
        r#",{"id":"level","categories":["l1","l2"]}"#,
        r#"{"A":1000.0,"B":0.0}"#,
        r#"{"group":{"g1":0.4,"g2":0.6},"type":{"t1":0.5,"t2":0.5},
        "level":{"by":"type","t1":{"l1":0.5,"l2":0.5},"t2":{"l1":1.0,"l2":0.0}}}"#,
    );
    let mut model = model;
    model.process.stratifications[1].conditions =
        Some(serde_json::from_str(r#"[{"stratification":"group","category":"g1"}]"#).unwrap());

    assert_eq!(
        initial_state_error(&model),
        "Stratification 'level': its fractions differ by 'type', but the compartment \
         'A_g2_l1' has a category of 'level' and none of 'type'"
    );
}

#[test]
fn a_split_by_category_names_no_category_it_does_not_reach() {
    // `type` applies only to `g2`, so no compartment of `type` has `g1`.
    let mut model = stratified_model(
        "",
        r#"{"A":1000.0,"B":0.0}"#,
        &fractions_with(TYPE_BY_GROUP),
    );
    model.process.stratifications[1].conditions =
        Some(serde_json::from_str(r#"[{"stratification":"group","categories":["g2"]}]"#).unwrap());

    assert_eq!(
        initial_state_error(&model),
        "Stratification 'type': its fractions by 'group' must name exactly the categories \
         of 'group' that its compartments have. Missing categories: [], Extra categories: \
         [\"g1\"]."
    );
}

#[test]
fn the_fractions_of_each_category_sum_to_one() {
    let model = stratified_model(
        "",
        r#"{"A":1000.0,"B":0.0}"#,
        &fractions_with(r#"{"by":"group","g1":{"t1":0.25,"t2":0.75},"g2":{"t1":0.5,"t2":0.4}}"#),
    );

    assert_eq!(
        initial_state_error(&model),
        "Stratification 'type' within category 'g2' of 'group': the fractions of its \
         categories sum to 1 within 1e-6, got 0.9"
    );
}

#[test]
fn every_stratification_has_its_fractions() {
    let model = stratified_model(
        "",
        r#"{"A":1000.0,"B":0.0}"#,
        r#"{"group":{"g1":0.4,"g2":0.6}}"#,
    );

    assert_eq!(
        initial_state_error(&model),
        "Initial stratification fractions keys must exactly match stratification ids. \
         Missing ids: [\"type\"], Extra ids: []."
    );
}

#[test]
fn a_bin_left_for_calibration_splits_a_value_by_its_fractions() {
    let model = stratified_model("", r#"{"A":null,"B":0.0}"#, &fractions_with(TYPE_BY_GROUP));

    let state = written_state(engine_of(&model), &["A"], None, &[2000.0]).unwrap();

    assert_close(&state, &[200.0, 600.0, 600.0, 600.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn a_remainder_of_empty_compartments_splits_by_its_fractions() {
    // A = 1000 starts, so B takes 1000 - 600 = 400 as `type` splits it.
    let model = stratified_model(
        "",
        r#"{"A":1000.0,"B":0.0}"#,
        &fractions_with(TYPE_BY_GROUP),
    );

    let state = written_state(engine_of(&model), &["A"], Some("B"), &[600.0]).unwrap();

    assert_close(
        &state,
        &[60.0, 180.0, 180.0, 180.0, 40.0, 120.0, 120.0, 120.0],
    );
}

#[test]
fn a_category_of_the_stratification_a_split_differs_by_keeps_the_split() {
    // g1 = 0.5 of A = 1000 is 500, which `type` splits 0.25 / 0.75; g2 keeps
    // its 0.5 / 0.5.
    let model = stratified_model(
        "",
        r#"{"A":1000.0,"B":0.0}"#,
        &fractions_with(TYPE_BY_GROUP),
    );

    let state = written_state(engine_of(&model), &["g1"], None, &[0.5]).unwrap();

    assert_close(&state, &[125.0, 375.0, 250.0, 250.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn a_category_of_a_split_by_category_takes_its_value_in_every_split() {
    let model = stratified_model(
        "",
        r#"{"A":1000.0,"B":0.0}"#,
        &fractions_with(TYPE_BY_GROUP),
    );

    let state = written_state(engine_of(&model), &["t1"], None, &[0.2]).unwrap();

    assert_close(&state, &[80.0, 320.0, 120.0, 480.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn a_category_keeps_a_chain_of_splits_that_differ_by_it() {
    // `level` differs by `type`, which differs by `group`: g1 = 0.5 keeps
    // both splits on each side.
    let level = r#",{"id":"level","categories":["l1","l2"]}"#;
    let fractions = format!(
        r#"{{"group":{{"g1":0.4,"g2":0.6}},"type":{TYPE_BY_GROUP},
        "level":{{"by":"type","t1":{{"l1":1.0,"l2":0.0}},"t2":{{"l1":0.0,"l2":1.0}}}}}}"#
    );
    let model = stratified_model(level, r#"{"A":1000.0,"B":0.0}"#, &fractions);

    let state = written_state(engine_of(&model), &["g1"], None, &[0.5]).unwrap();

    assert_close(
        &state[..8],
        &[125.0, 0.0, 0.0, 375.0, 250.0, 0.0, 0.0, 250.0],
    );
}

#[test]
fn a_side_that_starts_empty_keeps_its_declared_split() {
    // g1 starts with nothing, so its 500 takes the split `type` declares in
    // g1, 0.25 / 0.75.
    let fractions = format!(r#"{{"group":{{"g1":0.0,"g2":1.0}},"type":{TYPE_BY_GROUP}}}"#);
    let model = stratified_model("", r#"{"A":1000.0,"B":0.0}"#, &fractions);

    let state = written_state(engine_of(&model), &["g1"], None, &[0.5]).unwrap();

    assert_close(&state, &[125.0, 375.0, 250.0, 250.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn a_prefix_that_starts_empty_keeps_its_declared_split() {
    let fractions = format!(r#"{{"group":{{"g1":0.0,"g2":1.0}},"type":{TYPE_BY_GROUP}}}"#);
    let model = stratified_model("", r#"{"A":1000.0,"B":0.0}"#, &fractions);

    let state = written_state(engine_of(&model), &["A_g1"], None, &[100.0]).unwrap();

    assert_close(&state, &[25.0, 75.0, 500.0, 500.0, 0.0, 0.0, 0.0, 0.0]);
}
