//! Where an engine in continuous time stops (impulses, the values a rate
//! compares `t` with, the points of a series, every integer when a rate reads
//! `step`), the checks of impulse schedules, and the flows that drain a
//! compartment.

use commol_core::Model;
use commol_model::{CompiledModel, as_whole};

/// A → B and A → C at the rates `ab` and `ac`, from A = 800 and B = 200,
/// with the parameters `k = 0.3`, `z = 0` and `s`, a series of the given
/// mode with the points (1, 10), (5, 30) and (9, 50); `extra` is added to
/// the transition A → B.
fn model(ab: &str, ac: &str, mode: &str, extra: &str) -> Model {
    let json = format!(
        r#"{{"name":"m",
        "parameters":[{{"id":"k","value":0.3}},{{"id":"z","value":0.0}},
        {{"id":"s","value":{{"data":[[1,10],[5,30],[9,50]],"mode":"{mode}"}}}}],
        "engine":{{"type":"DifferentialEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}},
        {{"id":"C","name":"C"}}],"accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"{ab}"{extra}}},
        {{"id":"ac","source":["A"],"target":["C"],"rate":"{ac}"}}]}},
        "structure":{{"kind":"aggregate",
        "bin_amounts":{{"A":800.0,"B":200.0,"C":0.0}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn compiled(model: &Model) -> CompiledModel {
    CompiledModel::try_from_model(model).unwrap()
}

#[test]
fn a_value_a_rate_compares_t_with_is_a_breakpoint_inside_the_range() {
    // The series adds its points 1, 5 and 9.
    let mut model = compiled(&model("if(t > 6, k, 0)", "k", "step_function", ""));
    assert_eq!(model.breakpoints(2.0, 4.0).unwrap(), [4.0]);
    assert_eq!(model.breakpoints(2.0, 8.0).unwrap(), [5.0, 6.0, 8.0]);
}

#[test]
fn only_a_fixed_value_compared_with_t_is_a_breakpoint() {
    // A / 1000 + 0.5 reads a compartment, so it is no breakpoint.
    for rate in [
        "if(t > A / 1000 + 0.5, k, 0)",
        "if(A / 1000 + 0.5 < t, k, 0)",
    ] {
        let mut model = compiled(&model(rate, "k", "step_function", ""));
        assert_eq!(model.breakpoints(0.0, 0.9), Ok(vec![0.9]), "{rate}");
    }
}

#[test]
fn a_threshold_of_fixed_values_through_functions_is_a_breakpoint() {
    for (rate, point) in [
        ("if(t > max(k, 2), k, 0)", 2.0),
        ("if(t > -k + 3, k, 0)", 2.7),
        ("if(t > if(k > 0, 3, 4), k, 0)", 3.0),
    ] {
        let mut model = compiled(&model(rate, "k", "step_function", ""));
        assert_eq!(model.breakpoints(1.5, 4.5), Ok(vec![point, 4.5]), "{rate}");
    }
}

#[test]
fn a_rate_of_step_stops_at_every_integer_after_the_start() {
    let mut model = compiled(&model("k * step", "k", "step_function", ""));
    assert_eq!(model.breakpoints(2.5, 5.0).unwrap(), [3.0, 4.0, 5.0]);
}

#[test]
fn the_points_of_a_series_read_at_t_are_breakpoints_inside_the_range() {
    let mut model = compiled(&model("s", "k", "step_function", ""));
    assert_eq!(model.breakpoints(2.0, 6.0).unwrap(), [5.0, 6.0]);
}

#[test]
fn impulses_act_at_their_positions_inside_the_range() {
    let at = compiled(&model(
        "k",
        "k",
        "step_function",
        r#","impulse_at":[1,5,9]"#,
    ));
    assert_eq!(at.impulse_positions(2.0, 6.0), [5.0]);
    let every = compiled(&model(
        "k",
        "k",
        "step_function",
        r#","impulse_every":[2,0.5]"#,
    ));
    assert_eq!(every.impulse_positions(4.0, 8.5), [4.5, 6.5, 8.5]);
    assert_eq!(every.impulse_positions(6.0, 9.0), [6.5, 8.5]);
}

#[test]
fn an_impulse_position_is_finite_and_not_negative() {
    for (positions, accepted) in [
        (vec![0.0, 2.0], true),
        (vec![-1.0], false),
        (vec![f64::INFINITY], false),
    ] {
        let mut model = model("k", "k", "step_function", r#","impulse_at":[1]"#);
        model.process.transitions[0].impulse_at = Some(positions.clone());
        let result = CompiledModel::try_from_model(&model);
        assert_eq!(result.is_ok(), accepted, "{positions:?}");
        if !accepted {
            let error = result.err().unwrap();
            assert!(error.contains("finite and not negative"), "{error}");
        }
    }
}

#[test]
fn an_impulse_has_positions_or_a_period_not_both() {
    let mut model = model("k", "k", "step_function", r#","impulse_at":[1]"#);
    model.process.transitions[0].impulse_every = Some((2.0, 0.0));

    let error = CompiledModel::try_from_model(&model).err().unwrap();

    assert_eq!(
        error,
        "Transition 'ab' sets both impulse_at and impulse_every; set one"
    );
}

#[test]
fn an_impulse_period_is_positive_and_its_offset_finite() {
    for (period, offset) in [(0.0, 0.0), (f64::INFINITY, 0.0), (1.0, f64::INFINITY)] {
        let mut model = model("k", "k", "step_function", r#","impulse_every":[1,0]"#);
        model.process.transitions[0].impulse_every = Some((period, offset));
        let error = CompiledModel::try_from_model(&model).err();
        assert!(error.is_some(), "period {period}, offset {offset}");
    }
}

#[test]
fn a_flow_of_zero_does_not_drain() {
    let mut model = compiled(&model("k", "z", "step_function", ""));
    let state = model.initial_state().to_vec();
    let outflows = model.outflows(0.0, 0.0, &state, 0).unwrap();
    assert_eq!(outflows, [("ab".to_string(), 0.3 * 800.0)]);
}

#[test]
fn a_linear_series_reads_time_and_interpolates_inside_its_points() {
    let mut linear = compiled(&model("s", "z", "linear", ""));
    assert!(linear.reads_time());
    let state = linear.initial_state().to_vec();
    let mut rates = vec![0.0; 2];
    let mut rate_at = |t: f64| {
        linear.flow_rates(0.0, t, &state, &mut rates).unwrap();
        rates[0] / 800.0
    };
    assert_eq!(rate_at(0.5), 0.0);
    assert_eq!(rate_at(1.0), 10.0);
    assert_eq!(rate_at(3.0), 20.0);
    assert_eq!(rate_at(9.0), 50.0);
    assert!(!compiled(&model("s", "z", "step_function", "")).reads_time());
}

#[test]
fn a_whole_number_is_whole_within_a_relative_tolerance() {
    assert_eq!(as_whole(1e6 + 1e-4), Some(1e6));
    assert_eq!(as_whole(1.0 + 1e-6), None);
}

/// `model` with `k` left without a value, as before calibration.
fn without_k(ab: &str, extra: &str) -> CompiledModel {
    let mut model = model(ab, "z", "step_function", extra);
    model.parameters[0].value = None;
    compiled(&model)
}

#[test]
fn a_value_that_reads_a_parameter_without_a_value_fails_where_it_is_read() {
    let state = vec![800.0, 200.0, 0.0];
    let mut rates = vec![0.0; 2];

    let program = without_k("k * A + 1", "")
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap_err();
    let threshold = without_k("if(t > k, 1, 0)", "")
        .breakpoints(0.0, 2.0)
        .unwrap_err();
    let mut impulse = without_k("k", r#","impulse_at":[1]"#);
    let rate = impulse
        .impulse_rates(1.0, &state, &mut Vec::new())
        .unwrap_err();
    let applied = impulse.apply_impulses(1.0, &mut state.clone()).unwrap_err();

    for error in [&program, &rate] {
        assert!(
            error.contains("Parameter index '") && error.ends_with("' not found"),
            "{error}"
        );
    }
    assert!(
        threshold.starts_with("Failed to evaluate the time threshold 'k'"),
        "{threshold}"
    );
    assert_eq!(applied, rate);
}

#[test]
fn a_formula_parameter_that_reads_a_parameter_without_a_value_fails() {
    let mut model = model("f", "z", "step_function", "");
    model.parameters[0].value = None;
    model
        .parameters
        .push(serde_json::from_str(r#"{"id":"f","value":"k * 2"}"#).expect("valid parameter"));
    let mut compiled = compiled(&model);
    let state = compiled.initial_state().to_vec();

    let error = compiled
        .transition_rates(0.0, 0.0, &state, &mut [0.0; 2])
        .unwrap_err();

    assert!(
        error.starts_with("Failed to evaluate formula parameter 'f': "),
        "{error}"
    );
}

#[test]
fn only_the_impulses_of_a_position_act_and_an_absolute_one_moves_its_rate() {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.5}],
        "engine":{"type":"DifferentialEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"},
        {"id":"C","name":"C"}],"accumulators":[],"stratifications":[],
        "transitions":[
        {"id":"ab","source":["A"],"target":["B"],"rate":"k","impulse_at":[1]},
        {"id":"ac","source":["A"],"target":["C"],"rate":"k * A / 4","impulse_at":[2]}]},
        "structure":{"kind":"aggregate",
        "bin_amounts":{"A":800.0,"B":200.0,"C":0.0}}}"#;
    let mut model = compiled(&serde_json::from_str(json).expect("valid model JSON"));
    let mut state = model.initial_state().to_vec();

    let mut rates = Vec::new();
    model.impulse_rates(1.0, &state, &mut rates).unwrap();
    assert_eq!(rates, [(0, 0.5)]);
    assert!(model.apply_impulses(1.0, &mut state).unwrap());
    assert_eq!(state, [400.0, 600.0, 0.0]);
    assert!(model.apply_impulses(2.0, &mut state).unwrap());
    assert_eq!(state, [350.0, 600.0, 50.0]);
}

#[test]
fn a_flow_that_does_not_take_from_a_compartment_does_not_drain_it() {
    let mut model = compiled(&model("k", "z", "step_function", ""));
    let state = model.initial_state().to_vec();

    assert_eq!(model.outflows(0.0, 0.0, &state, 1).unwrap(), []);
}
