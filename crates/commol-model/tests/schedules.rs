//! Where an engine in continuous time stops (impulses, the values a rate
//! compares `t` with, the points of a series, every integer when a rate reads
//! `step`), the points each stop holds and the impulses that act there, the
//! checks of the series points and impulse positions an engine runs, and the
//! flows that drain a compartment. The rules of an impulse are in
//! `value_rules.rs`.

use commol_core::{EngineType, Model, ModelEngine, ParameterValue, SeriesMode};
use commol_model::{CompiledModel, as_whole, check_model};

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

/// Where the pieces of a run of `model` from `from` to `to` end: at each
/// stop, and at `to` when no stop holds it.
fn ends(model: &mut CompiledModel, from: f64, to: f64) -> Vec<f64> {
    let plan = model.breakpoints(from, to).unwrap();
    let mut ends: Vec<f64> = plan.stops.iter().map(|stop| stop.at).collect();
    if ends.last() != Some(&plan.end) {
        ends.push(plan.end);
    }
    ends
}

/// The stops of a run of `model` from `from` to `to` where impulses act,
/// with the impulses of each.
fn impulses(model: &mut CompiledModel, from: f64, to: f64) -> Vec<(f64, Vec<usize>)> {
    let plan = model.breakpoints(from, to).unwrap();
    plan.stops
        .into_iter()
        .filter(|stop| !stop.impulses.is_empty())
        .map(|stop| (stop.at, stop.impulses))
        .collect()
}

/// The state after a run of `model` from `from` to `to` that applies the
/// impulses of every stop, and how many stops had impulses.
fn after_impulses(model: &mut CompiledModel, from: f64, to: f64) -> (Vec<f64>, usize) {
    let mut state = model.initial_state().to_vec();
    let mut acted = 0;
    for stop in model.breakpoints(from, to).unwrap().stops {
        let applied = model.apply_impulses(stop.at, &stop.impulses, &mut state);
        acted += usize::from(applied.unwrap());
    }
    (state, acted)
}

#[test]
fn a_value_a_rate_compares_t_with_is_a_breakpoint_inside_the_range() {
    // The series adds its points 1, 5 and 9.
    let mut model = compiled(&model("if(t > 6, k, 0)", "k", "step_function", ""));
    assert_eq!(ends(&mut model, 2.0, 4.0), [4.0]);
    assert_eq!(ends(&mut model, 2.0, 8.0), [5.0, 6.0, 8.0]);
}

#[test]
fn only_a_fixed_value_compared_with_t_is_a_breakpoint() {
    // A / 1000 + 0.5 reads a compartment, so it is no breakpoint.
    for rate in [
        "if(t > A / 1000 + 0.5, k, 0)",
        "if(A / 1000 + 0.5 < t, k, 0)",
    ] {
        let mut model = compiled(&model(rate, "k", "step_function", ""));
        assert_eq!(ends(&mut model, 0.0, 0.9), [0.9], "{rate}");
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
        assert_eq!(ends(&mut model, 1.5, 4.5), [point, 4.5], "{rate}");
    }
}

#[test]
fn a_rate_of_step_stops_at_every_integer_after_the_start() {
    let mut model = compiled(&model("k * step", "k", "step_function", ""));
    assert_eq!(ends(&mut model, 2.5, 5.0), [3.0, 4.0, 5.0]);
}

/// Positions on either side of 3 that `same_position` calls 3: the float
/// just below it (`0.3 / 0.1`), the float just above it, and 1e-9 from it.
fn next_to_three() -> [f64; 4] {
    [0.3 / 0.1, 3.0 - 1e-9, 3.0_f64.next_up(), 3.0 + 1e-9]
}

#[test]
fn a_point_next_to_a_step_is_in_the_stop_of_that_step() {
    for on in next_to_three() {
        let rate = format!("if(t >= {on:?}, k, 0) + step");
        let mut model = compiled(&model(&rate, "k", "step_function", ""));
        let plan = model.breakpoints(2.0, 5.0).unwrap();
        assert_eq!((plan.stops[0].at, plan.stops[0].cut), (3.0, on.min(3.0)));
        assert_eq!(ends(&mut model, 2.0, 5.0), [3.0, 4.0, 5.0], "{on:?}");
        // A run that ends next to 3 ends at the stop 3, and the next run
        // starts there, with the point behind it.
        assert_eq!(model.breakpoints(2.0, on).unwrap().end, 3.0, "{on:?}");
        assert_eq!(ends(&mut model, 3.0, 5.0), [4.0, 5.0], "{on:?}");
    }
}

#[test]
fn a_position_next_to_a_step_goes_to_that_step_before_another_stop() {
    // 2.999999996 is a stop of its own, as it is not the same position as 3.
    // 2.9999999985 is the same position as both, and it goes to the step 3,
    // where the impulse acts. So does a run that starts at 2.999999996.
    let rate = "if(t >= 2.999999996, k, 0) + step";
    let mut model = compiled(&model(rate, "k", "step_function", r#","impulse_at":[3]"#));
    let next = 2.999_999_998_5;
    assert_eq!(ends(&mut model, 2.0, 4.0), [2.999999996, 3.0, 4.0]);
    assert_eq!(model.breakpoints(2.0, 4.0).unwrap().place(next), 3.0);
    for from in [2.0, 2.999999996] {
        let plan = model.breakpoints(from, next).unwrap();
        assert_eq!((plan.end, plan.place(next)), (3.0, 3.0), "{from}");
        assert_eq!(impulses(&mut model, from, next), [(3.0, vec![0])], "{from}");
    }
    assert_eq!(ends(&mut model, 3.0, 4.0), [4.0]);
    // A start that is the same position as 3 holds the step.
    let plan = model.breakpoints(2.9999999995, 3.000000001).unwrap();
    assert_eq!((plan.stops.len(), plan.end), (0, 2.9999999995));
}

#[test]
fn a_start_that_no_stop_holds_leaves_a_position_next_to_it_to_a_stop() {
    // A run ends at 3.5 - 5e-9, which no stop holds. The next run asks for
    // 3.5 - 2.5e-9, the same position as that start and as the stop 3.5: it
    // ends at the stop, as the whole run does.
    let (start, next) = (3.5 - 5e-9, 3.5 - 2.5e-9);
    for rate in ["k", "k * step"] {
        let mut model = compiled(&model(rate, "k", "step_function", r#","impulse_at":[3.5]"#));
        assert_eq!(
            model.breakpoints(2.0, 5.0).unwrap().place(next),
            3.5,
            "{rate}"
        );
        assert_eq!(model.breakpoints(2.0, start).unwrap().end, start, "{rate}");
        let plan = model.breakpoints(start, next).unwrap();
        assert_eq!((plan.end, plan.place(next)), (3.5, 3.5), "{rate}");
        assert_eq!(
            impulses(&mut model, start, next),
            [(3.5, vec![0])],
            "{rate}"
        );
    }
}

#[test]
fn a_point_just_below_0_is_in_the_step_0() {
    // The point rounds to -0.0, which is the step 0.
    let rate = "if(t >= -1e-10, k, 0) + step";
    let mut model = compiled(&model(rate, "k", "step_function", ""));
    let plan = model.breakpoints(-1.5, 2.0).unwrap();
    let cuts: Vec<(f64, f64)> = plan.stops.iter().map(|stop| (stop.at, stop.cut)).collect();
    assert_eq!(cuts, [(-1.0, -1.0), (0.0, -1e-10), (1.0, 1.0), (2.0, 2.0)]);
}

#[test]
fn a_run_from_a_point_has_that_point_as_its_stop() {
    // A run that ended at the point 2.5 of a threshold or of an impulse, or
    // at 5 of a series, starts from that stop, so a position next to it
    // stays there, as in the whole run. Each model has that one point near
    // the start, and no step.
    let impulse = r#","impulse_at":[2.5]"#;
    for (rate, extra, point) in [
        ("k * if(t >= 2.5, 1, 0)", "", 2.5),
        ("k", impulse, 2.5),
        ("s", "", 5.0),
    ] {
        let mut model = compiled(&model(rate, "k", "step_function", extra));
        let next = point + 1e-9;
        let whole = model.breakpoints(point - 0.5, point + 0.5).unwrap();
        assert_eq!(whole.place(next), point, "{rate}");
        let plan = model.breakpoints(point, next).unwrap();
        assert_eq!((plan.end, plan.place(next)), (point, point), "{rate}");
    }
}

#[test]
fn a_run_of_a_model_that_reads_the_step_stays_below_5e8_on_either_side() {
    let mut steps = compiled(&model("k * step", "k", "step_function", ""));
    let error = steps.breakpoints(-6e8, -6e8 + 3.0).unwrap_err();
    let start = "A run cannot reach -600000000: a model that reads `step`";
    assert!(error.starts_with(start), "{error}");
    let mut plain = compiled(&model("k", "k", "step_function", ""));
    assert_eq!(ends(&mut plain, -6e8, -6e8 + 3.0), [-6e8 + 3.0]);
}

#[test]
fn a_point_the_same_as_a_point_of_a_stop_is_in_that_stop() {
    // 2.5 and 2.500000004 are not the same position, and the impulse is the
    // same position as both: it acts at the stop 2.5 alone. A run that ends
    // at 2.500000003, the same position as the impulse and as the next stop,
    // ends at that stop.
    let rate = "k * if(t >= 2.5, 1, 0) + k * if(t >= 2.500000004, 1, 0)";
    let mut model = compiled(&model(
        rate,
        "k",
        "step_function",
        r#","impulse_at":[2.500000002]"#,
    ));
    assert_eq!(impulses(&mut model, 2.0, 4.0), [(2.5, vec![0])]);
    assert_eq!(ends(&mut model, 2.0, 4.0), [2.5, 2.500000004, 4.0]);
    assert_eq!(
        model.breakpoints(2.0, 2.500000003).unwrap().end,
        2.500000004
    );
    assert_eq!(model.breakpoints(2.0, 2.5000000015).unwrap().end, 2.5);
    // A run from the stop 2.5 has the impulse behind it. A position that is
    // the same position as the start and as the next stop stays at the start.
    let plan = model.breakpoints(2.5, 2.500000002).unwrap();
    assert_eq!((plan.stops.len(), plan.end), (0, 2.5));
    assert_eq!(impulses(&mut model, 2.5, 4.0), []);
    assert_eq!(ends(&mut model, 2.5, 4.0), [2.500000004, 4.0]);
    let plan = model.breakpoints(2.0, 2.4).unwrap();
    assert_eq!((plan.stops.len(), plan.end, plan.place(2.4)), (0, 2.4, 2.4));
    // Without the stop 2.500000004, 2.5000000035 is in the start 2.5, which
    // holds 2.500000002.
    let rate = "k * if(t >= 2.5, 1, 0) + k * if(t >= 2.500000002, 1, 0)";
    let mut pair = compiled(&self::model(rate, "k", "step_function", ""));
    assert_eq!(pair.breakpoints(2.5, 2.5000000035).unwrap().end, 2.5);
}

#[test]
fn an_impulse_of_a_schedule_just_below_a_step_reads_that_step() {
    // 90 × 0.7 is 62.99999999999999.
    let rate = "if(step == 63, 0.5, 0)";
    let schedule = r#","impulse_every":[0.7,0]"#;
    let mut model = compiled(&model(rate, "z", "step_function", schedule));
    let (state, acted) = after_impulses(&mut model, 62.5, 63.5);
    assert_eq!((state, acted), (vec![400.0, 600.0, 0.0], 1));
}

#[test]
fn an_impulse_next_to_a_step_reads_that_step_and_acts_once() {
    // This impulse moves half of A at step 3 and nothing at another step.
    for on in next_to_three() {
        let impulse = format!(r#","impulse_at":[{on:?}]"#);
        let rate = "if(step == 3, 0.5, 0)";
        let mut model = compiled(&model(rate, "z", "step_function", &impulse));
        assert_eq!(impulses(&mut model, 2.0, 5.0), [(3.0, vec![0])], "{on}");
        let (state, acted) = after_impulses(&mut model, 2.0, 5.0);
        assert_eq!((state, acted), (vec![400.0, 600.0, 0.0], 1), "{on}");
    }
}

#[test]
fn the_points_of_a_series_read_at_t_are_breakpoints_inside_the_range() {
    let mut model = compiled(&model("s", "k", "step_function", ""));
    assert_eq!(ends(&mut model, 2.0, 6.0), [5.0, 6.0]);
}

#[test]
fn impulses_act_at_their_positions_inside_the_range() {
    let mut at = compiled(&model(
        "k",
        "k",
        "step_function",
        r#","impulse_at":[1,5,9]"#,
    ));
    assert_eq!(impulses(&mut at, 2.0, 6.0), [(5.0, vec![0])]);
    assert_eq!(
        (at.impulses_at(5.0), at.impulses_at(4.0)),
        (vec![0], vec![])
    );
    let mut every = compiled(&model(
        "k",
        "k",
        "step_function",
        r#","impulse_every":[2,0.5]"#,
    ));
    let at_each = |positions: &[f64]| -> Vec<(f64, Vec<usize>)> {
        positions
            .iter()
            .map(|&position| (position, vec![0]))
            .collect()
    };
    assert_eq!(impulses(&mut every, 4.0, 8.5), at_each(&[4.5, 6.5, 8.5]));
    assert_eq!(impulses(&mut every, 6.0, 9.0), at_each(&[6.5, 8.5]));
}

/// `model` with the engine `engine`.
fn on(engine: EngineType, mut model: Model) -> Model {
    model.engine = ModelEngine::new(engine);
    model
}

#[test]
fn an_engine_that_steps_at_integers_rejects_a_position_between_steps_at_the_check() {
    for engine in [EngineType::DifferenceEquations, EngineType::StochasticSteps] {
        let stepper = format!("the {} engine", engine.name());
        for (extra, message) in [
            (
                r#","impulse_at":[3,1.5,2.5]"#,
                "Impulse transition 'ab' acts at 1.5, which is not an integer step",
            ),
            (
                r#","impulse_every":[2,0.5]"#,
                "Impulse transition 'ab' acts at 0.5, which is not an integer step",
            ),
            (
                r#","impulse_every":[1.5,0]"#,
                "Impulse transition 'ab' acts at 1.5, which is not an integer step",
            ),
        ] {
            let error =
                check_model(&on(engine, model("k", "k", "linear", extra))).expect_err(extra);
            assert_eq!(
                error,
                format!("{message}; {stepper} steps only at integers")
            );
        }
        let mut series = on(engine, model("k", "k", "step_function", ""));
        series.parameters[2].value = Some(ParameterValue::Series {
            data: vec![(4.0, 1.0), (3.5, 2.0), (2.5, 3.0)],
            mode: SeriesMode::StepFunction,
        });
        let error = check_model(&series).expect_err("2.5 is between steps");
        assert_eq!(
            error,
            format!(
                "Series parameter 's' has a point at 2.5, which is not an integer step; \
                 {stepper} steps only at integers"
            )
        );
        assert_eq!(
            check_model(&on(engine, model("k", "k", "linear", ""))),
            Ok(())
        );
    }
    let between = model("k", "k", "step_function", r#","impulse_at":[1.5]"#);
    assert_eq!(check_model(&between), Ok(()), "an ODE model runs at 1.5");
}

#[test]
fn a_pulse_between_steps_is_rejected_by_every_engine() {
    for engine in [
        EngineType::DifferentialEquations,
        EngineType::DifferenceEquations,
    ] {
        let mut pulse = on(engine, model("k", "k", "pulse", ""));
        pulse.parameters[2].value = Some(ParameterValue::Series {
            data: vec![(1.0, 1.0), (1.5, 2.0)],
            mode: SeriesMode::Pulse,
        });
        let error = check_model(&pulse).expect_err("1.5 is between steps");
        assert_eq!(
            error,
            "Series parameter 's' is a pulse at 1.5, which is not an integer step; a \
             pulse acts on one whole step"
        );
    }
}

#[test]
fn a_position_that_is_not_valid_keeps_its_own_message_on_an_integer_engine() {
    let engine = EngineType::DifferenceEquations;
    let mut negative = on(engine, model("k", "k", "linear", r#","impulse_at":[1]"#));
    negative.process.transitions[0].impulse_at = Some(vec![-1.5, f64::INFINITY]);
    let error = CompiledModel::try_from_model(&negative).err().unwrap();
    assert!(
        error.contains("is a finite number of at least 0"),
        "{error}"
    );

    let mut both = on(engine, model("k", "k", "linear", r#","impulse_at":[1.5]"#));
    both.process.transitions[0].impulse_every = Some((1.5, 0.0));
    let error = CompiledModel::try_from_model(&both).err().unwrap();
    assert!(error.contains("not both"), "{error}");

    let mut series = on(engine, model("k", "k", "linear", ""));
    series.parameters[2].value = Some(ParameterValue::Series {
        data: vec![(-1.5, 1.0)],
        mode: SeriesMode::Linear,
    });
    let error = CompiledModel::try_from_model(&series).err().unwrap();
    assert!(
        error.contains("is a finite number of at least 0"),
        "{error}"
    );
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
        .impulse_rates(1.0, &[0], &state, &mut Vec::new())
        .unwrap_err();
    let applied = impulse
        .apply_impulses(1.0, &[0], &mut state.clone())
        .unwrap_err();

    for error in [&program, &rate] {
        assert!(error.ends_with("the parameter 'k' has no value"), "{error}");
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
    let (at_one, at_two) = (model.impulses_at(1.0), model.impulses_at(2.0));
    assert_eq!((at_one.as_slice(), at_two.as_slice()), (&[0][..], &[1][..]));
    model
        .impulse_rates(1.0, &at_one, &state, &mut rates)
        .unwrap();
    assert_eq!(rates, [(0, 0.5)]);
    assert!(model.apply_impulses(1.0, &at_one, &mut state).unwrap());
    assert_eq!(state, [400.0, 600.0, 0.0]);
    assert!(!model.apply_impulses(1.5, &[], &mut state).unwrap());
    assert!(model.apply_impulses(2.0, &at_two, &mut state).unwrap());
    assert_eq!(state, [350.0, 600.0, 50.0]);
}

#[test]
fn a_flow_that_does_not_take_from_a_compartment_does_not_drain_it() {
    let mut model = compiled(&model("k", "z", "step_function", ""));
    let state = model.initial_state().to_vec();

    assert_eq!(model.outflows(0.0, 0.0, &state, 1).unwrap(), []);
}
