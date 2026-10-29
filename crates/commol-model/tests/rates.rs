//! Every rate takes the fastest path its shape allows (a specialized product,
//! a stack program or the JIT with resolved slots), and every path gives the
//! value of the formula.

use commol_core::math_expression::MAX_NESTING;
use commol_core::{MathExpression, MathExpressionContext, Model, ParameterValue};
use commol_model::CompiledModel;

const PARAMETERS: [(&str, f64); 3] = [("k1", 0.3), ("k2", 0.1), ("k3", 0.25)];
const AMOUNTS: [(&str, f64); 3] = [("A", 700.0), ("B", 200.0), ("C", 100.0)];

/// A → B at `rate`, absolute, with the parameters of `PARAMETERS`, `extra`
/// more (each `v<i>` worth `i`) and the amounts of `AMOUNTS`.
fn model(rate: &str, extra: usize) -> Model {
    let parameters: Vec<String> = PARAMETERS
        .iter()
        .map(|(id, value)| (id.to_string(), *value))
        .chain((0..extra).map(|i| (format!("v{i}"), i as f64)))
        .map(|(id, value)| format!(r#"{{"id":"{id}","value":{value}}}"#))
        .collect();
    let json = format!(
        r#"{{"name":"m","parameters":[{}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}},
        {{"id":"C","name":"C"}}],"accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"{rate}"}}]}},
        "structure":{{"kind":"aggregate",
        "bin_amounts":{{"A":700.0,"B":200.0,"C":100.0}}}}}}"#,
        parameters.join(",")
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

/// The value of `rate` computed by its compiled function from a context of
/// names, which no fast path reads.
fn expected(rate: &str, extra: usize, t: f64) -> f64 {
    let mut context = MathExpressionContext::new();
    for (id, value) in PARAMETERS {
        context.set_parameter(id.to_string(), value);
    }
    for i in 0..extra {
        context.set_parameter(format!("v{i}"), i as f64);
    }
    context.init_compartments(AMOUNTS.iter().map(|(id, _)| id.to_string()).collect());
    let amounts: Vec<f64> = AMOUNTS.iter().map(|&(_, value)| value).collect();
    context.set_compartments_by_index(&amounts);
    context.set_parameter("total".to_string(), 1000.0);
    context.set_step(t.floor());
    context.set_time(t);
    MathExpression::new(rate.to_string())
        .evaluate(&mut context)
        .unwrap()
}

fn rate_of(rate: &str, extra: usize, t: f64) -> f64 {
    let mut compiled = CompiledModel::try_from_model(&model(rate, extra)).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 1];
    compiled
        .transition_rates(t.floor(), t, &state, &mut rates)
        .unwrap();
    rates[0]
}

#[test]
fn every_shape_of_rate_gives_the_value_of_its_formula() {
    for rate in [
        // Products of three: nested on the left or on the right.
        "k1 * A * B",
        "k1 * (A * B)",
        // One minus a value times a product.
        "(1 - k1) * A * B",
        "(1 - k1) * (A * B)",
        "k2 * (1 - k1) * A",
        // A product of two by a sum, over a value, either side.
        "k1 * A * (B + C) / total",
        "(B + C) * (k1 * A) / total",
        // Quotients the specialized forms do not take.
        "(k1 + A) / total",
        "k1 * A * B / total",
        // One value in parentheses.
        "(k1)",
        "(0.5)",
        // A stack program, and the JIT for an operator the program lacks.
        "k1 * -A + k2 * (B - C) / k3",
        "k1 * A * if(t > 5, 0, 1)",
        "if(!(t > 1), k1 * A, 0)",
    ] {
        for t in [0.0, 2.5] {
            let value = rate_of(rate, 0, t);
            assert!(
                (value - expected(rate, 0, t)).abs() <= 1e-12 * value.abs().max(1.0),
                "{rate} at {t}: {value}"
            );
        }
    }
}

#[test]
fn a_rate_at_the_nesting_limit_builds_and_runs_on_a_worker_thread() {
    // Each call holds a sum, a product and a comparison of `t` with a
    // fixed value, which the engine stops at.
    let rate = format!(
        "{}k1{}",
        "k1 + A * if(t > 1, ".repeat(MAX_NESTING),
        ", 0)".repeat(MAX_NESTING)
    );
    let worker = std::thread::Builder::new().stack_size(2 << 20);
    let values = worker
        .spawn(move || [0.5, 2.5].map(|t| (rate_of(&rate, 0, t), expected(&rate, 0, t))))
        .unwrap()
        .join()
        .unwrap();

    assert_eq!(values[0], (0.3, 0.3));
    assert_eq!(values[1].0, values[1].1);
    assert!(values[1].0 > 700.0, "{values:?}");
}

#[test]
fn a_rate_of_more_variables_than_the_stack_buffer_holds_reads_them_all() {
    let sum: Vec<String> = (0..17).map(|i| format!("v{i}")).collect();
    let rate = format!("if(t > 5, 0, 1) * ({})", sum.join(" + "));

    assert_eq!(rate_of(&rate, 17, 0.0), 136.0);
    assert_eq!(rate_of(&rate, 17, 6.0), 0.0);
}

#[test]
fn a_subgroup_total_of_a_model_without_stratifications_names_none() {
    let error = CompiledModel::try_from_model(&model("k1 * total_g1", 0))
        .err()
        .unwrap();

    assert!(
        error.ends_with("Valid names: none (the model has no stratifications)"),
        "{error}"
    );
}

#[test]
fn an_initial_amount_is_of_a_compartment_and_not_negative() {
    let mut compiled = CompiledModel::try_from_model(&model("k1", 0)).unwrap();

    assert_eq!(
        compiled.set_initial_condition(3, 1.0),
        Err("Invalid compartment index: 3. Model has 3 compartments.".to_string())
    );
    assert_eq!(
        compiled.set_initial_condition(0, -1.0),
        Err("Compartment 'A': initial amount is a finite number of at least 0, got -1".to_string())
    );
    compiled.set_initial_condition(0, 650.0).unwrap();
    assert_eq!(compiled.initial_state(), [650.0, 200.0, 100.0]);
}

#[test]
fn a_rate_deeper_than_the_stack_of_a_program_takes_the_jit() {
    let mut rate = "A".to_string();
    for _ in 0..40 {
        rate = format!("B + ({rate})");
    }
    let rate = format!("k1 * ({rate}) / total");

    let value = rate_of(&rate, 0, 0.0);

    assert!((value - expected(&rate, 0, 0.0)).abs() < 1e-9, "{value}");
    assert!((value - 0.3 * (700.0 + 40.0 * 200.0) / 1000.0).abs() < 1e-9);
}

#[test]
fn a_formula_reads_an_earlier_formula_of_the_same_step() {
    let mut json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&model("f2", 0)).unwrap()).unwrap();
    let parameters = json["parameters"].as_array_mut().unwrap();
    parameters.push(serde_json::json!({"id": "f1", "value": "k1 * step"}));
    parameters.push(serde_json::json!({"id": "f2", "value": "f1 + 1"}));
    let model: Model = serde_json::from_value(json).unwrap();

    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 1];
    let mut rate_at = |step: f64| {
        compiled
            .transition_rates(step, step, &state, &mut rates)
            .unwrap();
        rates[0]
    };

    // f1 is 0.3 · step at the step f2 reads it, not at the step before.
    assert_eq!(rate_at(3.0), 0.3 * 3.0 + 1.0);
    assert_eq!(rate_at(5.0), 0.3 * 5.0 + 1.0);
}

#[test]
fn a_rate_outside_the_language_fails_at_build() {
    let error = CompiledModel::try_from_model(&model("step > 1", 0))
        .err()
        .unwrap();

    assert_eq!(
        error,
        "Transition 'ab': Formula 'step > 1': the comparison '>' is only allowed in the \
         condition of if(condition, a, b)"
    );
}

#[test]
fn an_error_of_a_rate_names_its_transition() {
    let mut model = model("k1 * A * B", 0);
    model.parameters[0].value = None;
    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 1];

    let error = compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap_err();

    assert_eq!(
        error,
        "The rate of transition 'ab' cannot be evaluated: Parameter index '4' not found"
    );
}

#[test]
fn a_formula_parameter_outside_the_language_fails_at_build() {
    let mut model = model("k2", 0);
    model.parameters[1].value = Some(ParameterValue::Formula("k1 * (step > 30)".to_string()));

    assert_eq!(
        CompiledModel::try_from_model(&model).err().unwrap(),
        "Parameter 'k2': Formula 'k1 * (step > 30)': the comparison '>' is only allowed in \
         the condition of if(condition, a, b)"
    );
}

/// A → B at `rate` in a model stratified by `group` into `g1` and `g2`, with
/// the matrix `M` of `group`: rows `g1` 2 and 1, `g2` 1 and 3.
fn grouped(rate: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k1","value":0.3}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "stratifications":[{{"id":"group","categories":["g1","g2"]}}],
        "matrices":[{{"id":"M","stratification":"group",
        "values":{{"g1":{{"g1":2.0,"g2":1.0}},"g2":{{"g1":1.0,"g2":3.0}}}}}}],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"{rate}"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":900.0,"B":100.0}},
        "stratification_fractions":{{"group":{{"g1":0.5,"g2":0.5}}}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

#[test]
fn a_rate_reads_the_largest_eigenvalue_of_a_matrix() {
    let mut compiled = CompiledModel::try_from_model(&grouped("k1 * M_radius")).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 2];

    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();

    let radius = (5.0 + 5.0_f64.sqrt()) / 2.0;
    for rate in rates {
        assert!((rate - 0.3 * radius).abs() < 1e-12, "{rate}");
    }
}
