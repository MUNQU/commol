//! Every rate takes the fastest path its shape allows (a specialized product,
//! a stack program or the JIT with resolved slots), and every path gives the
//! value of the formula.

use commol_core::{MathExpression, MathExpressionContext, Model};
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
        "structure":{{"kind":"aggregate","total":1000,
        "bin_fractions":{{"A":0.7,"B":0.2,"C":0.1}}}}}}"#,
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
    for (id, value) in AMOUNTS {
        context.set_compartment(id.to_string(), value);
    }
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
        "k1 * A * !(t > 5)",
        "k1 * A * !(t > 1)",
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
fn a_rate_of_more_variables_than_the_stack_buffer_holds_reads_them_all() {
    let sum: Vec<String> = (0..17).map(|i| format!("v{i}")).collect();
    let rate = format!("!(t > 5) * ({})", sum.join(" + "));

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
        Err("Initial condition value must be non-negative, got: -1".to_string())
    );
    compiled.set_initial_condition(0, 650.0).unwrap();
    assert_eq!(compiled.initial_state(), [650.0, 200.0, 100.0]);
}

#[test]
fn a_series_without_points_is_zero() {
    let mut json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&model("s", 0)).unwrap()).unwrap();
    json["parameters"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"id": "s", "value": {"data": [], "mode": "linear"}}));
    let model: Model = serde_json::from_value(json).unwrap();

    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 1];
    compiled
        .transition_rates(0.0, 0.5, &state, &mut rates)
        .unwrap();

    assert_eq!(rates, [0.0]);
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
