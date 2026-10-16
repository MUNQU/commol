//! The flows of a compiled model: `derivatives` is the sum of the change of
//! one event of each flow times its value (`flow_changes`ᵀ · `flow_rates`).

use commol_core::Model;
use commol_model::CompiledModel;

/// A model of bins A, B and C stratified by `group` into `g0` and `g1`, with
/// an accumulator `moved`, read from its JSON with the given transitions.
fn model(transitions: &[&str]) -> Model {
    let json = format!(
        r#"{{"name":"m",
        "parameters":[{{"id":"k1","value":0.3}},{{"id":"k2","value":0.1}},
        {{"id":"r","value":2.5}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}},
        {{"id":"C","name":"C"}}],"accumulators":[{{"id":"moved","name":"moved"}}],
        "stratifications":[{{"id":"group","categories":["g0","g1"]}}],
        "transitions":[{}]}},
        "structure":{{"kind":"aggregate","total":1000,
        "bin_fractions":{{"A":0.7,"B":0.2,"C":0.1}},
        "stratification_fractions":{{"group":{{"g0":0.4,"g1":0.6}}}}}}}}"#,
        transitions.join(",")
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

fn transition(id: &str, source: &[&str], target: &[&str], rate: &str, extra: &str) -> String {
    let quote = |ids: &[&str]| {
        ids.iter()
            .map(|id| format!(r#""{id}""#))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        r#"{{"id":"{id}","source":[{}],"target":[{}],"rate":"{rate}"{extra}}}"#,
        quote(source),
        quote(target)
    )
}

/// Every kind of flow: proportional, absolute, several sources and targets,
/// an inflow, an outflow, a category transition and an accumulator.
fn models() -> Vec<Model> {
    vec![
        model(&[
            &transition(
                "ab",
                &["A"],
                &["B"],
                "k1 * B / total",
                r#","accumulators":["moved"]"#,
            ),
            &transition("bc", &["B"], &["C"], "k2", ""),
        ]),
        model(&[
            &transition("pair", &["A", "B"], &["C", "C"], "k2 * A * B / total", ""),
            &transition("in", &[], &["A"], "r", ""),
            &transition("out", &["C"], &[], "k2", r#","accumulators":["moved"]"#),
        ]),
        model(&[
            &transition("split", &["A"], &["B", "C"], "k1", ""),
            &transition("swap", &["g0"], &["g1"], "k2 * (1 + sin(t))", ""),
        ]),
    ]
}

#[test]
fn derivatives_are_the_changes_of_one_event_times_each_flow() {
    for model in models() {
        let mut compiled = CompiledModel::try_from_model(&model).unwrap();
        let mut state = compiled.initial_state().to_vec();
        // Accumulators are not read; give them values that would show.
        for value in &mut state[compiled.num_compartments()..] {
            *value = 7.0;
        }
        for (step, t) in [(0.0, 0.0), (2.0, 2.5)] {
            let mut derivatives = vec![0.0; state.len()];
            compiled
                .derivatives(step, t, &state, &mut derivatives)
                .unwrap();
            let mut flows = vec![0.0; compiled.flow_changes().len()];
            compiled.flow_rates(step, t, &state, &mut flows).unwrap();
            let mut expected = vec![0.0; state.len()];
            for (change, flow) in compiled.flow_changes().iter().zip(&flows) {
                for &(index, sign) in change.changes() {
                    expected[index] += sign * flow;
                }
            }
            assert_eq!(derivatives, expected, "{}", model.name);
        }
    }
}

#[test]
fn a_proportional_flow_is_its_rate_times_its_source() {
    let model = model(&[&transition("bc", &["B"], &["C"], "k2", "")]);
    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let state = compiled.initial_state().to_vec();
    let flows = compiled.flow_changes().len();
    let (mut rates, mut terms) = (vec![0.0; flows], vec![0.0; flows]);
    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();
    compiled.flow_rates(0.0, 0.0, &state, &mut terms).unwrap();
    for (change, (rate, term)) in compiled.flow_changes().iter().zip(rates.iter().zip(&terms)) {
        let source = change.proportional_to().unwrap();
        assert_eq!(*rate, 0.1);
        assert_eq!(*term, 0.1 * state[source]);
        assert_eq!(change.takes(), &[(source, 1)]);
    }
}

#[test]
fn one_event_takes_one_unit_from_each_source_and_counts_once() {
    let model = model(&[&transition(
        "pair",
        &["A", "B"],
        &["C", "C"],
        "k2 * A * B / total",
        r#","accumulators":["moved"]"#,
    )]);
    let compiled = CompiledModel::try_from_model(&model).unwrap();
    let names = compiled.output_names();
    let changes = &compiled.flow_changes()[0];
    assert_eq!(changes.transition_id(), "pair");
    assert_eq!(changes.proportional_to(), None);
    let named: Vec<(&str, f64)> = changes
        .changes()
        .iter()
        .map(|&(index, change)| (names[index].as_str(), change))
        .collect();
    assert_eq!(
        named,
        vec![
            ("A_g0", -1.0),
            ("C_g0", 1.0),
            ("B_g0", -1.0),
            ("C_g0", 1.0),
            ("moved_g0", 1.0)
        ]
    );
    let takes: Vec<(&str, u32)> = changes
        .takes()
        .iter()
        .map(|&(index, count)| (names[index].as_str(), count))
        .collect();
    assert_eq!(takes, vec![("A_g0", 1), ("B_g0", 1)]);
}

#[test]
fn a_rate_of_t_or_a_linear_series_reads_time() {
    let [steady, _, periodic] = <[Model; 3]>::try_from(models()).ok().unwrap();
    assert!(!CompiledModel::try_from_model(&steady).unwrap().reads_time());
    assert!(
        CompiledModel::try_from_model(&periodic)
            .unwrap()
            .reads_time()
    );
}
