//! The flows of a compiled model: `derivatives` is the sum of the change of
//! one event of each flow times its value (`flow_changes`ᵀ · `flow_rates`).

use commol_core::Model;
use commol_model::{CompiledModel, subgroup_total_names};

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
        "structure":{{"kind":"aggregate",
        "bin_amounts":{{"A":700.0,"B":200.0,"C":100.0}},
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
fn an_event_takes_as_many_units_as_a_source_appears() {
    let model = model(&[&transition(
        "double",
        &["A", "A"],
        &["B", "C"],
        "k1 * A * A / total",
        "",
    )]);
    let compiled = CompiledModel::try_from_model(&model).unwrap();
    let names = compiled.output_names();
    let takes: Vec<(&str, u32)> = compiled.flow_changes()[0]
        .takes()
        .iter()
        .map(|&(index, count)| (names[index].as_str(), count))
        .collect();
    assert_eq!(takes, vec![("A_g0", 2)]);
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

/// The rate of `ab` per unit of `A_g0`, with the given stratified rates.
fn rate_of_a_g0(stratified_rates: &str) -> f64 {
    let extra = format!(r#","stratified_rates":[{stratified_rates}]"#);
    let model = model(&[&transition("ab", &["A"], &["B"], "k2", &extra)]);
    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; compiled.flow_changes().len()];
    compiled.flow_rates(0.0, 0.0, &state, &mut rates).unwrap();
    let a_g0 = compiled
        .output_names()
        .iter()
        .position(|name| name == "A_g0")
        .unwrap();
    rates[0] / state[a_g0]
}

#[test]
fn the_most_specific_stratified_rate_wins_and_the_first_on_a_tie() {
    let any = r#"{"conditions":[],"rate":"r"}"#;
    let g0 = |rate: &str| {
        format!(
            r#"{{"conditions":[{{"stratification":"group","category":"g0"}}],"rate":"{rate}"}}"#
        )
    };
    assert_eq!(rate_of_a_g0(&format!("{any},{}", g0("k1"))), 0.3);
    assert_eq!(rate_of_a_g0(&format!("{},{}", g0("k1"), g0("k2"))), 0.3);
}

#[test]
fn a_bin_at_the_start_of_a_rate_is_its_compartment() {
    // Per compartment, B reads B_g0 in the flow of g0 and B_g1 in that of g1.
    let rate = |expression: &str| {
        let extra = r#","per_compartment":true"#;
        let model = model(&[&transition("ab", &["A"], &["B"], expression, extra)]);
        let mut compiled = CompiledModel::try_from_model(&model).unwrap();
        let state = compiled.initial_state().to_vec();
        let mut rates = vec![0.0; compiled.flow_changes().len()];
        compiled.flow_rates(0.0, 0.0, &state, &mut rates).unwrap();
        rates
    };
    let rates = rate("B * k1 / total");
    assert_eq!(rates, rate("k1 * B / total"));
    assert_ne!(rates[0], rates[1]);
}

#[test]
fn a_category_transition_has_one_source_and_one_target() {
    let model = model(&[&transition("swap", &["g0"], &["g1", "g1"], "k2", "")]);
    let error = CompiledModel::try_from_model(&model).err().unwrap();
    assert!(
        error.contains("exactly one source and one target"),
        "{error}"
    );
}

#[test]
fn a_partial_bin_total_leaves_out_the_compartments() {
    let json = r#"{"name":"m","parameters":[],"engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"}],"accumulators":[],
        "stratifications":[{"id":"group","categories":["g0","g1"]},
        {"id":"age","categories":["y","o"]}],"transitions":[]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    let names = subgroup_total_names(&model.process);
    assert!(names.contains(&"A_g0".to_string()), "{names:?}");
    assert!(names.contains(&"total_g0_y".to_string()), "{names:?}");
    assert!(!names.contains(&"A_g0_y".to_string()), "{names:?}");
}

#[test]
fn a_subgroup_total_of_a_stratification_that_expands_nothing_names_none() {
    // `subgroup_total_names` reads a process before its conditions are checked:
    // a condition no compartment meets leaves every compartment without a
    // category.
    let process: commol_core::Process = serde_json::from_str(
        r#"{"bins":[{"id":"A","name":"A"}],"stratifications":[{"id":"g",
        "categories":["g1","g2"],"conditions":[{"stratification":"h","category":"h1"}]}],
        "transitions":[]}"#,
    )
    .expect("valid process JSON");

    assert!(subgroup_total_names(&process).is_empty());
}

#[test]
fn an_inflow_chooses_its_compartment_with_a_category_and_has_no_to() {
    let inflow = |condition: &str| {
        model(&[&transition(
            "in",
            &[],
            &["A"],
            "r",
            &format!(
                r#","stratified_rates":[{{"conditions":[{{"stratification":"group",{condition}}}],"rate":"k1"}}]"#
            ),
        )])
    };
    let error = CompiledModel::try_from_model(&inflow(r#""to":"g1""#))
        .err()
        .unwrap();
    assert!(error.contains("no target to redirect"), "{error}");
    let mut compiled = CompiledModel::try_from_model(&inflow(r#""category":"g1""#)).unwrap();
    let mut rates = vec![0.0; 2];
    let state = compiled.initial_state().to_vec();
    compiled.flow_rates(0.0, 0.0, &state, &mut rates).unwrap();
    assert_eq!(rates, [2.5, 0.3]);
}

#[test]
fn a_bin_in_a_rate_is_the_total_of_the_bin_without_per_compartment() {
    let mut compiled =
        CompiledModel::try_from_model(&model(&[&transition("ab", &["A"], &["B"], "k1 * A", "")]))
            .unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 2];
    compiled.flow_rates(0.0, 0.0, &state, &mut rates).unwrap();
    // A holds 700 split 280 and 420 by group; each flow reads the whole bin.
    assert_eq!(rates, [0.3 * 700.0, 0.3 * 700.0]);
}

/// Bins A and B with a stratification `h` and then a stratification `g` of the
/// given `conditions`, A → B at `k`.
fn conditional(conditions: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.1}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],
        "stratifications":[{{"id":"h","categories":["h1","h2"]}},
        {{"id":"g","categories":["g1","g2"],"conditions":{conditions}}}],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"k"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":1000.0,"B":0.0}},
        "stratification_fractions":{{"h":{{"h1":0.5,"h2":0.5}},"g":{{"g1":0.5,"g2":0.5}}}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

#[test]
fn a_stratification_condition_filters_by_a_category() {
    let compiled =
        CompiledModel::try_from_model(&conditional(r#"[{"stratification":"h","category":"h2"}]"#))
            .unwrap();
    assert_eq!(
        compiled.compartments(),
        ["A_h1", "A_h2_g1", "A_h2_g2", "B_h1", "B_h2_g1", "B_h2_g2"]
    );

    let model = conditional(r#"[{"stratification":"h"}]"#);

    assert_eq!(
        CompiledModel::try_from_model(&model).err().unwrap(),
        "Stratification 'g': the condition on 'h' gives neither a category nor a set of \
         categories; give one of them"
    );
}

#[test]
fn a_condition_names_a_stratification_declared_before_it() {
    for named in ["g", "z"] {
        let model = conditional(&format!(
            r#"[{{"stratification":"{named}","category":"g1"}}]"#
        ));

        assert_eq!(
            CompiledModel::try_from_model(&model).err().unwrap(),
            format!(
                "Stratification 'g': a condition names '{named}', which is not a stratification \
                 declared before it"
            )
        );
    }
}

#[test]
fn a_rate_that_is_not_a_number_names_its_flow_and_its_position() {
    let model = model(&[&transition("bc", &["B"], &["C"], "k2 * sqrt(1 - t)", "")]);
    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut derivatives = vec![0.0; state.len()];

    assert_eq!(
        compiled.derivatives(2.0, 2.5, &state, &mut derivatives),
        Err("The rate of flow 'bc' at 2.5 is not a number".to_string())
    );
}

#[test]
fn a_compartment_that_is_not_a_number_gives_flows_that_are_not_numbers() {
    let model = model(&[&transition("bc", &["B"], &["C"], "k2", "")]);
    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let mut state = compiled.initial_state().to_vec();
    let source = compiled.flow_changes()[0].proportional_to().unwrap();
    state[source] = f64::NAN;
    let mut derivatives = vec![0.0; state.len()];

    compiled
        .derivatives(0.0, 0.0, &state, &mut derivatives)
        .unwrap();

    assert!(derivatives[source].is_nan());
}

#[test]
fn a_bin_whose_id_starts_with_another_finds_only_its_compartments() {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"A_x","name":"A_x"},
        {"id":"B","name":"B"}],"accumulators":[],
        "stratifications":[{"id":"group","categories":["g0","g1"]}],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k * A"}]},
        "structure":{"kind":"aggregate",
        "bin_amounts":{"A":700.0,"A_x":50.0,"B":0.0},
        "stratification_fractions":{"group":{"g0":0.4,"g1":0.6}}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");

    let mut compiled = CompiledModel::try_from_model(&model).unwrap();
    let names = compiled.output_names();
    let sources: Vec<&str> = compiled
        .flow_changes()
        .iter()
        .flat_map(|change| change.takes())
        .map(|&(index, _)| names[index].as_str())
        .collect();
    assert_eq!(sources, ["A_g0", "A_g1"]);
    // The bin A in the rate is the total of A_g0 and A_g1 only.
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 2];
    compiled.flow_rates(0.0, 0.0, &state, &mut rates).unwrap();
    assert_eq!(rates, [0.1 * 700.0, 0.1 * 700.0]);
}
