//! The rate a compartment takes from the stratified rates of a transition:
//! the rate whose conditions it holds, and the most specific one of them.

use commol_core::Process;
use commol_model::describe_flows;

/// Bins A and B, `group` of `g1`, `g2` and `g3`, and `type` on `g2` and `g3`,
/// with a transition from A to B of the given stratified rates and `k0` for
/// the compartments that match none.
fn process(stratified_rates: &str) -> Process {
    let json = format!(
        r#"{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "stratifications":[{{"id":"group","categories":["g1","g2","g3"]}},
        {{"id":"type","categories":["t1","t2"],
          "conditions":[{{"stratification":"group","categories":["g2","g3"]}}]}}],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"k0",
          "stratified_rates":[{stratified_rates}]}}]}}"#
    );
    serde_json::from_str(&json).expect("valid process JSON")
}

/// The rate of the flow from each compartment of A.
fn rates(process: &Process) -> Vec<(String, String)> {
    describe_flows(process, None)
        .unwrap()
        .into_iter()
        .map(|flow| (flow.source.unwrap(), flow.rate))
        .collect()
}

#[test]
fn a_condition_on_a_set_counts_as_one_condition() {
    let process = process(
        r#"{"conditions":[{"stratification":"group","categories":["g2","g3"]}],"rate":"k1"},
           {"conditions":[{"stratification":"group","category":"g3"},
                          {"stratification":"type","category":"t2"}],"rate":"k2"}"#,
    );

    let expected = [
        ("A_g1", "k0"),
        ("A_g2_t1", "k1"),
        ("A_g2_t2", "k1"),
        ("A_g3_t1", "k1"),
        ("A_g3_t2", "k2"),
    ];
    assert_eq!(
        rates(&process),
        expected.map(|(source, rate)| (source.to_string(), rate.to_string()))
    );
}

#[test]
fn a_set_and_one_category_of_the_same_specificity_keep_the_rate_declared_first() {
    let process = process(
        r#"{"conditions":[{"stratification":"group","categories":["g2","g3"]}],"rate":"k1"},
           {"conditions":[{"stratification":"group","category":"g3"}],"rate":"k2"}"#,
    );

    let rate_of_g3: Vec<String> = rates(&process)
        .into_iter()
        .filter(|(source, _)| source.starts_with("A_g3"))
        .map(|(_, rate)| rate)
        .collect();
    assert_eq!(rate_of_g3, ["k1", "k1"]);
}

#[test]
fn the_conditions_of_a_stratified_rate_are_checked() {
    let twice = process(
        r#"{"conditions":[{"stratification":"group","category":"g2"},
                          {"stratification":"group","category":"g3"}],"rate":"k1"}"#,
    );
    let none = process(r#"{"conditions":[],"rate":"k1"}"#);

    assert_eq!(
        describe_flows(&twice, None).err().as_deref(),
        Some(
            "Transition 'ab', stratified rate 0: two conditions name 'group'; give one \
             condition per stratification"
        )
    );
    assert_eq!(
        describe_flows(&none, None).err().as_deref(),
        Some("Transition 'ab', stratified rate 0 has no condition; give it at least one")
    );
}
