//! The rules of a model that the engine checks when it builds one, whatever
//! path brings it: unique ids, the categories of a stratification, a process
//! with bins and transitions, the bin amounts, the values a run needs, and
//! the names the bins build.

use commol_core::Model;
use commol_model::CompiledModel;
use serde_json::{Value, json};

/// A → B at `k1 * A`, with `k1` of 0.1, bins A and B of 90 and 10, and the
/// stratification `group` of `g1` and `g2`, split half and half.
fn model_json() -> Value {
    json!({
        "name": "m",
        "parameters": [{"id": "k1", "value": 0.1}],
        "engine": {"type": "DifferenceEquations"},
        "process": {
            "bins": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}],
            "accumulators": [{"id": "n", "name": "n"}],
            "stratifications": [{"id": "group", "categories": ["g1", "g2"]}],
            "transitions": [{"id": "ab", "source": ["A"], "target": ["B"], "rate": "k1",
                             "accumulators": ["n"]}]
        },
        "structure": {
            "kind": "aggregate",
            "bin_amounts": {"A": 90.0, "B": 10.0},
            "stratification_fractions": {"group": {"g1": 0.5, "g2": 0.5}}
        }
    })
}

fn build(json: Value) -> Result<CompiledModel, String> {
    let model: Model = serde_json::from_value(json).expect("valid model JSON");
    CompiledModel::try_from_model(&model)
}

/// The JSON of the model with the item `item` added to the list at `path`.
fn with_one_more(path: &[&str], item: Value) -> Value {
    let mut json = model_json();
    let mut list = &mut json;
    for key in path {
        list = &mut list[key];
    }
    list.as_array_mut().expect("a list").push(item);
    json
}

#[test]
fn a_repeated_id_is_refused() {
    let cases = [
        (
            with_one_more(&["process", "bins"], json!({"id": "A", "name": "A2"})),
            "Two bins share the id 'A'; give each bin its own id",
        ),
        (
            with_one_more(
                &["process", "accumulators"],
                json!({"id": "n", "name": "n2"}),
            ),
            "Two accumulators share the id 'n'; give each accumulator its own id",
        ),
        (
            with_one_more(
                &["process", "stratifications"],
                json!({"id": "group", "categories": ["h1"]}),
            ),
            "Two stratifications share the id 'group'; give each stratification its own id",
        ),
        (
            with_one_more(
                &["process", "transitions"],
                json!({"id": "ab", "source": ["B"], "target": ["A"], "rate": "k1"}),
            ),
            "Two transitions share the id 'ab'; give each transition its own id",
        ),
        (
            with_one_more(&["parameters"], json!({"id": "k1", "value": 0.2})),
            "Two parameters share the id 'k1'; give each parameter its own id",
        ),
    ];

    for (json, message) in cases {
        assert_eq!(build(json).err().as_deref(), Some(message));
    }
}

#[test]
fn the_categories_of_a_stratification_are_checked() {
    let with_categories = |categories: Value| {
        let mut json = model_json();
        json["process"]["stratifications"][0]["categories"] = categories;
        json
    };

    assert_eq!(
        build(with_categories(json!(["g1", "g1"]))).err().as_deref(),
        Some("Stratification 'group' names the category 'g1' twice; name each category once")
    );
    assert_eq!(
        build(with_categories(json!([]))).err().as_deref(),
        Some("Stratification 'group' has no category; give it at least one")
    );
}

#[test]
fn a_process_without_bins_or_transitions_is_refused() {
    let mut without_transitions = model_json();
    without_transitions["process"]["transitions"] = json!([]);
    let mut without_bins = without_transitions.clone();
    without_bins["process"]["bins"] = json!([]);
    without_bins["structure"]["bin_amounts"] = json!({});

    assert_eq!(
        build(without_transitions).err().as_deref(),
        Some("The process has no transition; an aggregate model needs at least one transition")
    );
    assert_eq!(
        build(without_bins).err().as_deref(),
        Some("The process has no bin; an aggregate model needs at least one bin")
    );
}

#[test]
fn the_bin_amounts_are_the_bins() {
    let with_amounts = |amounts: Value| {
        let mut json = model_json();
        json["structure"]["bin_amounts"] = amounts;
        json
    };

    assert_eq!(
        build(with_amounts(json!({"A": 90.0}))).err().as_deref(),
        Some(
            "Initial bin amounts keys must exactly match bin ids. Missing ids: [\"B\"], \
             Extra ids: []."
        )
    );
    assert_eq!(
        build(with_amounts(json!({"A": 90.0, "B": 10.0, "Z": 1.0})))
            .err()
            .as_deref(),
        Some(
            "Initial bin amounts keys must exactly match bin ids. Missing ids: [], Extra ids: \
             [\"Z\"]."
        )
    );
}

#[test]
fn a_run_needs_every_value() {
    let mut json = model_json();
    json["parameters"][0]["value"] = Value::Null;
    json["structure"]["bin_amounts"]["B"] = Value::Null;
    let mut compiled = build(json).unwrap();
    let state = compiled.initial_state().to_vec();
    let mut rates = vec![0.0; 2];

    assert_eq!(
        compiled.check_values().err().as_deref(),
        Some(
            "The model cannot run: the parameters [\"k1\"] and the amounts of the bins [\"B\"] \
             have no value. Give them a value, or calibrate them and write the calibrated \
             values back"
        )
    );
    assert_eq!(
        compiled
            .transition_rates(0.0, 0.0, &state, &mut rates)
            .err()
            .as_deref(),
        Some("The rate of transition 'ab' cannot be evaluated: the parameter 'k1' has no value")
    );

    compiled.set_parameter("k1", 0.1).unwrap();
    assert_eq!(
        compiled.check_values().err().as_deref(),
        Some(
            "The model cannot run: the amounts of the bins [\"B\"] have no value. Give them a \
             value, or calibrate them and write the calibrated values back"
        )
    );

    let index = |name: &str| {
        compiled
            .compartments()
            .iter()
            .position(|compartment| compartment == name)
            .unwrap()
    };
    let (b_g1, b_g2) = (index("B_g1"), index("B_g2"));
    compiled.set_initial_condition(b_g2, 5.0).unwrap();
    assert_eq!(
        compiled.check_values().err().as_deref(),
        Some(
            "The model cannot run: the amounts of the compartments [\"B_g1\"] have no value. \
             Give them a value, or calibrate them and write the calibrated values back"
        )
    );

    compiled.set_initial_condition(b_g1, 5.0).unwrap();
    assert_eq!(compiled.check_values(), Ok(()));
}

/// The stratification `id` of `categories`.
fn stratification(id: &str, categories: &[&str]) -> Value {
    json!({"id": id, "categories": categories})
}

/// [`model_json`] with the stratifications `stratifications`, each split
/// evenly, and then `edit`.
fn stratified(stratifications: &[Value], edit: impl FnOnce(&mut Value)) -> Value {
    let mut json = model_json();
    let mut fractions = serde_json::Map::new();
    for stratification in stratifications {
        let categories = stratification["categories"].as_array().unwrap();
        let share = 1.0 / categories.len() as f64;
        let split = categories
            .iter()
            .map(|category| (category.as_str().unwrap().to_string(), json!(share)))
            .collect();
        let id = stratification["id"].as_str().unwrap().to_string();
        fractions.insert(id, Value::Object(split));
    }
    json["process"]["stratifications"] = json!(stratifications);
    json["structure"]["stratification_fractions"] = Value::Object(fractions);
    edit(&mut json);
    json
}

/// Add the bin `id`, of 5.
fn add_bin(json: &mut Value, id: &str) {
    let bins = json["process"]["bins"].as_array_mut().unwrap();
    bins.push(json!({"id": id, "name": id}));
    json["structure"]["bin_amounts"][id] = json!(5.0);
}

/// Give the accumulator `n` the id `id`.
fn rename_accumulator(json: &mut Value, id: &str) {
    json["process"]["accumulators"][0]["id"] = json!(id);
    json["process"]["transitions"][0]["accumulators"] = json!([id]);
}

/// The sentences of the refusal of a model that builds a name twice.
fn refusal(json: Value) -> String {
    let message = build(json).err().expect("the model is refused");
    let opening = "Two things or more build one name, so a rate, an initial condition or a \
                   result could read one for another: ";
    message
        .strip_prefix(opening)
        .expect("the message of a name built twice")
        .to_string()
}

const OWN_NAME: &str = ", so that each has a name of its own.";

#[test]
fn a_name_of_two_owners_is_refused() {
    let group = [stratification("group", &["g1", "g2"])];
    let group_and_type = [
        stratification("group", &["g1", "g2"]),
        stratification("type", &["t1", "t2"]),
    ];
    let joined = [stratification("group", &["g1_h", "h"])];

    let cases = [
        (
            stratified(&group, |json| add_bin(json, "A_g1")),
            "'A_g1' names the compartment of bin 'A' in 'group:g1' and the bin 'A_g1'. Rename \
             bin 'A', bin 'A_g1' or category 'group:g1'",
        ),
        (
            stratified(&group_and_type, |json| add_bin(json, "A_g1")),
            "'A_g1' names the compartments of bin 'A' in 'group:g1' and the bin 'A_g1'. Rename \
             bin 'A', bin 'A_g1' or category 'group:g1'",
        ),
        (
            stratified(&joined, |json| add_bin(json, "A_g1")),
            "'A_g1_h' names the compartment of bin 'A' in 'group:g1_h' and the compartment of \
             bin 'A_g1' in 'group:h'. Rename bin 'A', bin 'A_g1', category 'group:g1_h' or \
             category 'group:h'",
        ),
        (
            stratified(&group_and_type, |json| add_bin(json, "A_t1")),
            "'A_t1' names the bin 'A_t1' and the compartments of bin 'A' in 'type:t1'. Rename \
             bin 'A_t1', bin 'A' or category 'type:t1'",
        ),
        (
            stratified(&group, |json| rename_accumulator(json, "A")),
            "'A' names the bin 'A' and the accumulator 'A'. Rename bin 'A' or accumulator 'A'",
        ),
        (
            stratified(&group, |json| rename_accumulator(json, "A_g1")),
            "'A_g1' names the compartment of bin 'A' in 'group:g1' and the accumulator 'A_g1'. \
             Rename bin 'A', accumulator 'A_g1' or category 'group:g1'",
        ),
        (
            stratified(&group, |json| rename_accumulator(json, "total_g1")),
            "'total_g1' names the accumulator 'total_g1' and the subgroup total of \
             'group:g1'. Rename accumulator 'total_g1' or category 'group:g1'",
        ),
    ];

    for (json, sentence) in cases {
        assert_eq!(refusal(json), format!("{sentence}{OWN_NAME}"));
    }
}

#[test]
fn a_name_of_one_owner_from_two_lists_of_category_ids_is_refused() {
    let only_in_a = json!({"id": "type", "categories": ["b"],
                           "conditions": [{"stratification": "group", "category": "a"}]});
    let cases = [
        (
            stratification("type", &["b", "c"]),
            "compartments",
            "outputs",
        ),
        (only_in_a, "compartment", "output"),
    ];

    for (kind, of, of_outputs) in cases {
        let json = stratified(&[stratification("group", &["a", "a_b"]), kind], |_| {});
        let rename = format!(
            "Rename category 'group:a', category 'type:b' or category 'group:a_b'{OWN_NAME}"
        );
        let sentence = |bin: &str| {
            format!(
                "'{bin}_a_b' names the compartment of bin '{bin}' in 'group:a', 'type:b' and \
                 the {of} of bin '{bin}' in 'group:a_b'. {rename}"
            )
        };
        assert_eq!(
            refusal(json),
            format!(
                "{} {} 'n_a_b' names the output of accumulator 'n' in 'group:a', 'type:b' and \
                 the {of_outputs} of accumulator 'n' in 'group:a_b'. {rename} 'total_a_b' names \
                 the subgroup total of 'group:a', 'type:b' and the subgroup total of \
                 'group:a_b'. {rename}",
                sentence("A"),
                sentence("B")
            )
        );
    }
}

#[test]
fn the_message_gives_only_the_root_of_each_conflict() {
    let deeper = stratified(
        &[
            stratification("group", &["g1_h", "h"]),
            stratification("type", &["t1", "t2"]),
        ],
        |json| add_bin(json, "A_g1"),
    );
    let only_in_a = json!({"id": "type", "categories": ["b"],
                           "conditions": [{"stratification": "group", "category": "a"}]});
    let under_a_shared_category = stratified(
        &[
            stratification("level", &["l1"]),
            stratification("group", &["a", "a_b"]),
            only_in_a,
        ],
        |_| {},
    );

    assert_eq!(
        refusal(deeper),
        format!(
            "'A_g1_h' names the compartments of bin 'A' in 'group:g1_h' and the compartments \
             of bin 'A_g1' in 'group:h'. Rename bin 'A', bin 'A_g1', category 'group:g1_h' or \
             category 'group:h'{OWN_NAME}"
        )
    );
    let rename =
        format!("Rename category 'group:a', category 'type:b' or category 'group:a_b'{OWN_NAME}");
    let partial = |bin: &str| {
        format!(
            "'{bin}_a_b' names the compartments of bin '{bin}' in 'group:a', 'type:b' and the \
             compartments of bin '{bin}' in 'group:a_b'. {rename}"
        )
    };
    assert_eq!(
        refusal(under_a_shared_category),
        format!(
            "{} {} 'n_l1_a_b' names the output of accumulator 'n' in 'level:l1', 'group:a', \
             'type:b' and the output of accumulator 'n' in 'level:l1', 'group:a_b'. {rename} \
             'total_a_b' names the subgroup total of 'group:a', 'type:b' and the subgroup \
             total of 'group:a_b'. {rename}",
            partial("A"),
            partial("B")
        )
    );
}

#[test]
fn a_category_id_that_two_stratifications_share_builds() {
    let shared = stratified(
        &[
            stratification("group", &["c1", "c2"]),
            stratification("type", &["c1", "c2"]),
        ],
        |_| {},
    );

    assert!(build(shared).is_ok());
}

#[test]
fn a_bin_whose_id_starts_with_a_name_of_another_builds() {
    let group = [stratification("group", &["g1", "g2"])];

    assert!(build(stratified(&group, |json| add_bin(json, "A_g1x"))).is_ok());
}

/// The stratifications `group = [a, b]` and `type = [a, c]`, which share the
/// category id `a`.
fn sharing_a() -> [Value; 2] {
    [
        stratification("group", &["a", "b"]),
        stratification("type", &["a", "c"]),
    ]
}

/// Give the transition `ab` the rate `rate`.
fn with_rate(json: &mut Value, rate: &str) {
    json["process"]["transitions"][0]["rate"] = json!(rate);
}

/// What the refusal of a model whose formula reads a name of two meanings
/// says after its opening.
fn read_refusal(json: Value) -> String {
    let message = build(json).err().expect("the model is refused");
    let opening = "A rate or a formula parameter reads a name to which a category id that two \
                   stratifications share gives two meanings or more: ";
    message
        .strip_prefix(opening)
        .expect("the message of a name of two meanings that a formula reads")
        .to_string()
}

const RENAME_A: &str = "Rename category 'group:a' or category 'type:a', so that each has a \
                        name of its own.";

/// How the message reads each meaning of a name with names of one meaning.
fn reading(sums: &str) -> String {
    format!(" To keep the ids, read a meaning as a sum of names of one meaning: {sums}.")
}

#[test]
fn a_formula_that_reads_a_name_of_a_shared_category_is_refused() {
    let total_a = format!(
        "'total_a' names the subgroup total of 'group:a' and the subgroup total of 'type:a'. \
         {RENAME_A}{}",
        reading("'group:a' is 'total_a_a + total_a_c'; 'type:a' is 'total_a_a + total_b_a'")
    );
    let a_a = format!(
        "'A_a' names the compartments of bin 'A' in 'group:a' and the compartments of bin 'A' \
         in 'type:a'. {RENAME_A}{}",
        reading("'group:a' is 'A_a_a + A_a_c'; 'type:a' is 'A_a_a + A_b_a'")
    );
    let with_rate_of = |rate: &str| stratified(&sharing_a(), |json| with_rate(json, rate));
    let with_formula = stratified(&sharing_a(), |json| {
        with_rate(json, "k1 * total_a * p");
        json["parameters"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id": "p", "value": "total_a / A"}));
    });
    let with_stratified_rate = stratified(&sharing_a(), |json| {
        json["process"]["transitions"][0]["stratified_rates"] = json!([{
            "conditions": [{"stratification": "group", "category": "b"}],
            "rate": "k1 * A_a"
        }]);
    });

    let cases = [
        (
            with_rate_of("k1 * total_a"),
            format!("transition 'ab' reads 'total_a'. {total_a}"),
        ),
        (
            with_rate_of("k1 * A_a"),
            format!("transition 'ab' reads 'A_a'. {a_a}"),
        ),
        (
            with_rate_of("k1 * total_a * A_a / A_a_a"),
            format!("transition 'ab' reads 'A_a' and 'total_a'. {a_a} {total_a}"),
        ),
        (
            with_formula,
            format!("transition 'ab' reads 'total_a'; parameter 'p' reads 'total_a'. {total_a}"),
        ),
        (
            with_stratified_rate,
            format!("transition 'ab' reads 'A_a'. {a_a}"),
        ),
    ];

    for (json, sentences) in cases {
        assert_eq!(read_refusal(json), sentences);
    }
}

#[test]
fn a_meaning_that_no_name_of_one_meaning_reads_needs_a_rename() {
    // `type` splits only `b`, so `A_a` is the compartment of `group:a` and the
    // compartments of `A` in `type:a`, and no other name is that compartment.
    let only_in_b = json!({"id": "type", "categories": ["a", "c"],
                           "conditions": [{"stratification": "group", "category": "b"}]});
    let json = stratified(&[stratification("group", &["a", "b"]), only_in_b], |json| {
        with_rate(json, "k1 * A_a / total_a");
    });

    assert_eq!(
        read_refusal(json),
        format!(
            "transition 'ab' reads 'A_a' and 'total_a'. 'A_a' names the compartment of bin 'A' \
             in 'group:a' and the compartments of bin 'A' in 'type:a'. {RENAME_A}{} Only a \
             rename reads 'group:a', which has no other name. 'total_a' names the subgroup \
             total of 'group:a' and the subgroup total of 'type:a'. {RENAME_A}{} Only a rename \
             reads 'group:a', which is no sum of names of one meaning.",
            reading("'type:a' is 'A_b_a'"),
            reading("'type:a' is 'total_b_a'")
        )
    );
}

#[test]
fn stratifications_of_disjoint_conditions_that_share_an_id_read_each_meaning_by_a_total() {
    // `type` splits only `a` and `kind` only `b`: no compartment has both
    // `x`, and `total_x` is the total of `type:x` and that of `kind:x`.
    let only_in = |id: &str, categories: [&str; 2], category: &str| {
        json!({"id": id, "categories": categories,
               "conditions": [{"stratification": "group", "category": category}]})
    };
    let json = stratified(
        &[
            stratification("group", &["a", "b"]),
            only_in("type", ["x", "y"], "a"),
            only_in("kind", ["x", "z"], "b"),
        ],
        |json| with_rate(json, "k1 * total_x"),
    );

    assert_eq!(
        read_refusal(json),
        format!(
            "transition 'ab' reads 'total_x'. 'total_x' names the subgroup total of 'type:x' \
             and the subgroup total of 'kind:x'. Rename category 'type:x' or category \
             'kind:x', so that each has a name of its own.{}",
            reading("'type:x' is 'total_a_x'; 'kind:x' is 'total_b_x'")
        )
    );
}

#[test]
fn the_valid_subgroup_totals_leave_out_the_names_of_two_meanings() {
    let json = stratified(&sharing_a(), |json| with_rate(json, "k1 * total_d"));

    assert_eq!(
        build(json).err().as_deref(),
        Some(
            "Unknown subgroup total(s) 'total_d'. Category suffixes follow the declaration \
             order of the stratifications. Valid names: total_a_a, total_a_c, total_b, \
             total_b_a, total_b_c, total_c"
        )
    );
}

#[test]
fn a_name_of_two_meanings_stays_reserved() {
    let json = stratified(&sharing_a(), |json| {
        json["parameters"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id": "total_a", "value": 1.0}));
    });

    assert_eq!(
        build(json).err().as_deref(),
        Some(
            "Parameter ids cannot be names the engine provides: 'total_a' (a subgroup total). \
             Rename the parameter(s)."
        )
    );
}

#[test]
fn a_shared_category_builds_when_no_formula_reads_a_name_it_gives_two_meanings() {
    let rate = "k1 * (total_a_a + total_a_c) / total_c * A_b * A_c / (A_a_a + A_a_c) / A";
    let json = stratified(&sharing_a(), |json| with_rate(json, rate));

    assert!(build(json).is_ok());
}
