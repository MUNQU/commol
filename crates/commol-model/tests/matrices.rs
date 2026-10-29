//! A matrix between the categories of a stratification: its variables, the
//! matrix sums of a rate as the flows hold them, and what a build rejects.

use commol_core::{Model, Neighbors, ParameterValue};
use commol_model::{CompiledModel, MatrixVariable, check_model, describe_flows, matrix_variables};

/// A model of bins A and B stratified by `group` into `g1` and `g2`, and by
/// the stratifications of `more`, with the matrix `M` of `group` (rows `g1`
/// 2 and 1, `g2` 1 and 3) and A → B at `rate`.
fn model(rate: &str, more: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k1","value":0.3}}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "stratifications":[{{"id":"group","categories":["g1","g2"]}}{more}],
        "matrices":[{{"id":"M","stratification":"group",
        "values":{{"g1":{{"g1":2.0,"g2":1.0}},"g2":{{"g1":1.0,"g2":3.0}}}}}}],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"{rate}"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":900.0,"B":100.0}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

/// The rate and whether it is proportional, of each flow of `model`.
fn flows(model: &Model) -> Vec<(String, String, bool)> {
    describe_flows(&model.process, None)
        .unwrap()
        .into_iter()
        .map(|flow| {
            (
                flow.source.unwrap(),
                flow.rate,
                flow.proportional_to.is_some(),
            )
        })
        .collect()
}

#[test]
fn a_matrix_gives_its_entries_and_its_radius() {
    let names: Vec<(String, f64)> = matrix_variables(&model("k1", "").process)
        .into_iter()
        .map(
            |MatrixVariable {
                 name,
                 matrix,
                 value,
                 radius,
             }| {
                assert_eq!(radius, name == "M_radius");
                assert_eq!(matrix, "M");
                (name, value)
            },
        )
        .collect();

    assert_eq!(
        names,
        [
            ("M_g1_g1".to_string(), 2.0),
            ("M_g1_g2".to_string(), 1.0),
            ("M_g2_g1".to_string(), 1.0),
            ("M_g2_g2".to_string(), 3.0),
            ("M_radius".to_string(), names[4].1),
        ]
    );
    assert!((names[4].1 - (5.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
}

#[test]
fn a_matrix_sum_is_the_sum_over_the_categories_of_the_compartment() {
    let model = model("k1 * matrix_sum(M, B / total)", "");

    assert_eq!(
        flows(&model),
        [
            (
                "A_g1".to_string(),
                "k1 * (M_g1_g1 * (B_g1 / total_g1) + M_g1_g2 * (B_g2 / total_g2))".to_string(),
                true
            ),
            (
                "A_g2".to_string(),
                "k1 * (M_g2_g1 * (B_g1 / total_g1) + M_g2_g2 * (B_g2 / total_g2))".to_string(),
                true
            ),
        ]
    );
}

#[test]
fn a_split_category_reads_the_sum_of_its_compartments() {
    let model = model(
        "matrix_sum(M, B + k1)",
        r#",{"id":"sub","categories":["s1","s2"]}"#,
    );

    // `B` in `g1` is the partial bin total `B_g1`; `k1` is no bin.
    assert_eq!(
        flows(&model)[0],
        (
            "A_g1_s1".to_string(),
            "M_g1_g1 * (B_g1 + k1) + M_g1_g2 * (B_g2 + k1)".to_string(),
            true
        )
    );
}

#[test]
fn a_set_of_compartments_that_no_subgroup_total_sums_alone_reads_as_their_sum() {
    // `total_g_s` sums the compartments of `g` and `s` with those of `g_s`,
    // so the total of `g_s` is the sum of its compartments.
    let mut model = model(
        "matrix_sum(M, total)",
        r#",{"id":"sub","categories":["s","t"]}"#,
    );
    model.process.stratifications[0].categories = vec!["g".to_string(), "g_s".to_string()];
    model.process.matrices[0].values = [
        (
            "g".to_string(),
            [("g".to_string(), 0.0), ("g_s".to_string(), 1.0)].into(),
        ),
        (
            "g_s".to_string(),
            [("g".to_string(), 0.0), ("g_s".to_string(), 0.0)].into(),
        ),
    ]
    .into();

    assert_eq!(
        flows(&model)[0].1,
        "M_g_g_s * (A_g_s_s + A_g_s_t + B_g_s_s + B_g_s_t)"
    );
}

#[test]
fn a_term_of_a_zero_entry_is_left_out_and_a_zero_row_keeps_its_first_term() {
    let mut model = model("matrix_sum(M, B)", "");
    model.process.matrices[0].values["g1"]["g2"] = 0.0;
    model.process.matrices[0].values["g2"]["g1"] = 0.0;
    model.process.matrices[0].values["g2"]["g2"] = 0.0;

    let rates: Vec<String> = flows(&model).into_iter().map(|flow| flow.1).collect();

    // The row of `g2` is 0: its first term keeps the unit of the rate.
    assert_eq!(rates, ["M_g1_g1 * B_g1", "M_g2_g1 * B_g1"]);
}

#[test]
fn a_rate_that_reads_a_compartment_outside_its_matrix_sum_is_absolute() {
    let model = model("k1 * A * matrix_sum(M, B / total)", "");

    let (_, rate, proportional) = &flows(&model)[0];

    assert_eq!(
        rate,
        "k1 * A * (M_g1_g1 * (B_g1 / total_g1) + M_g1_g2 * (B_g2 / total_g2))"
    );
    assert!(!proportional);
}

#[test]
fn a_matrix_sum_runs_as_the_sum_it_is() {
    let rate = "k1 * matrix_sum(M, B / total)";
    let mut compiled = CompiledModel::try_from_model(&model(rate, "")).unwrap();
    // A_g1, A_g2, B_g1, B_g2.
    let state = [400.0, 500.0, 60.0, 40.0];
    let mut rates = vec![0.0; 2];

    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();

    let (b1, b2) = (60.0 / 460.0, 40.0 / 540.0);
    assert_eq!(
        rates,
        [0.3 * (2.0 * b1 + 1.0 * b2), 0.3 * (1.0 * b1 + 3.0 * b2)]
    );
}

/// The message of the build of `model`.
fn rejection(model: &Model) -> String {
    CompiledModel::try_from_model(model).err().unwrap()
}

#[test]
fn a_matrix_sum_of_an_unknown_matrix_is_rejected() {
    let model = model("matrix_sum(N, B)", "");

    assert_eq!(
        rejection(&model),
        "Transition 'ab' reads matrix_sum(N, …), but 'N' is not a matrix of the process; \
         its matrices are [\"M\"]"
    );
    assert_eq!(
        describe_flows(&model.process, None).unwrap_err(),
        "Transition 'ab' reads the matrix 'N', which the process does not have"
    );
}

#[test]
fn a_compartment_without_a_category_of_the_matrix_is_rejected() {
    let mut model = model("k1", "");
    model.process.stratifications[0].id = "other".to_string();
    model
        .process
        .stratifications
        .push(commol_core::Stratification {
            id: "group".to_string(),
            categories: vec!["g1".to_string(), "g2".to_string()],
            conditions: Some(vec![commol_core::StratificationCondition {
                stratification: "other".to_string(),
                category: Some("o1".to_string()),
                to: None,
            }]),
        });
    model.process.stratifications[0].categories = vec!["o1".to_string(), "o2".to_string()];
    model.process.transitions[0].rate = Some("matrix_sum(M, B)".to_string());

    assert_eq!(
        rejection(&model),
        "Transition 'ab' reads matrix_sum(M, …) from 'A_o2', which has no category of \
         'group', the stratification of the matrix"
    );
}

#[test]
fn a_formula_parameter_does_not_read_a_matrix_sum() {
    let mut model = model("k1", "");
    model.parameters[0].value = Some(ParameterValue::Formula("matrix_sum(M, B)".to_string()));

    assert_eq!(
        rejection(&model),
        "Parameter 'k1' reads matrix_sum, which reads the category of the compartment of a \
         rate; only the rate of a transition reads it"
    );
}

#[test]
fn a_matrix_sum_does_not_read_the_neighbors() {
    let mut model = model("matrix_sum(M, neighbors_B)", "");
    model.aggregate_mut().unwrap().neighbors = Some(Neighbors::Complete);

    assert_eq!(
        rejection(&model),
        "Transition 'ab' reads 'neighbors_B' inside matrix_sum(M, …); the expression of a \
         matrix sum reads bins, totals and parameters, not the neighbors of a unit"
    );
}

#[test]
fn two_matrices_of_one_id_are_rejected() {
    let mut model = model("k1", "");
    let matrix = model.process.matrices[0].clone();
    model.process.matrices.push(matrix);

    assert_eq!(rejection(&model), "Two matrices have the id 'M'");
}

#[test]
fn a_matrix_that_does_not_fit_its_stratification_is_rejected() {
    let mut model = model("k1", "");
    model.process.matrices[0].values["g2"].shift_remove("g1");

    assert_eq!(
        check_model(&model).unwrap_err(),
        "Matrix 'M' has no value in row 'g2' for the category 'g1' of 'group'; it needs \
         one value for each pair of its categories ([\"g1\", \"g2\"])"
    );
}

#[test]
fn a_variable_of_a_matrix_is_not_the_name_of_another() {
    for (id, kind) in [("M_radius", "a parameter"), ("M_g1_g2", "a parameter")] {
        let mut model = model("k1", "");
        model.parameters[0].id = id.to_string();
        model.process.transitions[0].rate = Some(id.to_string());

        assert_eq!(
            rejection(&model),
            format!(
                "Matrix 'M' gives the variable '{id}', which is also the name of {kind}; \
                 rename one of them"
            )
        );
    }
    let mut model = model("k1", "");
    model.process.bins[1].id = "M_radius".to_string();
    model.process.transitions[0].target = vec!["M_radius".to_string()];
    model
        .aggregate_mut()
        .unwrap()
        .initial_conditions
        .bin_amounts = [
        ("A".to_string(), Some(900.0)),
        ("M_radius".to_string(), Some(100.0)),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        rejection(&model),
        "Matrix 'M' gives the variable 'M_radius', which is also the name of a bin; rename \
         one of them"
    );
}

#[test]
fn the_id_of_a_matrix_is_not_the_name_of_another() {
    let mut model = model("k1", "");
    model.process.matrices[0].id = "k1".to_string();

    assert_eq!(
        rejection(&model),
        "Matrix 'k1' has the name of a parameter; rename one of them"
    );
}

#[test]
fn a_matrix_sum_inside_any_node_is_written_out_in_place() {
    let model = model("if(step > 0, -max(matrix_sum(M, 2 * B), 1) ^ 2, 0)", "");

    assert_eq!(
        flows(&model)[0].1,
        "if(step > 0, -max(M_g1_g1 * (2 * B_g1) + M_g1_g2 * (2 * B_g2), 1) ^ 2, 0)"
    );
}

#[test]
fn two_matrices_that_give_one_variable_are_rejected() {
    // `M` on `x` and `y_z` and `M_x` on `y` and `z` both give `M_x_y_z`.
    let mut model = model(
        "k1",
        r#",{"id":"left","categories":["x","y_z"]},{"id":"right","categories":["y","z"]}"#,
    );
    let mut first = model.process.matrices[0].clone();
    first.stratification = "left".to_string();
    first.values = serde_json::from_str(r#"{"x":{"x":1,"y_z":1},"y_z":{"x":1,"y_z":1}}"#).unwrap();
    let mut second = first.clone();
    second.id = "M_x".to_string();
    second.stratification = "right".to_string();
    second.values = serde_json::from_str(r#"{"y":{"y":1,"z":1},"z":{"y":1,"z":1}}"#).unwrap();
    model.process.matrices = vec![first, second];

    assert_eq!(
        rejection(&model),
        "Matrices 'M' and 'M_x' both give the variable 'M_x_y_z'; rename one of them"
    );
}
