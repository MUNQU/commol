//! A matrix between the categories of a stratification: its variables, the
//! matrix sums of a rate as the flows hold them, and what a build rejects.

use commol_core::{
    Model, Neighbors, Parameter, ParameterValue, SeriesMode, StratificationFractions,
};
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

/// `model` with an even split of every stratification, whatever its
/// stratifications and categories after a test changes them.
fn with_even_fractions(model: &Model) -> Model {
    let mut model = model.clone();
    let fractions = model
        .process
        .stratifications
        .iter()
        .map(|stratification| {
            let share = 1.0 / stratification.categories.len() as f64;
            let split = stratification
                .categories
                .iter()
                .map(|category| (category.clone(), share))
                .collect();
            (
                stratification.id.clone(),
                StratificationFractions::Split(split),
            )
        })
        .collect();
    model
        .aggregate_mut()
        .unwrap()
        .initial_conditions
        .stratification_fractions = fractions;
    model
}

/// The compiled `model`, with an even split of every stratification.
fn compile(model: &Model) -> Result<CompiledModel, String> {
    CompiledModel::try_from_model(&with_even_fractions(model))
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
    let names: Vec<(String, Option<f64>)> = matrix_variables(&model("k1", ""))
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
            ("M_g1_g1".to_string(), Some(2.0)),
            ("M_g1_g2".to_string(), Some(1.0)),
            ("M_g2_g1".to_string(), Some(1.0)),
            ("M_g2_g2".to_string(), Some(3.0)),
            ("M_radius".to_string(), names[4].1),
        ]
    );
    assert!((names[4].1.unwrap() - (5.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
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

    // The row of `g2` is 0: the term of its first column, under a condition
    // that is false, keeps the unit of the rate.
    assert_eq!(
        rates,
        ["M_g1_g1 * B_g1", "if(M_g2_g1 != 0, M_g2_g1 * B_g1, 0)"]
    );
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
    let mut compiled = compile(&model(rate, "")).unwrap();
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
    compile(model).err().unwrap()
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
                categories: None,
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
        check_model(&with_even_fractions(&model)).unwrap_err(),
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

#[test]
fn a_row_of_zeros_is_zero_where_its_first_category_is_empty() {
    let mut model = model("k1 * matrix_sum(M, B / total)", "");
    let values = &mut model.process.matrices[0].values;
    values["g1"]["g1"] = 0.0;
    values["g2"]["g1"] = 0.0;
    values["g2"]["g2"] = 0.0;
    let mut compiled = compile(&model).unwrap();
    // A_g1, A_g2, B_g1, B_g2: `g1` is empty, so `B_g1 / total_g1` is NaN.
    let state = [0.0, 500.0, 0.0, 40.0];
    let mut rates = vec![0.0; 2];

    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();

    assert!((rates[0] - 0.3 * 40.0 / 540.0).abs() < 1e-15);
    assert_eq!(rates[1], 0.0);
}

#[test]
fn a_variable_of_a_matrix_is_not_the_id_of_another() {
    let mut model = model("k1", "");
    let mut other = model.process.matrices[0].clone();
    other.id = "M_radius".to_string();
    model.process.matrices.push(other);

    assert_eq!(
        rejection(&model),
        "Matrix 'M' gives the variable 'M_radius', which is also the id of a matrix; rename \
         one of them"
    );
}

/// `model` with the rate `rate`, the row of `g1` of `M` scaled by `a1` of
/// value `a1` (`None` to calibrate) and the column of `g2` by `b2` at 1.
fn scaled(rate: &str, a1: Option<f64>) -> Model {
    let mut model = model(rate, "");
    for (id, value) in [("a1", a1), ("b2", Some(1.0))] {
        model.parameters.push(Parameter {
            id: id.to_string(),
            value: value.map(ParameterValue::Constant),
            description: None,
        });
    }
    let matrix = &mut model.process.matrices[0];
    matrix.row_scales.insert("g1".to_string(), "a1".to_string());
    matrix
        .column_scales
        .insert("g2".to_string(), "b2".to_string());
    model
}

/// The value of each variable of the matrices of `model`, by name.
fn values(model: &Model) -> Vec<(String, Option<f64>)> {
    matrix_variables(model)
        .into_iter()
        .map(|variable| (variable.name, variable.value))
        .collect()
}

#[test]
fn a_scale_multiplies_the_entries_and_the_radius_follows_it() {
    let rate = "k1 * matrix_sum(M, B / total)";
    let model = scaled(rate, Some(2.0));
    let radius = |a1: f64| {
        // The radius of [[2 a1, a1], [1, 3]].
        let (trace, det) = (2.0 * a1 + 3.0, 6.0 * a1 - a1);
        (trace + (trace * trace - 4.0 * det).sqrt()) / 2.0
    };
    assert_eq!(
        values(&model)[..4],
        [
            ("M_g1_g1".to_string(), Some(4.0)),
            ("M_g1_g2".to_string(), Some(2.0)),
            ("M_g2_g1".to_string(), Some(1.0)),
            ("M_g2_g2".to_string(), Some(3.0)),
        ]
    );
    assert!((values(&model)[4].1.unwrap() - radius(2.0)).abs() < 1e-12);

    let mut compiled = compile(&model).unwrap();
    let state = [400.0, 500.0, 60.0, 40.0];
    let (b1, b2) = (60.0 / 460.0, 40.0 / 540.0);
    let mut rates = vec![0.0; 2];
    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();
    assert_eq!(
        rates,
        [0.3 * (4.0 * b1 + 2.0 * b2), 0.3 * (1.0 * b1 + 3.0 * b2)]
    );

    // Setting the scale writes the entries and the radius again.
    compiled.set_parameter("a1", 1.0).unwrap();
    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();
    assert_eq!(
        rates,
        [0.3 * (2.0 * b1 + 1.0 * b2), 0.3 * (1.0 * b1 + 3.0 * b2)]
    );
    let radius_rate = scaled("M_radius", Some(2.0));
    let mut compiled = compile(&radius_rate).unwrap();
    compiled.set_parameter("a1", 1.0).unwrap();
    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();
    assert!((rates[0] - radius(1.0)).abs() < 1e-12);
    // A parameter that scales no matrix leaves the matrices as they are.
    compiled.set_parameter("k1", 0.5).unwrap();
    assert_eq!(
        compiled.matrices()[0].entries(|_| None),
        Ok(vec![2.0, 1.0, 1.0, 3.0])
    );
}

#[test]
fn a_scale_to_calibrate_gives_the_matrix_its_values_once_it_is_set() {
    let model = scaled("k1 * matrix_sum(M, B / total)", None);
    assert!(values(&model).iter().all(|(_, value)| value.is_none()));

    let mut compiled = compile(&model).unwrap();
    assert_eq!(
        compiled.matrices()[0]
            .entries(|_| None)
            .unwrap_err()
            .parameter,
        "a1"
    );
    compiled.set_parameter("a1", 2.0).unwrap();
    assert_eq!(
        compiled.matrices()[0].entries(|_| None),
        Ok(vec![4.0, 2.0, 1.0, 3.0])
    );
    let state = [400.0, 500.0, 60.0, 40.0];
    let (b1, b2) = (60.0 / 460.0, 40.0 / 540.0);
    let mut rates = vec![0.0; 2];
    compiled
        .transition_rates(0.0, 0.0, &state, &mut rates)
        .unwrap();
    assert_eq!(rates[0], 0.3 * (4.0 * b1 + 2.0 * b2));
}

#[test]
fn a_scale_is_a_constant_or_a_parameter_to_calibrate() {
    let mut unknown = scaled("k1", Some(2.0));
    unknown.process.matrices[0]
        .row_scales
        .insert("g2".to_string(), "a9".to_string());
    assert_eq!(
        check_model(&with_even_fractions(&unknown)).unwrap_err(),
        "Matrix 'M' scales the row of 'g2' by 'a9', which is not a parameter of the model; \
         a scale is a constant or a parameter to calibrate, which does not change during a run"
    );

    let mut formula = scaled("k1", Some(2.0));
    formula.parameters[1].value = Some(ParameterValue::Formula("2 * k1".to_string()));
    assert_eq!(
        rejection(&formula),
        "Matrix 'M' scales the row of 'g1' by 'a1', a formula parameter; a scale is a \
         constant or a parameter to calibrate, which does not change during a run"
    );

    let mut series = scaled("k1", Some(2.0));
    series.parameters[2].value = Some(ParameterValue::Series {
        data: vec![(0.0, 1.0), (1.0, 2.0)],
        mode: SeriesMode::Linear,
    });
    assert_eq!(
        check_model(&with_even_fractions(&series)).unwrap_err(),
        "Matrix 'M' scales the column of 'g2' by 'b2', a series parameter; a scale is a \
         constant or a parameter to calibrate, which does not change during a run"
    );
}
