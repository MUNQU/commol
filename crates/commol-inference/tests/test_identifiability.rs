//! Structural rank, collinearity indices and the identifiability of a sample.

use commol_calibration::{CalibrationParameter, CalibrationProblem, LossConfig, ObservedDataPoint};
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_inference::ParameterSample;
use commol_inference::identifiability::{
    IdentifiabilityError, StructuralVerdict, collinearity_index, sample_identifiability,
    structural_rank,
};
use commol_inference::sensitivity::{Design, FiniteDifferences, Matrix};
use commol_random::Generator;

/// A → B at the rate `rate`, from A = 1000, with the parameters `ids`.
fn decay(rate: &str, ids: &[&str]) -> DifferenceEquations {
    let parameters: Vec<String> = ids
        .iter()
        .map(|id| format!(r#"{{"id":"{id}","value":0.1}}"#))
        .collect();
    let json = format!(
        r#"{{"name":"m","parameters":[{}],
        "engine":{{"type":"DifferenceEquations"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"{rate}",
        "stratified_rates":null,"per_compartment":null}}]}},
        "structure":{{"kind":"aggregate","total":1000,"bin_fractions":{{"A":1.0,"B":0.0}}}}}}"#,
        parameters.join(",")
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    DifferenceEquations::try_from_model(&model).unwrap()
}

fn problem(
    engine: DifferenceEquations,
    bounds: &[(&str, f64, f64)],
) -> CalibrationProblem<DifferenceEquations> {
    CalibrationProblem::new(
        engine,
        (1..=4)
            .map(|t| ObservedDataPoint::new(f64::from(t), "A".to_string(), 500.0))
            .collect(),
        bounds
            .iter()
            .map(|&(id, min, max)| CalibrationParameter::new(id.to_string(), min, max))
            .collect(),
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

#[test]
fn an_unread_parameter_is_non_identifiable() {
    let problem = problem(
        decay("k", &["k", "u"]),
        &[("k", 0.05, 0.5), ("u", 0.0, 1.0)],
    );
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let analysis = structural_rank(&source, 5, 7).unwrap();

    assert_eq!(analysis.generic_rank, 1);
    assert_eq!(
        analysis.verdicts,
        [
            StructuralVerdict::Identifiable,
            StructuralVerdict::NonIdentifiable
        ]
    );
    assert!(analysis.combinations.is_empty());
    assert_eq!(analysis.points.len(), 5);
}

#[test]
fn a_product_is_identifiable_in_combination() {
    let problem = problem(
        decay("k * s", &["k", "s"]),
        &[("k", 0.1, 0.5), ("s", 0.2, 1.0)],
    );
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let analysis = structural_rank(&source, 5, 7).unwrap();

    assert_eq!(analysis.generic_rank, 1);
    assert_eq!(
        analysis.verdicts,
        [
            StructuralVerdict::InCombination,
            StructuralVerdict::InCombination
        ]
    );
    assert_eq!(analysis.combinations.len(), 1);
    assert_eq!(analysis.combinations[0].exponents, [(0, 1.0), (1, 1.0)]);
}

#[test]
fn every_point_infeasible_raises_with_the_count() {
    // Above k = 1 a step leaves A below zero.
    let problem = problem(decay("k", &["k"]), &[("k", 1.5, 2.0)]);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let error = structural_rank(&source, 4, 7).unwrap_err();

    assert!(
        matches!(
            &error,
            IdentifiabilityError::NoFeasiblePoint { points: 4, .. }
        ),
        "{error:?}"
    );
    let message = error.to_string();
    assert!(message.contains("4 points"), "{message}");
    assert!(
        message.contains("leaves the values the model allows"),
        "{message}"
    );
}

#[test]
fn proportional_columns_give_an_infinite_index() {
    let scaled = Matrix::from_row_slice(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);

    assert_eq!(collinearity_index(&scaled, &[0, 1]), f64::INFINITY);
}

#[test]
fn orthogonal_columns_give_index_one() {
    let scaled = Matrix::from_row_slice(3, 3, &[2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.5]);

    for subset in [&[0, 1][..], &[1, 2], &[0, 1, 2]] {
        assert!((collinearity_index(&scaled, subset) - 1.0).abs() < 1e-12);
    }
}

fn sample(columns: &[Vec<f64>]) -> ParameterSample {
    let draws = columns[0].len();
    let values: Vec<Vec<f64>> = (0..draws)
        .map(|i| columns.iter().map(|column| column[i]).collect())
        .collect();
    ParameterSample::new(
        (0..columns.len()).map(|j| format!("p{j}")).collect(),
        vec![(0.0, 1.0); columns.len()],
        values,
        &vec![0.0; draws],
        None,
    )
    .unwrap()
}

/// `n` standard normal draws (Box–Muller).
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut generator = Generator::from_seed(seed);
    (0..n)
        .map(|_| {
            let (u, v) = (generator.uniform_positive(), generator.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        })
        .collect()
}

#[test]
fn contraction_of_a_spread_and_a_tight_sample() {
    let n = 1000;
    let spread: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let tight: Vec<f64> = normals(n, 1).iter().map(|z| 0.5 + 0.001 * z).collect();
    let result = sample_identifiability(&sample(&[spread, tight]), 0.9).unwrap();

    assert!(
        result.contraction[0].abs() < 0.01,
        "{:?}",
        result.contraction
    );
    assert!(result.contraction[1] > 0.99, "{:?}", result.contraction);
}

#[test]
fn a_ridge_gives_a_correlated_pair() {
    let a: Vec<f64> = normals(500, 2).iter().map(|z| 0.5 + 0.1 * z).collect();
    let b: Vec<f64> = a.iter().map(|x| 1.0 - x).collect();
    let c: Vec<f64> = normals(500, 3).iter().map(|z| 0.5 + 0.1 * z).collect();
    let result = sample_identifiability(&sample(&[a, b, c]), 0.9).unwrap();

    assert!((result.correlation[0][1] + 1.0).abs() < 1e-9);
    assert_eq!(result.correlated_pairs.len(), 1);
    assert_eq!(
        (result.correlated_pairs[0].0, result.correlated_pairs[0].1),
        (0, 1)
    );
}

#[test]
fn a_bimodal_marginal_is_flagged() {
    let z = normals(400, 4);
    let bimodal: Vec<f64> = z
        .iter()
        .enumerate()
        .map(|(i, z)| if i % 2 == 0 { 0.2 } else { 0.8 } + 0.03 * z)
        .collect();
    let unimodal: Vec<f64> = normals(400, 5).iter().map(|z| 0.5 + 0.1 * z).collect();
    let result = sample_identifiability(&sample(&[bimodal, unimodal]), 0.9).unwrap();

    assert_eq!(result.multimodal, [true, false]);
}
