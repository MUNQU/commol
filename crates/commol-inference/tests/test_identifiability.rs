//! Structural rank, collinearity indices and the identifiability of a sample.

use commol_calibration::{CalibrationParameter, CalibrationProblem, LossConfig, ObservedDataPoint};
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_inference::ParameterSample;
use commol_inference::identifiability::{
    IdentifiabilityError, Reference, StructuralVerdict, collinearity, collinearity_index,
    sample_identifiability, structural_rank,
};
use commol_inference::sensitivity::{
    Design, FiniteDifferences, Matrix, Sensitivities, SensitivityError, SensitivitySource,
};
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
    let result = sample_identifiability(&sample(&[spread, tight]), Reference::Bounds, 0.9).unwrap();

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
    let result = sample_identifiability(&sample(&[a, b, c]), Reference::Bounds, 0.9).unwrap();

    assert!((result.correlation[0][1] + 1.0).abs() < 1e-9);
    assert_eq!(result.correlated_pairs.len(), 1);
    assert_eq!(
        (result.correlated_pairs[0].0, result.correlated_pairs[0].1),
        (0, 1)
    );
}

#[test]
fn a_correlation_equal_to_the_threshold_is_not_above_it() {
    // Centered columns with Σxy = 1 and Σx² = Σy² = 2: r = 1/2.
    let x = vec![0.75, 0.25, 0.5, 0.5];
    let y = vec![0.75, 0.5, 0.25, 0.5];
    let result = sample_identifiability(&sample(&[x, y]), Reference::Bounds, 0.5).unwrap();

    assert_eq!(result.correlation[0][1], 0.5);
    assert!(result.correlated_pairs.is_empty());
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
    let result =
        sample_identifiability(&sample(&[bimodal, unimodal]), Reference::Bounds, 0.9).unwrap();

    assert_eq!(result.multimodal, [true, false]);
}

/// A source of fixed scaled sensitivities: point `[i, …]` reads
/// `jacobians[i]`; `i = −1` is an infeasible point and `i = −2` a failure.
struct Fixed {
    parameters: Vec<CalibrationParameter>,
    jacobians: Vec<Matrix>,
}

impl Fixed {
    fn new(jacobians: Vec<Matrix>) -> Self {
        let columns = jacobians[0].ncols();
        Self {
            parameters: (0..columns)
                .map(|j| CalibrationParameter::new(format!("k{j}"), 0.0, 1.0))
                .collect(),
            jacobians,
        }
    }
}

impl SensitivitySource for Fixed {
    fn parameters(&self) -> &[CalibrationParameter] {
        &self.parameters
    }

    fn jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError> {
        match values[0] {
            -1.0 => Err(SensitivityError::InfeasiblePoint("outside".to_string())),
            -2.0 => Err(SensitivityError::Failed("broken".to_string())),
            index => {
                let jacobian = self.jacobians[index as usize].clone();
                Ok(Sensitivities {
                    predictions: vec![0.0; jacobian.nrows()],
                    errors: Matrix::zeros(jacobian.nrows(), jacobian.ncols()),
                    labels: vec!["A".to_string(); jacobian.nrows()],
                    jacobian,
                })
            }
        }
    }

    fn scales(&self, sensitivities: &Sensitivities) -> Vec<f64> {
        vec![1.0; sensitivities.jacobian.nrows()]
    }

    fn scaled_jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError> {
        self.jacobian(values)
    }
}

#[test]
fn a_singular_value_at_the_error_of_the_sensitivities_is_not_counted() {
    // Exact sensitivities: the zero singular value equals the zero error.
    let jacobian = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
    let analysis = structural_rank(&Fixed::new(vec![jacobian]), 3, 7).unwrap();

    assert_eq!(analysis.generic_rank, 1);
    assert_eq!(
        analysis.verdicts,
        [
            StructuralVerdict::Identifiable,
            StructuralVerdict::NonIdentifiable
        ]
    );
}

#[test]
fn a_failure_of_the_sensitivities_stops_the_structural_rank() {
    let source = Fixed {
        parameters: vec![CalibrationParameter::new("k0".to_string(), -2.0, -2.0)],
        jacobians: vec![],
    };
    let error = structural_rank(&source, 3, 7).unwrap_err();

    assert!(
        matches!(&error, IdentifiabilityError::Sensitivity(SensitivityError::Failed(m)) if m == "broken"),
        "{error:?}"
    );
}

/// Columns at 60° (k0, k1) and a third orthogonal to both (k2): the pair
/// has `γ = 1 / √(1 − cos 60°) = √2`, the others 1.
fn sixty_degrees() -> Matrix {
    let sine = 0.75f64.sqrt();
    Matrix::from_row_slice(
        4,
        3,
        &[1.0, 0.5, 0.0, 0.0, sine, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    )
}

#[test]
fn collinearity_reads_the_indices_and_sensitivities_of_known_columns() {
    let source = Fixed::new(vec![sixty_degrees()]);

    // One point of weight 3: its share is the whole weight.
    let result = collinearity(&source, &[vec![0.0; 3]], &[3.0], 2, 1.2).unwrap();

    assert!((result.pair_gamma[0][1] - 2f64.sqrt()).abs() < 1e-12);
    assert!((result.pair_gamma[0][2] - 1.0).abs() < 1e-12);
    // δ_j = ‖S_·j‖ / √4.
    for (actual, expected) in result.sensitivity.iter().zip([0.5, 0.5, 1.0]) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "{:?}",
            result.sensitivity
        );
    }
    let admissible: Vec<_> = result.admissible.iter().map(|s| &s.parameters).collect();
    assert_eq!(admissible, [&[0, 2], &[1, 2]]);
    assert!(result.admissible.iter().all(|s| s.share_below == 1.0));
    // Only sizes up to max_subset_size.
    assert_eq!(result.best_by_size.len(), 1);
    assert_eq!(result.best_by_size[0].parameters, [0, 2]);
}

#[test]
fn an_index_equal_to_the_threshold_is_not_below_it() {
    let source = Fixed::new(vec![sixty_degrees()]);

    let result = collinearity(&source, &[vec![0.0; 3]], &[1.0], 2, 1.0).unwrap();

    assert_eq!(result.best_by_size[0].gamma, 1.0);
    assert_eq!(result.best_by_size[0].share_below, 0.0);
    assert!(result.admissible.is_empty(), "{:?}", result.admissible);
}

#[test]
fn single_parameters_are_admissible_when_no_pair_is() {
    let proportional = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 6.0]);
    let source = Fixed::new(vec![proportional]);

    let result = collinearity(&source, &[vec![0.0; 2]], &[1.0], 2, 10.0).unwrap();

    let admissible: Vec<_> = result.admissible.iter().map(|s| &s.parameters).collect();
    assert_eq!(admissible, [&[0], &[1]]);
}

#[test]
fn collinearity_skips_an_infeasible_point_and_stops_at_a_failure() {
    let source = Fixed::new(vec![sixty_degrees()]);

    let result =
        collinearity(&source, &[vec![0.0; 3], vec![-1.0; 3]], &[1.0, 1.0], 2, 1.2).unwrap();
    assert_eq!((result.points, result.skipped.len()), (1, 1));

    let error =
        collinearity(&source, &[vec![0.0; 3], vec![-2.0; 3]], &[1.0, 1.0], 2, 1.2).unwrap_err();
    assert!(error.to_string().contains("broken"), "{error}");
}

#[test]
fn collinearity_checks_its_weights_and_subset_size() {
    let source = Fixed::new(vec![sixty_degrees()]);
    let point = [vec![0.0; 3]];

    for (weights, size) in [(&[1.0, 1.0][..], 2), (&[1.0], 1), (&[1.0], 4)] {
        let error = collinearity(&source, &point, weights, size, 1.2).unwrap_err();
        assert!(
            matches!(error, IdentifiabilityError::Invalid(_)),
            "{weights:?} {size}: {error}"
        );
    }
}

#[test]
fn a_correlation_threshold_is_strictly_between_zero_and_one() {
    let spread = |seed| normals(50, seed).iter().map(|z| 0.5 + 0.1 * z).collect();
    let draws = sample(&[spread(6), spread(7)]);

    assert!(sample_identifiability(&draws, Reference::Bounds, 0.5).is_ok());
    for threshold in [0.0, 1.0] {
        let error = sample_identifiability(&draws, Reference::Bounds, threshold).unwrap_err();
        assert!(matches!(error, IdentifiabilityError::Invalid(_)), "{error}");
    }
}

#[test]
fn a_greedy_search_extends_the_best_subset_past_the_exhaustive_sizes() {
    // Seven orthogonal columns: every subset has index 1.
    let source = Fixed::new(vec![Matrix::identity(7, 7)]);

    let result = collinearity(&source, &[vec![0.0; 7]], &[1.0], 7, 1.5).unwrap();

    let sizes: Vec<usize> = result
        .best_by_size
        .iter()
        .map(|s| s.parameters.len())
        .collect();
    assert_eq!(sizes, [2, 3, 4, 5, 6, 7]);
    assert_eq!(result.admissible[0].parameters, [0, 1, 2, 3, 4, 5, 6]);
    // At a threshold of 1, an index of 1 is not below it at any size.
    let at_one = collinearity(&source, &[vec![0.0; 7]], &[1.0], 7, 1.0).unwrap();
    assert!(
        at_one.admissible.iter().all(|s| s.parameters.len() == 1),
        "{:?}",
        at_one.admissible
    );
}

#[test]
fn a_structural_rank_needs_a_point_and_collinearity_a_feasible_one() {
    let jacobian = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let source = Fixed::new(vec![jacobian]);

    let no_points = structural_rank(&source, 0, 7).unwrap_err();
    let infeasible = collinearity(
        &source,
        &[vec![-1.0, 0.0], vec![-1.0, 0.0]],
        &[1.0, 1.0],
        2,
        20.0,
    )
    .unwrap_err();

    assert_eq!(
        no_points,
        IdentifiabilityError::Invalid("A structural rank needs at least one point".to_string())
    );
    assert!(
        matches!(&infeasible, IdentifiabilityError::NoFeasiblePoint { points: 2, reasons }
            if reasons.len() == 2),
        "{infeasible:?}"
    );
}
