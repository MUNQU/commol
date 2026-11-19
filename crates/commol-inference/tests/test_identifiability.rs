//! Structural rank, collinearity indices and the identifiability of a sample.

use commol_calibration::{
    CalibrationParameter, CalibrationParameterType, CalibrationProblem, LossConfig,
    ObservedDataPoint,
};
use commol_core::{Accuracy, AccuracySettings, Model};
use commol_difference::DifferenceEquations;
use commol_inference::identifiability::{
    PracticalVerdict, SampleSize, StructuralVerdict, Tightening, collinearity, collinearity_index,
    sample_identifiability, structural_rank,
};
use commol_inference::sensitivity::{
    Design, FiniteDifferences, Matrix, PartSensitivities, Sensitivities, SensitivityError,
    SensitivitySource,
};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use commol_core::json::Pairs;
use commol_inference::distribution::{BoundSide, Distribution, Family};
use commol_inference::settings::PriorsSettings;
use commol_inference::settings::identifiability::{Reference, SampleIdentifiabilitySettings};
use commol_inference::{MethodError, ParameterSample};
use commol_random::Generator;
use commol_random::sample::{gamma, normal};
use rayon::prelude::*;

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
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":1000.0,"B":0.0}}}}}}"#,
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
        None,
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
            Some(StructuralVerdict::Identifiable),
            Some(StructuralVerdict::NonIdentifiable)
        ]
    );
    assert!(analysis.combinations.is_empty());
    assert_eq!(analysis.points.len(), 5);
}

#[test]
fn an_exact_engine_is_not_evaluated_again() {
    let problem = problem(
        decay("k", &["k", "u"]),
        &[("k", 0.05, 0.5), ("u", 0.0, 1.0)],
    );
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let analysis = structural_rank(&source, 5, 7).unwrap();

    assert!(
        problem
            .forward()
            .at_accuracy(Accuracy {
                relative: 1e-9,
                absolute: 1e-9
            })
            .is_none()
    );
    assert_eq!(analysis.accuracy, Accuracy::EXACT);
    assert_eq!(analysis.tighter, None);
    assert_eq!(analysis.tighter_generic_rank, None);
    assert!(
        analysis
            .points
            .iter()
            .all(|point| point.rank == 1 && point.tighter.is_none())
    );
    assert_eq!(
        analysis.verdicts,
        [
            Some(StructuralVerdict::Identifiable),
            Some(StructuralVerdict::NonIdentifiable)
        ]
    );
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
            Some(StructuralVerdict::InCombination),
            Some(StructuralVerdict::InCombination)
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
        matches!(&error, MethodError::NoFeasiblePoint { points: 4, .. }),
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

/// The settings of a sample against its bounds, with `correlation_threshold`,
/// a contraction threshold of 0.5 and a bound share of 0.05.
fn bounds(correlation_threshold: f64) -> SampleIdentifiabilitySettings {
    SampleIdentifiabilitySettings {
        reference: Reference::Bounds,
        correlation_threshold,
        contraction_threshold: 0.5,
        bound_share: 0.05,
        log_parameters: vec![],
        priors: None,
    }
}

/// `n` standard normal draws.
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut generator = Generator::from_seed(seed);
    (0..n).map(|_| normal(&mut generator)).collect()
}

#[test]
fn contraction_of_a_spread_and_a_tight_sample() {
    let n = 1000;
    let spread: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let tight: Vec<f64> = normals(n, 1).iter().map(|z| 0.5 + 0.001 * z).collect();
    let result = sample_identifiability(&sample(&[spread, tight]), &bounds(0.9), 0).unwrap();

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
    let result = sample_identifiability(&sample(&[a, b, c]), &bounds(0.9), 0).unwrap();

    assert!((result.correlation[0][1] + 1.0).abs() < 1e-9);
    assert_eq!(result.correlated_pairs.len(), 1);
    assert_eq!(
        (result.correlated_pairs[0].0, result.correlated_pairs[0].1),
        (0, 1)
    );
}

#[test]
fn a_sample_gives_the_verdict_of_each_parameter() {
    let a: Vec<f64> = normals(500, 2).iter().map(|z| 0.5 + 0.1 * z).collect();
    let b: Vec<f64> = a.iter().map(|x| 1.0 - x).collect();
    let c: Vec<f64> = normals(500, 3).iter().map(|z| 0.5 + 0.1 * z).collect();
    let result = sample_identifiability(&sample(&[a, b, c]), &bounds(0.9), 0).unwrap();

    let r = result.correlation[0][1];
    let verdicts = &result.verdicts;
    assert_eq!(verdicts[0].verdict, PracticalVerdict::PoorlyDetermined);
    assert_eq!(verdicts[0].correlated_with, [(1, r)]);
    assert_eq!(verdicts[1].verdict, PracticalVerdict::PoorlyDetermined);
    assert_eq!(verdicts[1].correlated_with, [(0, r)]);
    assert_eq!(verdicts[2].verdict, PracticalVerdict::Determined);
    assert!(verdicts[2].correlated_with.is_empty());
    assert!(
        verdicts.iter().all(|verdict| verdict.modes == 1),
        "{verdicts:?}"
    );
}

#[test]
fn a_correlation_equal_to_the_threshold_is_not_above_it() {
    // Centered columns with Σxy = 1 and Σx² = Σy² = 2: r = 1/2.
    let x = vec![0.75, 0.25, 0.5, 0.5];
    let y = vec![0.75, 0.5, 0.25, 0.5];
    let result = sample_identifiability(&sample(&[x, y]), &bounds(0.5), 0).unwrap();

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
    let result = sample_identifiability(&sample(&[bimodal, unimodal]), &bounds(0.9), 0).unwrap();

    assert_eq!(result.verdicts[0].modes, 2);
    assert_eq!(result.verdicts[1].modes, 1);
    assert_eq!(
        result.verdicts[0].verdict,
        PracticalVerdict::PoorlyDetermined
    );
    assert_eq!(result.verdicts[1].verdict, PracticalVerdict::Determined);
}

/// A source of fixed scaled sensitivities: point `[i, …]` reads
/// `jacobians[i]`; `i = −1` is an infeasible point and `i = −2` a failure.
struct Fixed {
    parameters: Vec<CalibrationParameter>,
    jacobians: Vec<Matrix>,
}

impl Fixed {
    fn new(jacobians: Vec<Matrix>) -> Self {
        Self::with_noise(jacobians, &[])
    }

    /// The parameters `noise`, by index, are noise parameters.
    fn with_noise(jacobians: Vec<Matrix>, noise: &[usize]) -> Self {
        let columns = jacobians[0].ncols();
        Self {
            parameters: (0..columns)
                .map(|j| {
                    let kind = if noise.contains(&j) {
                        CalibrationParameterType::Noise
                    } else {
                        CalibrationParameterType::Parameter
                    };
                    CalibrationParameter::with_type(format!("k{j}"), kind, 0.0, 1.0)
                })
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
                let predictions = vec![0.0; jacobian.nrows()];
                let errors = Matrix::zeros(jacobian.nrows(), jacobian.ncols());
                Ok(Sensitivities {
                    parts: PartSensitivities::one_per_observation(&predictions, &jacobian, &errors),
                    predictions,
                    errors,
                    labels: vec!["A".to_string(); jacobian.nrows()],
                    jacobian,
                })
            }
        }
    }

    fn scales(&self, sensitivities: &Sensitivities) -> Vec<f64> {
        vec![1.0; sensitivities.jacobian.nrows()]
    }

    fn scaled_jacobian(
        &self,
        values: &[f64],
        columns: &[usize],
    ) -> Result<Sensitivities, SensitivityError> {
        self.jacobian_of(values, columns)
    }
}

#[test]
fn a_singular_value_at_the_error_of_the_sensitivities_is_not_counted() {
    // Exact sensitivities: the zero singular value equals the zero error.
    let jacobian = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
    let source = Fixed::new(vec![jacobian]);
    let analysis = structural_rank(&source, 3, 7).unwrap();

    // A source of exact runs has no other accuracy.
    assert!(source.at_accuracy(Accuracy::EXACT).is_none());
    assert_eq!(analysis.tighter, None);

    assert_eq!(analysis.generic_rank, 1);
    assert_eq!(
        analysis.verdicts,
        [
            Some(StructuralVerdict::Identifiable),
            Some(StructuralVerdict::NonIdentifiable)
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
        matches!(&error, MethodError::Sensitivity(SensitivityError::Failed(m)) if m == "broken"),
        "{error:?}"
    );
}

/// A source of the scaled sensitivities `diag(1, 10⁻³)` whose entries all
/// carry an error equal to the relative accuracy of its runs; at an
/// accuracy below `fails_below` every run fails. `settings` set its
/// accuracy.
#[derive(Clone)]
struct Tolerant {
    parameters: Vec<CalibrationParameter>,
    accuracy: Accuracy,
    fails_below: f64,
    settings: Option<AccuracySettings>,
}

const TOLERANCES: AccuracySettings = AccuracySettings {
    relative: "Solver.rtol",
    absolute: "Solver.atol",
};

impl Tolerant {
    fn new(relative: f64, fails_below: f64) -> Self {
        Self {
            parameters: vec![
                CalibrationParameter::new("k0".to_string(), 1.0, 2.0),
                CalibrationParameter::new("k1".to_string(), 1.0, 2.0),
            ],
            accuracy: Accuracy {
                relative,
                absolute: relative,
            },
            fails_below,
            settings: Some(TOLERANCES),
        }
    }
}

impl SensitivitySource for Tolerant {
    fn parameters(&self) -> &[CalibrationParameter] {
        &self.parameters
    }

    fn jacobian(&self, _values: &[f64]) -> Result<Sensitivities, SensitivityError> {
        if self.accuracy.relative < self.fails_below {
            return Err(SensitivityError::Failed("too stiff".to_string()));
        }
        let predictions = vec![0.0; 2];
        let jacobian = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1e-3]);
        let errors = Matrix::from_element(2, 2, self.accuracy.relative);
        Ok(Sensitivities {
            parts: PartSensitivities::one_per_observation(&predictions, &jacobian, &errors),
            predictions,
            jacobian,
            errors,
            labels: vec!["A".to_string(); 2],
        })
    }

    fn scales(&self, _sensitivities: &Sensitivities) -> Vec<f64> {
        vec![1.0; 2]
    }

    fn scaled_jacobian(
        &self,
        values: &[f64],
        columns: &[usize],
    ) -> Result<Sensitivities, SensitivityError> {
        self.jacobian_of(values, columns)
    }

    fn accuracy(&self) -> Accuracy {
        self.accuracy
    }

    fn accuracy_settings(&self) -> Option<AccuracySettings> {
        self.settings
    }

    fn at_accuracy(&self, accuracy: Accuracy) -> Option<Self> {
        Some(Self {
            accuracy,
            ..self.clone()
        })
    }
}

#[test]
fn a_rank_lost_to_the_accuracy_gives_no_verdict() {
    // ‖E‖_F = 2·10⁻³ hides the singular value 10⁻³; 2·10⁻⁵ does not.
    let analysis = structural_rank(&Tolerant::new(1e-3, 0.0), 3, 7).unwrap();

    let tighter = Accuracy {
        relative: 1e-5,
        absolute: 1e-5,
    };
    assert_eq!(
        analysis.tighter,
        Some(Tightening {
            accuracy: tighter,
            settings: TOLERANCES
        })
    );
    for point in &analysis.points {
        assert_eq!(point.rank, 1);
        let again = point.tighter.as_ref().expect("evaluated again");
        assert_eq!((again.accuracy, again.rank), (tighter, 2));
    }
    assert_eq!(analysis.generic_rank, 1);
    assert_eq!(analysis.tighter_generic_rank, Some(2));
    assert_eq!(
        analysis.verdicts,
        [Some(StructuralVerdict::Identifiable), None]
    );
    assert!(analysis.unchecked.is_empty());
}

#[test]
fn a_source_that_names_no_setting_of_its_accuracy_is_not_evaluated_again() {
    // It runs at other accuracies, but no setting sets them.
    let source = Tolerant {
        settings: None,
        ..Tolerant::new(1e-3, 0.0)
    };
    let analysis = structural_rank(&source, 3, 7).unwrap();

    assert_eq!(analysis.tighter, None);
    assert!(analysis.points.iter().all(|point| point.tighter.is_none()));
    assert_eq!(
        analysis.verdicts,
        [
            Some(StructuralVerdict::Identifiable),
            Some(StructuralVerdict::NonIdentifiable)
        ]
    );
}

#[test]
fn a_point_of_full_rank_is_not_evaluated_again() {
    let analysis = structural_rank(&Tolerant::new(1e-5, 0.0), 3, 7).unwrap();

    assert!(analysis.points.iter().all(|point| point.tighter.is_none()));
    assert_eq!(analysis.tighter_generic_rank, None);
    assert_eq!(
        analysis.verdicts,
        [Some(StructuralVerdict::Identifiable); 2]
    );
}

#[test]
fn the_tighter_accuracy_is_no_lower_than_the_tightest() {
    let near = structural_rank(&Tolerant::new(1e-13, 0.0), 1, 7).unwrap();
    let tightest = structural_rank(&Tolerant::new(1e-14, 0.0), 1, 7).unwrap();

    assert_eq!(
        near.tighter.map(|tighter| tighter.accuracy),
        Some(Accuracy {
            relative: 1e-14,
            absolute: 1e-14
        })
    );
    assert_eq!(tightest.tighter, None);
}

#[test]
fn a_point_that_fails_at_the_tighter_accuracy_leaves_the_rank_unchecked() {
    let analysis = structural_rank(&Tolerant::new(1e-3, 1e-4), 3, 7).unwrap();

    assert_eq!(analysis.unchecked, vec!["too stiff".to_string(); 3]);
    assert!(analysis.points.iter().all(|point| point.tighter.is_none()));
    assert_eq!(
        analysis.verdicts,
        [Some(StructuralVerdict::Identifiable), None]
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
fn collinearity_gives_the_verdict_of_each_parameter() {
    let sixty = Fixed::new(vec![sixty_degrees()]);
    // No observation is sensitive to k1: its column is zero.
    let insensitive = Fixed::new(vec![Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0])]);

    let collinear = collinearity(&sixty, &[vec![0.0; 3]], &[1.0], 2, 1.2).unwrap();
    let apart = collinearity(&sixty, &[vec![0.0; 3]], &[1.0], 2, 1.5).unwrap();
    let unread = collinearity(&insensitive, &[vec![0.0; 2]], &[1.0], 2, 1.5).unwrap();

    // The pair (k0, k1) has the index √2: at or above 1.2, below 1.5.
    let verdicts = &collinear.verdicts;
    assert_eq!(verdicts[0].verdict, PracticalVerdict::PoorlyDetermined);
    assert_eq!(verdicts[0].collinear_with, [1]);
    assert_eq!(verdicts[1].collinear_with, [0]);
    assert_eq!(verdicts[2].verdict, PracticalVerdict::Determined);
    assert!(verdicts[2].collinear_with.is_empty());
    assert!(
        apart
            .verdicts
            .iter()
            .all(|v| v.verdict == PracticalVerdict::Determined && !v.insensitive),
        "{:?}",
        apart.verdicts
    );
    let k1 = &unread.verdicts[1];
    assert_eq!(k1.verdict, PracticalVerdict::PoorlyDetermined);
    assert!(k1.insensitive);
    assert!(!unread.verdicts[0].insensitive);
}

#[test]
fn an_index_equal_to_the_threshold_is_not_below_it() {
    // One pair of columns at 60 degrees, whose index is the threshold.
    let sine = 0.75f64.sqrt();
    let pair = Matrix::from_row_slice(2, 2, &[1.0, 0.5, 0.0, sine]);
    let source = Fixed::new(vec![pair]);
    let gamma = collinearity(&source, &[vec![0.0; 2]], &[1.0], 2, 10.0)
        .unwrap()
        .best_by_size[0]
        .gamma;

    let result = collinearity(&source, &[vec![0.0; 2]], &[1.0], 2, gamma).unwrap();

    assert!(gamma > 1.0, "{gamma}");
    assert_eq!(result.best_by_size[0].share_below, 0.0);
    assert!(
        result.admissible.iter().all(|s| s.parameters.len() == 1),
        "{:?}",
        result.admissible
    );
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
            matches!(error, MethodError::Invalid(_)),
            "{weights:?} {size}: {error}"
        );
    }
}

#[test]
fn a_correlation_threshold_is_strictly_between_zero_and_one() {
    let spread = |seed| normals(50, seed).iter().map(|z| 0.5 + 0.1 * z).collect();
    let draws = sample(&[spread(6), spread(7)]);

    assert!(sample_identifiability(&draws, &bounds(0.5), 0).is_ok());
    for threshold in [0.0, 1.0] {
        let error = sample_identifiability(&draws, &bounds(threshold), 0).unwrap_err();
        assert!(matches!(error, MethodError::Invalid(_)), "{error}");
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
        MethodError::Invalid("points is a whole number of at least 1, got 0".to_string())
    );
    assert!(
        matches!(&infeasible, MethodError::NoFeasiblePoint { points: 2, reasons }
            if reasons.len() == 2),
        "{infeasible:?}"
    );
}

/// `k0` and `k2` read at 60 degrees, and `k1`, a noise parameter, read by
/// no observation.
fn sixty_degrees_and_noise() -> Fixed {
    let sine = 0.75f64.sqrt();
    let jacobian = Matrix::from_row_slice(2, 3, &[1.0, 0.0, 0.5, 0.0, 0.0, sine]);
    Fixed::with_noise(vec![jacobian], &[1])
}

#[test]
fn a_noise_parameter_is_left_out_of_the_structural_rank() {
    let analysis = structural_rank(&sixty_degrees_and_noise(), 3, 7).unwrap();

    assert_eq!(analysis.parameters, [0, 2]);
    assert_eq!(analysis.generic_rank, 2);
    assert!(
        analysis
            .points
            .iter()
            .all(|point| point.singular_values.len() == 2 && point.values.len() == 3)
    );
    assert_eq!(
        analysis.verdicts,
        [Some(StructuralVerdict::Identifiable); 2]
    );
    assert_eq!(analysis.projections.len(), 2);
}

#[test]
fn a_combination_names_the_parameters_by_their_index_in_the_problem() {
    // k0 and k2 read as one column; k1 is a noise parameter.
    let jacobian = Matrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let source = Fixed::with_noise(vec![jacobian], &[1]);

    let analysis = structural_rank(&source, 3, 7).unwrap();

    assert_eq!(analysis.generic_rank, 1);
    assert_eq!(analysis.combinations.len(), 1);
    assert_eq!(analysis.combinations[0].exponents, [(0, 1.0), (2, 1.0)]);
}

#[test]
fn a_rank_with_only_noise_parameters_is_an_error() {
    let jacobian = Matrix::zeros(2, 2);
    let source = Fixed::with_noise(vec![jacobian], &[0, 1]);

    let rank = structural_rank(&source, 3, 7).unwrap_err();
    let indices = collinearity(&source, &[vec![0.0; 2]], &[1.0], 2, 10.0).unwrap_err();

    for (error, method) in [(rank, "StructuralRank"), (indices, "Collinearity")] {
        assert_eq!(
            error,
            MethodError::Invalid(format!(
                "{method} has no parameter to differentiate: every calibrated parameter is a \
                 noise parameter of an observation model, which no prediction reads"
            ))
        );
    }
}

#[test]
fn a_noise_parameter_is_left_out_of_the_collinearity() {
    let result = collinearity(&sixty_degrees_and_noise(), &[vec![0.0; 3]], &[1.0], 2, 1.2).unwrap();

    assert_eq!(result.parameters, [0, 2]);
    assert_eq!(result.sensitivity.len(), 2);
    // The pair (k0, k2) has the index √2, at or above 1.2.
    assert!((result.pair_gamma[0][1] - 2f64.sqrt()).abs() < 1e-12);
    assert_eq!(result.best_by_size[0].parameters, [0, 2]);
    let admissible: Vec<_> = result.admissible.iter().map(|s| &s.parameters).collect();
    assert_eq!(admissible, [&[0], &[2]]);
    assert_eq!(result.verdicts.len(), 2);
    assert_eq!(result.verdicts[0].collinear_with, [2]);
    assert_eq!(result.verdicts[1].collinear_with, [0]);
    assert!(result.verdicts.iter().all(|v| !v.insensitive));
}

#[test]
fn a_subset_larger_than_the_differentiated_parameters_is_an_error() {
    let error =
        collinearity(&sixty_degrees_and_noise(), &[vec![0.0; 3]], &[1.0], 3, 10.0).unwrap_err();

    assert_eq!(
        error,
        MethodError::Invalid(
            "max_subset_size is a whole number from 2 to 2, got 3: 2 of the 3 parameters are \
             not noise parameters"
                .to_string()
        )
    );
}

#[test]
fn collinearity_needs_two_parameters_that_are_not_noise_parameters() {
    let jacobian = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
    let source = Fixed::with_noise(vec![jacobian], &[1]);

    let error = collinearity(&source, &[vec![0.0; 2]], &[1.0], 2, 10.0).unwrap_err();

    assert_eq!(
        error,
        MethodError::Invalid(
            "Collinearity needs at least 2 parameters that are not noise parameters, got 1"
                .to_string()
        )
    );
}

#[test]
fn the_priors_of_a_sample_identifiability_match_its_reference() {
    let read = |reference: &str, priors: &str| {
        commol_core::json::read::<SampleIdentifiabilitySettings>(&format!(
            r#"{{"reference": "{reference}", "correlation_threshold": 0.9,
            "contraction_threshold": 0.5, "bound_share": 0.05, "log_parameters": [],
            "priors": {priors}}}"#
        ))
    };
    let names = ["k".to_string()];

    assert!(
        read("prior", "null")
            .unwrap_err()
            .starts_with("reference='prior' needs priors")
    );
    assert!(
        read("bounds", r#"{"distributions": null}"#)
            .unwrap_err()
            .starts_with("priors are read only with reference='prior'")
    );
    let bounds = read("bounds", "null").unwrap();
    assert_eq!(bounds.reference_priors(&names, &[(0.0, 1.0)]), Ok(None));
    let prior = read("prior", r#"{"distributions": null}"#).unwrap();
    assert!(
        prior
            .reference_priors(&names, &[(0.0, 1.0)])
            .unwrap()
            .is_some()
    );
}

#[test]
fn a_measure_over_the_bounds_refuses_an_infinite_bound() {
    let problem = problem(decay("k", &["k"]), &[("k", 0.05, f64::INFINITY)]);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let half_line = ParameterSample::new(
        vec!["k".to_string()],
        vec![(0.0, f64::INFINITY)],
        vec![vec![0.2], vec![0.4], vec![0.3]],
        &[0.0; 3],
        None,
    )
    .unwrap();

    assert_eq!(
        structural_rank(&source, 3, 7).unwrap_err().to_string(),
        "StructuralRank evaluates points of a Latin hypercube over the bounds, so they must \
         be finite: parameter 'k' has the bounds [0.05, inf]. Give it finite bounds"
    );
    assert_eq!(
        sample_identifiability(&half_line, &bounds(0.9), 0)
            .unwrap_err()
            .to_string(),
        "reference='bounds' measures the sample against the uniform distribution on the \
         bounds, so they must be finite: parameter 'k' has the bounds [0, inf]. Give \
         reference='prior' with the priors of the sample"
    );
}

#[test]
fn the_structural_rank_reads_no_part() {
    use commol_calibration::{DiscrepancyConfig, ObservationModel};
    // One total of B over four steps, in four parts: the total reads
    // A₀ · (1 − (1 − k)⁴), one combination of A₀ and k; the parts would
    // separate them.
    let total = ObservedDataPoint {
        parts: 4,
        ..ObservedDataPoint::with_window(4.0, "B".to_string(), 300.0, 4.0)
    };
    let problem = CalibrationProblem::new(
        decay("k", &["k"]),
        vec![total],
        vec![
            CalibrationParameter::new("k".to_string(), 0.05, 0.5),
            CalibrationParameter::with_type(
                "A".to_string(),
                CalibrationParameterType::InitialCondition,
                500.0,
                1500.0,
            ),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![("B".to_string(), ObservationModel::Poisson)]),
    )
    .unwrap();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

    let sensitivities = source.jacobian(&[0.1, 1000.0]).unwrap();
    assert_eq!(sensitivities.jacobian.nrows(), 1);
    assert_eq!(sensitivities.parts.jacobian.nrows(), 4);
    assert_eq!(sensitivities.parts.jacobian.rank(1e-8), 2);
    let analysis = structural_rank(&source, 5, 7).unwrap();
    assert_eq!(analysis.generic_rank, 1);
    let indices = collinearity(&source, &[vec![0.1, 1000.0]], &[1.0], 2, 20.0).unwrap();
    assert_eq!(indices.pair_gamma[0][1], f64::INFINITY);
}

/// A sample of `columns` with `bounds`, equally weighted, of the parameters
/// `p0`, `p1`, …
fn sample_on(columns: &[Vec<f64>], bounds: &[(f64, f64)]) -> ParameterSample {
    let draws = columns[0].len();
    let values: Vec<Vec<f64>> = (0..draws)
        .map(|i| columns.iter().map(|column| column[i]).collect())
        .collect();
    ParameterSample::new(
        (0..columns.len()).map(|j| format!("p{j}")).collect(),
        bounds.to_vec(),
        values,
        &vec![0.0; draws],
        None,
    )
    .unwrap()
}

/// The settings of a sample against the priors `priors` of `p0`, `p1`, …,
/// with `log_parameters`.
fn with_priors(priors: &[Family], log_parameters: &[&str]) -> SampleIdentifiabilitySettings {
    let families = priors
        .iter()
        .enumerate()
        .map(|(j, family)| (format!("p{j}"), *family))
        .collect();
    SampleIdentifiabilitySettings {
        reference: Reference::Prior,
        priors: Some(PriorsSettings {
            distributions: Some(Pairs(families)),
        }),
        log_parameters: log_parameters.iter().map(|name| name.to_string()).collect(),
        ..bounds(0.9)
    }
}

/// `n` draws of `draw`.
fn draws(n: usize, generator: &mut Generator, draw: impl Fn(&mut Generator) -> f64) -> Vec<f64> {
    (0..n).map(|_| draw(generator)).collect()
}

/// A draw of the normal of `mean` and `sd` truncated to `[low, high]`.
fn truncated(generator: &mut Generator, mean: f64, sd: f64, (low, high): (f64, f64)) -> f64 {
    loop {
        let x = mean + sd * normal(generator);
        if (low..=high).contains(&x) {
            return x;
        }
    }
}

#[test]
fn a_sample_of_its_reference_is_poorly_determined_by_its_contraction() {
    // Uniform draws on the bounds, and draws of LogNormal(0, 2) read on the
    // log scale.
    let mut generator = Generator::from_seed(11);
    let uniform = draws(2000, &mut generator, |g| g.uniform());
    let log_normal = draws(2000, &mut generator, |g| (2.0 * normal(g)).exp());
    let sample = sample_on(&[uniform, log_normal], &[(0.0, 1.0), (0.0, f64::INFINITY)]);
    let priors = [
        Family::Uniform,
        Family::LogNormal {
            mu: 0.0,
            sigma: 2.0,
        },
    ];

    let result = sample_identifiability(&sample, &with_priors(&priors, &["p1"]), 0).unwrap();

    for (j, verdict) in result.verdicts.iter().enumerate() {
        assert!(
            result.contraction[j].abs() < 0.1,
            "{:?}",
            result.contraction
        );
        assert_eq!(verdict.verdict, PracticalVerdict::PoorlyDetermined);
        assert!(verdict.low_contraction, "{verdict:?}");
        assert!(verdict.correlated_with.is_empty(), "{verdict:?}");
        assert_eq!(verdict.modes, 1, "{verdict:?}");
    }
}

#[test]
fn a_flat_or_smooth_marginal_has_one_mode() {
    // Each parameter is drawn from its reference: Uniform, LogUniform read
    // on the log scale, Normal on infinite bounds and Gamma.
    let priors = [
        Family::Uniform,
        Family::LogUniform,
        Family::Normal { mean: 0.0, sd: 1.0 },
        Family::Gamma {
            shape: 2.0,
            rate: 1.0,
        },
    ];
    let bounds = [
        (0.0, 1.0),
        (0.01, 100.0),
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0, f64::INFINITY),
    ];
    let settings = with_priors(&priors, &["p1"]);
    let mut marginals = 0;
    let mut multimodal = Vec::new();
    for n in [300, 2000, 20_000] {
        let counts: Vec<Vec<usize>> = (0..200)
            .into_par_iter()
            .map(|seed| {
                let mut generator = Generator::from_seed(seed + 1000 * n as u64);
                let columns = [
                    draws(n, &mut generator, |g| g.uniform()),
                    draws(n, &mut generator, |g| 0.01 * 10_000f64.powf(g.uniform())),
                    draws(n, &mut generator, normal),
                    draws(n, &mut generator, |g| gamma(g, 2.0, 1.0)),
                ];
                let result =
                    sample_identifiability(&sample_on(&columns, &bounds), &settings, seed).unwrap();
                result.modes.iter().map(Vec::len).collect()
            })
            .collect();
        for count in counts {
            marginals += count.len();
            multimodal.extend(count.into_iter().filter(|&modes| modes > 1));
        }
    }

    assert_eq!(marginals, 2400);
    assert!(multimodal.len() <= 24, "{multimodal:?}");
}

#[test]
fn a_wide_range_or_a_heavy_draw_keeps_the_modes_of_the_mass() {
    // 90 % near 0 and two clusters of 5 % far away: the grid keeps its step
    // of h/4 over a range of 10⁹ bandwidths.
    let mut generator = Generator::from_seed(21);
    let wide: Vec<f64> = (0..3000)
        .map(|i| {
            let center = match i % 20 {
                0 => 3.3e5,
                1 => 1e6,
                _ => 0.0,
            };
            center + 0.01 * normal(&mut generator)
        })
        .collect();
    let sample = sample_on(&[wide], &[(-1e7, 1e7)]);
    let modes = &sample_identifiability(&sample, &bounds(0.9), 0)
        .unwrap()
        .modes[0];
    assert_eq!(modes.len(), 3, "{modes:?}");
    for (mode, center) in modes.iter().zip([0.0, 3.3e5, 1e6]) {
        assert!((mode - center).abs() < 0.01, "{modes:?}");
    }

    // One draw of 2 % of the weight far from 999 others is one draw, not a
    // mode: the standard error of the density reads the weight of each.
    let mut values: Vec<Vec<f64>> =
        draws(999, &mut generator, |g| truncated(g, 0.3, 0.05, (0.0, 1.0)))
            .into_iter()
            .map(|x| vec![x])
            .collect();
    values.push(vec![0.8]);
    let mut log_weights = vec![0.0; 999];
    log_weights.push((0.02 * 999.0 / 0.98f64).ln());
    let heavy = ParameterSample::new(
        vec!["p0".to_string()],
        vec![(0.0, 1.0)],
        values,
        &log_weights,
        None,
    )
    .unwrap();
    let result = sample_identifiability(&heavy, &bounds(0.9), 0).unwrap();
    assert_eq!(result.modes[0].len(), 1, "{:?}", result.modes);
}

#[test]
fn the_spread_is_that_of_the_weighted_draws() {
    // Weights of 1 and 4e-18: the spread is that of the weights, not NaN.
    let values = vec![vec![0.2, 0.3], vec![0.5, 0.1], vec![0.9, 0.7]];
    let nearly_one = ParameterSample::new(
        vec!["p0".to_string(), "p1".to_string()],
        vec![(0.0, 1.0); 2],
        values,
        &[0.0, -40.0, -40.0],
        None,
    )
    .unwrap();
    let result = sample_identifiability(&nearly_one, &bounds(0.9), 0).unwrap();
    assert!(
        result.contraction.iter().all(|c| (c - 1.0).abs() < 1e-12),
        "{:?}",
        result.contraction
    );

    // Half the weight at each end of [0, 1]: the variance 1/4, the largest
    // on the bounds, a contraction of 1 − 3 = −2, the same for copies.
    let ends = sample_on(&[vec![0.0, 1.0]], &[(0.0, 1.0)]);
    let copies = sample_on(&[vec![0.0, 0.0, 1.0, 1.0]], &[(0.0, 1.0)]);
    for sample in [ends, copies] {
        let contraction = sample_identifiability(&sample, &bounds(0.9), 0)
            .unwrap()
            .contraction;
        assert!((contraction[0] + 2.0).abs() < 1e-12, "{contraction:?}");
    }
}

#[test]
fn real_modes_are_kept() {
    let unit = (0.0, 1.0);
    for seed in 0..10 {
        let mut generator = Generator::from_seed(seed);
        let g = &mut generator;
        // Two weak modes; a minor mode of 15 %; two narrow modes of 97 % of
        // the mass, with 3 % of the draws spread far.
        let weak: Vec<f64> = (0..2000)
            .map(|i| truncated(g, if i % 2 == 0 { 0.35 } else { 0.65 }, 0.08, unit))
            .collect();
        let minor: Vec<f64> = (0..2000)
            .map(|i| truncated(g, if i % 20 < 3 { 0.7 } else { 0.3 }, 0.05, unit))
            .collect();
        let narrow: Vec<f64> = (0..2000)
            .map(|i| {
                if i % 100 < 3 {
                    100.0 * g.uniform()
                } else {
                    truncated(g, if i % 2 == 0 { 0.3 } else { 0.6 }, 0.01, unit)
                }
            })
            .collect();
        let sample = sample_on(&[weak, minor, narrow], &[unit, unit, (0.0, 100.0)]);

        let result = sample_identifiability(&sample, &bounds(0.9), seed).unwrap();

        for modes in &result.modes {
            assert_eq!(modes.len(), 2, "seed {seed}: {:?}", result.modes);
        }
    }

    // A unimodal sample under a normal or a gamma prior has its mode at the
    // mode of the sample, not where it is far from its prior.
    let mut generator = Generator::from_seed(3);
    let near_one = draws(20_000, &mut generator, |g| 1.0 + 0.5 * normal(g));
    let skewed = draws(20_000, &mut generator, |g| gamma(g, 20.0, 10.0));
    let sample = sample_on(
        &[near_one, skewed],
        &[(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
    );
    let priors = [
        Family::Normal { mean: 0.0, sd: 1.0 },
        Family::Gamma {
            shape: 2.0,
            rate: 1.0,
        },
    ];

    let result = sample_identifiability(&sample, &with_priors(&priors, &[]), 3).unwrap();

    assert_eq!(result.modes[0].len(), 1);
    assert!((result.modes[0][0] - 1.0).abs() < 0.1, "{:?}", result.modes);
    assert_eq!(result.modes[1].len(), 1);
    assert!((result.modes[1][0] - 1.9).abs() < 0.1, "{:?}", result.modes);
}

/// The columns of `a_bound_that_sets_a_side` against the uniform on their
/// bounds, with those bounds and the sides each bound sets.
type Column = (fn(&mut Generator) -> f64, (f64, f64), &'static [BoundSide]);

/// A sample of a prior: its draw, its prior and its bounds.
type OfPrior = (fn(&mut Generator) -> f64, Family, (f64, f64));

/// The sides each bound sets, of each column against its bounds and against
/// its prior.
type Sides = (Vec<Vec<BoundSide>>, Vec<Vec<BoundSide>>);

#[test]
fn a_bound_that_sets_a_side() {
    use BoundSide::{Lower, Upper};
    let unit = (0.0, 1.0);
    let against_bounds: [Column; 10] = [
        (|g| g.uniform(), unit, &[]),
        (|g| truncated(g, 0.93, 0.02, (0.0, 1.0)), unit, &[]),
        (|g| truncated(g, 0.95, 0.03, (0.0, 1.0)), unit, &[]),
        (|g| truncated(g, 0.99, 0.004, (0.0, 1.0)), unit, &[]),
        (|g| truncated(g, 0.985, 0.005, (0.0, 1.0)), unit, &[]),
        // A pile at the upper bound: 47 % of its mass in the top 5 %.
        (|g| truncated(g, 1.0, 0.08, (0.0, 1.0)), unit, &[Upper]),
        (|g| truncated(g, 0.0, 0.05, (0.0, 1.0)), unit, &[Lower]),
        (
            |g| truncated(g, 0.0, 0.05, (0.0, 0.2)),
            (0.0, 0.2),
            &[Lower],
        ),
        (|g| 0.5 + 0.5 * g.uniform(), unit, &[Upper]),
        // The density 2u.
        (|g| g.uniform().sqrt(), unit, &[Upper]),
    ];
    // Samples of their references, which no bound sets.
    let of_priors: [OfPrior; 3] = [
        (
            |g| gamma(g, 2.0, 10.0),
            Family::Gamma {
                shape: 2.0,
                rate: 10.0,
            },
            (0.0, f64::INFINITY),
        ),
        (
            |g| gamma(g, 0.5, 1.0),
            Family::Gamma {
                shape: 0.5,
                rate: 1.0,
            },
            (0.0, f64::INFINITY),
        ),
        (
            |g| g.uniform().powf(0.2),
            Family::Beta { a: 5.0, b: 1.0 },
            unit,
        ),
    ];
    let priors: Vec<Family> = of_priors.iter().map(|(_, family, _)| *family).collect();
    let prior_settings = with_priors(&priors, &[]);
    // Twenty seeds at 300 and 2000 draws, four at 20 000.
    for (n, seeds) in [(300, 20), (2000, 20), (20_000, 4)] {
        let runs: Vec<Sides> = (0..seeds)
            .into_par_iter()
            .map(|seed| {
                let mut generator = Generator::from_seed(seed + 1000 * n as u64);
                let columns: Vec<Vec<f64>> = against_bounds
                    .iter()
                    .map(|(draw, _, _)| draws(n, &mut generator, draw))
                    .collect();
                let limits: Vec<(f64, f64)> = against_bounds.iter().map(|c| c.1).collect();
                let by_bounds =
                    sample_identifiability(&sample_on(&columns, &limits), &bounds(0.9), seed)
                        .unwrap();
                let columns: Vec<Vec<f64>> = of_priors
                    .iter()
                    .map(|(draw, _, _)| draws(n, &mut generator, draw))
                    .collect();
                let limits: Vec<(f64, f64)> = of_priors.iter().map(|c| c.2).collect();
                let by_priors =
                    sample_identifiability(&sample_on(&columns, &limits), &prior_settings, seed)
                        .unwrap();
                let sides = |result: &commol_inference::identifiability::SampleIdentifiability| {
                    result
                        .verdicts
                        .iter()
                        .map(|verdict| verdict.at_bound.clone())
                        .collect()
                };
                (sides(&by_bounds), sides(&by_priors))
            })
            .collect();
        for (j, (_, _, expected)) in against_bounds.iter().enumerate() {
            let matching = runs.iter().filter(|run| run.0[j] == *expected).count();
            if expected.is_empty() {
                assert_eq!(matching, runs.len(), "column {j} at {n} draws");
            } else if n == 300 {
                // Few draws: the test of the side has less power.
                assert!(matching >= 15, "column {j}: {matching} of 20 at 300 draws");
            } else if n == 2000 {
                assert!(matching >= 18, "column {j}: {matching} of 20");
            } else {
                // The power of the test grows with the draws.
                assert_eq!(matching, runs.len(), "column {j} at {n} draws");
            }
        }
        for j in 0..of_priors.len() {
            assert!(
                runs.iter().all(|run| run.1[j].is_empty()),
                "prior {j} at {n} draws"
            );
        }
    }
}

#[test]
fn an_infinite_side_has_mass_and_no_bound() {
    // Draws far in the lower tail of a normal prior on infinite bounds, and
    // a pile at 0 under a gamma prior, whose upper side is infinite.
    let mut generator = Generator::from_seed(5);
    let tail = draws(2000, &mut generator, |g| -2.5 + 0.3 * normal(g));
    let pile = draws(2000, &mut generator, |g| 0.02 * normal(g).abs());
    let sample = sample_on(
        &[tail, pile],
        &[(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
    );
    let priors = [
        Family::Normal { mean: 0.0, sd: 1.0 },
        Family::Gamma {
            shape: 2.0,
            rate: 1.0,
        },
    ];

    let result = sample_identifiability(&sample, &with_priors(&priors, &[]), 0).unwrap();

    // P(N(−2.5, 0.3) < Φ⁻¹(0.05) = −1.645) = Φ(2.85) = 0.998.
    assert!((result.mass_below[0] - 0.998).abs() < 0.005, "{result:?}");
    assert_eq!(result.mass_above[0], 0.0);
    assert!(result.verdicts[0].at_bound.is_empty());
    // The gamma prior holds 5 % of its mass below 0.355: every draw.
    assert!((result.mass_below[1] - 1.0).abs() < 1e-12);
    assert_eq!(result.verdicts[1].at_bound, [BoundSide::Lower]);
}

#[test]
fn the_overlap_of_a_sample_with_its_reference() {
    let mean_overlap = |n: usize| {
        let overlaps: Vec<f64> = (0..200)
            .map(|seed| {
                let mut generator = Generator::from_seed(seed);
                let uniform = draws(n, &mut generator, |g| g.uniform());
                let result =
                    sample_identifiability(&sample_on(&[uniform], &[(0.0, 1.0)]), &bounds(0.9), 0)
                        .unwrap();
                result.overlap[0]
            })
            .collect();
        overlaps.iter().sum::<f64>() / overlaps.len() as f64
    };
    // B = ⌈2 · 500^{1/3}⌉ = 16 and ⌈2 · 30^{1/3}⌉ = 7.
    assert!(mean_overlap(500) > 0.85);
    assert!((mean_overlap(30) - 0.82).abs() < 0.02);
    let mut generator = Generator::from_seed(1);
    let uniform = draws(500, &mut generator, |g| g.uniform());
    let result =
        sample_identifiability(&sample_on(&[uniform], &[(0.0, 1.0)]), &bounds(0.9), 0).unwrap();
    assert_eq!(result.bins, 16);

    // A sample on one value fills one bin of B.
    let on_one = draws(1000, &mut generator, |g| 0.53 + 1e-6 * normal(g));
    let result =
        sample_identifiability(&sample_on(&[on_one], &[(0.0, 1.0)]), &bounds(0.9), 0).unwrap();
    assert_eq!(result.bins, 20);
    assert!((result.overlap[0] - 0.05).abs() < 1e-12);
    let one_point = sample_on(&[vec![0.3]], &[(0.0, 1.0)]);
    let result = sample_identifiability(&one_point, &bounds(0.9), 0).unwrap();
    assert_eq!((result.bins, result.overlap[0]), (2, 0.5));
}

#[test]
fn log_parameters_set_the_scale_of_every_shape_measure() {
    // The variance of ln θ under the reference, against closed forms:
    // σ² of the log-normal, the trigamma of the shape of the gamma, and the
    // width squared over 12 of the log-uniform.
    let half_line = (0.0, f64::INFINITY);
    let cases = [
        (
            Family::LogNormal {
                mu: 0.3,
                sigma: 0.7,
            },
            half_line,
            0.49,
        ),
        (
            Family::Gamma {
                shape: 2.0,
                rate: 3.0,
            },
            half_line,
            PI * PI / 6.0 - 1.0,
        ),
        (
            Family::Gamma {
                shape: 0.5,
                rate: 1.0,
            },
            half_line,
            PI * PI / 2.0,
        ),
        (Family::LogUniform, (0.1, 10.0), 100f64.ln().powi(2) / 12.0),
        // Shapes whose quantile falls below the smallest f64: the trigamma
        // ψ₁(a), and ψ₁(a) − ψ₁(a + b) for the beta.
        (
            Family::Gamma {
                shape: 0.01,
                rate: 1.0,
            },
            half_line,
            10_001.621_213_528_317,
        ),
        (
            Family::Gamma {
                shape: 0.001,
                rate: 0.001,
            },
            half_line,
            1_000_001.642_533_196,
        ),
        (
            Family::Beta { a: 0.002, b: 2.0 },
            (0.0, 1.0),
            250_000.996_011_968,
        ),
    ];
    for (family, (lower, upper), expected) in cases {
        let variance = Distribution::new(family, lower, upper)
            .unwrap()
            .log_variance();
        assert!(
            (variance - expected).abs() < 1e-6 * expected,
            "{family}: {variance} {expected}"
        );
    }

    // A product held fixed, and a pile at the bound 0.
    let mut generator = Generator::from_seed(8);
    let first = draws(2000, &mut generator, |g| {
        (5f64.ln() * (2.0 * g.uniform() - 1.0)).exp()
    });
    let second: Vec<f64> = first
        .iter()
        .map(|x| (-x.ln() + 0.01 * normal(&mut generator)).exp())
        .collect();
    let pile = draws(2000, &mut generator, |g| {
        truncated(g, 0.0, 0.02, (0.0, 1.0))
    });
    let sample = sample_on(
        &[first, second, pile],
        &[(0.1, 10.0), (0.1, 10.0), (0.0, 1.0)],
    );
    let natural = sample_identifiability(&sample, &bounds(0.9), 0).unwrap();
    let logs = SampleIdentifiabilitySettings {
        log_parameters: vec!["p0".to_string(), "p1".to_string(), "p2".to_string()],
        ..bounds(0.9)
    };
    let log = sample_identifiability(&sample, &logs, 0).unwrap();

    assert_eq!(log.mass_below, natural.mass_below);
    assert_eq!(log.mass_above, natural.mass_above);
    assert_eq!(log.overlap, natural.overlap);
    // p0 is log-uniform on [0.2, 5]: in ln θ it spreads (ln 25)² / 12 against
    // the variance of ln θ under the uniform on [0.1, 10], a contraction of
    // −0.10, and in θ a variance of 1.65 against 8.17, one of 0.80.
    let reference = Distribution::new(Family::Uniform, 0.1, 10.0).unwrap();
    let in_log = 1.0 - 25f64.ln().powi(2) / 12.0 / reference.log_variance();
    let in_natural =
        1.0 - (24.96 / (2.0 * 25f64.ln()) - (4.8 / 25f64.ln()).powi(2)) / reference.variance();
    assert!(
        (log.contraction[0] - in_log).abs() < 0.05,
        "{:?}",
        log.contraction
    );
    assert!(
        (natural.contraction[0] - in_natural).abs() < 0.05,
        "{:?}",
        natural.contraction
    );
    assert!(log.verdicts[0].low_contraction);
    assert!(!natural.verdicts[0].low_contraction);
    // On the log scale the product is a line: its direction holds both
    // parameters equally and contracts almost wholly.
    let ridge = log.directions.last().unwrap();
    assert!(ridge.contraction > 0.999, "{ridge:?}");
    assert!(
        (ridge.loadings[0] - FRAC_1_SQRT_2).abs() < 0.01,
        "{ridge:?}"
    );
    assert!(
        (ridge.loadings[1] - FRAC_1_SQRT_2).abs() < 0.01,
        "{ridge:?}"
    );
    // On the natural scale the product is a curve, which no direction of
    // the first two parameters follows as closely.
    assert!(
        natural
            .directions
            .iter()
            .filter(|direction| direction.loadings[2].abs() < 0.1)
            .all(|direction| direction.contraction < 0.99),
        "{:?}",
        natural.directions
    );
    // A log-uniform marginal is flat in ln θ, with one mode in parameter
    // units; the bound 0 is no bound of ln θ.
    assert_eq!(log.modes[0].len(), 1);
    assert!((0.1..=10.0).contains(&log.modes[0][0]));
    assert_eq!(natural.verdicts[2].at_bound, [BoundSide::Lower]);
    assert!(log.verdicts[2].at_bound.is_empty());
}

#[test]
fn directions_and_their_contraction() {
    // Against standard normal priors: the first two parameters have the
    // covariance [[0.25, 0.2], [0.2, 0.25]], eigenvalues 0.45 along (1, 1)
    // and 0.05 along (1, −1); the third spreads twice as wide as its prior.
    let mut generator = Generator::from_seed(9);
    let n = 20_000;
    let along = draws(n, &mut generator, normal);
    let across = draws(n, &mut generator, normal);
    let first: Vec<f64> = along
        .iter()
        .zip(&across)
        .map(|(a, b)| (0.45f64.sqrt() * a + 0.05f64.sqrt() * b) * FRAC_1_SQRT_2)
        .collect();
    let second: Vec<f64> = along
        .iter()
        .zip(&across)
        .map(|(a, b)| (0.45f64.sqrt() * a - 0.05f64.sqrt() * b) * FRAC_1_SQRT_2)
        .collect();
    let wide = draws(n, &mut generator, |g| 2.0 * normal(g));
    let infinite = (f64::NEG_INFINITY, f64::INFINITY);
    let sample = sample_on(&[first, second, wide], &[infinite; 3]);
    let priors = [Family::Normal { mean: 0.0, sd: 1.0 }; 3];

    let result = sample_identifiability(&sample, &with_priors(&priors, &[]), 0).unwrap();

    let found: Vec<(f64, Vec<f64>)> = result
        .directions
        .iter()
        .map(|d| (d.contraction, d.loadings.clone()))
        .collect();
    let expected = [
        (-3.0, [0.0, 0.0, 1.0]),
        (0.55, [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]),
        (0.95, [FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0]),
    ];
    for ((contraction, loadings), (want, want_loadings)) in found.iter().zip(&expected) {
        assert!(
            (contraction - want).abs() < 0.1 * want.abs().min(1.0),
            "{found:?}"
        );
        for (loading, want) in loadings.iter().zip(want_loadings) {
            assert!((loading - want).abs() < 0.03, "{found:?}");
        }
    }

    // Three distinct rows, ten copies each, give no directions for three
    // parameters.
    let rows: Vec<f64> = (0..30).map(|i| f64::from(i % 3)).collect();
    let few = sample_on(&[rows.clone(), rows.clone(), rows], &[infinite; 3]);
    let result = sample_identifiability(&few, &with_priors(&priors, &[]), 0).unwrap();
    assert_eq!(result.distinct_rows, 3);
    assert!(result.directions.is_empty());
    assert_eq!(result.hidden_ridges, None);
}

#[test]
fn the_count_of_hidden_ridges() {
    // Ten parameters of contraction 0.6 against standard normal priors, at
    // 300 draws: independent, or sharing a ridge of pairwise correlation
    // 1/6, whose direction contracts by 0.
    let priors = [Family::Normal { mean: 0.0, sd: 1.0 }; 10];
    let settings = with_priors(&priors, &[]);
    let counts = |ridge: bool| -> Vec<usize> {
        (0..50)
            .into_par_iter()
            .map(|seed| {
                let mut generator = Generator::from_seed(500 + seed);
                let common = draws(300, &mut generator, normal);
                let columns: Vec<Vec<f64>> = (0..10)
                    .map(|_| {
                        common
                            .iter()
                            .map(|c| {
                                let own = normal(&mut generator);
                                if ridge {
                                    (0.4f64 - 1.0 / 15.0).sqrt() * own + (1.0f64 / 15.0).sqrt() * c
                                } else {
                                    0.4f64.sqrt() * own
                                }
                            })
                            .collect()
                    })
                    .collect();
                let infinite = (f64::NEG_INFINITY, f64::INFINITY);
                let sample = sample_on(&columns, &[infinite; 10]);
                let result = sample_identifiability(&sample, &settings, seed).unwrap();
                assert!((result.draws_per_parameter - 30.0).abs() < 1e-9);
                result.hidden_ridges.unwrap()
            })
            .collect()
    };

    let independent = counts(false);
    let mean = independent.iter().sum::<usize>() as f64 / 50.0;
    assert!(mean < 0.3, "{independent:?}");
    assert!(counts(true).iter().all(|&count| count == 1));
}

#[test]
fn the_directions_do_not_depend_on_units() {
    let mut generator = Generator::from_seed(12);
    let a = draws(500, &mut generator, |g| truncated(g, 0.5, 0.1, (0.0, 1.0)));
    let b: Vec<f64> = a
        .iter()
        .map(|x| (0.5 * x + 0.2 * generator.uniform()).min(1.0))
        .collect();
    let sample = sample_on(&[a.clone(), b.clone()], &[(0.0, 1.0); 2]);
    let scaled: Vec<f64> = a.iter().map(|x| 1000.0 * x).collect();
    let rescaled = sample_on(&[scaled, b], &[(0.0, 1000.0), (0.0, 1.0)]);

    let first = sample_identifiability(&sample, &bounds(0.9), 0).unwrap();
    let second = sample_identifiability(&rescaled, &bounds(0.9), 0).unwrap();

    assert_eq!(first.directions.len(), 2);
    assert_eq!(second.directions.len(), 2);
    for (d, e) in first.directions.iter().zip(&second.directions) {
        assert!((d.contraction - e.contraction).abs() < 1e-9, "{d:?} {e:?}");
    }
}

#[test]
fn a_sample_of_too_few_effective_draws_has_no_determined_parameter() {
    // 500 uniform draws of two parameters whose weights put 0.999 on one
    // draw: the weighted variance is near 0, so each contraction is near 1,
    // but the effective size is about 1, below p + 1 = 3.
    let mut generator = Generator::from_seed(14);
    let columns = [
        draws(500, &mut generator, |g| g.uniform()),
        draws(500, &mut generator, |g| g.uniform()),
    ];
    let values: Vec<Vec<f64>> = (0..500)
        .map(|i| vec![columns[0][i], columns[1][i]])
        .collect();
    let log_weights: Vec<f64> = (0..500)
        .map(|i| {
            if i == 0 {
                0.999f64.ln()
            } else {
                (0.001f64 / 499.0).ln()
            }
        })
        .collect();
    let sample = ParameterSample::new(
        vec!["p0".to_string(), "p1".to_string()],
        vec![(0.0, 1.0); 2],
        values,
        &log_weights,
        None,
    )
    .unwrap();

    let result = sample_identifiability(&sample, &bounds(0.9), 0).unwrap();

    assert!(result.effective_size < 1.01, "{}", result.effective_size);
    assert_eq!(result.size, SampleSize::TooFew);
    for (j, verdict) in result.verdicts.iter().enumerate() {
        assert!(result.contraction[j] > 0.99, "{:?}", result.contraction);
        assert!(!verdict.low_contraction, "{verdict:?}");
        assert_eq!(verdict.verdict, PracticalVerdict::PoorlyDetermined);
    }
    assert_eq!(result.hidden_ridges, Some(0));

    // Equal weights: 30 draws of two parameters are above p + 1 and below
    // 30 per parameter; 500 are not.
    let tight = |n: usize, generator: &mut Generator| -> ParameterSample {
        let a = draws(n, generator, |g| 0.5 + 0.01 * g.uniform());
        let b = draws(n, generator, |g| 0.5 + 0.01 * g.uniform());
        sample_on(&[a, b], &[(0.0, 1.0); 2])
    };
    let few = sample_identifiability(&tight(30, &mut generator), &bounds(0.9), 0).unwrap();
    let enough = sample_identifiability(&tight(500, &mut generator), &bounds(0.9), 0).unwrap();
    assert_eq!(few.size, SampleSize::Few);
    assert_eq!(enough.size, SampleSize::Enough);
    for result in [&few, &enough] {
        assert!(
            result
                .verdicts
                .iter()
                .all(|v| v.verdict == PracticalVerdict::Determined),
            "{:?}",
            result.verdicts
        );
    }

    // Equal weights at a floor are at it, whichever way their Kish size
    // rounds: 5 draws of 4 parameters, 30 draws of 1.
    let at = |n: usize, p: usize, generator: &mut Generator| -> SampleSize {
        let columns: Vec<Vec<f64>> = (0..p)
            .map(|_| draws(n, generator, |g| g.uniform()))
            .collect();
        let sample = sample_on(&columns, &vec![(0.0, 1.0); p]);
        sample_identifiability(&sample, &bounds(0.9), 0)
            .unwrap()
            .size
    };
    assert_eq!(at(5, 4, &mut generator), SampleSize::Few);
    assert_eq!(at(4, 4, &mut generator), SampleSize::TooFew);
    assert_eq!(at(30, 1, &mut generator), SampleSize::Enough);
    assert_eq!(at(29, 1, &mut generator), SampleSize::Few);
}
