//! Local approximations around a fit: the curvature on the unbounded scale,
//! the Laplace approximation and the delta method, the sandwich, the Laplace
//! mixture, importance sampling and PSIS, on an engine whose outputs are
//! closed forms of its parameters.

use commol_core::NoSteps;
use commol_core::{STOPPED, Steps};
use commol_inference::runs::{RunEvent, RunSteps};
use std::collections::HashMap;
use std::sync::Mutex;

use commol_calibration::Workspace;
use commol_calibration::{
    Argument, CalibrationParameter, CalibrationParameterType, CalibrationProblem,
    DiscrepancyConfig, LossConfig, Noise, ObservationModel, ObservedDataPoint, ReplicateSummary,
    Replicates,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::distribution::Family;
use commol_inference::importance::TemperedPosterior;
use commol_inference::local::{
    Curvature, CurvatureKind, Gaussian, curvature_at, delta_band, laplace, laplace_mixture,
    sandwich_approximation, scaled_block_scores,
};
use commol_inference::psis::smooth;
use commol_inference::resampling::BlockScheme;
use commol_inference::sensitivity::{
    Design, FiniteDifferences, Matrix, Sensitivities, SensitivityError, SensitivitySource,
};
use commol_inference::smc::{LikelihoodTarget, information_matching_eta};
use commol_inference::target::Target;
use commol_inference::transform::Transform;
use commol_inference::{MethodError, Priors};
use commol_statistics::special::{normal_quantile_sf, student_t_two_sided_tail};

/// The closed form of the outputs of a [`Curve`].
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// One output `x = a + b·t`.
    Line,
    /// One output `x = a²`: two optima, `±√y`.
    Square,
    /// Two outputs, `x = a` and `y = b + a²`: with a precise `y`, the
    /// posterior of `(a, b)` bends along `b = y − a²`.
    Banana,
    /// One output `x = a·t + b·t²`, whose increments over equal parts grow
    /// along the axis.
    Quadratic,
    /// One output `x = a + unit·b·t`: a line whose slope is `b` units.
    Units(f64),
    /// Two outputs, `x = a` and `y = unit·(b (1 + t) + a)`.
    Coupled(f64),
    /// One output `x = a·b·(1 + t)`, which reads `a` and `b` only as their
    /// product.
    Product,
    /// Two outputs, `x = a + b·t` and `y = b`.
    LineAndSlope,
    /// One cumulative output `x = a·(t + b·(1 − cos 2πt)/2π)`: its increment
    /// over a window from one whole position to the next does not read `b`,
    /// and its increments over the parts of that window do.
    Wave,
    /// Two outputs, `x = a` and `y = b`, each read by one parameter.
    Pair,
    /// Two outputs, `x = a + b·t` and `y = c`.
    LineAndLevel,
}

/// An engine whose outputs at a position `t` are a closed form of its
/// parameters `a` and `b`.
#[derive(Clone, Debug)]
struct Curve {
    shape: Shape,
    parameters: HashMap<String, f64>,
}

impl Curve {
    fn new(shape: Shape) -> Self {
        Self {
            shape,
            parameters: HashMap::from([
                ("a".to_string(), 0.0),
                ("b".to_string(), 0.0),
                ("c".to_string(), 0.0),
            ]),
        }
    }
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for Curve {
    fn check_varied_values(&self, _plan: &Self::ValuePlan) -> Result<(), String> {
        Ok(())
    }

    fn check_values(&self) -> Result<(), String> {
        Ok(())
    }

    type ValuePlan = NoPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        let (a, b, c) = (
            self.parameters["a"],
            self.parameters["b"],
            self.parameters["c"],
        );
        buffer.clear();
        buffer.extend(positions.iter().map(|&t| match self.shape {
            Shape::Line => vec![a + b * t],
            Shape::Square => vec![a * a],
            Shape::Banana => vec![a, b + a * a],
            Shape::Quadratic => vec![a * t + b * t * t],
            Shape::Units(unit) => vec![a + unit * b * t],
            Shape::Coupled(unit) => vec![a, unit * (b * (1.0 + t) + a)],
            Shape::Product => vec![a * b * (1.0 + t)],
            Shape::LineAndSlope => vec![a + b * t, b],
            Shape::Pair => vec![a, b],
            Shape::LineAndLevel => vec![a + b * t, c],
            Shape::Wave => {
                let tau = std::f64::consts::TAU;
                vec![a * (t + b * (1.0 - (tau * t).cos()) / tau)]
            }
        }));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        match self.shape {
            Shape::Banana
            | Shape::Coupled(_)
            | Shape::LineAndSlope
            | Shape::Pair
            | Shape::LineAndLevel => {
                vec!["x".to_string(), "y".to_string()]
            }
            _ => vec!["x".to_string()],
        }
    }

    fn state_outputs(&self) -> usize {
        self.output_names().len()
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(&self, _: &[ModelValue], _: Option<&str>) -> Result<NoPlan, String> {
        Ok(NoPlan)
    }

    fn apply_values(&mut self, _: &NoPlan, _: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}

/// Observations of `output` at the positions 0, 1, ….
fn observed(output: &str, values: &[f64]) -> Vec<ObservedDataPoint> {
    values
        .iter()
        .enumerate()
        .map(|(t, &value)| ObservedDataPoint::new(t as f64, output.to_string(), value))
        .collect()
}

fn normal(output: &str, sd: Argument) -> (String, ObservationModel) {
    (output.to_string(), ObservationModel::Normal { sd })
}

fn noise(id: &str, min: f64, max: f64) -> CalibrationParameter {
    CalibrationParameter::with_type(id.to_string(), CalibrationParameterType::Noise, min, max)
}

/// The derivative `dθ/du` of the logit of `θ` between `lower` and `upper`.
fn logit_derivative(theta: f64, lower: f64, upper: f64) -> f64 {
    (theta - lower) * (upper - theta) / (upper - lower)
}

/// The line `x = a + b·t` at ten positions with a normal model of noise
/// parameter `sd`.
fn line_problem() -> CalibrationProblem<Curve> {
    let values: Vec<f64> = (0..10)
        .map(|t| 1.0 + 0.5 * t as f64 + [0.3, -0.2, 0.1][t % 3])
        .collect();
    CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &values),
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
            noise("sd", 0.1, 2.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Parameter("sd".to_string()))]),
    )
    .unwrap()
}

/// The information of the line at `fit` on the unbounded scale:
/// `D · I(θ) · D`, with `I = Σ [1, t]ᵀ[1, t] / sd²` for `(a, b)` and
/// `2n / sd²` for `sd`.
fn line_information(fit: &[f64]) -> [[f64; 3]; 3] {
    let sd = fit[2];
    let mut information = [[0.0; 3]; 3];
    for t in 0..10 {
        let row = [1.0, t as f64];
        for j in 0..2 {
            for k in 0..2 {
                information[j][k] += row[j] * row[k] / (sd * sd);
            }
        }
    }
    information[2][2] = 2.0 * 10.0 / (sd * sd);
    let d = [
        logit_derivative(fit[0], -10.0, 10.0),
        logit_derivative(fit[1], -5.0, 5.0),
        logit_derivative(fit[2], 0.1, 2.0),
    ];
    for j in 0..3 {
        for k in 0..3 {
            information[j][k] *= d[j] * d[k];
        }
    }
    information
}

/// The curvature `H_u = S Ĥ S` on the unbounded scale.
fn on_the_unbounded_scale(curvature: &Curvature) -> nalgebra::DMatrix<f64> {
    let (scales, matrix) = (curvature.scales(), curvature.scaled_matrix());
    nalgebra::DMatrix::from_fn(scales.len(), scales.len(), |j, k| {
        scales[j] * matrix[(j, k)] * scales[k]
    })
}

fn assert_matrix(actual: &nalgebra::DMatrix<f64>, expected: &[[f64; 3]; 3], relative: f64) {
    for j in 0..3 {
        for k in 0..3 {
            let scale = (expected[j][j] * expected[k][k]).sqrt();
            assert!(
                (actual[(j, k)] - expected[j][k]).abs() <= relative * scale,
                "entry ({j}, {k}): {} against {}",
                actual[(j, k)],
                expected[j][k]
            );
        }
    }
}

#[test]
fn curvatures_of_a_linear_model_equal_the_information() {
    let problem = line_problem();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let fit = [1.1, 0.48, 0.4];

    let curvature = Curvature::gauss_newton(&problem, &source, &fit).unwrap();

    let matrix = on_the_unbounded_scale(&curvature);
    assert_matrix(&matrix, &line_information(&fit), 1e-8);
    assert!(curvature.notes().is_empty(), "{:?}", curvature.notes());
    let identity = laplace(&curvature).unwrap().covariance() * matrix;
    assert!((identity - nalgebra::DMatrix::identity(3, 3)).amax() < 1e-9);
}

#[test]
fn the_hessian_of_a_linear_model_equals_the_information() {
    // At the maximum of the likelihood the observed information of the line
    // is its expected information: the cross terms of `sd` sum the residuals,
    // and its own term is `3 Σ r² / sd⁴ − n / sd² = 2n / sd²`.
    let problem = line_problem();
    let observed: Vec<f64> = problem.observed_data().iter().map(|o| o.value).collect();
    let n = observed.len() as f64;
    let mean_t = 4.5;
    let mean_x = observed.iter().sum::<f64>() / n;
    let sxy: f64 = observed
        .iter()
        .enumerate()
        .map(|(t, x)| (t as f64 - mean_t) * (x - mean_x))
        .sum();
    let sxx: f64 = (0..10).map(|t| (t as f64 - mean_t).powi(2)).sum();
    let b = sxy / sxx;
    let a = mean_x - b * mean_t;
    let sd = (observed
        .iter()
        .enumerate()
        .map(|(t, x)| (x - a - b * t as f64).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    let fit = [a, b, sd];

    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let curvature = Curvature::hessian(&problem, &source, &fit).unwrap();

    let matrix = on_the_unbounded_scale(&curvature);
    assert_matrix(&matrix, &line_information(&fit), 1e-5);
}

#[test]
fn a_hessian_keeps_the_scores_and_the_blocks_of_gauss_newton() {
    // Its units are not those of Gauss–Newton, and on the unbounded scale both
    // have the same scores and blocks.
    let problem = line_problem();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let fit = [1.1, 0.48, 0.4];
    let expected = Curvature::gauss_newton(&problem, &source, &fit).unwrap();
    let curvature = Curvature::hessian(&problem, &source, &fit).unwrap();

    assert_ne!(curvature.scales(), expected.scales());
    let block = [0, 3, 4];
    let read = |curvature: &Curvature| {
        let scales = nalgebra::DMatrix::from_diagonal(&curvature.scales().to_vec().into());
        (
            curvature.scaled_gradients() * &scales,
            &scales * curvature.scaled_block_matrix(&block) * &scales,
        )
    };
    let (gradients, matrix) = read(&curvature);
    let (expected_gradients, expected_matrix) = read(&expected);
    assert!((gradients - &expected_gradients).amax() < 1e-12 * expected_gradients.amax());
    assert!((matrix - &expected_matrix).amax() < 1e-12 * expected_matrix.amax());
}

#[test]
fn a_scale_is_a_power_of_two_times_the_derivative_of_its_parameter() {
    // So that the units of a curvature round nothing.
    let problem = line_problem();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let fit = [1.1, 0.48, 0.4];
    let derivatives = Transform::new(&problem.parameter_bounds()).derivatives(&fit);
    for curvature in [
        Curvature::gauss_newton(&problem, &source, &fit).unwrap(),
        Curvature::hessian(&problem, &source, &fit).unwrap(),
    ] {
        for (scale, derivative) in curvature.scales().iter().zip(&derivatives) {
            assert_eq!(
                libm::frexp(scale / derivative).0,
                0.5,
                "{scale} {derivative}"
            );
        }
    }
}

#[test]
fn draws_mapped_back_stay_inside_the_bounds() {
    // The fit 0.02 of `a` is 0.7 standard deviations above its bound 0: a
    // normal on the scale of `a` would put a quarter of its draws below it.
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &[0.05, -0.02, 0.03]),
        vec![CalibrationParameter::new("a".to_string(), 0.0, 10.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.05))]),
    )
    .unwrap();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let curvature = Curvature::gauss_newton(&problem, &source, &[0.02]).unwrap();

    let draws = laplace(&curvature).unwrap().draws(2000, 7);

    assert_eq!(draws.len(), 2000);
    assert!(draws.iter().all(|draw| draw[0] > 0.0 && draw[0] < 10.0));
    // The median of the draws is the fit: half of them lie between it and
    // the bound.
    let below = draws.iter().filter(|draw| draw[0] < 0.02).count();
    assert!((900..1100).contains(&below), "{below} draws below the fit");
}

/// The level `x = a` at sixteen positions of mean 1, with `a` in `bounds`
/// and a normal noise of sd 0.4: the sd of `a` is 0.1. The residuals, of sd
/// 0.79, give the sandwich an sd of about 0.2.
fn level_problem(bounds: (f64, f64)) -> CalibrationProblem<Curve> {
    let values: Vec<f64> = (0..16)
        .map(|t| 1.0 + [0.5, -0.5, 1.0, -1.0][t % 4])
        .collect();
    CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &values),
        vec![CalibrationParameter::new(
            "a".to_string(),
            bounds.0,
            bounds.1,
        )],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.4))]),
    )
    .unwrap()
}

/// The bound and the distance in sd of each note of `notes` that names a
/// bound near the fit.
fn noted_bounds(notes: &[String]) -> Vec<(f64, f64)> {
    notes
        .iter()
        .filter_map(|note| {
            let (_, rest) = note.split_once(" is ")?;
            let (distance, rest) = rest.split_once(" sd from its bound ")?;
            let bound = rest.split([',', ' ']).next()?;
            Some((bound.parse().ok()?, distance.parse().ok()?))
        })
        .collect()
}

#[test]
fn a_fit_within_a_few_sd_of_a_bound_is_noted() {
    let fit = [1.0];
    let laplace_notes = |bounds: (f64, f64)| {
        let problem = level_problem(bounds);
        let curvature = curvature_at(&problem, &fit, CurvatureKind::GaussNewton).unwrap();
        laplace(&curvature).unwrap().bound_notes(curvature.names())
    };
    // Each scale of a bound: a logit from either bound, a log and a log from
    // above, with distances of 0.001 to 2.9 sd.
    for (bounds, bound, distance) in [
        ((0.0, 1.05), 1.05, 0.5),
        ((0.95, 3.0), 0.95, 0.5),
        ((0.71, f64::INFINITY), 0.71, 2.9),
        ((f64::NEG_INFINITY, 1.0001), 1.0001, 0.001),
    ] {
        let notes = laplace_notes(bounds);
        let noted = noted_bounds(&notes);
        assert_eq!(noted.len(), 1, "{bounds:?}: {notes:?}");
        assert_eq!(noted[0].0, bound, "{notes:?}");
        assert!(
            (noted[0].1 / distance - 1.0).abs() < 0.02,
            "{bounds:?}: {notes:?}"
        );
        assert!(notes[0].starts_with("'a' is "), "{notes:?}");
    }
    // Both bounds within 3 sd, the lower first; none at 3 sd or more.
    let noted = noted_bounds(&laplace_notes((0.8, 1.1)));
    assert_eq!(noted, vec![(0.8, 2.0), (1.1, 1.0)]);
    assert!(laplace_notes((0.69, 1.31)).is_empty());
    assert!(laplace_notes((-1e6, 1e6)).is_empty());
    assert!(laplace_notes((f64::NEG_INFINITY, f64::INFINITY)).is_empty());

    // The sandwich reads its own covariance, of a scale twice that of
    // Laplace, and its Student t as the normal of the same mass past the
    // bound.
    let sandwich_near = |bounds: (f64, f64), block_length: usize| {
        let problem = level_problem(bounds);
        let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, block_length)
            .unwrap();
        let scale =
            run.gaussian.covariance()[(0, 0)].sqrt() * logit_derivative(1.0, bounds.0, bounds.1);
        let nu = run.gaussian.degrees_of_freedom().unwrap()[0];
        let scaled = (bounds.1 - 1.0) / scale;
        let tail = 0.5 * student_t_two_sided_tail(scaled * scaled, nu);
        let notes = run.gaussian.bound_notes(run.curvature.names());
        (scaled, normal_quantile_sf(tail), nu, notes)
    };
    let (scaled, z, nu, notes) = sandwich_near((0.0, 1.05), 1);
    let noted = noted_bounds(&notes);
    assert_eq!(noted.len(), 1, "{notes:?}");
    // The note gives two decimals.
    assert!((noted[0].1 - z).abs() <= 0.005, "{noted:?} {z}");
    assert!(z < scaled && scaled < 0.3, "{scaled} {z}");
    assert!(
        notes[0].contains(&format!("Student t of {nu:.1} degrees of freedom")),
        "{notes:?}"
    );
    // With few degrees of freedom, a bound beyond 3 units of the scale of the
    // t is noted when the t puts more mass past it than a normal at 3 sd.
    let (scaled, z, nu, notes) = sandwich_near((-10.0, 1.8), 5);
    assert!(nu < 5.0 && scaled > 3.0 && z < 3.0, "{nu} {scaled} {z}");
    let noted = noted_bounds(&notes);
    assert_eq!(noted.len(), 1, "{notes:?}");
    assert!((noted[0].1 - z).abs() <= 0.005, "{noted:?} {z}");

    // The mixture notes it for its component.
    let problem = level_problem((0.0, 1.05));
    let priors = Priors::uniform_on_bounds(&["a".to_string()], &[(0.0, 1.05)]).unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();
    let mixture = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 1),
        1.0,
        &[fit.to_vec()],
        200,
        CurvatureKind::GaussNewton,
        1,
        &NoSteps,
    )
    .unwrap();
    assert_eq!(noted_bounds(&mixture.notes), vec![(1.05, 0.5)]);
    assert!(mixture.notes[0].starts_with("At the end [1.0], 'a' is "));
    assert!(mixture.notes[0].contains("the importance weights correct it"));
    assert!(!mixture.notes[0].contains("ProfileLikelihood"));

    // A fit on its bound is still refused.
    let problem = level_problem((0.0, 1.0 + 1e-9));
    let error = curvature_at(&problem, &fit, CurvatureKind::GaussNewton).unwrap_err();
    assert!(error.to_string().contains("'a' is on its bound"), "{error}");
}

/// The weighted share of `values` whose first entry is above 0.
fn positive_share(values: &[Vec<f64>], log_weights: &[f64]) -> f64 {
    let weights: Vec<f64> = log_weights.iter().map(|w| w.exp()).collect();
    let total: f64 = weights.iter().sum();
    values
        .iter()
        .zip(&weights)
        .filter(|(draw, _)| draw[0] > 0.0)
        .map(|(_, w)| w)
        .sum::<f64>()
        / total
}

#[test]
fn mixture_finds_both_modes_in_proportion() {
    // x = a² observed near 4: the likelihood has the modes ±2, and the prior
    // N(0.5, 1) gives the positive one more mass.
    let values: Vec<f64> = (0..10).map(|i| 4.0 + [0.2, -0.3, 0.1][i % 3]).collect();
    let mean: f64 = values.iter().sum::<f64>() / 10.0;
    let sd = 0.5;
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Square),
        observed("x", &values),
        vec![CalibrationParameter::new("a".to_string(), -5.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(sd))]),
    )
    .unwrap();
    let priors = Priors::new(
        &["a".to_string()],
        &[(-5.0, 5.0)],
        &[("a".to_string(), Family::Normal { mean: 0.5, sd: 1.0 })],
    )
    .unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();
    let ends = [vec![mean.sqrt()], vec![-mean.sqrt()]];

    let run = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 3),
        1.0,
        &ends,
        4000,
        CurvatureKind::GaussNewton,
        3,
        &NoSteps,
    )
    .unwrap();

    // The oracle: the posterior mass above 0, by the midpoint rule on a fine
    // grid of the prior times the likelihood.
    let log_posterior = |a: f64| {
        let prior = -0.5 * ((a - 0.5) / 1.0_f64).powi(2);
        let likelihood: f64 = values
            .iter()
            .map(|y| -0.5 * ((y - a * a) / sd).powi(2))
            .sum();
        prior + likelihood
    };
    let grid: Vec<f64> = (0..200_000)
        .map(|i| -5.0 + (i as f64 + 0.5) * 5e-5)
        .collect();
    let densities: Vec<f64> = grid.iter().map(|&a| log_posterior(a).exp()).collect();
    let total: f64 = densities.iter().sum();
    let above: f64 = grid
        .iter()
        .zip(&densities)
        .filter(|(a, _)| **a > 0.0)
        .map(|(_, d)| d)
        .sum();
    let expected = above / total;

    assert_eq!(run.components.len(), 2);
    let share = positive_share(&run.sample.values, &run.sample.log_weights);
    let error = (expected * (1.0 - expected) / run.sample.effective_size).sqrt();
    assert!(
        (share - expected).abs() < 4.0 * error,
        "{share} against {expected} ± {error}"
    );
    assert!(run.reliable, "{}", run.sample.k_hat);
}

#[test]
fn a_component_reads_the_curvature_of_the_target() {
    // The target is 3 times as curved as the objective.
    let problem = line_problem();
    let names = problem.parameter_names();
    let bounds = problem.parameter_bounds();
    let fit = [1.1, 0.48, 0.4];
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let curvature = Curvature::gauss_newton(&problem, &source, &fit).unwrap();
    let priors = Priors::uniform_on_bounds(&names, &bounds).unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();

    let run = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 3),
        3.0,
        &[fit.to_vec()],
        100,
        CurvatureKind::GaussNewton,
        3,
        &NoSteps,
    )
    .unwrap();

    let laplace = laplace(&curvature).unwrap();
    let expected = laplace.covariance() / 3.0;
    let covariance = run.components[0].gaussian.covariance();
    assert!((covariance - &expected).amax() < 1e-12 * expected.amax());
}

/// `x = a²` observed near 4 with `a` in `[−5, 1.5]`: the positive optimum
/// 2 lies beyond the upper bound, so a start ends on it.
fn square_on_a_bound() -> (CalibrationProblem<Curve>, Priors) {
    let values: Vec<f64> = (0..10).map(|i| 4.0 + [0.2, -0.3, 0.1][i % 3]).collect();
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Square),
        observed("x", &values),
        vec![CalibrationParameter::new("a".to_string(), -5.0, 1.5)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let priors = Priors::uniform_on_bounds(&["a".to_string()], &[(-5.0, 1.5)]).unwrap();
    (problem, priors)
}

#[test]
fn an_end_on_a_bound_worse_than_every_component_is_only_noted() {
    let (problem, priors) = square_on_a_bound();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();

    // The end on the bound fits worse than the optimum −2.
    let run = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 3),
        1.0,
        &[vec![-2.0], vec![1.5]],
        400,
        CurvatureKind::GaussNewton,
        3,
        &NoSteps,
    )
    .unwrap();

    assert_eq!(run.components.len(), 1);
    assert!(
        run.notes
            .iter()
            .any(|note| note.contains("is on its bound 1.5"))
    );
    assert!(run.reliable, "{:?}", run.notes);
}

#[test]
fn an_end_on_a_bound_better_than_a_component_marks_the_sample_unreliable() {
    let (problem, priors) = square_on_a_bound();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();

    // The kept end −1 fits worse than the end on the bound: the mixture
    // misses a mode at least as good as one it holds.
    let run = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 3),
        1.0,
        &[vec![-1.0], vec![1.5]],
        400,
        CurvatureKind::GaussNewton,
        3,
        &NoSteps,
    )
    .unwrap();

    assert_eq!(run.components.len(), 1);
    assert!(!run.reliable);
    assert!(
        run.notes
            .iter()
            .any(|note| note.contains("misses an optimum that gives no component")),
        "{:?}",
        run.notes
    );
}

#[test]
fn banana_target_gives_a_large_k_hat() {
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Banana),
        vec![
            ObservedDataPoint::new(0.0, "x".to_string(), 0.0),
            ObservedDataPoint::new(0.0, "y".to_string(), 0.0),
        ],
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -50.0, 10.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![
            normal("x", Argument::Value(2.0)),
            normal("y", Argument::Value(0.1)),
        ]),
    )
    .unwrap();
    let names = ["a".to_string(), "b".to_string()];
    let priors = Priors::uniform_on_bounds(&names, &[(-10.0, 10.0), (-50.0, 10.0)]).unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();

    let run = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 5),
        1.0,
        &[vec![0.0, 0.0]],
        2000,
        CurvatureKind::GaussNewton,
        5,
        &NoSteps,
    )
    .unwrap();

    assert!(run.sample.k_hat > 0.7, "{}", run.sample.k_hat);
    assert!(!run.reliable);
    assert!(run.notes.iter().any(|note| note.contains("unreliable")));
}

/// Log weights of `n` draws in a fixed order: `2u − 1`, or `−m·log(1 − u)`
/// (a Pareto of shape `m` in the weights), at `u = (i + ½)/n`, the draw at
/// `37·i mod n` first.
fn reference_log_weights(n: usize, pareto: Option<f64>) -> Vec<f64> {
    let base: Vec<f64> = (0..n)
        .map(|i| {
            let u = (i as f64 + 0.5) / n as f64;
            match pareto {
                None => 2.0 * u - 1.0,
                Some(m) => -(-u).ln_1p() * m,
            }
        })
        .collect();
    (0..n).map(|i| base[(i * 37) % n]).collect()
}

/// The smoothed log weights of the largest draws, ascending by their log
/// weight before smoothing, and of the smallest draw.
fn tail_and_smallest(log_weights: &[f64], smoothed: &[f64], tail: usize) -> (Vec<f64>, f64) {
    let mut order: Vec<usize> = (0..log_weights.len()).collect();
    order.sort_by(|&i, &j| log_weights[i].total_cmp(&log_weights[j]));
    let tail_values = order[order.len() - tail..]
        .iter()
        .map(|&i| smoothed[i])
        .collect();
    (tail_values, smoothed[order[0]])
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{actual} differs from {expected}"
    );
}

#[test]
fn psis_matches_the_reference_values() {
    // Computed once with ArviZ 1.3.0 (arviz-stats 1.3.3), not a dependency
    // of commol: `arviz_stats.base.array_stats.psislw(-log_weights)`, whose
    // argument is the log-likelihood of LOO, the negated log weights.
    let uniform = reference_log_weights(100, None);
    let smoothed = smooth(&uniform).unwrap();
    assert_close(smoothed.k_hat, -0.31684968871602176);
    let (tail, smallest) = tail_and_smallest(&uniform, &smoothed.log_weights, 20);
    let expected = [
        -4.17354114557456,
        -4.153234386138045,
        -4.132590297543125,
        -4.111566393789103,
        -4.090113532336781,
        -4.068174300791891,
        -4.045680874289283,
        -4.0225521160040065,
        -3.998689568442881,
        -3.9739717725222343,
        -3.9482459813781157,
        -3.92131565475853,
        -3.892920794973578,
        -3.862705429750347,
        -3.8301603201312235,
        -3.7945133259372117,
        -3.783579793532023,
        -3.783579793532023,
        -3.783579793532023,
        -3.783579793532023,
    ];
    for (actual, expected) in tail.iter().zip(expected) {
        assert_close(*actual, expected);
    }
    assert_close(smallest, -5.763579793532022);

    let cases = [
        (
            1000,
            0.5,
            94,
            0.4966301751636287,
            -6.408328992825719,
            [-4.598972185995217, -4.344806705667466, -3.7985127731413586],
            -7.590339627135899,
        ),
        (
            400,
            0.9,
            60,
            0.8183704168942164,
            -5.9091529614789655,
            [-3.1968997692663184, -2.7757820590857962, -1.873210090064678],
            -7.615747553918605,
        ),
    ];
    for (n, shape, tail_count, k_hat, first, largest, smallest) in cases {
        let log_weights = reference_log_weights(n, Some(shape));
        let smoothed = smooth(&log_weights).unwrap();
        assert_close(smoothed.k_hat, k_hat);
        let (tail, lowest) = tail_and_smallest(&log_weights, &smoothed.log_weights, tail_count);
        assert_close(tail[0], first);
        for (actual, expected) in tail[tail_count - 3..].iter().zip(largest) {
            assert_close(*actual, expected);
        }
        assert_close(lowest, smallest);
    }
}

#[test]
fn every_scale_maps_the_curvature_by_its_derivative() {
    // `a` has no bound (the identity) and `b` a lower bound only (a log), so
    // `D = (1, b + 10)`.
    let values: Vec<f64> = (0..10).map(|t| 1.0 + 0.5 * t as f64).collect();
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &values),
        vec![
            CalibrationParameter::new("a".to_string(), f64::NEG_INFINITY, f64::INFINITY),
            CalibrationParameter::new("b".to_string(), -10.0, f64::INFINITY),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let fit = [1.0, 0.5];
    let d = [1.0, 10.5];
    let mut expected = [[0.0; 2]; 2];
    for t in 0..10 {
        let row = [1.0, t as f64];
        for j in 0..2 {
            for k in 0..2 {
                expected[j][k] += row[j] * row[k] / 0.25 * d[j] * d[k];
            }
        }
    }

    for curvature in [
        Curvature::gauss_newton(&problem, &source, &fit).unwrap(),
        Curvature::hessian(&problem, &source, &fit).unwrap(),
    ] {
        let matrix = on_the_unbounded_scale(&curvature);
        for j in 0..2 {
            for k in 0..2 {
                let scale = (expected[j][j] * expected[k][k]).sqrt();
                assert!(
                    (matrix[(j, k)] - expected[j][k]).abs() < 1e-5 * scale,
                    "({j}, {k}): {matrix}"
                );
            }
        }
    }
    let error = Curvature::gauss_newton(&problem, &source, &[1.0, -10.0]).unwrap_err();
    assert!(
        error.to_string().contains("'b' is on its bound -10"),
        "{error}"
    );
}

#[test]
fn a_parameter_the_data_do_not_read_has_no_laplace_approximation() {
    // One observation at t = 0 reads a but not b.
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &[1.0]),
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let curvature = Curvature::gauss_newton(&problem, &source, &[1.0, 0.0]).unwrap();

    let error = laplace(&curvature).unwrap_err();
    assert!(matches!(error, MethodError::Invalid(_)));
    assert!(
        error.to_string().contains("not positive definite"),
        "{error}"
    );
    let singular = nalgebra::DMatrix::zeros(2, 2);
    let transform = Transform::new(&problem.parameter_bounds());
    let error = Gaussian::new(transform, vec![0.0, 0.0], singular, None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("covariance is not positive definite")
    );
}

/// A Poisson model of the line, `a` in `[−1, 5]`, of the counts `values`.
fn poisson_line(values: &[f64]) -> CalibrationProblem<Curve> {
    CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", values),
        vec![CalibrationParameter::new("a".to_string(), -1.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![("x".to_string(), ObservationModel::Poisson)]),
    )
    .unwrap()
}

#[test]
fn a_mean_on_the_edge_of_its_model_has_no_finite_curvature() {
    // A Poisson mean of 0 has the information 1/0.
    let problem = poisson_line(&[0.0, 0.0, 0.0]);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

    let error = Curvature::gauss_newton(&problem, &source, &[0.0]).unwrap_err();

    assert!(matches!(error, MethodError::Failed(_)));
    assert_eq!(
        error.to_string(),
        "The curvature at the fit is not finite: a mean sits on the edge of the means its \
         observation model allows"
    );
}

/// The Poisson counts of `x = a·t`, `a` in `[−1, 5]`, `b = 0`: `first` at
/// `t = 0`, where the mean is 0 with no slope, then 1, 2 and 4.
fn counts_from_zero(first: f64) -> CalibrationProblem<Curve> {
    CalibrationProblem::new(
        Curve::new(Shape::Quadratic),
        observed("x", &[first, 1.0, 2.0, 4.0]),
        vec![CalibrationParameter::new("a".to_string(), -1.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![("x".to_string(), ObservationModel::Poisson)]),
    )
    .unwrap()
}

#[test]
fn a_count_of_zero_at_a_mean_of_zero_adds_no_score() {
    // The fit of the counts 1, 2 and 4 of means a, 2a and 3a: a = 7/6.
    let problem = counts_from_zero(0.0);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

    let curvature = Curvature::gauss_newton(&problem, &source, &[7.0 / 6.0]).unwrap();

    let gradients = curvature.scaled_gradients();
    assert_eq!(gradients[(0, 0)], 0.0);
    // The other terms keep their scores, which sum to 0 at the fit.
    assert!(
        gradients.iter().all(|entry| entry.is_finite()),
        "{gradients}"
    );
    let total: f64 = gradients.iter().sum();
    assert!(total.abs() < 1e-6, "{total}");
}

#[test]
fn a_count_with_no_probability_at_its_mean_has_no_curvature() {
    // A count of 1 at a mean of 0: the score 1/0 − 1, although the slope of
    // the mean is 0.
    let problem = counts_from_zero(1.0);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

    let error = Curvature::gauss_newton(&problem, &source, &[1.0]).unwrap_err();

    assert!(matches!(error, MethodError::Invalid(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .starts_with("Observation 'x' at 0 has no probability at the fit"),
        "{error}"
    );
}

/// Finite differences of a problem whose sensitivities `change` changes.
struct Changed<F: Fn(&mut Sensitivities) + Sync> {
    source: FiniteDifferences<Curve>,
    change: F,
}

impl<F: Fn(&mut Sensitivities) + Sync> SensitivitySource for Changed<F> {
    fn parameters(&self) -> &[CalibrationParameter] {
        self.source.parameters()
    }

    fn jacobian(&self, values: &[f64]) -> Result<Sensitivities, SensitivityError> {
        let mut sensitivities = self.source.jacobian(values)?;
        (self.change)(&mut sensitivities);
        Ok(sensitivities)
    }

    fn scales(&self, sensitivities: &Sensitivities) -> Vec<f64> {
        self.source.scales(sensitivities)
    }
}

/// The message of the Gauss–Newton curvature of `problem` at `fit` with the
/// finite differences that `change` changes, of kind [`MethodError::Failed`].
fn changed_failure(
    problem: &CalibrationProblem<Curve>,
    fit: &[f64],
    change: impl Fn(&mut Sensitivities) + Sync,
) -> String {
    let source = FiniteDifferences::new(problem, Design::Observations).unwrap();
    match Curvature::gauss_newton(problem, &Changed { source, change }, fit) {
        Err(MethodError::Failed(message)) => message,
        other => panic!("{other:?}"),
    }
}

/// [`line_problem`] with the weight `weight` for its observation at 3.
fn line_of_weight(weight: f64) -> CalibrationProblem<Curve> {
    let problem = line_problem();
    let values: Vec<f64> = problem.observed_data().iter().map(|o| o.value).collect();
    let mut weights = [1.0; 10];
    weights[3] = weight;
    problem.with_data(&values, &weights).unwrap()
}

#[test]
fn a_jacobian_that_is_not_finite_is_refused() {
    // The term of weight 0 has no curvature to show it, only its gradient.
    for weight in [0.0, 1.0] {
        let message = changed_failure(&line_of_weight(weight), &[1.1, 0.48, 0.4], |s| {
            s.jacobian[(3, 1)] = f64::INFINITY;
            s.parts.jacobian[(3, 1)] = f64::INFINITY;
        });

        assert_eq!(
            message,
            "The derivatives of the runs at the fit are not finite: observation 'x' at 3 has \
             the derivative inf with respect to 'b'"
        );
    }
}

#[test]
fn a_prediction_that_is_not_finite_is_refused() {
    // The curvature of a loss does not read the predictions, its gradient
    // does.
    for weight in [0.0, 1.0] {
        let values = [1.0, 1.5, 2.2, 2.4];
        let problem = CalibrationProblem::new(
            Curve::new(Shape::Line),
            observed("x", &values),
            vec![
                CalibrationParameter::new("a".to_string(), -10.0, 10.0),
                CalibrationParameter::new("b".to_string(), -5.0, 5.0),
            ],
            None,
            vec![],
            LossConfig::SumSquaredError,
        )
        .unwrap()
        .with_data(&values, &[1.0, 1.0, 1.0, weight])
        .unwrap();

        let message = changed_failure(&problem, &[1.0, 0.5], |s| {
            s.predictions[3] = f64::INFINITY;
        });

        assert_eq!(
            message,
            "The predictions of the runs at the fit are not finite: observation 'x' at 3 has \
             the prediction inf"
        );
    }

    // The increment of x = a t + b t² over 10, a sum of two Poisson parts.
    let sum = ObservedDataPoint {
        parts: 2,
        ..ObservedDataPoint::with_window(10.0, "x".to_string(), 40.0, 10.0)
    };
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Quadratic),
        vec![sum],
        vec![CalibrationParameter::new("a".to_string(), 0.1, 9.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![("x".to_string(), ObservationModel::Poisson)]),
    )
    .unwrap();
    let message = changed_failure(&problem, &[4.0], |s| s.predictions[0] = f64::NAN);
    assert!(
        message.ends_with("observation 'x' at 10 has the prediction NaN"),
        "{message}"
    );
    // An infinite mean of a Poisson part has the information 0.
    let message = changed_failure(&problem, &[4.0], |s| s.parts.means[1] = f64::INFINITY);
    assert!(
        message.ends_with("observation 'x' at 10 has the prediction inf in its part 2"),
        "{message}"
    );
}

/// A line of `x` at 0, 1, 2 with a normal model of sd 0.5, and at 3 the
/// series `far`, `scale` times the line, of weight `weight` and sd `sd`.
fn line_and_far(scale: f64, weight: f64, sd: f64) -> CalibrationProblem<Curve> {
    let values = [0.1, 0.2, 0.1, 1.0];
    let mut data = observed("x", &values);
    data[3].label = "far".to_string();
    data[3].scale = scale;
    CalibrationProblem::new(
        Curve::new(Shape::Line),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), -1.0, 1.0),
            CalibrationParameter::new("b".to_string(), -1.0, 1.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![
            normal("x", Argument::Value(0.5)),
            normal("far", Argument::Value(sd)),
        ]),
    )
    .unwrap()
    .with_data(&values, &[1.0, 1.0, 1.0, weight])
    .unwrap()
}

#[test]
fn a_difference_of_finite_runs_that_overflows_is_refused() {
    // 10³⁰⁸ (a + b t) at t = 3 and (a, b) near (0, 0.01): the runs of a
    // difference of b are finite, and their slope 3·10³⁰⁸ is above the
    // largest number. A weight of 0, or a noise whose information 1/sd²
    // underflows, gives a term of no curvature.
    let fit = [0.0, 0.01];
    for (weight, sd) in [(0.0, 0.5), (1.0, 0.5), (1.0, 1e308)] {
        let problem = line_and_far(1e308, weight, sd);
        let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
        let sensitivities = source.jacobian(&fit).unwrap();
        assert!(sensitivities.predictions.iter().all(|p| p.is_finite()));
        assert_eq!(sensitivities.jacobian[(3, 1)], f64::INFINITY);

        for error in [
            Curvature::gauss_newton(&problem, &source, &fit).unwrap_err(),
            Curvature::hessian(&problem, &source, &fit).unwrap_err(),
            sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 1).unwrap_err(),
        ] {
            assert_eq!(
                error,
                MethodError::Failed(
                    "The derivatives of the runs at the fit are not finite: observation 'far' \
                     at 3 has the derivative inf with respect to 'b'"
                        .to_string()
                )
            );
        }
    }
}

#[test]
fn a_loss_with_no_curvature_is_refused_before_a_derivative_that_is_not_finite() {
    // The same derivative of 3·10³⁰⁸, under the mean absolute error, which no
    // change of units gives a curvature.
    let mut data = observed("x", &[0.1, 0.2, 0.1, 1.0]);
    data[3].scale = 1e308;
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), -1.0, 1.0),
            CalibrationParameter::new("b".to_string(), -1.0, 1.0),
        ],
        None,
        vec![],
        LossConfig::MeanAbsoluteError,
    )
    .unwrap();

    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let error = Curvature::gauss_newton(&problem, &source, &[0.0, 0.01]).unwrap_err();

    assert!(matches!(error, MethodError::Invalid(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .starts_with("loss_function=\"mae\" has no curvature"),
        "{error}"
    );
}

/// `x = a + b·t` at the positions 0 to 8 with noise of sd 0.5, and at 4 the
/// series `far` of `y = b`, `scale` times its value, with noise of sd 1, seen
/// at its prediction at `fit`.
fn line_and_slope(scale: f64, fit: [f64; 2]) -> CalibrationProblem<Curve> {
    line_and_slope_of_sd(scale, fit, 1.0)
}

/// [`line_and_slope`] with noise of sd `sd` for far.
fn line_and_slope_of_sd(scale: f64, fit: [f64; 2], sd: f64) -> CalibrationProblem<Curve> {
    line_and_slope_at(scale, fit, sd, &[4])
}

/// [`line_and_slope_of_sd`] with far seen at each position of `far`.
fn line_and_slope_at(
    scale: f64,
    fit: [f64; 2],
    sd: f64,
    far: &[usize],
) -> CalibrationProblem<Curve> {
    let values = [0.1, 0.3, 0.2, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9];
    let mut data = observed("x", &values);
    for &t in far {
        let mut point = ObservedDataPoint::new(t as f64, "y".to_string(), scale * fit[1]);
        point.label = "far".to_string();
        point.scale = scale;
        data.push(point);
    }
    CalibrationProblem::new(
        Curve::new(Shape::LineAndSlope),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), -1.0, 1.0),
            CalibrationParameter::new("b".to_string(), -1.0, 1.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![
            normal("x", Argument::Value(0.5)),
            normal("far", Argument::Value(sd)),
        ]),
    )
    .unwrap()
}

#[test]
fn a_covariance_is_given_until_the_curvature_is_too_large_for_it() {
    // The curvature of b is about scale², so its variance is about 1/scale²
    // in the Laplace approximation, a number below the smallest normal one at
    // 1e157 and below any number at 1e200. In the sandwich, which reads the
    // scores of x alone, it is about 1/scale⁴, below the smallest normal number
    // at 1e78; there far is seen at every position, so that every block holds
    // a share of the curvature of b. With bounds that do not move, a
    // derivative whose square is above the largest number is read in its unit.
    let fit = [0.05, 0.1];
    let refused = MethodError::Failed(
        "The curvature of 'b' at the fit is too large for its covariance to be a number: its \
         variance on the unbounded scale is below the smallest number"
            .to_string(),
    );
    let laplace_of = |scale: f64| {
        let problem = line_and_slope(scale, fit);
        let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
        laplace(&Curvature::gauss_newton(&problem, &source, &fit).unwrap())
    };
    let sandwich_of = |scale: f64| {
        let problem = line_and_slope_at(scale, fit, 1.0, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 1)
    };
    let eta_of = |scale: f64| information_matching_eta(&line_and_slope(scale, fit), &fit, 1);

    let variance = laplace_of(1e157).unwrap().covariance()[(1, 1)];
    assert!(variance > 0.0 && variance < f64::MIN_POSITIVE, "{variance}");
    let variance = sandwich_of(1e78).unwrap().gaussian.covariance()[(1, 1)];
    assert!(variance > 0.0 && variance < f64::MIN_POSITIVE, "{variance}");
    for scale in [1e200, 1e300] {
        assert_eq!(laplace_of(scale).unwrap_err(), refused, "{scale:e}");
        assert_eq!(sandwich_of(scale).unwrap_err(), refused, "{scale:e}");
    }
    let expected = eta_of(1e10).unwrap();
    for scale in [1e157, 1e200, 1e300] {
        let calibration = eta_of(scale).unwrap();
        assert!(
            (calibration.eta / expected.eta - 1.0).abs() < 1e-12,
            "{scale:e}: {} against {}",
            calibration.eta,
            expected.eta
        );
    }
}

#[test]
fn a_hessian_steps_where_its_objective_is_a_number() {
    // At a scale of 1e159 the curvature of b on the unbounded scale is about
    // 2.5e317: a step of δ^{1/4} there moves the objective above the largest
    // number, while the variance, about 4e-318, is a number.
    let fit = [0.05, 0.1];
    let problem = line_and_slope(1e159, fit);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let expected = laplace(&Curvature::gauss_newton(&problem, &source, &fit).unwrap()).unwrap();

    let curvature = Curvature::hessian(&problem, &source, &fit).unwrap();

    let covariance = laplace(&curvature).unwrap().covariance().clone();
    let expected = expected.covariance();
    assert!(expected[(1, 1)] > 0.0 && expected[(1, 1)] < 1e-317);
    for j in 0..2 {
        let relative = (covariance[(j, j)] / expected[(j, j)] - 1.0).abs();
        assert!(relative < 1e-5, "{j}: {covariance} against {expected}");
    }
    // At a corner of (a, b) the objective of far is about 1e306, which rounds
    // away what x adds: the note says that the coupling is not read.
    let note = "The curvature of ('a', 'b') has an estimated error of more than 100% of the \
                scale of its row and column";
    assert_eq!(curvature.notes(), [note]);
}

#[test]
fn a_hessian_that_cannot_step_says_that_the_curvature_is_too_large() {
    // At a scale of 1e300 the objective one rounding of b away from the fit
    // is about 1e566: no step that moves b keeps it a number. With an sd of
    // 1e-150 for far the scale of b, about 5e449, is above the largest
    // number, and the bound of a step reads only its unit.
    let fit = [0.05, 0.1];
    for sd in [1.0, 1e-150] {
        let problem = line_and_slope_of_sd(1e300, fit, sd);
        let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

        let error = Curvature::hessian(&problem, &source, &fit).unwrap_err();

        assert_eq!(error, too_large_for_the_hessian("'b'"), "{sd:e}");
    }
}

/// The refusal of a Hessian whose curvature of `b` is too large for it, at
/// steps within the rounding of `what`.
fn too_large_for_the_hessian(what: &str) -> MethodError {
    MethodError::Failed(format!(
        "The curvature of 'b' at the fit is too large for the Hessian: a step that keeps its \
         objective well below the largest number is within the rounding of {what}"
    ))
}

#[test]
fn a_hessian_reads_no_step_within_the_rounding_of_its_parameter_or_its_runs() {
    // An unbounded b at 1e12, whose rounding is 2⁻¹³, with x on the line and
    // far at its prediction. At 1e148 the shorter step is 2¹⁷. At 1e157 it is
    // 2⁻¹³, whose points are exact, and the runs of far, 1e169, round by about
    // their change; at 3e157 it is 2⁻¹⁴, and a point of it rounds.
    let fit = [0.05, 1e12];
    let problem_at = |scale: f64| {
        let values: Vec<f64> = (0..9).map(|t| fit[0] + fit[1] * t as f64).collect();
        let mut data = observed("x", &values);
        let mut far = ObservedDataPoint::new(4.0, "y".to_string(), scale * fit[1]);
        far.label = "far".to_string();
        far.scale = scale;
        data.push(far);
        CalibrationProblem::new(
            Curve::new(Shape::LineAndSlope),
            data,
            vec![
                CalibrationParameter::new("a".to_string(), -1.0, 1.0),
                CalibrationParameter::new("b".to_string(), f64::NEG_INFINITY, f64::INFINITY),
            ],
            None,
            vec![],
            DiscrepancyConfig::Likelihood(vec![
                normal("x", Argument::Value(0.5)),
                normal("far", Argument::Value(1.0)),
            ]),
        )
        .unwrap()
    };
    let variances = |scale: f64| {
        let problem = problem_at(scale);
        let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
        let expected = Curvature::gauss_newton(&problem, &source, &fit).unwrap();
        let variance = |curvature: &Curvature| laplace(curvature).unwrap().covariance()[(1, 1)];
        Curvature::hessian(&problem, &source, &fit)
            .map(|curvature| (variance(&curvature), variance(&expected)))
    };

    let (variance, expected) = variances(1e148).unwrap();
    assert!(
        (variance / expected - 1.0).abs() < 1e-6,
        "{variance} against {expected}"
    );
    let runs = "the predictions of the model at the fit";
    for (scale, what) in [(1e157, runs), (3e157, "'b'")] {
        assert_eq!(
            variances(scale).unwrap_err(),
            too_large_for_the_hessian(what),
            "{scale:e}"
        );
    }
}

#[test]
fn a_sandwich_whose_scores_are_all_0_is_refused_by_its_covariance() {
    // The data lie on the line: every residual and every block score at the
    // fit is 0, so the sandwich is 0, which is no curvature too large for it.
    // Blocks of observations of weight 0 add no curvature and no score.
    let values: Vec<f64> = (0..8).map(|t| 1.0 + 0.5 * t as f64).collect();
    for silent in [0, 4] {
        let problem = line_with_silent(&values, silent, 0.0);

        let error = sandwich_approximation(&problem, &[1.0, 0.5], CurvatureKind::GaussNewton, 2)
            .unwrap_err();

        assert!(matches!(error, MethodError::Invalid(_)), "{error:?}");
        assert!(
            error
                .to_string()
                .starts_with("The covariance is not positive definite"),
            "{silent}: {error}"
        );
    }
}

#[test]
fn a_term_of_no_weight_adds_nothing_whatever_its_derivative() {
    // The derivative 3·10³⁰⁷ of the term of weight 0 is a number, and no
    // number in the unit of b, whose other derivatives are at most 2.
    let curvature = |scale: f64| {
        let values = [0.1, 0.2, 0.1, 1.0];
        let mut data = observed("x", &values);
        data[3].label = "far".to_string();
        data[3].scale = scale;
        let problem = CalibrationProblem::new(
            Curve::new(Shape::Line),
            data,
            vec![
                CalibrationParameter::new("a".to_string(), -1.0, 1.0),
                CalibrationParameter::new("b".to_string(), -1.0, 1.0),
            ],
            None,
            vec![],
            DiscrepancyConfig::Likelihood(vec![
                normal("x", Argument::Value(100.0)),
                normal("far", Argument::Value(1.0)),
            ]),
        )
        .unwrap()
        .with_data(&values, &[1.0, 1.0, 1.0, 0.0])
        .unwrap();
        let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
        Curvature::gauss_newton(&problem, &source, &[0.0, 0.01]).unwrap()
    };
    let (expected, far) = (curvature(1.0), curvature(1e307));

    assert_eq!(far.scaled_matrix(), expected.scaled_matrix());
    assert_eq!(far.scaled_gradients(), expected.scaled_gradients());
    assert_eq!(far.scales(), expected.scales());
}

#[test]
fn a_posterior_is_minus_infinity_outside_its_support() {
    // A Poisson mean below 0 is outside the model.
    let problem = poisson_line(&[1.0, 2.0]);
    let priors = Priors::uniform_on_bounds(&["a".to_string()], &[(-1.0, 5.0)]).unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();
    let posterior = TemperedPosterior::new(&target, 1);
    let mut workspace = Workspace::default();

    assert_eq!(posterior.dimension(), 1);
    assert_eq!(
        posterior.log_density(&[-0.5], &mut workspace),
        Ok(f64::NEG_INFINITY)
    );
    let inside = posterior.log_density(&[1.5], &mut workspace).unwrap();
    assert!(inside.is_finite());

    // An end outside the support, and one whose curvature is 0, give no
    // component; with no other end there is no mixture.
    let error = laplace_mixture(
        &problem,
        &posterior,
        1.0,
        &[vec![-0.5]],
        100,
        CurvatureKind::GaussNewton,
        1,
        &NoSteps,
    )
    .unwrap_err();
    assert!(error.to_string().contains("outside the support"), "{error}");
}

#[test]
fn an_end_whose_curvature_is_singular_gives_no_component() {
    // x = a² at a = 0 has the slope 0: the curvature of a is 0 there.
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Square),
        observed("x", &[4.0, 4.2, 3.8]),
        vec![CalibrationParameter::new("a".to_string(), -5.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let priors = Priors::uniform_on_bounds(&["a".to_string()], &[(-5.0, 5.0)]).unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();

    let run = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 1),
        1.0,
        &[vec![2.0], vec![0.0]],
        200,
        CurvatureKind::GaussNewton,
        1,
        &NoSteps,
    )
    .unwrap();

    assert_eq!(run.components.len(), 1);
    assert!(
        run.notes
            .iter()
            .any(|note| note.contains("[0.0] gives no component"))
    );
}

#[test]
fn a_refused_end_that_fits_best_marks_the_sample_unreliable() {
    // x = a² observed near 0: the end 0 fits best and its curvature is 0, so
    // the mixture keeps the end 1 alone and misses the optimum.
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Square),
        observed("x", &[0.1, -0.1, 0.05]),
        vec![CalibrationParameter::new("a".to_string(), -5.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let priors = Priors::uniform_on_bounds(&["a".to_string()], &[(-5.0, 5.0)]).unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();

    let run = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 1),
        1.0,
        &[vec![0.0], vec![1.0]],
        200,
        CurvatureKind::GaussNewton,
        1,
        &NoSteps,
    )
    .unwrap();

    assert_eq!(run.components.len(), 1);
    assert_eq!(run.components[0].center, vec![1.0]);
    assert!(!run.reliable);
    assert!(
        run.notes.iter().any(|note| note.starts_with(
            "The mixture misses an optimum that gives no component and fits at least as \
             well as its worst component"
        ) && note.contains("[0.0]")),
        "{:?}",
        run.notes
    );
}

/// The increments of [`Shape::Wave`] over the windows of the positions 1 to
/// 30, each split into `parts` parts, at `fit` and with noise.
fn wave_in_parts(parts: usize, fit: [f64; 2], seed: u64) -> CalibrationProblem<Curve> {
    let tau = std::f64::consts::TAU;
    let total = |t: f64| fit[0] * (t + fit[1] * (1.0 - (tau * t).cos()) / tau);
    let mut generator = commol_random::Generator::from_seed(seed);
    let data = (1..=30)
        .map(|n| {
            let at = n as f64;
            let value = total(at) - total(at - 1.0) + 0.1 * (generator.open_uniform() - 0.5);
            let mut point = ObservedDataPoint::with_window(at, "x".to_string(), value, 1.0);
            point.parts = parts;
            point
        })
        .collect();
    CalibrationProblem::new(
        Curve::new(Shape::Wave),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), 0.1, 3.0),
            CalibrationParameter::new("b".to_string(), -1.0, 1.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.3))]),
    )
    .unwrap()
}

#[test]
fn a_parameter_read_only_by_parts_that_cancel_is_refused_by_every_method() {
    // The total of each window does not read b, and each of its parts does:
    // the curvature of b sums terms that cancel to a rounding of their size.
    let mut generator = commol_random::Generator::from_seed(9);
    for parts in [3, 12] {
        for seed in 0..10 {
            let fit = [
                0.5 + generator.open_uniform(),
                -0.8 + 1.6 * generator.open_uniform(),
            ];
            let problem = wave_in_parts(parts, fit, seed);
            let priors =
                Priors::uniform_on_bounds(&problem.parameter_names(), &problem.parameter_bounds())
                    .unwrap();
            let target = LikelihoodTarget::new(&problem, &priors).unwrap();
            let curvature = curvature_at(&problem, &fit, CurvatureKind::GaussNewton).unwrap();
            let mixture = laplace_mixture(
                &problem,
                &TemperedPosterior::new(&target, 1),
                1.0,
                &[fit.to_vec()],
                100,
                CurvatureKind::GaussNewton,
                1,
                &NoSteps,
            );
            let sandwich = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 2);

            assert_eq!(curvature.scaled_matrix()[(1, 1)], 0.0, "{parts} at {fit:?}");
            for (method, outcome) in [
                ("laplace", laplace(&curvature).err()),
                ("sandwich", sandwich.err()),
                ("mixture", mixture.err()),
                (
                    "information matching",
                    information_matching_eta(&problem, &fit, 2).err(),
                ),
            ] {
                let error = outcome.unwrap_or_else(|| panic!("{method}, {parts} at {fit:?}"));
                assert!(
                    matches!(error, MethodError::Invalid(_))
                        && error.to_string().contains(
                            "The curvature at the fit is not positive definite: the data"
                        ),
                    "{method}, {parts} at {fit:?}: {error}"
                );
            }
        }
    }
}

/// The values of a line `x = 1 + t/2` at the positions `0..count`, with a
/// residual that repeats every four positions.
fn line_values(count: usize) -> Vec<f64> {
    (0..count)
        .map(|t| 1.0 + 0.5 * t as f64 + [0.3, -0.2, 0.1, -0.4][t % 4])
        .collect()
}

/// The line `x = a + b·t` at the positions 0, 1, … of `values`, with a normal
/// model of sd 1/2, and after them `silent` observations of weight `weight`,
/// one above the line.
fn line_with_silent(values: &[f64], silent: usize, weight: f64) -> CalibrationProblem<Curve> {
    let mut data = observed("x", values);
    for t in values.len()..values.len() + silent {
        let mut point = ObservedDataPoint::new(t as f64, "x".to_string(), 2.0 + 0.5 * t as f64);
        point.weight = weight;
        data.push(point);
    }
    CalibrationProblem::new(
        Curve::new(Shape::Line),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap()
}

/// `Σ_b a_b a_bᵀ` on the unbounded scale, the influences `a_b = H_u⁻¹ s_b`
/// of the blocks of `block_length` positions with no factor, from the
/// curvature of `kind`, its inverse and its block scores.
fn influences<E: SimulationEngine + Send + Sync>(
    problem: &CalibrationProblem<E>,
    fit: &[f64],
    kind: CurvatureKind,
    block_length: usize,
) -> Matrix {
    let curvature = curvature_at(problem, fit, kind).unwrap();
    let blocks = BlockScheme::new(problem.observed_data(), block_length).unwrap();
    let inverse = curvature.inverse().unwrap();
    let scaled = scaled_block_scores(&curvature, &blocks) * inverse.scaled_matrix();
    let scales = curvature.scales();
    Matrix::from_fn(fit.len(), fit.len(), |j, k| {
        scaled.column(j).dot(&scaled.column(k)) / scales[j] / scales[k]
    })
}

/// The factor `f_j = (J⁻¹)_jj / Σ_b v_bb` and the degrees of freedom
/// `ν_j = (Σ_b v_bb)² / Σ_bc v_bc²` of each parameter `j`, with
/// `v_bc = e_jᵀ J⁻¹ (J_b δ_bc − J_b J⁻¹ J_c) J⁻¹ e_j`, from the curvature
/// `J_b` of each block on any scale of the parameters and `J = Σ_b J_b`, by
/// the whole matrix of the `v_bc` (Bell and McCaffrey 2002).
fn bell_mccaffrey(blocks: &[Matrix]) -> (Vec<f64>, Vec<f64>) {
    let p = blocks[0].nrows();
    let total = blocks
        .iter()
        .fold(Matrix::zeros(p, p), |sum, block| sum + block);
    let inverse = total.try_inverse().unwrap();
    (0..p)
        .map(|j| {
            let u = inverse.column(j).clone_owned();
            let v = Matrix::from_fn(blocks.len(), blocks.len(), |b, c| {
                let own = if b == c {
                    u.dot(&(&blocks[b] * &u))
                } else {
                    0.0
                };
                own - u.dot(&(&blocks[b] * &inverse * &blocks[c] * &u))
            });
            let read = v.trace();
            (inverse[(j, j)] / read, read * read / v.norm_squared())
        })
        .unzip()
}

/// The curvature `Σ_{t∈b} [1, t]ᵀ [1, t]` of a line in each block of
/// `length` consecutive positions of `0..positions`, on the scale of `(a, b)`
/// and up to a factor that every block shares.
fn line_blocks(positions: usize, length: usize) -> Vec<Matrix> {
    (0..positions)
        .collect::<Vec<_>>()
        .chunks(length)
        .map(|block| {
            block.iter().fold(Matrix::zeros(2, 2), |sum, &t| {
                let row = Matrix::from_row_slice(1, 2, &[1.0, t as f64]);
                sum + row.transpose() * row
            })
        })
        .collect()
}

/// The largest relative difference between two numbers of `actual` and
/// `expected`.
fn largest_difference<'a>(
    actual: impl IntoIterator<Item = &'a f64>,
    expected: impl IntoIterator<Item = &'a f64>,
) -> f64 {
    actual
        .into_iter()
        .zip(expected)
        .map(|(a, e)| ((a - e) / e).abs())
        .fold(0.0, f64::max)
}

/// `x = a` and `y = b + a²` at six positions with noise of sd 1/2: they have
/// the same derivatives at every position, so each block of two positions
/// holds a third of the curvature.
fn banana_problem() -> CalibrationProblem<Curve> {
    let mut data = observed("x", &[0.9, 1.3, 0.7, 1.2, 1.1, 0.6]);
    data.extend(observed("y", &[1.6, 1.2, 1.9, 1.4, 1.3, 1.7]));
    CalibrationProblem::new(
        Curve::new(Shape::Banana),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![
            normal("x", Argument::Value(0.5)),
            normal("y", Argument::Value(0.5)),
        ]),
    )
    .unwrap()
}

#[test]
fn equal_blocks_give_cr1_and_one_degree_of_freedom_fewer_than_the_blocks() {
    // At the optimum, where the scores of the blocks sum to zero: a is the
    // mean of x and b that of y less a².
    let problem = banana_problem();
    let a = [0.9, 1.3, 0.7, 1.2, 1.1, 0.6].iter().sum::<f64>() / 6.0;
    let fit = [
        a,
        [1.6, 1.2, 1.9, 1.4, 1.3, 1.7].iter().sum::<f64>() / 6.0 - a * a,
    ];

    let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 2).unwrap();

    let expected = 1.5 * influences(&problem, &fit, CurvatureKind::GaussNewton, 2);
    let covariance = run.gaussian.covariance();
    assert!(
        largest_difference(covariance.iter(), expected.iter()) < 1e-12,
        "{covariance}"
    );
    let degrees = run.gaussian.degrees_of_freedom().unwrap();
    assert!(
        largest_difference(degrees, &[2.0, 2.0]) < 1e-12,
        "{degrees:?}"
    );
}

#[test]
fn a_sandwich_reads_the_share_of_the_curvature_of_each_block() {
    // The blocks of 3 of 8 positions of a line hold unequal shares of the
    // curvature of its intercept and its slope, and the last is short: CR1
    // would read 2 degrees of freedom for each. The fit is the optimum,
    // where the scores of the blocks sum to zero.
    let problem = line_with_silent(&line_values(8), 0, 1.0);
    let fit = line_optimum(&line_values(8));

    let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 3).unwrap();

    let (factors, degrees) = bell_mccaffrey(&line_blocks(8, 3));
    let sums = influences(&problem, &fit, CurvatureKind::GaussNewton, 3);
    let expected = Matrix::from_fn(2, 2, |j, k| (factors[j] * factors[k]).sqrt() * sums[(j, k)]);
    let covariance = run.gaussian.covariance();
    assert!(
        largest_difference(covariance.iter(), expected.iter()) < 1e-9,
        "{covariance}"
    );
    let actual = run.gaussian.degrees_of_freedom().unwrap();
    assert!(
        largest_difference(actual, &degrees) < 1e-9,
        "{actual:?} {degrees:?}"
    );
    assert!(
        degrees.iter().all(|nu| (nu - 2.0).abs() > 0.1),
        "{degrees:?}"
    );
}

#[test]
fn a_block_with_no_curvature_changes_nothing() {
    // Three observations of weight w after nine of the line make a fourth
    // block of 3. At w = 0 it has no curvature and no score, and as w goes to
    // 0 the sandwich goes to the one without it.
    let values = line_values(9);
    let fit = [1.0, 0.5];
    let run = |silent: usize, weight: f64| {
        let problem = line_with_silent(&values, silent, weight);
        sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 3).unwrap()
    };
    let without = run(0, 0.0);

    let silent = run(3, 0.0);

    assert_eq!((without.blocks, silent.blocks), (3, 4));
    assert_eq!(silent.gaussian.covariance(), without.gaussian.covariance());
    assert_eq!(
        silent.gaussian.degrees_of_freedom(),
        without.gaussian.degrees_of_freedom()
    );
    for weight in [1e-3, 1e-6, 1e-9] {
        let light = run(3, weight);
        let change = largest_difference(
            light.gaussian.covariance().iter(),
            without.gaussian.covariance().iter(),
        )
        .max(largest_difference(
            light.gaussian.degrees_of_freedom().unwrap(),
            without.gaussian.degrees_of_freedom().unwrap(),
        ));
        assert!(
            change > 0.0 && change < 100.0 * weight,
            "{weight}: {change}"
        );
    }
}

#[test]
fn a_sandwich_needs_more_blocks_with_curvature_than_parameters() {
    // Four observations of weight 0 after eight of the line make a third
    // block of 4, which has no curvature, and so do four of weight 10⁻¹²,
    // whose share of the curvature is about 10⁻¹².
    let fit = [1.0, 0.5];
    for (silent, weight, blocks) in [(0, 0.0, 2), (4, 0.0, 3), (4, 1e-12, 3)] {
        let problem = line_with_silent(&line_values(8), silent, weight);

        let error =
            sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 4).unwrap_err();

        assert_eq!(
            error,
            MethodError::Invalid(format!(
                "block_length 4 makes 2 blocks with curvature, of {blocks} in all, for 2 \
                 parameters: the sum of the scores over blocks needs more blocks with curvature \
                 than parameters, or its rank is below the parameters. Use shorter blocks"
            ))
        );
    }
    let problem = line_with_silent(&line_values(8), 0, 1.0);
    let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 3).unwrap();
    assert_eq!((run.blocks, run.block_length), (3, 3));
}

#[test]
fn blocks_that_each_hold_all_the_curvature_of_a_parameter_are_refused() {
    // x = a is observed only in the first of three blocks and y = b in all
    // three: the block scores of a are its gradient at the fit and 0.
    let mut data = observed("x", &[0.9, 1.3]);
    data.extend(observed("y", &[1.2, 0.8, 1.1, 0.7, 1.3, 0.9]));
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Pair),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![
            normal("x", Argument::Value(0.5)),
            normal("y", Argument::Value(0.5)),
        ]),
    )
    .unwrap();

    let error =
        sandwich_approximation(&problem, &[1.1, 1.0], CurvatureKind::GaussNewton, 2).unwrap_err();

    assert_eq!(
        error,
        MethodError::Invalid(
            "block_length 2 makes blocks that each hold all the curvature of 'a', where its \
             scores sum to zero at the fit. Use shorter blocks"
                .to_string()
        )
    );
}

/// The refusal of a sandwich whose block scores with blocks of `length`
/// positions have rank below the parameters.
fn below_rank(length: usize) -> MethodError {
    MethodError::Invalid(format!(
        "block_length {length} makes block scores of rank below the parameters: in every block \
         the scores of some parameters move together, so the sandwich gives them no variance \
         apart. Use another block length, or a method that does not read block scores, such \
         as Laplace or a bootstrap"
    ))
}

#[test]
fn block_scores_that_move_together_in_every_block_are_refused() {
    // y is `scale` times x and b is `scale` times a at the optimum, so in
    // every block the influence of b is `scale` times that of a: the sandwich
    // has rank 1, which the rounding of its Cholesky factor hides at 0.3 and
    // not at 3. With a off the optimum by `offset`, as an optimizer leaves
    // it, the scores of the blocks move together after one step to the zero
    // of their sum.
    let x = [0.9, 1.3, 0.7, 1.2, 1.1, 0.6];
    let a = x.iter().sum::<f64>() / x.len() as f64;
    for (scale, offset) in [0.3, 3.0]
        .into_iter()
        .flat_map(|scale| OFFSETS.map(|offset| (scale, offset)))
    {
        let y: Vec<f64> = x.iter().map(|value| scale * value).collect();
        let mut data = observed("x", &x);
        data.extend(observed("y", &y));
        let problem = CalibrationProblem::new(
            Curve::new(Shape::Pair),
            data,
            vec![
                CalibrationParameter::new("a".to_string(), -10.0, 10.0),
                CalibrationParameter::new("b".to_string(), -10.0, 10.0),
            ],
            None,
            vec![],
            DiscrepancyConfig::Likelihood(vec![
                normal("x", Argument::Value(0.5)),
                normal("y", Argument::Value(0.5 * scale)),
            ]),
        )
        .unwrap();

        let fit = [a * (1.0 + offset), scale * a];

        let error =
            sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 2).unwrap_err();

        assert_eq!(error, below_rank(2), "{scale} {offset}");
    }
}

#[test]
fn parameters_read_in_fewer_blocks_than_themselves_are_refused() {
    // a and b of x = a + b·t are read only in the first two of four blocks of
    // 2, and c of y = c in all four: four blocks with curvature for three
    // parameters. At the optimum (1, 0.5, 1) the scores of a and b in the
    // two blocks sum to zero, so the block scores have rank 2, and so they do
    // after one step to the zero of their sum with a off the optimum by
    // `offset`.
    let mut data = observed("x", &[1.1, 1.4, 1.9, 2.6]);
    data.extend(observed("y", &[0.9, 1.3, 0.7, 1.2, 1.1, 0.6, 1.0, 1.2]));
    let problem = CalibrationProblem::new(
        Curve::new(Shape::LineAndLevel),
        data,
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
            CalibrationParameter::new("c".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![
            normal("x", Argument::Value(0.5)),
            normal("y", Argument::Value(0.5)),
        ]),
    )
    .unwrap();

    for offset in OFFSETS {
        let fit = [1.0 + offset, 0.5, 1.0];

        let error =
            sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 2).unwrap_err();

        assert_eq!(error, below_rank(2), "{offset}");
    }
}

#[test]
fn the_sandwich_of_a_fit_off_its_optimum_is_that_of_the_optimum() {
    // The line is linear in a and b, so one step to the zero of the sum of
    // the scores reaches the scores of the optimum, and the covariance on the
    // scale of the parameters does not follow the error of the fit: it
    // changes by the error of the derivatives, about 10⁻⁸, where the scores
    // of the fit change it by 6·10⁻⁵ at an offset of 10⁻⁸.
    let values = line_values(8);
    let problem = line_with_silent(&values, 0, 1.0);
    let optimum = line_optimum(&values);
    let on_the_parameters = |fit: [f64; 2]| {
        let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 3).unwrap();
        let d = [
            logit_derivative(fit[0], -10.0, 10.0),
            logit_derivative(fit[1], -5.0, 5.0),
        ];
        Matrix::from_fn(2, 2, |j, k| d[j] * run.gaussian.covariance()[(j, k)] * d[k])
    };
    let expected = on_the_parameters(optimum);

    for offset in OFFSETS {
        let covariance = on_the_parameters(optimum.map(|value| value * (1.0 + offset)));

        let change = largest_difference(covariance.iter(), expected.iter()) / expected.amax();
        assert!(change < 1e-7, "{offset}: {change:e}");
    }
}

/// The relative errors of a fit that an optimizer can leave.
const OFFSETS: [f64; 4] = [0.0, 1e-8, 1e-5, 1e-3];

/// The least-squares fit `(a, b)` of the line `a + b·t` to `values` at the
/// positions 0, 1, ….
fn line_optimum(values: &[f64]) -> [f64; 2] {
    let n = values.len() as f64;
    let t: Vec<f64> = (0..values.len()).map(|t| t as f64).collect();
    let mean_t = t.iter().sum::<f64>() / n;
    let mean = values.iter().sum::<f64>() / n;
    let spread: f64 = t.iter().map(|t| (t - mean_t) * (t - mean_t)).sum();
    let product: f64 = t
        .iter()
        .zip(values)
        .map(|(t, value)| (t - mean_t) * (value - mean))
        .sum();
    let slope = product / spread;
    [mean - slope * mean_t, slope]
}

#[test]
fn a_hessian_sandwich_whose_gauss_newton_curvature_is_singular_says_so() {
    // x = a² at a = 0 has no first derivative, so no block holds a share of
    // the Gauss–Newton curvature, while the Hessian, 4 Σ (a² − x) / sd²
    // from the data below 0, is positive.
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Square),
        observed("x", &[-1.0, -0.5, -1.5, -0.8]),
        vec![CalibrationParameter::new("a".to_string(), -5.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    assert!(
        curvature_at(&problem, &[0.0], CurvatureKind::Hessian)
            .is_ok_and(|curvature| curvature.inverse().is_ok())
    );

    let error = sandwich_approximation(&problem, &[0.0], CurvatureKind::Hessian, 1).unwrap_err();

    assert_eq!(
        error,
        MethodError::Invalid(
            "The Gauss–Newton curvature of the blocks at the fit is not positive definite: the \
             sandwich reads the share of each block in it, and the first derivatives of the \
             model do not determine every parameter there. Calibrator.identifiability(...) \
             shows which parameters the data determine"
                .to_string()
        )
    );
}

#[test]
fn the_shares_of_a_hessian_sandwich_are_those_of_gauss_newton() {
    // The residuals of y sum to 0.1, so the Hessian of a is that of
    // Gauss–Newton less 2 · 0.1 / sd², and its blocks, which share the
    // curvature of Gauss–Newton equally, would not read 2 degrees of freedom.
    let problem = banana_problem();
    let fit = [1.0, 0.5];
    let [expected, hessian] = [CurvatureKind::GaussNewton, CurvatureKind::Hessian]
        .map(|kind| sandwich_approximation(&problem, &fit, kind, 2).unwrap());

    let curvatures = [CurvatureKind::GaussNewton, CurvatureKind::Hessian]
        .map(|kind| on_the_unbounded_scale(&curvature_at(&problem, &fit, kind).unwrap()));
    assert!((&curvatures[0] - &curvatures[1]).amax() > 1e-3 * curvatures[0].amax());
    let degrees = hessian.gaussian.degrees_of_freedom().unwrap();
    let expected = expected.gaussian.degrees_of_freedom().unwrap();
    assert!(
        largest_difference(degrees, expected) < 1e-12,
        "{degrees:?} {expected:?}"
    );
}

/// The line `x = a + unit·b·t` observed at eight positions, with a normal
/// model of sd `sd`, and a point near its fit.
fn line_in_units(unit: f64, sd: f64) -> (CalibrationProblem<Curve>, [f64; 2]) {
    let values: Vec<f64> = (0..8)
        .map(|t| 1.0 + 0.5 * t as f64 + [0.3, -0.2, 0.1, -0.4][t % 4])
        .collect();
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Units(unit)),
        observed("x", &values),
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0 / unit, 5.0 / unit),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(sd))]),
    )
    .unwrap();
    (problem, [1.0, 0.5 / unit])
}

/// The covariances of the Laplace approximation and of the sandwich, with
/// blocks of 2, of [`line_in_units`] with the curvature `kind`.
fn covariances_in_units(unit: f64, sd: f64, kind: CurvatureKind) -> [Matrix; 2] {
    let (problem, fit) = line_in_units(unit, sd);
    let curvature = curvature_at(&problem, &fit, kind).unwrap();
    let sandwich = sandwich_approximation(&problem, &fit, kind, 2).unwrap();
    [
        laplace(&curvature).unwrap().covariance().clone(),
        sandwich.gaussian.covariance().clone(),
    ]
}

#[test]
fn a_covariance_does_not_depend_on_the_unit_of_a_parameter() {
    // A unit that is a power of two rounds nothing, and the bounds of b move
    // with it, so on the unbounded scale every number is the same, bit for
    // bit. On the scale of b its curvature is below the smallest normal number
    // at 2⁻⁵³⁰, below any number at 2⁻¹⁰⁰⁰, and above the largest from 2⁵¹⁰.
    for (kind, exponents) in [
        (
            CurvatureKind::GaussNewton,
            &[-1000, -530, -300, 300, 530, 1000][..],
        ),
        (CurvatureKind::Hessian, &[-1000, 1000][..]),
    ] {
        let expected = covariances_in_units(1.0, 0.5, kind);
        for &exponent in exponents {
            assert_eq!(
                covariances_in_units(2f64.powi(exponent), 0.5, kind),
                expected,
                "{kind:?} at 2^{exponent}"
            );
        }
    }
}

#[test]
fn a_unit_holds_a_derivative_whose_product_with_the_root_of_its_information_overflows() {
    // With an sd of 2⁻³⁰ the root of the information is 2³⁰, and at 2¹⁰⁰⁰
    // its product with the derivative 7·2¹⁰⁰⁰ of the last mean by b is above
    // the largest number, while both are finite.
    let sd = 2f64.powi(-30);
    for kind in [CurvatureKind::GaussNewton, CurvatureKind::Hessian] {
        assert_eq!(
            covariances_in_units(2f64.powi(1000), sd, kind),
            covariances_in_units(1.0, sd, kind),
            "{kind:?}"
        );
    }
}

#[test]
fn a_sandwich_reads_scores_whose_squares_underflow() {
    // x = a has noise of sd 1/2, and y = unit·(b (1 + t) + a) has noise of sd 1
    // and data in the same unit: the scores and the curvature that y gives are
    // in unit², so their quotient does not depend on it, and the square of a
    // score is below any number from 2⁻³⁰⁰ on and above the largest from 2²⁵⁶.
    // From 2⁵¹² the square of a residual is above it too, and the likelihood
    // has no value.
    // With a small unit only x reads a, and b follows the change of a by its
    // coupling in y; with a large unit only y reads both. The fit is the
    // optimum at every unit: the residuals of x sum to zero, and those of y
    // too, also times 1 + t.
    let x = [0.9, 1.3, 0.7, 1.2, 1.1, 0.8];
    let y = [1.75, 1.95, 2.3, 2.8, 3.45, 4.25];
    let fit = [1.0, 0.5];
    let d = [
        logit_derivative(fit[0], -10.0, 10.0),
        logit_derivative(fit[1], -5.0, 5.0),
    ];
    let reads = |t: usize| 1.0 + t as f64;
    let coupling: f64 = (0..6).map(reads).sum();
    let curvature: f64 = (0..6).map(|t| reads(t) * reads(t)).sum();
    let determinant = 6.0 * curvature - coupling * coupling;
    // The curvature of each block, in a unit of b times the unit: with a
    // small unit x alone reads a, with a large one y alone reads both.
    let block = |small: bool, t: usize| {
        if small {
            Matrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, reads(t) * reads(t)])
        } else {
            Matrix::from_row_slice(2, 2, &[1.0, reads(t), reads(t), reads(t) * reads(t)])
        }
    };
    let expected = |small: bool| {
        let blocks: Vec<Matrix> = (0..6).map(|t| block(small, t)).collect();
        let (factors, _) = bell_mccaffrey(&blocks);
        let mut expected = [[0.0; 2]; 2];
        for t in 0..6 {
            let residual = y[t] - fit[1] * reads(t) - fit[0];
            let influence = if small {
                let first = -(x[t] - fit[0]) / 6.0;
                [first, (-residual * reads(t) - coupling * first) / curvature]
            } else {
                [
                    residual * (coupling * reads(t) - curvature) / determinant,
                    residual * (coupling - 6.0 * reads(t)) / determinant,
                ]
            };
            for j in 0..2 {
                for k in 0..2 {
                    expected[j][k] += (factors[j] * factors[k]).sqrt() * influence[j] / d[j]
                        * influence[k]
                        / d[k];
                }
            }
        }
        expected
    };

    for exponent in [-100, -300, -900, 300, 500] {
        let unit = 2f64.powi(exponent);
        let expected = expected(exponent < 0);
        let mut data = observed("x", &x);
        data.extend(observed("y", &y.map(|value| value * unit)));
        let problem = CalibrationProblem::new(
            Curve::new(Shape::Coupled(unit)),
            data,
            vec![
                CalibrationParameter::new("a".to_string(), -10.0, 10.0),
                CalibrationParameter::new("b".to_string(), -5.0, 5.0),
            ],
            None,
            vec![],
            DiscrepancyConfig::Likelihood(vec![
                normal("x", Argument::Value(0.5)),
                normal("y", Argument::Value(1.0)),
            ]),
        )
        .unwrap();

        let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 1).unwrap();

        let covariance = run.gaussian.covariance();
        for j in 0..2 {
            for k in 0..2 {
                assert!(
                    (covariance[(j, k)] / expected[j][k] - 1.0).abs() < 1e-8,
                    "2^{exponent}, ({j}, {k}): {covariance} against {expected:?}"
                );
            }
        }
        assert_eq!(covariance[(0, 1)], covariance[(1, 0)]);
    }
}

#[test]
fn parameters_read_through_their_product_are_refused_by_every_method() {
    // At these points and numbers of observations, the derivatives of the
    // closed form are exact enough that the curvature of a and b has an
    // eigenvalue of 0 but for rounding.
    let mut generator = commol_random::Generator::from_seed(5);
    let mut uniform = |low: f64, high: f64| low + (high - low) * generator.open_uniform();
    for count in [8, 60, 400] {
        for _ in 0..20 {
            let fit = [uniform(0.1, 2.0), uniform(0.1, 2.0)];
            let values: Vec<f64> = (0..count)
                .map(|t| fit[0] * fit[1] * (1.0 + t as f64) + uniform(-1.0, 1.0))
                .collect();
            let names = ["a".to_string(), "b".to_string()];
            let bounds = [(0.01, 5.0), (0.01, 5.0)];
            let problem = CalibrationProblem::new(
                Curve::new(Shape::Product),
                observed("x", &values),
                vec![
                    CalibrationParameter::new("a".to_string(), 0.01, 5.0),
                    CalibrationParameter::new("b".to_string(), 0.01, 5.0),
                ],
                None,
                vec![],
                DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.3))]),
            )
            .unwrap();
            let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
            let curvature = Curvature::gauss_newton(&problem, &source, &fit).unwrap();
            let priors = Priors::uniform_on_bounds(&names, &bounds).unwrap();
            let target = LikelihoodTarget::new(&problem, &priors).unwrap();
            let mixture = laplace_mixture(
                &problem,
                &TemperedPosterior::new(&target, 1),
                1.0,
                &[fit.to_vec()],
                100,
                CurvatureKind::GaussNewton,
                1,
                &NoSteps,
            );
            let sandwich = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 2);

            for (method, outcome) in [
                ("laplace", laplace(&curvature).err()),
                ("sandwich", sandwich.err()),
                ("mixture", mixture.err()),
                (
                    "information matching",
                    information_matching_eta(&problem, &fit, 2).err(),
                ),
            ] {
                let error = outcome.unwrap_or_else(|| panic!("{method}, {count} at {fit:?}"));
                assert!(
                    matches!(error, MethodError::Invalid(_)),
                    "{method}: {error:?}"
                );
                assert!(
                    error
                        .to_string()
                        .contains("The curvature at the fit is not positive definite: the data"),
                    "{method}, {count} at {fit:?}: {error}"
                );
            }
        }
    }
}

#[test]
fn a_band_reads_the_degrees_of_freedom_of_a_direction_beyond_the_range_of_a_quotient() {
    // With noise of sd 10¹⁰ the scale of b is about 10⁻⁹ of its derivative,
    // and at the position 10³⁰⁰ the gradient of x in b is 10³⁰⁰ times it: the
    // direction g/s of the band, about 10³⁰⁹ in b, is above the largest
    // number, and its degrees of freedom are those of b.
    let values: Vec<f64> = line_values(8).iter().map(|value| 1e10 * value).collect();
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &values),
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(1e10))]),
    )
    .unwrap();
    let fit = [1.0, 0.0];
    let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, 3).unwrap();
    let degrees = run.gaussian.degrees_of_freedom().unwrap()[1];
    assert!((degrees - 1.0).abs() > 0.1, "{degrees}");
    let names = ["a".to_string(), "b".to_string()];
    let fit = [("a".to_string(), 1.0), ("b".to_string(), 0.0)];
    let covariance = [vec![0.0, 0.0], vec![0.0, 1e-300]];

    let band = delta_band(&problem, &names, &fit, &covariance, Some(3), &[1e300], 0.9).unwrap();

    let derivative = logit_derivative(0.0, -5.0, 5.0);
    let deviation = 1e300 * derivative * 1e-150;
    let quantile = (band.band.upper[0][0] - band.band.mean[0][0]) / deviation;
    let expected = commol_statistics::special::student_t_quantile(0.95, degrees);
    assert!(
        (quantile - expected).abs() < 1e-9 * expected,
        "{quantile} {expected}"
    );
}

#[test]
fn a_band_of_replicates_is_refused() {
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &[1.0, 1.5, 2.0]),
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
    .with_replicates(Some(Replicates::new(2, 1, ReplicateSummary::Mean).unwrap()))
    .unwrap();
    let names = ["a".to_string(), "b".to_string()];
    let fit = [("a".to_string(), 1.0), ("b".to_string(), 0.5)];
    let covariance = [vec![1.0, 0.0], vec![0.0, 1.0]];

    let error = delta_band(&problem, &names, &fit, &covariance, None, &[1.0], 0.9).unwrap_err();

    assert_eq!(
        error,
        MethodError::Sensitivity(SensitivityError::Stochastic)
    );
}

#[test]
fn a_hessian_that_steps_outside_the_model_fails() {
    // A Poisson mean of 5·10⁻⁵: the Gauss–Newton curvature runs only the
    // engine, but the Hessian evaluates the likelihood a step of about 10⁻⁴
    // below, at a negative mean.
    let problem = poisson_line(&[1.0, 2.0]);
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    assert!(Curvature::gauss_newton(&problem, &source, &[5e-5]).is_ok());

    let error = Curvature::hessian(&problem, &source, &[5e-5]).unwrap_err();

    assert!(matches!(error, MethodError::Failed(_)));
    assert!(
        error
            .to_string()
            .starts_with("The Hessian of the objective at the fit failed"),
        "{error}"
    );
}

#[test]
fn a_hessian_of_an_objective_near_the_largest_number_is_not_finite() {
    // The square of a residual of 1.2·10¹⁵⁴ is 1.44·10³⁰⁸: twice it, in the
    // second difference, is +∞.
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &[1.2e154]),
        vec![CalibrationParameter::new("a".to_string(), -10.0, 10.0)],
        None,
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    assert!(Curvature::gauss_newton(&problem, &source, &[1.0]).is_ok());

    let error = Curvature::hessian(&problem, &source, &[1.0]).unwrap_err();

    assert_eq!(
        error,
        MethodError::Failed("The Hessian of the objective at the fit is not finite".to_string())
    );
}

#[test]
fn a_fit_on_a_bound_above_alone_is_refused() {
    let values: Vec<f64> = (0..10).map(|t| 1.0 + 0.5 * t as f64).collect();
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &values),
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), f64::NEG_INFINITY, 0.4),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

    let error = Curvature::gauss_newton(&problem, &source, &[1.0, 0.4]).unwrap_err();

    assert!(matches!(error, MethodError::Invalid(_)));
    assert!(
        error.to_string().contains("'b' is on its bound 0.4"),
        "{error}"
    );
    assert!(Curvature::gauss_newton(&problem, &source, &[1.0, 0.3]).is_ok());
}

#[test]
fn a_mixture_whose_optima_the_prior_rules_out_is_refused() {
    // a fits −0.5, inside its bounds [−1, 5], where its gamma prior, whose
    // support starts at 0, has density 0.
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &[-0.4, -0.6, -0.5]),
        vec![CalibrationParameter::new("a".to_string(), -1.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let priors = Priors::new(
        &["a".to_string()],
        &[(-1.0, 5.0)],
        &[(
            "a".to_string(),
            Family::Gamma {
                shape: 2.0,
                rate: 1.0,
            },
        )],
    )
    .unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();

    let error = laplace_mixture(
        &problem,
        &TemperedPosterior::new(&target, 1),
        1.0,
        &[vec![-0.5]],
        100,
        CurvatureKind::GaussNewton,
        1,
        &NoSteps,
    )
    .unwrap_err();

    assert_eq!(
        error,
        MethodError::Invalid("Every optimum of the starts has a target density of 0".to_string())
    );
}

#[test]
fn a_one_dimensional_student_t_has_the_density_of_an_observation_model() {
    let (df, scale, center) = (3.0, 2.0, 1.0);
    let unbounded = Transform::new(&[(f64::NEG_INFINITY, f64::INFINITY)]);
    let gaussian = Gaussian::new(
        unbounded,
        vec![center],
        nalgebra::DMatrix::from_element(1, 1, scale * scale),
        Some(vec![df]),
    )
    .unwrap();
    let noise = Noise::StudentT { df, scale };
    for point in [-4.0, 1.0, 3.0, 25.0] {
        let observation = noise.log_density(point, center);
        let local = gaussian.log_density(&[point]).unwrap();
        assert!(
            (observation - local).abs() <= 1e-15 * observation.abs(),
            "{point}: {observation} {local}"
        );
    }
}

/// A receiver whose loops have stopped before their first step.
struct Stopped;

impl<E> Steps<E> for Stopped {
    fn stopped(&self) -> bool {
        true
    }
}

/// The events of a loop of model runs, as text: the beginning first, then
/// the points that ran, sorted.
#[derive(Default)]
struct KeptRuns(Mutex<Vec<String>>);

impl KeptRuns {
    fn events(self) -> Vec<String> {
        let mut events = self.0.into_inner().unwrap();
        events[1..].sort();
        events
    }
}

impl Steps<RunEvent> for KeptRuns {
    fn send(&self, event: RunEvent) {
        self.0.lock().unwrap().push(match event {
            RunEvent::Began { total } => format!("began {total}"),
            RunEvent::Ran { index } => format!("ran {index}"),
        });
    }
}

#[test]
fn a_mixture_sends_each_draw_and_runs_none_once_the_steps_stop() {
    let values: Vec<f64> = (0..10).map(|i| 4.0 + [0.2, -0.3, 0.1][i % 3]).collect();
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Square),
        observed("x", &values),
        vec![CalibrationParameter::new("a".to_string(), -5.0, 5.0)],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let priors = Priors::new(
        &["a".to_string()],
        &[(-5.0, 5.0)],
        &[("a".to_string(), Family::Normal { mean: 0.5, sd: 1.0 })],
    )
    .unwrap();
    let target = LikelihoodTarget::new(&problem, &priors).unwrap();
    let mixture = |steps: &RunSteps<'_>| {
        laplace_mixture(
            &problem,
            &TemperedPosterior::new(&target, 3),
            1.0,
            &[vec![2.0]],
            30,
            CurvatureKind::GaussNewton,
            3,
            steps,
        )
    };

    let kept = KeptRuns::default();
    mixture(&kept).unwrap();
    let events = kept.events();
    assert_eq!(events[0], "began 30");
    assert_eq!(events.len(), 31);

    let error = mixture(&Stopped).unwrap_err();
    assert!(error.to_string().contains(STOPPED), "{error}");
}

/// A target whose density fails at every point, and a proposal that draws
/// 0.
struct Failing;

impl Target for Failing {
    fn dimension(&self) -> usize {
        1
    }

    fn log_density(&self, _values: &[f64], _workspace: &mut Workspace) -> Result<f64, String> {
        Err("the run failed".to_string())
    }
}

impl commol_inference::importance::Proposal for Failing {
    fn draw(&self, _generator: &mut commol_random::Generator) -> Vec<f64> {
        vec![0.0]
    }

    fn log_density(&self, _unbounded: &[f64]) -> f64 {
        0.0
    }
}

#[test]
fn a_draw_of_an_importance_sample_that_fails_is_sent_as_run() {
    let kept = KeptRuns::default();
    let error = commol_inference::importance::importance_sample(
        &Failing,
        &Failing,
        &Transform::new(&[(f64::NEG_INFINITY, f64::INFINITY)]),
        10,
        3,
        &kept,
    )
    .unwrap_err();
    assert_eq!(error, "the run failed");
    let events = kept.events();
    assert_eq!(events[0], "began 10");
    assert!(events.len() > 1, "{events:?}");
}

/// The central difference of `f` in argument `index` of `point`.
fn central(f: &dyn Fn(&[f64]) -> f64, point: &[f64], index: usize) -> f64 {
    let h = 1e-5 * point[index].abs().max(1e-3);
    let mut up = point.to_vec();
    let mut down = point.to_vec();
    up[index] += h;
    down[index] -= h;
    (f(&up) - f(&down)) / (2.0 * h)
}

/// The noise of a model of the arguments `(μ₁, …, μ_K, noise arguments…)`.
type NoiseOf = Box<dyn Fn(&[f64]) -> Noise>;

/// A model of a sum of parts with each reading: its name, its noise and the
/// count of its noise arguments.
fn sum_readings() -> Vec<(&'static str, NoiseOf, usize)> {
    use commol_calibration::Sum;
    let negative_binomial = |sum: Sum| -> NoiseOf {
        Box::new(move |arguments: &[f64]| Noise::NegativeBinomial {
            overdispersion: arguments[arguments.len() - 1],
            sum: Some(sum),
        })
    };
    vec![
        ("exact", negative_binomial(Sum::Exact), 1),
        ("moments", negative_binomial(Sum::Moments), 1),
        ("poisson", Box::new(|_: &[f64]| Noise::Poisson), 0),
        (
            "normal",
            Box::new(|arguments: &[f64]| Noise::Normal {
                sd: arguments[arguments.len() - 1],
            }),
            1,
        ),
    ]
}

#[test]
fn the_gradient_of_a_sum_is_that_of_its_density() {
    let parts = [1.5, 4.0, 2.5, 0.2];
    for (name, noise_of, arguments) in sum_readings() {
        let mut point: Vec<f64> = parts.to_vec();
        if arguments == 1 {
            point.push(0.4);
        }
        for y in [0.0, 3.0, 8.0, 21.0] {
            let log_density =
                |point: &[f64]| noise_of(point).sum_log_density(y, &point[..4]).unwrap();
            let scores = noise_of(&point).sum_scores(y, &parts).unwrap();
            assert_eq!(scores.len(), point.len(), "{name}");
            for (index, score) in scores.iter().enumerate() {
                let expected = central(&log_density, &point, index);
                assert!(
                    (score - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                    "{name}, y = {y}, argument {index}: {score} against {expected}"
                );
            }
        }
    }

    // Through the block of the parts: the gradient of the term of the sum on
    // the unbounded scale is the difference of its log density in the
    // parameters, times the derivative of the scale.
    use commol_calibration::Sum;
    let window = |at: f64, value: f64| ObservedDataPoint {
        parts: 5,
        ..ObservedDataPoint::with_window(at, "x".to_string(), value, 10.0)
    };
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Quadratic),
        vec![window(10.0, 40.0), window(20.0, 160.0)],
        vec![
            CalibrationParameter::new("a".to_string(), 0.1, 5.0),
            CalibrationParameter::new("b".to_string(), 0.01, 2.0),
            noise("phi", 0.01, 3.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![(
            "x".to_string(),
            ObservationModel::NegativeBinomial {
                overdispersion: Argument::Parameter("phi".to_string()),
                sum: Some(Sum::Exact),
            },
        )]),
    )
    .unwrap();
    let fit = [1.2, 0.3, 0.5];
    let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();
    let curvature = Curvature::gauss_newton(&problem, &source, &fit).unwrap();
    let bounds = [(0.1, 5.0), (0.01, 2.0), (0.01, 3.0)];
    for i in 0..2 {
        let term = |values: &[f64]| {
            -problem
                .pointwise_log_likelihood(values, &mut Workspace::default())
                .unwrap()[i]
        };
        for (j, &(lower, upper)) in bounds.iter().enumerate() {
            let expected = central(&term, &fit, j) * logit_derivative(fit[j], lower, upper);
            let actual = curvature.scaled_gradients()[(i, j)] * curvature.scales()[j];
            assert!(
                (actual - expected).abs() <= 1e-5 * expected.abs().max(1e-3),
                "term {i}, parameter {j}: {actual} against {expected}"
            );
        }
    }
}

#[test]
fn the_information_of_a_sum_is_that_of_its_density() {
    let parts = [1.5, 4.0, 2.5];
    for (name, noise_of, arguments) in sum_readings() {
        if name == "normal" {
            continue;
        }
        let mut point: Vec<f64> = parts.to_vec();
        if arguments == 1 {
            point.push(0.4);
        }
        let size = point.len();
        // Σ_y p(y) s(y) s(y)ᵀ, with the scores from differences of the log
        // density, over every count that carries mass.
        let mut expected = vec![0.0; size * size];
        for y in 0..200 {
            let y = f64::from(y);
            let log_density =
                |point: &[f64]| noise_of(point).sum_log_density(y, &point[..3]).unwrap();
            let probability = log_density(&point).exp();
            let scores: Vec<f64> = (0..size)
                .map(|index| central(&log_density, &point, index))
                .collect();
            for a in 0..size {
                for b in 0..size {
                    expected[a * size + b] += probability * scores[a] * scores[b];
                }
            }
        }
        let information = noise_of(&point).sum_information(&parts).unwrap();
        for a in 0..size {
            for b in 0..size {
                let scale = (expected[a * size + a] * expected[b * size + b]).sqrt();
                let (actual, expected) = (information[a * size + b], expected[a * size + b]);
                assert!(
                    (actual - expected).abs() <= 1e-6 * scale,
                    "{name}, entry ({a}, {b}): {actual} against {expected}"
                );
            }
        }
    }
    // A normal sum: the information of N(M, K sd²) over (μ₁, μ₂, sd).
    let information = Noise::Normal { sd: 2.0 }
        .sum_information(&[1.0, 3.0])
        .unwrap();
    let mean = 1.0 / 8.0;
    let expected = [mean, mean, 0.0, mean, mean, 0.0, 0.0, 0.0, 0.5];
    for (actual, expected) in information.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-15, "{information:?}");
    }
}
