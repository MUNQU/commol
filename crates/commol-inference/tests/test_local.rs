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
    Curvature, CurvatureKind, Gaussian, delta_band, laplace, laplace_mixture,
    sandwich_approximation,
};
use commol_inference::psis::smooth;
use commol_inference::sensitivity::{
    Design, FiniteDifferences, Sensitivities, SensitivityError, SensitivitySource,
};
use commol_inference::smc::LikelihoodTarget;
use commol_inference::target::Target;
use commol_inference::transform::Transform;
use commol_inference::{MethodError, Priors};

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
            parameters: HashMap::from([("a".to_string(), 0.0), ("b".to_string(), 0.0)]),
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
        let (a, b) = (self.parameters["a"], self.parameters["b"]);
        buffer.clear();
        buffer.extend(positions.iter().map(|&t| match self.shape {
            Shape::Line => vec![a + b * t],
            Shape::Square => vec![a * a],
            Shape::Banana => vec![a, b + a * a],
            Shape::Quadratic => vec![a * t + b * t * t],
        }));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        match self.shape {
            Shape::Line | Shape::Square | Shape::Quadratic => vec!["x".to_string()],
            Shape::Banana => vec!["x".to_string(), "y".to_string()],
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

    assert_matrix(curvature.matrix(), &line_information(&fit), 1e-8);
    assert!(curvature.notes().is_empty(), "{:?}", curvature.notes());
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

    assert_matrix(curvature.matrix(), &line_information(&fit), 1e-5);
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
            .any(|note| note.contains("misses an optimum on a bound")),
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
        for j in 0..2 {
            for k in 0..2 {
                let scale = (expected[j][j] * expected[k][k]).sqrt();
                assert!(
                    (curvature.matrix()[(j, k)] - expected[j][k]).abs() < 1e-5 * scale,
                    "({j}, {k}): {}",
                    curvature.matrix()
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
    assert!(error.to_string().contains("not finite"), "{error}");
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

    let gradients = curvature.gradients();
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
            sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, Some(1))
                .unwrap_err(),
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

#[test]
fn a_derivative_too_large_for_its_square_has_no_finite_curvature() {
    // The derivative 3 scale of scale (a + b t) at t = 3 is finite and its
    // square is not.
    for scale in [1e154, 1e157] {
        let problem = line_and_far(scale, 1.0, 1.0);
        let source = FiniteDifferences::new(&problem, Design::Observations).unwrap();

        let error = Curvature::gauss_newton(&problem, &source, &[0.0, 0.0]).unwrap_err();

        assert!(matches!(error, MethodError::Failed(_)), "{error:?}");
        assert!(
            error
                .to_string()
                .ends_with("or a derivative is too large for its square"),
            "{error}"
        );
    }
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
fn a_sandwich_needs_more_blocks_than_parameters_and_reads_the_automatic_length() {
    let values: Vec<f64> = (0..8)
        .map(|t| 1.0 + 0.5 * t as f64 + [0.3, -0.2, 0.1, -0.4][t % 4])
        .collect();
    let problem = CalibrationProblem::new(
        Curve::new(Shape::Line),
        observed("x", &values),
        vec![
            CalibrationParameter::new("a".to_string(), -10.0, 10.0),
            CalibrationParameter::new("b".to_string(), -5.0, 5.0),
        ],
        None,
        vec![],
        DiscrepancyConfig::Likelihood(vec![normal("x", Argument::Value(0.5))]),
    )
    .unwrap();
    let fit = [1.0, 0.5];

    let error =
        sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, Some(4)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("makes 2 blocks for 2 parameters"),
        "{error}"
    );
    let run = sandwich_approximation(&problem, &fit, CurvatureKind::GaussNewton, None).unwrap();
    let estimates = run.estimates.unwrap();
    assert_eq!(estimates.len(), 1);
    assert_eq!(run.blocks, 8_usize.div_ceil(run.block_length));
    assert_eq!(
        run.gaussian.degrees_of_freedom(),
        Some((run.blocks - 1) as f64)
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
        Some(df),
    )
    .unwrap();
    let noise = Noise::StudentT { df, scale };
    for point in [-4.0, 1.0, 3.0, 25.0] {
        let observation = noise.log_density(point, center);
        let local = gaussian.log_density(&[point]);
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
            let actual = curvature.gradients()[(i, j)];
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
