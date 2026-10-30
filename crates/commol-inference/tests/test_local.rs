//! Local approximations around a fit: the curvature on the unbounded scale,
//! the Laplace approximation and the delta method, the sandwich, the Laplace
//! mixture, importance sampling and PSIS, on an engine whose outputs are
//! closed forms of its parameters.

use std::collections::HashMap;

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
use commol_inference::sensitivity::{Design, FiniteDifferences, SensitivityError};
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
        }));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        match self.shape {
            Shape::Line | Shape::Square => vec!["x".to_string()],
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
