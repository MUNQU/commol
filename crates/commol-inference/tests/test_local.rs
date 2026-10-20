//! Local approximations around a fit: the curvature on the unbounded scale,
//! the Laplace approximation and the delta method, the sandwich, the Laplace
//! mixture, importance sampling and PSIS, on an engine whose outputs are
//! closed forms of its parameters.

use std::collections::HashMap;

use commol_calibration::{
    Argument, CalibrationParameter, CalibrationParameterType, CalibrationProblem,
    DiscrepancyConfig, ObservationModel, ObservedDataPoint,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::local::{Curvature, laplace};
use commol_inference::sensitivity::{Design, FiniteDifferences};

/// The closed form of the outputs of a [`Curve`].
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// One output `x = a + b·t`.
    Line,
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
        }));
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        match self.shape {
            Shape::Line => vec!["x".to_string()],
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
#[ignore = "waits for the stepped Hessian of MUNQU/argmin#6; run with cargo test -p commol-inference --test test_local -- --ignored"]
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

    let curvature = Curvature::hessian(&problem, &fit).unwrap();

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
