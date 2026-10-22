//! The curvature of the objective of a problem at a fit, on the unbounded
//! scale.
//!
//! Gauss–Newton reads the expected curvature of each observation's term
//! ([`commol_calibration::CalibrationProblem::term_derivatives`]): over the
//! arguments of the term, its mean `μ` and the noise arguments of its
//! observation model, it is `wᵢ` times the Fisher information of the model,
//! or `2 w n²` for the sum of squares. The chain rule through the Jacobian
//! `J` of the means and the identity of the noise parameters gives
//! `H_θ = Σᵢ Aᵢᵀ Cᵢ Aᵢ`, which is `Jᵀ W J` on the model parameters. On the
//! unbounded scale, `H_u = D · H_θ · D`; the curvature of the log Jacobian
//! is left out. The Hessian takes central differences of the objective
//! itself, the observed curvature, for a fit with large residuals or a wrong
//! model, where the expected and the observed curvature differ.

use std::cell::RefCell;

use commol_calibration::{CalibrationProblem, TermDerivatives, Workspace};
use commol_core::SimulationEngine;
use finitediff::vec::hessian::central_hessian_nograd_error_vec;

use crate::error::MethodError;
use crate::sensitivity::{Matrix, SensitivitySource};
use crate::transform::{Scale, Transform};

/// A value this close to a bound, relative to the width of the bounds, is on
/// it: about the square root of the machine epsilon, where an optimizer that
/// the bound stops ends.
const ON_BOUND: f64 = 1e-8;

/// An entry of the curvature is reported when its estimated error is above
/// this share of `√(H_jj · H_kk)`, the scale of its row and column.
const UNCERTAIN_ENTRY: f64 = 0.01;

/// The curvature of the objective at a fit, on the unbounded scale.
#[derive(Debug, Clone)]
pub struct Curvature {
    names: Vec<String>,
    transform: Transform,
    fit: Vec<f64>,
    unbounded_fit: Vec<f64>,
    derivatives: Vec<f64>,
    matrix: Matrix,
    gradients: Matrix,
    predictions: Vec<f64>,
    notes: Vec<String>,
}

impl Curvature {
    /// The Gauss–Newton curvature of the objective of `problem` at `fit`,
    /// with the Jacobian of `source`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when a value of `fit` sits on a bound of its
    /// parameter, the engine is stochastic, or the discrepancy has no
    /// curvature (the mean absolute error); [`MethodError::Failed`] when the
    /// derivatives fail or the curvature is not finite.
    pub fn gauss_newton<E, S>(
        problem: &CalibrationProblem<E>,
        source: &S,
        fit: &[f64],
    ) -> Result<Self, MethodError>
    where
        E: SimulationEngine,
        S: SensitivitySource + ?Sized,
    {
        let transform = Transform::new(&problem.parameter_bounds());
        check_inside(problem, &transform, fit)?;
        let sensitivities = source.jacobian(fit).map_err(MethodError::Sensitivity)?;
        let terms = problem
            .term_derivatives(fit, &sensitivities.predictions)
            .map_err(MethodError::Invalid)?;
        let p = fit.len();
        let mut matrix = Matrix::zeros(p, p);
        let mut errors = Matrix::zeros(p, p);
        let mut gradients = Matrix::zeros(terms.len(), p);
        for (i, term) in terms.iter().enumerate() {
            let jacobian: Vec<f64> = sensitivities.jacobian.row(i).iter().copied().collect();
            let error: Vec<f64> = sensitivities.errors.row(i).iter().copied().collect();
            add_term(&mut matrix, &mut errors, term, &jacobian, &error);
            for (j, gradient) in term_gradient(term, &jacobian).into_iter().enumerate() {
                gradients[(i, j)] = gradient;
            }
        }
        if matrix.iter().any(|entry| !entry.is_finite()) {
            return Err(MethodError::Failed(
                "The curvature at the fit is not finite: a mean sits on the edge of the means \
                 its observation model allows"
                    .to_string(),
            ));
        }
        let names = problem.parameter_names();
        let mut curvature = Self::on_the_unbounded_scale(names, transform, fit, matrix, errors);
        for i in 0..gradients.nrows() {
            for j in 0..p {
                gradients[(i, j)] *= curvature.derivatives[j];
            }
        }
        curvature.gradients = gradients;
        curvature.predictions = sensitivities.predictions;
        Ok(curvature)
    }

    /// The curvature of `problem` at `fit` by central differences of its
    /// objective: the stepped central Hessian of the `finitediff` crate of
    /// the argmin fork, with its estimated error. The gradients of the terms
    /// and the predictions are those of [`Self::gauss_newton`] with
    /// `source`.
    ///
    /// The step of parameter `j` is `h_j = δ^{1/4} · s_j`, which balances the
    /// truncation error of the central second difference, of order `h²`,
    /// with the error `δ/h²` of an objective of relative accuracy `δ`
    /// ([`SimulationEngine::output_accuracy`]). The scale `s_j` is `dθ/du`,
    /// the length of a unit step on the unbounded scale, for a parameter
    /// with a bound, and `max(|θ_j|, 1)` for one without: `dθ/du` is below
    /// the distance to each bound, so `θ ± 2h` stays inside them.
    ///
    /// # Errors
    ///
    /// As [`Self::gauss_newton`], and [`MethodError::Failed`] when a point of
    /// the differences is infeasible or the engine fails.
    pub fn hessian<E, S>(
        problem: &CalibrationProblem<E>,
        source: &S,
        fit: &[f64],
    ) -> Result<Self, MethodError>
    where
        E: SimulationEngine,
        S: SensitivitySource + ?Sized,
    {
        let expected = Self::gauss_newton(problem, source, fit)?;
        let accuracy = problem.forward().base_engine().output_accuracy();
        let factor = accuracy.powf(0.25);
        let steps: Vec<f64> = expected
            .transform
            .scales()
            .iter()
            .zip(&expected.derivatives)
            .zip(fit)
            .map(|((scale, &derivative), &value)| match scale {
                Scale::Identity => factor * value.abs().max(1.0),
                _ => factor * derivative,
            })
            .collect();
        let workspace = RefCell::new(Workspace::default());
        let objective = |values: &Vec<f64>| -> Result<f64, anyhow::Error> {
            problem
                .evaluate(values, &mut workspace.borrow_mut())
                .map_err(|error| anyhow::Error::msg(error.to_string()))
        };
        let estimate = central_hessian_nograd_error_vec(fit, &objective, &steps, accuracy)
            .map_err(|error| {
                MethodError::Failed(format!(
                    "The Hessian of the objective at the fit failed: {error}"
                ))
            })?;
        let p = fit.len();
        let error = estimate.error();
        let matrix = Matrix::from_fn(p, p, |j, k| estimate.hessian[j][k]);
        let errors = Matrix::from_fn(p, p, |j, k| error[j][k]);
        if matrix.iter().any(|entry| !entry.is_finite()) {
            return Err(MethodError::Failed(
                "The Hessian of the objective at the fit is not finite".to_string(),
            ));
        }
        let mut curvature = Self::on_the_unbounded_scale(
            expected.names.clone(),
            expected.transform.clone(),
            fit,
            matrix,
            errors,
        );
        curvature.gradients = expected.gradients;
        curvature.predictions = expected.predictions;
        Ok(curvature)
    }

    /// `H_u = D · H_θ · D` and its errors, with the notes of entries whose
    /// error is not small.
    fn on_the_unbounded_scale(
        names: Vec<String>,
        transform: Transform,
        fit: &[f64],
        natural: Matrix,
        natural_errors: Matrix,
    ) -> Self {
        let derivatives = transform.derivatives(fit);
        let p = fit.len();
        let scale =
            |m: &Matrix| Matrix::from_fn(p, p, |j, k| derivatives[j] * m[(j, k)] * derivatives[k]);
        let matrix = scale(&natural);
        let errors = scale(&natural_errors);
        let mut notes = Vec::new();
        for j in 0..p {
            for k in j..p {
                let size = (matrix[(j, j)] * matrix[(k, k)]).abs().sqrt();
                if errors[(j, k)] > UNCERTAIN_ENTRY * size {
                    notes.push(format!(
                        "The curvature of ('{}', '{}') is {:.6e} with an estimated error of \
                         {:.3e}, {:.1}% of the scale of its row and column",
                        names[j],
                        names[k],
                        matrix[(j, k)],
                        errors[(j, k)],
                        100.0 * errors[(j, k)] / size
                    ));
                }
            }
        }
        Self {
            names,
            unbounded_fit: transform.to_unbounded(fit),
            transform,
            fit: fit.to_vec(),
            derivatives,
            matrix,
            gradients: Matrix::zeros(0, p),
            predictions: Vec::new(),
            notes,
        }
    }

    /// The map between the scale of the parameters and the unbounded one.
    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    /// The fit, on the scale of the parameters.
    pub fn fit(&self) -> &[f64] {
        &self.fit
    }

    /// The fit on the unbounded scale: the centre of every approximation.
    pub fn unbounded_fit(&self) -> &[f64] {
        &self.unbounded_fit
    }

    /// `dθ/du` of each parameter at the fit: the diagonal of `D`.
    pub fn derivatives(&self) -> &[f64] {
        &self.derivatives
    }

    /// The curvature `H_u` on the unbounded scale.
    pub fn matrix(&self) -> &Matrix {
        &self.matrix
    }

    /// The gradient of each observation's term of the objective on the
    /// unbounded scale, one row per observation: `D · Aᵢᵀ ∇ᵢ`, with `∇ᵢ` the
    /// gradient of the term over its arguments. Their block sums make the
    /// meat of the sandwich.
    pub fn gradients(&self) -> &Matrix {
        &self.gradients
    }

    /// The predicted observations at the fit.
    pub fn predictions(&self) -> &[f64] {
        &self.predictions
    }

    /// The entries whose estimated error is above 1 % of the scale of their
    /// row and column, in words.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }
}

/// Refuse a fit on a bound of its parameter ([`on_bound`]): the unbounded
/// scale puts it at infinity, where the variance is infinite.
fn check_inside<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    transform: &Transform,
    fit: &[f64],
) -> Result<(), MethodError> {
    match on_bound(transform, fit) {
        Some((index, bound)) => Err(MethodError::Invalid(format!(
            "The fit of '{}' is on its bound {bound}: on the unbounded scale its variance is \
             infinite there. A profile likelihood describes the uncertainty of a parameter on \
             its bound (MUNQU/commol#132)",
            problem.parameters()[index].id
        ))),
        None => Ok(()),
    }
}

/// The first parameter of `values` on one of its bounds, and that bound: at
/// most [`ON_BOUND`] times the width of its bounds from it, or times the
/// larger of 1 and the bound's magnitude for a parameter with one bound. An
/// optimizer that a bound stops ends there up to its rounding.
pub(crate) fn on_bound(transform: &Transform, values: &[f64]) -> Option<(usize, f64)> {
    transform
        .scales()
        .iter()
        .zip(values)
        .enumerate()
        .find_map(|(index, (scale, &value))| {
            let (bounds, scale_of) = match *scale {
                Scale::Logit { lower, upper } => ([Some(lower), Some(upper)], upper - lower),
                Scale::Log { lower } => ([Some(lower), None], lower.abs().max(1.0)),
                Scale::LogFromAbove { upper } => ([None, Some(upper)], upper.abs().max(1.0)),
                Scale::Identity => ([None, None], 1.0),
            };
            bounds
                .into_iter()
                .flatten()
                .find(|&bound| (value - bound).abs() <= ON_BOUND * scale_of)
                .map(|bound| (index, bound))
        })
}

/// Add `Aᵀ C A` of one term to `matrix`, and the first-order error of the
/// Jacobian row to `errors`. Row 0 of `A` is the Jacobian of the mean; row
/// `1 + k` is the unit vector of the parameter of noise argument `k`, or 0
/// for a number.
fn add_term(
    matrix: &mut Matrix,
    errors: &mut Matrix,
    term: &TermDerivatives,
    jacobian: &[f64],
    error: &[f64],
) {
    let size = 1 + term.parameters.len();
    let p = jacobian.len();
    let row = |a: usize, j: usize| -> f64 {
        if a == 0 {
            jacobian[j]
        } else {
            f64::from(u8::from(term.parameters[a - 1] == Some(j)))
        }
    };
    let row_error = |a: usize, j: usize| if a == 0 { error[j] } else { 0.0 };
    for a in 0..size {
        for b in 0..size {
            let c = term.curvature[a * size + b];
            if c == 0.0 {
                continue;
            }
            for j in 0..p {
                let (u, du) = (row(a, j), row_error(a, j));
                if u == 0.0 && du == 0.0 {
                    continue;
                }
                for k in 0..p {
                    let (v, dv) = (row(b, k), row_error(b, k));
                    matrix[(j, k)] += c * u * v;
                    errors[(j, k)] += c.abs() * (du * v.abs() + u.abs() * dv);
                }
            }
        }
    }
}

/// `Aᵀ ∇` of one term: the gradient of its arguments through the Jacobian of
/// the mean and the parameters of its noise arguments.
fn term_gradient(term: &TermDerivatives, jacobian: &[f64]) -> Vec<f64> {
    let mut gradient: Vec<f64> = jacobian.iter().map(|j| term.gradient[0] * j).collect();
    for (parameter, slope) in term.parameters.iter().zip(&term.gradient[1..]) {
        if let Some(index) = parameter {
            gradient[*index] += slope;
        }
    }
    gradient
}
