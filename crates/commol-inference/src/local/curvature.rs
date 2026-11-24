//! The curvature of the objective of a problem at a fit, on the unbounded
//! scale.
//!
//! Gauss–Newton reads the expected curvature of each observation's term
//! ([`commol_calibration::CalibrationProblem::term_derivatives`]): over the
//! arguments of the term, the means `μₜ` of its parts and the noise
//! arguments of its observation model, it is `wᵢ` times the Fisher
//! information of the model, or `2 w n²` for the sum of squares. The chain
//! rule through the Jacobian `J` of the means of the parts
//! ([`crate::sensitivity::PartSensitivities`]) and the identity of the noise
//! parameters gives `H_θ = Σᵢ Aᵢᵀ Cᵢ Aᵢ`, which is `Jᵀ W J` on the model
//! parameters. On the
//! unbounded scale, `H_u = D · H_θ · D`; the curvature of the log Jacobian
//! is left out. The Hessian takes central differences of the objective
//! itself, the observed curvature, for a fit with large residuals or a wrong
//! model, where the expected and the observed curvature differ.

use std::cell::RefCell;

use commol_calibration::{CalibrationProblem, DerivativesError, TermDerivatives, Workspace};
use commol_core::SimulationEngine;
use finitediff::vec::hessian::central_hessian_nograd_error_vec;

use crate::error::MethodError;
use crate::sensitivity::{Matrix, Sensitivities, SensitivitySource};
use crate::transform::{Scale, Transform};

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
    terms: Vec<TermDerivatives>,
    part_jacobian: Matrix,
    first_parts: Vec<usize>,
    notes: Vec<String>,
}

impl Curvature {
    /// The Gauss–Newton curvature of the objective of `problem` at `fit`,
    /// with the Jacobian of `source`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when a value of `fit` sits on a bound of its
    /// parameter, the engine is stochastic, the discrepancy has no
    /// curvature (the mean absolute error), or an observation has no finite
    /// derivatives of its term ([`CalibrationProblem::term_derivatives`]);
    /// [`MethodError::Failed`] when the derivatives fail, a prediction or a
    /// derivative of `source` is not finite, also one that the curvature does
    /// not read, or the curvature is not finite.
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
        problem
            .discrepancy()
            .check_curvature()
            .map_err(MethodError::Invalid)?;
        let sensitivities = source.jacobian(fit).map_err(MethodError::Sensitivity)?;
        check_finite(problem, &sensitivities)?;
        let parts = &sensitivities.parts;
        let terms = problem
            .term_derivatives(fit, &sensitivities.predictions, &parts.means)
            .map_err(|error| match error {
                DerivativesError::Invalid(message) => MethodError::Invalid(message),
                DerivativesError::Failed(message) => MethodError::Failed(message),
            })?;
        let p = fit.len();
        let mut matrix = Matrix::zeros(p, p);
        let mut errors = Matrix::zeros(p, p);
        let mut gradients = Matrix::zeros(terms.len(), p);
        let mut first_parts = Vec::with_capacity(terms.len());
        let mut first = 0;
        for (i, term) in terms.iter().enumerate() {
            let jacobian = rows(&parts.jacobian, first, term.parts);
            let error = rows(&parts.errors, first, term.parts);
            first_parts.push(first);
            first += term.parts;
            add_term(&mut matrix, Some((&mut errors, &error)), term, &jacobian);
            for (j, gradient) in term_gradient(term, &jacobian).into_iter().enumerate() {
                gradients[(i, j)] = gradient;
            }
        }
        if matrix.iter().any(|entry| !entry.is_finite()) {
            return Err(MethodError::Failed(
                "The curvature at the fit is not finite: a mean sits on the edge of the means \
                 its observation model allows, or a derivative is too large for its square"
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
        curvature.terms = terms;
        curvature.part_jacobian = sensitivities.parts.jacobian;
        curvature.first_parts = first_parts;
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
    /// (the relative [`SimulationEngine::accuracy`]). The scale `s_j` is
    /// `dθ/du`, the length of a unit step on the unbounded scale, for a parameter
    /// with a bound, and `max(|θ_j|, 1)` for one without: `dθ/du` is below
    /// the distance to each bound, so `θ ± 2h` stays inside them.
    ///
    /// # Errors
    ///
    /// As [`Self::gauss_newton`], also for an entry of `source` that the
    /// differences of the objective do not read, and [`MethodError::Failed`]
    /// when a point of the differences is infeasible or the engine fails.
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
        let accuracy = problem.forward().base_engine().accuracy().relative;
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
        curvature.terms = expected.terms;
        curvature.part_jacobian = expected.part_jacobian;
        curvature.first_parts = expected.first_parts;
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
            terms: Vec::new(),
            part_jacobian: Matrix::zeros(0, p),
            first_parts: Vec::new(),
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

    /// The Gauss–Newton curvature of the terms of `observations` alone, on
    /// the unbounded scale: `J_b = D · Σ_{i∈b} Aᵢᵀ Cᵢ Aᵢ · D`. For a
    /// Gauss–Newton curvature it is the share of a block `b` of
    /// [`Self::matrix`], the sum of the shares of all observations; a
    /// Hessian keeps the Gauss–Newton terms, whose shares do not sum to it.
    pub fn block_matrix(&self, observations: &[usize]) -> Matrix {
        let p = self.fit.len();
        let mut matrix = Matrix::zeros(p, p);
        for &i in observations {
            let term = &self.terms[i];
            let jacobian = rows(&self.part_jacobian, self.first_parts[i], term.parts);
            add_term(&mut matrix, None, term, &jacobian);
        }
        Matrix::from_fn(p, p, |j, k| {
            self.derivatives[j] * matrix[(j, k)] * self.derivatives[k]
        })
    }

    /// The entries whose estimated error is above 1 % of the scale of their
    /// row and column, in words.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }
}

/// Refuse a fit on a bound of its parameter ([`Transform::on_bound`]): the
/// unbounded scale puts it at infinity, where the variance is infinite.
fn check_inside<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    transform: &Transform,
    fit: &[f64],
) -> Result<(), MethodError> {
    match transform.on_bound(fit) {
        Some((index, bound)) => Err(MethodError::Invalid(format!(
            "The fit of '{}' is on its bound {bound}: on the unbounded scale its variance is \
             infinite there. A profile likelihood describes the uncertainty of a parameter on \
             its bound: Calibrator.uncertainty(ProfileLikelihood(...))",
            problem.parameters()[index].id
        ))),
        None => Ok(()),
    }
}

/// Refuse a prediction or a derivative of `sensitivities` that is not
/// finite, whatever the weight of its observation. The check of the curvature
/// does not see one that the curvature does not read (in a term of no
/// curvature, or the prediction of a loss), and the gradient of its term can
/// be not finite.
fn check_finite<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    sensitivities: &Sensitivities,
) -> Result<(), MethodError> {
    let Some(entry) = sensitivities.first_not_finite() else {
        return Ok(());
    };
    let obs = &problem.observed_data()[entry.observation];
    let value = entry.value;
    let (what, named) = match entry.column {
        None => ("predictions", format!("prediction {value}")),
        Some(j) => (
            "derivatives",
            format!(
                "derivative {value} with respect to '{}'",
                problem.parameters()[j].id
            ),
        ),
    };
    let place = entry
        .part
        .map_or_else(String::new, |part| format!(" in its part {}", part + 1));
    Err(MethodError::Failed(format!(
        "The {what} of the runs at the fit are not finite: observation '{}' at {} has the \
         {named}{place}",
        obs.label, obs.at
    )))
}

/// Rows `first` to `first + count` of `matrix`.
fn rows(matrix: &Matrix, first: usize, count: usize) -> Vec<Vec<f64>> {
    (first..first + count)
        .map(|k| matrix.row(k).iter().copied().collect())
        .collect()
}

/// Add `Aᵀ C A` of one term to `matrix`, and, with `errors`, the
/// first-order error of the Jacobian rows, of the errors `error` of each
/// entry, to it. Row `t` of `A`, below the number of parts, is the Jacobian
/// of the mean of part `t`; row `K + k` is the unit vector of the parameter
/// of noise argument `k`, or 0 for a number.
fn add_term(
    matrix: &mut Matrix,
    errors: Option<(&mut Matrix, &[Vec<f64>])>,
    term: &TermDerivatives,
    jacobian: &[Vec<f64>],
) {
    let parts = term.parts;
    let size = parts + term.parameters.len();
    let p = matrix.ncols();
    let row = |a: usize, j: usize| -> f64 {
        if a < parts {
            jacobian[a][j]
        } else {
            f64::from(u8::from(term.parameters[a - parts] == Some(j)))
        }
    };
    let (mut errors, error) =
        errors.map_or((None, None), |(errors, error)| (Some(errors), Some(error)));
    let row_error = |a: usize, j: usize| match error {
        Some(error) if a < parts => error[a][j],
        _ => 0.0,
    };
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
                    if let Some(errors) = errors.as_mut() {
                        errors[(j, k)] += c.abs() * (du * v.abs() + u.abs() * dv);
                    }
                }
            }
        }
    }
}

/// `Aᵀ ∇` of one term: the gradient of its arguments through the Jacobian of
/// the means of its parts and the parameters of its noise arguments.
fn term_gradient(term: &TermDerivatives, jacobian: &[Vec<f64>]) -> Vec<f64> {
    let p = jacobian.first().map_or(0, Vec::len);
    let mut gradient = vec![0.0; p];
    for (row, slope) in jacobian.iter().zip(&term.gradient) {
        for (entry, derivative) in gradient.iter_mut().zip(row) {
            *entry += slope * derivative;
        }
    }
    for (parameter, slope) in term.parameters.iter().zip(&term.gradient[term.parts..]) {
        if let Some(index) = parameter {
            gradient[*index] += slope;
        }
    }
    gradient
}
