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
//! parameters. On the unbounded scale, `H_u = D · H_θ · D`; the curvature of
//! the log Jacobian is left out. The Hessian takes central differences of the
//! objective itself, the observed curvature, for a fit with large residuals or
//! a wrong model, where the expected and the observed curvature differ.
//!
//! A curvature is kept as `H_u = S Ĥ S`, with the scale of each parameter in
//! the diagonal `S` ([`Curvature::scales`]) and `Ĥ` in its units: the squares
//! and the products of derivatives stay inside the range of floating point
//! whatever the scale of a parameter. A scale is a power of two times `dθ/du`,
//! so the change of units rounds nothing. No method forms `H_u`: they read its
//! inverse, `S⁻¹ Ĥ⁻¹ S⁻¹` ([`InverseCurvature`]).
//!
//! A diagonal entry of a sum that is at most `ε` times the sum of the absolute
//! values of its terms is only the rounding of terms that cancel, and is 0.

use std::cell::RefCell;

use commol_calibration::{CalibrationProblem, DerivativesError, TermDerivatives, Workspace};
use commol_core::SimulationEngine;
use commol_statistics::numeric::CompensatedSum;
use finitediff::vec::hessian::central_hessian_nograd_error_vec;
use nalgebra::{DMatrix, DMatrixView};

use super::inverse::InverseCurvature;
use crate::error::MethodError;
use crate::sensitivity::{
    Matrix, Sensitivities, SensitivitySource, part_backgrounds, run_rounding,
};
use crate::transform::{Scale, Transform};

/// An entry of the curvature is reported when its estimated error is above
/// this share of `√(H_jj · H_kk)`, the scale of its row and column.
const UNCERTAIN_ENTRY: f64 = 0.01;

/// The largest `H_jj h_j²` of a step `h_j` of the Hessian on the unbounded
/// scale, with `H_jj` the Gauss–Newton curvature. By it the points of the
/// differences with the steps `2h` move the objective by at most 8 times this,
/// a quarter of the largest number, so a stencil of four of them sums to a
/// number, when the observed curvature is at most the Gauss–Newton one and the
/// objective grows like its quadratic in the residuals. A Student t of a small
/// `df`, which forms `z²/ν`, can still overflow.
const OBJECTIVE_ROOM: f64 = f64::MAX / 32.0;

/// The curvature of the objective at a fit. Its matrices, named `scaled`, are
/// in the units of [`Self::scales`].
#[derive(Debug, Clone)]
pub struct Curvature {
    names: Vec<String>,
    transform: Transform,
    fit: Vec<f64>,
    unbounded_fit: Vec<f64>,
    scales: Vec<f64>,
    matrix: Matrix,
    gradients: Matrix,
    terms: Vec<TermDerivatives>,
    /// One row per argument of each term ([`argument_rows`]).
    arguments: Matrix,
    /// The first row of each term.
    first_arguments: Vec<usize>,
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
    /// not read, or the curvature in its units is not finite (an infinite
    /// information).
    pub fn gauss_newton<E, S>(
        problem: &CalibrationProblem<E>,
        source: &S,
        fit: &[f64],
    ) -> Result<Self, MethodError>
    where
        E: SimulationEngine,
        S: SensitivitySource + ?Sized,
    {
        Self::gauss_newton_and_means(problem, source, fit).map(|(curvature, ..)| curvature)
    }

    /// [`Self::gauss_newton`], the mean of each part of each observation at
    /// `fit` that it reads, and the exponent of the unit of each parameter: a
    /// step of 1 in its units is one of `2^-unit` on the scale of `θ`. The
    /// unit is an exponent because its scale can be above the largest number.
    fn gauss_newton_and_means<E, S>(
        problem: &CalibrationProblem<E>,
        source: &S,
        fit: &[f64],
    ) -> Result<(Self, Vec<f64>, Vec<i32>), MethodError>
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
        let mut arguments = argument_rows(&terms, &parts.jacobian, 1.0);
        let mut argument_errors = argument_rows(&terms, &parts.errors, 0.0);
        let first_arguments: Vec<usize> = terms
            .iter()
            .scan(0, |next, term| {
                let first = *next;
                *next += argument_count(term);
                Some(first)
            })
            .collect();
        let units = units(&terms, &first_arguments, &arguments);
        for (j, &unit) in units.iter().enumerate() {
            for rows in [&mut arguments, &mut argument_errors] {
                rows.column_mut(j)
                    .apply(|entry| *entry = libm::ldexp(*entry, -unit));
            }
        }
        let p = fit.len();
        let mut sums = DMatrix::from_element(p, p, CompensatedSum::default());
        let mut errors = Matrix::zeros(p, p);
        let mut gradients = Matrix::zeros(terms.len(), p);
        for (i, (term, &first)) in terms.iter().zip(&first_arguments).enumerate() {
            let count = argument_count(term);
            let rows = arguments.rows(first, count);
            let row_errors = argument_errors.rows(first, count);
            add_term(&mut sums, Some((&mut errors, row_errors)), term, rows);
            for (a, &slope) in term.gradient.iter().enumerate() {
                // An argument with no slope adds nothing, whatever its row.
                if slope != 0.0 {
                    for j in 0..p {
                        gradients[(i, j)] += slope * rows[(a, j)];
                    }
                }
            }
        }
        let matrix = symmetric(&sums);
        if matrix.iter().any(|entry| !entry.is_finite()) {
            return Err(MethodError::Failed(
                "The curvature at the fit is not finite: a mean sits on the edge of the means \
                 its observation model allows"
                    .to_string(),
            ));
        }
        let names = problem.parameter_names();
        let scales: Vec<f64> = units
            .iter()
            .zip(&transform.derivatives(fit))
            .map(|(&unit, &derivative)| libm::ldexp(derivative, unit))
            .collect();
        let notes = notes(&names, &matrix, &errors);
        let curvature = Self {
            names,
            unbounded_fit: transform.to_unbounded(fit),
            transform,
            fit: fit.to_vec(),
            scales,
            matrix,
            gradients,
            terms,
            arguments,
            first_arguments,
            notes,
        };
        Ok((curvature, sensitivities.parts.means, units))
    }

    /// The curvature of `problem` at `fit` by central differences of its
    /// objective: the stepped central Hessian of the `finitediff` crate of
    /// the argmin fork, with its estimated error. The gradients of the terms
    /// are those of [`Self::gauss_newton`] with `source`, in the units of the
    /// Hessian.
    ///
    /// The step of parameter `j` is `h_j = δ^{1/4} · s_j`, which balances the
    /// truncation error of the central second difference, of order `h²`,
    /// with the error `δ/h²` of an objective of relative accuracy `δ`
    /// (the relative [`SimulationEngine::accuracy`]). The scale `s_j` is
    /// `dθ/du`, the length of a unit step on the unbounded scale, for a parameter
    /// with a bound, and `max(|θ_j|, 1)` for one without: `dθ/du` is below
    /// the distance to each bound, so `θ ± 2h` stays inside them.
    ///
    /// A step is at most the one whose `H_jj h_j²` on the unbounded scale,
    /// with `H_jj` the Gauss–Newton curvature, is `f64::MAX / 32`, so that
    /// the objective at the points of the differences stays a number when the
    /// curvature is large. That step is the power of two at or below the
    /// bound. It is refused unless `θ ± h` and `θ ± 2h` are numbers with no
    /// rounding, since the differences divide by the step, and unless the
    /// share of the rounding of the runs at the fit in the curvature,
    /// `Σ |C| |A| e / (h Ĥ_jj)` by the Gauss–Newton terms (`rounding_of_runs`),
    /// is at most `δ^{1/2}`: the error that the step `δ^{1/4}` accepts, so the
    /// shorter step is never less accurate. A coupling with a parameter of a
    /// far smaller curvature rounds away, and its note says so. The bound is
    /// read in the units of Gauss–Newton, so it is the same in any unit of `θ`.
    ///
    /// The differences are taken in the unit `q_j`, the power of two at or
    /// below `h_j`: at `θ/q` with steps `h/q`, which are the same points, so
    /// that the Hessian is in those units whatever the size of `h`.
    ///
    /// # Errors
    ///
    /// As [`Self::gauss_newton`], also for an entry of `source` that the
    /// differences of the objective do not read, and [`MethodError::Failed`]
    /// when a point of the differences is infeasible or the engine fails, or
    /// when the shorter step is refused: its curvature is too large for the
    /// Hessian.
    pub fn hessian<E, S>(
        problem: &CalibrationProblem<E>,
        source: &S,
        fit: &[f64],
    ) -> Result<Self, MethodError>
    where
        E: SimulationEngine,
        S: SensitivitySource + ?Sized,
    {
        let (expected, means, expected_units) = Self::gauss_newton_and_means(problem, source, fit)?;
        let derivatives = expected.transform.derivatives(fit);
        let accuracy = problem.forward().base_engine().accuracy().relative;
        let roundings: Vec<f64> = means
            .iter()
            .zip(part_backgrounds(problem.observed_data()))
            .map(|(&mean, background)| run_rounding(accuracy, mean, background))
            .collect();
        let factor = accuracy.powf(0.25);
        let mut steps = Vec::with_capacity(fit.len());
        for (j, scale) in expected.transform.scales().iter().enumerate() {
            let step = match scale {
                Scale::Identity => factor * fit[j].abs().max(1.0),
                _ => factor * derivatives[j],
            };
            // A step of 1 in the units of Gauss–Newton is one of `2^unit` on
            // the scale of θ.
            let unit = -expected_units[j];
            let largest = libm::ldexp((OBJECTIVE_ROOM / expected.matrix[(j, j)]).sqrt(), unit);
            if largest >= step {
                steps.push(step);
                continue;
            }
            // The power of two at or below the bound, whose points must be
            // exact, since the differences divide by the step.
            let step = libm::ldexp(1.0, libm::frexp(largest).1 - 1);
            let within = |what: &str| {
                MethodError::Failed(format!(
                    "The curvature of '{name}' at the fit is too large for the Hessian: a step \
                     that keeps its objective well below the largest number is within the \
                     rounding of {what}",
                    name = expected.names[j]
                ))
            };
            let exact = [-2.0, -1.0, 1.0, 2.0]
                .iter()
                .all(|&k| exact_sum(fit[j], k * step));
            if !(largest > 0.0 && exact) {
                return Err(within(&format!("'{}'", expected.names[j])));
            }
            let in_units = libm::ldexp(step, -unit);
            if rounding_of_runs(&expected, &roundings, j)
                > accuracy.sqrt() * in_units * expected.matrix[(j, j)]
            {
                return Err(within("the predictions of the model at the fit"));
            }
            steps.push(step);
        }
        let workspace = RefCell::new(Workspace::default());
        let objective = |values: &Vec<f64>| -> Result<f64, anyhow::Error> {
            problem
                .evaluate(values, &mut workspace.borrow_mut())
                .map_err(|error| anyhow::Error::msg(error.to_string()))
        };
        // The exponent of `q_j`, the power of two at or below a step, which is
        // finite and above 0. A power of two divides and multiplies with no
        // rounding.
        let units: Vec<i32> = steps.iter().map(|&step| libm::frexp(step).1 - 1).collect();
        let in_units = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .zip(&units)
                .map(|(&value, &unit)| libm::ldexp(value, -unit))
                .collect()
        };
        let in_units_objective = |values: &Vec<f64>| -> Result<f64, anyhow::Error> {
            let values: Vec<f64> = values
                .iter()
                .zip(&units)
                .map(|(&value, &unit)| libm::ldexp(value, unit))
                .collect();
            objective(&values)
        };
        let estimate = central_hessian_nograd_error_vec(
            &in_units(fit),
            &in_units_objective,
            &in_units(&steps),
            accuracy,
        )
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
        // A quotient, since the product of `dθ/du` and `1/q` can overflow.
        let scales = in_units(&derivatives);
        let notes = notes(&expected.names, &matrix, &errors);
        let mut curvature = Self {
            matrix,
            notes,
            ..expected
        };
        // The scores and the terms of the blocks, in the units of the Hessian.
        for (j, scale) in scales.iter().enumerate() {
            let ratio = curvature.scales[j] / scale;
            curvature.gradients.column_mut(j).scale_mut(ratio);
            curvature.arguments.column_mut(j).scale_mut(ratio);
        }
        curvature.scales = scales;
        Ok(curvature)
    }

    /// The inverse of the curvature, by
    /// [`commol_statistics::linear_algebra::symmetric_inverse`].
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when the curvature with its diagonal scaled to
    /// 1 is not positive definite, or is singular to working precision
    /// ([`commol_statistics::linear_algebra::singular`]): the data do not
    /// determine every parameter, or the fit is not a minimum. The rule reads
    /// neither the scale of a parameter nor the estimated error of the
    /// derivatives.
    pub fn inverse(&self) -> Result<InverseCurvature<'_>, MethodError> {
        InverseCurvature::new(self)
    }

    /// The names of the parameters.
    pub fn names(&self) -> &[String] {
        &self.names
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

    /// The scale `s_j` of each parameter: a step of 1 in the units of the
    /// curvature is one of `1/s_j` on the unbounded scale, so the curvature
    /// there is `H_u = S Ĥ S` and a gradient `ĝ S`, with `S = diag(s)`.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// The curvature `Ĥ`, in the units of [`Self::scales`].
    pub fn scaled_matrix(&self) -> &Matrix {
        &self.matrix
    }

    /// The gradient of each observation's term of the objective, one row per
    /// observation, in the units of [`Self::scales`]: `Aᵢᵀ ∇ᵢ`, with `∇ᵢ` the
    /// gradient of the term over its arguments. Their block sums are the
    /// scores of the sandwich.
    pub fn scaled_gradients(&self) -> &Matrix {
        &self.gradients
    }

    /// The Gauss–Newton curvature of the terms of `observations` alone, in
    /// the units of [`Self::scales`]: `J_b = Σ_{i∈b} Aᵢᵀ Cᵢ Aᵢ`. For a
    /// Gauss–Newton curvature it is the share of a block `b` of
    /// [`Self::scaled_matrix`], the sum of the shares of all observations; a
    /// Hessian keeps the Gauss–Newton terms, whose shares do not sum to it.
    pub fn scaled_block_matrix(&self, observations: &[usize]) -> Matrix {
        let p = self.fit.len();
        let mut sums = DMatrix::from_element(p, p, CompensatedSum::default());
        for &i in observations {
            let term = &self.terms[i];
            let rows = self
                .arguments
                .rows(self.first_arguments[i], argument_count(term));
            add_term(&mut sums, None, term, rows);
        }
        symmetric(&sums)
    }

    /// The entries whose estimated error is above 1 % of the scale of their
    /// row and column, in words.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }
}

/// The rounding of the runs at the fit in the Gauss–Newton curvature of
/// parameter `j`, per unit of its step, in its units: `Σ_i Σ_{a,b} |C_i,ab|
/// |A_i,bj| e_i,a`, with `e_i,a` the rounding of each mean of a part,
/// `roundings` ([`run_rounding`]), and 0 for a noise argument. A rounding `e`
/// of a mean moves the objective at a step `h` by about `|C A| h e`, and its
/// second difference by that over `h²`. The term of the residual at the fit is
/// left out: under the bound of a step the change of the residual over the
/// step is above it, unless the objective at the fit is near the largest
/// number.
fn rounding_of_runs(curvature: &Curvature, roundings: &[f64], j: usize) -> f64 {
    let mut sum = 0.0;
    let mut part = 0;
    for (term, &first) in curvature.terms.iter().zip(&curvature.first_arguments) {
        let count = argument_count(term);
        for a in 0..term.parts {
            let rounding = roundings[part + a];
            for b in 0..count {
                sum += term.curvature[a * count + b].abs()
                    * curvature.arguments[(first + b, j)].abs()
                    * rounding;
            }
        }
        part += term.parts;
    }
    sum
}

/// Whether `a + b` is a number with no rounding: the error of the sum by
/// Knuth's two-sum, which is exact, is 0.
fn exact_sum(a: f64, b: f64) -> bool {
    let sum = a + b;
    let b_part = sum - a;
    sum.is_finite() && (a - (sum - b_part)) + (b - b_part) == 0.0
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

/// The number of arguments of a term: the means of its parts and its noise
/// arguments.
fn argument_count(term: &TermDerivatives) -> usize {
    term.parts + term.parameters.len()
}

/// One row per argument of each term: the row of `parts` of the mean of each
/// part, and for a noise argument the value `own` at its parameter, or 0 for a
/// number. The rows of the Jacobian have an `own` of 1, and those of its errors
/// one of 0.
fn argument_rows(terms: &[TermDerivatives], parts: &Matrix, own: f64) -> Matrix {
    let count = terms.iter().map(argument_count).sum();
    let mut rows = Matrix::zeros(count, parts.ncols());
    let (mut row, mut part) = (0, 0);
    for term in terms {
        rows.rows_mut(row, term.parts)
            .copy_from(&parts.rows(part, term.parts));
        for (k, parameter) in term.parameters.iter().enumerate() {
            if let Some(j) = parameter {
                rows[(row + term.parts + k, *j)] = own;
            }
        }
        row += argument_count(term);
        part += term.parts;
    }
    rows
}

/// The exponent of the unit of each parameter: of the power of two at the
/// largest `√c_aa |A_aj|` over the arguments of every term, the root of what
/// one argument adds to the curvature of the parameter, or 0 when each is 0.
/// The product of a finite root and a finite derivative can be above the
/// largest number, so its exponent is the sum of those of its factors.
fn units(terms: &[TermDerivatives], first_arguments: &[usize], arguments: &Matrix) -> Vec<i32> {
    let mut largest = vec![None; arguments.ncols()];
    for (term, first) in terms.iter().zip(first_arguments) {
        let count = argument_count(term);
        for a in 0..count {
            let (root, root_exponent) = libm::frexp(term.curvature[a * count + a].sqrt());
            for (j, largest) in largest.iter_mut().enumerate() {
                let (derivative, exponent) = libm::frexp(arguments[(first + a, j)].abs());
                // The fractions are in [1/2, 1) for a finite size that is not
                // 0, and their product in [1/4, 1).
                let fraction = root * derivative;
                if fraction > 0.0 && fraction.is_finite() {
                    let (_, rest) = libm::frexp(fraction);
                    *largest = (*largest).max(Some(root_exponent + exponent + rest));
                }
            }
        }
    }
    largest
        .into_iter()
        .map(|exponent| exponent.map_or(0, |exponent| exponent - 1))
        .collect()
}

/// The entries of the curvature `matrix`, of errors `errors`, both in the
/// same units, whose error is not small, in words: the share of the error,
/// which is the same in any units.
fn notes(names: &[String], matrix: &Matrix, errors: &Matrix) -> Vec<String> {
    let mut notes = Vec::new();
    for j in 0..names.len() {
        for k in j..names.len() {
            let size = (matrix[(j, j)] * matrix[(k, k)]).abs().sqrt();
            let error = errors[(j, k)];
            if error > UNCERTAIN_ENTRY * size {
                let share = if error <= size {
                    format!("{:.1}%", 100.0 * error / size)
                } else {
                    "more than 100%".to_string()
                };
                notes.push(format!(
                    "The curvature of ('{}', '{}') has an estimated error of {share} of the \
                     scale of its row and column",
                    names[j], names[k],
                ));
            }
        }
    }
    notes
}

/// Add `Aᵀ C A` of one term to the upper triangle of `sums`, with `rows` the
/// rows of `A`, one per argument of the term; and, with `errors`, the
/// first-order error that the errors of those rows give it, to the upper
/// triangle of the matrix of errors.
///
/// The sums are compensated: the rounding of a plain sum grows with the number
/// of observations, and gives a direction that the data do not read an
/// eigenvalue above the limit of [`commol_statistics::linear_algebra::singular`].
fn add_term(
    sums: &mut DMatrix<CompensatedSum>,
    mut errors: Option<(&mut Matrix, DMatrixView<'_, f64>)>,
    term: &TermDerivatives,
    rows: DMatrixView<'_, f64>,
) {
    let size = rows.nrows();
    let p = rows.ncols();
    for a in 0..size {
        for b in 0..size {
            let c = term.curvature[a * size + b];
            if c == 0.0 {
                continue;
            }
            for j in 0..p {
                let u = rows[(a, j)];
                let du = errors.as_ref().map_or(0.0, |(_, error)| error[(a, j)]);
                if u == 0.0 && du == 0.0 {
                    continue;
                }
                for k in j..p {
                    let v = rows[(b, k)];
                    sums[(j, k)].add(c * u * v);
                    if let Some((errors, error)) = &mut errors {
                        errors[(j, k)] += c.abs() * (du * v.abs() + u.abs() * error[(b, k)]);
                    }
                }
            }
        }
    }
}

/// The symmetric matrix of the sums of its upper triangle. A diagonal entry
/// that is at most `ε` times the sum of the absolute values of its terms is
/// the rounding of terms that cancel, and is 0.
fn symmetric(sums: &DMatrix<CompensatedSum>) -> Matrix {
    Matrix::from_fn(sums.nrows(), sums.ncols(), |j, k| {
        let sum = &sums[(j.min(k), j.max(k))];
        if j == k && sum.value().abs() <= f64::EPSILON * sum.magnitude() {
            0.0
        } else {
            sum.value()
        }
    })
}
