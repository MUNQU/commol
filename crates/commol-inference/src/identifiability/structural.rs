//! Structural identifiability, local and numerical.
//!
//! The scaled sensitivities are evaluated at the rows of a Latin hypercube
//! over the bounds, with one column per parameter that is not a noise
//! parameter: a noise parameter of an observation model changes the
//! likelihood and never a prediction, so it is left out. At each point the
//! numerical rank counts the singular values above `‖E‖_F`, the Frobenius
//! norm of the estimated errors of the scaled sensitivities: by Weyl's
//! inequality, an error `E` moves no singular value by more than
//! `‖E‖₂ ≤ ‖E‖_F`. The generic rank is the rank at most points.
//!
//! A parameter is identifiable when the null space has no component on it,
//! non-identifiable when it lies in the null space, and identifiable only in
//! combination otherwise. The combinations the data determine are the
//! directions orthogonal to the null space among those parameters: a basis
//! of them in reduced row echelon form, rounded, read as products of powers
//! of the parameters.
//!
//! The error of the runs of an engine that integrates with a tolerance can
//! hide singular values that a tighter tolerance resolves. A point whose rank
//! is below the number of parameters differentiated is evaluated once more
//! with the engine at a tighter accuracy ([`TIGHTENING`], no lower than
//! [`TIGHTEST`]), and a parameter whose verdict changes there has no verdict:
//! the accuracy, not the structure, sets it.

use commol_calibration::{first_infinite_bound, start_points};

use super::{differentiated_columns, skippable};
use crate::error::MethodError;
use commol_core::{Accuracy, AccuracySettings};

use crate::sensitivity::{Matrix, SensitivitySource};

/// Below this projection on the null space a parameter is identifiable.
const IDENTIFIABLE: f64 = 0.05;
/// Above this projection on the null space a parameter is non-identifiable.
const NON_IDENTIFIABLE: f64 = 0.95;
/// Entries of a combination below this share of its largest entry go.
const NEGLIGIBLE: f64 = 0.05;
/// A point evaluated again runs at the accuracies of the engine divided by
/// this factor.
pub const TIGHTENING: f64 = 100.0;
/// The tightest accuracy a point is evaluated again at.
pub const TIGHTEST: f64 = 1e-14;

/// What the structure of the model and the design say about a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralVerdict {
    /// The null space has no component on it.
    Identifiable,
    /// Only combinations with other parameters are identifiable.
    InCombination,
    /// It lies in the null space: no change of it shows in the outputs.
    NonIdentifiable,
}

/// The rank of the scaled sensitivities at one point.
#[derive(Debug, Clone, PartialEq)]
pub struct PointRank {
    /// The parameter values of the point, one per parameter.
    pub values: Vec<f64>,
    /// The accuracy of the runs the sensitivities come from.
    pub accuracy: Accuracy,
    /// The singular values, largest first, one per parameter differentiated.
    pub singular_values: Vec<f64>,
    /// The threshold `‖E‖_F` a singular value must pass.
    pub threshold: f64,
    /// How many singular values pass it.
    pub rank: usize,
    /// The rank at the tighter accuracy, when the point was evaluated again
    /// there.
    pub tighter: Option<TighterRank>,
}

/// The rank of the scaled sensitivities at a point evaluated again at a
/// tighter accuracy.
#[derive(Debug, Clone, PartialEq)]
pub struct TighterRank {
    /// The accuracy of the runs.
    pub accuracy: Accuracy,
    /// The singular values, largest first, one per parameter differentiated.
    pub singular_values: Vec<f64>,
    /// The threshold `‖E‖_F` a singular value must pass.
    pub threshold: f64,
    /// How many singular values pass it.
    pub rank: usize,
}

/// The tighter accuracy a point is evaluated again at, and the settings of
/// the engine that set it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tightening {
    /// The accuracy of the runs.
    pub accuracy: Accuracy,
    /// The settings that set it, as a user writes them.
    pub settings: AccuracySettings,
}

/// A combination of parameters the data determine: the product of each
/// parameter (by index) raised to its exponent.
#[derive(Debug, Clone, PartialEq)]
pub struct Combination {
    /// `(parameter index, exponent)`, by index.
    pub exponents: Vec<(usize, f64)>,
}

/// The result of [`structural_rank`].
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralAnalysis {
    /// Each feasible point, in order.
    pub points: Vec<PointRank>,
    /// Why each infeasible point was skipped.
    pub skipped: Vec<String>,
    /// The parameters differentiated, by index: every parameter that is not
    /// a noise parameter. `projections` and `verdicts` hold one entry per
    /// parameter differentiated, in this order; the others are left out.
    pub parameters: Vec<usize>,
    /// The rank at most points (the lowest on a tie), at the accuracy of the
    /// engine.
    pub generic_rank: usize,
    /// The norm of the projection of each parameter differentiated on the
    /// null space, averaged over the points of generic rank, at the accuracy
    /// of the engine: 0 when the null space has no component on it, 1 when
    /// it lies in the null space.
    pub projections: Vec<f64>,
    /// The verdict of each parameter differentiated; `None` when the verdict
    /// changes at the tighter accuracy, or a point could not be evaluated
    /// there and the parameter is not identifiable.
    pub verdicts: Vec<Option<StructuralVerdict>>,
    /// The identifiable combinations of the parameters
    /// [`StructuralVerdict::InCombination`], at the first point of generic
    /// rank; each exponent is of a parameter by its index in the problem.
    pub combinations: Vec<Combination>,
    /// The accuracy of the runs of the engine.
    pub accuracy: Accuracy,
    /// The tighter accuracy a point of rank below the number of parameters
    /// differentiated is evaluated again at, with the settings that set it;
    /// `None` when no setting sets the accuracy of the runs, or the accuracy
    /// is already the tightest.
    pub tighter: Option<Tightening>,
    /// The rank at most points with each point evaluated again at the
    /// tighter accuracy in place of its rank at the accuracy of the engine;
    /// `None` when no point was evaluated again.
    pub tighter_generic_rank: Option<usize>,
    /// Why each point that the tighter accuracy should check could not be
    /// evaluated there.
    pub unchecked: Vec<String>,
}

/// The rank and the null space of the scaled sensitivities at a point.
struct Evaluation {
    singular_values: Vec<f64>,
    threshold: f64,
    rank: usize,
    null_space: Matrix,
}

impl Evaluation {
    fn of(sensitivities: &crate::sensitivity::Sensitivities) -> Self {
        let threshold = sensitivities.errors.norm();
        let (singular_values, right) = singular_decomposition(&sensitivities.jacobian);
        let rank = singular_values.iter().filter(|&&s| s > threshold).count();
        let null_space = right.rows(rank, right.nrows() - rank).into_owned();
        Self {
            singular_values,
            threshold,
            rank,
            null_space,
        }
    }
}

/// The accuracies of the engine divided by [`TIGHTENING`], no lower than
/// [`TIGHTEST`] and no looser than they are; `None` when that is no
/// tighter.
fn tighter_than(accuracy: Accuracy) -> Option<Accuracy> {
    let tighten = |value: f64| (value / TIGHTENING).max(TIGHTEST).min(value);
    let tighter = Accuracy {
        relative: tighten(accuracy.relative),
        absolute: tighten(accuracy.absolute),
    };
    (tighter != accuracy).then_some(tighter)
}

/// The structural rank of `source` at `points` rows of a Latin hypercube over
/// the bounds, drawn from `seed`, over the parameters that are not noise
/// parameters.
///
/// When a setting sets the accuracy of the runs of the source
/// ([`SensitivitySource::accuracy_settings`]) and the source runs at another
/// ([`SensitivitySource::at_accuracy`]), each point whose rank is below the
/// number of parameters differentiated is evaluated again at a tighter
/// accuracy, and the verdicts are compared with those of the points at that
/// accuracy.
///
/// # Errors
///
/// [`MethodError::Invalid`] when `points` is 0, a bound is not finite, or
/// every parameter is a noise parameter;
/// [`MethodError::NoFeasiblePoint`] when every point is infeasible;
/// [`MethodError::Sensitivity`] when the sensitivities fail for
/// another reason.
pub fn structural_rank<S: SensitivitySource>(
    source: &S,
    points: usize,
    seed: u64,
) -> Result<StructuralAnalysis, MethodError> {
    crate::ranges::POINTS
        .check(points)
        .map_err(MethodError::Invalid)?;
    let parameters = source.parameters();
    if let Some((name, (min, max))) = first_infinite_bound(
        parameters
            .iter()
            .map(|p| (p.id.as_str(), (p.min_bound, p.max_bound))),
    ) {
        return Err(MethodError::Invalid(format!(
            "StructuralRank evaluates points of a Latin hypercube over the bounds, so they \
             must be finite: parameter '{name}' has the bounds [{min}, {max}]. Give it finite \
             bounds"
        )));
    }
    let differentiated = differentiated_columns(parameters, "StructuralRank")?;
    let accuracy = source.accuracy();
    let tighter = source.accuracy_settings().and_then(|settings| {
        let accuracy = tighter_than(accuracy)?;
        let source = source.at_accuracy(accuracy)?;
        Some((Tightening { accuracy, settings }, source))
    });
    let mut ranks = Vec::with_capacity(points);
    let mut evaluations = Vec::with_capacity(points);
    let mut tighter_evaluations = Vec::with_capacity(points);
    let mut skipped = Vec::new();
    let mut unchecked = Vec::new();
    for values in start_points(parameters, points + 1, seed)
        .into_iter()
        .skip(1)
    {
        let sensitivities = match source.scaled_jacobian(&values, &differentiated) {
            Ok(sensitivities) => sensitivities,
            Err(error) if skippable(&error) => {
                skipped.push(error.to_string());
                continue;
            }
            Err(error) => return Err(MethodError::Sensitivity(error)),
        };
        let evaluation = Evaluation::of(&sensitivities);
        let again = match &tighter {
            Some((tighter, source)) if evaluation.rank < differentiated.len() => {
                match source.scaled_jacobian(&values, &differentiated) {
                    Ok(sensitivities) => Some((tighter.accuracy, Evaluation::of(&sensitivities))),
                    Err(error) => {
                        unchecked.push(error.to_string());
                        None
                    }
                }
            }
            _ => None,
        };
        ranks.push(PointRank {
            values,
            accuracy,
            singular_values: evaluation.singular_values.clone(),
            threshold: evaluation.threshold,
            rank: evaluation.rank,
            tighter: again.as_ref().map(|(accuracy, again)| TighterRank {
                accuracy: *accuracy,
                singular_values: again.singular_values.clone(),
                threshold: again.threshold,
                rank: again.rank,
            }),
        });
        evaluations.push(evaluation);
        tighter_evaluations.push(again.map(|(_, again)| again));
    }
    if ranks.is_empty() {
        return Err(MethodError::NoFeasiblePoint {
            points,
            reasons: skipped,
        });
    }
    let loose: Vec<&Evaluation> = evaluations.iter().collect();
    let (generic_rank, projections, generic) = projections_of(&loose, differentiated.len());
    let verdicts = verdicts_of(&projections);
    let tight: Vec<&Evaluation> = evaluations
        .iter()
        .zip(&tighter_evaluations)
        .map(|(loose, tight)| tight.as_ref().unwrap_or(loose))
        .collect();
    let (tight_rank, tight_projections, _) = projections_of(&tight, differentiated.len());
    let tight_verdicts = verdicts_of(&tight_projections);
    let tighter_generic_rank = tighter_evaluations
        .iter()
        .any(Option::is_some)
        .then_some(tight_rank);
    let resolved: Vec<Option<StructuralVerdict>> = verdicts
        .iter()
        .zip(&tight_verdicts)
        .map(|(&verdict, &tight)| {
            let checked = unchecked.is_empty() || verdict == StructuralVerdict::Identifiable;
            (verdict == tight && checked).then_some(verdict)
        })
        .collect();
    let combined: Vec<usize> = (0..differentiated.len())
        .filter(|&j| verdicts[j] == StructuralVerdict::InCombination)
        .collect();
    let combinations = combinations(&evaluations[generic].null_space, &combined)
        .into_iter()
        .filter(|combination| {
            combination
                .exponents
                .iter()
                .all(|&(j, _)| resolved[j].is_some())
        })
        .map(|combination| Combination {
            exponents: combination
                .exponents
                .into_iter()
                .map(|(j, exponent)| (differentiated[j], exponent))
                .collect(),
        })
        .collect();
    Ok(StructuralAnalysis {
        points: ranks,
        skipped,
        parameters: differentiated,
        generic_rank,
        projections,
        verdicts: resolved,
        combinations,
        accuracy,
        tighter: tighter.map(|(tighter, _)| tighter),
        tighter_generic_rank,
        unchecked,
    })
}

/// The generic rank of `evaluations`, the projection of each of the
/// `parameters` on the null space averaged over the points of generic rank,
/// and the index of the first of them.
fn projections_of(evaluations: &[&Evaluation], parameters: usize) -> (usize, Vec<f64>, usize) {
    let generic_rank = mode(evaluations.iter().map(|evaluation| evaluation.rank));
    let generic: Vec<&Matrix> = evaluations
        .iter()
        .filter(|evaluation| evaluation.rank == generic_rank)
        .map(|evaluation| &evaluation.null_space)
        .collect();
    let projections = (0..parameters)
        .map(|j| {
            generic
                .iter()
                .map(|null| null.column(j).norm())
                .sum::<f64>()
                / generic.len() as f64
        })
        .collect();
    let first = evaluations
        .iter()
        .position(|evaluation| evaluation.rank == generic_rank)
        .expect("the generic rank is the rank of a point");
    (generic_rank, projections, first)
}

/// The verdict of each projection on the null space.
fn verdicts_of(projections: &[f64]) -> Vec<StructuralVerdict> {
    projections
        .iter()
        .map(|&projection| {
            if projection < IDENTIFIABLE {
                StructuralVerdict::Identifiable
            } else if projection > NON_IDENTIFIABLE {
                StructuralVerdict::NonIdentifiable
            } else {
                StructuralVerdict::InCombination
            }
        })
        .collect()
}

/// The singular values of `matrix`, largest first, one per column, and the
/// matching right singular vectors as rows.
fn singular_decomposition(matrix: &Matrix) -> (Vec<f64>, Matrix) {
    let columns = matrix.ncols();
    // Zero rows give one singular value per column even with fewer rows.
    let mut padded = Matrix::zeros(matrix.nrows().max(columns), columns);
    padded.rows_mut(0, matrix.nrows()).copy_from(matrix);
    let svd = padded.svd(false, true);
    let right = svd.v_t.expect("the right singular vectors were asked for");
    let mut order: Vec<usize> = (0..columns).collect();
    order.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
    let values = order.iter().map(|&k| svd.singular_values[k]).collect();
    let rows = Matrix::from_fn(columns, columns, |r, c| right[(order[r], c)]);
    (values, rows)
}

/// The most frequent value, the lowest on a tie.
fn mode(values: impl Iterator<Item = usize>) -> usize {
    let mut counts = std::collections::BTreeMap::new();
    for value in values {
        *counts.entry(value).or_insert(0usize) += 1;
    }
    let most = counts.values().copied().max().unwrap_or(0);
    counts
        .into_iter()
        .find(|&(_, count)| count == most)
        .map_or(0, |(value, _)| value)
}

/// A basis of the directions orthogonal to the rows of `null` among the
/// parameters `combined`, rounded: one combination per free column of the
/// reduced row echelon form of `null` restricted to them.
fn combinations(null: &Matrix, combined: &[usize]) -> Vec<Combination> {
    if combined.is_empty() {
        return Vec::new();
    }
    let mut reduced = Matrix::from_fn(null.nrows(), combined.len(), |r, c| null[(r, combined[c])]);
    let pivots = row_echelon(&mut reduced);
    (0..combined.len())
        .filter(|column| !pivots.contains(column))
        .filter_map(|free| {
            let mut direction = vec![0.0; combined.len()];
            direction[free] = 1.0;
            for (row, &pivot) in pivots.iter().enumerate() {
                direction[pivot] = -reduced[(row, free)];
            }
            rounded(&direction).map(|exponents| Combination {
                exponents: exponents
                    .into_iter()
                    .map(|(c, exponent)| (combined[c], exponent))
                    .collect(),
            })
        })
        .collect()
}

/// Reduce `matrix` in place to reduced row echelon form, with partial
/// pivoting; returns the pivot column of each leading row.
fn row_echelon(matrix: &mut Matrix) -> Vec<usize> {
    let (rows, columns) = matrix.shape();
    let tolerance = 1e-8 * matrix.amax().max(1.0);
    let mut pivots = Vec::new();
    let mut row = 0;
    for column in 0..columns {
        if row == rows {
            break;
        }
        let (best, size) = (row..rows)
            .map(|r| (r, matrix[(r, column)].abs()))
            .fold((row, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        if size <= tolerance {
            continue;
        }
        matrix.swap_rows(row, best);
        let pivot = matrix[(row, column)];
        for c in 0..columns {
            matrix[(row, c)] /= pivot;
        }
        for r in 0..rows {
            if r != row {
                let factor = matrix[(r, column)];
                for c in 0..columns {
                    matrix[(r, c)] -= factor * matrix[(row, c)];
                }
            }
        }
        pivots.push(column);
        row += 1;
    }
    pivots
}

/// The entries of `direction` above [`NEGLIGIBLE`] of the largest, divided
/// by the smallest of them in absolute value and rounded to the nearest
/// half, with a positive first entry; `None` when nothing is left.
fn rounded(direction: &[f64]) -> Option<Vec<(usize, f64)>> {
    let largest = direction.iter().map(|d| d.abs()).fold(0.0, f64::max);
    let kept: Vec<(usize, f64)> = direction
        .iter()
        .enumerate()
        .filter(|(_, d)| d.abs() >= NEGLIGIBLE * largest && **d != 0.0)
        .map(|(i, &d)| (i, d))
        .collect();
    let smallest = kept
        .iter()
        .map(|(_, d)| d.abs())
        .fold(f64::INFINITY, f64::min);
    let sign = kept.first()?.1.signum();
    let exponents: Vec<(usize, f64)> = kept
        .into_iter()
        .map(|(i, d)| (i, (2.0 * sign * d / smallest).round() / 2.0))
        .filter(|(_, e)| *e != 0.0)
        .collect();
    (!exponents.is_empty()).then_some(exponents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generic_rank_is_the_most_frequent_and_the_lowest_on_a_tie() {
        assert_eq!(mode([1, 2, 2].into_iter()), 2);
        assert_eq!(mode([3, 1, 3, 1].into_iter()), 1);
    }

    #[test]
    fn the_reduced_row_echelon_form_has_one_pivot_per_independent_row() {
        let mut matrix = Matrix::from_row_slice(2, 3, &[2.0, 4.0, 2.0, 1.0, 3.0, 2.0]);
        let pivots = row_echelon(&mut matrix);
        assert_eq!(pivots, vec![0, 1]);
        let expected = Matrix::from_row_slice(2, 3, &[1.0, 0.0, -1.0, 0.0, 1.0, 1.0]);
        assert!((matrix - expected).amax() < 1e-12);
    }

    #[test]
    fn an_entry_below_the_tolerance_of_the_largest_is_no_pivot() {
        // 1e-6 is below 1e-8 × 1e4.
        let mut matrix = Matrix::from_row_slice(2, 2, &[1e4, 0.0, 0.0, 1e-6]);
        assert_eq!(row_echelon(&mut matrix), vec![0]);
    }

    #[test]
    fn a_direction_is_rounded_to_halves_of_its_smallest_entry() {
        // The sign makes the first entry positive.
        assert_eq!(rounded(&[-0.5, 1.0]), Some(vec![(0, 1.0), (1, -2.0)]));
        assert_eq!(rounded(&[0.3, 0.45]), Some(vec![(0, 1.0), (1, 1.5)]));
    }

    #[test]
    fn entries_below_a_twentieth_of_the_largest_are_dropped() {
        assert_eq!(rounded(&[2.0, 1.0, 0.05]), Some(vec![(0, 2.0), (1, 1.0)]));
        assert_eq!(rounded(&[0.0, 0.0]), None);
    }
}
