//! Global sensitivity analysis: which inputs move the quantities of interest
//! over the whole input space.
//!
//! Each method is an estimator on a design: the design comes from
//! [`crate::design`], its points run through [`crate::batch::BatchRunner`],
//! and the estimator here reads the inputs and the quantities of each point.
//! [`ForwardAnalysis`] joins the three for each method.
//!
//! | Module | Estimator |
//! |---|---|
//! | [`prcc`](mod@prcc) | Partial rank correlation coefficients and their significance, weighted |
//! | [`morris`](mod@morris) | Elementary effects along Morris trajectories: `μ`, `μ*` and `σ` |
//! | [`sobol`](mod@sobol) | First-order and total Sobol indices on the design of Saltelli, with bootstrap intervals |
//! | [`analysis`] | [`ForwardAnalysis`]: the forward model, the inputs and the seed, and every method on them |
//! | [`chaos`] | Polynomial chaos expansions in Legendre polynomials of `F(x)`, and the Sobol indices of their coefficients |

pub mod analysis;
pub mod chaos;
pub mod morris;
pub mod prcc;
pub mod sobol;

pub use analysis::{ForwardAnalysis, Propagation};
pub use chaos::ChaosExpansion;
pub use morris::{Morris, morris};
pub use prcc::{Prcc, prcc};
pub use sobol::{Sobol, sobol};

use crate::predictive::PredictiveError;

/// Check that no quantity is NaN at a point of a design, before an
/// estimator reduces the quantities (`[point][quantity]`) across the points.
///
/// # Errors
///
/// [`PredictiveError::NotANumber`] with the first quantity that is NaN at a
/// point, and every point where it is.
pub(crate) fn check_numbers(outputs: &[Vec<f64>]) -> Result<(), PredictiveError> {
    let quantities = outputs.iter().map(Vec::len).max().unwrap_or(0);
    let nan = |row: &Vec<f64>, quantity: usize| row.get(quantity).is_some_and(|v| v.is_nan());
    match (0..quantities).find(|&q| outputs.iter().any(|row| nan(row, q))) {
        None => Ok(()),
        Some(quantity) => Err(PredictiveError::NotANumber {
            quantity,
            points: (0..outputs.len())
                .filter(|&point| nan(&outputs[point], quantity))
                .collect(),
            run: None,
        }),
    }
}
