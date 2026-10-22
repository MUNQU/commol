//! Morris screening (Morris 1991), with the `μ*` of Campolongo, Cariboni and
//! Saltelli (2007).
//!
//! Each step of a trajectory moves one input, so the change of a quantity
//! over the step, divided by the change of the input, is an elementary
//! effect of that input. The changes are on the scale of probabilities,
//! `F(x)`, so the effects of inputs of different units compare. Per input,
//! `μ*`, the mean of the absolute effects, ranks the inputs by importance;
//! `σ`, the standard deviation of the effects, is large when the effect of
//! the input depends on where the other inputs are (interactions or a
//! non-linear effect); `μ` is the signed mean.

use super::check_numbers;
use crate::design::MorrisDesign;
use crate::error::MethodError;

/// The Morris indices of every input for every quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct Morris {
    /// The mean of the elementary effects, `[quantity][input]`.
    pub mu: Vec<Vec<f64>>,
    /// The mean of the absolute elementary effects, `[quantity][input]`.
    pub mu_star: Vec<Vec<f64>>,
    /// The standard deviation of the elementary effects (divisor `r − 1`),
    /// `[quantity][input]`; NaN with one trajectory.
    pub sigma: Vec<Vec<f64>>,
}

/// The Morris indices of `design` from the quantities of its points,
/// `[point][quantity]`.
///
/// # Errors
///
/// [`MethodError::Prediction`] of [`PredictiveError::NotANumber`](crate::predictive::PredictiveError::NotANumber) with the first quantity that is NaN at a
/// point, before any effect; [`MethodError::Invalid`] when there is not
/// one row of quantities per point.
pub fn morris(design: &MorrisDesign, outputs: &[Vec<f64>]) -> Result<Morris, MethodError> {
    check_numbers(outputs)?;
    effects(design, outputs).map_err(MethodError::Invalid)
}

fn effects(design: &MorrisDesign, outputs: &[Vec<f64>]) -> Result<Morris, String> {
    if outputs.len() != design.points.len() {
        return Err(format!(
            "Morris needs one row of quantities per point: {} points, {} rows",
            design.points.len(),
            outputs.len()
        ));
    }
    let inputs = design.steps.first().map_or(0, Vec::len);
    let quantities = outputs.first().map_or(0, Vec::len);
    let trajectories = design.steps.len();
    // effects[quantity][input][trajectory]
    let mut effects = vec![vec![Vec::with_capacity(trajectories); inputs]; quantities];
    for (t, steps) in design.steps.iter().enumerate() {
        let start = t * (inputs + 1);
        for (s, step) in steps.iter().enumerate() {
            let (before, after) = (&outputs[start + s], &outputs[start + s + 1]);
            for q in 0..quantities {
                effects[q][step.input].push((after[q] - before[q]) / step.change);
            }
        }
    }
    let count = trajectories as f64;
    let reduce = |f: &dyn Fn(&[f64]) -> f64| -> Vec<Vec<f64>> {
        effects
            .iter()
            .map(|by_input| by_input.iter().map(|e| f(e)).collect())
            .collect()
    };
    let mean = |e: &[f64]| e.iter().sum::<f64>() / count;
    Ok(Morris {
        mu: reduce(&mean),
        mu_star: reduce(&|e| e.iter().map(|v| v.abs()).sum::<f64>() / count),
        sigma: reduce(&|e| {
            let m = mean(e);
            (e.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (count - 1.0)).sqrt()
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::MorrisStep;

    #[test]
    fn an_effect_is_the_change_of_the_quantity_over_the_change_of_the_input() {
        // Two trajectories of one input: up by ½ from 1 to 3, down by ½ from
        // 5 to 4.
        let design = MorrisDesign {
            points: vec![vec![0.25], vec![0.75], vec![0.75], vec![0.25]],
            steps: vec![
                vec![MorrisStep {
                    input: 0,
                    change: 0.5,
                }],
                vec![MorrisStep {
                    input: 0,
                    change: -0.5,
                }],
            ],
        };
        let outputs = vec![vec![1.0], vec![3.0], vec![5.0], vec![4.0]];

        let result = morris(&design, &outputs).unwrap();

        // Effects 4 and 2.
        assert_eq!(result.mu, vec![vec![3.0]]);
        assert_eq!(result.mu_star, vec![vec![3.0]]);
        assert!((result.sigma[0][0] - 2f64.sqrt()).abs() < 1e-12);
        assert!(morris(&design, &outputs[..3]).is_err());
    }
}
