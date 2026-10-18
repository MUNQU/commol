//! The block jackknife: one fit per block left out.
//!
//! With `B` fits `θ̂₍₋ᵦ₎` around the fit `θ̂` and their mean `θ̄`, the
//! jackknife bias is `(B − 1)(θ̄ − θ̂)`, its standard error
//! `√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)`, the pseudo-value of block `b` is
//! `B θ̂ − (B − 1) θ̂₍₋ᵦ₎`, and the estimate is the fit minus the bias, the
//! mean of the pseudo-values (Künsch 1989). The pseudo-values are not
//! draws: their spread needs the `(B − 1)` scaling to give an interval.

use commol_core::SimulationEngine;
use rayon::prelude::*;

use crate::resampling::blocks::BlockScheme;
use crate::resampling::resamplers::Perturbation;
use crate::resampling::runner::{ReplicateRunner, every_replicate_failed};

/// The statistics of a jackknife, one value per parameter in each vector.
#[derive(Debug, Clone, PartialEq)]
pub struct JackknifeStatistics {
    /// The bias-corrected estimate.
    pub estimate: Vec<f64>,
    /// The jackknife bias.
    pub bias: Vec<f64>,
    /// The jackknife standard error.
    pub standard_error: Vec<f64>,
    /// The pseudo-values, one row per block.
    pub pseudo_values: Vec<Vec<f64>>,
    /// How far leaving each block out moves each parameter, in standard
    /// errors: `(θ̂₍₋ᵦ₎ − θ̂) / se`, one row per block. NaN or infinite when
    /// a standard error is 0.
    pub influence: Vec<Vec<f64>>,
}

/// The jackknife statistics of `leave_outs`, the fits with one block left
/// out each, around the fit `fit`.
///
/// # Panics
///
/// Panics with fewer than two leave-outs.
pub fn jackknife_statistics(fit: &[f64], leave_outs: &[Vec<f64>]) -> JackknifeStatistics {
    assert!(leave_outs.len() >= 2, "a jackknife needs two leave-outs");
    let blocks = leave_outs.len() as f64;
    let parameters = fit.len();
    let mean: Vec<f64> = (0..parameters)
        .map(|j| leave_outs.iter().map(|row| row[j]).sum::<f64>() / blocks)
        .collect();
    let bias: Vec<f64> = (0..parameters)
        .map(|j| (blocks - 1.0) * (mean[j] - fit[j]))
        .collect();
    let standard_error: Vec<f64> = (0..parameters)
        .map(|j| {
            let squares: f64 = leave_outs
                .iter()
                .map(|row| (row[j] - mean[j]).powi(2))
                .sum();
            ((blocks - 1.0) / blocks * squares).sqrt()
        })
        .collect();
    let pseudo_values = leave_outs
        .iter()
        .map(|row| {
            (0..parameters)
                .map(|j| blocks * fit[j] - (blocks - 1.0) * row[j])
                .collect()
        })
        .collect();
    let influence = leave_outs
        .iter()
        .map(|row| {
            (0..parameters)
                .map(|j| (row[j] - fit[j]) / standard_error[j])
                .collect()
        })
        .collect();
    JackknifeStatistics {
        estimate: (0..parameters).map(|j| fit[j] - bias[j]).collect(),
        bias,
        standard_error,
        pseudo_values,
        influence,
    }
}

/// A block jackknife: its statistics and what happened to its fits.
#[derive(Debug, Clone, PartialEq)]
pub struct Jackknife {
    /// The statistics over the blocks that fitted.
    pub statistics: JackknifeStatistics,
    /// The index of the block of each row of the statistics.
    pub blocks: Vec<usize>,
    /// The index and reason of each block whose fit failed.
    pub failed: Vec<(usize, String)>,
    /// The number of fits kept although their optimizer did not converge.
    pub not_converged: usize,
}

/// The block jackknife of `runner`'s problem with the blocks of `scheme`:
/// each block in turn gets weight 0, in parallel.
///
/// # Errors
///
/// Returns a message when `scheme` has fewer than two blocks, or fewer than
/// two fits succeed.
pub fn block_jackknife<E>(
    runner: &ReplicateRunner<'_, E>,
    scheme: &BlockScheme,
) -> Result<Jackknife, String>
where
    E: SimulationEngine + Send + Sync,
{
    if scheme.len() < 2 {
        return Err(format!(
            "block_length {} makes {} block; a jackknife needs at least two",
            scheme.length(),
            scheme.len()
        ));
    }
    let problem = runner.problem();
    let weights: Vec<f64> = problem.observed_data().iter().map(|o| o.weight).collect();
    let outcomes: Vec<_> = scheme
        .blocks()
        .par_iter()
        .enumerate()
        .map(|(block, members)| {
            let mut left_out = weights.clone();
            for &observation in members {
                left_out[observation] = 0.0;
            }
            runner.fit_one(block, &Perturbation::Weights(left_out))
        })
        .collect();
    let mut leave_outs = Vec::new();
    let mut blocks = Vec::new();
    let mut failed = Vec::new();
    let mut not_converged = 0;
    for (block, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Ok(result) => {
                not_converged += usize::from(!result.converged);
                leave_outs.push(problem.applied_parameter_values(&result.best_parameters));
                blocks.push(block);
            }
            Err(reason) => failed.push((block, reason)),
        }
    }
    if leave_outs.len() < 2 {
        if leave_outs.is_empty() {
            return Err(every_replicate_failed(&failed));
        }
        return Err(format!(
            "Only one block left out could be fitted; a jackknife needs two. {}",
            every_replicate_failed(&failed)
        ));
    }
    Ok(Jackknife {
        statistics: jackknife_statistics(runner.fit(), &leave_outs),
        blocks,
        failed,
        not_converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_jackknife_of_a_mean_is_the_classical_standard_error() {
        let values = [2.0, 4.0, 9.0, 1.0, 5.0];
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let leave_outs: Vec<Vec<f64>> = (0..values.len())
            .map(|i| vec![(mean * n - values[i]) / (n - 1.0)])
            .collect();
        let statistics = jackknife_statistics(&[mean], &leave_outs);
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert!((statistics.standard_error[0] - (variance / n).sqrt()).abs() < 1e-12);
        assert!(statistics.bias[0].abs() < 1e-12);
        // The pseudo-values of a mean are the values.
        for (pseudo, value) in statistics.pseudo_values.iter().zip(values) {
            assert!((pseudo[0] - value).abs() < 1e-12);
        }
    }

    #[test]
    fn a_biased_fit_moves_the_estimate_by_its_bias() {
        // Three blocks, mean 4: bias 2 × (4 − 1), standard error √(2/3 × 8).
        let statistics = jackknife_statistics(&[1.0], &[vec![2.0], vec![4.0], vec![6.0]]);
        let se = (16.0f64 / 3.0).sqrt();
        assert_eq!(statistics.bias, vec![6.0]);
        assert_eq!(statistics.estimate, vec![-5.0]);
        assert!((statistics.standard_error[0] - se).abs() < 1e-12);
        assert_eq!(
            statistics.pseudo_values,
            vec![vec![-1.0], vec![-5.0], vec![-9.0]]
        );
        for (influence, moved) in statistics.influence.iter().zip([1.0, 3.0, 5.0]) {
            assert!((influence[0] - moved / se).abs() < 1e-12);
        }
    }
}
