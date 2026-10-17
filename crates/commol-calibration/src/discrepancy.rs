//! The discrepancy: how far predicted observations are from the observed
//! ones.
//!
//! It holds the observed values, their weights and per-series normalization,
//! the loss, and the summary of the replicates of a stochastic engine. It
//! never runs an engine.

use std::collections::HashMap;

use crate::types::{LossConfig, ObservedDataPoint, ReplicateSummary};

/// The loss of predicted observations against the observed ones.
#[derive(Debug, Clone)]
pub struct Discrepancy {
    loss: LossConfig,
    observed: Vec<f64>,
    weights: Vec<f64>,
    labels: Vec<String>,
    normalization: Vec<f64>,
    summary: Option<ReplicateSummary>,
}

impl Discrepancy {
    /// The discrepancy of `observed_data` under `loss`, with no
    /// normalization and one run per evaluation.
    pub fn new(loss: LossConfig, observed_data: &[ObservedDataPoint]) -> Self {
        Self {
            loss,
            observed: observed_data.iter().map(|obs| obs.value).collect(),
            weights: observed_data.iter().map(|obs| obs.weight).collect(),
            labels: observed_data.iter().map(|obs| obs.label.clone()).collect(),
            normalization: vec![1.0; observed_data.len()],
            summary: None,
        }
    }

    /// Enable or disable per-series normalization
    /// ([`observation_normalization`]).
    pub(crate) fn set_normalization(&mut self, enabled: bool) {
        self.normalization = if enabled {
            observation_normalization(&self.labels, &self.observed)
        } else {
            vec![1.0; self.observed.len()]
        };
    }

    /// Compare the observations with this summary of the replicates; `None`
    /// compares them with the one run of a deterministic engine.
    pub(crate) fn set_summary(&mut self, summary: Option<ReplicateSummary>) {
        self.summary = summary;
    }

    /// The loss.
    pub fn loss(&self) -> LossConfig {
        self.loss
    }

    /// The factor each residual is multiplied by: `1 / rms` of its series
    /// when normalization is enabled, otherwise 1.
    pub fn normalization(&self) -> &[f64] {
        &self.normalization
    }

    /// The discrepancy of `predictions`, which hold one value per
    /// observation and replicate, replicate after replicate. The predictions
    /// are reordered.
    pub fn of(&self, predictions: &mut [f64]) -> f64 {
        let n = self.observed.len();
        match self.summary {
            None => self.loss.aggregate(
                self.observed
                    .iter()
                    .zip(predictions.iter())
                    .zip(&self.weights)
                    .zip(&self.normalization)
                    .map(|(((observed, predicted), &weight), normalization)| {
                        ((observed - predicted) * normalization, weight)
                    }),
            ),
            Some(summary) => {
                let replicates = predictions.len() / n.max(1);
                let mut per_observation: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..replicates).map(|r| predictions[r * n + i]).collect())
                    .collect();
                self.loss.replicate_loss(
                    summary,
                    self.observed
                        .iter()
                        .zip(per_observation.iter_mut())
                        .zip(&self.weights)
                        .zip(&self.normalization)
                        .map(|(((&observed, values), &weight), &normalization)| {
                            (observed, values.as_mut_slice(), weight, normalization)
                        }),
                )
            }
        }
    }
}

/// The normalization factor of each observation: `1 / rms` of the observed
/// values of its series (its label), or 1 for a series whose values are all
/// (near) zero.
pub fn observation_normalization(labels: &[String], observed: &[f64]) -> Vec<f64> {
    series_rms(labels, observed)
        .into_iter()
        .map(|rms| 1.0 / rms)
        .collect()
}

/// The scale of each value: the root mean square of the values of its series
/// (its label), or 1 for a series whose values are all (near) zero.
pub fn series_rms(labels: &[String], values: &[f64]) -> Vec<f64> {
    let mut sums: HashMap<&str, (f64, usize)> = HashMap::new();
    for (label, value) in labels.iter().zip(values) {
        let entry = sums.entry(label.as_str()).or_insert((0.0, 0));
        entry.0 += value * value;
        entry.1 += 1;
    }
    labels
        .iter()
        .map(|label| {
            let (sum_squares, count) = sums[label.as_str()];
            let rms = (sum_squares / count as f64).sqrt();
            if rms > f64::EPSILON {
                rms
            } else {
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed() -> Vec<ObservedDataPoint> {
        vec![
            ObservedDataPoint::new(1.0, "A".to_string(), 3.0),
            ObservedDataPoint::new(2.0, "A".to_string(), 4.0),
            ObservedDataPoint::new(1.0, "B".to_string(), 0.0),
        ]
    }

    #[test]
    fn each_series_is_scaled_by_the_inverse_of_its_rms() {
        let mut discrepancy = Discrepancy::new(LossConfig::SumSquaredError, &observed());
        discrepancy.set_normalization(true);
        let rms = (12.5_f64).sqrt();
        assert_eq!(discrepancy.normalization(), [1.0 / rms, 1.0 / rms, 1.0]);
    }

    #[test]
    fn replicates_are_summarized_before_the_loss() {
        let mut discrepancy = Discrepancy::new(LossConfig::SumSquaredError, &observed());
        discrepancy.set_summary(Some(ReplicateSummary::Mean));
        // Two replicates: means 3, 5 and 1 against 3, 4 and 0.
        let mut predictions = [2.0, 4.0, 0.0, 4.0, 6.0, 2.0];
        assert_eq!(discrepancy.of(&mut predictions), 2.0);
    }
}
