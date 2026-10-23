//! The discrepancy: how far predicted observations are from the observed
//! ones.
//!
//! It is a loss of the residuals or the negative log-likelihood of
//! observation models, one or the other. It holds the observed values, their
//! weights, the per-series normalization of a loss, and the summary of the
//! replicates of a stochastic engine. It never runs an engine.

use std::collections::HashMap;

use crate::feasibility::Violation;
use crate::likelihood::{Likelihood, ObservationModel, TermDerivatives, unnamed_noise};
use crate::types::{
    CalibrationParameter, CalibrationParameterType, LossConfig, ObservedDataPoint, ReplicateSummary,
};

/// What a problem compares predictions with data by: a loss, or one
/// observation model per series label.
#[derive(Debug, Clone, PartialEq)]
pub enum DiscrepancyConfig {
    /// A loss of the residuals.
    Loss(LossConfig),
    /// The negative log-likelihood of an observation model per series.
    Likelihood(Vec<(String, ObservationModel)>),
}

impl From<LossConfig> for DiscrepancyConfig {
    fn from(loss: LossConfig) -> Self {
        Self::Loss(loss)
    }
}

/// Why a discrepancy has no derivatives of its terms at a point.
#[derive(Debug, Clone, PartialEq)]
pub enum TermError {
    /// The discrepancy has no curvature there, in words.
    NoCurvature(String),
    /// The mean of this observation lies outside its observation model.
    MeanOutside(usize),
}

/// The measure of a discrepancy.
#[derive(Debug, Clone)]
enum Measure {
    Loss(LossConfig),
    Likelihood(Likelihood),
}

/// The loss, or the negative log-likelihood, of predicted observations
/// against the observed ones.
#[derive(Debug, Clone)]
pub struct Discrepancy {
    measure: Measure,
    observed: Vec<f64>,
    weights: Vec<f64>,
    labels: Vec<String>,
    normalized: bool,
    normalization: Vec<f64>,
    summary: Option<ReplicateSummary>,
}

impl Discrepancy {
    /// The discrepancy of `observed_data` under `config`, whose observation
    /// models name `parameters`, with no normalization and one run per
    /// evaluation.
    ///
    /// # Errors
    ///
    /// Returns a message when the observation models do not fit the
    /// observations or the parameters ([`Likelihood::new`]), or a loss comes
    /// with a noise parameter.
    pub fn new(
        config: DiscrepancyConfig,
        observed_data: &[ObservedDataPoint],
        parameters: &[CalibrationParameter],
    ) -> Result<Self, String> {
        let measure = match config {
            DiscrepancyConfig::Loss(loss) => {
                if let Some(noise) = parameters
                    .iter()
                    .find(|p| p.parameter_type == CalibrationParameterType::Noise)
                {
                    return Err(unnamed_noise(&noise.id));
                }
                Measure::Loss(loss)
            }
            DiscrepancyConfig::Likelihood(models) => {
                Measure::Likelihood(Likelihood::new(&models, observed_data, parameters)?)
            }
        };
        Ok(Self::of_measure(measure, observed_data))
    }

    fn of_measure(measure: Measure, observed_data: &[ObservedDataPoint]) -> Self {
        Self {
            measure,
            observed: observed_data.iter().map(|obs| obs.value).collect(),
            weights: observed_data.iter().map(|obs| obs.weight).collect(),
            labels: observed_data.iter().map(|obs| obs.label.clone()).collect(),
            normalized: false,
            normalization: vec![1.0; observed_data.len()],
            summary: None,
        }
    }

    /// The discrepancy of other observed values and weights, built as this
    /// one was built: the same loss or models, normalization setting and
    /// summary. The normalization factors come from the new values.
    ///
    /// # Errors
    ///
    /// Returns a message that names the observation when a new value lies
    /// outside the support of its observation model.
    pub(crate) fn with_data(&self, observed_data: &[ObservedDataPoint]) -> Result<Self, String> {
        let measure = match &self.measure {
            Measure::Loss(loss) => Measure::Loss(*loss),
            Measure::Likelihood(likelihood) => {
                Measure::Likelihood(likelihood.with_data(observed_data)?)
            }
        };
        let mut discrepancy = Self::of_measure(measure, observed_data);
        discrepancy.normalized = self.normalized;
        if self.normalized {
            discrepancy.normalization =
                observation_normalization(&discrepancy.labels, &discrepancy.observed);
        }
        discrepancy.summary = self.summary;
        Ok(discrepancy)
    }

    /// Enable or disable per-series normalization
    /// ([`observation_normalization`]).
    ///
    /// # Errors
    ///
    /// Returns a message when it is enabled with observation models, which
    /// set the scale of each series themselves.
    pub(crate) fn set_normalization(&mut self, enabled: bool) -> Result<(), String> {
        if enabled && self.likelihood().is_some() {
            return Err("normalize_observations needs a loss: the observation models set the                         scale of each series with their noise"
                .to_string());
        }
        self.normalized = enabled;
        self.normalization = if enabled {
            observation_normalization(&self.labels, &self.observed)
        } else {
            vec![1.0; self.observed.len()]
        };
        Ok(())
    }

    /// Compare the observations with this summary of the replicates; `None`
    /// compares them with the one run of a deterministic engine.
    ///
    /// # Errors
    ///
    /// Returns a message when a summary comes with observation models: a
    /// stochastic engine needs a loss.
    pub(crate) fn set_summary(&mut self, summary: Option<ReplicateSummary>) -> Result<(), String> {
        if summary.is_some() && self.likelihood().is_some() {
            return Err("A stochastic engine needs a loss: the likelihood of a summary of its                         replicates ignores the noise of the process, so observation models                         need a deterministic engine"
                .to_string());
        }
        self.summary = summary;
        Ok(())
    }

    /// The factor each residual is multiplied by: `1 / rms` of its series
    /// when normalization is enabled, otherwise 1.
    pub fn normalization(&self) -> &[f64] {
        &self.normalization
    }

    /// The loss, or `None` for observation models.
    pub fn loss(&self) -> Option<LossConfig> {
        match &self.measure {
            Measure::Loss(loss) => Some(*loss),
            Measure::Likelihood(_) => None,
        }
    }

    /// The likelihood of the observation models, or `None` for a loss.
    pub fn likelihood(&self) -> Option<&Likelihood> {
        match &self.measure {
            Measure::Loss(_) => None,
            Measure::Likelihood(likelihood) => Some(likelihood),
        }
    }

    /// The observed values, one per observation.
    pub fn observed(&self) -> &[f64] {
        &self.observed
    }

    /// The weight of each observation.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The discrepancy of the predictions of one run, one value per
    /// observation, at the parameter `values`, whatever the summary of the
    /// replicates: the loss with the weights and the normalization, or the
    /// negative log-likelihood.
    ///
    /// # Errors
    ///
    /// [`Violation::MeanOutsideModel`] when a prediction lies outside the
    /// means its observation model allows.
    pub fn of_run(&self, values: &[f64], predictions: &[f64]) -> Result<f64, Violation> {
        match &self.measure {
            Measure::Loss(loss) => Ok(loss.aggregate(
                self.observed
                    .iter()
                    .zip(predictions)
                    .zip(&self.weights)
                    .zip(&self.normalization)
                    .map(|(((observed, predicted), &weight), normalization)| {
                        ((observed - predicted) * normalization, weight)
                    }),
            )),
            Measure::Likelihood(likelihood) => likelihood
                .log_likelihood(values, predictions)
                .map(|log_likelihood| -log_likelihood),
        }
    }

    /// The discrepancy of `predictions`, which hold one value per
    /// observation and replicate, replicate after replicate, at the
    /// parameter `values`. The predictions are reordered.
    ///
    /// # Errors
    ///
    /// As [`Self::of_run`].
    pub fn of(&self, values: &[f64], predictions: &mut [f64]) -> Result<f64, Violation> {
        let n = self.observed.len();
        match (self.summary, &self.measure) {
            (Some(summary), Measure::Loss(loss)) => {
                let replicates = predictions.len() / n.max(1);
                let mut per_observation: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..replicates).map(|r| predictions[r * n + i]).collect())
                    .collect();
                Ok(loss.replicate_loss(
                    summary,
                    self.observed
                        .iter()
                        .zip(per_observation.iter_mut())
                        .zip(&self.weights)
                        .zip(&self.normalization)
                        .map(|(((&observed, values), &weight), &normalization)| {
                            (observed, values.as_mut_slice(), weight, normalization)
                        }),
                ))
            }
            _ => self.of_run(values, predictions),
        }
    }

    /// The derivatives of each observation's term of the discrepancy of one
    /// run, at the parameter `values` and the means `predictions`, with
    /// respect to its arguments ([`TermDerivatives`]).
    ///
    /// | Discrepancy | Gradient | Curvature |
    /// |---|---|---|
    /// | `SumSquaredError` | `−2 w n² r` | `2 w n²` |
    /// | `RootMeanSquaredError` | `−2 w n² r · c` | `2 w n² · c`, `c = 1 / (2 √(W · S))` |
    /// | Observation models | `−w` times the scores ([`crate::Noise::scores`]) | `w` times the Fisher information ([`crate::Noise::information`]) |
    ///
    /// `r = y − μ` is the residual, `n` the normalization of the observation,
    /// `S = Σ w (n r)²` and
    /// `W = Σ w`: the root mean squared error `√(S / W)` changes by `c` per
    /// unit of `S`, so its terms are those of the sum of squares times `c`.
    ///
    /// # Errors
    ///
    /// [`TermError::NoCurvature`] for the mean absolute error, and for the
    /// root mean squared error of a perfect fit; [`TermError::MeanOutside`]
    /// when a mean lies outside its observation model.
    pub fn term_derivatives(
        &self,
        values: &[f64],
        predictions: &[f64],
    ) -> Result<Vec<TermDerivatives>, TermError> {
        let loss = match &self.measure {
            Measure::Likelihood(likelihood) => {
                return likelihood
                    .term_derivatives(values, predictions)
                    .map_err(TermError::MeanOutside);
            }
            Measure::Loss(loss) => *loss,
        };
        let scale = match loss {
            LossConfig::SumSquaredError => 1.0,
            LossConfig::RootMeanSquaredError => {
                let (sum, weight_sum) = self.squares(predictions);
                if sum <= 0.0 || weight_sum <= 0.0 {
                    return Err(TermError::NoCurvature(
                        "loss_function=\"rmse\" has no derivative at a perfect fit, where \
                         every residual is 0"
                            .to_string(),
                    ));
                }
                0.5 / (weight_sum * sum).sqrt()
            }
            LossConfig::MeanAbsoluteError => {
                return Err(TermError::NoCurvature(
                    "loss_function=\"mae\" has no curvature: the sandwich of the absolute \
                     error needs a density of the residuals at 0 and a bandwidth. Use a \
                     bootstrap (BayesianBootstrap, ResidualBootstrap or WildBootstrap)"
                        .to_string(),
                ));
            }
        };
        Ok(self
            .observed
            .iter()
            .zip(predictions)
            .zip(&self.weights)
            .zip(&self.normalization)
            .map(|(((&y, &mean), &weight), &normalization)| {
                let factor = 2.0 * weight * normalization * normalization * scale;
                TermDerivatives {
                    parameters: Vec::new(),
                    gradient: vec![-factor * (y - mean)],
                    curvature: vec![factor],
                }
            })
            .collect())
    }

    /// `(Σ w (n r)², Σ w)` of the predictions of one run.
    fn squares(&self, predictions: &[f64]) -> (f64, f64) {
        self.observed
            .iter()
            .zip(predictions)
            .zip(&self.weights)
            .zip(&self.normalization)
            .fold((0.0, 0.0), |(sum, weights), (((y, mean), &w), n)| {
                let residual = (y - mean) * n;
                (sum + w * residual * residual, weights + w)
            })
    }

    /// The base of the penalty of an infeasible point, at least `1e10`: for
    /// a loss, `1000 · n · max_obs²`; for observation models, `1000 · Σ wᵢ tᵢ`,
    /// with `tᵢ` the cost of observation `i` at the narrowest noise the bounds
    /// allow and a mean `max(max_obs, 1)` from it.
    pub fn penalty_base(&self) -> f64 {
        let largest = self.observed.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = match &self.measure {
            Measure::Loss(_) => largest * largest * self.observed.len() as f64,
            Measure::Likelihood(likelihood) => likelihood.far_cost(largest.max(1.0)),
        };
        (scale * 1000.0).max(1e10)
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
            if rms > f64::EPSILON { rms } else { 1.0 }
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

    fn sse() -> Discrepancy {
        Discrepancy::new(LossConfig::SumSquaredError.into(), &observed(), &[]).unwrap()
    }

    #[test]
    fn each_series_is_scaled_by_the_inverse_of_its_rms() {
        let mut discrepancy = sse();
        discrepancy.set_normalization(true).unwrap();
        let rms = (12.5_f64).sqrt();
        assert_eq!(discrepancy.normalization(), [1.0 / rms, 1.0 / rms, 1.0]);
    }

    #[test]
    fn other_data_get_the_normalization_of_their_own_values() {
        let mut discrepancy = sse();
        discrepancy.set_normalization(true).unwrap();
        let mut other = observed();
        other[0].value = 6.0;
        other[1].value = 8.0;
        let rms = (50.0_f64).sqrt();
        assert_eq!(
            discrepancy.with_data(&other).unwrap().normalization(),
            [1.0 / rms, 1.0 / rms, 1.0]
        );
    }

    #[test]
    fn replicates_are_summarized_before_the_loss() {
        let mut discrepancy = sse();
        discrepancy
            .set_summary(Some(ReplicateSummary::Mean))
            .unwrap();
        // Two replicates: means 3, 5 and 1 against 3, 4 and 0.
        let mut predictions = [2.0, 4.0, 0.0, 4.0, 6.0, 2.0];
        assert_eq!(discrepancy.of(&[], &mut predictions), Ok(2.0));
    }
}
