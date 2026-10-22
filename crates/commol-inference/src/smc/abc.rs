//! Approximate Bayesian computation: a particle is inside the tolerance `ε`
//! when the distance between its run and the observations is at most `ε`.
//!
//! The level is the tolerance: the log kernel of a kept distance `d` at `ε`
//! is 0 when `d ≤ ε` and `−∞` otherwise. With a stochastic engine each
//! evaluation runs one fresh replicate, seeded from the particle's stream,
//! and the particle keeps the distance of its own simulation until a move
//! replaces it, so the sampler samples the ABC posterior; the problem's
//! replicates and their summary do not apply.

use commol_calibration::discrepancy::observation_normalization;
use commol_calibration::{CalibrationProblem, EvaluationError, LossConfig, Workspace};
use commol_core::SimulationEngine;
use commol_random::Generator;

use super::TemperedTarget;
use crate::error::MethodError;
use crate::priors::Priors;

/// The metric of a [`Distance::Series`] within each series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesMetric {
    /// The weighted root mean square of the scaled residuals.
    Rmse,
    /// The weighted mean absolute scaled residual.
    Mae,
}

impl SeriesMetric {
    /// The metric of a name: `"rmse"` or `"mae"`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`], which lists the names, when `name` is none
    /// of them.
    pub fn from_name(name: &str) -> Result<Self, MethodError> {
        match name {
            "rmse" => Ok(Self::Rmse),
            "mae" => Ok(Self::Mae),
            _ => Err(MethodError::Invalid(format!(
                "metric is '{name}'; it must be \"rmse\" or \"mae\""
            ))),
        }
    }

    fn loss(self) -> LossConfig {
        match self {
            Self::Rmse => LossConfig::RootMeanSquaredError,
            Self::Mae => LossConfig::MeanAbsoluteError,
        }
    }
}

/// How far a run is from the observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    /// The loss of the problem, with its weights and normalization, on one
    /// run.
    Loss,
    /// The metric of each series on its residuals divided by the root mean
    /// square of its observed values, then the largest over the series.
    Series(SeriesMetric),
}

/// ABC on a problem, with priors and a distance.
pub struct AbcTarget<'a, E: SimulationEngine> {
    problem: &'a CalibrationProblem<E>,
    priors: &'a Priors,
    distance: Distance,
    bounds: Vec<(f64, f64)>,
    stochastic: bool,
    /// The series of each observation, by index.
    series: Vec<usize>,
    count: usize,
    /// `1 / rms` of the observed values of the series of each observation.
    scale: Vec<f64>,
}

impl Distance {
    /// The distance of a series metric, or the loss without one.
    ///
    /// # Errors
    ///
    /// As [`SeriesMetric::from_name`].
    pub fn from_metric(metric: Option<&str>) -> Result<Self, MethodError> {
        metric.map_or(Ok(Self::Loss), |name| {
            SeriesMetric::from_name(name).map(Self::Series)
        })
    }
}

impl<'a, E: SimulationEngine> AbcTarget<'a, E> {
    /// ABC on `problem` with `priors` and `distance`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when the distance is the loss and the
    /// problem has observation models instead.
    pub fn new(
        problem: &'a CalibrationProblem<E>,
        priors: &'a Priors,
        distance: Distance,
    ) -> Result<Self, MethodError> {
        if distance == Distance::Loss && problem.discrepancy().loss().is_none() {
            return Err(MethodError::Invalid(
                "LossDistance() needs a loss; the problem has observation models. Use \
                 SeriesDistance(...), or BayesianPosterior(...) for their likelihood"
                    .to_string(),
            ));
        }
        let labels: Vec<String> = problem
            .observed_data()
            .iter()
            .map(|obs| obs.label.clone())
            .collect();
        let mut names: Vec<&str> = Vec::new();
        let series = labels
            .iter()
            .map(|label| match names.iter().position(|name| name == label) {
                Some(index) => index,
                None => {
                    names.push(label);
                    names.len() - 1
                }
            })
            .collect();
        let observed = problem.discrepancy().observed().to_vec();
        Ok(Self {
            problem,
            priors,
            distance,
            bounds: problem.parameter_bounds(),
            stochastic: problem.forward().replicates().is_some(),
            series,
            count: names.len(),
            scale: observation_normalization(&labels, &observed),
        })
    }

    /// Whether each evaluation runs one fresh replicate of a stochastic
    /// engine.
    pub fn stochastic(&self) -> bool {
        self.stochastic
    }

    /// The distance of the predictions of one run at the parameter
    /// `values`.
    pub fn distance(&self, values: &[f64], predictions: &[f64]) -> f64 {
        let discrepancy = self.problem.discrepancy();
        match self.distance {
            Distance::Loss => discrepancy
                .of_run(values, predictions)
                .expect("the loss of a run has no violation"),
            Distance::Series(metric) => {
                let loss = metric.loss();
                let mut sums = vec![(0.0, 0.0); self.count];
                for (i, &predicted) in predictions.iter().enumerate() {
                    let residual = (discrepancy.observed()[i] - predicted) * self.scale[i];
                    let weight = discrepancy.weights()[i];
                    let sum = &mut sums[self.series[i]];
                    sum.0 += loss.term(residual, weight);
                    sum.1 += weight;
                }
                sums.into_iter()
                    .map(|(sum, weight)| loss.finish(sum, weight))
                    .fold(0.0, f64::max)
            }
        }
    }
}

impl<E: SimulationEngine + Sync> TemperedTarget for AbcTarget<'_, E> {
    fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    fn log_prior(&self, values: &[f64]) -> f64 {
        self.priors.log_density(values)
    }

    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64> {
        self.priors.sample(stream)
    }

    fn evaluate(
        &self,
        values: &[f64],
        stream: &mut Generator,
        workspace: &mut Workspace,
    ) -> Result<Option<f64>, String> {
        let seed = self.stochastic.then(|| stream.next_u64());
        if self.problem.feasibility().check_bounds(values).is_err() {
            return Ok(None);
        }
        match self.problem.predict_once(values, seed, workspace) {
            Ok(()) => {
                let distance = self.distance(values, &workspace.predictions);
                Ok(distance.is_finite().then_some(distance))
            }
            Err(EvaluationError::Violation(_)) => Ok(None),
            Err(EvaluationError::Failed(message)) => Err(message),
        }
    }

    fn log_kernel(&self, value: f64, level: f64) -> f64 {
        if value <= level {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    fn first_level(&self, kept: &[f64]) -> f64 {
        kept.iter().copied().fold(0.0, f64::max)
    }

    fn last_level(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_distance_is_the_loss_or_a_metric_of_the_series() {
        assert_eq!(Distance::from_metric(None), Ok(Distance::Loss));
        assert_eq!(
            Distance::from_metric(Some("mae")),
            Ok(Distance::Series(SeriesMetric::Mae))
        );
        assert_eq!(
            Distance::from_metric(Some("max")),
            Err(MethodError::Invalid(
                "metric is 'max'; it must be \"rmse\" or \"mae\"".to_string()
            ))
        );
    }
}
