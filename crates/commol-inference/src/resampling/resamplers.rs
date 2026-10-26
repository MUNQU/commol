//! The resamplers: each draws the data of one replicate, as new weights or
//! new observed values.

use commol_calibration::{CalibrationProblem, ObservedDataPoint, Workspace};
use commol_core::SimulationEngine;
use commol_random::Generator;
use commol_random::sample::exponential;

use crate::error::MethodError;
use crate::predictive::{DistributionNoise, ObservationNoise};
use crate::resampling::blocks::BlockScheme;
use crate::resampling::residuals::{ResidualBlocks, Residuals, SeriesResiduals, source_positions};

/// What a replicate changes in the data of the problem: one weight or one
/// observed value per observation, in problem order.
#[derive(Debug, Clone, PartialEq)]
pub enum Perturbation {
    /// New weights; the observed values stay.
    Weights(Vec<f64>),
    /// New observed values; the weights stay.
    Values(Vec<f64>),
}

/// The port of a resampling method: the data of one replicate.
pub trait Resampler: Sync {
    /// The perturbation of one replicate, drawn with `generator`.
    fn replicate(&self, generator: &mut Generator) -> Perturbation;
}

/// The Bayesian (weighted) bootstrap: one Dirichlet(1, …, 1) weight per
/// block, scaled to mean 1, multiplies the weight of every observation of
/// the block.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianBootstrap {
    weights: Vec<f64>,
    block_of: Vec<usize>,
    blocks: usize,
}

impl BayesianBootstrap {
    /// The bootstrap of `observations` with `blocks`.
    pub fn new(observations: &[ObservedDataPoint], blocks: &BlockScheme) -> Self {
        Self {
            weights: observations.iter().map(|obs| obs.weight).collect(),
            block_of: blocks.block_of(),
            blocks: blocks.count(),
        }
    }
}

impl BayesianBootstrap {
    /// The weights of one replicate, drawn with `generator`.
    pub fn draw_weights(&self, generator: &mut Generator) -> Vec<f64> {
        // Normalized Exp(1) draws are a Dirichlet(1, …, 1) draw.
        let draws: Vec<f64> = (0..self.blocks)
            .map(|_| exponential(generator, 1.0))
            .collect();
        let scale = self.blocks as f64 / draws.iter().sum::<f64>();
        self.weights
            .iter()
            .zip(&self.block_of)
            .map(|(weight, &block)| weight * draws[block] * scale)
            .collect()
    }
}

impl Resampler for BayesianBootstrap {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        Perturbation::Weights(self.draw_weights(generator))
    }
}

/// The parametric bootstrap: each replicate draws every observed value from
/// the observation model of its series at the fit, around the fit's
/// predicted mean and with the fit's noise parameters.
#[derive(Debug, Clone)]
pub struct ParametricResampler {
    noise: DistributionNoise,
    observations: Vec<ObservedDataPoint>,
    fit: Vec<f64>,
    predicted: Vec<f64>,
}

impl ParametricResampler {
    /// The bootstrap of the observation models of `problem` at `fit`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when the problem has a loss instead of
    /// observation models; as [`MethodError::of_point`] when the fit is not
    /// feasible or has no run.
    pub fn new<E: SimulationEngine>(
        problem: &CalibrationProblem<E>,
        fit: &[f64],
    ) -> Result<Self, MethodError> {
        let noise = DistributionNoise::new(problem).map_err(|_| {
            MethodError::Invalid(
                "ParametricBootstrap(...) needs observation models; the problem has a loss. \
                 Use ResidualBootstrap(...) or WildBootstrap(...)"
                    .to_string(),
            )
        })?;
        let mut workspace = Workspace::default();
        problem
            .evaluate(fit, &mut workspace)
            .map_err(|error| MethodError::of_point("The fit has no likelihood", error))?;
        Ok(Self {
            noise,
            observations: problem.observed_data().to_vec(),
            fit: fit.to_vec(),
            predicted: workspace.predictions,
        })
    }
}

impl Resampler for ParametricResampler {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        let mut values = vec![0.0; self.predicted.len()];
        self.noise.observe(
            &self.observations,
            &self.fit,
            &self.predicted,
            generator,
            &mut values,
        );
        Perturbation::Values(values)
    }
}

/// The residual block bootstrap: a synthetic series joins blocks of
/// residuals of the fit, drawn with replacement across every series at
/// once, and adds them to the fitted values on their scale.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualBootstrap {
    residuals: Residuals,
    grid: SeriesResiduals,
    series_of: Vec<usize>,
    position_of: Vec<usize>,
    length: usize,
    blocks: ResidualBlocks,
}

impl ResidualBootstrap {
    /// The bootstrap of the residuals of `observations`, with the positions
    /// of `scheme` and blocks of mean `scheme.length()`.
    pub fn new(
        observations: &[ObservedDataPoint],
        residuals: Residuals,
        scheme: &BlockScheme,
        blocks: ResidualBlocks,
    ) -> Self {
        let grid = SeriesResiduals::new(observations, residuals.residuals());
        let series_of = observations
            .iter()
            .map(|obs| grid.series(&obs.label).expect("every series has residuals"))
            .collect();
        Self {
            residuals,
            grid,
            series_of,
            position_of: scheme.position_of().to_vec(),
            length: scheme.length(),
            blocks,
        }
    }
}

impl Resampler for ResidualBootstrap {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        let positions = self.grid.positions();
        let source = source_positions(positions, self.length, positions, self.blocks, generator);
        let residuals: Vec<f64> = self
            .series_of
            .iter()
            .zip(&self.position_of)
            .map(|(&series, &position)| self.grid.at(series, source[position]))
            .collect();
        Perturbation::Values(self.residuals.values(&residuals))
    }
}

/// The distribution of the multiplier of a wild bootstrap: mean 0 and
/// variance 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Multipliers {
    /// `±1` with probability 1/2 each.
    Rademacher,
    /// `−(√5 − 1)/2` with probability `(√5 + 1)/(2√5)`, otherwise
    /// `(√5 + 1)/2` (Mammen 1993): third moment 1.
    Mammen,
}

impl Multipliers {
    /// One multiplier.
    pub fn draw(self, generator: &mut Generator) -> f64 {
        let u = generator.uniform();
        match self {
            Self::Rademacher => {
                if u < 0.5 {
                    -1.0
                } else {
                    1.0
                }
            }
            Self::Mammen => {
                let root5 = 5f64.sqrt();
                if u < (root5 + 1.0) / (2.0 * root5) {
                    -(root5 - 1.0) / 2.0
                } else {
                    (root5 + 1.0) / 2.0
                }
            }
        }
    }
}

/// The wild block bootstrap: the residuals of each block are multiplied by
/// one multiplier and applied to the fitted values on their scale, so every
/// residual stays at its own position.
#[derive(Debug, Clone, PartialEq)]
pub struct WildBootstrap {
    residuals: Residuals,
    block_of: Vec<usize>,
    blocks: usize,
    multipliers: Multipliers,
}

impl WildBootstrap {
    /// The bootstrap of `residuals` with the blocks of `scheme`.
    pub fn new(residuals: Residuals, scheme: &BlockScheme, multipliers: Multipliers) -> Self {
        Self {
            residuals,
            block_of: scheme.block_of(),
            blocks: scheme.count(),
            multipliers,
        }
    }
}

impl Resampler for WildBootstrap {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        let factors: Vec<f64> = (0..self.blocks)
            .map(|_| self.multipliers.draw(generator))
            .collect();
        let residuals: Vec<f64> = self
            .residuals
            .residuals()
            .iter()
            .zip(&self.block_of)
            .map(|(residual, &block)| residual * factors[block])
            .collect();
        Perturbation::Values(self.residuals.values(&residuals))
    }
}
