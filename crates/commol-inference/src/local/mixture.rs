//! The Laplace mixture: a Laplace approximation at each distinct optimum of a
//! multi-start fit, weighted by its Laplace evidence, as the proposal of an
//! importance sampler of a posterior.
//!
//! The ends of the starts are taken in order of their objective, and an end
//! closer than one Mahalanobis unit to a kept component, under that
//! component's covariance, merges into it. The component at `θ̂_k` is
//! `N(û_k, (s·H_u)⁻¹)` on the unbounded scale, with `s` the curvature of the
//! target's log density per unit of the objective (1 for a likelihood, `η`
//! for a Gibbs posterior), and its log evidence is
//! `log p(θ̂_k) + log |dθ/du| + (d/2) log 2π + ½ log |Σ_k|`. Every draw is
//! weighted by the target over the mixture and smoothed by PSIS. An end on a
//! bound gives no component; when it fits at least as well as the worst
//! component, the sample misses an optimum and is unreliable.

use commol_calibration::{CalibrationProblem, Workspace};
use commol_core::SimulationEngine;
use commol_random::Generator;

use super::approximation::{CurvatureKind, curvature_at};
use super::curvature::on_bound;
use super::gaussian::{Gaussian, inverse};
use crate::distribution::open_uniform;
use crate::error::MethodError;
use crate::importance::{ImportanceRun, Proposal, importance_sample};
use crate::psis::{RELIABLE_K_HAT, log_sum_exp};
use crate::target::{Target, discrepancy_in_support};
use crate::transform::Transform;

/// Ends closer than this Mahalanobis distance to a kept component merge into
/// it.
const MERGE_DISTANCE: f64 = 1.0;

/// One component of a Laplace mixture.
#[derive(Debug, Clone)]
pub struct Component {
    /// The optimum on the scale of the parameters.
    pub center: Vec<f64>,
    /// The normal on the unbounded scale.
    pub gaussian: Gaussian,
    /// The log of its Laplace evidence.
    pub log_evidence: f64,
    /// Its weight in the mixture: its evidence over the sum.
    pub weight: f64,
}

/// The mixture of the components, as a proposal.
struct Mixture<'a> {
    components: &'a [Component],
    log_weights: Vec<f64>,
}

impl Proposal for Mixture<'_> {
    fn draw(&self, generator: &mut Generator) -> Vec<f64> {
        let point = open_uniform(generator);
        let mut cumulative = 0.0;
        let index = self
            .components
            .iter()
            .position(|component| {
                cumulative += component.weight;
                point < cumulative
            })
            .unwrap_or(self.components.len() - 1);
        self.components[index].gaussian.draw_unbounded(generator)
    }

    fn log_density(&self, unbounded: &[f64]) -> f64 {
        let terms: Vec<f64> = self
            .components
            .iter()
            .zip(&self.log_weights)
            .map(|(component, log_weight)| log_weight + component.gaussian.log_density(unbounded))
            .collect();
        log_sum_exp(&terms)
    }
}

/// A Laplace mixture and its importance sample.
#[derive(Debug, Clone)]
pub struct MixtureRun {
    /// The components, in order of their objective.
    pub components: Vec<Component>,
    /// Whether the sample is reliable: `k̂ ≤ 0.7`, and no end on a bound
    /// fits at least as well as the worst component.
    pub reliable: bool,
    /// The draws and their smoothed weights.
    pub sample: ImportanceRun,
    /// What the reader should know: ends that gave no component, and an
    /// unreliable sample.
    pub notes: Vec<String>,
}

/// The Laplace mixture at the optima `ends` of `problem` as the proposal of
/// `draws` draws of `target`, whose log density has the curvature
/// `curvature_scale` per unit of the objective.
///
/// # Errors
///
/// [`MethodError::Invalid`] when no end gives a component, or the curvature
/// does not apply (a stochastic engine, the mean absolute error);
/// [`MethodError::Failed`] when a run, the derivatives or the importance
/// sampler fail.
#[allow(clippy::too_many_arguments)]
pub fn laplace_mixture<E, T>(
    problem: &CalibrationProblem<E>,
    target: &T,
    curvature_scale: f64,
    ends: &[Vec<f64>],
    draws: usize,
    kind: CurvatureKind,
    seed: u64,
) -> Result<MixtureRun, MethodError>
where
    E: SimulationEngine + Send + Sync,
    T: Target + Sync,
{
    let names = problem.parameter_names();
    let transform = Transform::new(&problem.parameter_bounds());
    let mut workspace = Workspace::default();
    let mut notes = Vec::new();
    let mut scored = Vec::new();
    for end in ends {
        match discrepancy_in_support(problem, end, &mut workspace).map_err(MethodError::Failed)? {
            Some(objective) => scored.push((objective, end)),
            None => notes.push(format!("The end {end:?} is outside the support")),
        }
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut components: Vec<Component> = Vec::new();
    // The objective of each kept component, and of each end on a bound.
    let mut kept = Vec::new();
    let mut on_bounds = Vec::new();
    for (objective, end) in scored {
        let unbounded = transform.to_unbounded(end);
        let merged = components.iter().any(|component| {
            let squared: f64 = component
                .gaussian
                .standardized(&unbounded)
                .iter()
                .map(|z| z * z)
                .sum();
            squared.sqrt() < MERGE_DISTANCE
        });
        if merged {
            continue;
        }
        if let Some((j, bound)) = on_bound(&transform, end) {
            on_bounds.push((objective, end));
            notes.push(format!(
                "The end {end:?} gives no component: '{}' is on its bound {bound}",
                names[j]
            ));
            continue;
        }
        let curvature = curvature_at(problem, end, kind)?;
        let matrix = curvature.matrix() * curvature_scale;
        let covariance = match inverse(&matrix) {
            Ok(covariance) => covariance,
            Err(error) => {
                notes.push(format!("The end {end:?} gives no component: {error}"));
                continue;
            }
        };
        let gaussian = Gaussian::new(transform.clone(), unbounded, covariance, None)?;
        let log_target = target
            .log_density(end, &mut workspace)
            .map_err(MethodError::Failed)?;
        let d = end.len() as f64;
        let log_determinant: f64 = gaussian
            .factor()
            .diagonal()
            .iter()
            .map(|l| 2.0 * l.ln())
            .sum();
        kept.push(objective);
        components.push(Component {
            center: end.clone(),
            log_evidence: log_target
                + transform.log_jacobian(end)
                + 0.5 * d * (2.0 * std::f64::consts::PI).ln()
                + 0.5 * log_determinant,
            gaussian,
            weight: 0.0,
        });
    }
    if components.is_empty() {
        return Err(MethodError::Invalid(format!(
            "No end of the starts gives a component of the mixture: {}",
            notes.join("; ")
        )));
    }
    let evidences: Vec<f64> = components.iter().map(|c| c.log_evidence).collect();
    let total = log_sum_exp(&evidences);
    if !total.is_finite() {
        return Err(MethodError::Invalid(
            "Every optimum of the starts has a target density of 0".to_string(),
        ));
    }
    for component in &mut components {
        component.weight = (component.log_evidence - total).exp();
    }
    let mixture = Mixture {
        log_weights: components.iter().map(|c| c.log_evidence - total).collect(),
        components: &components,
    };
    let sample = importance_sample(target, &mixture, &transform, draws, seed)
        .map_err(MethodError::Failed)?;
    if !sample.reliable {
        notes.push(format!(
            "The Pareto shape of the importance weights is k̂ = {:.2}, above {RELIABLE_K_HAT}: \
             the mixture misses mass of the target, and the sample is unreliable",
            sample.k_hat
        ));
    }
    // An end on a bound that fits at least as well as a kept component is
    // an optimum the mixture leaves out: its mass is missing from the
    // sample, which k̂ cannot see.
    let worst = kept.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let missed: Vec<String> = on_bounds
        .iter()
        .filter(|(objective, _)| *objective <= worst)
        .map(|(objective, end)| format!("{end:?} (objective {objective})"))
        .collect();
    if !missed.is_empty() {
        notes.push(format!(
            "The mixture misses an optimum on a bound that fits at least as well as its \
             worst component (objective {worst}): {}; the sample is unreliable",
            missed.join(", ")
        ));
    }
    Ok(MixtureRun {
        reliable: sample.reliable && missed.is_empty(),
        components,
        sample,
        notes,
    })
}
