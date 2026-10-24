//! The SMC sampler: it starts from the prior and, step by step, chooses the
//! next level, reweights, resamples and moves the particles until the last
//! level.
//!
//! # A step
//!
//! 1. **Level.** The next level `ℓ'` is the one nearest the last whose
//!    conditional effective sample size (Zhou, Johansen and Aston 2016),
//!    `CESS(ℓ') / N = (Σ Wᵢ wᵢ)² / Σ Wᵢ wᵢ²` with the incremental weights
//!    `wᵢ = exp(log_kernel(vᵢ, ℓ') − log_kernel(vᵢ, ℓ))`, is at least
//!    `target_cess`, by bisection until the two ends are neighbouring
//!    numbers ([`next_level`]). With weights of 0 or 1, as in ABC, the
//!    share is the weight of the particles inside, so the next tolerance is
//!    the smallest kept value whose share reaches `target_cess`.
//! 2. **Reweighting.** `Wᵢ ∝ Wᵢ wᵢ`, from the kept values only; the log
//!    evidence adds `log Σ Wᵢ wᵢ`, the log of the mean incremental weight.
//! 3. **Resampling.** Systematic, when Kish's effective size falls below
//!    `resample_threshold · N`.
//! 4. **Moves.** The kernel fits the particles and moves each one of
//!    positive weight, in parallel, each drawing from its own stream. With
//!    `a` the share of the first move's proposals accepted, every particle
//!    moves `R = ⌈log c / log(1 − a)⌉` times, at least once (Drovandi and
//!    Pettitt 2011), so it moves at least once with probability `1 − c`,
//!    where `c` is `stay_probability`.
//!
//! The sampler stops at the last level, or, with a `min_acceptance`, when
//! the first move of a step accepts less: it then keeps the particles of
//! the step before, the last one whose moves completed.

use commol_calibration::Workspace;
use commol_random::Generator;
use rayon::prelude::*;

use super::TemperedTarget;
use super::kernel::Kernel;
use crate::error::MethodError;
use crate::statistics::{effective_size, normalized_weights};
use commol_core::check_open_unit;
use commol_random::Stream;

/// One step of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct SmcStep {
    /// The level the step moved to.
    pub level: f64,
    /// Kish's effective size of the weights after reweighting.
    pub effective_size: f64,
    /// Whether the step resampled.
    pub resampled: bool,
    /// The share of the proposals of the first move that were accepted.
    pub acceptance_rate: f64,
    /// How many times each particle moved.
    pub moves: usize,
}

/// Why a run stopped before its last level.
#[derive(Debug, Clone, PartialEq)]
pub enum Stop {
    /// The first move at `level` accepted `acceptance_rate`, below the
    /// minimum.
    LowAcceptance {
        /// The level of the step that was not kept.
        level: f64,
        /// Its acceptance rate.
        acceptance_rate: f64,
    },
    /// No level after `level` keeps the conditional effective sample size:
    /// the kept values do not separate the particles any more.
    NoNextLevel {
        /// The last level reached.
        level: f64,
    },
}

/// The particles of a run, its steps and its log evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct SmcRun {
    /// One row per particle.
    pub values: Vec<Vec<f64>>,
    /// The log weight of each particle, normalized; `−∞` for a particle of
    /// weight 0.
    pub log_weights: Vec<f64>,
    /// The value each particle keeps, or `None` outside the support.
    pub kept: Vec<Option<f64>>,
    /// Every step, in order.
    pub steps: Vec<SmcStep>,
    /// The log of the product of the mean incremental weights.
    pub log_evidence: f64,
    /// The number of model runs.
    pub evaluations: usize,
    /// Why the run stopped before the last level, if it did.
    pub stopped_at: Option<Stop>,
}

impl SmcRun {
    /// The weights, which sum to 1.
    pub fn weights(&self) -> Vec<f64> {
        normalized_weights(&self.log_weights)
    }
}

/// The state of a run between steps.
#[derive(Debug, Clone)]
pub struct SmcState {
    values: Vec<Vec<f64>>,
    kept: Vec<Option<f64>>,
    log_weights: Vec<f64>,
    streams: Vec<Generator>,
    main: Generator,
    level: f64,
    log_evidence: f64,
    evaluations: usize,
    steps: Vec<SmcStep>,
    stopped_at: Option<Stop>,
}

impl SmcState {
    /// The current level.
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Multiply the current level by `factor`: the same distribution read
    /// on a target whose levels are `factor` times as large.
    pub fn rescale_level(&mut self, factor: f64) {
        self.level *= factor;
    }

    /// The particles, their weights and kept values.
    pub fn particles(&self) -> (&[Vec<f64>], &[f64], &[Option<f64>]) {
        (&self.values, &self.log_weights, &self.kept)
    }

    /// The run so far.
    pub fn run(&self) -> SmcRun {
        SmcRun {
            values: self.values.clone(),
            log_weights: normalized_log_weights(&self.log_weights),
            kept: self.kept.clone(),
            steps: self.steps.clone(),
            log_evidence: self.log_evidence,
            evaluations: self.evaluations,
            stopped_at: self.stopped_at.clone(),
        }
    }
}

/// The sequential Monte Carlo sampler.
#[derive(Debug, Clone)]
pub struct SmcSampler<K: Kernel> {
    particles: usize,
    kernel: K,
    resample_threshold: f64,
    target_cess: f64,
    stay_probability: f64,
    min_acceptance: Option<f64>,
}

impl<K: Kernel> SmcSampler<K> {
    /// A sampler of `particles` particles moved by `kernel`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`], which names the setting, when `particles`
    /// is below 2, `resample_threshold` is not in `(0, 1]`, or `target_cess`
    /// or `stay_probability` is not strictly between 0 and 1.
    pub fn new(
        particles: usize,
        kernel: K,
        resample_threshold: f64,
        target_cess: f64,
        stay_probability: f64,
    ) -> Result<Self, MethodError> {
        if particles < 2 {
            return Err(MethodError::Invalid(format!(
                "particles must be at least 2, got {particles}"
            )));
        }
        if !(resample_threshold > 0.0 && resample_threshold <= 1.0) {
            return Err(MethodError::Invalid(format!(
                "resample_threshold must be in (0, 1], got {resample_threshold}"
            )));
        }
        check_open_unit("target_cess", target_cess).map_err(MethodError::Invalid)?;
        check_open_unit("stay_probability", stay_probability).map_err(MethodError::Invalid)?;
        Ok(Self {
            particles,
            kernel,
            resample_threshold,
            target_cess,
            stay_probability,
            min_acceptance: None,
        })
    }

    /// Stop when the first move of a step accepts less than
    /// `min_acceptance`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when `min_acceptance` is not strictly between
    /// 0 and 1.
    pub fn with_min_acceptance(mut self, min_acceptance: f64) -> Result<Self, MethodError> {
        check_open_unit("min_acceptance", min_acceptance).map_err(MethodError::Invalid)?;
        self.min_acceptance = Some(min_acceptance);
        Ok(self)
    }

    /// Run from the prior to the last level of `target`, with every random
    /// number drawn from `seed`.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when every draw of the prior is outside the
    /// support; [`MethodError::Failed`] when the model fails, the kernel
    /// cannot fit the particles, or a step accepts no proposal and no
    /// `min_acceptance` stops it.
    pub fn run<T: TemperedTarget>(&mut self, target: &T, seed: u64) -> Result<SmcRun, MethodError> {
        let mut state = self.start(target, seed)?;
        self.advance(&mut state, target, |_| {})?;
        Ok(state.run())
    }

    /// Draw the particles from the prior and run the model at each.
    ///
    /// # Errors
    ///
    /// As [`Self::run`].
    pub fn start<T: TemperedTarget>(&self, target: &T, seed: u64) -> Result<SmcState, MethodError> {
        let mut streams: Vec<Generator> = (0..self.particles)
            .map(|i| Generator::from_seed(Stream::SmcParticles.replicate(seed, i as u64)))
            .collect();
        let draws: Vec<(Vec<f64>, Option<f64>)> = streams
            .par_iter_mut()
            .map_init(Workspace::default, |workspace, stream| {
                let values = target.sample_prior(stream);
                let kept = target.evaluate(&values, stream, workspace)?;
                Ok((values, kept))
            })
            .collect::<Result<_, String>>()
            .map_err(MethodError::Failed)?;
        let (values, kept): (Vec<_>, Vec<_>) = draws.into_iter().unzip();
        let alive = kept.iter().filter(|k| k.is_some()).count();
        if alive == 0 {
            return Err(MethodError::Invalid(
                "Every draw of the prior is outside the support of the problem: each run is \
                 infeasible or violates a constraint"
                    .to_string(),
            ));
        }
        let log_weights: Vec<f64> = kept
            .iter()
            .map(|k| if k.is_some() { 0.0 } else { f64::NEG_INFINITY })
            .collect();
        let finite: Vec<f64> = kept.iter().flatten().copied().collect();
        Ok(SmcState {
            level: target.first_level(&finite),
            values,
            kept,
            log_weights,
            streams,
            main: Generator::from_seed(Stream::SmcResampling.seed(seed)),
            log_evidence: (alive as f64 / self.particles as f64).ln(),
            evaluations: self.particles,
            steps: Vec::new(),
            stopped_at: None,
        })
    }

    /// Run the steps of `state` until the last level of `target`, calling
    /// `after_step` after each completed step.
    ///
    /// # Errors
    ///
    /// As [`Self::run`].
    pub fn advance<T: TemperedTarget>(
        &mut self,
        state: &mut SmcState,
        target: &T,
        mut after_step: impl FnMut(&SmcState),
    ) -> Result<(), MethodError> {
        let last = target.last_level();
        while state.level != last {
            let before = state.clone();
            let level = next_level(
                target,
                &state.kept,
                &state.log_weights,
                state.level,
                self.target_cess,
            );
            if level == state.level {
                state.stopped_at = Some(Stop::NoNextLevel { level });
                return Ok(());
            }
            self.reweight(state, target, level);
            let effective_size = effective_size(&normalized_weights(&state.log_weights));
            let resampled = effective_size < self.resample_threshold * self.particles as f64;
            if resampled {
                self.resample(state);
            }
            let weights = normalized_weights(&state.log_weights);
            self.kernel
                .adapt(target.bounds(), &state.values, &weights)
                .map_err(MethodError::Failed)?;
            let acceptance_rate = self
                .move_particles(state, target, level, 1)
                .map_err(MethodError::Failed)?;
            if let Some(minimum) = self.min_acceptance
                && acceptance_rate < minimum
            {
                let evaluations = state.evaluations;
                *state = before;
                state.evaluations = evaluations;
                state.stopped_at = Some(Stop::LowAcceptance {
                    level,
                    acceptance_rate,
                });
                return Ok(());
            }
            if acceptance_rate == 0.0 {
                return Err(MethodError::Failed(format!(
                    "The moves at level {level} accept no proposal, so the particles cannot \
                     move; use more particles or a larger target_cess"
                )));
            }
            let moves = move_count(self.stay_probability, acceptance_rate);
            if moves > 1 {
                self.move_particles(state, target, level, moves - 1)
                    .map_err(MethodError::Failed)?;
            }
            state.level = level;
            state.steps.push(SmcStep {
                level,
                effective_size,
                resampled,
                acceptance_rate,
                moves,
            });
            after_step(state);
        }
        Ok(())
    }

    fn reweight<T: TemperedTarget>(&self, state: &mut SmcState, target: &T, level: f64) {
        let weights = normalized_weights(&state.log_weights);
        let mut mean = 0.0;
        for ((log_weight, kept), weight) in
            state.log_weights.iter_mut().zip(&state.kept).zip(&weights)
        {
            let increment = match kept {
                Some(value) if *weight > 0.0 => increment(target, *value, state.level, level),
                _ => f64::NEG_INFINITY,
            };
            mean += weight * increment.exp();
            *log_weight += increment;
        }
        state.log_evidence += mean.ln();
    }

    /// Systematic resampling with one uniform number of the main generator.
    fn resample(&self, state: &mut SmcState) {
        let weights = normalized_weights(&state.log_weights);
        let n = self.particles;
        let start = state.main.uniform() / n as f64;
        let mut indices = Vec::with_capacity(n);
        let mut cumulative = 0.0;
        let mut source = 0;
        for k in 0..n {
            let point = start + k as f64 / n as f64;
            while source + 1 < n && cumulative + weights[source] <= point {
                cumulative += weights[source];
                source += 1;
            }
            indices.push(source);
        }
        state.values = indices.iter().map(|&i| state.values[i].clone()).collect();
        state.kept = indices.iter().map(|&i| state.kept[i]).collect();
        state.log_weights = vec![0.0; n];
    }

    /// Move every particle of positive weight `moves` times at `level`;
    /// returns the share of accepted proposals of the first move.
    fn move_particles<T: TemperedTarget>(
        &self,
        state: &mut SmcState,
        target: &T,
        level: f64,
        moves: usize,
    ) -> Result<f64, String> {
        let kernel = &self.kernel;
        let outcomes: Vec<(bool, usize)> = state
            .values
            .par_iter_mut()
            .zip(state.kept.par_iter_mut())
            .zip(state.streams.par_iter_mut())
            .zip(state.log_weights.par_iter())
            .map_init(
                Workspace::default,
                |workspace, (((values, kept), stream), log_weight)| {
                    if *log_weight == f64::NEG_INFINITY {
                        return Ok((false, 0));
                    }
                    let mut first = false;
                    let mut evaluations = 0;
                    for index in 0..moves {
                        let (accepted, ran) = metropolis_move(
                            target, kernel, values, kept, stream, workspace, level,
                        )?;
                        evaluations += usize::from(ran);
                        if index == 0 {
                            first = accepted;
                        }
                    }
                    Ok((first, evaluations))
                },
            )
            .collect::<Result<_, String>>()?;
        let moved = state
            .log_weights
            .iter()
            .filter(|w| **w > f64::NEG_INFINITY)
            .count();
        let accepted = outcomes.iter().filter(|(accepted, _)| *accepted).count();
        state.evaluations += outcomes.iter().map(|(_, ran)| ran).sum::<usize>();
        Ok(accepted as f64 / moved as f64)
    }
}

/// One Metropolis–Hastings move of a particle: whether it was accepted and
/// whether the model ran.
fn metropolis_move<T: TemperedTarget, K: Kernel>(
    target: &T,
    kernel: &K,
    values: &mut Vec<f64>,
    kept: &mut Option<f64>,
    stream: &mut Generator,
    workspace: &mut Workspace,
    level: f64,
) -> Result<(bool, bool), String> {
    let proposal = kernel.propose(values, stream);
    let log_prior = target.log_prior(&proposal.values);
    if log_prior == f64::NEG_INFINITY || proposal.log_ratio == f64::NEG_INFINITY {
        return Ok((false, false));
    }
    let proposed = target.evaluate(&proposal.values, stream, workspace)?;
    let u = stream.open_uniform();
    let Some(value) = proposed else {
        return Ok((false, true));
    };
    let current =
        target.log_prior(values) + kept.map_or(f64::NEG_INFINITY, |v| target.log_kernel(v, level));
    let log_ratio = log_prior + target.log_kernel(value, level) - current + proposal.log_ratio;
    let accepted = u.ln() < log_ratio;
    if accepted {
        *values = proposal.values;
        *kept = Some(value);
    }
    Ok((accepted, true))
}

/// `log_kernel(value, to) − log_kernel(value, from)`, `−∞` when the kernel
/// at `to` is 0.
fn increment<T: TemperedTarget>(target: &T, value: f64, from: f64, to: f64) -> f64 {
    let at = target.log_kernel(value, to);
    if at == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        at - target.log_kernel(value, from)
    }
}

/// `⌈log c / log(1 − a)⌉`, at least 1.
fn move_count(stay_probability: f64, acceptance_rate: f64) -> usize {
    let count = (stay_probability.ln() / (1.0 - acceptance_rate).ln()).ceil();
    if count.is_finite() && count >= 1.0 {
        count as usize
    } else {
        1
    }
}

/// Log weights that sum to 1 on the weight scale.
fn normalized_log_weights(log_weights: &[f64]) -> Vec<f64> {
    normalized_weights(log_weights)
        .into_iter()
        .map(f64::ln)
        .collect()
}

/// The conditional effective sample size over `N` of a move of the
/// particles from `from` to `to`.
fn relative_cess<T: TemperedTarget>(
    target: &T,
    kept: &[Option<f64>],
    weights: &[f64],
    from: f64,
    to: f64,
) -> f64 {
    let increments: Vec<f64> = kept
        .iter()
        .zip(weights)
        .map(|(kept, &w)| match kept {
            Some(value) if w > 0.0 => increment(target, *value, from, to),
            _ => f64::NEG_INFINITY,
        })
        .collect();
    let max = increments.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return 0.0;
    }
    let (mut first, mut second) = (0.0, 0.0);
    for (increment, w) in increments.iter().zip(weights) {
        let scaled = (increment - max).exp();
        first += w * scaled;
        second += w * scaled * scaled;
    }
    first * first / second
}

/// The next level from `current`: the level nearest the last at which the
/// conditional effective sample size over `N` is at least `target_cess`, by
/// bisection until the two ends are neighbouring numbers. No model runs.
pub fn next_level<T: TemperedTarget>(
    target: &T,
    kept: &[Option<f64>],
    log_weights: &[f64],
    current: f64,
    target_cess: f64,
) -> f64 {
    let weights = normalized_weights(log_weights);
    let last = target.last_level();
    let cess = |level: f64| relative_cess(target, kept, &weights, current, level);
    if cess(last) >= target_cess {
        return last;
    }
    let (mut near, mut far) = (current, last);
    loop {
        let middle = near + (far - near) / 2.0;
        if middle == near || middle == far {
            return near;
        }
        if cess(middle) >= target_cess {
            near = middle;
        } else {
            far = middle;
        }
    }
}

/// ABC rejection: `draws` draws of the prior, of which the closest share
/// `quantile` is kept, as one choice of the level of [`next_level`] with
/// `target_cess = quantile` and no move. The draws outside the tolerance
/// have weight 0.
///
/// # Errors
///
/// [`MethodError::Invalid`] when `draws` is below 2 or `quantile` is not
/// strictly between 0 and 1; otherwise as [`SmcSampler::run`].
pub fn abc_rejection<T: TemperedTarget>(
    target: &T,
    draws: usize,
    quantile: f64,
    seed: u64,
) -> Result<SmcRun, MethodError> {
    check_open_unit("quantile", quantile).map_err(MethodError::Invalid)?;
    let sampler = SmcSampler::new(draws, super::RandomWalk::new(), 1.0, quantile, 0.5)
        .map_err(|error| MethodError::Invalid(error.to_string().replace("particles", "draws")))?;
    let mut state = sampler.start(target, seed)?;
    let level = next_level(
        target,
        &state.kept,
        &state.log_weights,
        state.level,
        quantile,
    );
    sampler.reweight(&mut state, target, level);
    let weights = normalized_weights(&state.log_weights);
    let accepted = weights.iter().filter(|w| **w > 0.0).count();
    state.steps.push(SmcStep {
        level,
        effective_size: effective_size(&weights),
        resampled: false,
        acceptance_rate: accepted as f64 / draws as f64,
        moves: 0,
    });
    state.level = level;
    Ok(state.run())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_particle_moves_at_least_once() {
        assert_eq!(move_count(0.01, 1.0), 1);
        assert_eq!(move_count(0.5, 0.5), 1);
        assert_eq!(move_count(0.01, 0.5), 7);
    }
}
