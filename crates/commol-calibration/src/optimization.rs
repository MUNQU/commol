//! Optimizers of a calibration problem: Nelder-Mead and particle swarm.
//!
//! Each optimizer minimizes the [`OptimizationObjective`] of a problem from a
//! start point. Nothing here writes to standard output or standard error: a
//! caller that wants progress passes an observer
//! ([`optimize_with_observer`]).

use std::cell::RefCell;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{
    CostFunction, Error, Executor, IterState, PopulationState, TerminationReason, TerminationStatus,
};
use argmin::solver::neldermead::NelderMead;
use argmin::solver::particleswarm::{
    InitializationStrategy as ArgminInitializationStrategy,
    MutationApplication as ArgminMutationApplication, MutationStrategy as ArgminMutationStrategy,
    Particle, ParticleSwarm,
};
use commol_core::SimulationEngine;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::calibration_problem::CalibrationProblem;
use crate::objective::{OptimizationObjective, Recorded};
use crate::types::{CalibrationEvaluation, CalibrationResult, CalibrationResultWithHistory};

/// Configuration for Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations
    pub max_iterations: u64,

    /// Sample standard deviation tolerance (convergence criterion)
    /// Must be non-negative, defaults to EPSILON
    pub sd_tolerance: f64,

    /// Size of the initial simplex, as a fraction of each parameter's bound
    /// width. Each vertex moves one parameter by this fraction from the start
    /// value (towards the farther bound when the step would leave the bounds).
    /// Must be in (0, 1]; defaults to 0.1.
    pub initial_simplex_size: f64,

    /// Reflection parameter (alpha)
    /// Must be > 0, defaults to 1.0
    pub alpha: Option<f64>,

    /// Expansion parameter (gamma)
    /// Must be > 1, defaults to 2.0
    pub gamma: Option<f64>,

    /// Contraction parameter (rho)
    /// Must be in (0, 0.5], defaults to 0.5
    pub rho: Option<f64>,

    /// Shrinking parameter (sigma)
    /// Must be in (0, 1], defaults to 0.5
    pub sigma: Option<f64>,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            sd_tolerance: 1e-6,
            initial_simplex_size: 0.1,
            alpha: None, // Use argmin's default: 1.0
            gamma: None, // Use argmin's default: 2.0
            rho: None,   // Use argmin's default: 0.5
            sigma: None, // Use argmin's default: 0.5
        }
    }
}

/// Inertia weight strategy for PSO
#[derive(Debug, Clone)]
pub enum InertiaWeightStrategy {
    /// Constant inertia weight (default: 1/(2*ln(2)) ≈ 0.721)
    Constant(f64),
    /// Chaotic inertia weight using logistic map
    Chaotic {
        /// Lowest inertia weight.
        w_min: f64,
        /// Highest inertia weight; greater than `w_min`.
        w_max: f64,
    },
}

/// Acceleration coefficient strategy for PSO
#[derive(Debug, Clone)]
pub enum AccelerationStrategy {
    /// Constant coefficients (default: cognitive=0.5+ln(2), social=0.5+ln(2))
    Constant {
        /// Pull towards each particle's best position (c1).
        cognitive: f64,
        /// Pull towards the swarm's best position (c2).
        social: f64,
    },
    /// Time-Varying Acceleration Coefficients (TVAC)
    TimeVarying {
        /// Cognitive coefficient at the first iteration.
        c1_initial: f64,
        /// Cognitive coefficient at the last iteration.
        c1_final: f64,
        /// Social coefficient at the first iteration.
        c2_initial: f64,
        /// Social coefficient at the last iteration.
        c2_final: f64,
    },
}

/// Particle initialization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializationStrategy {
    /// Standard uniform random initialization (default)
    UniformRandom,
    /// Latin Hypercube Sampling - ensures uniform distribution across search space
    LatinHypercube,
    /// Opposition-Based Learning - generates both random and opposite positions, selects best
    OppositionBased,
}

/// Mutation strategy for avoiding local optima
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MutationStrategy {
    /// No mutation (default)
    None,
    /// Gaussian mutation with specified standard deviation
    Gaussian(f64),
    /// Cauchy mutation with specified scale parameter
    Cauchy(f64),
}

/// Determines which particles to apply mutation to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationApplication {
    /// No mutation application (default)
    None,
    /// Apply mutation only to the global best particle (most efficient)
    GlobalBestOnly,
    /// Apply mutation to all particles (maximum diversity)
    AllParticles,
    /// Apply mutation only to particles with below-average fitness (balanced approach)
    BelowAverage,
}

/// Configuration for Particle Swarm Optimization
#[derive(Debug, Clone)]
pub struct ParticleSwarmConfig {
    /// Number of particles in the swarm
    pub num_particles: usize,

    /// Maximum number of iterations
    pub max_iterations: u64,

    /// Target cost (convergence criterion)
    /// Optimization stops when cost reaches this value
    pub target_cost: Option<f64>,

    /// Inertia weight strategy (constant or chaotic)
    /// Defaults to Constant(1/(2*ln(2)) ≈ 0.721)
    pub inertia_strategy: Option<InertiaWeightStrategy>,

    /// Acceleration coefficient strategy (constant or time-varying)
    /// Defaults to Constant with cognitive=0.5+ln(2), social=0.5+ln(2)
    pub acceleration_strategy: Option<AccelerationStrategy>,

    /// Particle initialization strategy
    /// Defaults to UniformRandom
    pub initialization_strategy: InitializationStrategy,

    /// Velocity clamping factor (as fraction of search space range, None = disabled)
    /// Typical values: 0.1 to 0.2
    pub velocity_clamp_factor: Option<f64>,

    /// Threshold below which velocity is considered near-zero and should be reinitialized
    /// None = disabled. Typical values: 0.001 to 0.01
    pub velocity_mutation_threshold: Option<f64>,

    /// Mutation strategy to help escape local optima
    pub mutation_strategy: MutationStrategy,

    /// Mutation probability (applied per iteration, 0.0 to 1.0)
    pub mutation_probability: f64,

    /// Which particles to apply mutation to
    pub mutation_application: MutationApplication,

    /// Random seed for reproducibility (None = use system entropy)
    pub seed: Option<u64>,
}

impl Default for ParticleSwarmConfig {
    fn default() -> Self {
        Self {
            num_particles: 20,
            max_iterations: 1000,
            target_cost: None,
            inertia_strategy: None,      // Use argmin's default
            acceleration_strategy: None, // Use argmin's default
            initialization_strategy: InitializationStrategy::UniformRandom,
            velocity_clamp_factor: None,
            velocity_mutation_threshold: None,
            mutation_strategy: MutationStrategy::None,
            mutation_probability: 0.0,
            mutation_application: MutationApplication::None,
            seed: None,
        }
    }
}

/// Algorithm-specific optimization configuration
#[derive(Debug, Clone)]
pub enum OptimizationConfig {
    /// Nelder-Mead simplex method (gradient-free)
    /// Good for: 2-10 parameters, non-smooth objectives
    /// Most reliable for compartment models
    NelderMead(NelderMeadConfig),

    /// Particle Swarm Optimization (gradient-free, global search)
    /// Good for: Multiple local minima, parallelizable
    /// Use when you suspect multiple local optima
    ParticleSwarm(ParticleSwarmConfig),
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig::NelderMead(NelderMeadConfig::default())
    }
}

/// Minimize the objective of `problem` from its initial guess.
///
/// # Errors
///
/// Returns a message when the configuration is not valid, or the engine or
/// a constraint cannot be evaluated.
pub fn optimize<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
) -> Result<CalibrationResult, String> {
    optimize_from(problem, config, &problem.initial_parameters(), None)
}

/// Minimize the objective of `problem` from `start`.
///
/// Nelder-Mead builds its simplex around `start`. A particle swarm draws its
/// population over the bounds; `seed`, when given, replaces the seed of its
/// configuration.
///
/// # Errors
///
/// As [`optimize`].
pub fn optimize_from<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
    start: &[f64],
    seed: Option<u64>,
) -> Result<CalibrationResult, String> {
    let objective = OptimizationObjective::new(problem);
    run(problem, objective, config, start, seed, None::<NoObserver>).map(|(result, _)| result)
}

/// Minimize the objective of `problem` from its initial guess, with an
/// observer that sees every iteration.
///
/// # Errors
///
/// As [`optimize`].
pub fn optimize_with_observer<E, O>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
    observer: O,
) -> Result<CalibrationResult, String>
where
    E: SimulationEngine,
    O: Observe<NelderMeadState> + Observe<ParticleSwarmState> + 'static,
{
    let objective = OptimizationObjective::new(problem);
    let start = problem.initial_parameters();
    run(problem, objective, config, &start, None, Some(observer)).map(|(result, _)| result)
}

/// Minimize the objective of `problem` from its initial guess, and keep every
/// evaluation of a feasible point.
///
/// # Errors
///
/// As [`optimize`].
pub fn optimize_with_history<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
) -> Result<CalibrationResultWithHistory, String> {
    let recorded = Recorded {
        objective: OptimizationObjective::new(problem),
        evaluations: RefCell::new(Vec::new()),
    };
    let start = problem.initial_parameters();
    let (result, recorded) = run(problem, recorded, config, &start, None, None::<NoObserver>)?;
    let evaluations: Vec<CalibrationEvaluation> = recorded
        .map(|recorded| recorded.evaluations.into_inner())
        .unwrap_or_default();
    Ok(CalibrationResultWithHistory {
        result,
        evaluations,
    })
}

/// Run the optimizer of `config` on `cost` from `start`, and return the
/// result and the cost function.
fn run<E, C, O>(
    problem: &CalibrationProblem<E>,
    cost: C,
    config: &OptimizationConfig,
    start: &[f64],
    seed: Option<u64>,
    observer: Option<O>,
) -> Result<(CalibrationResult, Option<C>), String>
where
    E: SimulationEngine,
    C: CostFunction<Param = Vec<f64>, Output = f64>,
    O: Observe<NelderMeadState> + Observe<ParticleSwarmState> + 'static,
{
    if start.len() != problem.num_parameters() {
        return Err(format!(
            "Expected {} start values, got {}",
            problem.num_parameters(),
            start.len()
        ));
    }
    match config {
        OptimizationConfig::NelderMead(nm_config) => {
            optimize_nelder_mead(problem, cost, start, nm_config, observer)
        }
        OptimizationConfig::ParticleSwarm(ps_config) => {
            let mut ps_config = ps_config.clone();
            if seed.is_some() {
                ps_config.seed = seed;
            }
            optimize_particle_swarm(problem, cost, start, &ps_config, observer)
        }
    }
}

/// State of a Nelder-Mead run.
pub type NelderMeadState = IterState<Vec<f64>, (), (), (), (), f64>;

/// State of a Particle Swarm run.
pub type ParticleSwarmState = PopulationState<Particle<Vec<f64>, f64>, f64>;

/// An observer that observes nothing, for runs without one.
pub struct NoObserver;

impl<I> Observe<I> for NoObserver {}

/// Whether a run converged: the solver met its convergence criterion or the
/// target cost. A run that stops at `max_iterations` did not converge.
pub fn converged(status: &TerminationStatus) -> bool {
    matches!(
        status,
        TerminationStatus::Terminated(
            TerminationReason::SolverConverged | TerminationReason::TargetCostReached
        )
    )
}

/// Maps an unbounded coordinate `u` to a value `x` inside its bounds:
/// `x = lo + (hi − lo)·(sin u + 1)/2` (the MINUIT and lmfit transform).
///
/// Nelder-Mead searches `u`, so every point it tries is inside the bounds and
/// the simplex never collapses on a bound where the clamped loss is flat.
#[derive(Debug, Clone)]
pub(crate) struct SineTransform {
    bounds: Vec<(f64, f64)>,
}

impl SineTransform {
    pub(crate) fn new(bounds: Vec<(f64, f64)>) -> Self {
        Self { bounds }
    }

    /// Bounded values of the unbounded coordinates `u`.
    pub(crate) fn to_x(&self, u: &[f64]) -> Vec<f64> {
        u.iter()
            .zip(&self.bounds)
            .map(|(u, (lo, hi))| lo + (hi - lo) * (u.sin() + 1.0) / 2.0)
            .collect()
    }

    /// Unbounded coordinates of the values `x`, clamped to the bounds first.
    pub(crate) fn to_u(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.bounds)
            .map(|(x, (lo, hi))| {
                if hi > lo {
                    (2.0 * (x.clamp(*lo, *hi) - lo) / (hi - lo) - 1.0).asin()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Initial simplex in `u`: the start, and one vertex per parameter that
    /// moves it by `step × (hi − lo)` in `x`, towards the farther bound.
    pub(crate) fn initial_simplex(&self, x0: &[f64], step: f64) -> Vec<Vec<f64>> {
        let x0: Vec<f64> = x0
            .iter()
            .zip(&self.bounds)
            .map(|(x, (lo, hi))| x.clamp(*lo, *hi))
            .collect();
        let u0 = self.to_u(&x0);
        let mut vertices = vec![u0.clone()];
        for (i, (lo, hi)) in self.bounds.iter().enumerate() {
            let mut vertex = u0.clone();
            if hi > lo {
                let delta = step * (hi - lo);
                let moved = if hi - x0[i] >= x0[i] - lo {
                    x0[i] + delta
                } else {
                    x0[i] - delta
                };
                vertex[i] = self.to_u(&with_value(&x0, i, moved))[i];
            } else {
                vertex[i] += 1.0;
            }
            vertices.push(vertex);
        }
        vertices
    }
}

fn with_value(values: &[f64], index: usize, value: f64) -> Vec<f64> {
    let mut values = values.to_vec();
    values[index] = value;
    values
}

/// A cost function seen through a [`SineTransform`]: the optimizer proposes
/// `u`, the cost function scores `x`.
pub(crate) struct BoundedProblem<C> {
    pub(crate) cost: C,
    transform: SineTransform,
}

impl<C: CostFunction<Param = Vec<f64>, Output = f64>> CostFunction for BoundedProblem<C> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, u: &Self::Param) -> Result<Self::Output, Error> {
        self.cost.cost(&self.transform.to_x(u))
    }
}

/// Build Nelder-Mead solver from configuration, with its simplex in `u`
fn build_nelder_mead_solver(
    transform: &SineTransform,
    initial_params: &[f64],
    config: &NelderMeadConfig,
) -> Result<NelderMead<Vec<f64>, f64>, String> {
    if !(config.initial_simplex_size > 0.0 && config.initial_simplex_size <= 1.0) {
        return Err(format!(
            "initial_simplex_size must be in (0, 1], got {}",
            config.initial_simplex_size
        ));
    }
    let vertices = transform.initial_simplex(initial_params, config.initial_simplex_size);

    // Build solver with configuration
    let mut solver = NelderMead::new(vertices)
        .with_sd_tolerance(config.sd_tolerance)
        .map_err(|e| format!("Failed to set sd_tolerance: {}", e))?;

    // Apply optional parameters if provided
    if let Some(alpha) = config.alpha {
        solver = solver
            .with_alpha(alpha)
            .map_err(|e| format!("Failed to set alpha: {}", e))?;
    }

    if let Some(gamma) = config.gamma {
        solver = solver
            .with_gamma(gamma)
            .map_err(|e| format!("Failed to set gamma: {}", e))?;
    }

    if let Some(rho) = config.rho {
        solver = solver
            .with_rho(rho)
            .map_err(|e| format!("Failed to set rho: {}", e))?;
    }

    if let Some(sigma) = config.sigma {
        solver = solver
            .with_sigma(sigma)
            .map_err(|e| format!("Failed to set sigma: {}", e))?;
    }

    Ok(solver)
}

/// Nelder-Mead on `cost` from `start`, with its simplex in `u`.
fn optimize_nelder_mead<E, C, O>(
    problem: &CalibrationProblem<E>,
    cost: C,
    start: &[f64],
    config: &NelderMeadConfig,
    observer: Option<O>,
) -> Result<(CalibrationResult, Option<C>), String>
where
    E: SimulationEngine,
    C: CostFunction<Param = Vec<f64>, Output = f64>,
    O: Observe<NelderMeadState> + 'static,
{
    let transform = SineTransform::new(problem.parameter_bounds());
    let solver = build_nelder_mead_solver(&transform, start, config)?;
    let bounded = BoundedProblem {
        cost,
        transform: transform.clone(),
    };
    let mut executor = Executor::new(bounded, solver)
        .configure(|state| state.max_iters(config.max_iterations).counting(true))
        .timer(true);
    if let Some(observer) = observer {
        executor = executor.add_observer(observer, ObserverMode::Always);
    }
    let mut result = executor
        .run()
        .map_err(|e| format!("Optimization failed: {}", e))?;

    let state = &result.state;
    let best = state
        .best_param
        .as_ref()
        .map(|u| transform.to_x(u))
        .unwrap_or_else(|| problem.applied_parameter_values(start));
    let calibration = CalibrationResult {
        best_parameters: best,
        parameter_names: problem.parameter_names(),
        final_loss: state.best_cost,
        iterations: state.iter as usize,
        converged: converged(&state.termination_status),
        termination_reason: format!("{:?}", state.termination_status),
    };
    let cost = result.problem.take_problem().map(|bounded| bounded.cost);
    Ok((calibration, cost))
}

/// Particle swarm on `cost`, its population drawn over the bounds.
fn optimize_particle_swarm<E, C, O>(
    problem: &CalibrationProblem<E>,
    cost: C,
    start: &[f64],
    config: &ParticleSwarmConfig,
    observer: Option<O>,
) -> Result<(CalibrationResult, Option<C>), String>
where
    E: SimulationEngine,
    C: CostFunction<Param = Vec<f64>, Output = f64>,
    O: Observe<ParticleSwarmState> + 'static,
{
    let bounds = problem.parameter_bounds();
    let lower_bound: Vec<f64> = bounds.iter().map(|(min, _)| *min).collect();
    let upper_bound: Vec<f64> = bounds.iter().map(|(_, max)| *max).collect();

    // Always use SmallRng for consistency, either seeded or from system entropy
    let mut solver = if let Some(seed) = config.seed {
        ParticleSwarm::new((lower_bound, upper_bound), config.num_particles)
            .with_rng_generator(SmallRng::seed_from_u64(seed))
    } else {
        ParticleSwarm::new((lower_bound, upper_bound), config.num_particles)
            .with_rng_generator(SmallRng::from_rng(&mut rand::rng()))
    };

    // Apply inertia strategy if provided
    solver = match &config.inertia_strategy {
        Some(InertiaWeightStrategy::Constant(w)) => solver
            .with_inertia_factor(*w)
            .map_err(|e| format!("Failed to set inertia_factor: {}", e))?,
        Some(InertiaWeightStrategy::Chaotic { w_min, w_max }) => solver
            .with_chaotic_inertia(*w_min, *w_max)
            .map_err(|e| format!("Failed to set chaotic_inertia: {}", e))?,
        None => solver, // Use argmin's default
    };

    // Apply acceleration strategy if provided
    solver = match &config.acceleration_strategy {
        Some(AccelerationStrategy::Constant { cognitive, social }) => {
            solver = solver
                .with_cognitive_factor(*cognitive)
                .map_err(|e| format!("Failed to set cognitive_factor: {}", e))?;
            solver
                .with_social_factor(*social)
                .map_err(|e| format!("Failed to set social_factor: {}", e))?
        }
        Some(AccelerationStrategy::TimeVarying {
            c1_initial,
            c1_final,
            c2_initial,
            c2_final,
        }) => solver
            .with_tvac(
                *c1_initial,
                *c1_final,
                *c2_initial,
                *c2_final,
                config.max_iterations as usize,
            )
            .map_err(|e| format!("Failed to set TVAC: {}", e))?,
        None => solver, // Use argmin's default
    };

    // Apply initialization strategy
    let argmin_init_strategy = match config.initialization_strategy {
        InitializationStrategy::UniformRandom => ArgminInitializationStrategy::UniformRandom,
        InitializationStrategy::LatinHypercube => ArgminInitializationStrategy::LatinHypercube,
        InitializationStrategy::OppositionBased => ArgminInitializationStrategy::OppositionBased,
    };
    solver = solver.with_initialization_strategy(argmin_init_strategy);

    // Apply velocity clamping if enabled
    if let Some(clamp_factor) = config.velocity_clamp_factor {
        solver = solver
            .with_velocity_clamping(clamp_factor)
            .map_err(|e| format!("Failed to set velocity_clamping: {}", e))?;
    }

    // Apply velocity mutation if enabled
    if let Some(threshold) = config.velocity_mutation_threshold {
        solver = solver
            .with_velocity_mutation(threshold)
            .map_err(|e| format!("Failed to set velocity_mutation: {}", e))?;
    }

    // Apply mutation if enabled
    if !matches!(config.mutation_strategy, MutationStrategy::None)
        && !matches!(config.mutation_application, MutationApplication::None)
    {
        let argmin_mutation_strategy = match config.mutation_strategy {
            MutationStrategy::None => ArgminMutationStrategy::None,
            MutationStrategy::Gaussian(std_dev) => ArgminMutationStrategy::Gaussian(std_dev),
            MutationStrategy::Cauchy(scale) => ArgminMutationStrategy::Cauchy(scale),
        };

        let argmin_mutation_application = match config.mutation_application {
            MutationApplication::None => ArgminMutationApplication::None,
            MutationApplication::GlobalBestOnly => ArgminMutationApplication::GlobalBestOnly,
            MutationApplication::AllParticles => ArgminMutationApplication::AllParticles,
            MutationApplication::BelowAverage => ArgminMutationApplication::BelowAverage,
        };

        solver = solver
            .with_mutation(
                argmin_mutation_strategy,
                config.mutation_probability,
                argmin_mutation_application,
            )
            .map_err(|e| format!("Failed to set mutation: {}", e))?;
    }

    let mut executor = Executor::new(cost, solver)
        .configure(|state| {
            let mut state = state.max_iters(config.max_iterations).counting(true);
            if let Some(target) = config.target_cost {
                state = state.target_cost(target);
            }
            state
        })
        .timer(true);
    if let Some(observer) = observer {
        executor = executor.add_observer(observer, ObserverMode::Always);
    }
    let mut result = executor
        .run()
        .map_err(|e| format!("Optimization failed: {}", e))?;

    let state = &result.state;
    let (best, best_cost) = match &state.best_individual {
        Some(particle) => (particle.position.clone(), particle.cost),
        None => (start.to_vec(), f64::INFINITY),
    };
    let calibration = CalibrationResult {
        // The values the objective scored, clamped to their bounds.
        best_parameters: problem.applied_parameter_values(&best),
        parameter_names: problem.parameter_names(),
        final_loss: best_cost,
        iterations: state.iter as usize,
        converged: converged(&state.termination_status),
        termination_reason: format!("{:?}", state.termination_status),
    };
    Ok((calibration, result.problem.take_problem()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_convergence_and_target_cost_count_as_converged() {
        let status = TerminationStatus::Terminated;
        assert!(converged(&status(TerminationReason::SolverConverged)));
        assert!(converged(&status(TerminationReason::TargetCostReached)));
        assert!(!converged(&status(TerminationReason::MaxItersReached)));
        assert!(!converged(&status(TerminationReason::Timeout)));
        assert!(!converged(&TerminationStatus::NotTerminated));
    }

    #[test]
    fn the_transform_keeps_every_value_inside_its_bounds() {
        let transform = SineTransform::new(vec![(0.0, 1.0), (-2.0, 3.0)]);
        for u in [-10.0, -1.6, 0.0, 0.7, 1.6, 25.0] {
            let x = transform.to_x(&[u, u]);
            assert!((0.0..=1.0).contains(&x[0]));
            assert!((-2.0..=3.0).contains(&x[1]));
        }
    }

    #[test]
    fn the_transform_round_trips_inside_the_bounds() {
        let transform = SineTransform::new(vec![(0.0, 1.0), (-2.0, 3.0)]);
        let x = [0.25, 1.5];
        let back = transform.to_x(&transform.to_u(&x));
        assert!((back[0] - 0.25).abs() < 1e-12);
        assert!((back[1] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn the_initial_simplex_moves_each_value_by_the_step_of_its_width() {
        let transform = SineTransform::new(vec![(0.0, 1.0), (0.0, 10.0)]);
        // A start at the lower bound (0) is not degenerate.
        let simplex = transform.initial_simplex(&[0.0, 9.5], 0.1);
        let x: Vec<Vec<f64>> = simplex.iter().map(|u| transform.to_x(u)).collect();

        assert_eq!(x.len(), 3);
        assert!((x[0][0] - 0.0).abs() < 1e-12 && (x[0][1] - 9.5).abs() < 1e-12);
        assert!((x[1][0] - 0.1).abs() < 1e-12, "{:?}", x[1]);
        // 9.5 + 1.0 leaves the bounds, so the vertex moves down.
        assert!((x[2][1] - 8.5).abs() < 1e-12, "{:?}", x[2]);
    }

    #[test]
    fn the_initial_simplex_moves_towards_the_farther_bound() {
        let transform = SineTransform::new(vec![(-2.0, 3.0)]);
        // 0.6 is 2.4 from the upper bound and 2.6 from the lower one.
        let simplex = transform.initial_simplex(&[0.6], 0.1);
        let moved = transform.to_x(&simplex[1])[0];
        assert!((moved - 0.1).abs() < 1e-12, "{moved}");
    }

    #[test]
    fn a_fixed_value_stays_at_every_vertex_of_a_simplex_that_still_spans_it() {
        let transform = SineTransform::new(vec![(2.0, 2.0)]);
        assert_eq!(transform.to_u(&[2.0]), vec![0.0]);
        let simplex = transform.initial_simplex(&[2.0], 0.1);
        assert_ne!(simplex[0], simplex[1]);
        for vertex in &simplex {
            assert_eq!(transform.to_x(vertex), vec![2.0]);
        }
    }

    #[test]
    fn the_initial_simplex_size_is_a_fraction_of_the_width() {
        let transform = SineTransform::new(vec![(0.0, 1.0)]);
        let solver = |size| {
            let config = NelderMeadConfig {
                initial_simplex_size: size,
                ..NelderMeadConfig::default()
            };
            build_nelder_mead_solver(&transform, &[0.5], &config)
        };
        assert!(solver(1.0).is_ok());
        for size in [0.0, 1.5] {
            let error = solver(size).err().expect("an invalid size");
            assert!(error.contains("must be in (0, 1]"), "{error}");
        }
    }
}
