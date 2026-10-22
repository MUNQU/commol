//! Optimizers of a calibration problem: Nelder-Mead and particle swarm.
//!
//! Each optimizer minimizes the [`OptimizationObjective`] of a problem from a
//! start point. Nothing here writes to standard output or standard error: a
//! caller that wants progress passes an observer
//! ([`optimize_with_observer`]).

use std::cell::RefCell;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{
    BoundedProblem, CostFunction, Error, Executor, IterState, Operator, PopulationState,
    SineTransform, State, TerminationReason, TerminationStatus,
};
use argmin::solver::augmentedlagrangian::{AugmentedLagrangian, AugmentedLagrangianState};
use argmin::solver::neldermead::NelderMead;
use argmin::solver::particleswarm::{
    InitializationStrategy as ArgminInitializationStrategy,
    MutationApplication as ArgminMutationApplication, MutationStrategy as ArgminMutationStrategy,
    Particle, ParticleSwarm,
};
use commol_core::SimulationEngine;
use commol_random::replicate_seed;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::calibration_problem::CalibrationProblem;
use crate::objective::{OptimizationObjective, Recorded};
use crate::subproblem::{
    ConstraintSettings, Fixed, FixedParameter, PredictionCost, Subproblem, SubproblemResult,
};
use crate::types::{CalibrationEvaluation, CalibrationResult, CalibrationResultWithHistory};

/// Configuration for Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations
    pub max_iterations: u64,

    /// The run stops when the sample standard deviation of the costs of the
    /// simplex falls below it. Must be non-negative; defaults to 1e-6.
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

impl InitializationStrategy {
    /// The strategy of a name: `"uniform"`, `"latin_hypercube"` or
    /// `"opposition_based"`.
    ///
    /// # Errors
    ///
    /// Returns a message that lists the names when `name` is none of them.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "uniform" => Ok(Self::UniformRandom),
            "latin_hypercube" => Ok(Self::LatinHypercube),
            "opposition_based" => Ok(Self::OppositionBased),
            _ => Err(format!(
                "Unknown initialization strategy '{name}'; the strategies are 'uniform', \
                 'latin_hypercube' and 'opposition_based'"
            )),
        }
    }
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

impl MutationStrategy {
    /// The mutation of a name, `"gaussian"` or `"cauchy"`, at `scale`.
    ///
    /// # Errors
    ///
    /// Returns a message that lists the names when `name` is none of them.
    pub fn from_name(name: &str, scale: f64) -> Result<Self, String> {
        match name {
            "gaussian" => Ok(Self::Gaussian(scale)),
            "cauchy" => Ok(Self::Cauchy(scale)),
            _ => Err(format!(
                "Unknown mutation strategy '{name}'; the strategies are 'gaussian' and 'cauchy'"
            )),
        }
    }
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

impl MutationApplication {
    /// The particles of a name: `"global_best"`, `"all_particles"` or
    /// `"below_average"`.
    ///
    /// # Errors
    ///
    /// Returns a message that lists the names when `name` is none of them.
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "global_best" => Ok(Self::GlobalBestOnly),
            "all_particles" => Ok(Self::AllParticles),
            "below_average" => Ok(Self::BelowAverage),
            _ => Err(format!(
                "Unknown mutation application '{name}'; the applications are 'global_best', \
                 'all_particles' and 'below_average'"
            )),
        }
    }
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
    run(
        &SearchSpace::of(problem),
        objective,
        config,
        start,
        seed,
        None::<NoObserver>,
    )
    .map(|(result, _)| result)
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
    run(
        &SearchSpace::of(problem),
        objective,
        config,
        &start,
        None,
        Some(observer),
    )
    .map(|(result, _)| result)
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
    let (result, recorded) = run(
        &SearchSpace::of(problem),
        recorded,
        config,
        &start,
        None,
        None::<NoObserver>,
    )?;
    let evaluations: Vec<CalibrationEvaluation> = recorded
        .map(|recorded| recorded.evaluations.into_inner())
        .unwrap_or_default();
    Ok(CalibrationResultWithHistory {
        result,
        evaluations,
    })
}

/// What an optimizer searches: the bounds and the name of each free value.
///
/// It is the problem's parameters, or those a subproblem leaves free
/// ([`crate::Subproblem`]).
#[derive(Debug, Clone)]
pub(crate) struct SearchSpace {
    pub(crate) bounds: Vec<(f64, f64)>,
    pub(crate) names: Vec<String>,
}

impl SearchSpace {
    /// Every parameter of `problem`.
    pub(crate) fn of<E: SimulationEngine>(problem: &CalibrationProblem<E>) -> Self {
        Self {
            bounds: problem.parameter_bounds(),
            names: problem.parameter_names(),
        }
    }

    /// Each value of `values` clamped to its bounds.
    pub(crate) fn clamp(&self, values: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(&self.bounds)
            .map(|(value, (lo, hi))| value.clamp(*lo, *hi))
            .collect()
    }

    /// The sine transform of the bounds, which Nelder-Mead searches through.
    fn transform(&self) -> Result<SineTransform<f64>, String> {
        SineTransform::new(self.bounds.clone())
            .map_err(|e| format!("Nelder-Mead needs finite bounds: {e}"))
    }

    pub(crate) fn check_start(&self, start: &[f64]) -> Result<(), String> {
        if start.len() == self.bounds.len() {
            Ok(())
        } else {
            Err(format!(
                "Expected {} start values, got {}",
                self.bounds.len(),
                start.len()
            ))
        }
    }
}

/// Fit the values `subproblem` leaves free with the optimizer of `config`,
/// from `start`, which holds every parameter of the problem.
///
/// A held parameter is left out of the values the optimizer searches, and
/// its value replaces the one in `start`. A held prediction is an equality
/// constraint `g(θ) = c`, met by the augmented Lagrangian of the argmin
/// fork: each outer iteration runs the optimizer of `config` on
/// `f(θ) + λ (g(θ) − c) + (μ/2)(g(θ) − c)²`, Nelder-Mead from the point of
/// the iteration before, then updates `λ` and `μ` by the rule of Algencan.
/// A particle swarm runs to its `max_iterations` in each outer iteration,
/// so the run stops at the outer limit of the settings.
///
/// `seed`, when given, replaces the seed of a particle swarm; outer
/// iteration `k` of a held prediction takes `replicate_seed(seed, k)`.
///
/// # Errors
///
/// As [`optimize`], and a message when `start` has the wrong length.
pub fn optimize_subproblem<E: SimulationEngine>(
    subproblem: &Subproblem<'_, E>,
    config: &OptimizationConfig,
    start: &[f64],
    seed: Option<u64>,
) -> Result<SubproblemResult, String> {
    let problem = subproblem.problem;
    SearchSpace::of(problem).check_start(start)?;
    match &subproblem.fixed {
        Fixed::Parameter { index, value } => {
            let cost = FixedParameter {
                objective: OptimizationObjective::new(problem),
                index: *index,
                value: *value,
                evaluations: std::cell::Cell::new(0),
            };
            let mut space = SearchSpace::of(problem);
            space.bounds.remove(*index);
            space.names.remove(*index);
            let mut free = start.to_vec();
            free.remove(*index);
            let (mut fit, cost) = if space.bounds.is_empty() {
                let loss = cost.cost(&free).map_err(|e| e.to_string())?;
                let fit = CalibrationResult {
                    best_parameters: Vec::new(),
                    parameter_names: Vec::new(),
                    final_loss: loss,
                    iterations: 0,
                    converged: true,
                    termination_reason: "No free parameter".to_string(),
                };
                (fit, Some(cost))
            } else {
                run(&space, cost, config, &free, seed, None::<NoObserver>)?
            };
            let cost = cost.ok_or("Optimization failed: the solver kept the cost function")?;
            fit.best_parameters = cost.full(&fit.best_parameters);
            fit.parameter_names = problem.parameter_names();
            Ok(SubproblemResult {
                fit,
                prediction: None,
                multiplier: None,
                penalty: None,
                evaluations: cost.evaluations.get(),
            })
        }
        Fixed::Prediction(held) => {
            let space = SearchSpace::of(problem);
            let run = run_constrained(
                &space,
                PredictionCost::new(held),
                config,
                start,
                seed,
                &held.settings,
            )?;
            let (_, prediction) = run
                .cost
                .evaluate(&run.fit.best_parameters)
                .map_err(|e| e.to_string())?;
            Ok(SubproblemResult {
                fit: run.fit,
                prediction,
                multiplier: Some(run.multiplier),
                penalty: Some(run.penalty),
                evaluations: run.cost.evaluations.get(),
            })
        }
    }
}

/// The end of a constrained run.
struct ConstrainedRun<C> {
    fit: CalibrationResult,
    multiplier: f64,
    penalty: f64,
    cost: C,
}

/// The state of an augmented Lagrangian run with one constraint.
type LagrangianState = AugmentedLagrangianState<Vec<f64>, Vec<f64>, f64>;

/// The augmented Lagrangian of `settings` on `cost` and its constraint, over
/// `space` from `start`, with the optimizer of `config` as its inner solver.
fn run_constrained<C>(
    space: &SearchSpace,
    cost: C,
    config: &OptimizationConfig,
    start: &[f64],
    seed: Option<u64>,
    settings: &ConstraintSettings,
) -> Result<ConstrainedRun<C>, String>
where
    C: CostFunction<Param = Vec<f64>, Output = f64> + Operator<Param = Vec<f64>, Output = Vec<f64>>,
{
    let outer = |state: LagrangianState, param: Vec<f64>| {
        state
            .param(param)
            .max_iters(settings.max_outer_iterations)
            .counting(true)
    };
    let (best, state, cost) = match config {
        OptimizationConfig::NelderMead(nm_config) => {
            let transform = space.transform()?;
            build_nelder_mead_solver(&transform, start, nm_config)?;
            let factory = |u: &Vec<f64>| {
                let solver = build_nelder_mead_solver(&transform, &transform.to_x(u), nm_config)
                    .map_err(Error::msg)?;
                Ok((
                    solver,
                    NelderMeadState::new().max_iters(nm_config.max_iterations),
                ))
            };
            let bounded = BoundedProblem::new(cost, transform.clone());
            let u0 = transform.to_u(start);
            let mut result = Executor::new(bounded, lagrangian(factory, settings)?)
                .configure(|state| outer(state, u0))
                .run()
                .map_err(|e| format!("Optimization failed: {e}"))?;
            let best = transform.to_x(result.state.get_best_param().ok_or(NO_POINT)?);
            let cost = result.problem.take_problem().ok_or(NO_COST)?.into_inner();
            (best, result.state, cost)
        }
        OptimizationConfig::ParticleSwarm(ps_config) => {
            let base = seed.or(ps_config.seed);
            let outer_iteration = std::cell::Cell::new(0u64);
            let factory = |_: &Vec<f64>| {
                let k = outer_iteration.get();
                outer_iteration.set(k + 1);
                let swarm = seeded(ps_config, base.map(|base| replicate_seed(base, k)));
                let solver =
                    build_particle_swarm_solver(&space.bounds, &swarm).map_err(Error::msg)?;
                Ok((solver, swarm_state(ParticleSwarmState::new(), &swarm)))
            };
            let mut result = Executor::new(cost, lagrangian(factory, settings)?)
                .configure(|state| outer(state, space.clamp(start)))
                .run()
                .map_err(|e| format!("Optimization failed: {e}"))?;
            let best = space.clamp(result.state.get_best_param().ok_or(NO_POINT)?);
            let cost = result.problem.take_problem().ok_or(NO_COST)?;
            (best, result.state, cost)
        }
    };
    let fit = CalibrationResult {
        best_parameters: best,
        parameter_names: space.names.clone(),
        final_loss: state.get_best_cost(),
        iterations: state.get_inner_iters_total() as usize,
        converged: converged(state.get_termination_status()),
        termination_reason: format!("{:?}", state.get_termination_status()),
    };
    Ok(ConstrainedRun {
        fit,
        multiplier: state
            .get_multipliers()
            .and_then(|multipliers| multipliers.first().copied())
            .unwrap_or(0.0),
        penalty: state.get_penalty(),
        cost,
    })
}

/// The augmented Lagrangian of `settings` around the inner solver that
/// `factory` builds.
fn lagrangian<B>(
    factory: B,
    settings: &ConstraintSettings,
) -> Result<AugmentedLagrangian<B, Vec<f64>, f64>, String> {
    let mut solver = AugmentedLagrangian::new(factory)
        .with_tolerance(settings.tolerance)
        .map_err(|e| format!("Invalid constraint tolerance: {e}"))?;
    if let Some(multiplier) = settings.multiplier {
        solver = solver.with_multipliers(vec![multiplier]);
    }
    if let Some(penalty) = settings.penalty {
        solver = solver
            .with_penalty(penalty)
            .map_err(|e| format!("Invalid constraint penalty: {e}"))?;
    }
    Ok(solver)
}

const NO_POINT: &str = "Optimization failed: the solver returned no point";
const NO_COST: &str = "Optimization failed: the solver kept the cost function";

/// Run the optimizer of `config` on `cost` over `space` from `start`, and
/// return the result and the cost function.
pub(crate) fn run<C, O>(
    space: &SearchSpace,
    cost: C,
    config: &OptimizationConfig,
    start: &[f64],
    seed: Option<u64>,
    observer: Option<O>,
) -> Result<(CalibrationResult, Option<C>), String>
where
    C: CostFunction<Param = Vec<f64>, Output = f64>,
    O: Observe<NelderMeadState> + Observe<ParticleSwarmState> + 'static,
{
    space.check_start(start)?;
    match config {
        OptimizationConfig::NelderMead(nm_config) => {
            optimize_nelder_mead(space, cost, start, nm_config, observer)
        }
        OptimizationConfig::ParticleSwarm(ps_config) => {
            optimize_particle_swarm(space, cost, &seeded(ps_config, seed), observer)
        }
    }
}

/// The swarm of `config`, with `seed` in place of its own when given.
pub(crate) fn seeded(config: &ParticleSwarmConfig, seed: Option<u64>) -> ParticleSwarmConfig {
    let mut config = config.clone();
    if seed.is_some() {
        config.seed = seed;
    }
    config
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

/// Build Nelder-Mead solver from configuration, with its simplex in `u`
fn build_nelder_mead_solver(
    transform: &SineTransform<f64>,
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
fn optimize_nelder_mead<C, O>(
    space: &SearchSpace,
    cost: C,
    start: &[f64],
    config: &NelderMeadConfig,
    observer: Option<O>,
) -> Result<(CalibrationResult, Option<C>), String>
where
    C: CostFunction<Param = Vec<f64>, Output = f64>,
    O: Observe<NelderMeadState> + 'static,
{
    let transform = space.transform()?;
    let solver = build_nelder_mead_solver(&transform, start, config)?;
    let bounded = BoundedProblem::new(cost, transform.clone());
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
        .unwrap_or_else(|| space.clamp(start));
    let calibration = CalibrationResult {
        best_parameters: best,
        parameter_names: space.names.clone(),
        final_loss: state.best_cost,
        iterations: state.iter as usize,
        converged: converged(&state.termination_status),
        termination_reason: format!("{:?}", state.termination_status),
    };
    let cost = result
        .problem
        .take_problem()
        .map(BoundedProblem::into_inner);
    Ok((calibration, cost))
}

/// Particle swarm on `cost`, its population drawn over the bounds.
fn optimize_particle_swarm<C, O>(
    space: &SearchSpace,
    cost: C,
    config: &ParticleSwarmConfig,
    observer: Option<O>,
) -> Result<(CalibrationResult, Option<C>), String>
where
    C: CostFunction<Param = Vec<f64>, Output = f64>,
    O: Observe<ParticleSwarmState> + 'static,
{
    let solver = build_particle_swarm_solver(&space.bounds, config)?;
    let mut executor = Executor::new(cost, solver)
        .configure(|state| swarm_state(state, config))
        .timer(true);
    if let Some(observer) = observer {
        executor = executor.add_observer(observer, ObserverMode::Always);
    }
    let mut result = executor
        .run()
        .map_err(|e| format!("Optimization failed: {}", e))?;

    let state = &result.state;
    let best = state
        .best_individual
        .as_ref()
        .expect("the swarm keeps a best particle, a NaN cost counting as the worst");
    let calibration = CalibrationResult {
        // The values the objective scored, clamped to their bounds.
        best_parameters: space.clamp(&best.position),
        parameter_names: space.names.clone(),
        final_loss: best.cost,
        iterations: state.iter as usize,
        converged: converged(&state.termination_status),
        termination_reason: format!("{:?}", state.termination_status),
    };
    Ok((calibration, result.problem.take_problem()))
}

/// The limits of a swarm run: `max_iterations`, and `target_cost` when set.
pub(crate) fn swarm_state(
    state: ParticleSwarmState,
    config: &ParticleSwarmConfig,
) -> ParticleSwarmState {
    let mut state = state.max_iters(config.max_iterations).counting(true);
    if let Some(target) = config.target_cost {
        state = state.target_cost(target);
    }
    state
}

/// The swarm of `config` over `bounds`, its random numbers from its seed or,
/// without one, from the system.
pub(crate) fn build_particle_swarm_solver(
    bounds: &[(f64, f64)],
    config: &ParticleSwarmConfig,
) -> Result<ParticleSwarm<Vec<f64>, f64, SmallRng>, String> {
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

    // Apply mutation if a strategy and the particles it acts on are set
    let mutation_strategy = match config.mutation_strategy {
        MutationStrategy::None => None,
        MutationStrategy::Gaussian(std_dev) => Some(ArgminMutationStrategy::Gaussian(std_dev)),
        MutationStrategy::Cauchy(scale) => Some(ArgminMutationStrategy::Cauchy(scale)),
    };
    let mutation_application = match config.mutation_application {
        MutationApplication::None => None,
        MutationApplication::GlobalBestOnly => Some(ArgminMutationApplication::GlobalBestOnly),
        MutationApplication::AllParticles => Some(ArgminMutationApplication::AllParticles),
        MutationApplication::BelowAverage => Some(ArgminMutationApplication::BelowAverage),
    };
    if let (Some(strategy), Some(application)) = (mutation_strategy, mutation_application) {
        solver = solver
            .with_mutation(strategy, config.mutation_probability, application)
            .map_err(|e| format!("Failed to set mutation: {}", e))?;
    }

    Ok(solver)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(x − 0.7)²`, NaN below 0.5.
    struct HalfNan;

    impl CostFunction for HalfNan {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            Ok(if x[0] < 0.5 {
                f64::NAN
            } else {
                (x[0] - 0.7).powi(2)
            })
        }
    }

    #[test]
    fn a_swarm_with_nan_costs_finds_the_minimum_where_the_cost_is_a_number() {
        let space = SearchSpace {
            bounds: vec![(0.0, 1.0)],
            names: vec!["x".to_string()],
        };
        let config = ParticleSwarmConfig {
            max_iterations: 200,
            // A seed whose first particle falls where the cost is NaN.
            seed: Some(0),
            ..ParticleSwarmConfig::default()
        };
        let (fit, _) =
            optimize_particle_swarm(&space, HalfNan, &config, None::<NoObserver>).unwrap();
        assert!((fit.best_parameters[0] - 0.7).abs() < 1e-3);
        assert!(fit.final_loss < 1e-6);
    }

    /// `(x − 0.7)²`.
    struct Bowl;

    impl CostFunction for Bowl {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            Ok((x[0] - 0.7).powi(2))
        }
    }

    #[test]
    fn a_swarm_without_a_seed_draws_one_from_the_system() {
        let space = SearchSpace {
            bounds: vec![(0.0, 1.0)],
            names: vec!["x".to_string()],
        };
        let config = ParticleSwarmConfig {
            max_iterations: 200,
            seed: None,
            ..ParticleSwarmConfig::default()
        };
        let (fit, _) = optimize_particle_swarm(&space, Bowl, &config, None::<NoObserver>).unwrap();
        assert!((fit.best_parameters[0] - 0.7).abs() < 1e-3);
    }

    #[test]
    fn a_name_gives_its_strategy_and_an_unknown_name_lists_the_names() {
        assert!(matches!(
            InitializationStrategy::from_name("latin_hypercube"),
            Ok(InitializationStrategy::LatinHypercube)
        ));
        assert!(matches!(
            InitializationStrategy::from_name("opposition_based"),
            Ok(InitializationStrategy::OppositionBased)
        ));
        assert_eq!(
            MutationStrategy::from_name("cauchy", 0.2),
            Ok(MutationStrategy::Cauchy(0.2))
        );
        assert_eq!(
            MutationApplication::from_name("below_average"),
            Ok(MutationApplication::BelowAverage)
        );
        let unknown = [
            InitializationStrategy::from_name("sobol").map(|_| ()),
            MutationStrategy::from_name("uniform", 0.1).map(|_| ()),
            MutationApplication::from_name("worst").map(|_| ()),
        ];
        assert_eq!(
            unknown.map(Result::unwrap_err),
            [
                "Unknown initialization strategy 'sobol'; the strategies are 'uniform', \
                 'latin_hypercube' and 'opposition_based'"
                    .to_string(),
                "Unknown mutation strategy 'uniform'; the strategies are 'gaussian' and 'cauchy'"
                    .to_string(),
                "Unknown mutation application 'worst'; the applications are 'global_best', \
                 'all_particles' and 'below_average'"
                    .to_string(),
            ]
        );
    }

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
    fn the_initial_simplex_size_is_a_fraction_of_the_width() {
        let transform = SineTransform::new(vec![(0.0, 1.0)]).unwrap();
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

    #[test]
    fn nelder_mead_refuses_an_infinite_bound() {
        let space = SearchSpace {
            bounds: vec![(0.0, f64::INFINITY)],
            names: vec!["x".to_string()],
        };
        let config = NelderMeadConfig::default();
        let error = optimize_nelder_mead(&space, HalfNan, &[1.0], &config, None::<NoObserver>)
            .err()
            .expect("an infinite bound");
        assert!(
            error.starts_with("Nelder-Mead needs finite bounds"),
            "{error}"
        );
    }
}
