//! Optimization solver setup and execution

use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::particleswarm::{
    InitializationStrategy as ArgminInitializationStrategy,
    MutationApplication as ArgminMutationApplication, MutationStrategy as ArgminMutationStrategy,
    ParticleSwarm,
};
use commol_core::SimulationEngine;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::calibration_problem::CalibrationProblem;
use crate::types::{CalibrationEvaluation, CalibrationResult};

/// Print optimization header for verbose output
fn print_optimization_header(
    algorithm: &str,
    parameter_names: &[String],
    initial_values: &[f64],
    max_iterations: u64,
) {
    eprintln!("=== {} Optimization (Verbose Mode) ===", algorithm);
    eprintln!("Parameters: {:?}", parameter_names);
    eprintln!("Initial values: {:?}", initial_values);
    eprintln!("Max iterations: {}", max_iterations);
}

/// Run executor with logging observer
fn run_with_logging<O, S, I>(
    executor: Executor<O, S, I>,
) -> Result<argmin::core::OptimizationResult<O, S, I>, String>
where
    O: argmin::core::CostFunction,
    S: argmin::core::Solver<O, I>,
    I: argmin::core::State,
{
    use argmin::core::observers::ObserverMode;
    use argmin_observer_slog::SlogLogger;

    executor
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .map_err(|e| format!("Optimization failed: {}", e))
}

/// Configuration for Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations
    pub max_iterations: u64,

    /// Sample standard deviation tolerance (convergence criterion)
    /// Must be non-negative, defaults to EPSILON
    pub sd_tolerance: f64,

    /// Simplex perturbation multiplier for creating initial vertices
    /// Each vertex is created by multiplying one parameter dimension by this factor.
    /// A value of 1.1 means 10% perturbation. Must be > 1.0.
    pub simplex_perturbation: f64,

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

    /// Enable verbose output
    pub verbose: bool,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            sd_tolerance: 1e-6,
            simplex_perturbation: 1.1, // 10% perturbation
            alpha: None,               // Use argmin's default: 1.0
            gamma: None,               // Use argmin's default: 2.0
            rho: None,                 // Use argmin's default: 0.5
            sigma: None,               // Use argmin's default: 0.5
            verbose: false,
        }
    }
}

impl NelderMeadConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set sample standard deviation tolerance (convergence criterion)
    pub fn with_sd_tolerance(mut self, tolerance: f64) -> Self {
        self.sd_tolerance = tolerance;
        self
    }

    /// Set simplex perturbation multiplier
    /// Each simplex vertex is created by multiplying one parameter dimension by this factor.
    /// A value of 1.1 means 10% perturbation. Must be > 1.0.
    pub fn with_simplex_perturbation(mut self, perturbation: f64) -> Self {
        self.simplex_perturbation = perturbation;
        self
    }

    /// Set reflection parameter (alpha)
    /// Must be > 0, defaults to 1.0
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Set expansion parameter (gamma)
    /// Must be > 1, defaults to 2.0
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set contraction parameter (rho)
    /// Must be in (0, 0.5], defaults to 0.5
    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = Some(rho);
        self
    }

    /// Set shrinking parameter (sigma)
    /// Must be in (0, 1], defaults to 0.5
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Inertia weight strategy for PSO
#[derive(Debug, Clone)]
pub enum InertiaWeightStrategy {
    /// Constant inertia weight (default: 1/(2*ln(2)) ≈ 0.721)
    Constant(f64),
    /// Chaotic inertia weight using logistic map
    /// Parameters: (w_min, w_max)
    Chaotic { w_min: f64, w_max: f64 },
}

/// Acceleration coefficient strategy for PSO
#[derive(Debug, Clone)]
pub enum AccelerationStrategy {
    /// Constant coefficients (default: cognitive=0.5+ln(2), social=0.5+ln(2))
    Constant { cognitive: f64, social: f64 },
    /// Time-Varying Acceleration Coefficients (TVAC)
    TimeVarying {
        c1_initial: f64,
        c1_final: f64,
        c2_initial: f64,
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

    /// Enable verbose output
    pub verbose: bool,
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
            verbose: false,
        }
    }
}

impl ParticleSwarmConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of particles
    pub fn with_num_particles(mut self, num_particles: usize) -> Self {
        self.num_particles = num_particles;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set target cost (convergence criterion)
    pub fn with_target_cost(mut self, target_cost: f64) -> Self {
        self.target_cost = Some(target_cost);
        self
    }

    /// Set constant inertia weight factor
    /// Controls the influence of previous velocity on current velocity
    /// Defaults to 1/(2*ln(2)) ≈ 0.721
    pub fn with_inertia_factor(mut self, factor: f64) -> Self {
        self.inertia_strategy = Some(InertiaWeightStrategy::Constant(factor));
        self
    }

    /// Enable chaotic inertia weight using logistic map
    /// Uses a logistic map to generate chaotic variation in the inertia weight
    /// This helps particles escape local optima through non-linear dynamics.
    ///
    /// # Arguments
    /// * `w_min` - Minimum inertia weight
    /// * `w_max` - Maximum inertia weight
    pub fn with_chaotic_inertia(mut self, w_min: f64, w_max: f64) -> Self {
        self.inertia_strategy = Some(InertiaWeightStrategy::Chaotic { w_min, w_max });
        self
    }

    /// Set constant cognitive and social acceleration factors
    /// * `cognitive` - Controls attraction to particle's personal best position (default: 0.5 + ln(2))
    /// * `social` - Controls attraction to swarm's best position (default: 0.5 + ln(2))
    pub fn with_acceleration_factors(mut self, cognitive: f64, social: f64) -> Self {
        self.acceleration_strategy = Some(AccelerationStrategy::Constant { cognitive, social });
        self
    }

    /// Enable Time-Varying Acceleration Coefficients (TVAC)
    /// TVAC adjusts cognitive (c1) and social (c2) acceleration coefficients over iterations
    /// to balance exploration and exploitation. Typically, c1 decreases and c2 increases over
    /// time to shift from exploration to exploitation.
    ///
    /// # Arguments
    /// * `c1_initial` - Initial cognitive factor
    /// * `c1_final` - Final cognitive factor
    /// * `c2_initial` - Initial social factor
    /// * `c2_final` - Final social factor
    pub fn with_tvac(
        mut self,
        c1_initial: f64,
        c1_final: f64,
        c2_initial: f64,
        c2_final: f64,
    ) -> Self {
        self.acceleration_strategy = Some(AccelerationStrategy::TimeVarying {
            c1_initial,
            c1_final,
            c2_initial,
            c2_final,
        });
        self
    }

    /// Set particle initialization strategy
    /// Choose between different strategies for initializing particle positions:
    ///
    /// - **UniformRandom** (default): Standard uniform random initialization
    /// - **LatinHypercube**: Latin Hypercube Sampling ensures uniform distribution across search space
    /// - **OppositionBased**: Opposition-Based Learning generates both random and opposite positions
    pub fn with_initialization_strategy(mut self, strategy: InitializationStrategy) -> Self {
        self.initialization_strategy = strategy;
        self
    }

    /// Enable velocity clamping
    /// Limits particle velocities to prevent explosive behavior. The velocity is clamped to
    /// +-clamp_factor * (upper_bound - lower_bound) component-wise.
    ///
    /// # Arguments
    /// * `clamp_factor` - Fraction of search space range (typically 0.1 to 0.2)
    pub fn with_velocity_clamping(mut self, clamp_factor: f64) -> Self {
        self.velocity_clamp_factor = Some(clamp_factor);
        self
    }

    /// Enable velocity mutation when velocity approaches zero
    /// Reinitializes velocity components when they fall below a threshold, preventing
    /// particle stagnation.
    ///
    /// # Arguments
    /// * `threshold` - Velocity threshold below which reinitialization occurs (typically 0.001 to 0.01)
    pub fn with_velocity_mutation(mut self, threshold: f64) -> Self {
        self.velocity_mutation_threshold = Some(threshold);
        self
    }

    /// Enable mutation with specified strategy and application method
    /// Applies Gaussian or Cauchy mutation to particles with specified probability
    /// to help escape local optima.
    ///
    /// # Arguments
    /// * `strategy` - Mutation strategy (Gaussian or Cauchy with scale parameter)
    /// * `probability` - Mutation probability per iteration (0.0 to 1.0)
    /// * `application` - Which particles to mutate (GlobalBestOnly, AllParticles, or BelowAverage)
    pub fn with_mutation(
        mut self,
        strategy: MutationStrategy,
        probability: f64,
        application: MutationApplication,
    ) -> Self {
        self.mutation_strategy = strategy;
        self.mutation_probability = probability;
        self.mutation_application = application;
        self
    }

    /// Set random seed for reproducibility
    ///
    /// When set, the particle swarm will produce deterministic results,
    /// allowing you to reproduce the same optimization trajectory across runs.
    ///
    /// Note: The seed is set via this builder method (rather than as a required
    /// constructor parameter) to allow easy modification without creating a new
    /// configuration. This is particularly useful when running multiple
    /// calibrations with different seeds while keeping other parameters constant.
    ///
    /// # Arguments
    /// * `seed` - Random seed value
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = ParticleSwarmConfig::new()
    ///     .with_num_particles(40)
    ///     .with_max_iterations(500)
    ///     .with_seed(42);  // Reproducible results
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
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

/// Available optimization algorithms (kept for backward compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationAlgorithm {
    /// Nelder-Mead simplex method (gradient-free)
    NelderMead,
    /// Particle Swarm Optimization (gradient-free, global search)
    ParticleSwarm,
}

impl std::fmt::Display for OptimizationAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationAlgorithm::NelderMead => write!(f, "Nelder-Mead"),
            OptimizationAlgorithm::ParticleSwarm => write!(f, "Particle Swarm"),
        }
    }
}

/// Run optimization on a calibration problem
///
/// # Arguments
///
/// * `problem` - The calibration problem to solve
/// * `config` - Algorithm-specific optimization configuration
///
/// # Returns
///
/// Returns a `CalibrationResult` containing the best parameters found
///
/// # Example
///
/// ```rust,ignore
/// use commol_calibration::{optimize, OptimizationConfig, NelderMeadConfig};
///
/// let config = OptimizationConfig::NelderMead(
///     NelderMeadConfig::new()
///         .with_max_iterations(1000)
///         .with_tolerance(1e-6)
/// );
///
/// let result = optimize(problem, config)?;
/// println!("Best parameters: {:?}", result.best_parameters);
/// println!("Final loss: {}", result.final_loss);
/// ```
pub fn optimize<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    config: OptimizationConfig,
) -> Result<CalibrationResult, String> {
    let initial_params = problem.initial_parameters();
    let parameter_names = problem.parameter_names();

    match config {
        OptimizationConfig::NelderMead(nm_config) => {
            optimize_nelder_mead(problem, initial_params, parameter_names, nm_config)
        }
        OptimizationConfig::ParticleSwarm(ps_config) => {
            optimize_particle_swarm(problem, initial_params, parameter_names, ps_config)
        }
    }
}

/// Build Nelder-Mead solver from configuration
fn build_nelder_mead_solver(
    initial_params: &[f64],
    config: &NelderMeadConfig,
) -> Result<NelderMead<Vec<f64>, f64>, String> {
    // Create simplex vertices (n+1 vertices for n parameters)
    let mut vertices = vec![initial_params.to_vec()];

    // Create n additional vertices by perturbing each parameter
    for i in 0..initial_params.len() {
        let mut vertex = initial_params.to_vec();
        vertex[i] *= config.simplex_perturbation;
        vertices.push(vertex);
    }

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

/// Internal optimization function for Nelder-Mead that optionally returns evaluations
fn optimize_nelder_mead_internal<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    initial_params: Vec<f64>,
    parameter_names: Vec<String>,
    config: NelderMeadConfig,
    with_history: bool,
) -> Result<(CalibrationResult, Option<Vec<CalibrationEvaluation>>), String> {
    let solver = build_nelder_mead_solver(&initial_params, &config)?;
    let executor =
        Executor::new(problem, solver).configure(|state| state.max_iters(config.max_iterations));

    let result = if config.verbose {
        print_optimization_header(
            "Nelder-Mead",
            &parameter_names,
            &initial_params,
            config.max_iterations,
        );
        eprintln!("SD tolerance: {}", config.sd_tolerance);
        eprintln!("===============================================");

        run_with_logging(executor)?
    } else {
        executor
            .run()
            .map_err(|e| format!("Optimization failed: {}", e))?
    };

    // Extract evaluations if requested before consuming result
    let evaluations = if with_history {
        // Access the problem directly from the Problem wrapper
        result
            .problem()
            .problem
            .as_ref()
            .map(|p| p.get_evaluations())
    } else {
        None
    };

    let state = result.state();
    let best_params = state.best_param.clone().unwrap_or(initial_params.clone());

    // Use CalibrationProblem's method to fix auto-calculated parameters
    let corrected_params = result
        .problem()
        .problem
        .as_ref()
        .map(|p| p.fix_auto_calculated_parameters(best_params.clone()))
        .unwrap_or(best_params);

    Ok((
        CalibrationResult {
            best_parameters: corrected_params,
            parameter_names,
            final_loss: state.best_cost,
            iterations: state.iter as usize,
            converged: state.termination_status.terminated(),
            termination_reason: format!("{:?}", state.termination_status),
        },
        evaluations,
    ))
}

/// Optimize using Nelder-Mead algorithm
fn optimize_nelder_mead<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    initial_params: Vec<f64>,
    parameter_names: Vec<String>,
    config: NelderMeadConfig,
) -> Result<CalibrationResult, String> {
    let (result, _) =
        optimize_nelder_mead_internal(problem, initial_params, parameter_names, config, false)?;
    Ok(result)
}

/// Internal optimization function for Particle Swarm that optionally returns evaluations
fn optimize_particle_swarm_internal<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    initial_params: Vec<f64>,
    parameter_names: Vec<String>,
    config: ParticleSwarmConfig,
    with_history: bool,
) -> Result<(CalibrationResult, Option<Vec<CalibrationEvaluation>>), String> {
    // Get bounds from CalibrationParameter definitions
    let bounds = problem.parameter_bounds();
    let lower_bound: Vec<f64> = bounds.iter().map(|(min, _)| *min).collect();
    let upper_bound: Vec<f64> = bounds.iter().map(|(_, max)| *max).collect();

    // Build solver with configuration and RNG
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

    let executor = Executor::new(problem, solver).configure(|state| {
        let mut state = state.max_iters(config.max_iterations);
        if let Some(target) = config.target_cost {
            state = state.target_cost(target);
        }
        state
    });

    let result = if config.verbose {
        print_optimization_header(
            "Particle Swarm",
            &parameter_names,
            &initial_params,
            config.max_iterations,
        );
        eprintln!("Bounds: {:?}", bounds);
        eprintln!("Num particles: {}", config.num_particles);
        eprintln!(
            "Initialization strategy: {:?}",
            config.initialization_strategy
        );
        if let Some(ref strategy) = config.inertia_strategy {
            eprintln!("Inertia strategy: {:?}", strategy);
        }
        if let Some(ref strategy) = config.acceleration_strategy {
            eprintln!("Acceleration strategy: {:?}", strategy);
        }
        if let Some(clamp) = config.velocity_clamp_factor {
            eprintln!("Velocity clamping: {}", clamp);
        }
        if let Some(threshold) = config.velocity_mutation_threshold {
            eprintln!("Velocity mutation threshold: {}", threshold);
        }
        if !matches!(config.mutation_strategy, MutationStrategy::None) {
            eprintln!(
                "Mutation: {:?} (prob: {}, application: {:?})",
                config.mutation_strategy, config.mutation_probability, config.mutation_application
            );
        }
        if let Some(target) = config.target_cost {
            eprintln!("Target cost: {}", target);
        }
        eprintln!("===================================================");

        run_with_logging(executor)?
    } else {
        executor
            .run()
            .map_err(|e| format!("Optimization failed: {}", e))?
    };

    // Extract evaluations if requested before consuming result
    let evaluations = if with_history {
        // Access the problem directly from the Problem wrapper
        result
            .problem()
            .problem
            .as_ref()
            .map(|p| p.get_evaluations())
    } else {
        None
    };

    let state = result.state();

    // For ParticleSwarm, best_individual contains the best particle found
    let (best_params, best_cost) = match &state.best_individual {
        Some(particle) => (particle.position.clone(), particle.cost),
        None => (initial_params.clone(), f64::INFINITY),
    };

    // Use CalibrationProblem's method to fix auto-calculated parameters
    let corrected_params = result
        .problem()
        .problem
        .as_ref()
        .map(|p| p.fix_auto_calculated_parameters(best_params.clone()))
        .unwrap_or(best_params);

    Ok((
        CalibrationResult {
            best_parameters: corrected_params,
            parameter_names,
            final_loss: best_cost,
            iterations: state.iter as usize,
            converged: state.termination_status.terminated(),
            termination_reason: format!("{:?}", state.termination_status),
        },
        evaluations,
    ))
}

/// Optimize using Particle Swarm algorithm
fn optimize_particle_swarm<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    initial_params: Vec<f64>,
    parameter_names: Vec<String>,
    config: ParticleSwarmConfig,
) -> Result<CalibrationResult, String> {
    let (result, _) =
        optimize_particle_swarm_internal(problem, initial_params, parameter_names, config, false)?;
    Ok(result)
}

/// Optimize calibration problem with evaluation history tracking
///
/// This function runs optimization and returns all objective function evaluations
/// that occurred during the optimization process.
///
/// # Arguments
/// * `problem` - The calibration problem to optimize
/// * `config` - The optimization algorithm configuration
///
/// # Returns
/// * `CalibrationResultWithHistory` containing best parameters and all evaluations
pub fn optimize_with_history<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    config: OptimizationConfig,
) -> Result<crate::types::CalibrationResultWithHistory, String> {
    let initial_params = problem.initial_parameters();
    let parameter_names = problem.parameter_names();

    let (result, evaluations) = match config {
        OptimizationConfig::NelderMead(nm_config) => optimize_nelder_mead_internal(
            problem,
            initial_params,
            parameter_names,
            nm_config,
            true,
        )?,
        OptimizationConfig::ParticleSwarm(ps_config) => optimize_particle_swarm_internal(
            problem,
            initial_params,
            parameter_names,
            ps_config,
            true,
        )?,
    };

    Ok(crate::types::CalibrationResultWithHistory {
        best_parameters: result.best_parameters,
        parameter_names: result.parameter_names,
        final_loss: result.final_loss,
        iterations: result.iterations,
        converged: result.converged,
        termination_reason: result.termination_reason,
        evaluations: evaluations.unwrap_or_default(),
    })
}
