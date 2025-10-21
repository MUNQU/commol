//! Optimization solver setup and execution

use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::particleswarm::ParticleSwarm;
use epimodel_core::SimulationEngine;

use crate::calibration_problem::CalibrationProblem;
use crate::types::CalibrationResult;

/// Configuration for Nelder-Mead optimization
#[derive(Debug, Clone)]
pub struct NelderMeadConfig {
    /// Maximum number of iterations
    pub max_iterations: u64,

    /// Sample standard deviation tolerance (convergence criterion)
    /// Must be non-negative, defaults to EPSILON
    pub sd_tolerance: f64,

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
            alpha: None, // Use argmin's default: 1.0
            gamma: None, // Use argmin's default: 2.0
            rho: None,   // Use argmin's default: 0.5
            sigma: None, // Use argmin's default: 0.5
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

    /// Inertia weight applied to particle velocity
    /// Defaults to 1/(2*ln(2)) ≈ 0.721
    pub inertia_factor: Option<f64>,

    /// Cognitive acceleration factor (attraction to personal best)
    /// Defaults to 0.5 + ln(2) ≈ 1.193
    pub cognitive_factor: Option<f64>,

    /// Social acceleration factor (attraction to swarm best)
    /// Defaults to 0.5 + ln(2) ≈ 1.193
    pub social_factor: Option<f64>,

    /// Enable verbose output
    pub verbose: bool,
}

impl Default for ParticleSwarmConfig {
    fn default() -> Self {
        Self {
            num_particles: 20,
            max_iterations: 1000,
            target_cost: None,
            inertia_factor: None,   // Use argmin's default: 1/(2*ln(2))
            cognitive_factor: None, // Use argmin's default: 0.5 + ln(2)
            social_factor: None,    // Use argmin's default: 0.5 + ln(2)
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

    /// Set inertia weight factor
    /// Controls the influence of previous velocity on current velocity
    /// Defaults to 1/(2*ln(2)) ≈ 0.721
    pub fn with_inertia_factor(mut self, factor: f64) -> Self {
        self.inertia_factor = Some(factor);
        self
    }

    /// Set cognitive acceleration factor
    /// Controls attraction to particle's personal best position
    /// Defaults to 0.5 + ln(2) ≈ 1.193
    pub fn with_cognitive_factor(mut self, factor: f64) -> Self {
        self.cognitive_factor = Some(factor);
        self
    }

    /// Set social acceleration factor
    /// Controls attraction to swarm's best position
    /// Defaults to 0.5 + ln(2) ≈ 1.193
    pub fn with_social_factor(mut self, factor: f64) -> Self {
        self.social_factor = Some(factor);
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
    /// Most reliable for epidemiological models
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
/// use epimodel_calibration::{optimize, OptimizationConfig, NelderMeadConfig};
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

/// Optimize using Nelder-Mead algorithm
fn optimize_nelder_mead<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    initial_params: Vec<f64>,
    parameter_names: Vec<String>,
    config: NelderMeadConfig,
) -> Result<CalibrationResult, String> {
    // Create simplex vertices (n+1 vertices for n parameters)
    let n = initial_params.len();
    let mut vertices = vec![initial_params.clone()];

    // Create n additional vertices by perturbing each parameter
    for i in 0..n {
        let mut vertex = initial_params.clone();
        vertex[i] *= 1.1; // 10% perturbation
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

    let executor =
        Executor::new(problem, solver).configure(|state| state.max_iters(config.max_iterations));

    let result = if config.verbose {
        use argmin::core::observers::ObserverMode;
        use argmin_observer_slog::SlogLogger;

        eprintln!("=== Nelder-Mead Optimization (Verbose Mode) ===");
        eprintln!("Parameters: {:?}", parameter_names);
        eprintln!("Initial values: {:?}", initial_params);
        eprintln!("Max iterations: {}", config.max_iterations);
        eprintln!("SD tolerance: {}", config.sd_tolerance);
        eprintln!("===============================================");

        executor
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()
            .map_err(|e| format!("Optimization failed: {}", e))?
    } else {
        executor
            .run()
            .map_err(|e| format!("Optimization failed: {}", e))?
    };

    let state = result.state();

    Ok(CalibrationResult {
        best_parameters: state.best_param.clone().unwrap_or(initial_params),
        parameter_names,
        final_loss: state.best_cost,
        iterations: state.iter as usize,
        converged: state.termination_status.terminated(),
        termination_reason: format!("{:?}", state.termination_status),
    })
}

/// Optimize using Particle Swarm algorithm
fn optimize_particle_swarm<E: SimulationEngine>(
    problem: CalibrationProblem<E>,
    initial_params: Vec<f64>,
    parameter_names: Vec<String>,
    config: ParticleSwarmConfig,
) -> Result<CalibrationResult, String> {
    // Get bounds from CalibrationParameter definitions
    let bounds = problem.get_parameter_bounds();
    let lower_bound: Vec<f64> = bounds.iter().map(|(min, _)| *min).collect();
    let upper_bound: Vec<f64> = bounds.iter().map(|(_, max)| *max).collect();

    // Build solver with configuration
    let mut solver = ParticleSwarm::new((lower_bound, upper_bound), config.num_particles);

    // Apply optional parameters if provided
    if let Some(inertia) = config.inertia_factor {
        solver = solver
            .with_inertia_factor(inertia)
            .map_err(|e| format!("Failed to set inertia_factor: {}", e))?;
    }

    if let Some(cognitive) = config.cognitive_factor {
        solver = solver
            .with_cognitive_factor(cognitive)
            .map_err(|e| format!("Failed to set cognitive_factor: {}", e))?;
    }

    if let Some(social) = config.social_factor {
        solver = solver
            .with_social_factor(social)
            .map_err(|e| format!("Failed to set social_factor: {}", e))?;
    }

    let executor = Executor::new(problem, solver).configure(|state| {
        let mut state = state.max_iters(config.max_iterations);
        if let Some(target) = config.target_cost {
            state = state.target_cost(target);
        }
        state
    });

    let result = if config.verbose {
        use argmin::core::observers::ObserverMode;
        use argmin_observer_slog::SlogLogger;

        eprintln!("=== Particle Swarm Optimization (Verbose Mode) ===");
        eprintln!("Parameters: {:?}", parameter_names);
        eprintln!("Bounds: {:?}", bounds);
        eprintln!("Num particles: {}", config.num_particles);
        eprintln!("Max iterations: {}", config.max_iterations);
        if let Some(target) = config.target_cost {
            eprintln!("Target cost: {}", target);
        }
        eprintln!("===================================================");

        executor
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()
            .map_err(|e| format!("Optimization failed: {}", e))?
    } else {
        executor
            .run()
            .map_err(|e| format!("Optimization failed: {}", e))?
    };

    let state = result.state();

    // For ParticleSwarm, the state is PopulationState
    // best_individual is an Option containing the best particle found
    let (best_params, best_cost) = match &state.best_individual {
        Some(particle) => (particle.position.clone(), particle.cost),
        None => (initial_params.clone(), f64::INFINITY),
    };

    Ok(CalibrationResult {
        best_parameters: best_params,
        parameter_names,
        final_loss: best_cost,
        iterations: state.iter as usize,
        converged: state.termination_status.terminated(),
        termination_reason: format!("{:?}", state.termination_status),
    })
}
