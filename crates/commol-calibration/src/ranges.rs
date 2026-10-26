//! The range rule of each setting of a calibration: its observations,
//! constraints, observation models, optimizers and problem.
//!
//! Each rule is declared once, here. The function or the constructor of the
//! engine that takes the setting checks it with the rule, and [`RANGES`]
//! lists the rules of each object of the Python API, so the Python model of
//! the object checks them when it is built
//! ([`commol_core::Ranges::check`]).

use commol_core::{Bound, Count, Ranges, Real, Rule};

/// The position of an observation or a constraint on the axis.
pub const AT: Real = Real::non_negative("at");
/// The observed value of an observation.
pub const VALUE: Real = Real::non_negative_or_infinite("value");
/// The weight of an observation or a constraint.
pub const WEIGHT: Real = Real::positive_or_infinite("weight");
/// The window of an observation of a cumulative output, and the length of
/// a window of the grid of windows.
pub const WINDOW: Real = Real::positive("window");
/// The index of a window of the grid of windows.
pub const WINDOW_INDEX: Count = Count::at_least("index", 0);
/// The last position the ends of the grid of windows cover.
pub const GRID_END: Real = Real::finite("end");
/// The trials of a binomial observation.
pub const TRIALS: Count = Count::at_least("trials", 0);

/// The standard deviation of normal observations.
pub const SD: Real = Real::positive("sd");
/// The standard deviation of the logarithm of log-normal observations.
pub const LOG_NORMAL_SIGMA: Real = Real::positive("sigma");
/// The degrees of freedom of Student-t observations.
pub const DF: Real = Real::positive("df");
/// The scale of Student-t observations.
pub const STUDENT_T_SCALE: Real = Real::positive("scale");
/// The overdispersion of negative binomial observations; 0 is the Poisson.
pub const OVERDISPERSION: Real = Real::non_negative("overdispersion");

/// The iterations of an optimizer.
pub const MAX_ITERATIONS: Count = Count::at_least("max_iterations", 1);
/// The iterations between the headers of a verbose optimizer.
pub const HEADER_INTERVAL: Count = Count::at_least("header_interval", 1);
/// The tolerance of the standard deviation of a Nelder-Mead simplex.
pub const SD_TOLERANCE: Real = Real::positive_or_infinite("sd_tolerance");
/// The size of the initial simplex of Nelder-Mead, as a fraction of the
/// width of each bound.
pub const INITIAL_SIMPLEX_SIZE: Real =
    Real::new("initial_simplex_size", Bound::Open(0.0), Bound::Closed(1.0));
/// The reflection coefficient of Nelder-Mead.
pub const ALPHA: Real = Real::positive_or_infinite("alpha");
/// The expansion coefficient of Nelder-Mead.
pub const GAMMA: Real = Real::positive_or_infinite("gamma");
/// The contraction coefficient of Nelder-Mead.
pub const RHO: Real = Real::positive_or_infinite("rho");
/// The shrink coefficient of Nelder-Mead.
pub const SIGMA: Real = Real::positive_or_infinite("sigma");

/// The particles of a swarm.
pub const PARTICLES: Count = Count::at_least("particles", 1);
/// The constant inertia weight of a swarm.
pub const INERTIA_FACTOR: Real = Real::positive_or_infinite("inertia_factor");
/// The smallest chaotic inertia weight of a swarm.
pub const INERTIA_MIN: Real = Real::positive_or_infinite("inertia_min");
/// The largest chaotic inertia weight of a swarm.
pub const INERTIA_MAX: Real = Real::positive_or_infinite("inertia_max");
/// The constant cognitive coefficient of a swarm.
pub const COGNITIVE: Real = Real::positive_or_infinite("cognitive");
/// The constant social coefficient of a swarm.
pub const SOCIAL: Real = Real::positive_or_infinite("social");
/// The first cognitive coefficient of time-varying acceleration.
pub const COGNITIVE_INITIAL: Real = Real::positive_or_infinite("cognitive_initial");
/// The last cognitive coefficient of time-varying acceleration.
pub const COGNITIVE_FINAL: Real = Real::positive_or_infinite("cognitive_final");
/// The first social coefficient of time-varying acceleration.
pub const SOCIAL_INITIAL: Real = Real::positive_or_infinite("social_initial");
/// The last social coefficient of time-varying acceleration.
pub const SOCIAL_FINAL: Real = Real::positive_or_infinite("social_final");
/// The scale of a mutation of a swarm.
pub const MUTATION_SCALE: Real = Real::positive_or_infinite("mutation_scale");
/// The probability of a mutation of a swarm, per iteration.
pub const MUTATION_PROBABILITY: Real = Real::new(
    "mutation_probability",
    Bound::Closed(0.0),
    Bound::Closed(1.0),
);
/// The velocity clamp of a swarm, as a fraction of the width of each bound.
pub const VELOCITY_CLAMP_FACTOR: Real = Real::new(
    "velocity_clamp_factor",
    Bound::Open(0.0),
    Bound::Closed(1.0),
);
/// The velocity below which a particle of a swarm is reinitialized.
pub const VELOCITY_MUTATION_THRESHOLD: Real =
    Real::non_negative_or_infinite("velocity_mutation_threshold");
/// The iterations of the improvement stop of a swarm.
pub const IMPROVEMENT_WINDOW: Count = Count::at_least("improvement_window", 1);
/// The absolute tolerance of the improvement stop of a swarm.
pub const IMPROVEMENT_TOL_ABS: Real = Real::non_negative_or_infinite("improvement_tol_abs");
/// The relative tolerance of the improvement stop of a swarm.
pub const IMPROVEMENT_TOL_REL: Real = Real::non_negative_or_infinite("improvement_tol_rel");
/// The tolerance of the radius stop of a swarm.
pub const RADIUS_TOL: Real = Real::non_negative_or_infinite("radius_tol");

/// The starts of a multi-start optimization, the first from the initial
/// guess or the fit.
pub const STARTS: Count = Count::at_least("starts", 1);
/// The replicates of each evaluation of a stochastic engine.
pub const REPLICATES: Count = Count::at_least("replicates", 1);
/// The iterations of a calibration result.
pub const ITERATIONS: Count = Count::at_least("iterations", 0);

/// The rules of a normal observation model.
pub const NORMAL_OBSERVATIONS: Ranges = Ranges {
    object: "NormalObservations",
    rules: &[Rule::Real(SD)],
};

/// The rules of a log-normal observation model.
pub const LOG_NORMAL_OBSERVATIONS: Ranges = Ranges {
    object: "LogNormalObservations",
    rules: &[Rule::Real(LOG_NORMAL_SIGMA)],
};

/// The rules of a Student-t observation model.
pub const STUDENT_T_OBSERVATIONS: Ranges = Ranges {
    object: "StudentTObservations",
    rules: &[Rule::Real(DF), Rule::Real(STUDENT_T_SCALE)],
};

/// The rules of a negative binomial observation model.
pub const NEGATIVE_BINOMIAL_OBSERVATIONS: Ranges = Ranges {
    object: "NegativeBinomialObservations",
    rules: &[Rule::Real(OVERDISPERSION)],
};

/// The rules of a binomial observation model.
pub const BINOMIAL_OBSERVATIONS: Ranges = Ranges {
    object: "BinomialObservations",
    rules: &[Rule::Count(TRIALS)],
};

/// The rules of the settings of each object of a calibration, as the Python
/// API names its class.
pub const RANGES: &[Ranges] = &[
    crate::settings::Observation::RANGES,
    crate::types::CalibrationConstraint::RANGES,
    NORMAL_OBSERVATIONS,
    LOG_NORMAL_OBSERVATIONS,
    STUDENT_T_OBSERVATIONS,
    NEGATIVE_BINOMIAL_OBSERVATIONS,
    BINOMIAL_OBSERVATIONS,
    crate::settings::NelderMeadSettings::RANGES,
    crate::settings::ParticleSwarmSettings::RANGES,
    crate::settings::ProblemSettings::RANGES,
    crate::settings::FitRecord::RANGES,
];
