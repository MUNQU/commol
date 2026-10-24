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
/// The window of an observation of a cumulative output.
pub const WINDOW: Real = Real::positive("window");
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
pub const FACTOR: Real = Real::positive_or_infinite("factor");
/// The smallest chaotic inertia weight of a swarm.
pub const W_MIN: Real = Real::positive_or_infinite("w_min");
/// The largest chaotic inertia weight of a swarm.
pub const W_MAX: Real = Real::positive_or_infinite("w_max");
/// The constant cognitive coefficient of a swarm.
pub const COGNITIVE: Real = Real::positive_or_infinite("cognitive");
/// The constant social coefficient of a swarm.
pub const SOCIAL: Real = Real::positive_or_infinite("social");
/// The first cognitive coefficient of time-varying acceleration.
pub const C1_INITIAL: Real = Real::positive_or_infinite("c1_initial");
/// The last cognitive coefficient of time-varying acceleration.
pub const C1_FINAL: Real = Real::positive_or_infinite("c1_final");
/// The first social coefficient of time-varying acceleration.
pub const C2_INITIAL: Real = Real::positive_or_infinite("c2_initial");
/// The last social coefficient of time-varying acceleration.
pub const C2_FINAL: Real = Real::positive_or_infinite("c2_final");
/// The scale of a mutation of a swarm.
pub const SCALE: Real = Real::positive_or_infinite("scale");
/// The probability of a mutation of a swarm, per iteration.
pub const PROBABILITY: Real = Real::new("probability", Bound::Closed(0.0), Bound::Closed(1.0));
/// The velocity clamp of a swarm, as a fraction of the width of each bound.
pub const CLAMP_FACTOR: Real = Real::new("clamp_factor", Bound::Open(0.0), Bound::Closed(1.0));
/// The velocity below which a particle of a swarm is reinitialized.
pub const MUTATION_THRESHOLD: Real = Real::non_negative_or_infinite("mutation_threshold");
/// The iterations of the improvement stop of a swarm.
pub const STOP_WINDOW: Count = Count::at_least("window", 1);
/// The absolute tolerance of the improvement stop of a swarm.
pub const TOL_ABS: Real = Real::non_negative_or_infinite("tol_abs");
/// The relative tolerance of the improvement stop of a swarm.
pub const TOL_REL: Real = Real::non_negative_or_infinite("tol_rel");
/// The tolerance of the radius stop of a swarm.
pub const TOL: Real = Real::non_negative_or_infinite("tol");

/// The starts of a multi-start optimization, the first from the initial
/// guess or the fit.
pub const STARTS: Count = Count::at_least("starts", 1);
/// The replicates of each evaluation of a stochastic engine.
pub const REPLICATES: Count = Count::at_least("replicates", 1);
/// The iterations of a calibration result.
pub const ITERATIONS: Count = Count::at_least("iterations", 0);

/// The rules of the settings of each object of a calibration, as the Python
/// API names its class.
pub const RANGES: &[Ranges] = &[
    Ranges {
        object: "ObservedDataPoint",
        rules: &[
            Rule::Real(AT),
            Rule::Real(VALUE),
            Rule::Real(WEIGHT),
            Rule::Real(WINDOW),
            Rule::Count(TRIALS),
        ],
    },
    Ranges {
        object: "CalibrationConstraint",
        rules: &[Rule::Real(WEIGHT), Rule::Real(AT)],
    },
    Ranges {
        object: "NormalObservations",
        rules: &[Rule::Real(SD)],
    },
    Ranges {
        object: "LogNormalObservations",
        rules: &[Rule::Real(LOG_NORMAL_SIGMA)],
    },
    Ranges {
        object: "StudentTObservations",
        rules: &[Rule::Real(DF), Rule::Real(STUDENT_T_SCALE)],
    },
    Ranges {
        object: "NegativeBinomialObservations",
        rules: &[Rule::Real(OVERDISPERSION)],
    },
    Ranges {
        object: "BinomialObservations",
        rules: &[Rule::Count(TRIALS)],
    },
    Ranges {
        object: "NelderMeadConfig",
        rules: &[
            Rule::Count(MAX_ITERATIONS),
            Rule::Real(SD_TOLERANCE),
            Rule::Real(INITIAL_SIMPLEX_SIZE),
            Rule::Real(ALPHA),
            Rule::Real(GAMMA),
            Rule::Real(RHO),
            Rule::Real(SIGMA),
            Rule::Count(HEADER_INTERVAL),
        ],
    },
    Ranges {
        object: "PSOConstantInertia",
        rules: &[Rule::Real(FACTOR)],
    },
    Ranges {
        object: "PSOChaoticInertia",
        rules: &[Rule::Real(W_MIN), Rule::Real(W_MAX)],
    },
    Ranges {
        object: "PSOConstantAcceleration",
        rules: &[Rule::Real(COGNITIVE), Rule::Real(SOCIAL)],
    },
    Ranges {
        object: "PSOTimeVaryingAcceleration",
        rules: &[
            Rule::Real(C1_INITIAL),
            Rule::Real(C1_FINAL),
            Rule::Real(C2_INITIAL),
            Rule::Real(C2_FINAL),
        ],
    },
    Ranges {
        object: "PSOMutationConfig",
        rules: &[Rule::Real(SCALE), Rule::Real(PROBABILITY)],
    },
    Ranges {
        object: "PSOVelocityConfig",
        rules: &[Rule::Real(CLAMP_FACTOR), Rule::Real(MUTATION_THRESHOLD)],
    },
    Ranges {
        object: "PSOImprovementStop",
        rules: &[
            Rule::Count(STOP_WINDOW),
            Rule::Real(TOL_ABS),
            Rule::Real(TOL_REL),
        ],
    },
    Ranges {
        object: "PSORadiusStop",
        rules: &[Rule::Real(TOL)],
    },
    Ranges {
        object: "ParticleSwarmConfig",
        rules: &[Rule::Count(PARTICLES), Rule::Count(MAX_ITERATIONS)],
    },
    Ranges {
        object: "CalibrationProblem",
        rules: &[Rule::Count(REPLICATES), Rule::Count(commol_core::SEED)],
    },
    Ranges {
        object: "CalibrationResult",
        rules: &[Rule::Count(ITERATIONS), Rule::Count(commol_core::SEED)],
    },
];
