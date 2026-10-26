//! The range rule of each setting of an uncertainty, identifiability or
//! sensitivity method, of a prior and of a quantity of interest.
//!
//! Each rule is declared once, here or in `commol_calibration::ranges` for
//! the settings a calibration shares, such as `starts`. The function or the
//! constructor of the engine that takes the setting checks it with the
//! rule, and [`RANGES`] lists the rules of each object of the Python API, so
//! the Python model of the object checks them when it is built
//! ([`commol_core::Ranges::check`]).

use commol_calibration::ranges::{AT, WINDOW};
use commol_core::{Bound, Count, Ranges, Real, Rule};

/// The replicates of a bootstrap.
pub const REPLICATES: Count = Count::at_least("replicates", 1);
/// The replicates of a method that needs two, such as EnbPI or the bootstrap
/// of Sobol indices.
pub const TWO_REPLICATES: Count = Count::at_least("replicates", 2);
/// The positions of a block of the observations.
pub const BLOCK_LENGTH: Count = Count::at_least("block_length", 1);
/// The draws a sample or a prior gives when asked for some.
pub const COUNT: Count = Count::at_least("count", 0);
/// The draws of a method.
pub const DRAWS: Count = Count::at_least("draws", 1);
/// The draws of a method that needs two, such as the base of Sobol, the
/// points of polynomial chaos or ABC rejection.
pub const TWO_DRAWS: Count = Count::at_least("draws", 2);
/// The draws of the Laplace mixture, which Pareto smoothing needs.
pub const MIXTURE_DRAWS: Count = Count::at_least("draws", 25);

/// The particles of sequential Monte Carlo.
pub const PARTICLES: Count = Count::at_least("particles", 2);
/// The share of the effective sample size each step of sequential Monte
/// Carlo keeps.
pub const TARGET_CESS: Real = Real::share("target_cess");
/// The share of the effective sample size below which sequential Monte Carlo
/// resamples.
pub const RESAMPLE_THRESHOLD: Real =
    Real::new("resample_threshold", Bound::Open(0.0), Bound::Closed(1.0));
/// The probability that a particle of sequential Monte Carlo does not move.
pub const STAY_PROBABILITY: Real = Real::share("stay_probability");
/// The acceptance below which ABC stops.
pub const MIN_ACCEPTANCE: Real = Real::share("min_acceptance");
/// The share of the draws ABC rejection keeps.
pub const QUANTILE: Real = Real::share("quantile");
/// The scale of the loss of a Gibbs posterior.
pub const ETA: Real = Real::positive("eta");

/// The step of the working level of adaptive conformal intervals.
pub const STEP: Real = Real::positive("step");

/// The points of a structural rank.
pub const POINTS: Count = Count::at_least("points", 1);
/// The largest subset of the collinearity indices; also at most the number
/// of parameters.
pub const MAX_SUBSET_SIZE: Count = Count::at_least("max_subset_size", 2);
/// The collinearity index below which a subset can be fitted together.
pub const THRESHOLD: Real = Real::new("threshold", Bound::Open(1.0), Bound::Closed(f64::INFINITY));
/// The absolute correlation above which a pair of a sample is listed.
pub const CORRELATION_THRESHOLD: Real = Real::share("correlation_threshold");
/// The spread of the ends of the starts, as a share of the bound width,
/// above which a parameter is flagged.
pub const PARAMETER_TOLERANCE: Real = Real::non_negative_or_infinite("parameter_tolerance");
/// How far above the best loss, relative to it, a start is near the best.
pub const LOSS_TOLERANCE: Real = Real::non_negative_or_infinite("loss_tolerance");

/// The steps of the tracer of a profile.
pub const MAX_STEPS: Count = Count::at_least("max_steps", 1);

/// The trajectories of Morris.
pub const TRAJECTORIES: Count = Count::at_least("trajectories", 1);
/// The levels of the grid of Morris; it is also even.
pub const LEVELS: Count = Count::at_least("levels", 2);
/// The degree of a polynomial chaos basis.
pub const DEGREE: Count = Count::at_least("degree", 1);
/// The q of the q-norm of a polynomial chaos basis.
pub const Q_NORM: Real = Real::new("q_norm", Bound::Open(0.0), Bound::Closed(1.0));
/// The largest number of inputs of a term of a polynomial chaos basis.
pub const MAX_INTERACTION: Count = Count::at_least("max_interaction", 1);

/// The first shape of a beta prior.
pub const A: Real = Real::positive("a");
/// The second shape of a beta prior.
pub const B: Real = Real::positive("b");
/// The mean of a normal prior.
pub const MEAN: Real = Real::finite("mean");
/// The standard deviation of a normal prior.
pub const SD: Real = Real::positive("sd");
/// The mean of the logarithm of a log-normal prior.
pub const MU: Real = Real::finite("mu");
/// The standard deviation of the logarithm of a log-normal prior.
pub const SIGMA: Real = Real::positive("sigma");
/// The shape of a gamma prior.
pub const SHAPE: Real = Real::positive("shape");
/// The rate of a gamma prior.
pub const RATE: Real = Real::positive("rate");

/// The start of the window of a gain.
pub const START: Real = Real::non_negative("start");
/// The end of the window of a gain.
pub const END: Real = Real::non_negative("end");
/// The positions a quantity reads.
pub const POSITIONS: Real = Real::non_negative("positions");

/// The rules of a beta distribution.
pub const BETA: Ranges = Ranges {
    object: "Beta",
    rules: &[Rule::Real(A), Rule::Real(B)],
};

/// The rules of a normal distribution.
pub const NORMAL: Ranges = Ranges {
    object: "Normal",
    rules: &[Rule::Real(MEAN), Rule::Real(SD)],
};

/// The rules of a log-normal distribution.
pub const LOG_NORMAL: Ranges = Ranges {
    object: "LogNormal",
    rules: &[Rule::Real(MU), Rule::Real(SIGMA)],
};

/// The rules of a gamma distribution.
pub const GAMMA: Ranges = Ranges {
    object: "Gamma",
    rules: &[Rule::Real(SHAPE), Rule::Real(RATE)],
};

/// The rules of the value of a series at a position.
pub const VALUE_AT: Ranges = Ranges {
    object: "ValueAt",
    rules: &[Rule::Real(AT)],
};

/// The rules of a gain.
pub const GAIN: Ranges = Ranges {
    object: "Gain",
    rules: &[Rule::Real(START), Rule::Real(END)],
};

/// The rules of a peak.
pub const PEAK: Ranges = Ranges {
    object: "Peak",
    rules: &[Rule::Real(POSITIONS)],
};

/// The rules of the position of a peak.
pub const PEAK_POSITION: Ranges = Ranges {
    object: "PeakPosition",
    rules: &[Rule::Real(POSITIONS)],
};

/// The rules of a trajectory.
pub const TRAJECTORY: Ranges = Ranges {
    object: "Trajectory",
    rules: &[Rule::Real(POSITIONS)],
};

/// The rules of the increments of a series.
pub const INCREMENTS: Ranges = Ranges {
    object: "Increments",
    rules: &[Rule::Real(WINDOW)],
};

/// The rules of the settings of each object of the methods, as the Python
/// API names its class.
pub const RANGES: &[Ranges] = &[
    crate::settings::smc::InformationMatchingEta::RANGES,
    crate::settings::smc::BootstrapCoverageEta::RANGES,
    crate::settings::smc::GibbsPosteriorSettings::RANGES,
    crate::settings::smc::BayesianPosteriorSettings::RANGES,
    crate::settings::smc::AbcSmcSettings::RANGES,
    crate::settings::smc::AbcRejectionSettings::RANGES,
    crate::settings::resampling::BayesianBootstrapSettings::RANGES,
    crate::settings::resampling::ParametricBootstrapSettings::RANGES,
    crate::settings::resampling::ResidualBootstrapSettings::RANGES,
    crate::settings::resampling::WildBootstrapSettings::RANGES,
    crate::settings::resampling::BlockJackknifeSettings::RANGES,
    crate::settings::resampling::ResidualNoiseSettings::RANGES,
    crate::settings::conformal::SplitConformalSettings::RANGES,
    crate::settings::conformal::RollingConformalSettings::RANGES,
    crate::settings::conformal::JackknifePlusSettings::RANGES,
    crate::settings::conformal::EnbPiSettings::RANGES,
    crate::settings::conformal::AdaptiveConformalSettings::RANGES,
    crate::settings::profiles::Threshold::RANGES,
    crate::settings::profiles::PredictionSpec::RANGES,
    crate::settings::profiles::ProfileLikelihoodSettings::RANGES,
    crate::settings::profiles::PredictionProfileSettings::RANGES,
    crate::settings::identifiability::StructuralRankSettings::RANGES,
    crate::settings::identifiability::CollinearitySettings::RANGES,
    crate::settings::identifiability::SampleIdentifiabilitySettings::RANGES,
    crate::settings::identifiability::StartsIdentifiabilitySettings::RANGES,
    crate::settings::local::LaplaceSettings::RANGES,
    crate::settings::local::SandwichSettings::RANGES,
    crate::settings::local::GibbsTargetSettings::RANGES,
    crate::settings::local::LaplaceMixtureSettings::RANGES,
    crate::settings::sensitivity::PropagationSettings::RANGES,
    crate::settings::sensitivity::PrccSettings::RANGES,
    crate::settings::sensitivity::MorrisSettings::RANGES,
    crate::settings::sensitivity::SobolSettings::RANGES,
    crate::settings::sensitivity::PolynomialChaosSettings::RANGES,
    BETA,
    NORMAL,
    LOG_NORMAL,
    GAMMA,
    VALUE_AT,
    GAIN,
    PEAK,
    PEAK_POSITION,
    TRAJECTORY,
    INCREMENTS,
];
