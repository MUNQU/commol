//! Profile likelihood: confidence intervals of parameters and predictions
//! by fitting again with each held fixed (Raue et al. 2009; Kreutz et al.
//! 2012).
//!
//! | Module | Holds |
//! |---|---|
//! | [`tracer`] | [`ProfileTracer`]: one quantity held at values that step away from the fit, the others fitted again |
//! | [`threshold`] | The thresholds of the statistic: `½ χ²₁`, Beale's F and the bootstrap |
//!
//! A subproblem of `commol-calibration` ([`commol_calibration::Subproblem`])
//! holds the quantity: a parameter leaves the values the optimizer searches,
//! a prediction is an equality constraint of an augmented Lagrangian. Both
//! run the problem's own optimizer.

pub mod threshold;
pub mod tracer;

pub use threshold::{BootstrapThresholds, beale, bootstrap_thresholds, chi_square};
pub use tracer::{PenaltyRule, Profile, ProfileEnd, ProfilePoint, ProfileTracer, ProfiledQuantity};
