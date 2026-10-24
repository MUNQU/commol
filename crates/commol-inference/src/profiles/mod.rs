//! Profile likelihood: confidence intervals of parameters and predictions
//! by fitting again with each held fixed (Raue et al. 2009; Kreutz et al.
//! 2012).
//!
//! | Module | Holds |
//! |---|---|
//! | [`tracer`] | [`ProfileTracer`]: one quantity held at values that step away from the fit, the others fitted again |
//! | [`threshold`] | The thresholds of the statistic: `½ χ²₁`, Beale's F and the bootstrap |
//! | [`resets`] | What a flat profile says: resets of the other parameters, [`StructuralReading`] and the compensating parameters |
//! | [`verdict`] | [`ProfileVerdict`]: what the shape of a profile says about its quantity |
//!
//! A subproblem of `commol-calibration` ([`commol_calibration::Subproblem`])
//! holds the quantity: a parameter leaves the values the optimizer searches,
//! a prediction is an equality constraint of an augmented Lagrangian. Both
//! run the problem's own optimizer.

pub mod resets;
pub mod threshold;
pub mod tracer;
pub mod verdict;

pub use resets::{StructuralReading, Structure, Undetermined};
pub use threshold::{BootstrapThresholds, beale, bootstrap_thresholds, chi_square};
pub use tracer::{Profile, ProfileEnd, ProfilePoint, ProfileTracer, ProfiledQuantity};
pub use verdict::ProfileVerdict;
