//! # Commol Calibration
//!
//! The calibration problem of a model, for any engine that implements
//! `SimulationEngine` from `commol-core`, and the optimizers that minimize
//! it.
//!
//! A [`CalibrationProblem`] is built once and holds one layer per job:
//!
//! | Layer | Job |
//! |---|---|
//! | [`ForwardModel`] | Writes values into the engine and runs it, one run per replicate of a stochastic engine |
//! | [`ObservationOperator`] | Turns the rows of a run into one prediction per observation |
//! | [`Feasibility`] | Bounds, constraints, infeasible starts and runs, as a [`Violation`] |
//! | [`Discrepancy`] | Compares predictions with the observations: a loss, or the likelihood of an observation model per series ([`Likelihood`]) |
//! | [`OptimizationObjective`] | The argmin cost function: the only place that turns a violation into a penalty |
//!
//! [`optimize`] minimizes the objective with Nelder-Mead or a particle
//! swarm; [`optimize_starts`] runs several starts in parallel
//! ([`start_points`]).
//!
//! ## Example
//!
//! ```rust,ignore
//! use commol_calibration::{
//!     optimize, CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig,
//!     ObservedDataPoint, OptimizationConfig,
//! };
//!
//! let problem = CalibrationProblem::new(
//!     engine,
//!     vec![ObservedDataPoint::new(10.0, "B".to_string(), 501.0)],
//!     vec![CalibrationParameter::new("k1".to_string(), 0.0, 1.0)],
//!     vec![],
//!     LossConfig::SumSquaredError,
//! )?;
//! let config = OptimizationConfig::NelderMead(NelderMeadConfig::default());
//! let result = optimize(&problem, &config)?;
//! ```

#![warn(missing_docs)]

pub mod calibration_problem;
pub mod discrepancy;
pub mod feasibility;
pub mod forward;
pub mod likelihood;
pub mod objective;
pub mod observation;
pub mod optimization;
pub mod starts;
pub mod types;
pub mod values;

pub use calibration_problem::{check_positions, CalibrationProblem};
pub use discrepancy::{Discrepancy, DiscrepancyConfig};
pub use feasibility::{EvaluationError, Feasibility, Violation};
pub use forward::{ForwardModel, Workspace};
pub use likelihood::{Argument, Center, Likelihood, Noise, ObservationModel, Term};
pub use objective::{OptimizationObjective, Score};
pub use observation::ObservationOperator;
pub use optimization::{
    converged, optimize, optimize_from, optimize_with_history, optimize_with_observer,
    AccelerationStrategy, InertiaWeightStrategy, InitializationStrategy, MutationApplication,
    MutationStrategy, NelderMeadConfig, NelderMeadState, NoObserver, OptimizationConfig,
    ParticleSwarmConfig, ParticleSwarmState,
};
pub use starts::{optimize_starts, start_points, StartOutcome};
pub use types::{
    crps, weighted_crps, CalibrationConstraint, CalibrationEvaluation, CalibrationParameter,
    CalibrationParameterType, CalibrationResult, CalibrationResultWithHistory, LossConfig,
    ObservedDataPoint, ReplicateSummary, Replicates,
};
pub use values::{check_model_values, configured_engine};
