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
//! Every method that reads the observations series by series takes the
//! series from [`SeriesIndex`].
//!
//! The summary of the replicates of a stochastic engine and every method of
//! `commol-inference` take the one quantile of [`statistics`].
//!
//! [`optimize_starts`] minimizes the objective with Nelder-Mead or a
//! particle swarm from several start points ([`start_points`]) in parallel,
//! and chooses the best ([`Starts`]): a calibration and every fit of a
//! replicate run it. [`optimize`] is its start 0, from the initial guess. A
//! particle swarm takes the seed its caller gives.
//!
//! ## Example
//!
//! ```rust
//! use commol_calibration::{
//!     CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig, ObservedDataPoint,
//!     OptimizationConfig, optimize,
//! };
//! use commol_core::Model;
//! use commol_difference::DifferenceEquations;
//!
//! // A → B at the rate k1, from A = 900 and B = 100.
//! let json = r#"{"name":"m","parameters":[{"id":"k1","value":0.1}],
//!     "engine":{"type":"DifferenceEquations"},
//!     "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
//!     "accumulators":[],"stratifications":[],
//!     "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k1"}]},
//!     "structure":{"kind":"aggregate","bin_amounts":{"A":900.0,"B":100.0}}}"#;
//! let model: Model = serde_json::from_str(json).unwrap();
//! let engine = DifferenceEquations::try_from_model(&model).unwrap();
//! // B at step 10 when k1 is 0.1.
//! let observed = 1000.0 - 900.0 * 0.9_f64.powi(10);
//! let problem = CalibrationProblem::new(
//!     engine,
//!     vec![ObservedDataPoint::new(10.0, "B".to_string(), observed)],
//!     vec![CalibrationParameter::new("k1".to_string(), 0.0, 1.0)],
//!     vec![],
//!     LossConfig::SumSquaredError,
//! )
//! .unwrap();
//! let config = OptimizationConfig::NelderMead(NelderMeadConfig::default());
//! let result = optimize(&problem, &config, 7).unwrap();
//! assert!((result.best_parameters[0] - 0.1).abs() < 1e-4);
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
pub mod series;
pub mod starts;
pub mod statistics;
pub mod subproblem;
pub mod types;
pub mod values;

pub use calibration_problem::CalibrationProblem;
pub use discrepancy::{Discrepancy, DiscrepancyConfig, TermError};
pub use feasibility::{EvaluationError, Feasibility, Violation};
pub use forward::{ForwardModel, Workspace, replicate_mean, schedule_of};
pub use likelihood::{
    Argument, Center, Likelihood, Noise, ObservationModel, Term, TermDerivatives,
};
pub use objective::{OptimizationObjective, Score};
pub use observation::{
    ObservationOperator, WindowError, index_at, window_increments, window_start,
};
pub use optimization::{
    AccelerationStrategy, ImprovementStop, InertiaWeightStrategy, InitializationStrategy,
    MutationApplication, MutationStrategy, NelderMeadConfig, NelderMeadState, NoObserver,
    OptimizationConfig, ParticleSwarmConfig, ParticleSwarmState, converged, optimize,
    optimize_from, optimize_subproblem, optimize_with_history,
};
pub use series::SeriesIndex;
pub use starts::{
    StartOutcome, Starts, optimize_starts, optimize_starts_with_observer, start_points,
};
pub use subproblem::{ConstraintSettings, Subproblem, SubproblemResult, predict};
pub use types::{
    CalibrationConstraint, CalibrationEvaluation, CalibrationParameter, CalibrationParameterType,
    CalibrationResult, CalibrationResultWithHistory, LossConfig, ObservedDataPoint,
    ReplicateSummary, Replicates, crps, weighted_crps,
};
pub use values::{check_model_values, configured_engine};
