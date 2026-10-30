//! The settings of a calibration as the Python API writes them: the JSON of
//! the `model_dump()` of its pydantic classes, read with
//! [`commol_core::json::read`].
//!
//! Each type here has the fields of its Python class, with the same names,
//! so the value a reader reads writes back to the same JSON. Where the type
//! that the engine computes with has another shape, the settings type gives
//! it: an [`Observation`] its [`ObservedDataPoint`], an [`OptimizerSettings`]
//! its [`OptimizationConfig`], a [`ProblemSettings`] its
//! [`CalibrationProblem`].
//!
//! | Type | Python class |
//! |---|---|
//! | [`Observation`] | `ObservedDataPoint` |
//! | [`NelderMeadSettings`] | `NelderMeadConfig` |
//! | [`ParticleSwarmSettings`] | `ParticleSwarmConfig` |
//! | [`PolishSettings`] | `NelderMeadConfig`, as the `polish` of a `ParticleSwarmConfig` |
//! | [`OptimizerSettings`] | `NelderMeadConfig \| ParticleSwarmConfig`, by `kind` |
//! | [`ProblemSettings`] | `CalibrationProblem` |
//! | [`FitRecord`] | `CalibrationResult` |

use commol_core::json::Pairs;
use commol_core::{Rule, SimulationEngine, checked, tagged};
use serde::{Deserialize, Serialize};

use crate::calibration_problem::CalibrationProblem;
use crate::discrepancy::DiscrepancyConfig;
use crate::likelihood::ObservationModel;
use crate::optimization::{
    AccelerationStrategy, ImprovementStop, InertiaWeightStrategy, InitializationStrategy,
    MutationApplication, MutationStrategy, NelderMeadConfig, OptimizationConfig,
    ParticleSwarmConfig,
};
use crate::ranges::{
    ALPHA, AT, COGNITIVE, COGNITIVE_FINAL, COGNITIVE_INITIAL, GAMMA, HEADER_INTERVAL,
    IMPROVEMENT_TOL_ABS, IMPROVEMENT_TOL_REL, IMPROVEMENT_WINDOW, INERTIA_FACTOR, INERTIA_MAX,
    INERTIA_MIN, INITIAL_SIMPLEX_SIZE, ITERATIONS, MAX_ITERATIONS, MUTATION_PROBABILITY,
    MUTATION_SCALE, PARTICLES, RADIUS_TOL, REPLICATES, RHO, SCALE, SD_TOLERANCE, SIGMA, SOCIAL,
    SOCIAL_FINAL, SOCIAL_INITIAL, TRIALS, VALUE, VELOCITY_CLAMP_FACTOR,
    VELOCITY_MUTATION_THRESHOLD, WEIGHT, WINDOW,
};
use crate::types::{
    CalibrationConstraint, CalibrationParameter, LossConfig, ObservedDataPoint, ReplicateSummary,
};

/// An observation as the Python class `ObservedDataPoint` writes it: one
/// output, or the sum of several outputs under a label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct Observation {
    /// Position of the observation on the axis.
    pub at: f64,
    /// The output observed, or `None` with `outputs`.
    pub output: Option<String>,
    /// The outputs summed, or `None` with `output`.
    pub outputs: Option<Vec<String>>,
    /// The name of the series.
    pub label: String,
    /// The observed value.
    pub value: f64,
    /// The weight of the observation.
    pub weight: f64,
    /// The id of a factor that multiplies the prediction, if any.
    pub factor_id: Option<String>,
    /// The known number that multiplies the prediction.
    pub scale: f64,
    /// The length of the window of a cumulative output, if any.
    pub window: Option<f64>,
    /// The trials of a binomial observation model that reads them here.
    pub trials: Option<u64>,
}

checked!(
    Observation,
    "ObservedDataPoint",
    [
        Rule::Real(AT),
        Rule::Real(VALUE),
        Rule::Real(WEIGHT),
        Rule::Real(SCALE),
        Rule::Real(WINDOW),
        Rule::Count(TRIALS),
    ],
    check = Observation::check
);

impl Observation {
    /// Check that the observation has one of `output` and `outputs`, a
    /// label, and a window that starts at 0 or after it.
    ///
    /// # Errors
    ///
    /// Returns a message that says which rule the observation breaks.
    pub fn check(&self) -> Result<(), String> {
        match (&self.output, &self.outputs) {
            (Some(_), None) => {}
            (None, Some(outputs)) if outputs.is_empty() => {
                return Err("outputs must be non-empty when provided".to_string());
            }
            (None, Some(_)) => {}
            _ => return Err("Give exactly one of 'output' and 'outputs'.".to_string()),
        }
        if self.label.is_empty() {
            return Err("An observation needs a 'label' that names its series.".to_string());
        }
        self.window.map_or(Ok(()), |window| {
            crate::observation::check_window_end(self.at, window).map(|_| ())
        })
    }
}

impl From<&Observation> for ObservedDataPoint {
    fn from(observation: &Observation) -> Self {
        Self {
            at: observation.at,
            label: observation.label.clone(),
            outputs: match (&observation.output, &observation.outputs) {
                (Some(output), _) => vec![output.clone()],
                (None, outputs) => outputs.clone().unwrap_or_default(),
            },
            value: observation.value,
            weight: observation.weight,
            factor_id: observation.factor_id.clone(),
            scale: observation.scale,
            window: observation.window,
            trials: observation.trials,
        }
    }
}

/// The observed data points of `observations`.
pub fn observed_data(observations: &[Observation]) -> Vec<ObservedDataPoint> {
    observations.iter().map(ObservedDataPoint::from).collect()
}

/// The settings of Nelder-Mead, as the Python class `NelderMeadConfig`
/// writes them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct NelderMeadSettings {
    /// Maximum number of iterations.
    pub max_iterations: u64,
    /// Tolerance on the standard deviation of the costs of the simplex.
    pub sd_tolerance: f64,
    /// Size of the initial simplex, as a fraction of each bound width.
    pub initial_simplex_size: f64,
    /// Reflection coefficient, or argmin's default.
    pub alpha: Option<f64>,
    /// Expansion coefficient, or argmin's default.
    pub gamma: Option<f64>,
    /// Contraction coefficient, or argmin's default.
    pub rho: Option<f64>,
    /// Shrink coefficient, or argmin's default.
    pub sigma: Option<f64>,
    /// Whether the optimizer writes its progress.
    pub verbose: bool,
    /// Iterations between two headers of the progress table.
    pub header_interval: u64,
}

checked!(
    NelderMeadSettings,
    "NelderMeadConfig",
    [
        Rule::Count(MAX_ITERATIONS),
        Rule::Real(SD_TOLERANCE),
        Rule::Real(INITIAL_SIMPLEX_SIZE),
        Rule::Real(ALPHA),
        Rule::Real(GAMMA),
        Rule::Real(RHO),
        Rule::Real(SIGMA),
        Rule::Count(HEADER_INTERVAL),
    ]
);

impl NelderMeadSettings {
    /// The configuration of the optimizer.
    pub fn config(&self) -> NelderMeadConfig {
        NelderMeadConfig {
            max_iterations: self.max_iterations,
            sd_tolerance: self.sd_tolerance,
            initial_simplex_size: self.initial_simplex_size,
            alpha: self.alpha,
            gamma: self.gamma,
            rho: self.rho,
            sigma: self.sigma,
        }
    }
}

/// The inertia of a swarm, as the Python API names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Inertia {
    /// A constant weight, `inertia_factor`.
    Constant,
    /// A chaotic weight between `inertia_min` and `inertia_max`.
    Chaotic,
}

/// The acceleration of a swarm, as the Python API names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Acceleration {
    /// Constant coefficients, `cognitive` and `social`.
    Constant,
    /// Coefficients that move from their initial to their final value.
    TimeVarying,
}

/// The mutation of a swarm, as the Python API names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mutation {
    /// Gaussian steps of standard deviation `mutation_scale`.
    Gaussian,
    /// Cauchy steps of scale `mutation_scale`.
    Cauchy,
}

/// The particles a mutation moves, as the Python API names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mutated {
    /// The best particle of the swarm.
    GlobalBest,
    /// Every particle.
    AllParticles,
    /// The particles of a cost above the mean.
    BelowAverage,
}

/// How a swarm places its particles, as the Python API names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Initialization {
    /// Uniform draws over the bounds.
    Uniform,
    /// A Latin hypercube over the bounds.
    LatinHypercube,
    /// Uniform draws and their opposites, the better of each pair kept.
    OppositionBased,
}

/// The settings of a particle swarm, as the Python class
/// `ParticleSwarmConfig` writes them: each strategy is a name, and its
/// numbers are fields of their own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ParticleSwarmSettings {
    /// Number of particles.
    pub particles: usize,
    /// Maximum number of iterations.
    pub max_iterations: u64,
    /// Whether the optimizer writes its progress.
    pub verbose: bool,
    /// Iterations between two headers of the progress table.
    pub header_interval: u64,
    /// How the particles start.
    pub initialization: Initialization,
    /// The inertia, or the weight of canonical PSO.
    pub inertia: Option<Inertia>,
    /// The constant weight, or the weight of canonical PSO.
    pub inertia_factor: Option<f64>,
    /// The lowest chaotic weight.
    pub inertia_min: Option<f64>,
    /// The highest chaotic weight.
    pub inertia_max: Option<f64>,
    /// The acceleration, or the coefficients of canonical PSO.
    pub acceleration: Option<Acceleration>,
    /// The constant cognitive coefficient, or that of canonical PSO.
    pub cognitive: Option<f64>,
    /// The constant social coefficient, or that of canonical PSO.
    pub social: Option<f64>,
    /// The cognitive coefficient at the first iteration.
    pub cognitive_initial: Option<f64>,
    /// The cognitive coefficient at the last iteration.
    pub cognitive_final: Option<f64>,
    /// The social coefficient at the first iteration.
    pub social_initial: Option<f64>,
    /// The social coefficient at the last iteration.
    pub social_final: Option<f64>,
    /// The mutation, if any.
    pub mutation: Option<Mutation>,
    /// The scale of a mutation step.
    pub mutation_scale: Option<f64>,
    /// The probability of a mutation at each iteration.
    pub mutation_probability: Option<f64>,
    /// The particles a mutation moves.
    pub mutation_application: Option<Mutated>,
    /// The largest velocity, as a fraction of each bound width.
    pub velocity_clamp_factor: Option<f64>,
    /// The speed below which a velocity is drawn again.
    pub velocity_mutation_threshold: Option<f64>,
    /// The iterations of the improvement stop rule.
    pub improvement_window: Option<u64>,
    /// The absolute tolerance of the improvement stop rule.
    pub improvement_tol_abs: Option<f64>,
    /// The relative tolerance of the improvement stop rule.
    pub improvement_tol_rel: Option<f64>,
    /// The tolerance of the radius stop rule.
    pub radius_tol: Option<f64>,
    /// Nelder-Mead from the best point of the swarm, if any.
    pub polish: Option<PolishSettings>,
}

/// The polish of a swarm: the Python class `NelderMeadConfig`, by its
/// `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case")]
pub enum PolishSettings {
    /// Nelder-Mead.
    NelderMead(NelderMeadSettings),
}

tagged!(PolishSettings);

checked!(
    ParticleSwarmSettings,
    "ParticleSwarmConfig",
    [
        Rule::Count(PARTICLES),
        Rule::Count(MAX_ITERATIONS),
        Rule::Count(HEADER_INTERVAL),
        Rule::Real(INERTIA_FACTOR),
        Rule::Real(INERTIA_MIN),
        Rule::Real(INERTIA_MAX),
        Rule::Real(COGNITIVE),
        Rule::Real(SOCIAL),
        Rule::Real(COGNITIVE_INITIAL),
        Rule::Real(COGNITIVE_FINAL),
        Rule::Real(SOCIAL_INITIAL),
        Rule::Real(SOCIAL_FINAL),
        Rule::Real(MUTATION_SCALE),
        Rule::Real(MUTATION_PROBABILITY),
        Rule::Real(VELOCITY_CLAMP_FACTOR),
        Rule::Real(VELOCITY_MUTATION_THRESHOLD),
        Rule::Count(IMPROVEMENT_WINDOW),
        Rule::Real(IMPROVEMENT_TOL_ABS),
        Rule::Real(IMPROVEMENT_TOL_REL),
        Rule::Real(RADIUS_TOL),
    ],
    check = |settings: &ParticleSwarmSettings| settings.config().map(|_| ())
);

/// The value of `field` that the strategy `strategy` needs, or the message
/// that it is missing.
fn needed(value: Option<f64>, field: &str, strategy: &str) -> Result<f64, String> {
    value.ok_or_else(|| format!("{strategy} needs {field}"))
}

/// Check that no field of `fields` is set: the strategy `strategy` does not
/// read them.
fn unread(fields: &[(&str, bool)], strategy: &str) -> Result<(), String> {
    match fields.iter().find(|(_, set)| *set) {
        Some((field, _)) => Err(format!("{field} is read only with another {strategy}")),
        None => Ok(()),
    }
}

impl ParticleSwarmSettings {
    /// The configuration of the optimizer.
    ///
    /// # Errors
    ///
    /// Returns a message when a strategy lacks a number it needs, a number
    /// is set that no strategy set reads, or the chaotic weights are not in
    /// order.
    pub fn config(&self) -> Result<ParticleSwarmConfig, String> {
        Ok(ParticleSwarmConfig {
            particles: self.particles,
            max_iterations: self.max_iterations,
            target_cost: None,
            inertia_strategy: self.inertia_strategy()?,
            acceleration_strategy: self.acceleration_strategy()?,
            initialization_strategy: match self.initialization {
                Initialization::Uniform => InitializationStrategy::UniformRandom,
                Initialization::LatinHypercube => InitializationStrategy::LatinHypercube,
                Initialization::OppositionBased => InitializationStrategy::OppositionBased,
            },
            velocity_clamp_factor: self.velocity_clamp_factor,
            velocity_mutation_threshold: self.velocity_mutation_threshold,
            mutation_strategy: self.mutation_strategy()?,
            mutation_probability: self.mutation_probability.unwrap_or(0.0),
            mutation_application: match self.mutation_application {
                None => MutationApplication::None,
                Some(Mutated::GlobalBest) => MutationApplication::GlobalBestOnly,
                Some(Mutated::AllParticles) => MutationApplication::AllParticles,
                Some(Mutated::BelowAverage) => MutationApplication::BelowAverage,
            },
            improvement_stop: self.improvement_stop()?,
            radius_stop: self.radius_tol,
            polish: self
                .polish
                .as_ref()
                .map(|PolishSettings::NelderMead(settings)| Box::new(settings.config())),
        })
    }

    fn inertia_strategy(&self) -> Result<Option<InertiaWeightStrategy>, String> {
        let strategy = |name: &str| format!("inertia='{name}'");
        match self.inertia {
            None => unread(
                &[
                    ("inertia_factor", self.inertia_factor.is_some()),
                    ("inertia_min", self.inertia_min.is_some()),
                    ("inertia_max", self.inertia_max.is_some()),
                ],
                "inertia",
            )
            .map(|()| None),
            Some(Inertia::Constant) => {
                unread(
                    &[
                        ("inertia_min", self.inertia_min.is_some()),
                        ("inertia_max", self.inertia_max.is_some()),
                    ],
                    "inertia",
                )?;
                Ok(Some(InertiaWeightStrategy::Constant(
                    self.inertia_factor
                        .unwrap_or(InertiaWeightStrategy::DEFAULT_FACTOR),
                )))
            }
            Some(Inertia::Chaotic) => {
                unread(
                    &[("inertia_factor", self.inertia_factor.is_some())],
                    "inertia",
                )?;
                let w_min = needed(self.inertia_min, "inertia_min", &strategy("chaotic"))?;
                let w_max = needed(self.inertia_max, "inertia_max", &strategy("chaotic"))?;
                if w_max <= w_min {
                    return Err(format!(
                        "inertia_max ({w_max}) must be greater than inertia_min ({w_min})"
                    ));
                }
                Ok(Some(InertiaWeightStrategy::Chaotic { w_min, w_max }))
            }
        }
    }

    fn acceleration_strategy(&self) -> Result<Option<AccelerationStrategy>, String> {
        let constant = [
            ("cognitive", self.cognitive.is_some()),
            ("social", self.social.is_some()),
        ];
        let varying = [
            ("cognitive_initial", self.cognitive_initial.is_some()),
            ("cognitive_final", self.cognitive_final.is_some()),
            ("social_initial", self.social_initial.is_some()),
            ("social_final", self.social_final.is_some()),
        ];
        match self.acceleration {
            None => {
                unread(&constant, "acceleration")?;
                unread(&varying, "acceleration").map(|()| None)
            }
            Some(Acceleration::Constant) => {
                unread(&varying, "acceleration")?;
                let default = AccelerationStrategy::DEFAULT_COEFFICIENT;
                Ok(Some(AccelerationStrategy::Constant {
                    cognitive: self.cognitive.unwrap_or(default),
                    social: self.social.unwrap_or(default),
                }))
            }
            Some(Acceleration::TimeVarying) => {
                unread(&constant, "acceleration")?;
                let strategy = "acceleration='time_varying'";
                Ok(Some(AccelerationStrategy::TimeVarying {
                    c1_initial: needed(self.cognitive_initial, "cognitive_initial", strategy)?,
                    c1_final: needed(self.cognitive_final, "cognitive_final", strategy)?,
                    c2_initial: needed(self.social_initial, "social_initial", strategy)?,
                    c2_final: needed(self.social_final, "social_final", strategy)?,
                }))
            }
        }
    }

    fn mutation_strategy(&self) -> Result<MutationStrategy, String> {
        let Some(mutation) = self.mutation else {
            return unread(
                &[
                    ("mutation_scale", self.mutation_scale.is_some()),
                    ("mutation_probability", self.mutation_probability.is_some()),
                    ("mutation_application", self.mutation_application.is_some()),
                ],
                "mutation",
            )
            .map(|()| MutationStrategy::None);
        };
        let strategy = format!(
            "mutation='{}'",
            match mutation {
                Mutation::Gaussian => "gaussian",
                Mutation::Cauchy => "cauchy",
            }
        );
        let scale = needed(self.mutation_scale, "mutation_scale", &strategy)?;
        needed(self.mutation_probability, "mutation_probability", &strategy)?;
        if self.mutation_application.is_none() {
            return Err(format!("{strategy} needs mutation_application"));
        }
        Ok(match mutation {
            Mutation::Gaussian => MutationStrategy::Gaussian(scale),
            Mutation::Cauchy => MutationStrategy::Cauchy(scale),
        })
    }

    fn improvement_stop(&self) -> Result<Option<ImprovementStop>, String> {
        match (
            self.improvement_window,
            self.improvement_tol_abs,
            self.improvement_tol_rel,
        ) {
            (None, None, None) => Ok(None),
            (Some(window), Some(tol_abs), Some(tol_rel)) => Ok(Some(ImprovementStop {
                window,
                tol_abs,
                tol_rel,
            })),
            _ => Err(
                "The improvement stop rule needs improvement_window, improvement_tol_abs \
                 and improvement_tol_rel, all three"
                    .to_string(),
            ),
        }
    }
}

/// The optimizer of a calibration problem: the Python classes
/// `NelderMeadConfig` and `ParticleSwarmConfig`, by their `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case")]
pub enum OptimizerSettings {
    /// Nelder-Mead.
    NelderMead(NelderMeadSettings),
    /// A particle swarm.
    ParticleSwarm(Box<ParticleSwarmSettings>),
}

tagged!(OptimizerSettings);

impl OptimizerSettings {
    /// The configuration of the optimizer.
    ///
    /// # Errors
    ///
    /// Returns the message of [`ParticleSwarmSettings::config`].
    pub fn config(&self) -> Result<OptimizationConfig, String> {
        Ok(match self {
            Self::NelderMead(settings) => OptimizationConfig::NelderMead(settings.config()),
            Self::ParticleSwarm(settings) => OptimizationConfig::ParticleSwarm(settings.config()?),
        })
    }

    /// The progress that each stage of the optimizer writes, each by its own
    /// `verbose` and `header_interval`: a swarm and its polish are two
    /// stages.
    pub fn progress(&self) -> Progress {
        let table = |verbose: bool, interval: u64| verbose.then_some(interval);
        match self {
            Self::NelderMead(settings) => Progress {
                nelder_mead: table(settings.verbose, settings.header_interval),
                particle_swarm: None,
            },
            Self::ParticleSwarm(settings) => Progress {
                nelder_mead: settings.polish.as_ref().and_then(
                    |PolishSettings::NelderMead(polish)| {
                        table(polish.verbose, polish.header_interval)
                    },
                ),
                particle_swarm: table(settings.verbose, settings.header_interval),
            },
        }
    }
}

/// The progress table of each stage of an optimizer: the iterations between
/// two headers, or `None` for a stage that writes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Nelder-Mead, alone or as the polish of a swarm.
    pub nelder_mead: Option<u64>,
    /// The particle swarm.
    pub particle_swarm: Option<u64>,
}

impl Progress {
    /// Whether a stage writes its progress.
    pub fn any(&self) -> bool {
        self.nelder_mead.is_some() || self.particle_swarm.is_some()
    }
}

/// A calibration problem as the Python class `CalibrationProblem` writes
/// it: the data, the unknowns, what measures the fit and the optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ProblemSettings {
    /// The observations to fit.
    pub observed_data: Vec<Observation>,
    /// The values to calibrate, in problem order.
    pub parameters: Vec<CalibrationParameter>,
    /// The constraints on the parameters and the outputs.
    pub constraints: Vec<CalibrationConstraint>,
    /// The loss, or `None` with observation models.
    pub loss_function: Option<LossConfig>,
    /// The observation model of each series, in order, or `None` with a
    /// loss.
    pub observation_models: Option<Pairs<ObservationModel>>,
    /// Whether each residual is divided by the scale of its series.
    pub normalize_observations: bool,
    /// The optimizer.
    pub optimization_config: OptimizerSettings,
    /// The id that takes what the initial conditions leave, if any.
    pub initial_condition_remainder: Option<String>,
    /// The replicates of each evaluation of a stochastic engine.
    pub replicates: Option<usize>,
    /// What the observations are compared with, for a stochastic engine.
    pub replicate_summary: Option<ReplicateSummary>,
    /// The seed of every random number of the calibration, if set.
    pub seed: Option<u64>,
}

checked!(
    ProblemSettings,
    "CalibrationProblem",
    [Rule::Count(REPLICATES), Rule::Count(commol_core::SEED)]
);

impl ProblemSettings {
    /// The calibration problem of these settings on `engine`. The settings
    /// of the replicates go with a loss.
    ///
    /// # Errors
    ///
    /// Returns a message when the problem has both or neither of a loss and
    /// observation models, or the problem does not fit the engine.
    pub fn problem<E: SimulationEngine>(&self, engine: E) -> Result<CalibrationProblem<E>, String> {
        let discrepancy = match (&self.loss_function, &self.observation_models) {
            (Some(loss), None) => DiscrepancyConfig::Loss(*loss),
            (None, Some(models)) => DiscrepancyConfig::Likelihood(models.0.clone()),
            _ => {
                return Err(
                    "A calibration problem has a loss or observation models: exactly one of \
                     them"
                        .to_string(),
                );
            }
        };
        let mut problem = CalibrationProblem::new(
            engine,
            observed_data(&self.observed_data),
            self.parameters.clone(),
            self.constraints.clone(),
            discrepancy,
        )?;
        if self.loss_function.is_some() {
            problem = problem
                .with_observation_normalization(self.normalize_observations)?
                .with_replicate_settings(self.replicates, self.seed, self.replicate_summary)?;
        }
        if let Some(remainder) = &self.initial_condition_remainder {
            problem = problem.with_initial_condition_remainder(remainder)?;
        }
        Ok(problem)
    }
}

/// One start of a calibration, as the Python class `StartSummary` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartRecord {
    /// The start point, by parameter.
    pub start: Pairs<f64>,
    /// The best point found, by parameter.
    pub end: Pairs<f64>,
    /// The loss at the end.
    pub loss: f64,
    /// Whether the optimizer converged.
    pub converged: bool,
}

/// The result of a calibration, as the Python class `CalibrationResult`
/// writes it: what a method that reads a fit takes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct FitRecord {
    /// The best value of each parameter.
    pub best_parameters: Pairs<f64>,
    /// The loss at the best values.
    pub final_loss: f64,
    /// The iterations of the best start.
    pub iterations: usize,
    /// Whether the best start converged.
    pub converged: bool,
    /// Why the best start stopped.
    pub termination_reason: String,
    /// Every start that did not fail.
    pub starts: Vec<StartRecord>,
    /// Why each start that failed failed.
    pub failed: Vec<String>,
    /// The seed of the calibration.
    pub seed: u64,
}

checked!(
    FitRecord,
    "CalibrationResult",
    [Rule::Count(ITERATIONS), Rule::Count(commol_core::SEED)]
);

impl FitRecord {
    /// The best value of each of `names`, in their order.
    ///
    /// # Errors
    ///
    /// Returns a message that names the parameters the fit lacks.
    pub fn values(&self, names: &[String]) -> Result<Vec<f64>, String> {
        let missing: Vec<&str> = names
            .iter()
            .filter(|name| self.best_parameters.get(name).is_none())
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "The fit is not a fit of this problem: it lacks {missing:?}"
            ));
        }
        Ok(names
            .iter()
            .map(|name| *self.best_parameters.get(name).expect("checked above"))
            .collect())
    }
}
