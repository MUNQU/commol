//! The list of model types: the engines a model can name.
//!
//! [`EngineType`] is the one list of engines. An engine names a method, not
//! a kind: each entry names the model kinds it runs, validates its settings,
//! and says the form of its dynamics ([`DynamicsForm`]) and the meaning of a
//! rate. A model names its engine and the settings of that
//! engine in [`ModelEngine`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The kind of model an engine runs: the kind of its structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// Amounts per compartment: bins expanded over stratifications.
    Aggregate,
    /// The model kind of the tests (`types::test_kind`).
    #[cfg(feature = "test-kind")]
    Test,
}

impl ModelKind {
    /// Every kind.
    pub const ALL: &'static [Self] = &[
        Self::Aggregate,
        #[cfg(feature = "test-kind")]
        Self::Test,
    ];

    /// Name of the kind, as the `kind` of a model structure.
    pub fn name(self) -> &'static str {
        match self {
            Self::Aggregate => "aggregate",
            #[cfg(feature = "test-kind")]
            Self::Test => "test",
        }
    }

    /// The kind with a name.
    ///
    /// # Errors
    ///
    /// Returns a message with every name when no kind has it.
    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|kind| kind.name() == name)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|kind| kind.name()).collect();
                format!("Unknown model kind '{name}'; the kinds are {names:?}")
            })
    }

    /// The types of value of the kind that calibration may set, besides
    /// parameters and factors.
    pub fn value_types(self) -> &'static [ModelValueType] {
        match self {
            Self::Aggregate => &[ModelValueType::InitialCondition],
            #[cfg(feature = "test-kind")]
            Self::Test => &[],
        }
    }
}

/// A type of value that calibration sets in a model, besides its parameters
/// and the factors of its observations. Each model kind accepts some of them
/// ([`ModelKind::value_types`]), and its engine resolves and writes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelValueType {
    /// A value of the state the model starts from.
    InitialCondition,
}

impl ModelValueType {
    /// Name of the type, as Python's `CalibrationParameterType` names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::InitialCondition => "initial_condition",
        }
    }
}

/// What the value of a rate expression means to an engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RateMeaning {
    /// The amount (absolute) or the fraction (proportional) a flow moves in one
    /// step.
    PerStep,
    /// The instantaneous rate of a flow, per unit of the axis.
    Continuous,
}

impl RateMeaning {
    /// Name of the meaning, as Python shows it.
    pub fn name(self) -> &'static str {
        match self {
            Self::PerStep => "per_step",
            Self::Continuous => "continuous",
        }
    }

    /// The rate of one outflow whose residence is exponential with a mean
    /// span of `span` steps.
    ///
    /// Per step: `1 - e^(-1/span)`, the fraction an exponential process of
    /// that mean moves in one step, so the bin keeps `e^(-n/span)` at each
    /// step `n`; its mean residence is `1/rate` steps, a little more than
    /// `span`. Continuous: `1/span`.
    ///
    /// # Errors
    ///
    /// Returns a message when `span` is not positive and finite.
    pub fn rate_from_mean_span(self, span: f64) -> Result<f64, String> {
        if !(span.is_finite() && span > 0.0) {
            return Err(format!(
                "The mean span must be positive and finite, got {span} steps"
            ));
        }
        Ok(match self {
            Self::PerStep => -(-1.0 / span).exp_m1(),
            Self::Continuous => 1.0 / span,
        })
    }

    /// The rate of one outflow that moves `probability` of the bin over
    /// `period` steps.
    ///
    /// Per step: `1 - (1 - p)^(1/period)`, so the bin keeps `(1 - rate)^period
    /// = 1 - p`. Continuous: `-ln(1 - p) / period`, so it keeps
    /// `e^(-rate * period) = 1 - p`.
    ///
    /// # Errors
    ///
    /// Returns a message when `period` is not positive and finite, or when
    /// `probability` is outside `[0, 1]`. A continuous rate needs
    /// `probability < 1`: no finite rate empties a bin.
    pub fn rate_from_probability(self, probability: f64, period: f64) -> Result<f64, String> {
        if !(period.is_finite() && period > 0.0) {
            return Err(format!(
                "The period must be positive and finite, got {period} steps"
            ));
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(format!(
                "The probability must be between 0 and 1, got {probability}"
            ));
        }
        let log_survival = (-probability).ln_1p();
        Ok(match self {
            Self::PerStep => -(log_survival / period).exp_m1(),
            Self::Continuous if probability == 1.0 => {
                return Err(
                    "A continuous rate cannot move a probability of 1: no finite rate \
                     empties a bin"
                        .to_string(),
                );
            }
            Self::Continuous => -log_survival / period,
        })
    }
}

/// How an engine advances the state, as a description of a model shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DynamicsForm {
    /// One equation per state component, applied once per step.
    DifferenceEquation,
    /// One equation per state component, integrated in continuous time.
    DifferentialEquation,
    /// An event table: each event, its change of the state and its
    /// propensity.
    StochasticEvents,
}

impl DynamicsForm {
    /// Every form.
    pub const ALL: &'static [Self] = &[
        Self::DifferenceEquation,
        Self::DifferentialEquation,
        Self::StochasticEvents,
    ];

    /// Name of the form, as Python shows it.
    pub fn name(self) -> &'static str {
        match self {
            Self::DifferenceEquation => "difference_equation",
            Self::DifferentialEquation => "differential_equation",
            Self::StochasticEvents => "stochastic_events",
        }
    }

    /// How the engine advances, in words: the `schedule` section of a
    /// description.
    pub fn schedule(self) -> &'static str {
        match self {
            Self::DifferenceEquation => {
                "Each step applies every flow once, computed from the state at the start of \
                 the step: X(n + 1) = X(n) + sum of the flows. The impulses of a step act \
                 first, and its flows run from the new state."
            }
            Self::DifferentialEquation => {
                "The solver integrates dX/dt = sum of the flows in continuous time: a rate \
                 is an instantaneous rate per unit of the axis. The integration stops and \
                 restarts at each impulse and at each position where a rate jumps."
            }
            Self::StochasticEvents => {
                "Events happen one at a time, at random times: each event changes the \
                 state by its change, with a probability per unit of the axis given by its \
                 propensity."
            }
        }
    }

    /// Left-hand side of the equation of a variable, in plain text or in
    /// LaTeX, or `None` when the form has no equations.
    pub fn equation_lhs(self, variable: &str, latex: bool) -> Option<String> {
        Some(match (self, latex) {
            (Self::DifferenceEquation, false) => {
                format!("[{variable}(t+Dt) - {variable}(t)] / Dt")
            }
            (Self::DifferenceEquation, true) => {
                format!("\\frac{{{variable}(t+\\Delta t) - {variable}(t)}}{{\\Delta t}}")
            }
            (Self::DifferentialEquation, false) => format!("d{variable}/dt"),
            (Self::DifferentialEquation, true) => format!("\\frac{{d{variable}}}{{dt}}"),
            (Self::StochasticEvents, _) => return None,
        })
    }
}

/// The list of engines a model can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngineType {
    /// Discrete time: each step applies every flow once (`commol-difference`).
    DifferenceEquations,
    /// Continuous time: the engine integrates `dX/dt = Σ flows` with an ODE
    /// solver (`commol-differential`).
    DifferentialEquations,
    /// Continuous time, at random: whole units move one event at a time, each
    /// flow at its rate as a hazard (Gillespie's direct method, or
    /// tau-leaping; `commol-stochastic`).
    StochasticEvents,
    /// Discrete time, at random: each step draws how many whole units each
    /// flow moves, a proportional rate being a per-step probability (the
    /// chain binomial; `commol-stochastic`).
    StochasticSteps,
    /// The engine of the test model kind: each output decays exponentially
    /// (`types::test_kind`).
    #[cfg(feature = "test-kind")]
    TestEngine,
    /// The stochastic engine of the test model kind: each unit of an output
    /// leaves after an exponential time.
    #[cfg(feature = "test-kind")]
    TestStochasticEngine,
}

impl EngineType {
    /// Every entry of the list.
    pub const ALL: &'static [Self] = &[
        Self::DifferenceEquations,
        Self::DifferentialEquations,
        Self::StochasticEvents,
        Self::StochasticSteps,
        #[cfg(feature = "test-kind")]
        Self::TestEngine,
        #[cfg(feature = "test-kind")]
        Self::TestStochasticEngine,
    ];

    /// Name of the engine, as a model names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::DifferenceEquations => "DifferenceEquations",
            Self::DifferentialEquations => "DifferentialEquations",
            Self::StochasticEvents => "StochasticEvents",
            Self::StochasticSteps => "StochasticSteps",
            #[cfg(feature = "test-kind")]
            Self::TestEngine => "TestEngine",
            #[cfg(feature = "test-kind")]
            Self::TestStochasticEngine => "TestStochasticEngine",
        }
    }

    /// The entry with a name.
    ///
    /// # Errors
    ///
    /// Returns a message with every name of the list when no entry has it.
    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|engine| engine.name() == name)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|engine| engine.name()).collect();
                format!("Unknown engine '{name}'; the engines are {names:?}")
            })
    }

    /// The kinds of model the engine runs.
    pub fn kinds(self) -> &'static [ModelKind] {
        match self {
            Self::DifferenceEquations
            | Self::DifferentialEquations
            | Self::StochasticEvents
            | Self::StochasticSteps => &[ModelKind::Aggregate],
            #[cfg(feature = "test-kind")]
            Self::TestEngine | Self::TestStochasticEngine => &[ModelKind::Test],
        }
    }

    /// Whether the engine draws random numbers: a run depends on its seed
    /// ([`crate::SimulationEngine::set_seed`]).
    pub fn is_stochastic(self) -> bool {
        match self {
            Self::DifferenceEquations | Self::DifferentialEquations => false,
            Self::StochasticEvents | Self::StochasticSteps => true,
            #[cfg(feature = "test-kind")]
            Self::TestEngine => false,
            #[cfg(feature = "test-kind")]
            Self::TestStochasticEngine => true,
        }
    }

    /// Whether the engine runs a model kind.
    pub fn runs(self, kind: ModelKind) -> bool {
        self.kinds().contains(&kind)
    }

    /// The engines that run a model kind, in the order of the list.
    pub fn engines_of(kind: ModelKind) -> Vec<Self> {
        Self::ALL
            .iter()
            .copied()
            .filter(|engine| engine.runs(kind))
            .collect()
    }

    /// Check that the engine runs a model kind.
    ///
    /// # Errors
    ///
    /// Returns a message that names the engine, the kinds it runs, the given
    /// kind and the engines that run it.
    pub fn check_kind(self, kind: ModelKind) -> Result<(), String> {
        if self.runs(kind) {
            return Ok(());
        }
        let kinds: Vec<&str> = self.kinds().iter().map(|kind| kind.name()).collect();
        let engines: Vec<&str> = Self::engines_of(kind)
            .into_iter()
            .map(|engine| engine.name())
            .collect();
        Err(format!(
            "The {} engine runs {} models, not {} models; the engines of {} models are {:?}",
            self.name(),
            kinds.join(" and "),
            kind.name(),
            kind.name(),
            engines
        ))
    }

    /// What the value of a rate expression means to the engine.
    pub fn rate_meaning(self) -> RateMeaning {
        match self {
            Self::DifferenceEquations | Self::StochasticSteps => RateMeaning::PerStep,
            Self::DifferentialEquations | Self::StochasticEvents => RateMeaning::Continuous,
            #[cfg(feature = "test-kind")]
            Self::TestEngine | Self::TestStochasticEngine => RateMeaning::Continuous,
        }
    }

    /// How the engine advances the state: the form of the dynamics a
    /// description of the model shows.
    pub fn dynamics_form(self) -> DynamicsForm {
        match self {
            Self::DifferenceEquations => DynamicsForm::DifferenceEquation,
            Self::DifferentialEquations => DynamicsForm::DifferentialEquation,
            Self::StochasticEvents | Self::StochasticSteps => DynamicsForm::StochasticEvents,
            #[cfg(feature = "test-kind")]
            Self::TestEngine => DynamicsForm::DifferentialEquation,
            #[cfg(feature = "test-kind")]
            Self::TestStochasticEngine => DynamicsForm::StochasticEvents,
        }
    }

    /// How the engine advances, in words: the `schedule` section of a
    /// description. It is the text of its form of dynamics
    /// ([`DynamicsForm::schedule`]), unless the engine says more.
    pub fn schedule(self) -> &'static str {
        match self {
            Self::StochasticEvents => {
                "Events happen one at a time, at random times: each event of a flow moves one \
                 unit from its sources to its targets, with a probability per unit of the axis \
                 (its propensity) of the rate times its source for a proportional rate, or the \
                 rate for an absolute one. The engine stops at each impulse, where the impulses \
                 act first, and at each position where a rate jumps."
            }
            Self::StochasticSteps => {
                "Each step draws at random how many whole units each flow moves, from the state \
                 at the start of the step: the units of a compartment leave by its proportional \
                 flows in one multinomial draw, each rate the probability that one unit leaves \
                 in the step; an absolute flow moves a Poisson number of units of mean its rate, \
                 capped by what its sources still hold. The impulses of a step act first."
            }
            _ => self.dynamics_form().schedule(),
        }
    }

    /// Read and validate the settings of the engine; `None` gives the
    /// defaults.
    ///
    /// # Errors
    ///
    /// Returns a message when the settings are not valid for the engine. When
    /// they are the settings of another engine, the message names both.
    pub fn settings(self, settings: Option<&Value>) -> Result<EngineSettings, String> {
        let Some(value) = settings else {
            return Ok(self.default_settings());
        };
        self.parse_settings(value).map_err(|error| {
            let owners: Vec<&str> = Self::ALL
                .iter()
                .copied()
                .filter(|other| *other != self && other.takes_settings())
                .filter(|other| other.parse_settings(value).is_ok())
                .map(|other| other.name())
                .collect();
            match owners.as_slice() {
                [] => format!("Invalid settings for the {} engine: {error}", self.name()),
                _ => format!(
                    "The settings {value} are settings of the {} engine, not of the {} engine",
                    owners.join(" or "),
                    self.name()
                ),
            }
        })
    }

    fn default_settings(self) -> EngineSettings {
        match self {
            Self::DifferenceEquations => EngineSettings::DifferenceEquations,
            Self::DifferentialEquations => {
                EngineSettings::DifferentialEquations(OdeSolver::default())
            }
            Self::StochasticEvents => {
                EngineSettings::StochasticEvents(StochasticEventsSettings::default())
            }
            Self::StochasticSteps => {
                EngineSettings::StochasticSteps(StochasticStepsSettings::default())
            }
            #[cfg(feature = "test-kind")]
            Self::TestEngine => EngineSettings::TestEngine,
            #[cfg(feature = "test-kind")]
            Self::TestStochasticEngine => EngineSettings::TestStochasticEngine,
        }
    }

    /// Whether the engine has settings of its own.
    fn takes_settings(self) -> bool {
        match self {
            Self::DifferenceEquations => false,
            Self::DifferentialEquations | Self::StochasticEvents | Self::StochasticSteps => true,
            #[cfg(feature = "test-kind")]
            Self::TestEngine | Self::TestStochasticEngine => false,
        }
    }

    fn parse_settings(self, value: &Value) -> Result<EngineSettings, String> {
        let no_settings = |settings: EngineSettings| match value {
            Value::Object(fields) if fields.is_empty() => Ok(settings),
            _ => Err("this engine takes no settings".to_string()),
        };
        match self {
            Self::DifferenceEquations => no_settings(EngineSettings::DifferenceEquations),
            #[cfg(feature = "test-kind")]
            Self::TestEngine => no_settings(EngineSettings::TestEngine),
            #[cfg(feature = "test-kind")]
            Self::TestStochasticEngine => no_settings(EngineSettings::TestStochasticEngine),
            Self::DifferentialEquations => {
                let solver: OdeSolver =
                    serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
                solver.validate()?;
                Ok(EngineSettings::DifferentialEquations(solver))
            }
            Self::StochasticEvents => {
                let settings: StochasticEventsSettings =
                    serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
                settings.validate()?;
                Ok(EngineSettings::StochasticEvents(settings))
            }
            Self::StochasticSteps => {
                let settings: StochasticStepsSettings =
                    serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
                Ok(EngineSettings::StochasticSteps(settings))
            }
        }
    }
}

/// The settings of one engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EngineSettings {
    /// The difference engine has no settings.
    DifferenceEquations,
    /// Solver settings of the differential engine.
    DifferentialEquations(OdeSolver),
    /// Method and rounding of the continuous-time stochastic engine.
    StochasticEvents(StochasticEventsSettings),
    /// Rounding of the discrete-time stochastic engine.
    StochasticSteps(StochasticStepsSettings),
    /// The engine of the test model kind has no settings.
    #[cfg(feature = "test-kind")]
    TestEngine,
    /// The stochastic engine of the test model kind has no settings.
    #[cfg(feature = "test-kind")]
    TestStochasticEngine,
}

impl EngineSettings {
    /// The engine these settings belong to.
    pub fn engine_type(&self) -> EngineType {
        match self {
            Self::DifferenceEquations => EngineType::DifferenceEquations,
            Self::DifferentialEquations(_) => EngineType::DifferentialEquations,
            Self::StochasticEvents(_) => EngineType::StochasticEvents,
            Self::StochasticSteps(_) => EngineType::StochasticSteps,
            #[cfg(feature = "test-kind")]
            Self::TestEngine => EngineType::TestEngine,
            #[cfg(feature = "test-kind")]
            Self::TestStochasticEngine => EngineType::TestStochasticEngine,
        }
    }

    /// How a stochastic engine makes whole the initial amounts that are not,
    /// if its settings name a way; `None` for the other engines.
    pub fn rounding(&self) -> Option<Rounding> {
        match self {
            Self::StochasticEvents(settings) => settings.rounding,
            Self::StochasticSteps(settings) => settings.rounding,
            _ => None,
        }
    }

    /// The settings as JSON, or `None` when the engine has none.
    fn to_value(self) -> Option<Value> {
        match self {
            Self::DifferenceEquations => None,
            Self::DifferentialEquations(solver) => serde_json::to_value(solver).ok(),
            Self::StochasticEvents(settings) => serde_json::to_value(settings).ok(),
            Self::StochasticSteps(settings) => serde_json::to_value(settings).ok(),
            #[cfg(feature = "test-kind")]
            Self::TestEngine | Self::TestStochasticEngine => None,
        }
    }
}

/// The engine of a model and its settings, which always belong to it.
///
/// In JSON it is `{"type": <engine name>, "settings": <object or null>}`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawModelEngine", into = "RawModelEngine")]
pub struct ModelEngine {
    settings: EngineSettings,
}

impl ModelEngine {
    /// An engine with its default settings.
    pub fn new(engine_type: EngineType) -> Self {
        Self {
            settings: engine_type.default_settings(),
        }
    }

    /// An engine with the given settings.
    pub fn with_settings(settings: EngineSettings) -> Self {
        Self { settings }
    }

    /// The engine.
    pub fn engine_type(&self) -> EngineType {
        self.settings.engine_type()
    }

    /// The settings of the engine.
    pub fn settings(&self) -> &EngineSettings {
        &self.settings
    }
}

/// [`ModelEngine`] as it is written in JSON.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelEngine {
    #[serde(rename = "type")]
    engine_type: EngineType,
    #[serde(default)]
    settings: Option<Value>,
}

impl TryFrom<RawModelEngine> for ModelEngine {
    type Error = String;

    fn try_from(raw: RawModelEngine) -> Result<Self, String> {
        let settings = raw.engine_type.settings(raw.settings.as_ref())?;
        Ok(Self { settings })
    }
}

impl From<ModelEngine> for RawModelEngine {
    fn from(engine: ModelEngine) -> Self {
        Self {
            engine_type: engine.engine_type(),
            settings: engine.settings.to_value(),
        }
    }
}

/// Integration method of the differential engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdeMethod {
    /// Explicit Tsitouras 5(4) Runge–Kutta method, for non-stiff models.
    #[default]
    #[serde(rename = "tsit45")]
    Tsit45,
    /// Backward differentiation formulas, for stiff models.
    #[serde(rename = "bdf")]
    Bdf,
}

/// Solver settings of the differential engine.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeSolver {
    /// Integration method.
    #[serde(default)]
    pub method: OdeMethod,
    /// Relative tolerance of each state component.
    #[serde(default = "OdeSolver::default_rtol")]
    pub rtol: f64,
    /// Absolute tolerance of each state component.
    #[serde(default = "OdeSolver::default_atol")]
    pub atol: f64,
}

impl OdeSolver {
    /// Default relative tolerance.
    pub const DEFAULT_RTOL: f64 = 1e-6;
    /// Default absolute tolerance.
    pub const DEFAULT_ATOL: f64 = 1e-6;

    fn default_rtol() -> f64 {
        Self::DEFAULT_RTOL
    }

    fn default_atol() -> f64 {
        Self::DEFAULT_ATOL
    }

    /// Check that both tolerances are positive numbers.
    ///
    /// # Errors
    ///
    /// Returns a message that names the first tolerance that is not.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [("rtol", self.rtol), ("atol", self.atol)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!(
                    "The solver tolerance {name} must be a positive number, got {value}"
                ));
            }
        }
        Ok(())
    }
}

impl Default for OdeSolver {
    fn default() -> Self {
        Self {
            method: OdeMethod::default(),
            rtol: Self::DEFAULT_RTOL,
            atol: Self::DEFAULT_ATOL,
        }
    }
}

/// How a stochastic engine makes whole the initial amounts of a model that
/// are not. There is no default: with small amounts the rounding decides
/// which compartments start with a unit, so it is a choice of the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rounding {
    /// The same whole amounts in every run: each amount rounded down, and the
    /// units left over to the largest remainders, so they sum to the total.
    #[serde(rename = "largest_remainder")]
    LargestRemainder,
    /// A multinomial draw of the total with the fractions of the model in
    /// each run, so the mean start equals the amounts of the model.
    #[serde(rename = "random")]
    Random,
}

impl Rounding {
    /// Every rounding, as a stochastic engine lists them.
    pub const ALL: &'static [Self] = &[Self::LargestRemainder, Self::Random];

    /// Name of the rounding, as the settings write it.
    pub fn name(self) -> &'static str {
        match self {
            Self::LargestRemainder => "largest_remainder",
            Self::Random => "random",
        }
    }
}

/// Method of the continuous-time stochastic engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventMethod {
    /// Gillespie's direct method: one event at a time, exact.
    #[default]
    #[serde(rename = "direct")]
    Direct,
    /// Tau-leaping: a Poisson number of events of each flow in a leap of
    /// length `tau`, an approximation for many events.
    #[serde(rename = "tau_leaping")]
    TauLeaping,
}

/// Settings of the continuous-time stochastic engine.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StochasticEventsSettings {
    /// The method.
    #[serde(default)]
    pub method: EventMethod,
    /// The length of a leap of tau-leaping; `None` chooses each leap as Cao,
    /// Gillespie and Petzold (2006) do, from `epsilon`.
    #[serde(default)]
    pub tau: Option<f64>,
    /// How much a propensity may change in one adaptive leap, relative to
    /// its value.
    #[serde(default = "StochasticEventsSettings::default_epsilon")]
    pub epsilon: f64,
    /// Relative tolerance of the time of an event when a rate reads `t`.
    #[serde(default = "StochasticEventsSettings::default_tolerance")]
    pub tolerance: f64,
    /// How to make whole the initial amounts that are not.
    #[serde(default)]
    pub rounding: Option<Rounding>,
}

impl StochasticEventsSettings {
    /// Default `epsilon` of an adaptive leap (Cao, Gillespie and Petzold 2006).
    pub const DEFAULT_EPSILON: f64 = 0.03;
    /// Default relative tolerance of the time of an event.
    pub const DEFAULT_TOLERANCE: f64 = 1e-8;

    fn default_epsilon() -> f64 {
        Self::DEFAULT_EPSILON
    }

    fn default_tolerance() -> f64 {
        Self::DEFAULT_TOLERANCE
    }

    /// Check the settings.
    ///
    /// # Errors
    ///
    /// Returns a message when `tau` is not a positive number or is given
    /// for the direct method, when `epsilon` is not in `(0, 1)`, or when the
    /// tolerance is not a positive number.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(tau) = self.tau {
            if !(tau.is_finite() && tau > 0.0) {
                return Err(format!("tau must be a positive number, got {tau}"));
            }
            if self.method != EventMethod::TauLeaping {
                return Err(
                    "tau is the length of a leap: it needs the method \"tau_leaping\"".to_string(),
                );
            }
        }
        if !(self.epsilon > 0.0 && self.epsilon < 1.0) {
            return Err(format!(
                "epsilon must be between 0 and 1, got {}",
                self.epsilon
            ));
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(format!(
                "The tolerance must be a positive number, got {}",
                self.tolerance
            ));
        }
        Ok(())
    }
}

impl Default for StochasticEventsSettings {
    fn default() -> Self {
        Self {
            method: EventMethod::default(),
            tau: None,
            epsilon: Self::DEFAULT_EPSILON,
            tolerance: Self::DEFAULT_TOLERANCE,
            rounding: None,
        }
    }
}

/// Settings of the discrete-time stochastic engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StochasticStepsSettings {
    /// How to make whole the initial amounts that are not.
    #[serde(default)]
    pub rounding: Option<Rounding>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_entry_has_a_unique_name_that_reads_back() {
        for &engine in EngineType::ALL {
            assert_eq!(EngineType::from_name(engine.name()), Ok(engine));
            let json = serde_json::to_value(engine).unwrap();
            assert_eq!(json, json!(engine.name()));
        }
        let error = EngineType::from_name("Stochastic").unwrap_err();
        assert!(error.contains("DifferenceEquations") && error.contains("DifferentialEquations"));
    }

    #[test]
    fn every_entry_names_its_kinds_and_rate_meaning() {
        for &engine in EngineType::ALL {
            assert!(!engine.kinds().is_empty(), "{engine:?} runs no kind");
        }
        assert_eq!(
            EngineType::DifferenceEquations.kinds(),
            &[ModelKind::Aggregate]
        );
        assert_eq!(
            EngineType::DifferentialEquations.kinds(),
            &[ModelKind::Aggregate]
        );
        assert_eq!(
            EngineType::engines_of(ModelKind::Aggregate),
            vec![
                EngineType::DifferenceEquations,
                EngineType::DifferentialEquations,
                EngineType::StochasticEvents,
                EngineType::StochasticSteps,
            ]
        );
        assert_eq!(
            EngineType::StochasticSteps.rate_meaning(),
            RateMeaning::PerStep
        );
        assert_eq!(
            EngineType::StochasticEvents.rate_meaning(),
            RateMeaning::Continuous
        );
        assert_eq!(
            EngineType::DifferenceEquations.rate_meaning(),
            RateMeaning::PerStep
        );
        assert_eq!(
            EngineType::DifferentialEquations.rate_meaning(),
            RateMeaning::Continuous
        );
    }

    #[test]
    fn a_rate_from_a_mean_span_keeps_the_exponential_survival() {
        let per_step = RateMeaning::PerStep.rate_from_mean_span(10.0).unwrap();
        assert!((per_step - (1.0 - (-0.1_f64).exp())).abs() < 1e-15);
        assert!((1.0 / per_step - 10.508).abs() < 1e-3);
        assert_eq!(RateMeaning::Continuous.rate_from_mean_span(10.0), Ok(0.1));
        for span in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(RateMeaning::PerStep.rate_from_mean_span(span).is_err());
        }
    }

    #[test]
    fn a_rate_from_a_probability_moves_it_over_the_period() {
        let per_step = RateMeaning::PerStep
            .rate_from_probability(0.3, 7.0)
            .unwrap();
        assert!(((1.0 - per_step).powf(7.0) - 0.7).abs() < 1e-14);
        let continuous = RateMeaning::Continuous
            .rate_from_probability(0.3, 7.0)
            .unwrap();
        assert!(((-continuous * 7.0).exp() - 0.7).abs() < 1e-14);
        assert_eq!(
            RateMeaning::PerStep.rate_from_probability(1.0, 2.0),
            Ok(1.0)
        );
        assert!(
            RateMeaning::Continuous
                .rate_from_probability(1.0, 2.0)
                .unwrap_err()
                .contains("probability of 1")
        );
        assert!(
            RateMeaning::PerStep
                .rate_from_probability(1.5, 2.0)
                .is_err()
        );
        assert!(
            RateMeaning::PerStep
                .rate_from_probability(0.5, 0.0)
                .is_err()
        );
    }

    #[test]
    fn every_entry_gives_a_form_of_its_dynamics() {
        for &engine in EngineType::ALL {
            let form = engine.dynamics_form();
            assert!(DynamicsForm::ALL.contains(&form));
            assert!(!form.schedule().is_empty());
        }
        assert_eq!(
            EngineType::DifferenceEquations.dynamics_form(),
            DynamicsForm::DifferenceEquation
        );
        assert_eq!(
            EngineType::DifferentialEquations.dynamics_form(),
            DynamicsForm::DifferentialEquation
        );
    }

    #[test]
    fn only_the_stochastic_engines_draw_random_numbers() {
        assert!(!EngineType::DifferenceEquations.is_stochastic());
        assert!(!EngineType::DifferentialEquations.is_stochastic());
        assert!(EngineType::StochasticEvents.is_stochastic());
        assert!(EngineType::StochasticSteps.is_stochastic());
        for engine in [EngineType::StochasticEvents, EngineType::StochasticSteps] {
            assert_eq!(engine.dynamics_form(), DynamicsForm::StochasticEvents);
            assert_ne!(
                engine.schedule(),
                DynamicsForm::DifferenceEquation.schedule()
            );
        }
    }

    #[test]
    fn stochastic_settings_read_back_and_name_their_rounding() {
        let engine: ModelEngine = serde_json::from_value(json!({
            "type": "StochasticEvents",
            "settings": {"method": "tau_leaping", "tau": 0.5, "rounding": "random"}
        }))
        .unwrap();
        assert!(matches!(
            engine.settings(),
            EngineSettings::StochasticEvents(settings)
                if settings.method == EventMethod::TauLeaping && settings.tau == Some(0.5)
        ));
        assert_eq!(engine.settings().rounding(), Some(Rounding::Random));
        let again: ModelEngine =
            serde_json::from_value(serde_json::to_value(engine).unwrap()).unwrap();
        assert_eq!(again, engine);
        let steps_engine: ModelEngine = serde_json::from_value(json!({
            "type": "StochasticSteps",
            "settings": {"rounding": "random"}
        }))
        .unwrap();
        let steps_again: ModelEngine =
            serde_json::from_value(serde_json::to_value(steps_engine).unwrap()).unwrap();
        assert_eq!(steps_again, steps_engine);
        assert_eq!(
            ModelEngine::new(EngineType::DifferenceEquations)
                .settings()
                .rounding(),
            None
        );

        let steps = EngineType::StochasticSteps
            .settings(Some(&json!({"rounding": "largest_remainder"})))
            .unwrap();
        assert_eq!(steps.rounding(), Some(Rounding::LargestRemainder));
        assert_eq!(
            EngineType::StochasticSteps
                .settings(None)
                .unwrap()
                .rounding(),
            None
        );
    }

    #[test]
    fn a_leap_needs_tau_leaping_and_a_positive_length() {
        for settings in [
            json!({"tau": 0.5}),
            json!({"method": "tau_leaping", "tau": 0.0}),
            json!({"epsilon": 1.5}),
            json!({"tolerance": -1.0}),
        ] {
            let error = EngineType::StochasticEvents
                .settings(Some(&settings))
                .unwrap_err();
            assert!(error.contains("StochasticEvents"), "{error}");
        }
    }

    #[test]
    fn a_form_with_equations_gives_their_left_hand_side() {
        for &form in DynamicsForm::ALL {
            for latex in [false, true] {
                match form.equation_lhs("A", latex) {
                    Some(lhs) => assert!(lhs.contains('A'), "{lhs}"),
                    None => assert_eq!(form, DynamicsForm::StochasticEvents),
                }
            }
        }
        assert_eq!(
            DynamicsForm::DifferentialEquation.equation_lhs("A", false),
            Some("dA/dt".to_string())
        );
        assert_eq!(
            DynamicsForm::DifferenceEquation.equation_lhs("A", false),
            Some("[A(t+Dt) - A(t)] / Dt".to_string())
        );
    }

    #[test]
    fn every_entry_takes_its_defaults() {
        for &engine in EngineType::ALL {
            let settings = engine.settings(None).unwrap();
            assert_eq!(settings.engine_type(), engine);
        }
    }

    #[test]
    fn every_entry_rejects_wrong_settings() {
        let wrong = json!({"no_such_setting": 1});
        for &engine in EngineType::ALL {
            let error = engine.settings(Some(&wrong)).unwrap_err();
            assert!(error.contains(engine.name()), "{error}");
        }
    }

    #[test]
    fn settings_of_another_engine_name_both_engines() {
        let ode = json!({"method": "bdf", "rtol": 1e-8, "atol": 1e-8});
        let error = EngineType::DifferenceEquations
            .settings(Some(&ode))
            .unwrap_err();
        assert!(error.contains("DifferentialEquations"), "{error}");
        assert!(error.contains("DifferenceEquations"), "{error}");
    }

    #[test]
    fn an_engine_rejects_a_model_of_another_kind() {
        assert_eq!(
            EngineType::DifferenceEquations.check_kind(ModelKind::Aggregate),
            Ok(())
        );
        for &kind in ModelKind::ALL {
            for &engine in EngineType::ALL {
                if engine.runs(kind) {
                    continue;
                }
                let error = engine.check_kind(kind).unwrap_err();
                assert!(error.contains(engine.name()), "{error}");
                for own in engine.kinds() {
                    assert!(error.contains(own.name()), "{error}");
                }
                assert!(error.contains(kind.name()), "{error}");
                for other in EngineType::engines_of(kind) {
                    assert!(error.contains(other.name()), "{error}");
                }
            }
        }
    }

    #[test]
    fn every_kind_has_a_unique_name_that_reads_back() {
        for &kind in ModelKind::ALL {
            assert_eq!(ModelKind::from_name(kind.name()), Ok(kind));
        }
        let error = ModelKind::from_name("network").unwrap_err();
        assert!(error.contains("aggregate"), "{error}");
        assert_eq!(
            ModelKind::Aggregate.value_types(),
            &[ModelValueType::InitialCondition]
        );
    }

    #[test]
    fn a_tolerance_must_be_positive() {
        let error = EngineType::DifferentialEquations
            .settings(Some(&json!({"rtol": 0.0})))
            .unwrap_err();
        assert!(error.contains("rtol"), "{error}");
    }

    #[test]
    fn a_model_engine_reads_back_with_its_settings() {
        let engine: ModelEngine = serde_json::from_value(json!({
            "type": "DifferentialEquations",
            "settings": {"method": "bdf", "rtol": 1e-8}
        }))
        .unwrap();
        assert!(matches!(
            engine.settings(),
            EngineSettings::DifferentialEquations(solver)
                if solver.method == OdeMethod::Bdf && solver.atol == OdeSolver::DEFAULT_ATOL
        ));
        let again: ModelEngine =
            serde_json::from_value(serde_json::to_value(engine).unwrap()).unwrap();
        assert_eq!(again, engine);

        let difference = ModelEngine::new(EngineType::DifferenceEquations);
        assert_eq!(
            serde_json::to_value(difference).unwrap(),
            json!({"type": "DifferenceEquations", "settings": null})
        );
    }

    #[cfg(feature = "test-kind")]
    #[test]
    fn a_test_engine_has_no_settings() {
        let engine = ModelEngine::new(EngineType::TestEngine);

        assert_eq!(
            serde_json::to_value(engine).unwrap(),
            json!({"type": "TestEngine", "settings": null})
        );
    }

    #[test]
    fn a_model_engine_rejects_settings_of_another_engine() {
        let error = serde_json::from_value::<ModelEngine>(json!({
            "type": "DifferenceEquations",
            "settings": {"method": "bdf"}
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("DifferentialEquations"), "{error}");
    }
}
