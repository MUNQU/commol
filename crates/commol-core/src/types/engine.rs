//! The list of model types: the engines a model can name.
//!
//! [`EngineType`] is the one list of engines. Each entry names the model kind
//! it runs, validates its settings, and says the form of its equations and
//! the meaning of a rate. A model names its engine and the settings of that
//! engine in [`ModelEngine`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The kind of model an engine runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// Bins expanded over stratifications, joined by transitions with rates.
    Compartmental,
}

impl ModelKind {
    /// Name of the kind, as Python shows it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Compartmental => "compartmental",
        }
    }
}

/// What the value of a rate expression means to an engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RateMeaning {
    /// The amount (absolute) or the fraction (per-capita) a flow moves in one
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
}

/// The list of engines a model can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngineType {
    /// Discrete time: each step applies every flow once (`commol-difference`).
    DifferenceEquations,
    /// Continuous time: the engine integrates `dX/dt = Σ flows` with an ODE
    /// solver (`commol-differential`).
    DifferentialEquations,
}

impl EngineType {
    /// Every entry of the list.
    pub const ALL: [Self; 2] = [Self::DifferenceEquations, Self::DifferentialEquations];

    /// Name of the engine, as a model names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::DifferenceEquations => "DifferenceEquations",
            Self::DifferentialEquations => "DifferentialEquations",
        }
    }

    /// The entry with a name.
    ///
    /// # Errors
    ///
    /// Returns a message with every name of the list when no entry has it.
    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|engine| engine.name() == name)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|engine| engine.name()).collect();
                format!("Unknown engine '{name}'; the engines are {names:?}")
            })
    }

    /// The kind of model the engine runs.
    pub fn kind(self) -> ModelKind {
        match self {
            Self::DifferenceEquations | Self::DifferentialEquations => ModelKind::Compartmental,
        }
    }

    /// What the value of a rate expression means to the engine.
    pub fn rate_meaning(self) -> RateMeaning {
        match self {
            Self::DifferenceEquations => RateMeaning::PerStep,
            Self::DifferentialEquations => RateMeaning::Continuous,
        }
    }

    /// Left-hand side of the equation of a variable, in plain text or in
    /// LaTeX.
    pub fn equation_lhs(self, variable: &str, latex: bool) -> String {
        match (self, latex) {
            (Self::DifferenceEquations, false) => {
                format!("[{variable}(t+Dt) - {variable}(t)] / Dt")
            }
            (Self::DifferenceEquations, true) => {
                format!("\\frac{{{variable}(t+\\Delta t) - {variable}(t)}}{{\\Delta t}}")
            }
            (Self::DifferentialEquations, false) => format!("d{variable}/dt"),
            (Self::DifferentialEquations, true) => format!("\\frac{{d{variable}}}{{dt}}"),
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
                .into_iter()
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
        }
    }

    /// Whether the engine has settings of its own.
    fn takes_settings(self) -> bool {
        match self {
            Self::DifferenceEquations => false,
            Self::DifferentialEquations => true,
        }
    }

    fn parse_settings(self, value: &Value) -> Result<EngineSettings, String> {
        match self {
            Self::DifferenceEquations => match value {
                Value::Object(fields) if fields.is_empty() => {
                    Ok(EngineSettings::DifferenceEquations)
                }
                _ => Err("this engine takes no settings".to_string()),
            },
            Self::DifferentialEquations => {
                let solver: OdeSolver =
                    serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
                solver.validate()?;
                Ok(EngineSettings::DifferentialEquations(solver))
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
}

impl EngineSettings {
    /// The engine these settings belong to.
    pub fn engine_type(&self) -> EngineType {
        match self {
            Self::DifferenceEquations => EngineType::DifferenceEquations,
            Self::DifferentialEquations(_) => EngineType::DifferentialEquations,
        }
    }

    /// The settings as JSON, or `None` when the engine has none.
    fn to_value(self) -> Option<Value> {
        match self {
            Self::DifferenceEquations => None,
            Self::DifferentialEquations(solver) => serde_json::to_value(solver).ok(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_entry_has_a_unique_name_that_reads_back() {
        for engine in EngineType::ALL {
            assert_eq!(EngineType::from_name(engine.name()), Ok(engine));
            let json = serde_json::to_value(engine).unwrap();
            assert_eq!(json, json!(engine.name()));
        }
        let error = EngineType::from_name("Stochastic").unwrap_err();
        assert!(error.contains("DifferenceEquations") && error.contains("DifferentialEquations"));
    }

    #[test]
    fn every_entry_names_its_kind_and_rate_meaning() {
        for engine in EngineType::ALL {
            assert_eq!(engine.kind().name(), "compartmental");
        }
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
    fn every_entry_prints_its_equations() {
        for engine in EngineType::ALL {
            for latex in [false, true] {
                assert!(engine.equation_lhs("S", latex).contains('S'));
            }
        }
        assert_eq!(
            EngineType::DifferentialEquations.equation_lhs("S", false),
            "dS/dt"
        );
        assert_eq!(
            EngineType::DifferenceEquations.equation_lhs("S", false),
            "[S(t+Dt) - S(t)] / Dt"
        );
    }

    #[test]
    fn every_entry_takes_its_defaults() {
        for engine in EngineType::ALL {
            let settings = engine.settings(None).unwrap();
            assert_eq!(settings.engine_type(), engine);
        }
    }

    #[test]
    fn every_entry_rejects_wrong_settings() {
        let wrong = json!({"no_such_setting": 1});
        for engine in EngineType::ALL {
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
        let EngineSettings::DifferentialEquations(solver) = engine.settings() else {
            panic!("expected differential settings");
        };
        assert_eq!(solver.method, OdeMethod::Bdf);
        assert_eq!(solver.atol, OdeSolver::DEFAULT_ATOL);
        let again: ModelEngine =
            serde_json::from_value(serde_json::to_value(engine).unwrap()).unwrap();
        assert_eq!(again, engine);

        let difference = ModelEngine::new(EngineType::DifferenceEquations);
        assert_eq!(
            serde_json::to_value(difference).unwrap(),
            json!({"type": "DifferenceEquations", "settings": null})
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
