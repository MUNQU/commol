//! The settings of the profile likelihood: the threshold, the predictions
//! and the two methods.

use commol_calibration::ranges::{AT, SCALE, STARTS, WINDOW};
use commol_calibration::{Observation, ObservedDataPoint};
use commol_core::{CONFIDENCE_LEVEL, Rule, checked};
use serde::{Deserialize, Serialize};

use super::resampling::Bootstrap;
use crate::ranges::MAX_STEPS;

/// The rule of a threshold, as the Python class `Threshold` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdKind {
    /// `½ χ²₁(confidence_level)` on `−log L`.
    ChiSquare,
    /// Beale's threshold of a sum of squares.
    Beale,
    /// The threshold a bootstrap calibrates.
    BootstrapCalibrated,
}

/// The threshold of a profile, as the Python class `Threshold` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct Threshold {
    /// The rule.
    pub kind: ThresholdKind,
    /// The level of each pointwise interval.
    pub confidence_level: f64,
    /// The bootstrap of a calibrated threshold, and of no other.
    pub bootstrap: Option<Bootstrap>,
}

checked!(
    Threshold,
    "Threshold",
    [Rule::Real(CONFIDENCE_LEVEL)],
    check = Threshold::check
);

impl Threshold {
    /// A bootstrap goes with `"bootstrap_calibrated"`, and only with it.
    fn check(&self) -> Result<(), String> {
        if (self.kind == ThresholdKind::BootstrapCalibrated) == self.bootstrap.is_some() {
            Ok(())
        } else {
            Err(
                "A bootstrap calibrates the threshold 'bootstrap_calibrated' and no \
                 other; build thresholds with Threshold.chi_square, Threshold.beale \
                 or Threshold.bootstrap_calibrated."
                    .to_string(),
            )
        }
    }
}

/// A prediction whose profile is traced, as the Python class
/// `PredictionSpec` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct PredictionSpec {
    /// The position of the prediction.
    pub at: f64,
    /// The output predicted, or None with `outputs`.
    pub output: Option<String>,
    /// The outputs summed, or None with `output`.
    pub outputs: Option<Vec<String>>,
    /// The label of the prediction; empty for the name of `output`.
    pub label: String,
    /// The width of the window of an increment, or None for a level.
    pub window: Option<f64>,
    /// The factor that multiplies the prediction, or None.
    pub factor_id: Option<String>,
    /// The known number that multiplies the prediction, with the factor, as
    /// the scale of an observation does.
    pub scale: f64,
}

checked!(
    PredictionSpec,
    "PredictionSpec",
    [Rule::Real(AT), Rule::Real(SCALE), Rule::Real(WINDOW)],
    check = |spec: &PredictionSpec| spec.observation().check()
);

impl PredictionSpec {
    /// The observation that reads this prediction, with no value; an empty
    /// label is the name of `output`, as an observation of the Python API
    /// names its series.
    fn observation(&self) -> Observation {
        let label = match (self.label.is_empty(), &self.output) {
            (true, Some(output)) => output.clone(),
            _ => self.label.clone(),
        };
        Observation {
            at: self.at,
            output: self.output.clone(),
            outputs: self.outputs.clone(),
            label,
            value: 0.0,
            weight: 1.0,
            factor_id: self.factor_id.clone(),
            scale: self.scale,
            window: self.window,
            trials: None,
        }
    }

    /// The observed data point that reads this prediction; its value is not
    /// read.
    pub fn point(&self) -> ObservedDataPoint {
        ObservedDataPoint::from(&self.observation())
    }
}

/// The profile likelihood of parameters, as the Python class
/// `ProfileLikelihood` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct ProfileLikelihoodSettings {
    /// The parameters profiled, by name.
    pub parameters: Vec<String>,
    /// The threshold of the intervals.
    pub threshold: Threshold,
    /// The most points on each side of a profile.
    pub max_steps: usize,
    /// Starts of the fit.
    pub starts: usize,
}

checked!(
    ProfileLikelihoodSettings,
    "ProfileLikelihood",
    [Rule::Count(MAX_STEPS), Rule::Count(STARTS)],
    check = ProfileLikelihoodSettings::check
);

impl ProfileLikelihoodSettings {
    /// Check that there is a parameter to profile, and that each is
    /// profiled once.
    ///
    /// # Errors
    ///
    /// Returns a message that names the parameter listed twice.
    pub fn check(&self) -> Result<(), String> {
        if self.parameters.is_empty() {
            return Err("Give at least one parameter to profile".to_string());
        }
        match repeated(&self.parameters) {
            Some(name) => Err(format!(
                "The parameter '{name}' is listed twice: each parameter is profiled once"
            )),
            None => Ok(()),
        }
    }
}

/// The profile likelihood of predictions, as the Python class
/// `PredictionProfile` writes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", deny_unknown_fields)]
pub struct PredictionProfileSettings {
    /// The predictions profiled.
    pub specs: Vec<PredictionSpec>,
    /// The threshold of the intervals.
    pub threshold: Threshold,
    /// The most points on each side of a profile.
    pub max_steps: usize,
    /// Starts of the fit.
    pub starts: usize,
}

checked!(
    PredictionProfileSettings,
    "PredictionProfile",
    [Rule::Count(MAX_STEPS), Rule::Count(STARTS)],
    check = PredictionProfileSettings::check
);

impl PredictionProfileSettings {
    /// Check that there is a prediction to profile, and that each has a
    /// label of its own: the label of an observation that reads it.
    ///
    /// # Errors
    ///
    /// Returns a message that names the label two predictions share.
    pub fn check(&self) -> Result<(), String> {
        if self.specs.is_empty() {
            return Err("Give at least one prediction to profile in specs".to_string());
        }
        let labels: Vec<String> = self
            .specs
            .iter()
            .map(|spec| spec.observation().label)
            .collect();
        match repeated(&labels) {
            Some(label) => Err(format!(
                "Two predictions have the label '{label}': give each a label of its own"
            )),
            None => Ok(()),
        }
    }
}

/// The first name of `names` that an earlier one repeats.
fn repeated(names: &[String]) -> Option<&String> {
    names
        .iter()
        .enumerate()
        .find(|(index, name)| names[..*index].contains(name))
        .map(|(_, name)| name)
}
