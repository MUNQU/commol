//! The likelihood of the observations: one observation model per series,
//! which says how an observed value scatters around its predicted mean.
//!
//! The observation operator gives the mean of each observation (its sums,
//! windows and factors); a model only adds the scatter. The log-likelihood
//! is `Σ wᵢ · log p(yᵢ | μᵢ)`: a weight is a power of its term, so a weight
//! of 2 counts the observation twice, as it does in every loss.
//!
//! | Module | Holds |
//! |---|---|
//! | [`density`] | [`Noise`]: a model with numbers for its noise, and its log density |

pub mod density;

use std::collections::HashMap;

pub use density::{Center, Noise, log_location};

use commol_core::Real;
use serde::{Deserialize, Serialize};

use crate::feasibility::Violation;
use crate::ranges::{DF, LOG_NORMAL_SIGMA, OVERDISPERSION, SD, STUDENT_T_SCALE};
use crate::types::{CalibrationParameter, CalibrationParameterType, ObservedDataPoint};

/// A noise argument of an observation model: a number, or the id of a noise
/// parameter that is calibrated. JSON writes it as a number or a string; a
/// string is always an id, since no argument is a number that is not finite.
#[derive(Debug, Clone, PartialEq)]
pub enum Argument {
    /// A fixed number.
    Value(f64),
    /// The id of a [`CalibrationParameterType::Noise`] parameter.
    Parameter(String),
}

impl Serialize for Argument {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Value(value) => serializer.serialize_f64(*value),
            Self::Parameter(id) => serializer.serialize_str(id),
        }
    }
}

impl<'de> Deserialize<'de> for Argument {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use commol_core::json::Value;
        use serde::de::Error;
        match Value::deserialize(deserializer)? {
            Value::Int(value) => Ok(Self::Value(value as f64)),
            Value::Float(value) => Ok(Self::Value(value)),
            Value::String(id) => Ok(Self::Parameter(id)),
            _ => Err(D::Error::custom(
                "a noise argument is a number or the id of a noise parameter",
            )),
        }
    }
}

/// How the observed values of one series scatter around their predicted
/// mean `μ`. The Python API writes it with its `kind`: `"normal"`,
/// `"log_normal"`, `"student_t"`, `"poisson"`, `"negative_binomial"` or
/// `"binomial"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationModel {
    /// Normal of mean `μ` and standard deviation `sd`.
    Normal {
        /// The standard deviation, above 0.
        sd: Argument,
    },
    /// Log-normal whose mean or median is `μ`; `sigma` is the standard
    /// deviation of `log Y`.
    LogNormal {
        /// The standard deviation of `log Y`, above 0.
        sigma: Argument,
        /// Whether `μ` is the mean or the median.
        center: Center,
    },
    /// Student-t of center `μ`.
    StudentT {
        /// The degrees of freedom, above 0.
        df: Argument,
        /// The scale, above 0.
        scale: Argument,
    },
    /// Poisson of mean `μ`.
    Poisson,
    /// Negative binomial of mean `μ` and variance `μ + φμ²`.
    NegativeBinomial {
        /// The overdispersion `φ`, at least 0; 0 is the Poisson.
        overdispersion: Argument,
    },
    /// Binomial of mean `μ`: `trials` for every observation of the series,
    /// or `None` to read [`ObservedDataPoint::trials`] of each.
    Binomial {
        /// The trials of the series, or `None`.
        trials: Option<u64>,
    },
}

commol_core::tagged!(ObservationModel, ranges = {
    "normal" => crate::ranges::NORMAL_OBSERVATIONS,
    "log_normal" => crate::ranges::LOG_NORMAL_OBSERVATIONS,
    "student_t" => crate::ranges::STUDENT_T_OBSERVATIONS,
    "negative_binomial" => crate::ranges::NEGATIVE_BINOMIAL_OBSERVATIONS,
    "binomial" => crate::ranges::BINOMIAL_OBSERVATIONS,
});

/// A noise argument resolved against the parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Slot {
    Value(f64),
    Parameter(usize),
}

impl Slot {
    fn at(self, values: &[f64]) -> f64 {
        match self {
            Self::Value(value) => value,
            Self::Parameter(index) => values[index],
        }
    }

    /// The value the bounds allow at their `lower` or upper end.
    fn end(self, bounds: &[(f64, f64)], lower: bool) -> f64 {
        match self {
            Self::Value(value) => value,
            Self::Parameter(index) if lower => bounds[index].0,
            Self::Parameter(index) => bounds[index].1,
        }
    }
}

/// An observation model resolved against the parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Resolved {
    Normal { sd: Slot },
    LogNormal { sigma: Slot, center: Center },
    StudentT { df: Slot, scale: Slot },
    Poisson,
    NegativeBinomial { overdispersion: Slot },
    Binomial { trials: Option<u64> },
}

impl Resolved {
    fn name(&self) -> &'static str {
        match self {
            Self::Normal { .. } => "Normal",
            Self::LogNormal { .. } => "LogNormal",
            Self::StudentT { .. } => "StudentT",
            Self::Poisson => "Poisson",
            Self::NegativeBinomial { .. } => "NegativeBinomial",
            Self::Binomial { .. } => "Binomial",
        }
    }
}

/// Which model an observation reads, and the trials it brings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Term {
    model: usize,
    trials: Option<u64>,
}

/// The derivatives of one observation's term of the objective with respect
/// to its arguments: its mean `μ` first, then the noise arguments of its
/// observation model. A loss has the mean only.
#[derive(Debug, Clone, PartialEq)]
pub struct TermDerivatives {
    /// The parameter each noise argument reads, `None` for a number.
    pub parameters: Vec<Option<usize>>,
    /// The first derivatives of the term over `(μ, noise arguments…)`: its
    /// score, whose outer products the sandwich sums.
    pub gradient: Vec<f64>,
    /// The expected second derivatives of the term, row-major over
    /// `(μ, noise arguments…)`.
    pub curvature: Vec<f64>,
}

/// The likelihood of the observations of a problem.
#[derive(Debug, Clone)]
pub struct Likelihood {
    /// The label and model of each series.
    series: Vec<(String, Resolved)>,
    /// The bounds of every calibration parameter, in problem order.
    bounds: Vec<(f64, f64)>,
    terms: Vec<Term>,
    observed: Vec<f64>,
    weights: Vec<f64>,
}

impl Likelihood {
    /// The likelihood of `observed_data` under `models`, one per series
    /// label, whose noise arguments name `parameters`.
    ///
    /// # Errors
    ///
    /// Returns a message when a series has no model or a model names no
    /// series, an argument is not valid or names a parameter that is not a
    /// noise parameter or whose bounds leave its range, a noise parameter
    /// is named by no model, a binomial series has trials on its model and
    /// on an observation or on neither, an observation of another series has
    /// trials, or an observed value is outside the support of its model.
    pub fn new(
        models: &[(String, ObservationModel)],
        observed_data: &[ObservedDataPoint],
        parameters: &[CalibrationParameter],
    ) -> Result<Self, String> {
        let index_of: HashMap<&str, usize> = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.id.as_str(), index))
            .collect();
        let mut named = vec![false; parameters.len()];
        let mut series = Vec::with_capacity(models.len());
        for (label, model) in models {
            if series
                .iter()
                .any(|(other, _): &(String, Resolved)| other == label)
            {
                return Err(format!("The series '{label}' has two observation models"));
            }
            if !observed_data.iter().any(|obs| &obs.label == label) {
                return Err(format!(
                    "The observation model of '{label}' names a series that no observation has"
                ));
            }
            let mut slot = |argument: &Argument, rule: Real| {
                resolve(label, argument, rule, parameters, &index_of, &mut named)
            };
            let resolved = match model {
                ObservationModel::Normal { sd } => Resolved::Normal { sd: slot(sd, SD)? },
                ObservationModel::LogNormal { sigma, center } => Resolved::LogNormal {
                    sigma: slot(sigma, LOG_NORMAL_SIGMA)?,
                    center: *center,
                },
                ObservationModel::StudentT { df, scale } => Resolved::StudentT {
                    df: slot(df, DF)?,
                    scale: slot(scale, STUDENT_T_SCALE)?,
                },
                ObservationModel::Poisson => Resolved::Poisson,
                ObservationModel::NegativeBinomial { overdispersion } => {
                    Resolved::NegativeBinomial {
                        overdispersion: slot(overdispersion, OVERDISPERSION)?,
                    }
                }
                ObservationModel::Binomial { trials } => Resolved::Binomial { trials: *trials },
            };
            series.push((label.clone(), resolved));
        }
        if let Some(parameter) = parameters
            .iter()
            .zip(&named)
            .find(|&(parameter, &named)| {
                parameter.parameter_type == CalibrationParameterType::Noise && !named
            })
            .map(|(parameter, _)| parameter)
        {
            return Err(unnamed_noise(&parameter.id));
        }
        let likelihood = Self {
            series,
            bounds: parameters
                .iter()
                .map(|p| (p.min_bound, p.max_bound))
                .collect(),
            terms: Vec::new(),
            observed: Vec::new(),
            weights: Vec::new(),
        };
        likelihood.with_data(observed_data)
    }

    /// The likelihood of other observed values and weights of the same
    /// observations: the same models and parameters.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], for the observations.
    pub fn with_data(&self, observed_data: &[ObservedDataPoint]) -> Result<Self, String> {
        let terms = self.terms(observed_data)?;
        for (obs, &term) in observed_data.iter().zip(&terms) {
            self.check_value(obs, term)?;
        }
        Ok(Self {
            terms,
            observed: observed_data.iter().map(|obs| obs.value).collect(),
            weights: observed_data.iter().map(|obs| obs.weight).collect(),
            ..self.clone()
        })
    }

    /// The term of each of `observations`, which need not be the problem's:
    /// the model of its series and its trials.
    ///
    /// # Errors
    ///
    /// Returns a message that names the observation when its series has no
    /// model, a binomial series has trials on its model and on the
    /// observation or on neither, or another series has trials.
    pub fn terms(&self, observations: &[ObservedDataPoint]) -> Result<Vec<Term>, String> {
        observations
            .iter()
            .map(|obs| {
                let model = self
                    .series
                    .iter()
                    .position(|(label, _)| label == &obs.label)
                    .ok_or_else(|| {
                        format!(
                            "Observation '{}' at {}: the series '{}' has no observation \
                             model; with observation models every observed series needs one",
                            obs.label, obs.at, obs.label
                        )
                    })?;
                let trials = match (self.series[model].1, obs.trials) {
                    (Resolved::Binomial { trials: Some(n) }, Some(_)) => {
                        return Err(format!(
                            "Observation '{}' at {} has trials, and the Binomial model of its \
                             series has trials={n}; give the trials in one place",
                            obs.label, obs.at
                        ));
                    }
                    (Resolved::Binomial { trials: None }, None) => {
                        return Err(format!(
                            "Observation '{}' at {} needs trials: the Binomial model of its \
                             series has none, so it reads them from each observation",
                            obs.label, obs.at
                        ));
                    }
                    (Resolved::Binomial { trials: Some(n) }, None) => Some(n),
                    (Resolved::Binomial { trials: None }, Some(n)) => Some(n),
                    (resolved, Some(_)) => {
                        return Err(format!(
                            "Observation '{}' at {} has trials, but its series has a {} model; \
                             only a Binomial model reads trials",
                            obs.label,
                            obs.at,
                            resolved.name()
                        ));
                    }
                    (_, None) => None,
                };
                Ok(Term { model, trials })
            })
            .collect()
    }

    /// The model of `term` with the noise of the parameter `values`.
    pub fn noise(&self, term: Term, values: &[f64]) -> Noise {
        self.noise_with(term, |slot, _| slot.at(values))
    }

    /// The model of `term` with each noise argument given by `value`, which
    /// also gets whether the argument narrows the model as it falls.
    fn noise_with(&self, term: Term, value: impl Fn(Slot, bool) -> f64) -> Noise {
        match self.series[term.model].1 {
            Resolved::Normal { sd } => Noise::Normal {
                sd: value(sd, true),
            },
            Resolved::LogNormal { sigma, center } => Noise::LogNormal {
                sigma: value(sigma, true),
                center,
            },
            Resolved::StudentT { df, scale } => Noise::StudentT {
                df: value(df, false),
                scale: value(scale, true),
            },
            Resolved::Poisson => Noise::Poisson,
            Resolved::NegativeBinomial { overdispersion } => Noise::NegativeBinomial {
                overdispersion: value(overdispersion, true),
            },
            Resolved::Binomial { .. } => Noise::Binomial {
                trials: term.trials.expect("a binomial term has trials"),
            },
        }
    }

    /// The term of each of the problem's observations.
    pub fn problem_terms(&self) -> &[Term] {
        &self.terms
    }

    /// The parameter each noise argument of the model of `term` reads, in
    /// the order of [`Noise::information`] after the mean: the index of a
    /// noise parameter, or `None` for a number.
    pub fn argument_parameters(&self, term: Term) -> Vec<Option<usize>> {
        let of = |slot: Slot| match slot {
            Slot::Parameter(index) => Some(index),
            Slot::Value(_) => None,
        };
        match self.series[term.model].1 {
            Resolved::Normal { sd } => vec![of(sd)],
            Resolved::LogNormal { sigma, .. } => vec![of(sigma)],
            Resolved::StudentT { df, scale } => vec![of(df), of(scale)],
            Resolved::NegativeBinomial { overdispersion } => vec![of(overdispersion)],
            Resolved::Poisson | Resolved::Binomial { .. } => Vec::new(),
        }
    }

    /// The derivatives of each observation's term of `−log L` at the
    /// parameter `values` and the means `predictions`: its gradient
    /// `−wᵢ s(yᵢ)` ([`Noise::scores`]) and its expected curvature `wᵢ`
    /// times the Fisher information of its model ([`Noise::information`]),
    /// with the parameters its noise arguments read.
    ///
    /// # Errors
    ///
    /// The index of the first observation with weight whose mean lies
    /// outside the means its model allows.
    pub fn term_derivatives(
        &self,
        values: &[f64],
        predictions: &[f64],
    ) -> Result<Vec<TermDerivatives>, usize> {
        self.terms
            .iter()
            .zip(&self.observed)
            .zip(&self.weights)
            .zip(predictions)
            .enumerate()
            .map(|(index, (((&term, &y), &weight), &mean))| {
                let noise = self.noise(term, values);
                let parameters = self.argument_parameters(term);
                if weight == 0.0 {
                    let size = 1 + parameters.len();
                    return Ok(TermDerivatives {
                        parameters,
                        gradient: vec![0.0; size],
                        curvature: vec![0.0; size * size],
                    });
                }
                if noise.mean_outside(mean).is_some() {
                    return Err(index);
                }
                Ok(TermDerivatives {
                    parameters,
                    gradient: noise
                        .scores(y, mean)
                        .into_iter()
                        .map(|score| -weight * score)
                        .collect(),
                    curvature: noise
                        .information(mean)
                        .into_iter()
                        .map(|entry| weight * entry)
                        .collect(),
                })
            })
            .collect()
    }

    fn check_value(&self, obs: &ObservedDataPoint, term: Term) -> Result<(), String> {
        let y = obs.value;
        let noise = self.noise_with(term, |slot, _| slot.end(&self.bounds, true));
        let fails = |needs: &str| {
            Err(format!(
                "Observation '{}' at {} has the value {y}; its {} model needs {needs}",
                obs.label,
                obs.at,
                self.series[term.model].1.name()
            ))
        };
        match noise {
            Noise::Normal { .. } | Noise::StudentT { .. } => Ok(()),
            Noise::LogNormal { .. } if y > 0.0 => Ok(()),
            Noise::LogNormal { .. } => fails("a value above 0"),
            Noise::Poisson | Noise::NegativeBinomial { .. } | Noise::Binomial { .. }
                if !(y >= 0.0 && y.fract() == 0.0) =>
            {
                fails("a whole number, at least 0")
            }
            Noise::Binomial { trials } if y > trials as f64 => {
                fails(&format!("a value of at most its {trials} trials"))
            }
            _ => Ok(()),
        }
    }

    /// The weighted log density of each observation of the problem,
    /// `wᵢ · log p(yᵢ | μᵢ)`, where `predictions` holds `μᵢ`; 0 for an
    /// observation of weight 0.
    ///
    /// # Errors
    ///
    /// [`Violation::MeanOutsideModel`] when a mean of an observation with
    /// weight lies outside its model: `amount` is the sum of the distances.
    pub fn pointwise(&self, values: &[f64], predictions: &[f64]) -> Result<Vec<f64>, Violation> {
        let mut outside = 0.0;
        let mut any = false;
        let terms = self
            .terms
            .iter()
            .zip(&self.observed)
            .zip(&self.weights)
            .zip(predictions)
            .map(|(((&term, &y), &weight), &mean)| {
                if weight == 0.0 {
                    return 0.0;
                }
                let noise = self.noise(term, values);
                match noise.mean_outside(mean) {
                    Some(distance) => {
                        any = true;
                        outside += distance;
                        0.0
                    }
                    None => weight * noise.log_density(y, mean),
                }
            })
            .collect();
        if any {
            Err(Violation::MeanOutsideModel { amount: outside })
        } else {
            Ok(terms)
        }
    }

    /// The log-likelihood `Σ wᵢ · log p(yᵢ | μᵢ)` of the predictions of one
    /// run; `−∞` where an observation has no mass.
    ///
    /// # Errors
    ///
    /// As [`Self::pointwise`].
    pub fn log_likelihood(&self, values: &[f64], predictions: &[f64]) -> Result<f64, Violation> {
        self.pointwise(values, predictions)
            .map(|terms| terms.into_iter().sum())
    }

    /// `Σ wᵢ tᵢ`, where `tᵢ ≥ 0` is `−log p(yᵢ)` at the narrowest noise the
    /// bounds allow and a mean `distance` from `yᵢ`: the scale of the
    /// penalty of an infeasible point. Every `tᵢ` is finite: the far mean
    /// gives every observed value in the support some mass.
    pub(crate) fn far_cost(&self, distance: f64) -> f64 {
        self.terms
            .iter()
            .zip(&self.observed)
            .zip(&self.weights)
            .map(|((&term, &y), &weight)| {
                let narrowest =
                    self.noise_with(term, |slot, narrows| slot.end(&self.bounds, narrows));
                weight * (-narrowest.far_log_density(y, distance)).max(0.0)
            })
            .sum()
    }
}

/// The message of a noise parameter that no observation model names.
pub(crate) fn unnamed_noise(id: &str) -> String {
    format!(
        "Noise parameter '{id}' is named by no observation model; a noise parameter is an \
         argument of the observation model of a series"
    )
}

/// Resolve `argument`, the argument of the model of `label` that `rule`
/// names.
fn resolve(
    label: &str,
    argument: &Argument,
    rule: Real,
    parameters: &[CalibrationParameter],
    index_of: &HashMap<&str, usize>,
    named: &mut [bool],
) -> Result<Slot, String> {
    let name = rule.name();
    match argument {
        Argument::Value(value) => rule
            .check(*value)
            .map(|()| Slot::Value(*value))
            .map_err(|error| format!("The observation model of '{label}': {error}")),
        Argument::Parameter(id) => {
            let &index = index_of.get(id.as_str()).ok_or_else(|| {
                format!(
                    "The observation model of '{label}' names '{id}' as {name}, which is not a \
                     calibration parameter"
                )
            })?;
            let parameter = &parameters[index];
            if parameter.parameter_type != CalibrationParameterType::Noise {
                return Err(format!(
                    "The observation model of '{label}' names '{id}' as {name}, a {:?} \
                     parameter; a noise argument must be a Noise parameter",
                    parameter.parameter_type
                ));
            }
            if rule
                .check(parameter.min_bound)
                .and(rule.check(parameter.max_bound))
                .is_err()
            {
                return Err(format!(
                    "Noise parameter '{id}' is {name} of the observation model of '{label}', \
                     so its bounds must be {}; they are [{}, {}]",
                    rule.range(),
                    parameter.min_bound,
                    parameter.max_bound
                ));
            }
            named[index] = true;
            Ok(Slot::Parameter(index))
        }
    }
}
