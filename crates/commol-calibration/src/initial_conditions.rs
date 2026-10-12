//! Application of calibrated values to a simulation engine.
//!
//! This module is the single source of truth for how calibration values reach
//! an engine: how an initial-condition id resolves to compartments, in which
//! order the values are written, and how the remainder keeps the population.
//! The calibration objective, the calibrated predictions, and the model
//! write-back all use it.
//!
//! Initial conditions are written in the same layers as a model definition:
//! bin values first, then stratification-category splits, then exact
//! compartments, and finally the remainder, which absorbs the difference to the
//! population size. Within a layer, values are written in parameter order.

use commol_core::{OutputLabels, SimulationEngine};

use crate::types::{CalibrationParameter, CalibrationParameterType};

/// Separator of a qualified category id, as in `sex:female`.
const CATEGORY_SEPARATOR: char = ':';

/// Relative tolerance under which the initial population is taken to match the
/// population size.
const POPULATION_TOLERANCE: f64 = 1e-9;

/// Compartments an initial-condition value is written to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InitialConditionTarget {
    /// A bin or a bin prefix. The value is the fraction of the population the
    /// covered compartments hold, shared by weights taken from the base
    /// population.
    Aggregate { entries: Vec<(usize, f64)> },
    /// One expanded compartment. The value is its fraction of the population.
    Exact { index: usize },
    /// A category of a binary stratification. The value splits each
    /// `(category, complement)` pair of compartments.
    PairedCategory { pairs: Vec<(usize, usize)> },
}

impl InitialConditionTarget {
    fn aggregate(indices: Vec<usize>, population: &[f64]) -> Self {
        let total: f64 = indices.iter().map(|&idx| population[idx]).sum();
        let entries = indices
            .iter()
            .map(|&idx| {
                let weight = if total > 0.0 {
                    population[idx] / total
                } else {
                    1.0 / indices.len() as f64
                };
                (idx, weight)
            })
            .collect();
        Self::Aggregate { entries }
    }

    /// Compartments whose head count the value sets. Category splits keep the
    /// pair totals, so they cover no population.
    fn covered_population(&self) -> Vec<usize> {
        match self {
            Self::Aggregate { entries } => entries.iter().map(|(idx, _)| *idx).collect(),
            Self::Exact { index } => vec![*index],
            Self::PairedCategory { .. } => Vec::new(),
        }
    }

    /// Layer in which the value is written.
    fn layer(&self) -> u8 {
        match self {
            Self::Aggregate { .. } => 0,
            Self::PairedCategory { .. } => 1,
            Self::Exact { .. } => 2,
        }
    }
}

/// Resolved initial-condition targets of a set of calibration parameters.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InitialConditionPlan {
    /// Target of each parameter, parallel to the parameters. `None` for
    /// parameters that are not initial conditions.
    targets: Vec<Option<InitialConditionTarget>>,
    /// Compartments that absorb the difference to the population size, if
    /// any, with the id they were resolved from.
    remainder: Option<(String, Vec<usize>)>,
}

impl InitialConditionPlan {
    /// Resolve the initial-condition target of every calibration parameter.
    ///
    /// An initial-condition id names, in order of precedence:
    /// - an expanded compartment (`A_g1`);
    /// - a qualified stratification category (`group:g1`);
    /// - a bin or a bin prefix (`A`, or `A_g1` over a deeper stratification);
    /// - a category of exactly one binary stratification (`g1`).
    ///
    /// # Errors
    ///
    /// Returns a message when an id matches none of these, names a category
    /// declared by more than one stratification, or names a category of a
    /// stratification that is not binary for some compartment.
    pub(crate) fn new<E: SimulationEngine>(
        engine: &E,
        parameters: &[CalibrationParameter],
    ) -> Result<Self, String> {
        let compartments = engine.compartments();
        let labels = compartment_labels(engine);
        let population = engine.population();

        let targets = parameters
            .iter()
            .map(|param| match param.parameter_type {
                CalibrationParameterType::InitialCondition => {
                    resolve_target(&param.id, &compartments, &labels, &population).map(Some)
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            targets,
            remainder: None,
        })
    }

    /// Name the compartments that absorb the difference to the population size.
    ///
    /// # Errors
    ///
    /// Returns a message when the id is not a compartment, bin or bin prefix,
    /// or when it covers a compartment that a calibrated bin or compartment
    /// value already sets.
    pub(crate) fn with_remainder<E: SimulationEngine>(
        mut self,
        engine: &E,
        parameters: &[CalibrationParameter],
        remainder_id: &str,
    ) -> Result<Self, String> {
        let compartments = engine.compartments();
        let indices =
            population_indices(remainder_id, &compartment_labels(engine)).ok_or_else(|| {
                format!(
                    "Initial-condition remainder '{}' is not a compartment or a bin \
                 (available: {})",
                    remainder_id,
                    compartments.join(", ")
                )
            })?;

        for (param, target) in parameters.iter().zip(&self.targets) {
            let Some(target) = target else { continue };
            if target
                .covered_population()
                .iter()
                .any(|idx| indices.contains(idx))
            {
                return Err(format!(
                    "Initial-condition remainder '{}' overlaps the calibrated initial \
                     condition '{}'",
                    remainder_id, param.id
                ));
            }
        }

        self.remainder = Some((remainder_id.to_string(), indices));
        Ok(self)
    }

    /// Compartments whose head count a calibrated bin or compartment value, or
    /// the remainder, sets, in engine order. Category splits keep the pair
    /// totals, so they cover none.
    pub(crate) fn covered_population(&self) -> Vec<usize> {
        let mut covered: Vec<usize> = self
            .targets
            .iter()
            .flatten()
            .flat_map(InitialConditionTarget::covered_population)
            .chain(
                self.remainder
                    .iter()
                    .flat_map(|(_, indices)| indices.iter().copied()),
            )
            .collect();
        covered.sort_unstable();
        covered.dedup();
        covered
    }

    /// Write calibration values into an engine.
    ///
    /// Parameters are set by id. Initial conditions are written layer by layer
    /// (bins, category splits, exact compartments) and then the remainder
    /// takes the difference to the population size, keeping its own split.
    /// Scale values are skipped because they apply to observations.
    ///
    /// # Returns
    ///
    /// How far the initial population exceeds the population size, as a
    /// fraction of it: `0.0` for a feasible state. With a remainder, the
    /// remainder is floored at zero and the deficit is reported. Without one,
    /// the excess of the total over the population size is reported.
    ///
    /// # Errors
    ///
    /// Returns a message when the engine rejects a parameter or initial
    /// condition.
    pub(crate) fn apply<E: SimulationEngine>(
        &self,
        engine: &mut E,
        parameters: &[CalibrationParameter],
        population_size: f64,
        values: &[f64],
    ) -> Result<f64, String> {
        for (param, value) in parameters.iter().zip(values) {
            if param.parameter_type == CalibrationParameterType::Parameter {
                engine
                    .set_parameter(&param.id, *value)
                    .map_err(|e| format!("Failed to set parameter '{}': {}", param.id, e))?;
            }
        }

        for layer in 0..3 {
            for ((param, value), target) in parameters.iter().zip(values).zip(&self.targets) {
                let Some(target) = target else { continue };
                if target.layer() == layer {
                    write_target(engine, target, *value, population_size).map_err(|e| {
                        format!("Failed to set initial condition for '{}': {}", param.id, e)
                    })?;
                }
            }
        }

        let population = engine.population();
        let Some((_, indices)) = &self.remainder else {
            let total: f64 = population.iter().sum();
            return Ok((total / population_size - 1.0).max(0.0));
        };

        let others: f64 = population
            .iter()
            .enumerate()
            .filter(|(idx, _)| !indices.contains(idx))
            .map(|(_, value)| value)
            .sum();
        let remainder = population_size - others;
        let deficit = (-remainder / population_size).max(0.0);
        let remainder = remainder.max(0.0);

        let current: f64 = indices.iter().map(|&idx| population[idx]).sum();
        for &idx in indices {
            let share = if current > 0.0 {
                population[idx] / current
            } else {
                1.0 / indices.len() as f64
            };
            engine
                .set_initial_condition(idx, remainder * share)
                .map_err(|e| format!("Failed to set the initial-condition remainder: {}", e))?;
        }
        Ok(deficit)
    }
}

/// Whether an initial-population excess returned by
/// [`InitialConditionPlan::apply`] makes the state infeasible.
pub(crate) fn is_infeasible(excess: f64) -> bool {
    excess > POPULATION_TOLERANCE
}

fn write_target<E: SimulationEngine>(
    engine: &mut E,
    target: &InitialConditionTarget,
    value: f64,
    population_size: f64,
) -> Result<(), String> {
    match target {
        InitialConditionTarget::Aggregate { entries } => {
            for &(idx, weight) in entries {
                engine.set_initial_condition(idx, value * weight * population_size)?;
            }
        }
        InitialConditionTarget::Exact { index } => {
            engine.set_initial_condition(*index, value * population_size)?;
        }
        InitialConditionTarget::PairedCategory { pairs } => {
            let population = engine.population();
            for &(category_idx, complement_idx) in pairs {
                let pair_total = population[category_idx] + population[complement_idx];
                engine.set_initial_condition(category_idx, value * pair_total)?;
                engine.set_initial_condition(complement_idx, (1.0 - value) * pair_total)?;
            }
        }
    }
    Ok(())
}

/// Labels of the compartments of an engine (its first outputs).
fn compartment_labels<E: SimulationEngine>(engine: &E) -> Vec<OutputLabels> {
    let mut labels = engine.output_labels();
    labels.truncate(engine.compartments().len());
    labels
}

/// Indices of the compartments a bin or bin prefix covers.
///
/// A compartment is covered when the id is its bin (`A`) or its bin followed
/// by its leading categories (`A_g1`, `A_g1_s2`). The match is built from the
/// labels, so a bin `A` does not cover the compartments of a bin `A_x`.
fn population_indices(id: &str, labels: &[OutputLabels]) -> Option<Vec<usize>> {
    let indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, label)| label.prefixes().iter().any(|prefix| prefix == id))
        .map(|(idx, _)| idx)
        .collect();
    (!indices.is_empty()).then_some(indices)
}

fn resolve_target(
    id: &str,
    compartments: &[String],
    labels: &[OutputLabels],
    population: &[f64],
) -> Result<InitialConditionTarget, String> {
    if let Some(index) = compartments.iter().position(|name| name == id) {
        return Ok(InitialConditionTarget::Exact { index });
    }
    if let Some((stratification, category)) = id.split_once(CATEGORY_SEPARATOR) {
        return paired_category(stratification, category, labels);
    }
    if let Some(indices) = population_indices(id, labels) {
        return Ok(InitialConditionTarget::aggregate(indices, population));
    }

    let mut stratifications: Vec<&str> = Vec::new();
    for label in labels {
        for (stratification, category) in &label.categories {
            if category == id && !stratifications.contains(&stratification.as_str()) {
                stratifications.push(stratification);
            }
        }
    }
    match stratifications.as_slice() {
        [stratification] => paired_category(stratification, id, labels),
        [] => Err(format!(
            "Invalid bin, compartment or category '{}' for initial condition \
             calibration (available compartments: {})",
            id,
            compartments.join(", ")
        )),
        _ => Err(format!(
            "Initial condition category '{}' is ambiguous: it is declared by \
             stratifications {:?}. Qualify it as {}",
            id,
            stratifications,
            stratifications
                .iter()
                .map(|stratification| format!("'{stratification}{CATEGORY_SEPARATOR}{id}'"))
                .collect::<Vec<_>>()
                .join(" or ")
        )),
    }
}

/// Pairs each compartment of a category with the compartment that differs
/// only in that stratification.
fn paired_category(
    stratification: &str,
    category: &str,
    labels: &[OutputLabels],
) -> Result<InitialConditionTarget, String> {
    let qualified = format!("{stratification}{CATEGORY_SEPARATOR}{category}");
    let mut pairs = Vec::new();
    for (category_idx, label) in labels.iter().enumerate() {
        if label.category(stratification) != Some(category) {
            continue;
        }
        let complements: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate.source == label.source
                    && candidate.categories.len() == label.categories.len()
                    && candidate.categories.iter().zip(&label.categories).all(
                        |((id, value), (other_id, other_value))| {
                            id == other_id
                                && if id == stratification {
                                    value != other_value
                                } else {
                                    value == other_value
                                }
                        },
                    )
            })
            .map(|(idx, _)| idx)
            .collect();
        match complements.as_slice() {
            [complement_idx] => pairs.push((category_idx, *complement_idx)),
            _ => {
                return Err(format!(
                    "Initial condition category '{}' needs a binary stratification: \
                     found {} complementary categories for one of its compartments",
                    qualified,
                    complements.len()
                ));
            }
        }
    }

    if pairs.is_empty() {
        return Err(format!(
            "Initial condition category '{}' matches no compartment",
            qualified
        ));
    }
    Ok(InitialConditionTarget::PairedCategory { pairs })
}

/// Clamp parameter values to their defined bounds.
pub(crate) fn clamp_to_bounds(parameters: &[CalibrationParameter], values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .zip(parameters)
        .map(|(value, param)| value.clamp(param.min_bound, param.max_bound))
        .collect()
}

/// Build an engine clone with calibration values applied exactly as in
/// objective evaluation.
///
/// # Returns
///
/// The configured engine and the values it applied (clamped to their bounds).
///
/// # Errors
///
/// Returns a message when the vector length is wrong, the engine rejects a
/// value, or the initial population exceeds the population size.
pub(crate) fn configure_engine<E: SimulationEngine>(
    base_engine: &E,
    parameters: &[CalibrationParameter],
    plan: &InitialConditionPlan,
    population_size: f64,
    values: &[f64],
) -> Result<(E, Vec<f64>), String> {
    if values.len() != parameters.len() {
        return Err(format!(
            "Expected {} parameters, got {}",
            parameters.len(),
            values.len()
        ));
    }
    let clamped = clamp_to_bounds(parameters, values);
    let mut engine = base_engine.clone();
    engine.reset();
    let excess = plan.apply(&mut engine, parameters, population_size, &clamped)?;
    if is_infeasible(excess) {
        return Err(match &plan.remainder {
            Some((id, _)) => format!(
                "Invalid initial conditions: the calibrated values leave the \
                 remainder '{}' {:.6} of the population below zero",
                id, excess
            ),
            None => format!(
                "Invalid initial conditions: the calibrated values exceed the \
                 population size by {:.6} of it",
                excess
            ),
        });
    }
    Ok((engine, clamped))
}

/// Initial state an engine starts from once calibration values are applied.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedInitialPopulation {
    /// Values the engine applied, parallel to the parameters (clamped to their
    /// bounds).
    pub values: Vec<f64>,
    /// Initial population of each compartment, in engine order.
    pub population: Vec<f64>,
    /// Compartments whose head count a calibrated bin or compartment value, or
    /// the remainder, sets, in engine order.
    pub covered: Vec<usize>,
}

/// Initial population an engine starts from once calibration values are
/// applied, exactly as the calibration objective scores them.
///
/// This is the authority for persisting calibrated initial conditions: a model
/// definition that rebuilds this population reproduces the scored state.
///
/// # Arguments
///
/// * `base_engine` - Engine built from the model definition, before calibration.
/// * `parameters` - Calibration parameters, in problem order.
/// * `remainder` - Compartment or bin that absorbs the difference to the
///   population size, if any.
/// * `values` - Value of each parameter, parallel to `parameters`.
/// * `population_size` - Population the fractions are taken of.
///
/// # Errors
///
/// Returns a message when an initial-condition id or the remainder cannot be
/// resolved, the vector length is wrong, or the initial population exceeds the
/// population size.
pub fn calibrated_initial_population<E: SimulationEngine>(
    base_engine: &E,
    parameters: &[CalibrationParameter],
    remainder: Option<&str>,
    values: &[f64],
    population_size: u64,
) -> Result<CalibratedInitialPopulation, String> {
    let mut plan = InitialConditionPlan::new(base_engine, parameters)?;
    if let Some(remainder) = remainder {
        plan = plan.with_remainder(base_engine, parameters, remainder)?;
    }
    let (engine, values) = configure_engine(
        base_engine,
        parameters,
        &plan,
        population_size as f64,
        values,
    )?;
    Ok(CalibratedInitialPopulation {
        values,
        population: engine.population(),
        covered: plan.covered_population(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(entries: &[(&str, &[(&str, &str)])]) -> Vec<OutputLabels> {
        entries
            .iter()
            .map(|(bin, categories)| OutputLabels {
                source: bin.to_string(),
                categories: categories
                    .iter()
                    .map(|(id, category)| (id.to_string(), category.to_string()))
                    .collect(),
            })
            .collect()
    }

    fn shared_category_labels() -> Vec<OutputLabels> {
        labels(&[
            ("A", &[("age", "y"), ("sex", "y")]),
            ("A", &[("age", "y"), ("sex", "o")]),
            ("A", &[("age", "o"), ("sex", "y")]),
            ("A", &[("age", "o"), ("sex", "o")]),
        ])
    }

    fn names(count: usize) -> Vec<String> {
        (0..count).map(|idx| format!("C{idx}")).collect()
    }

    #[test]
    fn binary_category_resolves_to_disjoint_pairs() {
        let labels = labels(&[
            ("A", &[("age", "y"), ("sex", "m")]),
            ("A", &[("age", "y"), ("sex", "f")]),
            ("A", &[("age", "o"), ("sex", "m")]),
            ("A", &[("age", "o"), ("sex", "f")]),
        ]);

        let target = resolve_target("y", &names(4), &labels, &[0.0; 4]).unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(0, 2), (1, 3)]
            }
        );
    }

    #[test]
    fn category_shared_by_two_stratifications_is_ambiguous() {
        let error =
            resolve_target("y", &names(4), &shared_category_labels(), &[0.0; 4]).unwrap_err();

        assert!(error.contains("ambiguous"), "{error}");
        assert!(error.contains("'age:y' or 'sex:y'"), "{error}");
    }

    #[test]
    fn qualified_category_resolves_one_stratification() {
        let target =
            resolve_target("sex:y", &names(4), &shared_category_labels(), &[0.0; 4]).unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(0, 1), (2, 3)]
            }
        );
    }

    #[test]
    fn category_of_a_conditional_stratification_pairs_only_where_it_applies() {
        let labels = labels(&[
            ("A", &[("age", "y")]),
            ("A", &[("age", "o"), ("care", "home")]),
            ("A", &[("age", "o"), ("care", "hosp")]),
        ]);

        let target = resolve_target("home", &names(3), &labels, &[0.0; 3]).unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(1, 2)]
            }
        );
    }

    #[test]
    fn category_of_a_non_binary_stratification_is_rejected() {
        let labels = labels(&[
            ("A", &[("age", "a")]),
            ("A", &[("age", "b")]),
            ("A", &[("age", "c")]),
        ]);

        let error = resolve_target("a", &names(3), &labels, &[0.0; 3]).unwrap_err();

        assert!(error.contains("binary"), "{error}");
    }

    #[test]
    fn bin_prefix_matches_labels_not_name_prefixes() {
        let labels = labels(&[
            ("A", &[("group", "g1")]),
            ("A", &[("group", "g2")]),
            ("A_x", &[("group", "g1")]),
        ]);

        assert_eq!(population_indices("A", &labels), Some(vec![0, 1]));
        assert_eq!(population_indices("A_g1", &labels), Some(vec![0]));
        assert_eq!(population_indices("A_x", &labels), Some(vec![2]));
        assert_eq!(population_indices("B", &labels), None);
    }
}
