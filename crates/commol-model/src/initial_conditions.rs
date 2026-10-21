//! Initial conditions that calibration writes into a compartmental engine.
//!
//! This module is the single source of truth for how an initial-condition
//! value reaches a compartmental engine: how its id resolves to compartments,
//! in which order the values are written, and how the remainder keeps the
//! sum of the amounts the model states. Both engines resolve and write values of their kind through it
//! ([`commol_core::SimulationEngine::plan_values`] and
//! [`commol_core::SimulationEngine::apply_values`]), so the calibration
//! objective, the calibrated predictions and the model write-back all use it.
//!
//! Initial conditions are written in the same layers as a model definition:
//! bin values first, then stratification-category splits, then exact
//! compartments, and finally the remainder. A bin or compartment value is an
//! amount in the unit of its bin; a category value is a split. The remainder
//! takes the sum of the amounts the model states less every other
//! compartment, so a compartment that no value covers, such as a bin of
//! another unit, keeps its amount and cancels out. Within a layer, values are
//! written in the order given.

use commol_core::{ModelValue, ModelValueType, ValuePlan};

use crate::engine::CompartmentalEngine;
use crate::labels::OutputLabels;

/// Separator of a qualified category id, as in `group:g1`.
const CATEGORY_SEPARATOR: char = ':';

/// Relative tolerance under which an excess over the sum the remainder keeps
/// is taken to be zero.
const EXCESS_TOLERANCE: f64 = 1e-9;

/// Compartments an initial-condition value is written to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InitialConditionTarget {
    /// A bin or a bin prefix. The value is the amount the covered
    /// compartments hold, shared by weights taken from the initial state.
    Aggregate { entries: Vec<(usize, f64)> },
    /// One expanded compartment. The value is its amount.
    Exact { index: usize },
    /// A category of a binary stratification. The value splits each
    /// `(category, complement)` pair of compartments.
    PairedCategory { pairs: Vec<(usize, usize)> },
}

impl InitialConditionTarget {
    fn aggregate(indices: Vec<usize>, state: &[f64]) -> Self {
        let total: f64 = indices.iter().map(|&idx| state[idx]).sum();
        let entries = indices
            .iter()
            .map(|&idx| {
                let weight = if total > 0.0 {
                    state[idx] / total
                } else {
                    1.0 / indices.len() as f64
                };
                (idx, weight)
            })
            .collect();
        Self::Aggregate { entries }
    }

    /// Compartments whose amount the value sets. Category splits keep the
    /// pair totals, so they cover no compartment.
    fn covered_compartments(&self) -> Vec<usize> {
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

/// Resolved initial-condition targets of a set of values.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialConditionPlan {
    /// Id and target of each value, in the order their numbers come.
    targets: Vec<(String, InitialConditionTarget)>,
    /// The sum of the initial amounts the model states.
    total: f64,
    /// Compartments that take what the other compartments leave of `total`,
    /// if any, with the id they were resolved from.
    remainder: Option<(String, Vec<usize>)>,
}

impl InitialConditionPlan {
    /// Resolve the target of every initial-condition value, and the
    /// remainder.
    ///
    /// An initial-condition id names, in order of precedence:
    /// - an expanded compartment (`A_g1`);
    /// - a qualified stratification category (`group:g1`);
    /// - a bin or a bin prefix (`A`, or `A_g1` over a deeper stratification);
    /// - a category of exactly one binary stratification (`g1`).
    ///
    /// The remainder, when given, names the compartments that take what the
    /// other compartments leave of the sum of the amounts the model states.
    ///
    /// # Errors
    ///
    /// Returns a message when a value is not an initial condition, when an id
    /// matches none of these, names a category declared by more than one
    /// stratification, or names a category of a stratification that is not
    /// binary for some compartment, or when the remainder does not resolve.
    pub fn new<E: CompartmentalEngine>(
        engine: &E,
        values: &[ModelValue],
        remainder: Option<&str>,
    ) -> Result<Self, String> {
        let compartments = engine.compartments();
        let labels = compartment_labels(engine);
        let state = engine.initial_state();

        let targets = values
            .iter()
            .map(|value| match value.value_type {
                ModelValueType::InitialCondition => {
                    resolve_target(&value.id, &compartments, &labels, &state)
                        .map(|target| (value.id.clone(), target))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;

        let plan = Self {
            targets,
            total: engine.total(),
            remainder: None,
        };
        match remainder {
            Some(remainder) => plan.with_remainder(engine, remainder),
            None => Ok(plan),
        }
    }

    /// Name the compartments that take what the others leave.
    ///
    /// # Errors
    ///
    /// Returns a message when the id is not a compartment, bin or bin prefix,
    /// or when it covers a compartment that a calibrated bin or compartment
    /// value already sets.
    fn with_remainder<E: CompartmentalEngine>(
        mut self,
        engine: &E,
        remainder_id: &str,
    ) -> Result<Self, String> {
        let compartments = engine.compartments();
        let indices =
            prefix_indices(remainder_id, &compartment_labels(engine)).ok_or_else(|| {
                format!(
                    "Initial-condition remainder '{}' is not a compartment or a bin \
                 (available: {})",
                    remainder_id,
                    compartments.join(", ")
                )
            })?;

        for (id, target) in &self.targets {
            if target
                .covered_compartments()
                .iter()
                .any(|idx| indices.contains(idx))
            {
                return Err(format!(
                    "Initial-condition remainder '{}' overlaps the calibrated initial \
                     condition '{}'",
                    remainder_id, id
                ));
            }
        }

        self.remainder = Some((remainder_id.to_string(), indices));
        Ok(self)
    }

    /// Compartments whose amount a calibrated bin or compartment value, or
    /// the remainder, sets, in engine order. Category splits keep the pair
    /// totals, so they cover none.
    pub fn covered_compartments(&self) -> Vec<usize> {
        let mut covered: Vec<usize> = self
            .targets
            .iter()
            .flat_map(|(_, target)| target.covered_compartments())
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

    /// The compartments each value sets, in the order of the values, and the
    /// compartments of the remainder (empty without one). A category split
    /// keeps the pair totals, so it sets none.
    pub fn coverage(&self) -> (Vec<Vec<usize>>, Vec<usize>) {
        let values = self
            .targets
            .iter()
            .map(|(_, target)| target.covered_compartments())
            .collect();
        let remainder = self
            .remainder
            .as_ref()
            .map_or_else(Vec::new, |(_, indices)| indices.clone());
        (values, remainder)
    }

    /// Write initial-condition values into an engine.
    ///
    /// The values are written layer by layer (bins, category splits, exact
    /// compartments) and then the remainder takes the sum of the amounts the
    /// model states less every other compartment, keeping its own split.
    ///
    /// # Returns
    ///
    /// How far the amounts the values set exceed the sum the remainder keeps
    /// (the declared amounts of every compartment the values and the
    /// remainder cover), as a fraction of those amounts: `0.0` for a feasible
    /// state. The remainder is floored at zero. Without a remainder every
    /// state is feasible, because the engine rejects a negative amount. An
    /// excess within the tolerance is `0.0`.
    ///
    /// # Errors
    ///
    /// Returns a message when the number of values is wrong or the engine
    /// rejects an initial condition.
    pub fn apply<E: CompartmentalEngine>(
        &self,
        engine: &mut E,
        values: &[f64],
    ) -> Result<f64, String> {
        if values.len() != self.targets.len() {
            return Err(format!(
                "Expected {} initial-condition values, got {}",
                self.targets.len(),
                values.len()
            ));
        }
        for layer in 0..3 {
            for ((id, target), value) in self.targets.iter().zip(values) {
                if target.layer() == layer {
                    write_target(engine, target, *value).map_err(|e| {
                        format!("Failed to set initial condition for '{}': {}", id, e)
                    })?;
                }
            }
        }

        let Some((_, indices)) = &self.remainder else {
            return Ok(0.0);
        };
        let state = engine.initial_state();
        let covered = self.covered_compartments();
        let others: f64 = state
            .iter()
            .enumerate()
            .filter(|(idx, _)| !indices.contains(idx))
            .map(|(_, value)| value)
            .sum();
        let uncovered: f64 = state
            .iter()
            .enumerate()
            .filter(|(idx, _)| !covered.contains(idx))
            .map(|(_, value)| value)
            .sum();
        // The remainder is what the values leave of the sum it keeps, so a
        // deficit is never larger than the amounts the values set.
        let remainder = self.total - others;
        let set = others - uncovered;
        let excess = if remainder < 0.0 {
            -remainder / set
        } else {
            0.0
        };
        let remainder = remainder.max(0.0);

        let current: f64 = indices.iter().map(|&idx| state[idx]).sum();
        for &idx in indices {
            let share = if current > 0.0 {
                state[idx] / current
            } else {
                1.0 / indices.len() as f64
            };
            engine
                .set_initial_condition(idx, remainder * share)
                .map_err(|e| format!("Failed to set the initial-condition remainder: {}", e))?;
        }
        Ok(feasible_or(excess))
    }
}

impl ValuePlan for InitialConditionPlan {
    fn infeasibility(&self, excess: f64) -> String {
        let remainder = self.remainder.as_ref().map_or("", |(id, _)| id.as_str());
        format!(
            "Invalid initial conditions: the calibrated amounts exceed the sum the \
             remainder '{remainder}' keeps by {excess:.6} of them"
        )
    }
}

/// An excess over the sum the remainder keeps, or `0.0` when it is within
/// the tolerance.
fn feasible_or(excess: f64) -> f64 {
    if excess > EXCESS_TOLERANCE {
        excess
    } else {
        0.0
    }
}

fn write_target<E: CompartmentalEngine>(
    engine: &mut E,
    target: &InitialConditionTarget,
    value: f64,
) -> Result<(), String> {
    match target {
        InitialConditionTarget::Aggregate { entries } => {
            for &(idx, weight) in entries {
                engine.set_initial_condition(idx, value * weight)?;
            }
        }
        InitialConditionTarget::Exact { index } => {
            engine.set_initial_condition(*index, value)?;
        }
        InitialConditionTarget::PairedCategory { pairs } => {
            let state = engine.initial_state();
            for &(category_idx, complement_idx) in pairs {
                let pair_total = state[category_idx] + state[complement_idx];
                engine.set_initial_condition(category_idx, value * pair_total)?;
                engine.set_initial_condition(complement_idx, (1.0 - value) * pair_total)?;
            }
        }
    }
    Ok(())
}

/// Labels of the compartments of an engine (its first outputs).
fn compartment_labels<E: CompartmentalEngine>(engine: &E) -> Vec<OutputLabels> {
    let mut labels = engine.output_labels();
    labels.truncate(engine.compartments().len());
    labels
}

/// Indices of the compartments a bin or bin prefix covers.
///
/// A compartment is covered when the id is its bin (`A`) or its bin followed
/// by its leading categories (`A_g1`, `A_g1_s2`). The match is built from the
/// labels, so a bin `A` does not cover the compartments of a bin `A_x`.
fn prefix_indices(id: &str, labels: &[OutputLabels]) -> Option<Vec<usize>> {
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
    state: &[f64],
) -> Result<InitialConditionTarget, String> {
    if let Some(index) = compartments.iter().position(|name| name == id) {
        return Ok(InitialConditionTarget::Exact { index });
    }
    if let Some((stratification, category)) = id.split_once(CATEGORY_SEPARATOR) {
        return paired_category(stratification, category, labels);
    }
    if let Some(indices) = prefix_indices(id, labels) {
        return Ok(InitialConditionTarget::aggregate(indices, state));
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
            ("A", &[("group", "c1"), ("type", "c1")]),
            ("A", &[("group", "c1"), ("type", "c2")]),
            ("A", &[("group", "c2"), ("type", "c1")]),
            ("A", &[("group", "c2"), ("type", "c2")]),
        ])
    }

    fn names(count: usize) -> Vec<String> {
        (0..count).map(|idx| format!("C{idx}")).collect()
    }

    #[test]
    fn binary_category_resolves_to_disjoint_pairs() {
        let labels = labels(&[
            ("A", &[("group", "c1"), ("type", "t1")]),
            ("A", &[("group", "c1"), ("type", "t2")]),
            ("A", &[("group", "c2"), ("type", "t1")]),
            ("A", &[("group", "c2"), ("type", "t2")]),
        ]);

        let target = resolve_target("c1", &names(4), &labels, &[0.0; 4]).unwrap();

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
            resolve_target("c1", &names(4), &shared_category_labels(), &[0.0; 4]).unwrap_err();

        assert!(error.contains("ambiguous"), "{error}");
        assert!(error.contains("'group:c1' or 'type:c1'"), "{error}");
    }

    #[test]
    fn qualified_category_resolves_one_stratification() {
        let target =
            resolve_target("type:c1", &names(4), &shared_category_labels(), &[0.0; 4]).unwrap();

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
            ("A", &[("group", "c1")]),
            ("A", &[("group", "c2"), ("status", "s1")]),
            ("A", &[("group", "c2"), ("status", "s2")]),
        ]);

        let target = resolve_target("s1", &names(3), &labels, &[0.0; 3]).unwrap();

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
            ("A", &[("group", "a")]),
            ("A", &[("group", "b")]),
            ("A", &[("group", "c")]),
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

        assert_eq!(prefix_indices("A", &labels), Some(vec![0, 1]));
        assert_eq!(prefix_indices("A_g1", &labels), Some(vec![0]));
        assert_eq!(prefix_indices("A_x", &labels), Some(vec![2]));
        assert_eq!(prefix_indices("B", &labels), None);
    }
}
