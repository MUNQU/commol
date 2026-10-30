//! Initial conditions that calibration writes into a compartmental engine.
//!
//! This module is the single source of truth for how an initial-condition
//! value reaches a compartmental engine: how its id resolves to compartments,
//! in which order the values are written, and how the remainder keeps the
//! sum of the amounts the model states. Every engine of the aggregate kind
//! resolves and writes the values of its kind through it
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
//!
//! A bin value, and the remainder, split among their compartments as the
//! initial state does; compartments that hold nothing, such as those of a bin
//! whose amount is left for calibration, split as the stratification
//! fractions declare within them ([`crate::CompiledModel::fractions`]).

use commol_core::{ModelValue, ModelValueType, ValuePlan};

use crate::engine::CompartmentalEngine;
use crate::labels::OutputLabels;

/// Separator of a qualified category id, as in `group:g1`.
pub const CATEGORY_SEPARATOR: char = ':';

/// Relative tolerance under which an excess over the sum the remainder keeps
/// is taken to be zero.
const EXCESS_TOLERANCE: f64 = 1e-9;

/// Compartments an initial-condition value is written to.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InitialConditionTarget {
    /// A bin or a bin prefix. The value is the amount the covered
    /// compartments hold, shared by weights taken from the initial state, or
    /// from the fractions when the compartments hold nothing ([`weights`]).
    Aggregate { entries: Vec<(usize, f64)> },
    /// One expanded compartment. The value is its amount.
    Exact { index: usize },
    /// A category of a binary stratification. The value splits each
    /// `(category, complement)` pair of groups of compartments: one
    /// compartment each, or, when a later stratification is split by this one,
    /// every compartment of that stratification, which keeps its split within
    /// its side ([`weights`]).
    PairedCategory {
        pairs: Vec<(Vec<usize>, Vec<usize>)>,
    },
}

/// The share of a value each compartment of `indices` takes: its share of
/// their amounts in `state`; when they hold nothing, its share of the split
/// the fractions declare within them; and an even split when that split
/// gives them nothing either.
///
/// The split within them is the product of the fractions that differ
/// between them. A fraction they all take scales every share alike, so it
/// leaves the split as it is, or, when it is 0, empties every share: a side
/// of a category, or a bin prefix, whose own fraction is 0 still takes the
/// split of the stratifications within it.
///
/// `fractions` holds the fraction of each category of each compartment, in
/// the order of the categories of its `labels`.
fn weights(
    indices: &[usize],
    state: &[f64],
    labels: &[OutputLabels],
    fractions: &[Vec<f64>],
) -> Vec<f64> {
    let normalized = |of: Vec<f64>| {
        let total: f64 = of.iter().sum();
        (total > 0.0).then(|| of.iter().map(|value| value / total).collect())
    };
    let amounts = indices.iter().map(|&idx| state[idx]).collect();
    if let Some(weights) = normalized(amounts) {
        return weights;
    }
    let of = |idx: usize| {
        labels[idx]
            .categories
            .iter()
            .map(|(id, _)| id.as_str())
            .zip(fractions[idx].iter().copied())
    };
    // A compartment outside a conditional stratification is not split by it.
    let fraction = |idx: usize, id: &str| {
        of(idx)
            .find(|(stratification, _)| *stratification == id)
            .map_or(1.0, |(_, value)| value)
    };
    // Each compartment against the first, once per stratification of either.
    let first = indices[0];
    let mut differing: Vec<&str> = Vec::new();
    for &idx in &indices[1..] {
        let mut differ = |id, value, other| {
            if value != fraction(other, id) && !differing.contains(&id) {
                differing.push(id);
            }
        };
        of(idx).for_each(|(id, value)| differ(id, value, first));
        of(first).for_each(|(id, value)| differ(id, value, idx));
    }
    let split = indices
        .iter()
        .map(|&idx| {
            of(idx)
                .filter(|(id, _)| differing.contains(id))
                .map(|(_, value)| value)
                .product()
        })
        .collect();
    normalized(split).unwrap_or_else(|| vec![1.0 / indices.len() as f64; indices.len()])
}

impl InitialConditionTarget {
    fn aggregate(
        indices: Vec<usize>,
        state: &[f64],
        labels: &[OutputLabels],
        fractions: &[Vec<f64>],
    ) -> Self {
        let entries = indices
            .iter()
            .copied()
            .zip(weights(&indices, state, labels, fractions))
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
    /// - a bin or a bin prefix (`A`, or `A_g1` over a deeper stratification),
    ///   when it holds no [`CATEGORY_SEPARATOR`];
    /// - a category of a binary stratification, qualified (`group:g1`) or
    ///   bare (`g1`) when one stratification has it ([`resolve_category`],
    ///   among the categories the compartments have).
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
        let fractions = engine.model().fractions();
        let splits_by = engine.model().splits_by();

        let targets = values
            .iter()
            .map(|value| match value.value_type {
                ModelValueType::InitialCondition => resolve_target(
                    &value.id,
                    &compartments,
                    &labels,
                    &state,
                    fractions,
                    splits_by,
                )
                .map(|target| (value.id.clone(), target)),
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

        let model = engine.model();
        let weights = weights(indices, &state, model.output_labels(), model.fractions());
        for (&idx, weight) in indices.iter().zip(weights) {
            engine
                .set_initial_condition(idx, remainder * weight)
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
            let model = engine.model();
            let mut writes = Vec::new();
            for (category, complement) in pairs {
                let pair_total: f64 = category.iter().chain(complement).map(|&i| state[i]).sum();
                for (side, part) in [(category, value), (complement, 1.0 - value)] {
                    let weights = weights(side, &state, model.output_labels(), model.fractions());
                    writes.extend(
                        side.iter()
                            .zip(weights)
                            .map(|(&idx, weight)| (idx, part * pair_total * weight)),
                    );
                }
            }
            // A split of a pair is a share of the amounts already there,
            // so a compartment that had no value still has none.
            let indices: Vec<usize> = writes.iter().map(|&(idx, _)| idx).collect();
            let unvalued = engine.model().unvalued_of(&indices);
            for (idx, amount) in writes {
                engine.set_initial_condition(idx, amount)?;
            }
            engine.model_mut().keep_unvalued(&unvalued);
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

/// The stratification and the bare name of a category id, among
/// `stratifications`, each with its categories.
///
/// A bare id (`g1`) is a category; a qualified id (`group:g1`) is a
/// stratification, [`CATEGORY_SEPARATOR`] and one of its categories. Ids
/// have no rule on their characters, so an id matches whole ids, never a
/// split at one separator: `a:b:c` is the category `b:c` of `a` or the
/// category `c` of `a:b`, whichever the stratifications declare. `Ok(None)`
/// when no stratification has the category.
///
/// # Errors
///
/// Returns a message when several stratifications match the id; it names
/// each of them and the qualified ids to write.
pub fn resolve_category<'a, S: AsRef<str>>(
    id: &str,
    stratifications: &'a [(S, Vec<S>)],
) -> Result<Option<(&'a str, &'a str)>, String> {
    let matches: Vec<(&str, &str)> = stratifications
        .iter()
        .flat_map(|(stratification, categories)| {
            let stratification = stratification.as_ref();
            categories
                .iter()
                .map(AsRef::as_ref)
                .filter(move |&category| {
                    category == id
                        || id
                            .strip_prefix(stratification)
                            .and_then(|rest| rest.strip_prefix(CATEGORY_SEPARATOR))
                            == Some(category)
                })
                .map(move |category| (stratification, category))
        })
        .collect();
    match matches.as_slice() {
        [] => Ok(None),
        [found] => Ok(Some(*found)),
        _ => {
            let owners: Vec<&str> = matches.iter().map(|(owner, _)| *owner).collect();
            let qualified: Vec<String> = matches
                .iter()
                .map(|(owner, name)| format!("'{owner}{CATEGORY_SEPARATOR}{name}'"))
                .collect();
            let fix = if qualified.iter().any(|other| other != &qualified[0]) {
                format!("Qualify it as {}", qualified.join(" or "))
            } else {
                format!(
                    "Each qualified id is {}, so rename a stratification or a category",
                    qualified[0]
                )
            };
            Err(format!(
                "Category '{id}' is ambiguous: it is declared by stratifications {owners:?}. \
                 {fix}."
            ))
        }
    }
}

/// The stratification and the bare name of a category id of the initial
/// conditions ([`resolve_category`]), among `stratifications`, each with its
/// categories.
///
/// # Errors
///
/// Returns the message of [`resolve_category`], or a message with every
/// category there is when no stratification has the category.
pub fn resolve_declared_category<'a, S: AsRef<str>>(
    id: &str,
    stratifications: &'a [(S, Vec<S>)],
) -> Result<(&'a str, &'a str), String> {
    resolve_category(id, stratifications)?.ok_or_else(|| {
        let mut available: Vec<&str> = stratifications
            .iter()
            .flat_map(|(_, categories)| categories.iter().map(AsRef::as_ref))
            .collect();
        available.sort_unstable();
        available.dedup();
        format!(
            "Category '{id}' not found in initial conditions. Available categories: \
             {available:?}"
        )
    })
}

fn resolve_target(
    id: &str,
    compartments: &[String],
    labels: &[OutputLabels],
    state: &[f64],
    fractions: &[Vec<f64>],
    splits_by: &[(String, String)],
) -> Result<InitialConditionTarget, String> {
    if let Some(index) = compartments.iter().position(|name| name == id) {
        return Ok(InitialConditionTarget::Exact { index });
    }
    if !id.contains(CATEGORY_SEPARATOR)
        && let Some(indices) = prefix_indices(id, labels)
    {
        return Ok(InitialConditionTarget::aggregate(
            indices, state, labels, fractions,
        ));
    }
    // Each stratification with the categories its compartments have.
    let mut stratifications: Vec<(&str, Vec<&str>)> = Vec::new();
    for label in labels {
        for (stratification, category) in &label.categories {
            match stratifications
                .iter_mut()
                .find(|(id, _)| id == stratification)
            {
                Some((_, categories)) if !categories.contains(&category.as_str()) => {
                    categories.push(category);
                }
                Some(_) => {}
                None => stratifications.push((stratification, vec![category])),
            }
        }
    }
    match resolve_category(id, &stratifications)? {
        Some((stratification, category)) => {
            paired_category(stratification, category, labels, splits_by)
        }
        None => Err(format!(
            "Invalid bin, compartment or category '{}' for initial condition \
             calibration (available compartments: {})",
            id,
            compartments.join(", ")
        )),
    }
}

/// Pairs the compartments of a category with those that differ from them
/// only in that stratification and in the stratifications split by it,
/// directly or through another (`splits_by`, `(stratification, by)` pairs),
/// so a later stratification that differs by the category keeps its split
/// on each side.
fn paired_category(
    stratification: &str,
    category: &str,
    labels: &[OutputLabels],
    splits_by: &[(String, String)],
) -> Result<InitialConditionTarget, String> {
    let qualified = format!("{stratification}{CATEGORY_SEPARATOR}{category}");
    let mut follows: Vec<&str> = vec![stratification];
    while let Some((split, _)) = splits_by
        .iter()
        .find(|(split, by)| follows.contains(&by.as_str()) && !follows.contains(&split.as_str()))
    {
        follows.push(split);
    }
    // The source and the categories outside `follows` of each compartment.
    let keys: Vec<(&str, Vec<&(String, String)>)> = labels
        .iter()
        .map(|label| {
            let rest = label
                .categories
                .iter()
                .filter(|(id, _)| !follows.contains(&id.as_str()))
                .collect();
            (label.source.as_str(), rest)
        })
        .collect();
    let mut pairs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut paired: Vec<usize> = Vec::new();
    for (index, label) in labels.iter().enumerate() {
        if label.category(stratification) != Some(category) || paired.contains(&index) {
            continue;
        }
        let (mut side, mut complement, mut others) = (Vec::new(), Vec::new(), Vec::new());
        for (candidate_index, candidate) in labels.iter().enumerate() {
            let Some(of) = candidate.category(stratification) else {
                continue;
            };
            if keys[candidate_index] != keys[index] {
                continue;
            }
            if of == category {
                side.push(candidate_index);
            } else {
                complement.push(candidate_index);
                if !others.contains(&of) {
                    others.push(of);
                }
            }
        }
        if others.len() != 1 {
            return Err(format!(
                "Initial condition category '{}' needs a binary stratification: \
                 found {} complementary categories for one of its compartments",
                qualified,
                others.len()
            ));
        }
        paired.extend(&side);
        pairs.push((side, complement));
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

        let target = resolve_target(
            "c1",
            &names(4),
            &labels,
            &[0.0; 4],
            &vec![Vec::new(); 4],
            &[],
        )
        .unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(vec![0], vec![2]), (vec![1], vec![3])]
            }
        );
    }

    #[test]
    fn a_category_pairs_the_groups_of_a_stratification_split_by_it() {
        let labels = labels(&[
            ("A", &[("group", "g1"), ("type", "t1")]),
            ("A", &[("group", "g1"), ("type", "t2")]),
            ("A", &[("group", "g2"), ("type", "t1")]),
            ("A", &[("group", "g2"), ("type", "t2")]),
        ]);
        let splits_by = [("type".to_string(), "group".to_string())];

        let target = resolve_target(
            "g1",
            &names(4),
            &labels,
            &[0.0; 4],
            &vec![Vec::new(); 4],
            &splits_by,
        )
        .unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(vec![0, 1], vec![2, 3])]
            }
        );
    }

    #[test]
    fn category_shared_by_two_stratifications_is_ambiguous() {
        let error = resolve_target(
            "c1",
            &names(4),
            &shared_category_labels(),
            &[0.0; 4],
            &vec![Vec::new(); 4],
            &[],
        )
        .unwrap_err();

        assert!(error.contains("ambiguous"), "{error}");
        assert!(error.contains("'group:c1' or 'type:c1'"), "{error}");
    }

    #[test]
    fn a_qualified_category_matches_whole_ids_that_hold_the_separator() {
        let category_with_separator = [("a", vec!["b:c", "d"])];
        let stratification_with_separator = [("a:b", vec!["c", "d"])];

        assert_eq!(
            resolve_category("a:b:c", &category_with_separator),
            Ok(Some(("a", "b:c")))
        );
        assert_eq!(
            resolve_category("a:b:c", &stratification_with_separator),
            Ok(Some(("a:b", "c")))
        );
        let both = [("a", vec!["b:c"]), ("a:b", vec!["c"])];
        assert_eq!(
            resolve_category("a:b:c", &both),
            Err(
                "Category 'a:b:c' is ambiguous: it is declared by stratifications \
                 [\"a\", \"a:b\"]. Each qualified id is 'a:b:c', so rename a stratification \
                 or a category."
                    .to_string()
            )
        );
        assert_eq!(
            resolve_declared_category("a:x", &category_with_separator),
            Err(
                "Category 'a:x' not found in initial conditions. Available categories: \
                 [\"b:c\", \"d\"]"
                    .to_string()
            )
        );
    }

    #[test]
    fn qualified_category_resolves_one_stratification() {
        let target = resolve_target(
            "type:c1",
            &names(4),
            &shared_category_labels(),
            &[0.0; 4],
            &vec![Vec::new(); 4],
            &[],
        )
        .unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(vec![0], vec![1]), (vec![2], vec![3])]
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

        let target = resolve_target(
            "s1",
            &names(3),
            &labels,
            &[0.0; 3],
            &vec![Vec::new(); 3],
            &[],
        )
        .unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(vec![1], vec![2])]
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

        let error = resolve_target(
            "a",
            &names(3),
            &labels,
            &[0.0; 3],
            &vec![Vec::new(); 3],
            &[],
        )
        .unwrap_err();

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
