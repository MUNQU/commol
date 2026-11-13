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
//! another unit, keeps its amount and cancels out. It needs an amount that
//! the model states for a compartment that it or a value covers: without
//! one, it would give a value no one gave. Within a layer, values are
//! written in the order given.
//!
//! A bin value, and the remainder, split among their compartments as the
//! initial state does; compartments that hold nothing, such as those of a bin
//! whose amount is left for calibration, split as the stratification
//! fractions declare within them ([`crate::CompiledModel::fractions`]).

use commol_core::{ModelValue, ModelValueType, ValuePlan};

use crate::engine::CompartmentalEngine;
use crate::labels::{CATEGORY_SEPARATOR, OutputLabels};

/// Relative tolerance under which an excess over the sum the remainder keeps
/// is taken to be zero.
const EXCESS_TOLERANCE: f64 = 1e-9;

/// What the id of an initial-condition value names, and so what the value
/// is: an amount of a bin, an amount of a compartment or a split of a
/// category. An id that names an expanded compartment is a compartment,
/// because the compartment comes first in the order of resolution
/// ([`InitialConditionPlan::new`]); as a bin or a bin prefix, such an id
/// covers that one compartment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialTargetKind {
    /// A bin or a bin prefix. The value is the amount its compartments hold.
    Bin,
    /// One expanded compartment. The value is its amount.
    Compartment,
    /// A category of a binary stratification. The value is a split.
    Category,
}

impl InitialTargetKind {
    /// The name of the kind: `bin`, `compartment` or `category`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bin => "bin",
            Self::Compartment => "compartment",
            Self::Category => "category",
        }
    }

    /// Layer in which a value of this kind is written.
    fn layer(self) -> u8 {
        match self {
            Self::Bin => 0,
            Self::Category => 1,
            Self::Compartment => 2,
        }
    }
}

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

    /// What the id of the value names.
    fn kind(&self) -> InitialTargetKind {
        match self {
            Self::Aggregate { .. } => InitialTargetKind::Bin,
            Self::Exact { .. } => InitialTargetKind::Compartment,
            Self::PairedCategory { .. } => InitialTargetKind::Category,
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
    /// An initial-condition id names one of:
    /// - an expanded compartment (`A_g1`);
    /// - a bin or a bin prefix (`A`, or `A_g1` over a deeper stratification);
    /// - a category of a binary stratification, qualified (`group:g1`) or
    ///   bare (`g1`) when one stratification has it ([`resolve_category`],
    ///   among the categories the compartments have).
    ///
    /// A name of a compartment, a bin or a bin prefix names one bin and one
    /// list of its categories, because [`crate::check_model`] refuses a name
    /// that two things build, as bin `A` in `g1` and a bin `A_g1` build
    /// `A_g1`. So an id never covers the compartments of two bins, and an id
    /// that is a compartment and a bin or a prefix of that one compartment
    /// is a compartment. An id that is a compartment, a bin or a bin prefix
    /// and also a category, bare or qualified, as a bin `A` and the category
    /// `group:A`, is refused: the message gives each other id that sets the
    /// category alone, bare or qualified, and what to rename, because a bin
    /// has no other id.
    ///
    /// The remainder, when given, names the compartments that take what the
    /// other compartments leave of the sum of the amounts the model states.
    ///
    /// # Errors
    ///
    /// Returns a message when a value is not an initial condition, when an id
    /// matches none of these, is a compartment, a bin or a bin prefix and
    /// also a category, names a category declared by more than one
    /// stratification, or names a category of a stratification that is not
    /// binary for some compartment, or when the remainder does not resolve
    /// or keeps no amount the model states.
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
        let stratifications = compartment_categories(&labels);

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
                    &stratifications,
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
    /// The remainder keeps the sum of the amounts the model states for the
    /// compartments that it and the values cover. When the model states none
    /// of them ([`crate::CompiledModel::states_none_of`]), as for a remainder
    /// of a bin left `None` with no value of a bin that has an amount, that
    /// sum is a `0` that no one gave, so the remainder is refused. An amount
    /// of `0` that the model states is a sum, so a remainder that keeps only
    /// that `0` resolves, and every positive value of the others is
    /// infeasible.
    ///
    /// The plan reads which compartments have no value from the engine when
    /// it is built: a compartment of a bin left `None` that a Rust caller has
    /// written before ([`CompartmentalEngine::set_initial_condition`]) counts
    /// as stated, and a compartment override on such a bin does not
    /// (<https://github.com/MUNQU/commol/issues/482>).
    ///
    /// # Errors
    ///
    /// Returns a message when the id is not a compartment, bin or bin prefix,
    /// when it covers a compartment that a bin or compartment value already
    /// sets, or when the model states no amount of a compartment that it or
    /// a value covers.
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
                    "Initial-condition remainder '{}' overlaps the initial-condition \
                     value '{}'",
                    remainder_id, id
                ));
            }
        }

        self.remainder = Some((remainder_id.to_string(), indices));
        let covered = self.covered_compartments();
        let model = engine.model();
        if model.states_none_of(&covered) {
            return Err(format!(
                "Initial-condition remainder '{remainder_id}' would give a value no one gave: \
                 it takes what the initial-condition values leave of the amounts the model \
                 states for them and for it, and {} have no value. Give one of them an \
                 amount, or add an initial-condition value of a bin that has one",
                model.missing_amounts(&covered).join(" and ")
            ));
        }
        Ok(self)
    }

    /// The id of each value, in the order their numbers come.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.targets.iter().map(|(id, _)| id.as_str())
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

    /// The kind of each value and the compartments it sets, in the order of
    /// the values, and the compartments of the remainder (empty without
    /// one). A category split keeps the pair totals, so it sets none. The
    /// remainder is not a value, so it has no kind.
    pub fn coverage(&self) -> (Vec<(InitialTargetKind, Vec<usize>)>, Vec<usize>) {
        let values = self
            .targets
            .iter()
            .map(|(_, target)| (target.kind(), target.covered_compartments()))
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
                if target.kind().layer() == layer {
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
/// labels, so a bin `A` does not cover the compartments of a bin `A_x`, and
/// the compartments an id covers share one bin and its leading categories
/// ([`crate::check_model`] refuses a name that two things build).
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
    let matches = category_matches(id, stratifications);
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

/// Every category of `stratifications` that the category id `id` names,
/// bare or qualified ([`resolve_category`]), as its stratification and its
/// bare name.
fn category_matches<'a, S: AsRef<str>>(
    id: &str,
    stratifications: &'a [(S, Vec<S>)],
) -> Vec<(&'a str, &'a str)> {
    stratifications
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
        .collect()
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

/// The target of an initial-condition id: its compartment, its bin or bin
/// prefix, or its category ([`InitialConditionPlan::new`]), among
/// `stratifications`, each with the categories the compartments have
/// ([`compartment_categories`]).
///
/// # Errors
///
/// Returns a message when the id names nothing, or a compartment, a bin or
/// a bin prefix and also a category ([`two_meanings`]), or when the category
/// does not resolve ([`resolve_category`], [`paired_category`]).
fn resolve_target(
    id: &str,
    compartments: &[String],
    labels: &[OutputLabels],
    state: &[f64],
    fractions: &[Vec<f64>],
    splits_by: &[(String, String)],
    stratifications: &[(&str, Vec<&str>)],
) -> Result<InitialConditionTarget, String> {
    let Some((kind, indices)) = compartments_named(id, compartments, labels) else {
        return match resolve_category(id, stratifications)? {
            Some((stratification, category)) => {
                paired_category(stratification, category, labels, splits_by)
            }
            None => Err(format!(
                "Invalid bin, compartment or category '{}' for initial condition \
                 calibration (available compartments: {})",
                id,
                compartments.join(", ")
            )),
        };
    };
    let categories = category_matches(id, stratifications);
    if !categories.is_empty() {
        // The ids, other than `id`, that set one of the categories alone.
        let others: Vec<String> = categories
            .iter()
            .filter(|(stratification, category)| {
                paired_category(stratification, category, labels, splits_by).is_ok()
            })
            .flat_map(|&(stratification, category)| {
                let qualified = format!("{stratification}{CATEGORY_SEPARATOR}{category}");
                [qualified, category.to_string()]
                    .into_iter()
                    .filter(move |other| {
                        other != id
                            && category_matches(other, stratifications)
                                == [(stratification, category)]
                            && compartments_named(other, compartments, labels).is_none()
                    })
            })
            .collect();
        let bin = labels[indices[0]].source.as_str();
        return Err(two_meanings(id, kind, bin, &categories, &others));
    }
    Ok(match kind {
        InitialTargetKind::Compartment => InitialConditionTarget::Exact { index: indices[0] },
        _ => InitialConditionTarget::aggregate(indices, state, labels, fractions),
    })
}

/// The kind and the compartments of `id` as an expanded compartment, a bin
/// or a bin prefix. The compartment comes first, so an id that is also the
/// bin or a prefix of that one compartment is a compartment.
fn compartments_named(
    id: &str,
    compartments: &[String],
    labels: &[OutputLabels],
) -> Option<(InitialTargetKind, Vec<usize>)> {
    match compartments.iter().position(|name| name == id) {
        Some(index) => Some((InitialTargetKind::Compartment, vec![index])),
        None => prefix_indices(id, labels).map(|indices| (InitialTargetKind::Bin, indices)),
    }
}

/// The message of an id that is a compartment, a bin or a bin prefix
/// (`kind`) of the bin `bin`, and also each category of `categories`, as
/// `(stratification, category)`: what it names, the ids of `others`, which
/// set a category alone, and what to rename, because a bin has no other id.
fn two_meanings(
    id: &str,
    kind: InitialTargetKind,
    bin: &str,
    categories: &[(&str, &str)],
    others: &[String],
) -> String {
    let what = if kind == InitialTargetKind::Bin && bin != id {
        format!("{} prefix", kind.name())
    } else {
        kind.name().to_string()
    };
    let qualified: Vec<String> = categories
        .iter()
        .map(|(stratification, category)| {
            format!("'{stratification}{CATEGORY_SEPARATOR}{category}'")
        })
        .collect();
    let (last, rest) = qualified
        .split_last()
        .expect("an id of two meanings names a category");
    let named = if rest.is_empty() {
        format!("category {last}")
    } else {
        format!("categories {} and {last}", rest.join(", "))
    };
    let write = if others.is_empty() {
        String::new()
    } else {
        let quoted: Vec<String> = others.iter().map(|other| format!("'{other}'")).collect();
        format!(" Write {} for the category.", quoted.join(" or "))
    };
    format!(
        "Initial-condition id '{id}' names the {what} '{id}' and the {named}.{write} A {what} \
         has no other id: rename bin '{bin}' or {named}, so that each has an id of its own."
    )
}

/// Each stratification with the categories the compartments of `labels`
/// have, in the order they come.
fn compartment_categories(labels: &[OutputLabels]) -> Vec<(&str, Vec<&str>)> {
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
    stratifications
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

    /// The target of `id` among compartments of `labels` that hold nothing.
    fn resolved(
        id: &str,
        labels: &[OutputLabels],
        splits_by: &[(String, String)],
    ) -> Result<InitialConditionTarget, String> {
        let count = labels.len();
        resolve_target(
            id,
            &names(count),
            labels,
            &vec![0.0; count],
            &vec![Vec::new(); count],
            splits_by,
            &compartment_categories(labels),
        )
    }

    #[test]
    fn binary_category_resolves_to_disjoint_pairs() {
        let labels = labels(&[
            ("A", &[("group", "c1"), ("type", "t1")]),
            ("A", &[("group", "c1"), ("type", "t2")]),
            ("A", &[("group", "c2"), ("type", "t1")]),
            ("A", &[("group", "c2"), ("type", "t2")]),
        ]);

        let target = resolved("c1", &labels, &[]).unwrap();

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

        let target = resolved("g1", &labels, &splits_by).unwrap();

        assert_eq!(
            target,
            InitialConditionTarget::PairedCategory {
                pairs: vec![(vec![0, 1], vec![2, 3])]
            }
        );
    }

    #[test]
    fn category_shared_by_two_stratifications_is_ambiguous() {
        let error = resolved("c1", &shared_category_labels(), &[]).unwrap_err();

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
        let target = resolved("type:c1", &shared_category_labels(), &[]).unwrap();

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

        let target = resolved("s1", &labels, &[]).unwrap();

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

        let error = resolved("a", &labels, &[]).unwrap_err();

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
