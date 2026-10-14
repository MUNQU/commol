//! Planning of the transition flows of a model.
//!
//! A plan says which compartment each transition moves population from and
//! to, with which rate, and whether the rate is an absolute flow or a
//! per-capita rate. It is built from the compartment labels, never from
//! compartment names. The engine compiles the plan into its runtime flows,
//! and [`describe_flows`] publishes it, so what is printed is what is
//! simulated.

use std::collections::{HashMap, HashSet};

use commol_core::{CompartmentalStructure, RateMathExpression, Stratification, Transition};

use crate::builder::{accumulator_output_ref, generate_accumulators, subpopulation_names};
use crate::helpers::{
    category_overrides, get_rate_string_for_compartment, replace_bin_in_rate, target_categories,
};
use crate::layout::CompartmentLayout;
use crate::types::AccumulatorOutputRef;

/// One planned flow of a transition.
///
/// The flow is computed once and applied to every piece: each piece removes it
/// from its source compartment and adds it to its target compartment. A
/// transition with several sources or targets (`A + B -> C + C`) has one piece
/// per position. A per-capita rate is multiplied by the source, so it needs a
/// single source.
pub(crate) struct PlannedFlow {
    /// Id of the transition the flow comes from.
    pub(crate) transition_id: String,
    /// `(source, target)` compartments of each piece. `None` on one side for
    /// source-less or target-less transitions.
    pub(crate) pieces: Vec<(Option<usize>, Option<usize>)>,
    /// Rate expression, with bin names replaced by compartment names when the
    /// transition is `per_compartment`.
    pub(crate) rate: String,
    /// Whether the rate is an absolute flow rather than a per-capita rate.
    pub(crate) absolute: bool,
    /// Accumulator output indices the flow increments.
    pub(crate) accumulator_indices: Vec<usize>,
}

/// What a transition moves: whole bins, or the categories of a stratification.
#[derive(Clone, Copy)]
enum TransitionKind<'a> {
    Bins,
    Categories(&'a Stratification),
}

fn transition_kind<'a>(
    structure: &'a CompartmentalStructure,
    transition: &Transition,
) -> Result<TransitionKind<'a>, String> {
    let ids: Vec<&String> = transition.source.iter().chain(&transition.target).collect();
    if ids
        .iter()
        .all(|id| structure.bins.iter().any(|bin| &bin.id == *id))
    {
        return Ok(TransitionKind::Bins);
    }
    let stratification = structure
        .stratifications
        .iter()
        .find(|strat| ids.iter().all(|id| strat.categories.contains(id)))
        .ok_or_else(|| {
            format!(
                "Transition '{}' must move between bins or between the categories \
                 of one stratification",
                transition.id
            )
        })?;
    if transition.source.len() != 1 || transition.target.len() != 1 {
        return Err(format!(
            "Category transition '{}' needs exactly one source and one target category",
            transition.id
        ));
    }
    Ok(TransitionKind::Categories(stratification))
}

/// Compartments a transition starts from: those of its first source bin, of its
/// first target bin when it has no source, or those carrying its source
/// category.
fn driving_compartments(
    transition: &Transition,
    kind: TransitionKind<'_>,
    layout: &CompartmentLayout,
) -> Vec<usize> {
    match kind {
        TransitionKind::Bins => {
            let bin = transition
                .source
                .first()
                .unwrap_or_else(|| &transition.target[0]);
            layout.of_bin(bin).collect()
        }
        TransitionKind::Categories(stratification) => layout
            .with_category(&stratification.id, &transition.source[0])
            .collect(),
    }
}

/// Plan every transition flow of a model.
///
/// A bin transition starts from each compartment of its first source bin. Its
/// sources are the compartments of the source bins with the same categories,
/// and its targets the compartments of the target bins with the same
/// categories, or with the categories its matched stratified rate redirects
/// with `to`. A category transition moves each compartment that carries its
/// source category to the compartment of the same bin with its target category
/// instead. A source-less transition feeds every compartment of its target bin;
/// a target-less one drains every compartment of its source bin.
///
/// # Errors
///
/// Returns a message when a transition mixes bins and categories, when a
/// category transition does not have exactly one source and one target, when
/// a stratified rate matches no compartment the transition starts from, when a
/// transition with several sources has a per-capita rate, or when a flow ends
/// in categories no compartment has.
pub(crate) fn plan_transition_flows(
    structure: &CompartmentalStructure,
    layout: &CompartmentLayout,
    accumulator_map: &HashMap<AccumulatorOutputRef, usize>,
    subpopulation_names: &HashSet<String>,
) -> Result<Vec<PlannedFlow>, String> {
    let stratifications = &structure.stratifications;
    let compartment_names: HashSet<&str> = layout.names().iter().map(String::as_str).collect();
    let is_absolute = |rate: &str, explicit: Option<bool>| {
        explicit.unwrap_or_else(|| {
            RateMathExpression::from_string(rate.to_string())
                .get_variables()
                .iter()
                .any(|v| compartment_names.contains(v.as_str()) || subpopulation_names.contains(v))
        })
    };

    let mut flows = Vec::new();
    for transition in &structure.transitions {
        if transition.source.is_empty() && transition.target.is_empty() {
            continue;
        }
        let kind = transition_kind(structure, transition)?;
        let driving = driving_compartments(transition, kind, layout);
        check_stratified_rates_match(transition, &driving, layout)?;

        for start in driving {
            let applied = layout.applied(start);
            let Some(matched) = get_rate_string_for_compartment(transition, &applied) else {
                continue;
            };
            let source_categories = layout.labels()[start].categories.clone();
            let rate_overrides = matched
                .stratified_rate
                .map(|sr| category_overrides(&sr.conditions))
                .unwrap_or_default();

            // Categories the flow ends in. A category transition always moves
            // its stratification; `to` overrides redirect the rest.
            let mut overrides = rate_overrides;
            if let TransitionKind::Categories(stratification) = kind {
                overrides.insert(stratification.id.as_str(), transition.target[0].as_str());
            }
            let end_categories = if transition.source.is_empty() || overrides.is_empty() {
                source_categories.clone()
            } else {
                target_categories(&applied, stratifications, &overrides)
            };

            let (source_bins, target_bins): (Vec<&str>, Vec<&str>) = match kind {
                TransitionKind::Bins => (
                    transition.source.iter().map(String::as_str).collect(),
                    transition.target.iter().map(String::as_str).collect(),
                ),
                TransitionKind::Categories(_) => {
                    let bin = layout.labels()[start].source.as_str();
                    (vec![bin], vec![bin])
                }
            };
            let sources: Option<Vec<usize>> = source_bins
                .iter()
                .map(|bin| layout.find(bin, &source_categories))
                .collect();
            let targets: Option<Vec<usize>> = target_bins
                .iter()
                .map(|bin| layout.find(bin, &end_categories))
                .collect();
            let (Some(sources), Some(targets)) = (sources, targets) else {
                return Err(format!(
                    "Transition '{}' sends '{}' to categories {:?} that no compartment \
                     of its target bin has. Add a 'to' condition for every \
                     stratification the target needs.",
                    transition.id,
                    layout.name(start),
                    end_categories
                        .iter()
                        .map(|(id, category)| format!("{id}={category}"))
                        .collect::<Vec<_>>()
                ));
            };

            let mut rate = matched.rate_string.clone();
            if transition.per_compartment.unwrap_or(false) && matches!(kind, TransitionKind::Bins) {
                for (bin, idx) in source_bins.iter().zip(&sources) {
                    rate = replace_bin_in_rate(&rate, bin, layout.name(*idx));
                }
                for (bin, idx) in target_bins.iter().zip(&targets) {
                    rate = replace_bin_in_rate(&rate, bin, layout.name(*idx));
                }
            }

            let explicit_absolute = matched.stratified_rate.and_then(|sr| sr.absolute);
            let absolute = sources.is_empty() || is_absolute(&rate, explicit_absolute);
            if !absolute && sources.len() > 1 {
                return Err(format!(
                    "Transition '{}' has several sources and a per-capita rate, which \
                     has no single source to multiply. Use an absolute rate that \
                     references the compartments (for example 'k * A * B'), or the \
                     '$compartment' placeholder for independent flows.",
                    transition.id
                ));
            }

            // Accumulators count the flow in the categories it ends in.
            let accumulator_categories: HashMap<String, String> = if overrides.is_empty() {
                applied.clone()
            } else {
                target_categories(&applied, stratifications, &overrides)
                    .into_iter()
                    .collect()
            };

            let pieces = (0..sources.len().max(targets.len()))
                .map(|k| (sources.get(k).copied(), targets.get(k).copied()))
                .collect();

            flows.push(PlannedFlow {
                transition_id: transition.id.clone(),
                pieces,
                rate,
                absolute,
                accumulator_indices: accumulator_indices(
                    transition,
                    &accumulator_categories,
                    stratifications,
                    accumulator_map,
                ),
            });
        }
    }

    Ok(flows)
}

/// Reject a stratified rate whose category conditions hold for none of the
/// compartments the transition starts from: it could never be applied.
fn check_stratified_rates_match(
    transition: &Transition,
    driving: &[usize],
    layout: &CompartmentLayout,
) -> Result<(), String> {
    for stratified_rate in transition.stratified_rates.iter().flatten() {
        let matches_some = driving.iter().any(|&idx| {
            let label = &layout.labels()[idx];
            stratified_rate.conditions.iter().all(|condition| {
                condition.category.as_deref().is_none_or(|category| {
                    label.category(&condition.stratification) == Some(category)
                })
            })
        });
        if !matches_some {
            return Err(format!(
                "Transition '{}' has a stratified rate whose conditions match none of \
                 its source compartments. A condition with a category filters the \
                 source; to send the flow to a category of a stratification the \
                 source does not carry, use 'to' without a category.",
                transition.id
            ));
        }
    }
    Ok(())
}

/// Accumulator outputs a flow increments, in the categories it ends in.
fn accumulator_indices(
    transition: &Transition,
    categories: &HashMap<String, String>,
    stratifications: &[Stratification],
    accumulator_map: &HashMap<AccumulatorOutputRef, usize>,
) -> Vec<usize> {
    transition
        .accumulators
        .iter()
        .filter_map(|accumulator_id| {
            let output_ref =
                accumulator_output_ref(accumulator_id.clone(), categories, stratifications);
            accumulator_map.get(&output_ref).copied()
        })
        .collect()
}

/// One flow of population between two compartments of a model.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowDescription {
    /// Id of the transition the flow comes from.
    pub transition: String,
    /// Compartment the flow leaves, or `None` for a source-less transition.
    pub source: Option<String>,
    /// Compartment the flow enters, or `None` for a target-less transition.
    pub target: Option<String>,
    /// Rate expression as the engine evaluates it.
    pub rate: String,
    /// Compartment a per-capita rate is multiplied by, or `None` when the rate
    /// is an absolute flow. Every piece of a transition shares the same flow.
    pub per_capita_of: Option<String>,
}

/// Every transition flow of a compartmental structure, exactly as the engine
/// simulates it.
///
/// It does not depend on the engine of the model, so any structure can be
/// described.
///
/// # Errors
///
/// Returns a message when a transition mixes bins and categories, or when a
/// category transition has no source or no target.
pub fn describe_flows(structure: &CompartmentalStructure) -> Result<Vec<FlowDescription>, String> {
    let layout = CompartmentLayout::new(structure);
    let accumulator_map: HashMap<AccumulatorOutputRef, usize> = generate_accumulators(structure)
        .into_iter()
        .enumerate()
        .map(|(idx, output)| (output.output_ref, idx))
        .collect();
    let names = subpopulation_names(&layout, structure);
    let flows = plan_transition_flows(structure, &layout, &accumulator_map, &names)?;
    let name = |idx: Option<usize>| idx.map(|idx| layout.name(idx).to_string());
    Ok(flows
        .into_iter()
        .flat_map(|flow| {
            let per_capita_of = if flow.absolute {
                None
            } else {
                name(flow.pieces[0].0)
            };
            flow.pieces
                .iter()
                .map(|&(source, target)| FlowDescription {
                    transition: flow.transition_id.clone(),
                    source: name(source),
                    target: name(target),
                    rate: flow.rate.clone(),
                    per_capita_of: per_capita_of.clone(),
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A structure with the given bins, stratifications and transitions.
    fn model(bins: &[&str], stratifications: &str, transitions: &str) -> CompartmentalStructure {
        let bins: Vec<String> = bins
            .iter()
            .map(|id| format!(r#"{{"id":"{id}","name":"{id}"}}"#))
            .collect();
        let fractions: Vec<String> = bins
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                let id = bins[idx].split('"').nth(3).unwrap();
                format!(r#"{{"bin":"{id}","fraction":null}}"#)
            })
            .collect();
        let json = format!(
            r#"{{"bins":[{}],"stratifications":[{}],
            "transitions":[{}],"initial_conditions":{{"population_size":100,
            "bin_fractions":[{}],"stratification_fractions":[]}}}}"#,
            bins.join(","),
            stratifications,
            transitions,
            fractions.join(",")
        );
        serde_json::from_str(&json).expect("valid structure JSON")
    }

    fn flows(
        model: &CompartmentalStructure,
    ) -> Vec<(Option<String>, Option<String>, String, bool)> {
        describe_flows(model)
            .unwrap()
            .into_iter()
            .map(|flow| {
                let absolute = flow.per_capita_of.is_none();
                (flow.source, flow.target, flow.rate, absolute)
            })
            .collect()
    }

    fn some(name: &str) -> Option<String> {
        Some(name.to_string())
    }

    #[test]
    fn target_less_transition_drains_its_source() {
        let model = model(
            &["A"],
            "",
            r#"{"id":"death","source":["A"],"target":[],"rate":"d"}"#,
        );

        assert_eq!(
            flows(&model),
            vec![(some("A"), None, "d".to_string(), false)]
        );
    }

    #[test]
    fn a_bin_does_not_capture_bins_that_share_its_prefix() {
        let model = model(
            &["A", "AB", "B", "BB"],
            "",
            r#"{"id":"t","source":["A"],"target":["B"],"rate":"k"}"#,
        );

        assert_eq!(
            flows(&model),
            vec![(some("A"), some("B"), "k".to_string(), false)]
        );
    }

    #[test]
    fn category_transition_moves_every_bin_between_categories() {
        let model = model(
            &["S", "I"],
            r#"{"id":"age","categories":["young","old"]}"#,
            r#"{"id":"aging","source":["young"],"target":["old"],"rate":"a"}"#,
        );

        assert_eq!(
            flows(&model),
            vec![
                (some("S_young"), some("S_old"), "a".to_string(), false),
                (some("I_young"), some("I_old"), "a".to_string(), false),
            ]
        );
    }

    #[test]
    fn rate_referencing_a_compartment_is_an_absolute_flow() {
        let model = model(
            &["S", "I"],
            "",
            r#"{"id":"inf","source":["S"],"target":["I"],"rate":"b * S * I"}"#,
        );

        assert_eq!(
            flows(&model),
            vec![(some("S"), some("I"), "b * S * I".to_string(), true)]
        );
    }

    #[test]
    fn transition_mixing_bins_and_categories_is_rejected() {
        let model = model(
            &["S", "I"],
            r#"{"id":"age","categories":["young","old"]}"#,
            r#"{"id":"bad","source":["S"],"target":["old"],"rate":"a"}"#,
        );

        assert!(describe_flows(&model).unwrap_err().contains("bad"));
    }

    #[test]
    fn several_sources_and_targets_become_pieces_of_one_flow() {
        let model = model(
            &["A", "B", "C"],
            "",
            r#"{"id":"r","source":["A","B"],"target":["C","C"],"rate":"k * A * B"}"#,
        );

        assert_eq!(
            flows(&model),
            vec![
                (some("A"), some("C"), "k * A * B".to_string(), true),
                (some("B"), some("C"), "k * A * B".to_string(), true),
            ]
        );
    }

    #[test]
    fn several_sources_with_a_per_capita_rate_are_rejected() {
        let model = model(
            &["A", "B"],
            "",
            r#"{"id":"out","source":["A","B"],"target":[],"rate":"mu"}"#,
        );

        assert!(describe_flows(&model).unwrap_err().contains("$compartment"));
    }

    #[test]
    fn stratified_rate_that_matches_no_source_is_rejected() {
        let model = model(
            &["S"],
            r#"{"id":"age","categories":["y","o"]},
               {"id":"care","categories":["home","hosp"],
                "conditions":[{"stratification":"age","category":"o"}]}"#,
            r#"{"id":"aging","source":["S"],"target":["S"],"stratified_rates":[{
                "conditions":[{"stratification":"age","category":"y","to":"o"},
                              {"stratification":"care","category":"home"}],
                "rate":"a"}]}"#,
        );

        let error = describe_flows(&model).unwrap_err();

        assert!(error.contains("use 'to'"), "{error}");
    }

    #[test]
    fn to_without_category_routes_into_a_new_stratification() {
        let model = model(
            &["S"],
            r#"{"id":"age","categories":["y","o"]},
               {"id":"care","categories":["home","hosp"],
                "conditions":[{"stratification":"age","category":"o"}]}"#,
            r#"{"id":"aging","source":["S"],"target":["S"],"stratified_rates":[{
                "conditions":[{"stratification":"age","category":"y","to":"o"},
                              {"stratification":"care","to":"home"}],
                "rate":"a"}]}"#,
        );

        assert_eq!(
            flows(&model),
            vec![(some("S_y"), some("S_o_home"), "a".to_string(), false)]
        );
    }

    #[test]
    fn moving_into_a_conditioning_category_needs_the_dependent_category() {
        let stratifications = r#"{"id":"age","categories":["y","o"]},
               {"id":"care","categories":["home","hosp"],
                "conditions":[{"stratification":"age","category":"o"}]}"#;
        let bare = model(
            &["S"],
            stratifications,
            r#"{"id":"aging","source":["y"],"target":["o"],"rate":"a"}"#,
        );
        let routed = model(
            &["S"],
            stratifications,
            r#"{"id":"aging","source":["y"],"target":["o"],"stratified_rates":[{
                "conditions":[{"stratification":"care","to":"home"}],"rate":"a"}]}"#,
        );

        assert!(
            describe_flows(&bare)
                .unwrap_err()
                .contains("'to' condition")
        );
        assert_eq!(
            flows(&routed),
            vec![(some("S_y"), some("S_o_home"), "a".to_string(), false)]
        );
    }
}
