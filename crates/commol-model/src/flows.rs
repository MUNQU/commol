//! Planning of the transition flows of a model.
//!
//! A plan says which compartment each transition moves an amount from and
//! to, with which rate, and whether the rate is an absolute flow or a
//! proportional rate. It is built from the compartment labels, never from
//! compartment names. The engine compiles the plan into its runtime flows,
//! and [`describe_flows`] publishes it, so what is printed is what is
//! simulated.

use std::collections::{HashMap, HashSet};

use commol_core::{
    NEIGHBORS_PREFIX, Neighbors, Process, RateMathExpression, SPECIAL_VAR_TOTAL, Stratification,
    Transition,
};

use crate::builder::{
    accumulator_output_ref, generate_accumulators, missing_neighbors, neighbor_terms,
    subgroup_total_name_set,
};
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
/// per position. A proportional rate is multiplied by the source, so it needs a
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
    /// Whether the rate is an absolute flow rather than a proportional rate.
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
    process: &'a Process,
    transition: &Transition,
) -> Result<TransitionKind<'a>, String> {
    let ids: Vec<&String> = transition.source.iter().chain(&transition.target).collect();
    if ids
        .iter()
        .all(|id| process.bins.iter().any(|bin| &bin.id == *id))
    {
        return Ok(TransitionKind::Bins);
    }
    let stratification = process
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
/// transition without a source has a `to` condition, when a transition with
/// several sources has a proportional rate, or when a flow ends
/// in categories no compartment has.
pub(crate) fn plan_transition_flows(
    process: &Process,
    neighbors: Option<&Neighbors>,
    layout: &CompartmentLayout,
    accumulator_map: &HashMap<AccumulatorOutputRef, usize>,
    subgroup_total_names: &HashSet<String>,
) -> Result<Vec<PlannedFlow>, String> {
    let stratifications = &process.stratifications;
    let compartment_names: HashSet<&str> = layout.names().iter().map(String::as_str).collect();
    let is_absolute = |rate: &str, explicit: Option<bool>| {
        explicit.unwrap_or_else(|| {
            RateMathExpression::from_string(rate.to_string())
                .get_variables()
                .iter()
                .any(|v| compartment_names.contains(v.as_str()) || subgroup_total_names.contains(v))
        })
    };

    let mut flows = Vec::new();
    for transition in &process.transitions {
        if transition.source.is_empty() && transition.target.is_empty() {
            continue;
        }
        let kind = transition_kind(process, transition)?;
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
            let end_categories = if overrides.is_empty() {
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
            let rate = resolve_neighbors(&rate, start, &transition.id, process, neighbors, layout)?;
            if !absolute && sources.len() > 1 {
                return Err(format!(
                    "Transition '{}' has several sources and a proportional rate, which \
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

/// A rate with its reads of the neighbors of a unit (`neighbors_<bin>`,
/// `neighbors`) written as the sums they are for the compartment `start`.
///
/// Under [`Neighbors::Complete`], `neighbors_<bin>` is the bin total and
/// `neighbors` is `total`. Under [`Neighbors::Groups`], they are the weighted
/// sums of the compartments of each category of its stratification, found by
/// label. Whether the rate is absolute is decided before, so a rate that
/// reads the neighbors and no compartment stays proportional.
///
/// # Errors
///
/// Returns a message when the rate reads the neighbors and the structure
/// has none, or `start` carries no category of the stratification of
/// `Groups`.
fn resolve_neighbors(
    rate: &str,
    start: usize,
    transition_id: &str,
    process: &Process,
    neighbors: Option<&Neighbors>,
    layout: &CompartmentLayout,
) -> Result<String, String> {
    let terms = neighbor_terms(rate);
    let Some(first) = terms.first() else {
        return Ok(rate.to_string());
    };
    let Some(neighbors) = neighbors else {
        return Err(missing_neighbors(transition_id, first));
    };
    let mut resolved = rate.to_string();
    for term in &terms {
        let bin = term.strip_prefix(NEIGHBORS_PREFIX);
        let sum = match neighbors {
            Neighbors::Complete => match bin {
                Some(bin) => bin.to_string(),
                None => SPECIAL_VAR_TOTAL.to_string(),
            },
            Neighbors::Groups { stratification, .. } => {
                let Some(own) = layout.labels()[start].category(stratification) else {
                    return Err(format!(
                        "Transition '{transition_id}' reads '{term}' from '{}', which has no \
                         category of '{stratification}', the stratification the neighbors \
                         group by",
                        layout.name(start)
                    ));
                };
                let categories = process
                    .stratifications
                    .iter()
                    .find(|candidate| &candidate.id == stratification)
                    .map(|found| found.categories.as_slice())
                    .unwrap_or_default();
                group_sum(neighbors, stratification, own, categories, bin, layout)
            }
        };
        resolved = replace_bin_in_rate(&resolved, term, &format!("({sum})"));
    }
    Ok(resolved)
}

/// `Σ_g' w[own][g'] × (the compartments of category g', of `bin` or of
/// every bin)`, as an expression; `0` when every weight is 0.
fn group_sum(
    neighbors: &Neighbors,
    stratification: &str,
    own: &str,
    categories: &[String],
    bin: Option<&str>,
    layout: &CompartmentLayout,
) -> String {
    let terms: Vec<String> = categories
        .iter()
        .filter_map(|category| {
            let weight = neighbors.weight(own, category);
            let compartments: Vec<&str> = layout
                .with_category(stratification, category)
                .filter(|&idx| bin.is_none_or(|bin| layout.labels()[idx].source == bin))
                .map(|idx| layout.name(idx))
                .collect();
            if weight == 0.0 || compartments.is_empty() {
                return None;
            }
            let amount = match compartments.as_slice() {
                [one] => (*one).to_string(),
                many => format!("({})", many.join(" + ")),
            };
            Some(if weight == 1.0 {
                amount
            } else {
                format!("{weight} * {amount}")
            })
        })
        .collect();
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// Reject a stratified rate whose category conditions hold for none of the
/// compartments the transition starts from: it could never be applied. Reject
/// a `to` condition on a transition without a source: it starts from the
/// compartments it enters, so there is no target to redirect.
fn check_stratified_rates_match(
    transition: &Transition,
    driving: &[usize],
    layout: &CompartmentLayout,
) -> Result<(), String> {
    for stratified_rate in transition.stratified_rates.iter().flatten() {
        if transition.source.is_empty() && stratified_rate.conditions.iter().any(|c| c.to.is_some())
        {
            return Err(format!(
                "Transition '{}' has no source, so a 'to' condition has no target to \
                 redirect. Use a category to choose the compartment the flow enters.",
                transition.id
            ));
        }
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

/// One flow of an amount between two compartments of a model.
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
    /// Compartment a proportional rate is multiplied by, or `None` when the rate
    /// is an absolute flow. Every piece of a transition shares the same flow.
    pub proportional_to: Option<String>,
    /// Index of the planned flow the piece belongs to: the pieces of one flow
    /// move together, as one event in a stochastic engine.
    pub flow: usize,
}

/// Every transition flow of a process laid out in compartments, exactly as
/// the engine simulates it, with the reads of the neighbors of a unit
/// resolved by `neighbors`.
///
/// It does not depend on the engine of the model, so any process can be
/// described.
///
/// # Errors
///
/// Returns a message when a transition mixes bins and categories, when a
/// category transition has no source or no target, or when a rate reads the
/// neighbors and `neighbors` is `None`.
pub fn describe_flows(
    process: &Process,
    neighbors: Option<&Neighbors>,
) -> Result<Vec<FlowDescription>, String> {
    let layout = CompartmentLayout::new(process);
    let accumulator_map: HashMap<AccumulatorOutputRef, usize> = generate_accumulators(process)
        .into_iter()
        .enumerate()
        .map(|(idx, output)| (output.output_ref, idx))
        .collect();
    let names = subgroup_total_name_set(&layout, process);
    let flows = plan_transition_flows(process, neighbors, &layout, &accumulator_map, &names)?;
    let name = |idx: Option<usize>| idx.map(|idx| layout.name(idx).to_string());
    Ok(flows
        .into_iter()
        .enumerate()
        .flat_map(|(index, flow)| {
            let proportional_to = if flow.absolute {
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
                    proportional_to: proportional_to.clone(),
                    flow: index,
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A process with the given bins, stratifications and transitions.
    fn model(bins: &[&str], stratifications: &str, transitions: &str) -> Process {
        let bins: Vec<String> = bins
            .iter()
            .map(|id| format!(r#"{{"id":"{id}","name":"{id}"}}"#))
            .collect();
        let json = format!(
            r#"{{"bins":[{}],"stratifications":[{}],"transitions":[{}]}}"#,
            bins.join(","),
            stratifications,
            transitions,
        );
        serde_json::from_str(&json).expect("valid process JSON")
    }

    fn flows(model: &Process) -> Vec<(Option<String>, Option<String>, String, bool)> {
        flows_with(model, None)
    }

    fn flows_with(
        model: &Process,
        neighbors: Option<&Neighbors>,
    ) -> Vec<(Option<String>, Option<String>, String, bool)> {
        describe_flows(model, neighbors)
            .unwrap()
            .into_iter()
            .map(|flow| {
                let absolute = flow.proportional_to.is_none();
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
            r#"{"id":"out","source":["A"],"target":[],"rate":"d"}"#,
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
            &["A", "B"],
            r#"{"id":"group","categories":["g1","g2"]}"#,
            r#"{"id":"move","source":["g1"],"target":["g2"],"rate":"a"}"#,
        );

        assert_eq!(
            flows(&model),
            vec![
                (some("A_g1"), some("A_g2"), "a".to_string(), false),
                (some("B_g1"), some("B_g2"), "a".to_string(), false),
            ]
        );
    }

    #[test]
    fn rate_referencing_a_compartment_is_an_absolute_flow() {
        let model = model(
            &["A", "B"],
            "",
            r#"{"id":"t1","source":["A"],"target":["B"],"rate":"b * A * B"}"#,
        );

        assert_eq!(
            flows(&model),
            vec![(some("A"), some("B"), "b * A * B".to_string(), true)]
        );
    }

    #[test]
    fn transition_mixing_bins_and_categories_is_rejected() {
        let model = model(
            &["A", "B"],
            r#"{"id":"group","categories":["g1","g2"]}"#,
            r#"{"id":"bad","source":["A"],"target":["g2"],"rate":"a"}"#,
        );

        assert!(describe_flows(&model, None).unwrap_err().contains("bad"));
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
    fn several_sources_with_a_proportional_rate_are_rejected() {
        let model = model(
            &["A", "B"],
            "",
            r#"{"id":"out","source":["A","B"],"target":[],"rate":"k"}"#,
        );

        assert!(
            describe_flows(&model, None)
                .unwrap_err()
                .contains("$compartment")
        );
    }

    #[test]
    fn stratified_rate_that_matches_no_source_is_rejected() {
        let model = model(
            &["A"],
            r#"{"id":"group","categories":["g1","g2"]},
               {"id":"type","categories":["t1","t2"],
                "conditions":[{"stratification":"group","category":"g2"}]}"#,
            r#"{"id":"move","source":["A"],"target":["A"],"stratified_rates":[{
                "conditions":[{"stratification":"group","category":"g1","to":"g2"},
                              {"stratification":"type","category":"t1"}],
                "rate":"a"}]}"#,
        );

        let error = describe_flows(&model, None).unwrap_err();

        assert!(error.contains("use 'to'"), "{error}");
    }

    #[test]
    fn to_without_category_routes_into_a_new_stratification() {
        let model = model(
            &["A"],
            r#"{"id":"group","categories":["g1","g2"]},
               {"id":"type","categories":["t1","t2"],
                "conditions":[{"stratification":"group","category":"g2"}]}"#,
            r#"{"id":"move","source":["A"],"target":["A"],"stratified_rates":[{
                "conditions":[{"stratification":"group","category":"g1","to":"g2"},
                              {"stratification":"type","to":"t1"}],
                "rate":"a"}]}"#,
        );

        assert_eq!(
            flows(&model),
            vec![(some("A_g1"), some("A_g2_t1"), "a".to_string(), false)]
        );
    }

    #[test]
    fn moving_into_a_conditioning_category_needs_the_dependent_category() {
        let stratifications = r#"{"id":"group","categories":["g1","g2"]},
               {"id":"type","categories":["t1","t2"],
                "conditions":[{"stratification":"group","category":"g2"}]}"#;
        let bare = model(
            &["A"],
            stratifications,
            r#"{"id":"move","source":["g1"],"target":["g2"],"rate":"a"}"#,
        );
        let routed = model(
            &["A"],
            stratifications,
            r#"{"id":"move","source":["g1"],"target":["g2"],"stratified_rates":[{
                "conditions":[{"stratification":"type","to":"t1"}],"rate":"a"}]}"#,
        );

        assert!(
            describe_flows(&bare, None)
                .unwrap_err()
                .contains("'to' condition")
        );
        assert_eq!(
            flows(&routed),
            vec![(some("A_g1"), some("A_g2_t1"), "a".to_string(), false)]
        );
    }

    fn groups(weights: Option<&str>) -> Neighbors {
        let weights = weights.unwrap_or("null");
        serde_json::from_str(&format!(
            r#"{{"type":"groups","stratification":"group","weights":{weights}}}"#
        ))
        .unwrap()
    }

    const AB_GROUPS: &str = r#"{"id":"group","categories":["g1","g2"]}"#;

    #[test]
    fn complete_neighbors_read_the_bin_total_and_the_total() {
        let process = model(
            &["A", "B"],
            "",
            r#"{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * neighbors_B / neighbors"}"#,
        );

        assert_eq!(
            flows_with(&process, Some(&Neighbors::Complete)),
            vec![(
                some("A"),
                some("B"),
                "k1 * (B) / (total)".to_string(),
                false
            )]
        );
    }

    #[test]
    fn group_neighbors_sum_the_weighted_compartments_of_each_category() {
        let process = model(
            &["A", "B"],
            AB_GROUPS,
            r#"{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * neighbors_B / neighbors"}"#,
        );
        let weights = r#"{"g1":{"g1":1.0,"g2":0.5},"g2":{"g1":0.25,"g2":1.0}}"#;

        assert_eq!(
            flows_with(&process, Some(&groups(Some(weights)))),
            vec![
                (
                    some("A_g1"),
                    some("B_g1"),
                    "k1 * (B_g1 + 0.5 * B_g2) / ((A_g1 + B_g1) + 0.5 * (A_g2 + B_g2))".to_string(),
                    false
                ),
                (
                    some("A_g2"),
                    some("B_g2"),
                    "k1 * (0.25 * B_g1 + B_g2) / (0.25 * (A_g1 + B_g1) + (A_g2 + B_g2))"
                        .to_string(),
                    false
                ),
            ]
        );
    }

    #[test]
    fn group_neighbors_of_weight_zero_are_zero() {
        let process = model(
            &["A", "B"],
            AB_GROUPS,
            r#"{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * neighbors_B"}"#,
        );
        let weights = r#"{"g1":{"g1":0.0,"g2":0.0},"g2":{"g1":0.0,"g2":1.0}}"#;

        let rates: Vec<String> = flows_with(&process, Some(&groups(Some(weights))))
            .into_iter()
            .map(|(_, _, rate, _)| rate)
            .collect();
        assert_eq!(rates, ["k1 * (0)", "k1 * (B_g2)"]);
    }

    #[test]
    fn a_transition_with_no_source_and_no_target_moves_nothing() {
        let process = model(
            &["A", "B"],
            "",
            r#"{"id":"none","source":[],"target":[],"rate":"k1"},
            {"id":"ab","source":["A"],"target":["B"],"rate":"k1"}"#,
        );

        assert_eq!(
            flows(&process),
            vec![(some("A"), some("B"), "k1".to_string(), false)]
        );
    }

    #[test]
    fn group_neighbors_without_weights_see_their_own_category() {
        let process = model(
            &["A", "B"],
            AB_GROUPS,
            r#"{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * neighbors_B"}"#,
        );

        let rates: Vec<String> = flows_with(&process, Some(&groups(None)))
            .into_iter()
            .map(|(_, _, rate, _)| rate)
            .collect();
        assert_eq!(rates, ["k1 * (B_g1)", "k1 * (B_g2)"]);
    }

    #[test]
    fn a_rate_that_reads_the_neighbors_and_a_compartment_is_absolute() {
        let process = model(
            &["A", "B"],
            "",
            r#"{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * A * neighbors_B / neighbors"}"#,
        );

        assert!(flows_with(&process, Some(&Neighbors::Complete))[0].3);
    }

    #[test]
    fn a_rate_that_reads_the_neighbors_needs_them() {
        let process = model(
            &["A", "B"],
            "",
            r#"{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * neighbors_B"}"#,
        );

        let error = describe_flows(&process, None).unwrap_err();
        assert!(
            error.contains("'ab'") && error.contains("neighbors_B"),
            "{error}"
        );
        assert!(
            error.contains("Complete()") && error.contains("Groups("),
            "{error}"
        );
    }

    #[test]
    fn a_compartment_with_no_category_of_the_groups_names_it() {
        let process = model(
            &["A", "B"],
            r#"{"id":"level","categories":["l1","l2"]},
               {"id":"group","categories":["g1","g2"],
                "conditions":[{"stratification":"level","category":"l2"}]}"#,
            r#"{"id":"ab","source":["A"],"target":["B"],"rate":"k1 * neighbors_B"}"#,
        );

        let error = describe_flows(&process, Some(&groups(None))).unwrap_err();
        assert!(error.contains("A_l1") && error.contains("group"), "{error}");
    }
}
