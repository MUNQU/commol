//! Helper functions for stratification handling and rate resolution.

use commol_core::{
    RateMathExpression, Stratification, StratificationCondition, StratifiedRate, Transition,
};
use std::collections::HashMap;

/// Convert a RateMathExpression to its string representation.
///
/// This helper extracts the string form of a rate expression, whether it's
/// a parameter name, formula, or constant value.
///
/// # Arguments
///
/// * `rate` - The rate expression to convert
///
/// # Returns
///
/// The string representation of the rate expression.
pub(crate) fn rate_to_string(rate: &RateMathExpression) -> String {
    match rate {
        RateMathExpression::Parameter(param) => param.clone(),
        RateMathExpression::Formula(formula) => formula.formula.clone(),
        RateMathExpression::Constant(value) => value.to_string(),
    }
}

/// Extract stratification categories from a compartment name.
///
/// Handles conditional stratifications: when a stratification has `conditions`,
/// it is only present in the compartment name if those conditions were satisfied
/// at generation time. This function processes stratifications in declaration
/// order, checks conditions against already-extracted categories, and only
/// consumes a token from the compartment name suffix when the stratification
/// applies.
///
/// # Arguments
///
/// * `compartment_name` - The full compartment name (e.g., "S_young_urban")
/// * `bin` - The bin prefix (e.g., "S")
/// * `stratifications` - List of stratifications in the model (in declaration order)
///
/// # Returns
///
/// A HashMap mapping stratification IDs to their category values for the
/// stratifications that apply to this compartment.
///
/// # Example
///
/// ```text
/// Model has: age=[y60, oe60], vaccination=[nv,v] (cond: age=oe60)
/// Input: "S_y60"  → Output: { "age" -> "y60" }
/// Input: "S_oe60_nv" → Output: { "age" -> "oe60", "vaccination" -> "nv" }
/// ```
pub(crate) fn extract_stratifications(
    compartment_name: &str,
    bin: &str,
    stratifications: &[Stratification],
) -> HashMap<String, String> {
    let mut result = HashMap::new();

    if !compartment_name.starts_with(bin) {
        return result;
    }

    let stratification_part = &compartment_name[bin.len()..];
    if stratification_part.is_empty() {
        return result;
    }

    let stratification_part = match stratification_part.strip_prefix('_') {
        Some(stripped) => stripped,
        None => return result,
    };

    let tokens: Vec<&str> = stratification_part.split('_').collect();
    let mut token_index = 0;

    for stratification in stratifications {
        // Check if this stratification applies given already-extracted categories
        let applies = match &stratification.conditions {
            None => true,
            Some(conds) => conds.iter().all(|c| match &c.category {
                Some(cat) => result.get(&c.stratification) == Some(cat),
                None => true,
            }),
        };

        if applies && token_index < tokens.len() {
            result.insert(stratification.id.clone(), tokens[token_index].to_string());
            token_index += 1;
        }
        // If applies is false: skip this stratification, don't consume a token
    }

    result
}

/// Result of matching a stratified rate for a compartment.
///
/// Contains the rate string and optionally the matched `StratifiedRate` reference,
/// which may contain `to` overrides on its conditions for cross-category transitions.
pub(crate) struct MatchedRate<'a> {
    pub rate_string: String,
    pub stratified_rate: Option<&'a StratifiedRate>,
}

/// Get the appropriate rate string for a compartment based on its stratifications.
///
/// This function resolves stratified rates by finding the most specific match
/// for the given stratification values. If no stratified rate matches, it falls
/// back to the default transition rate.
///
/// # Arguments
///
/// * `transition` - The transition containing rate information
/// * `stratification_values` - Map of stratification IDs to their category values
///
/// # Returns
///
/// A `MatchedRate` containing the rate string and the matched stratified rate
/// (if any), or `None` if no rate is defined.
pub(crate) fn get_rate_string_for_compartment<'a>(
    transition: &'a Transition,
    stratification_values: &HashMap<String, String>,
) -> Option<MatchedRate<'a>> {
    // If no stratified rates defined, use default rate
    if transition.stratified_rates.is_none() {
        return transition.rate.as_ref().map(|r| MatchedRate {
            rate_string: match r {
                RateMathExpression::Parameter(p) => p.clone(),
                RateMathExpression::Formula(f) => f.formula.clone(),
                RateMathExpression::Constant(c) => c.to_string(),
            },
            stratified_rate: None,
        });
    }

    let stratified_rates = transition.stratified_rates.as_ref().unwrap();

    // Find the best match (most specific). Target-only overrides have zero
    // source-filtering conditions, so they must still win when no more
    // specific source-filtering rate matches.
    let mut best_match: Option<&StratifiedRate> = None;
    let mut best_match_count: Option<usize> = None;

    for stratified_rate in stratified_rates {
        let mut matches = true;
        let mut match_count = 0;

        // Check if all conditions in this stratified rate match.
        // Conditions with category=None are target-only overrides: they never
        // filter source compartments and do not contribute to specificity.
        for condition in &stratified_rate.conditions {
            if let Some(cat) = &condition.category {
                match stratification_values.get(&condition.stratification) {
                    Some(actual_category) if actual_category == cat => {
                        match_count += 1;
                    }
                    _ => {
                        matches = false;
                        break;
                    }
                }
            }
        }

        // If this matches and is more specific than previous best, use it
        if matches && best_match_count.is_none_or(|count| match_count > count) {
            best_match = Some(stratified_rate);
            best_match_count = Some(match_count);
        }
    }

    // If we found a match, return the rate string and the matched stratified rate
    if let Some(matched_rate) = best_match {
        return Some(MatchedRate {
            rate_string: matched_rate.rate.clone(),
            stratified_rate: Some(matched_rate),
        });
    }

    // Fall back to default rate
    transition.rate.as_ref().map(|r| MatchedRate {
        rate_string: rate_to_string(r),
        stratified_rate: None,
    })
}

/// Compute the target compartment name by applying category overrides from
/// `StratificationCondition::to` fields.
///
/// When a condition has a `to` value, the corresponding stratification category
/// in the target compartment name is replaced with the `to` value. Categories
/// without `to` overrides are preserved from the source compartment.
///
/// Handles conditional stratifications: a stratification is only included in
/// the target name if its conditions are satisfied by the target's accumulated
/// categories (determined by applying `to` overrides to the source categories).
///
/// # Arguments
///
/// * `target_bin` - The target bin ID
/// * `stratification_values` - Source compartment's stratification categories
/// * `stratifications` - All stratifications in the model (defines ordering)
/// * `conditions` - The matched conditions (may contain `to` overrides)
///
/// # Returns
///
/// The target compartment name with category overrides applied.
pub(crate) fn compute_target_with_category_overrides(
    target_bin: &str,
    stratification_values: &HashMap<String, String>,
    stratifications: &[Stratification],
    conditions: &[StratificationCondition],
) -> String {
    // Build override map: stratification_id -> to_category
    let override_map: HashMap<&str, &str> = conditions
        .iter()
        .filter_map(|c| {
            c.to.as_ref()
                .map(|to| (c.stratification.as_str(), to.as_str()))
        })
        .collect();

    let mut parts = Vec::with_capacity(stratifications.len() + 1);
    parts.push(target_bin.to_string());

    // Track the target's effective categories to evaluate conditions for later stratifications
    let mut target_applied: HashMap<String, String> = HashMap::new();

    for strat in stratifications {
        // Determine the effective category for this stratification in the target:
        // override takes priority, then the source value
        let effective_cat = if let Some(&to_cat) = override_map.get(strat.id.as_str()) {
            Some(to_cat.to_string())
        } else {
            stratification_values.get(&strat.id).cloned()
        };

        // Check if this stratification applies to the TARGET
        // (evaluated against target's accumulated categories so far)
        let applies = match &strat.conditions {
            None => true,
            Some(conds) => conds.iter().all(|c| match &c.category {
                Some(cat) => target_applied.get(&c.stratification) == Some(cat),
                None => true,
            }),
        };

        if applies && let Some(cat) = effective_cat {
            target_applied.insert(strat.id.clone(), cat.clone());
            parts.push(cat);
        }
    }

    parts.join("_")
}

/// Check whether any condition in the slice has a `to` override set.
pub(crate) fn has_category_overrides(conditions: &[StratificationCondition]) -> bool {
    conditions.iter().any(|c| c.to.is_some())
}

/// Check if a byte is a valid identifier character (alphanumeric or underscore).
fn is_identifier_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Replace a base bin name in a rate expression with a specific compartment name.
///
/// Only replaces occurrences that appear as standalone identifiers (word boundaries)
pub(crate) fn replace_bin_in_rate(rate: &str, bin_name: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(rate.len() + 32);
    let rate_bytes = rate.as_bytes();
    let bin_bytes = bin_name.as_bytes();
    let bin_len = bin_bytes.len();
    let mut i = 0;

    while i < rate_bytes.len() {
        if i + bin_len <= rate_bytes.len() && &rate_bytes[i..i + bin_len] == bin_bytes {
            let before_ok = i == 0 || !is_identifier_char(rate_bytes[i - 1]);
            let after_ok =
                i + bin_len >= rate_bytes.len() || !is_identifier_char(rate_bytes[i + bin_len]);

            if before_ok && after_ok {
                result.push_str(replacement);
                i += bin_len;
                continue;
            }
        }
        result.push(rate_bytes[i] as char);
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use commol_core::{StratificationCondition, StratifiedRate};

    #[test]
    fn target_only_stratified_rate_can_match() {
        let transition = Transition {
            id: "route".to_string(),
            source: vec!["A".to_string()],
            target: vec!["B".to_string()],
            rate: None,
            stratified_rates: Some(vec![StratifiedRate {
                conditions: vec![StratificationCondition {
                    stratification: "group".to_string(),
                    category: None,
                    to: Some("g2".to_string()),
                }],
                rate: "1.0".to_string(),
                absolute: None,
            }]),
            condition: None,
            per_compartment: None,
        };

        let stratification_values = HashMap::from([("group".to_string(), "g1".to_string())]);
        let matched = get_rate_string_for_compartment(&transition, &stratification_values)
            .expect("target-only rate should match");

        assert_eq!(matched.rate_string, "1.0");
    }
}
