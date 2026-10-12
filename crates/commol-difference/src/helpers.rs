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

/// Categories of a target compartment, in stratification declaration order.
///
/// Each stratification takes its `overrides` category when there is one, and
/// the source category otherwise. A conditional stratification only applies
/// when its conditions hold for the categories the target has so far, so moving
/// a compartment into or out of a conditioning category adds or drops the
/// stratifications that depend on it.
pub(crate) fn target_categories(
    source: &HashMap<String, String>,
    stratifications: &[Stratification],
    overrides: &HashMap<&str, &str>,
) -> Vec<(String, String)> {
    let mut target_applied: HashMap<String, String> = HashMap::new();
    let mut categories = Vec::new();

    for strat in stratifications {
        let effective_cat = overrides
            .get(strat.id.as_str())
            .map(|to_cat| to_cat.to_string())
            .or_else(|| source.get(&strat.id).cloned());

        let applies = match &strat.conditions {
            None => true,
            Some(conds) => conds.iter().all(|c| match &c.category {
                Some(cat) => target_applied.get(&c.stratification) == Some(cat),
                None => true,
            }),
        };

        if applies && let Some(cat) = effective_cat {
            target_applied.insert(strat.id.clone(), cat.clone());
            categories.push((strat.id.clone(), cat));
        }
    }

    categories
}

/// `to` overrides of a set of conditions, keyed by stratification id.
pub(crate) fn category_overrides(conditions: &[StratificationCondition]) -> HashMap<&str, &str> {
    conditions
        .iter()
        .filter_map(|c| {
            c.to.as_ref()
                .map(|to| (c.stratification.as_str(), to.as_str()))
        })
        .collect()
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
            accumulators: vec![],
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
