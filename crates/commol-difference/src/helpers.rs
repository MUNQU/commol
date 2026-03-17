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
/// # Arguments
///
/// * `compartment_name` - The full compartment name (e.g., "S_young_urban")
/// * `bin` - The bin prefix (e.g., "S")
/// * `stratifications` - List of stratifications in the model
///
/// # Returns
///
/// A HashMap mapping stratification IDs to their category values.
///
/// # Example
///
/// ```text
/// Input: "S_young_urban" with bin "S" and stratifications ["age", "location"]
/// Output: { "age" -> "young", "location" -> "urban" }
/// ```
pub(crate) fn extract_stratifications(
    compartment_name: &str,
    bin: &str,
    stratifications: &[Stratification],
) -> HashMap<String, String> {
    let mut result = HashMap::new();

    // Remove bin prefix
    if !compartment_name.starts_with(bin) {
        return result;
    }

    // Get the stratification part (everything after bin and first underscore)
    let stratification_part = &compartment_name[bin.len()..];
    if stratification_part.is_empty() {
        return result; // No stratifications
    }

    // Remove leading underscore, return empty if invalid format
    let stratification_part = match stratification_part.strip_prefix('_') {
        Some(stripped) => stripped,
        None => return result,
    };

    // Split by underscore to get categories
    let categories: Vec<&str> = stratification_part.split('_').collect();

    // Match categories with stratification IDs (in order)
    for (i, stratification) in stratifications.iter().enumerate() {
        if i < categories.len() {
            result.insert(stratification.id.clone(), categories[i].to_string());
        }
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

    // Find the best match (most specific)
    let mut best_match: Option<&StratifiedRate> = None;
    let mut best_match_count = 0;

    for stratified_rate in stratified_rates {
        let mut matches = true;
        let mut match_count = 0;

        // Check if all conditions in this stratified rate match
        for condition in &stratified_rate.conditions {
            match stratification_values.get(&condition.stratification) {
                Some(actual_category) if actual_category == &condition.category => {
                    match_count += 1;
                }
                _ => {
                    matches = false;
                    break;
                }
            }
        }

        // If this matches and is more specific than previous best, use it
        if matches && match_count > best_match_count {
            best_match = Some(stratified_rate);
            best_match_count = match_count;
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
        .filter_map(|c| c.to.as_ref().map(|to| (c.stratification.as_str(), to.as_str())))
        .collect();

    // Reconstruct target compartment name using stratification declaration order
    let mut parts = Vec::with_capacity(stratifications.len() + 1);
    parts.push(target_bin.to_string());

    for strat in stratifications {
        let category = if let Some(&to_cat) = override_map.get(strat.id.as_str()) {
            to_cat.to_string()
        } else {
            stratification_values
                .get(&strat.id)
                .cloned()
                .unwrap_or_default()
        };
        parts.push(category);
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
