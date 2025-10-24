//! Helper functions for stratification handling and rate resolution.

use commol_core::{RateMathExpression, Stratification, StratifiedRate, Transition};
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
/// The rate string to use for this compartment, or `None` if no rate is defined.
pub(crate) fn get_rate_string_for_compartment(
    transition: &Transition,
    stratification_values: &HashMap<String, String>,
) -> Option<String> {
    // If no stratified rates defined, use default rate
    if transition.stratified_rates.is_none() {
        return transition.rate.as_ref().map(|r| match r {
            RateMathExpression::Parameter(p) => p.clone(),
            RateMathExpression::Formula(f) => f.formula.clone(),
            RateMathExpression::Constant(c) => c.to_string(),
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

    // If we found a match, return the rate string
    if let Some(matched_rate) = best_match {
        return Some(matched_rate.rate.clone());
    }

    // Fall back to default rate
    transition.rate.as_ref().map(rate_to_string)
}
