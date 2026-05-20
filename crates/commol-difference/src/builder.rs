//! Model builder for constructing DifferenceEquations from a compartment model.

use crate::helpers::{
    compute_target_with_category_overrides, extract_stratifications,
    get_rate_string_for_compartment, has_category_overrides, replace_bin_in_rate,
};
use crate::types::{DifferenceEquations, SubpopulationMapping, TransitionFlow};
use commol_core::{MathExpressionContext, Model, RateMathExpression};
use std::collections::{HashMap, HashSet};

impl DifferenceEquations {
    /// Create a new DifferenceEquations instance from a model.
    ///
    /// This constructor performs several pre-computation steps for efficiency:
    /// 1. Generates all stratified compartment combinations
    /// 2. Initializes population distribution across compartments
    /// 3. Pre-computes transition flows with parsed rate expressions
    /// 4. Pre-computes subpopulation mappings for stratifications
    ///
    /// # Arguments
    ///
    /// * `model` - The compartment model to compile
    ///
    /// # Returns
    ///
    /// A new `DifferenceEquations` instance ready for simulation.
    pub fn from_model(model: &Model) -> Self {
        // Generate all compartment combinations
        let (compartments, compartment_map) = generate_compartments(model);

        // Initialize population distribution
        let population = initialize_population(model, &compartments);

        // Store constant parameters for quick lookup
        // Formula parameters will be evaluated on each step
        let mut constant_parameters: HashMap<String, f64> = HashMap::new();
        let mut formula_parameters: Vec<(String, RateMathExpression)> = Vec::new();

        for p in &model.parameters {
            match &p.value {
                Some(commol_core::ParameterValue::Constant(val)) => {
                    constant_parameters.insert(p.id.clone(), *val);
                }
                Some(commol_core::ParameterValue::Formula(formula)) => {
                    // Parse formula once and store for later evaluation
                    let rate_expr = RateMathExpression::from_string(formula.clone());
                    formula_parameters.push((p.id.clone(), rate_expr));
                }
                None => {
                    // Parameter needs calibration - skip it for now
                    // During calibration, set_parameter() will provide the value
                }
            }
        }

        // Initialize expression context with constant parameters
        let mut expression_context = MathExpressionContext::new();
        expression_context.set_parameters(constant_parameters);
        expression_context.init_compartments(compartments.clone());

        // Store initial population for reset functionality
        let initial_population = population.clone();

        // Pre-compute subpopulation mappings for stratifications
        // (must be done before transition flows, as references_compartments
        // detection needs to know about subpopulation variable names)
        let subpopulation_mappings = build_subpopulation_mappings(
            &compartments,
            &model.population.stratifications,
            &model.population.bins,
        );

        let subpopulation_param_names: HashSet<String> = subpopulation_mappings
            .iter()
            .map(|m| m.parameter_name.clone())
            .collect();

        // Pre-compute all transition flows
        let transition_flows = build_transition_flows(
            model,
            &compartments,
            &compartment_map,
            &model.population.stratifications,
            &subpopulation_param_names,
        );

        // Initialize compartment flows buffer
        let num_compartments = compartments.len();
        let compartment_flows = vec![0.0; num_compartments];

        Self {
            compartments,
            population,
            expression_context,
            current_step: 0.0,
            initial_population,
            transition_flows,
            compartment_flows,
            subpopulation_mappings,
            formula_parameters,
        }
    }
}

/// Check whether a stratification's conditions are all satisfied by the
/// already-applied categories of the current compartment being built.
fn stratification_conditions_met(
    conditions: &Option<Vec<commol_core::StratificationCondition>>,
    applied: &HashMap<String, String>,
) -> bool {
    match conditions {
        None => true,
        Some(conds) => conds.iter().all(|c| {
            applied
                .get(&c.stratification)
                .map_or(false, |v| v == &c.category)
        }),
    }
}

/// Generate all stratified compartment combinations.
///
/// When a stratification has `conditions`, it only expands compartments whose
/// already-applied categories satisfy all of those conditions. Compartments
/// that do not satisfy the conditions are kept as-is (without appending this
/// stratification's categories), so the result is a non-uniform Cartesian
/// product.
///
/// Returns a tuple of (compartments vector, compartment_map for lookups).
fn generate_compartments(model: &Model) -> (Vec<String>, HashMap<String, usize>) {
    // Each element: (compartment_name, applied_categories)
    // applied_categories maps stratification_id → chosen_category for that compartment
    let mut partials: Vec<(String, HashMap<String, String>)> = model
        .population
        .bins
        .iter()
        .map(|b| (b.id.clone(), HashMap::new()))
        .collect();

    for stratification in &model.population.stratifications {
        let mut new_partials: Vec<(String, HashMap<String, String>)> = Vec::new();

        for (name, applied) in partials {
            if stratification_conditions_met(&stratification.conditions, &applied) {
                // Conditions met: expand into one entry per category
                for cat in &stratification.categories {
                    let mut new_applied = applied.clone();
                    new_applied.insert(stratification.id.clone(), cat.clone());
                    new_partials.push((format!("{}_{}", name, cat), new_applied));
                }
            } else {
                // Conditions not met: keep compartment unchanged
                new_partials.push((name, applied));
            }
        }

        partials = new_partials;
    }

    let compartments: Vec<String> = partials.into_iter().map(|(name, _)| name).collect();

    let compartment_map: HashMap<String, usize> = compartments
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();

    (compartments, compartment_map)
}

/// Initialize population distribution across compartments.
///
/// Mirrors the conditional expansion logic of `generate_compartments`: when a
/// stratification has conditions that are not satisfied by a compartment's
/// already-applied categories, that compartment is kept as-is (its population
/// is not split by that stratification's fractions).
fn initialize_population(model: &Model, compartments: &[String]) -> Vec<f64> {
    let total_population = model.population.initial_conditions.population_size as f64;

    let bin_fraction_map: HashMap<String, Option<f64>> = model
        .population
        .initial_conditions
        .bin_fractions
        .iter()
        .map(|bf| (bf.bin.clone(), bf.fraction))
        .collect();

    // Each element: (compartment_name, applied_categories, population_value)
    let mut partials: Vec<(String, HashMap<String, String>, f64)> = model
        .population
        .bins
        .iter()
        .map(|bin| {
            let fraction = bin_fraction_map
                .get(&bin.id)
                .and_then(|f| *f)
                .unwrap_or(0.0);
            (bin.id.clone(), HashMap::new(), total_population * fraction)
        })
        .collect();

    for stratification in &model.population.stratifications {
        let fraction_map: HashMap<String, f64> = model
            .population
            .initial_conditions
            .stratification_fractions
            .iter()
            .find(|sf| sf.stratification == stratification.id)
            .map(|item| {
                item.fractions
                    .iter()
                    .map(|frac| (frac.category.clone(), frac.fraction))
                    .collect()
            })
            .unwrap_or_default();

        let mut new_partials: Vec<(String, HashMap<String, String>, f64)> = Vec::new();

        for (name, applied, pop) in partials {
            if stratification_conditions_met(&stratification.conditions, &applied) {
                // Conditions met: split population by stratification fractions
                for cat in &stratification.categories {
                    let fraction = fraction_map.get(cat).unwrap_or(&0.0);
                    let mut new_applied = applied.clone();
                    new_applied.insert(stratification.id.clone(), cat.clone());
                    new_partials.push((format!("{}_{}", name, cat), new_applied, pop * fraction));
                }
            } else {
                // Conditions not met: keep compartment and population unchanged
                new_partials.push((name, applied, pop));
            }
        }

        partials = new_partials;
    }

    // Convert to vector indexed by compartment order
    let distribution: HashMap<String, f64> = partials
        .into_iter()
        .map(|(name, _, pop)| (name, pop))
        .collect();

    compartments
        .iter()
        .map(|comp| *distribution.get(comp).unwrap_or(&0.0))
        .collect()
}

/// Build pre-computed transition flows.
fn build_transition_flows(
    model: &Model,
    compartments: &[String],
    compartment_map: &HashMap<String, usize>,
    stratifications: &[commol_core::Stratification],
    subpopulation_names: &HashSet<String>,
) -> Vec<TransitionFlow> {
    let mut transition_flows = Vec::new();

    for transition in &model.dynamics.transitions {
        if !transition.source.is_empty() && !transition.target.is_empty() {
            let source_bin = &transition.source[0];
            let target_bin = &transition.target[0];

            // Process each compartment
            for (i, compartment_name) in compartments.iter().enumerate() {
                if compartment_name.starts_with(source_bin) {
                    let source_index = i;

                    // Extract stratifications for this compartment
                    let stratification_values =
                        extract_stratifications(compartment_name, source_bin, stratifications);

                    // Get the appropriate rate for this compartment
                    if let Some(matched) =
                        get_rate_string_for_compartment(transition, &stratification_values)
                    {
                        // Compute target compartment name:
                        // If matched stratified rate has `to` overrides, use them
                        // to remap categories. Otherwise, use standard bin replacement.
                        let target_compartment_name = if let Some(sr) = matched.stratified_rate {
                            if has_category_overrides(&sr.conditions) {
                                compute_target_with_category_overrides(
                                    target_bin,
                                    &stratification_values,
                                    stratifications,
                                    &sr.conditions,
                                )
                            } else {
                                compartment_name.replacen(source_bin, target_bin, 1)
                            }
                        } else {
                            compartment_name.replacen(source_bin, target_bin, 1)
                        };

                        if let Some(&target_index) = compartment_map.get(&target_compartment_name) {
                            // If per_compartment is enabled, replace base bin names
                            // with the specific stratified compartment names
                            let rate_string = if transition.per_compartment.unwrap_or(false) {
                                let mut modified = replace_bin_in_rate(
                                    &matched.rate_string,
                                    source_bin,
                                    compartment_name,
                                );
                                modified = replace_bin_in_rate(
                                    &modified,
                                    target_bin,
                                    &target_compartment_name,
                                );
                                modified
                            } else {
                                matched.rate_string
                            };

                            // Parse the rate expression once
                            let rate_expression =
                                RateMathExpression::from_string(rate_string.clone());

                            // Check if rate expression references compartment or
                            // subpopulation variables (partial bin sums)
                            let rate_variables = rate_expression.get_variables();
                            let references_compartments = rate_variables.iter().any(|v| {
                                compartment_map.contains_key(v) || subpopulation_names.contains(v)
                            });

                            transition_flows.push(TransitionFlow {
                                source_index,
                                target_index,
                                rate_expression,
                                references_compartments,
                            });
                        }
                    }
                }
            }
        }
    }

    transition_flows
}

/// Build pre-computed subpopulation mappings for stratifications.
///
/// This function creates mappings for:
/// 1. Subpopulation totals (N_young, N_old, N_young_urban, etc.)
/// 2. Base compartment totals (S, I, R) - sum of all stratified versions
///
/// When stratifications are present, base compartment names (S, I, R) become
/// available as variables representing the sum of all their stratified versions.
/// For example, S = S_young + S_old when age stratification is applied.
fn build_subpopulation_mappings(
    compartments: &[String],
    stratifications: &[commol_core::Stratification],
    bins: &[commol_core::Bin],
) -> Vec<SubpopulationMapping> {
    if stratifications.is_empty() {
        return Vec::new();
    }

    let mut subpopulation_map: HashMap<String, Vec<usize>> = HashMap::new();

    // Build mappings for subpopulation totals (N_young, N_old, etc.)
    for (compartment_index, compartment_name) in compartments.iter().enumerate() {
        let categories: Vec<_> = compartment_name.split('_').skip(1).collect();

        if !categories.is_empty() {
            // Generate all non-empty subsets using bitmask iteration
            for subset_mask in 1..(1 << categories.len()) {
                let subset: Vec<&str> = categories
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (subset_mask >> k) & 1 == 1)
                    .map(|(_, category)| *category)
                    .collect();

                let combination_name = subset.join("_");
                subpopulation_map
                    .entry(combination_name)
                    .or_default()
                    .push(compartment_index);
            }
        }
    }

    // Build mappings for base compartment totals (S, I, R)
    // Each base compartment name maps to the sum of all its stratified versions
    let mut base_compartment_map: HashMap<String, Vec<usize>> = HashMap::new();

    for bin in bins {
        let bin_id = &bin.id;
        for (compartment_index, compartment_name) in compartments.iter().enumerate() {
            // Check if this compartment belongs to this bin
            // (starts with bin_id followed by underscore)
            if compartment_name.starts_with(bin_id)
                && compartment_name
                    .chars()
                    .nth(bin_id.len())
                    .map_or(false, |c| c == '_')
            {
                base_compartment_map
                    .entry(bin_id.clone())
                    .or_default()
                    .push(compartment_index);
            }
        }
    }

    // Convert subpopulation mappings to vector
    let mut mappings: Vec<SubpopulationMapping> = subpopulation_map
        .into_iter()
        .map(|(combination_name, indices)| SubpopulationMapping {
            contributing_compartment_indices: indices,
            parameter_name: format!("N_{}", combination_name),
        })
        .collect();

    // Add base compartment mappings
    // These use the bin name directly (S, I, R) instead of N_ prefix
    for (bin_name, indices) in base_compartment_map {
        mappings.push(SubpopulationMapping {
            contributing_compartment_indices: indices,
            parameter_name: bin_name,
        });
    }

    // Build mappings for partial bin-stratification sums
    // These represent the sum of all compartments for a given bin that match
    // a partial subset of stratification categories. Only meaningful with 2+
    // stratifications (with 1 stratification, partial sums would duplicate
    // either the full compartment or the base compartment total).
    if stratifications.len() >= 2 {
        let mut bin_strat_partial_map: HashMap<String, Vec<usize>> = HashMap::new();

        for bin in bins {
            let bin_id = &bin.id;
            let bin_prefix_len = bin_id.len();

            for (compartment_index, compartment_name) in compartments.iter().enumerate() {
                // Only process compartments belonging to this bin
                if !compartment_name.starts_with(bin_id.as_str())
                    || compartment_name
                        .chars()
                        .nth(bin_prefix_len)
                        .map_or(true, |c| c != '_')
                {
                    continue;
                }

                // Extract categories from the stratification suffix
                let strat_suffix = &compartment_name[bin_prefix_len + 1..];
                let categories: Vec<&str> = strat_suffix.split('_').collect();
                let num_cats = categories.len();

                // Iterate over all proper non-empty subsets (exclude full mask)
                let full_mask: u32 = (1 << num_cats) - 1;
                for subset_mask in 1..full_mask {
                    let subset: Vec<&str> = categories
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| (subset_mask >> *k) & 1 == 1)
                        .map(|(_, cat)| *cat)
                        .collect();

                    let var_name = format!("{}_{}", bin_id, subset.join("_"));
                    bin_strat_partial_map
                        .entry(var_name)
                        .or_default()
                        .push(compartment_index);
                }
            }
        }

        for (var_name, indices) in bin_strat_partial_map {
            mappings.push(SubpopulationMapping {
                contributing_compartment_indices: indices,
                parameter_name: var_name,
            });
        }
    }

    mappings
}
