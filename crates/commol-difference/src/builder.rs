//! Model builder for constructing DifferenceEquations from a compartment model.

use crate::helpers::{
    compute_target_with_category_overrides, extract_stratifications,
    get_rate_string_for_compartment, has_category_overrides, replace_bin_in_rate,
};
use crate::types::{
    AccumulatorOutputRef, DifferenceEquations, FastRateExpression, FastRateOp,
    GeneratedAccumulatorOutput, SubpopulationLayout, SubpopulationMapping, TimeSeriesParameter,
    TransitionFlow, VarSlot,
};
use commol_core::math_expression::jit::ast::{BinaryOperator, Expr, UnaryOperator};
use commol_core::math_expression::jit::parser::parse_expression;
use commol_core::{MathExpressionContext, Model, RateMathExpression};
use std::collections::{HashMap, HashSet};

/// Time-series parameter payload collected before parameter slots are
/// reserved, so the final [`TimeSeriesParameter`] can be constructed with the
/// correct index up front.
struct TimeSeriesPayload {
    name: String,
    data: Vec<(u64, f64)>,
    mode: commol_core::SeriesMode,
}

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
        // Generate all compartment and accumulator output combinations
        let (compartments, compartment_map) = generate_compartments(model);
        let accumulator_outputs = generate_accumulators(model);
        let accumulator_names: Vec<String> = accumulator_outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        let accumulator_map: HashMap<AccumulatorOutputRef, usize> = accumulator_outputs
            .iter()
            .enumerate()
            .map(|(idx, output)| (output.output_ref.clone(), idx))
            .collect();
        let output_names: Vec<String> = compartments
            .iter()
            .chain(&accumulator_names)
            .cloned()
            .collect();

        // Initialize population distribution
        let population = initialize_population(model, &compartments);

        // Store constant parameters for quick lookup
        // Formula parameters will be evaluated on each step
        let mut constant_parameters: HashMap<String, f64> = HashMap::new();
        let mut formula_parameters: Vec<(String, RateMathExpression)> = Vec::new();
        // Collect time-series payloads first; final struct construction happens
        // after parameter slots are reserved so indices can be filled in once,
        // up front, instead of being patched in afterwards.
        let mut series_payloads: Vec<TimeSeriesPayload> = Vec::new();

        for p in &model.parameters {
            match &p.value {
                Some(commol_core::ParameterValue::Constant(val)) => {
                    constant_parameters.insert(p.id.clone(), *val);
                }
                Some(commol_core::ParameterValue::Formula(formula)) => {
                    let rate_expr = RateMathExpression::from_string(formula.clone());
                    formula_parameters.push((p.id.clone(), rate_expr));
                }
                Some(commol_core::ParameterValue::TimeSeries { data, mode }) => {
                    series_payloads.push(TimeSeriesPayload {
                        name: p.id.clone(),
                        data: data.clone(),
                        mode: mode.clone(),
                    });
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
        // (must be done before transition flows, as absolute-flow inference
        // needs to know about subpopulation variable names)
        let subpopulation_layout = build_subpopulation_mappings(
            &compartments,
            &model.population.stratifications,
            &model.population.bins,
        );

        let subpopulation_param_names: HashSet<String> = subpopulation_layout
            .iter()
            .map(|m| m.parameter_name.clone())
            .collect();

        // Pre-allocate every parameter slot the per-step loop will touch so
        // `set_parameter_str` can update in place and JIT input resolution can
        // use direct indexed parameter reads where possible.
        expression_context.reserve_parameters(
            std::iter::once("N".to_string())
                .chain(std::iter::once("t".to_string()))
                .chain(model.parameters.iter().map(|p| p.id.clone()))
                .chain(
                    subpopulation_layout
                        .iter()
                        .map(|m| m.parameter_name.clone()),
                )
                .chain(series_payloads.iter().map(|payload| payload.name.clone()))
                .chain(formula_parameters.iter().map(|(name, _)| name.clone())),
        );
        let n_parameter_index = expression_context
            .parameter_index("N")
            .expect("N parameter slot must be reserved");
        let t_parameter_index = expression_context
            .parameter_index("t")
            .expect("t parameter slot must be reserved");

        let mut subpopulation_mappings: Vec<SubpopulationMapping> = subpopulation_layout
            .into_iter()
            .map(|layout| {
                let parameter_index = expression_context
                    .parameter_index(&layout.parameter_name)
                    .expect("subpopulation parameter slot must be reserved");
                SubpopulationMapping {
                    contributing_compartment_indices: layout.contributing_compartment_indices,
                    parameter_name: layout.parameter_name,
                    parameter_index,
                }
            })
            .collect();

        let series_parameters: Vec<TimeSeriesParameter> = series_payloads
            .into_iter()
            .map(|payload| {
                let parameter_index = expression_context
                    .parameter_index(&payload.name)
                    .expect("time-series parameter slot must be reserved");
                TimeSeriesParameter::new(payload.name, parameter_index, payload.data, payload.mode)
            })
            .collect();

        // Pre-compute all transition flows
        let transition_flows = build_transition_flows(
            model,
            &compartments,
            &compartment_map,
            &accumulator_map,
            &model.population.stratifications,
            &subpopulation_param_names,
            &expression_context,
        );

        let used_context_variables =
            collect_used_context_variables(&transition_flows, &formula_parameters);
        subpopulation_mappings
            .retain(|mapping| used_context_variables.contains(&mapping.parameter_name));

        // The HashMap-backed compartment context is only needed when an
        // expression must fall back to evalexpr. Both the JIT path (via
        // `resolved_slots`) and the fast path (via `fast_rate_expression`)
        // read compartment values directly from `population`, so they do
        // not require the HashMap to be kept up to date.
        let requires_compartment_context = !formula_parameters.is_empty()
            || transition_flows.iter().any(|flow| {
                matches!(flow.rate_expression, RateMathExpression::Formula(_))
                    && flow.resolved_slots.is_none()
                    && flow.fast_rate_expression.is_none()
            });

        // Initialize compartment flows buffer
        let num_compartments = compartments.len();
        let compartment_flows = vec![0.0; num_compartments];

        Self {
            compartments,
            output_names,
            population,
            accumulators: vec![0.0; accumulator_map.len()],
            expression_context,
            current_step: 0.0,
            initial_population,
            initial_accumulators: vec![0.0; accumulator_map.len()],
            transition_flows,
            compartment_flows,
            subpopulation_mappings,
            formula_parameters,
            series_parameters,
            n_parameter_index,
            t_parameter_index,
            requires_compartment_context,
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
        Some(conds) => conds.iter().all(|c| match &c.category {
            Some(cat) => applied.get(&c.stratification) == Some(cat),
            None => true,
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

fn accumulator_output_ref(
    accumulator_id: String,
    applied: &HashMap<String, String>,
    stratifications: &[commol_core::Stratification],
) -> AccumulatorOutputRef {
    AccumulatorOutputRef {
        accumulator_id,
        categories: stratifications
            .iter()
            .map(|stratification| applied.get(&stratification.id).cloned())
            .collect(),
    }
}

fn generate_accumulators(model: &Model) -> Vec<GeneratedAccumulatorOutput> {
    let mut partials: Vec<(String, String, HashMap<String, String>)> = model
        .population
        .accumulators
        .iter()
        .map(|a| (a.id.clone(), a.id.clone(), HashMap::new()))
        .collect();

    for stratification in &model.population.stratifications {
        let mut new_partials: Vec<(String, String, HashMap<String, String>)> = Vec::new();

        for (accumulator_id, name, applied) in partials {
            if stratification_conditions_met(&stratification.conditions, &applied) {
                for cat in &stratification.categories {
                    let mut new_applied = applied.clone();
                    new_applied.insert(stratification.id.clone(), cat.clone());
                    new_partials.push((
                        accumulator_id.clone(),
                        format!("{}_{}", name, cat),
                        new_applied,
                    ));
                }
            } else {
                new_partials.push((accumulator_id, name, applied));
            }
        }

        partials = new_partials;
    }

    partials
        .into_iter()
        .map(
            |(accumulator_id, name, applied)| GeneratedAccumulatorOutput {
                name,
                output_ref: accumulator_output_ref(
                    accumulator_id,
                    &applied,
                    &model.population.stratifications,
                ),
            },
        )
        .collect()
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
    accumulator_map: &HashMap<AccumulatorOutputRef, usize>,
    stratifications: &[commol_core::Stratification],
    subpopulation_names: &HashSet<String>,
    expression_context: &MathExpressionContext,
) -> Vec<TransitionFlow> {
    let mut transition_flows = Vec::new();

    for transition in &model.dynamics.transitions {
        let source_empty = transition.source.is_empty();
        let target_empty = transition.target.is_empty();

        if source_empty && target_empty {
            continue;
        }

        if source_empty {
            // Source-less transition: add to target compartments from outside the system.
            // Rate is always treated as absolute (no source population to multiply by).
            let target_bin = &transition.target[0];
            for (i, compartment_name) in compartments.iter().enumerate() {
                if compartment_name.starts_with(target_bin) {
                    let stratification_values =
                        extract_stratifications(compartment_name, target_bin, stratifications);
                    if let Some(matched) =
                        get_rate_string_for_compartment(transition, &stratification_values)
                    {
                        let rate_expression =
                            RateMathExpression::from_string(matched.rate_string.clone());
                        let resolved_slots = resolve_jit_slots(
                            &rate_expression,
                            compartment_map,
                            expression_context,
                        );
                        let fast_rate_expression = build_fast_rate_expression(
                            &rate_expression,
                            compartment_map,
                            expression_context,
                        );
                        transition_flows.push(TransitionFlow {
                            source_index: None,
                            target_index: Some(i),
                            accumulator_indices: accumulator_indices_for_transition(
                                transition,
                                target_bin,
                                compartment_name,
                                &stratification_values,
                                stratifications,
                                accumulator_map,
                                matched.stratified_rate.map(|sr| sr.conditions.as_slice()),
                            ),
                            rate_expression,
                            fast_rate_expression,
                            is_absolute_flow: true,
                            resolved_slots,
                        });
                    }
                }
            }
        } else if target_empty {
            // Target-less transition: remove from source compartments out of the system.
            let source_bin = &transition.source[0];
            for (i, compartment_name) in compartments.iter().enumerate() {
                if compartment_name.starts_with(source_bin) {
                    let stratification_values =
                        extract_stratifications(compartment_name, source_bin, stratifications);
                    if let Some(matched) =
                        get_rate_string_for_compartment(transition, &stratification_values)
                    {
                        let rate_expression =
                            RateMathExpression::from_string(matched.rate_string.clone());
                        let is_absolute_flow =
                            match matched.stratified_rate.and_then(|sr| sr.absolute) {
                                Some(abs) => abs,
                                None => {
                                    let rate_variables = rate_expression.get_variables();
                                    rate_variables.iter().any(|v| {
                                        compartment_map.contains_key(v)
                                            || subpopulation_names.contains(v)
                                    })
                                }
                            };
                        let resolved_slots = resolve_jit_slots(
                            &rate_expression,
                            compartment_map,
                            expression_context,
                        );
                        let fast_rate_expression = build_fast_rate_expression(
                            &rate_expression,
                            compartment_map,
                            expression_context,
                        );
                        transition_flows.push(TransitionFlow {
                            source_index: Some(i),
                            target_index: None,
                            accumulator_indices: accumulator_indices_for_transition(
                                transition,
                                source_bin,
                                compartment_name,
                                &stratification_values,
                                stratifications,
                                accumulator_map,
                                matched.stratified_rate.map(|sr| sr.conditions.as_slice()),
                            ),
                            rate_expression,
                            fast_rate_expression,
                            is_absolute_flow,
                            resolved_slots,
                        });
                    }
                }
            }
        } else {
            // Normal transition: move from source to target compartments.
            let source_bin = &transition.source[0];
            let target_bin = &transition.target[0];

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
                                matched.rate_string.clone()
                            };

                            // Parse the rate expression once
                            let rate_expression =
                                RateMathExpression::from_string(rate_string.clone());

                            // Determine absolute vs per-capita mode: the matched
                            // stratified rate's explicit flag takes precedence;
                            // otherwise infer from whether the expression references
                            // any compartment or subpopulation variable.
                            let is_absolute_flow =
                                match matched.stratified_rate.and_then(|sr| sr.absolute) {
                                    Some(abs) => abs,
                                    None => {
                                        let rate_variables = rate_expression.get_variables();
                                        rate_variables.iter().any(|v| {
                                            compartment_map.contains_key(v)
                                                || subpopulation_names.contains(v)
                                        })
                                    }
                                };

                            let resolved_slots = resolve_jit_slots(
                                &rate_expression,
                                compartment_map,
                                expression_context,
                            );
                            let fast_rate_expression = build_fast_rate_expression(
                                &rate_expression,
                                compartment_map,
                                expression_context,
                            );

                            transition_flows.push(TransitionFlow {
                                source_index: Some(source_index),
                                target_index: Some(target_index),
                                accumulator_indices: accumulator_indices_for_transition(
                                    transition,
                                    source_bin,
                                    compartment_name,
                                    &stratification_values,
                                    stratifications,
                                    accumulator_map,
                                    matched.stratified_rate.map(|sr| sr.conditions.as_slice()),
                                ),
                                rate_expression,
                                fast_rate_expression,
                                is_absolute_flow,
                                resolved_slots,
                            });
                        }
                    }
                }
            }
        }
    }

    transition_flows
}

fn accumulator_indices_for_transition(
    transition: &commol_core::Transition,
    _source_bin: &str,
    _source_compartment_name: &str,
    stratification_values: &HashMap<String, String>,
    stratifications: &[commol_core::Stratification],
    accumulator_map: &HashMap<AccumulatorOutputRef, usize>,
    matched_conditions: Option<&[commol_core::StratificationCondition]>,
) -> Vec<usize> {
    transition
        .accumulators
        .iter()
        .filter_map(|accumulator_id| {
            let output_ref = if let Some(conditions) = matched_conditions {
                if has_category_overrides(conditions) {
                    accumulator_output_ref_with_category_overrides(
                        accumulator_id.clone(),
                        stratification_values,
                        stratifications,
                        conditions,
                    )
                } else {
                    accumulator_output_ref(
                        accumulator_id.clone(),
                        stratification_values,
                        stratifications,
                    )
                }
            } else {
                accumulator_output_ref(
                    accumulator_id.clone(),
                    stratification_values,
                    stratifications,
                )
            };
            accumulator_map.get(&output_ref).copied()
        })
        .collect()
}

fn accumulator_output_ref_with_category_overrides(
    accumulator_id: String,
    stratification_values: &HashMap<String, String>,
    stratifications: &[commol_core::Stratification],
    conditions: &[commol_core::StratificationCondition],
) -> AccumulatorOutputRef {
    let override_map: HashMap<&str, &str> = conditions
        .iter()
        .filter_map(|condition| {
            condition
                .to
                .as_ref()
                .map(|target| (condition.stratification.as_str(), target.as_str()))
        })
        .collect();
    let mut target_applied: HashMap<String, String> = HashMap::new();

    for stratification in stratifications {
        let effective_category = override_map
            .get(stratification.id.as_str())
            .map(|category| (*category).to_string())
            .or_else(|| stratification_values.get(&stratification.id).cloned());

        let applies = match &stratification.conditions {
            None => true,
            Some(conditions) => conditions
                .iter()
                .all(|condition| match &condition.category {
                    Some(category) => {
                        target_applied.get(&condition.stratification) == Some(category)
                    }
                    None => true,
                }),
        };

        if applies && let Some(category) = effective_category {
            target_applied.insert(stratification.id.clone(), category);
        }
    }

    accumulator_output_ref(accumulator_id, &target_applied, stratifications)
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
) -> Vec<SubpopulationLayout> {
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
                && (compartment_name.chars().nth(bin_id.len()) == Some('_'))
            {
                base_compartment_map
                    .entry(bin_id.clone())
                    .or_default()
                    .push(compartment_index);
            }
        }
    }

    // Convert subpopulation mappings to vector
    let mut mappings: Vec<SubpopulationLayout> = subpopulation_map
        .into_iter()
        .map(|(combination_name, indices)| SubpopulationLayout {
            contributing_compartment_indices: indices,
            parameter_name: format!("N_{}", combination_name),
        })
        .collect();

    // Add base compartment mappings
    // These use the bin name directly (S, I, R) instead of N_ prefix
    for (bin_name, indices) in base_compartment_map {
        mappings.push(SubpopulationLayout {
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
                    || (compartment_name.chars().nth(bin_prefix_len) != Some('_'))
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
            mappings.push(SubpopulationLayout {
                contributing_compartment_indices: indices,
                parameter_name: var_name,
            });
        }
    }

    mappings
}

fn collect_used_context_variables(
    transition_flows: &[TransitionFlow],
    formula_parameters: &[(String, RateMathExpression)],
) -> HashSet<String> {
    let mut variables = HashSet::new();

    for flow in transition_flows {
        variables.extend(flow.rate_expression.get_variables());
    }

    for (_, expression) in formula_parameters {
        variables.extend(expression.get_variables());
    }

    variables
}

/// Resolve the input variables of a JIT-compiled rate expression to direct
/// storage slots (compartment index, parameter name, or step alias) so the
/// per-step inner loop can fill the JIT input buffer without HashMap probes
/// for compartments. Returns `None` for non-JIT rates (constants, single
/// parameter references, or evalexpr fallbacks) — those have a fast path
/// of their own.
fn resolve_jit_slots(
    rate_expression: &RateMathExpression,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<Vec<VarSlot>> {
    let expr = match rate_expression {
        RateMathExpression::Formula(expr) => expr,
        _ => return None,
    };
    let names = expr.jit_variable_names()?;
    Some(
        names
            .iter()
            .map(|name| resolve_var_slot(name, compartment_map, expression_context))
            .collect(),
    )
}

fn build_fast_rate_expression(
    rate_expression: &RateMathExpression,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<FastRateExpression> {
    let mut ops = Vec::new();
    match rate_expression {
        RateMathExpression::Constant(value) => return Some(FastRateExpression::Constant(*value)),
        RateMathExpression::Parameter(name) => {
            return Some(FastRateExpression::Slot(resolve_var_slot(
                name,
                compartment_map,
                expression_context,
            )));
        }
        RateMathExpression::Formula(expr) => {
            let ast = parse_expression(expr.preprocessed()).ok()?;
            if let Some(specialized) =
                specialized_fast_expr(&ast, compartment_map, expression_context)
            {
                return Some(specialized);
            }
            push_fast_ops_from_ast(&ast, compartment_map, expression_context, &mut ops)?;
        }
    }
    Some(FastRateExpression::Program(ops))
}

fn specialized_fast_expr(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<FastRateExpression> {
    match expr {
        Expr::BinaryOp {
            op: BinaryOperator::Mul,
            left,
            right,
        } => {
            if let Some((a, b, c)) = match_mul3(left, right, compartment_map, expression_context) {
                return Some(FastRateExpression::Mul3(a, b, c));
            }
            if let Some((subtract, factor, value)) =
                match_one_minus_mul3(left, right, compartment_map, expression_context)
            {
                return Some(FastRateExpression::OneMinusMul3 {
                    subtract,
                    factor,
                    value,
                });
            }
            let left = slot_from_ast(left, compartment_map, expression_context)?;
            let right = slot_from_ast(right, compartment_map, expression_context)?;
            Some(FastRateExpression::Mul2(left, right))
        }
        Expr::BinaryOp {
            op: BinaryOperator::Div,
            left,
            right,
        } => match_mul_add_div(left, right, compartment_map, expression_context).map(
            |(first, second, add_left, add_right, denominator)| FastRateExpression::MulAddDiv {
                first,
                second,
                add_left,
                add_right,
                denominator,
            },
        ),
        Expr::Variable(name) => Some(FastRateExpression::Slot(resolve_var_slot(
            name,
            compartment_map,
            expression_context,
        ))),
        Expr::Constant(value) => Some(FastRateExpression::Constant(*value)),
        _ => None,
    }
}

fn match_mul3(
    left: &Expr,
    right: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<(VarSlot, VarSlot, VarSlot)> {
    if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = left
    {
        return Some((
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
            slot_from_ast(right, compartment_map, expression_context)?,
        ));
    }

    if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = right
    {
        return Some((
            slot_from_ast(left, compartment_map, expression_context)?,
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
        ));
    }

    None
}

fn match_one_minus_mul3(
    left: &Expr,
    right: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<(VarSlot, VarSlot, VarSlot)> {
    if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = left
    {
        if let Some(subtract) = match_one_minus(inner_left, compartment_map, expression_context) {
            return Some((
                subtract,
                slot_from_ast(inner_right, compartment_map, expression_context)?,
                slot_from_ast(right, compartment_map, expression_context)?,
            ));
        }
        if let Some(subtract) = match_one_minus(inner_right, compartment_map, expression_context) {
            return Some((
                subtract,
                slot_from_ast(inner_left, compartment_map, expression_context)?,
                slot_from_ast(right, compartment_map, expression_context)?,
            ));
        }
    }

    if let Some(subtract) = match_one_minus(left, compartment_map, expression_context)
        && let Expr::BinaryOp {
            op: BinaryOperator::Mul,
            left: inner_left,
            right: inner_right,
        } = right
    {
        return Some((
            subtract,
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
        ));
    }

    None
}

fn match_one_minus(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<VarSlot> {
    if let Expr::BinaryOp {
        op: BinaryOperator::Sub,
        left,
        right,
    } = expr
        && matches!(left.as_ref(), Expr::Constant(value) if *value == 1.0)
    {
        return slot_from_ast(right, compartment_map, expression_context);
    }
    None
}

fn match_mul_add_div(
    numerator: &Expr,
    denominator: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<(VarSlot, VarSlot, VarSlot, VarSlot, VarSlot)> {
    let denominator = slot_from_ast(denominator, compartment_map, expression_context)?;
    let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left,
        right,
    } = numerator
    else {
        return None;
    };

    let (first, second, add_expr) = if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = left.as_ref()
    {
        (
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
            right.as_ref(),
        )
    } else if let Expr::BinaryOp {
        op: BinaryOperator::Mul,
        left: inner_left,
        right: inner_right,
    } = right.as_ref()
    {
        (
            slot_from_ast(inner_left, compartment_map, expression_context)?,
            slot_from_ast(inner_right, compartment_map, expression_context)?,
            left.as_ref(),
        )
    } else {
        return None;
    };

    let Expr::BinaryOp {
        op: BinaryOperator::Add,
        left: add_left,
        right: add_right,
    } = add_expr
    else {
        return None;
    };

    Some((
        first,
        second,
        slot_from_ast(add_left, compartment_map, expression_context)?,
        slot_from_ast(add_right, compartment_map, expression_context)?,
        denominator,
    ))
}

fn slot_from_ast(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> Option<VarSlot> {
    match expr {
        Expr::Variable(name) => Some(resolve_var_slot(name, compartment_map, expression_context)),
        _ => None,
    }
}

fn push_fast_ops_from_ast(
    expr: &Expr,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
    ops: &mut Vec<FastRateOp>,
) -> Option<()> {
    match expr {
        Expr::Constant(value) => ops.push(FastRateOp::Constant(*value)),
        Expr::Variable(name) => ops.push(FastRateOp::Slot(resolve_var_slot(
            name,
            compartment_map,
            expression_context,
        ))),
        Expr::UnaryOp { op, operand } => match op {
            UnaryOperator::Neg => {
                push_fast_ops_from_ast(operand, compartment_map, expression_context, ops)?;
                ops.push(FastRateOp::Neg);
            }
            UnaryOperator::Not => return None,
        },
        Expr::BinaryOp { op, left, right } => {
            push_fast_ops_from_ast(left, compartment_map, expression_context, ops)?;
            push_fast_ops_from_ast(right, compartment_map, expression_context, ops)?;
            match op {
                BinaryOperator::Add => ops.push(FastRateOp::Add),
                BinaryOperator::Sub => ops.push(FastRateOp::Sub),
                BinaryOperator::Mul => ops.push(FastRateOp::Mul),
                BinaryOperator::Div => ops.push(FastRateOp::Div),
                BinaryOperator::Mod
                | BinaryOperator::Pow
                | BinaryOperator::Lt
                | BinaryOperator::Gt
                | BinaryOperator::Le
                | BinaryOperator::Ge
                | BinaryOperator::Eq
                | BinaryOperator::Ne
                | BinaryOperator::And
                | BinaryOperator::Or => return None,
            }
        }
        Expr::FunctionCall { .. } | Expr::Conditional { .. } => return None,
    }
    Some(())
}

fn resolve_var_slot(
    name: &str,
    compartment_map: &HashMap<String, usize>,
    expression_context: &MathExpressionContext,
) -> VarSlot {
    if let Some(&idx) = compartment_map.get(name) {
        VarSlot::Compartment(idx)
    } else if name == "step" || name == "t" {
        VarSlot::Step
    } else if let Some(idx) = expression_context.parameter_index(name) {
        VarSlot::ParameterIndex(idx)
    } else {
        VarSlot::Parameter(name.to_string())
    }
}
