//! Model builder for constructing DifferenceEquations from a compartment model.

use crate::helpers::{extract_stratifications, get_rate_string_for_compartment};
use crate::types::{DifferenceEquations, SubpopulationMapping, TransitionFlow};
use commol_core::{MathExpressionContext, Model, RateMathExpression};
use std::collections::HashMap;

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

        // Pre-compute all transition flows
        let transition_flows = build_transition_flows(
            model,
            &compartments,
            &compartment_map,
            &model.population.stratifications,
        );

        // Initialize compartment flows buffer
        let num_compartments = compartments.len();
        let compartment_flows = vec![0.0; num_compartments];

        // Pre-compute subpopulation mappings for stratifications
        let subpopulation_mappings =
            build_subpopulation_mappings(&compartments, &model.population.stratifications);

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

/// Generate all stratified compartment combinations.
///
/// Returns a tuple of (compartments vector, compartment_map for lookups).
fn generate_compartments(model: &Model) -> (Vec<String>, HashMap<String, usize>) {
    // Start with disease states
    let mut compartments: Vec<String> = model
        .population
        .bins
        .iter()
        .map(|ds| ds.id.clone())
        .collect();

    // Iteratively apply stratifications
    for stratification in &model.population.stratifications {
        compartments = compartments
            .iter()
            .flat_map(|compartment_name| {
                stratification
                    .categories
                    .iter()
                    .map(move |category| format!("{}_{}", compartment_name, category))
            })
            .collect();
    }

    // Create the compartment map for quick lookups
    let compartment_map: HashMap<String, usize> = compartments
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();

    (compartments, compartment_map)
}

/// Initialize population distribution across compartments.
fn initialize_population(model: &Model, compartments: &[String]) -> Vec<f64> {
    let total_population = model.population.initial_conditions.population_size as f64;

    // Build bin fraction map
    // Note: None fractions indicate calibration is needed
    // We store them as Option to distinguish from 0.0
    let bin_fraction_map: HashMap<String, Option<f64>> = model
        .population
        .initial_conditions
        .bin_fractions
        .iter()
        .map(|bf| (bf.bin.clone(), bf.fraction))
        .collect();

    // Initialize with bins
    let mut population_distribution: HashMap<String, f64> = model
        .population
        .bins
        .iter()
        .map(|bin| {
            let fraction = bin_fraction_map
                .get(&bin.id)
                .and_then(|f| *f)
                .unwrap_or(0.0);
            (bin.id.clone(), total_population * fraction)
        })
        .collect();

    // Apply stratifications iteratively
    for stratification in &model.population.stratifications {
        // Find the stratification fractions for this stratification
        let stratification_fractions_item = model
            .population
            .initial_conditions
            .stratification_fractions
            .iter()
            .find(|sf| sf.stratification == stratification.id);

        if let Some(fractions_item) = stratification_fractions_item {
            // Build fraction map for this stratification
            let fraction_map: HashMap<String, f64> = fractions_item
                .fractions
                .iter()
                .map(|frac| (frac.category.clone(), frac.fraction))
                .collect();

            // Apply stratification to all current compartments
            let mut stratified_distribution = HashMap::new();
            for (compartment_name, population) in &population_distribution {
                for category in &stratification.categories {
                    let fraction = fraction_map.get(category).unwrap_or(&0.0);
                    stratified_distribution.insert(
                        format!("{}_{}", compartment_name, category),
                        population * fraction,
                    );
                }
            }

            population_distribution = stratified_distribution;
        }
    }

    // Convert to vector indexed by compartment order
    compartments
        .iter()
        .map(|comp| *population_distribution.get(comp).unwrap_or(&0.0))
        .collect()
}

/// Build pre-computed transition flows.
fn build_transition_flows(
    model: &Model,
    compartments: &[String],
    compartment_map: &HashMap<String, usize>,
    stratifications: &[commol_core::Stratification],
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

                    // Construct the target compartment name
                    let target_compartment_name =
                        compartment_name.replacen(source_bin, target_bin, 1);

                    if let Some(&target_index) = compartment_map.get(&target_compartment_name) {
                        // Extract stratifications for this compartment
                        let stratification_values =
                            extract_stratifications(compartment_name, source_bin, stratifications);

                        // Get the appropriate rate for this compartment
                        if let Some(rate_string) =
                            get_rate_string_for_compartment(transition, &stratification_values)
                        {
                            // Parse the rate expression once
                            let rate_expression =
                                RateMathExpression::from_string(rate_string.clone());

                            // Check if rate expression references compartment variables
                            let rate_variables = rate_expression.get_variables();
                            let references_compartments = rate_variables
                                .iter()
                                .any(|v| compartment_map.contains_key(v));

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
fn build_subpopulation_mappings(
    compartments: &[String],
    stratifications: &[commol_core::Stratification],
) -> Vec<SubpopulationMapping> {
    if stratifications.is_empty() {
        return Vec::new();
    }

    let mut subpopulation_map: HashMap<String, Vec<usize>> = HashMap::new();

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

    // Convert to vector for faster iteration during simulation
    subpopulation_map
        .into_iter()
        .map(|(combination_name, indices)| SubpopulationMapping {
            contributing_compartment_indices: indices,
            parameter_name: format!("N_{}", combination_name),
        })
        .collect()
}
