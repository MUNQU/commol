use epimodel_core::{MathExpressionContext, Model, RateMathExpression, StratifiedRate, Transition};
use std::collections::HashMap;

#[derive(Clone)]
pub struct DifferenceEquations {
    compartments: Vec<String>,
    compartment_map: HashMap<String, usize>,
    population: Vec<f64>,
    transitions: Vec<Transition>,
    expression_context: MathExpressionContext,
    current_step: f64,
    stratifications: Vec<epimodel_core::Stratification>,
    // Store initial state for reset functionality
    initial_population: Vec<f64>,
}

impl DifferenceEquations {
    /// Create a new DifferenceEquations instance from a model
    pub fn from_model(model: &Model) -> Self {
        // Generate all compartment combinations
        let mut compartments = Vec::new();
        let mut compartment_map = HashMap::new();

        // Start with just disease states
        for ds in &model.population.disease_states {
            compartments.push(ds.id.clone());
        }

        // Iteratively apply stratifications
        for strat in &model.population.stratifications {
            let mut next_compartments = Vec::new();
            for comp in &compartments {
                for cat in &strat.categories {
                    next_compartments.push(format!("{}_{}", comp, cat));
                }
            }
            compartments = next_compartments;
        }

        // Create the compartment map for quick lookups
        for (i, comp) in compartments.iter().enumerate() {
            compartment_map.insert(comp.clone(), i);
        }

        // Initialize population distribution
        let disease_state_fraction_map: HashMap<String, f64> = model
            .population
            .initial_conditions
            .disease_state_fractions
            .iter()
            .map(|dsf| (dsf.disease_state.clone(), dsf.fraction))
            .collect();

        let mut pop_dist: HashMap<String, f64> = model
            .population
            .disease_states
            .iter()
            .map(|ds| {
                let fraction = disease_state_fraction_map.get(&ds.id).unwrap_or(&0.0);
                (
                    ds.id.clone(),
                    model.population.initial_conditions.population_size as f64 * fraction,
                )
            })
            .collect();

        for strat in &model.population.stratifications {
            let mut next_pop_dist = HashMap::new();

            // Find the stratification fractions for this stratification
            let strat_fractions_opt = model
                .population
                .initial_conditions
                .stratification_fractions
                .iter()
                .find(|sf| sf.stratification == strat.id);

            if let Some(strat_fractions_item) = strat_fractions_opt {
                let mut strat_fractions = HashMap::new();
                for frac in &strat_fractions_item.fractions {
                    strat_fractions.insert(frac.category.clone(), frac.fraction);
                }

                for (comp, pop) in &pop_dist {
                    for cat in &strat.categories {
                        let fraction = strat_fractions.get(cat).unwrap_or(&0.0);
                        next_pop_dist.insert(format!("{}_{}", comp, cat), pop * fraction);
                    }
                }
                pop_dist = next_pop_dist;
            }
        }

        let num_compartments = compartments.len();
        let population: Vec<f64> = (0..num_compartments)
            .map(|i| *pop_dist.get(&compartments[i]).unwrap_or(&0.0))
            .collect();

        // Store parameters for quick lookup
        let parameters: HashMap<String, f64> = model
            .parameters
            .iter()
            .map(|p| (p.id.clone(), p.value))
            .collect();

        // Clone transitions for use in step function
        let transitions = model.dynamics.transitions.clone();

        // Initialize expression context
        let mut expression_context = MathExpressionContext::new();
        expression_context.set_parameters(parameters);

        // Clone stratifications for later use
        let stratifications = model.population.stratifications.clone();

        // Store initial population for reset functionality
        let initial_population = population.clone();

        Self {
            compartments,
            compartment_map,
            population,
            transitions,
            expression_context,
            current_step: 0.0,
            stratifications,
            initial_population,
        }
    }

    /// Extract stratification categories from a compartment name
    ///
    /// Example: "S_young_urban" with disease_state "S" and stratifications ["age", "location"]
    /// Returns: HashMap { "age" -> "young", "location" -> "urban" }
    fn extract_stratifications(
        &self,
        comp_name: &str,
        disease_state: &str,
    ) -> HashMap<String, String> {
        let mut result = HashMap::new();

        // Remove disease state prefix
        if !comp_name.starts_with(disease_state) {
            return result;
        }

        // Get the stratification part (everything after disease state and first underscore)
        let strat_part = &comp_name[disease_state.len()..];
        if strat_part.is_empty() {
            return result; // No stratifications
        }

        // Remove leading underscore
        let strat_part = if let Some(stripped) = strat_part.strip_prefix('_') {
            stripped
        } else {
            return result; // Invalid format
        };

        // Split by underscore to get categories
        let categories: Vec<&str> = strat_part.split('_').collect();

        // Match categories with stratification IDs (in order)
        for (i, stratification) in self.stratifications.iter().enumerate() {
            if i < categories.len() {
                result.insert(stratification.id.clone(), categories[i].to_string());
            }
        }

        result
    }

    /// Get the appropriate rate string for a compartment based on its stratifications
    fn get_rate_string_for_compartment(
        &self,
        transition: &Transition,
        strat_values: &HashMap<String, String>,
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
            for strat_cond in &stratified_rate.conditions {
                match strat_values.get(&strat_cond.stratification) {
                    Some(actual_category) if actual_category == &strat_cond.category => {
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
        transition.rate.as_ref().map(|r| match r {
            RateMathExpression::Parameter(p) => p.clone(),
            RateMathExpression::Formula(f) => f.formula.clone(),
            RateMathExpression::Constant(c) => c.to_string(),
        })
    }
}

impl DifferenceEquations {
    pub fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    pub fn compartments(&self) -> Vec<String> {
        self.compartments.clone()
    }

    pub fn step(&mut self) -> Result<(), String> {
        let mut flows = vec![0.0; self.compartments.len()];

        // Update expression context with current population values
        self.expression_context.set_step(self.current_step);

        // Calculate and set total population N
        let total_population: f64 = self.population.iter().sum();
        self.expression_context
            .set_parameter("N".to_string(), total_population);

        // Set t as an alias for step (for convenience in formulas)
        self.expression_context
            .set_parameter("t".to_string(), self.current_step);

        for (i, comp_name) in self.compartments.iter().enumerate() {
            self.expression_context
                .set_compartment(comp_name.clone(), self.population[i]);
        }

        let mut subpopulation_totals: HashMap<String, f64> = HashMap::new();

        for (i, comp_name) in self.compartments.iter().enumerate() {
            let pop_value = self.population[i];
            let categories: Vec<_> = comp_name.split('_').skip(1).collect();

            if categories.is_empty() {
                continue;
            }

            // Iterate through all non-empty subsets of categories for this compartment
            for i in 1..(1 << categories.len()) {
                let mut subset = Vec::new();
                for (k, category) in categories.iter().enumerate() {
                    if (i >> k) & 1 == 1 {
                        subset.push(*category);
                    }
                }

                let combo_name: String = subset.join("_");

                *subpopulation_totals.entry(combo_name).or_insert(0.0) += pop_value;
            }
        }

        for (combo_name, total) in subpopulation_totals {
            let var_name = format!("N_{}", combo_name);
            self.expression_context.set_parameter(var_name, total);
        }

        for transition in &self.transitions {
            if !transition.source.is_empty() && !transition.target.is_empty() {
                let source_disease_state = &transition.source[0];
                let target_disease_state = &transition.target[0];

                // Process each compartment
                for (i, comp_name) in self.compartments.iter().enumerate() {
                    if comp_name.starts_with(source_disease_state) {
                        let source_idx = i;
                        let source_pop = self.population[source_idx];

                        // Construct the target compartment name
                        let target_comp_name =
                            comp_name.replacen(source_disease_state, target_disease_state, 1);

                        if let Some(target_idx) = self.compartment_map.get(&target_comp_name) {
                            // Extract stratifications for this compartment
                            let strat_values =
                                self.extract_stratifications(comp_name, source_disease_state);

                            // Get the appropriate rate for this compartment
                            let rate_result =
                                self.get_rate_string_for_compartment(transition, &strat_values);

                            if let Some(rate_str) = rate_result {
                                // Parse and evaluate the rate
                                let rate_expr = RateMathExpression::from_string(rate_str.clone());
                                let rate = match rate_expr.evaluate(&self.expression_context) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        return Err(format!(
                                            "Failed to evaluate rate for transition '{}' in compartment '{}': {}",
                                            transition.id, comp_name, e
                                        ));
                                    }
                                };

                                // Check if rate expression references compartment variables
                                let rate_vars = rate_expr.get_variables();
                                let references_compartments = rate_vars
                                    .iter()
                                    .any(|v| self.compartment_map.contains_key(v));

                                let flow = if references_compartments {
                                    // Absolute rate: use directly
                                    rate
                                } else {
                                    // Per-capita rate: multiply by source population
                                    source_pop * rate
                                };

                                flows[source_idx] -= flow;
                                flows[*target_idx] += flow;
                            }
                        }
                    }
                }
            }
        }

        // Apply the calculated flows to the population vector.
        for (i, flow) in flows.iter().enumerate().take(self.population.len()) {
            self.population[i] += flow;
        }

        // Increment step
        self.current_step += 1.0;

        Ok(())
    }

    pub fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
        // Pre-allocate memory for efficiency
        let mut steps = Vec::with_capacity(num_steps as usize + 1);

        // Store initial state (t=0)
        steps.push(self.population.clone());

        for _ in 0..num_steps {
            self.step()?;
            steps.push(self.population.clone());
        }

        Ok(steps)
    }
}

// Implement SimulationEngine trait for DifferenceEquations
impl epimodel_core::SimulationEngine for DifferenceEquations {
    fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
        // Delegate to existing implementation
        DifferenceEquations::run(self, num_steps)
    }

    fn step(&mut self) -> Result<(), String> {
        // Delegate to existing implementation
        DifferenceEquations::step(self)
    }

    fn compartments(&self) -> Vec<String> {
        self.compartments.clone()
    }

    fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    fn reset(&mut self) {
        // Reset population to initial state
        self.population = self.initial_population.clone();
        // Reset step counter
        self.current_step = 0.0;
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.expression_context
            .set_parameter(parameter_id.to_string(), value);
        Ok(())
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        self.expression_context.get_parameters()
    }

    fn current_step(&self) -> f64 {
        self.current_step
    }
}
