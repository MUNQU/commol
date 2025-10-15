use epimodel_core::{MathExpressionContext, Model, RateMathExpression, StratifiedRate, Transition};
use std::collections::HashMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass(name = "DifferenceEquations"))]
pub struct DifferenceEquations {
    compartments: Vec<String>,
    compartment_map: HashMap<String, usize>,
    population: Vec<f64>,
    transitions: Vec<Transition>,
    expression_context: MathExpressionContext,
    current_step: f64,
    stratifications: Vec<epimodel_core::Stratification>,
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
        let population = (0..num_compartments)
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

        Self {
            compartments,
            compartment_map,
            population,
            transitions,
            expression_context,
            current_step: 0.0,
            stratifications,
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
        let strat_part = if strat_part.starts_with('_') {
            &strat_part[1..]
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
    /// Returns (rate_string, is_from_stratified_rates)
    fn get_rate_string_for_compartment(
        &self,
        transition: &Transition,
        strat_values: &HashMap<String, String>,
    ) -> Option<(String, bool)> {
        // If no stratified rates defined, use default rate
        if transition.stratified_rates.is_none() {
            return transition.rate.as_ref().map(|r| {
                let rate_str = match r {
                    RateMathExpression::Parameter(p) => p.clone(),
                    RateMathExpression::Formula(f) => f.formula.clone(),
                    RateMathExpression::Constant(c) => c.to_string(),
                };
                (rate_str, false)
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
            return Some((matched_rate.rate.clone(), true));
        }

        // Fall back to default rate
        transition.rate.as_ref().map(|r| {
            let rate_str = match r {
                RateMathExpression::Parameter(p) => p.clone(),
                RateMathExpression::Formula(f) => f.formula.clone(),
                RateMathExpression::Constant(c) => c.to_string(),
            };
            (rate_str, false)
        })
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl DifferenceEquations {
    #[new]
    pub fn new(model: PyRef<Model>) -> Self {
        Self::from_model(&model)
    }

    #[getter]
    pub fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    #[getter]
    pub fn compartments(&self) -> Vec<String> {
        self.compartments.clone()
    }

    pub fn step(&mut self) -> PyResult<()> {
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
                for k in 0..categories.len() {
                    if (i >> k) & 1 == 1 {
                        subset.push(categories[k]);
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
                                self.get_rate_string_for_compartment(&transition, &strat_values);

                            if let Some((rate_str, _is_stratified)) = rate_result {
                                // Parse and evaluate the rate
                                let rate_expr = RateMathExpression::from_string(rate_str.clone());
                                let rate = match rate_expr.evaluate(&self.expression_context) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                            format!(
                                                "Failed to evaluate rate for transition '{}' in compartment '{}': {}",
                                                transition.id, comp_name, e
                                            ),
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
        for i in 0..self.population.len() {
            self.population[i] += flows[i];
        }

        // Increment step
        self.current_step += 1.0;

        Ok(())
    }

    pub fn run(&mut self, num_steps: u32) -> PyResult<Vec<Vec<f64>>> {
        // Pre-allocate memory for efficiency
        let mut steps = Vec::with_capacity(num_steps as usize + 1);

        // Store initial state (t=0)
        steps.push(self.population.clone());

        for _ in 0..num_steps {
            self.step()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            steps.push(self.population.clone());
        }

        Ok(steps)
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn epimodel_difference(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DifferenceEquations>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use epimodel_core::*;

    fn create_test_sir_model() -> Model {
        let disease_states = vec![
            DiseaseState {
                id: "S".to_string(),
                name: "Susceptible".to_string(),
            },
            DiseaseState {
                id: "I".to_string(),
                name: "Infected".to_string(),
            },
            DiseaseState {
                id: "R".to_string(),
                name: "Recovered".to_string(),
            },
        ];

        let parameters = vec![
            Parameter {
                id: "beta".to_string(),
                value: 0.3,
                description: Some("Transmission rate".to_string()),
            },
            Parameter {
                id: "gamma".to_string(),
                value: 0.1,
                description: Some("Recovery rate".to_string()),
            },
            Parameter {
                id: "N".to_string(),
                value: 1000.0,
                description: Some("Population size".to_string()),
            },
        ];

        let transitions = vec![
            Transition {
                id: "infection".to_string(),
                source: vec!["S".to_string()],
                target: vec!["I".to_string()],
                rate: Some(RateMathExpression::Formula(MathExpression::new(
                    "beta * S * I / N".to_string(),
                ))),
                stratified_rates: None,
                condition: None,
            },
            Transition {
                id: "recovery".to_string(),
                source: vec!["I".to_string()],
                target: vec!["R".to_string()],
                rate: Some(RateMathExpression::Parameter("gamma".to_string())),
                stratified_rates: None,
                condition: None,
            },
        ];

        let disease_state_fractions = vec![
            DiseaseStateFraction {
                disease_state: "S".to_string(),
                fraction: 0.99,
            },
            DiseaseStateFraction {
                disease_state: "I".to_string(),
                fraction: 0.01,
            },
            DiseaseStateFraction {
                disease_state: "R".to_string(),
                fraction: 0.0,
            },
        ];

        let initial_conditions = InitialConditions {
            population_size: 1000,
            disease_state_fractions,
            stratification_fractions: vec![],
        };

        let population = Population {
            disease_states,
            stratifications: vec![],
            transitions: transitions.clone(),
            initial_conditions,
        };

        let dynamics = Dynamics {
            typology: ModelTypes::DifferenceEquations,
            transitions,
        };

        Model {
            name: "Test SIR Model".to_string(),
            description: Some("SIR model for testing".to_string()),
            version: Some("1.0".to_string()),
            population,
            parameters,
            dynamics,
        }
    }

    #[test]
    fn test_difference_equations_creation() {
        let model = create_test_sir_model();
        let de = DifferenceEquations::from_model(&model);

        assert_eq!(de.compartments.len(), 3);
        assert_eq!(de.compartments, vec!["S", "I", "R"]);
        assert_eq!(de.population.len(), 3);

        // Check initial populations
        assert!((de.population[0] - 990.0).abs() < 1e-6); // S
        assert!((de.population[1] - 10.0).abs() < 1e-6); // I
        assert!((de.population[2] - 0.0).abs() < 1e-6); // R
    }

    #[test]
    fn test_formula_evaluation() {
        let model = create_test_sir_model();
        let mut de = DifferenceEquations::from_model(&model);

        // Store initial populations for comparison
        let initial_s = de.population[0];
        let initial_i = de.population[1];
        let initial_r = de.population[2];

        // Run one step
        de.step().unwrap();

        // Check that populations have changed according to the formulas
        let new_s = de.population[0];
        let new_i = de.population[1];
        let new_r = de.population[2];

        // S should decrease (people getting infected)
        assert!(new_s < initial_s);

        // R should increase (people recovering)
        assert!(new_r > initial_r);

        // Total population should remain constant
        let total_initial = initial_s + initial_i + initial_r;
        let total_new = new_s + new_i + new_r;
        assert!((total_initial - total_new).abs() < 1e-10);
    }

    #[test]
    fn test_step_progression() {
        let model = create_test_sir_model();
        let mut de = DifferenceEquations::from_model(&model);

        assert_eq!(de.current_step, 0.0);

        de.step().unwrap();
        assert_eq!(de.current_step, 1.0);

        de.step().unwrap();
        assert_eq!(de.current_step, 2.0);
    }

    #[test]
    fn test_run_multiple_steps() {
        let model = create_test_sir_model();
        let mut de = DifferenceEquations::from_model(&model);

        let results = de.run(10).unwrap();

        // Should have 11 time points (0 through 10)
        assert_eq!(results.len(), 11);

        // Each time point should have 3 compartments
        for step_result in &results {
            assert_eq!(step_result.len(), 3);
        }

        // Initial state should be preserved
        assert!((results[0][0] - 990.0).abs() < 1e-6); // S
        assert!((results[0][1] - 10.0).abs() < 1e-6); // I
        assert!((results[0][2] - 0.0).abs() < 1e-6); // R

        // Population should be conserved at all time points
        for step_result in &results {
            let total: f64 = step_result.iter().sum();
            assert!((total - 1000.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_constant_rate_expression() {
        let mut model = create_test_sir_model();

        // Replace recovery with constant rate
        model.dynamics.transitions[1].rate = Some(RateMathExpression::Constant(0.05));
        model.population.transitions[1].rate = Some(RateMathExpression::Constant(0.05));

        let mut de = DifferenceEquations::from_model(&model);

        let initial_i = de.population[1];
        let initial_r = de.population[2];

        de.step().unwrap();

        // Recovery should follow constant rate
        let expected_recovery = initial_i * 0.05;
        let actual_recovery = de.population[2] - initial_r;

        assert!((actual_recovery - expected_recovery).abs() < 1e-10);
    }
}
