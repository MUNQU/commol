use epimodel_core::{MathExpressionContext, Model, Transition};
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
        let mut pop_dist: HashMap<String, f64> = model
            .population
            .disease_states
            .iter()
            .map(|ds| {
                let fraction = model
                    .population
                    .initial_conditions
                    .disease_state_fraction
                    .get(&ds.id)
                    .unwrap_or(&0.0);
                (
                    ds.id.clone(),
                    model.population.initial_conditions.population_size as f64 * fraction,
                )
            })
            .collect();

        for strat in &model.population.stratifications {
            let mut next_pop_dist = HashMap::new();
            let strat_fractions = model
                .population
                .initial_conditions
                .stratification_fractions
                .get(&strat.id)
                .unwrap();
            for (comp, pop) in &pop_dist {
                for cat in &strat.categories {
                    let fraction = strat_fractions.get(cat).unwrap_or(&0.0);
                    next_pop_dist.insert(format!("{}_{}", comp, cat), pop * fraction);
                }
            }
            pop_dist = next_pop_dist;
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

        Self {
            compartments,
            compartment_map,
            population,
            transitions,
            expression_context,
            current_step: 0.0,
        }
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

        for transition in &self.transitions {
            if let Some(rate_expr) = &transition.rate {
                if !transition.source.is_empty() && !transition.target.is_empty() {
                    // Evaluate the rate expression
                    let rate = match rate_expr.evaluate(&self.expression_context) {
                        Ok(r) => r,
                        Err(e) => {
                            // Return error instead of just warning
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "Failed to evaluate rate for transition '{}': {}",
                                transition.id, e
                            )));
                        }
                    };

                    let source_disease_state = &transition.source[0];
                    let target_disease_state = &transition.target[0];

                    for (i, comp_name) in self.compartments.iter().enumerate() {
                        if comp_name.starts_with(source_disease_state) {
                            let source_idx = i;
                            let source_pop = self.population[source_idx];

                            // Construct the target compartment name by replacing the disease state
                            // part.
                            let target_comp_name =
                                comp_name.replacen(source_disease_state, target_disease_state, 1);
                            if let Some(target_idx) = self.compartment_map.get(&target_comp_name) {
                                // Check if rate expression references compartment variables
                                // If it does, it's an absolute rate (total flow)
                                // If not, it's a per-capita rate that needs to be multiplied by source pop
                                let rate_vars = transition.rate.as_ref().unwrap().get_variables();
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
    use std::collections::HashMap;

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
                condition: None,
            },
            Transition {
                id: "recovery".to_string(),
                source: vec!["I".to_string()],
                target: vec!["R".to_string()],
                rate: Some(RateMathExpression::Parameter("gamma".to_string())),
                condition: None,
            },
        ];

        let mut disease_state_fraction = HashMap::new();
        disease_state_fraction.insert("S".to_string(), 0.99);
        disease_state_fraction.insert("I".to_string(), 0.01);
        disease_state_fraction.insert("R".to_string(), 0.0);

        let initial_conditions = InitialConditions {
            population_size: 1000,
            disease_state_fraction,
            stratification_fractions: HashMap::new(),
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
