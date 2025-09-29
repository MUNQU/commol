use epimodel_core::{Model, Transition};
use std::collections::HashMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass(name = "DifferenceEquations"))]
pub struct DifferenceEquations {
    compartments: Vec<String>,
    compartment_map: HashMap<String, usize>,
    population: Vec<f64>,
    parameters: HashMap<String, f64>,
    transitions: Vec<Transition>,
}

#[cfg_attr(feature = "python", pymethods)]
impl DifferenceEquations {
    #[cfg(feature = "python")]
    #[new]
    pub fn new(model: PyRef<Model>) -> Self {
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
        let parameters = model
            .parameters
            .iter()
            .map(|p| (p.id.clone(), p.value))
            .collect();

        // Clone transitions for use in step function
        let transitions = model.dynamics.transitions.clone();

        Self {
            compartments,
            compartment_map,
            population,
            parameters,
            transitions,
        }
    }

    #[cfg(feature = "python")]
    #[getter]
    pub fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    pub fn step(&mut self) {
        let mut flows = vec![0.0; self.compartments.len()];

        for transition in &self.transitions {
            // For this basic version, we assume a simple rate and a single source/target disease state.
            if let Some(rate_id) = &transition.rate {
                if !transition.source.is_empty() && !transition.target.is_empty() {
                    let rate = self.parameters.get(rate_id).unwrap_or(&0.0);
                    let source_disease_state = &transition.source[0];
                    let target_disease_state = &transition.target[0];

                    for (i, comp_name) in self.compartments.iter().enumerate() {
                        if comp_name.starts_with(source_disease_state) {
                            let source_idx = i;
                            let source_pop = self.population[source_idx];

                            // Construct the target compartment name by replacing the disease state part.
                            let target_comp_name =
                                comp_name.replacen(source_disease_state, target_disease_state, 1);
                            if let Some(target_idx) = self.compartment_map.get(&target_comp_name) {
                                let flow = source_pop * rate;
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
    }

    pub fn run(&mut self, num_steps: u32) -> Vec<Vec<f64>> {
        // Pre-allocate memory for efficiency
        let mut history = Vec::with_capacity(num_steps as usize + 1);

        // Store initial state (t=0)
        history.push(self.population.clone());

        for _ in 0..num_steps {
            self.step();
            history.push(self.population.clone());
        }

        history
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn epimodel_difference(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DifferenceEquations>()?;
    Ok(())
}
