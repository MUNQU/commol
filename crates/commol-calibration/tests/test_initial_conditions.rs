use std::collections::HashMap;

use argmin::core::CostFunction;
use commol_calibration::{
    calibrated_initial_population, CalibrationParameter, CalibrationParameterType,
    CalibrationProblem, LossConfig, ObservedDataPoint,
};
use commol_core::{OutputLabels, SimulationEngine};

#[derive(Clone)]
struct TestEngine {
    compartments: Vec<String>,
    initial_population: Vec<f64>,
    population: Vec<f64>,
    parameters: HashMap<String, f64>,
}

impl TestEngine {
    fn new() -> Self {
        let population = vec![300.0, 200.0, 400.0, 100.0];
        Self {
            compartments: vec![
                "A_cat0".to_string(),
                "A_cat1".to_string(),
                "B_cat0".to_string(),
                "B_cat1".to_string(),
            ],
            initial_population: population.clone(),
            population,
            parameters: HashMap::new(),
        }
    }

    fn with_fixed_h() -> Self {
        let population = vec![300.0, 200.0, 100.0, 300.0, 100.0];
        Self {
            compartments: vec![
                "A_cat0".to_string(),
                "A_cat1".to_string(),
                "C".to_string(),
                "B_cat0".to_string(),
                "B_cat1".to_string(),
            ],
            initial_population: population.clone(),
            population,
            parameters: HashMap::new(),
        }
    }

    fn with_alt_categories() -> Self {
        let population = vec![250.0, 250.0, 100.0, 400.0];
        Self {
            compartments: vec![
                "A_seg0".to_string(),
                "A_seg1".to_string(),
                "B_seg0".to_string(),
                "B_seg1".to_string(),
            ],
            initial_population: population.clone(),
            population,
            parameters: HashMap::new(),
        }
    }
}

impl SimulationEngine for TestEngine {
    fn run(&mut self, num_steps: u32) -> Result<Vec<Vec<f64>>, String> {
        Ok((0..=num_steps).map(|_| self.population.clone()).collect())
    }

    fn step(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn compartments(&self) -> Vec<String> {
        self.compartments.clone()
    }

    fn output_labels(&self) -> Vec<OutputLabels> {
        // Names are `<bin>` or `<bin>_<category>` of a single stratification.
        self.compartments
            .iter()
            .map(|name| match name.split_once('_') {
                Some((bin, category)) => OutputLabels {
                    source: bin.to_string(),
                    categories: vec![("group".to_string(), category.to_string())],
                },
                None => OutputLabels {
                    source: name.clone(),
                    categories: vec![],
                },
            })
            .collect()
    }

    fn population(&self) -> Vec<f64> {
        self.population.clone()
    }

    fn reset(&mut self) {
        self.population.clone_from(&self.initial_population);
    }

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn get_parameters(&self) -> &HashMap<String, f64> {
        &self.parameters
    }

    fn current_step(&self) -> f64 {
        0.0
    }

    fn set_initial_condition(
        &mut self,
        compartment_index: usize,
        value: f64,
    ) -> Result<(), String> {
        self.initial_population[compartment_index] = value;
        self.population[compartment_index] = value;
        Ok(())
    }
}

fn initial_condition(id: &str) -> CalibrationParameter {
    CalibrationParameter::with_type_and_guess(
        id.to_string(),
        CalibrationParameterType::InitialCondition,
        0.0,
        1.0,
        0.5,
    )
}

fn initial_population(
    engine: &TestEngine,
    ids: &[&str],
    remainder: Option<&str>,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    let parameters: Vec<_> = ids.iter().map(|id| initial_condition(id)).collect();
    calibrated_initial_population(engine, &parameters, remainder, values, 1000)
        .map(|state| state.population)
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} != {expected:?}");
    }
}

#[test]
fn remainder_absorbs_the_difference_and_keeps_its_split() {
    // A = 0.65 keeps its 300:200 split; B takes the remaining 350 at 400:100.
    let population = initial_population(&TestEngine::new(), &["A"], Some("B"), &[0.65]).unwrap();

    assert_close(&population, &[390.0, 260.0, 280.0, 70.0]);
}

#[test]
fn remainder_leaves_fixed_compartments_alone() {
    // C stays at 100, so B takes 1000 - 300 - 100 = 600 at 300:100.
    let population =
        initial_population(&TestEngine::with_fixed_h(), &["A"], Some("B"), &[0.3]).unwrap();

    assert_close(&population, &[180.0, 120.0, 100.0, 450.0, 150.0]);
}

#[test]
fn single_exact_value_is_free_with_a_remainder() {
    let population =
        initial_population(&TestEngine::new(), &["A_cat1"], Some("B"), &[0.1]).unwrap();

    assert_close(&population, &[300.0, 100.0, 480.0, 120.0]);
}

#[test]
fn without_remainder_values_are_applied_as_given() {
    let population = initial_population(&TestEngine::new(), &["A_cat0"], None, &[0.1]).unwrap();

    assert_close(&population, &[100.0, 200.0, 400.0, 100.0]);
}

#[test]
fn without_remainder_an_excess_over_the_population_is_rejected() {
    let error = initial_population(&TestEngine::new(), &["A_cat0"], None, &[0.5]).unwrap_err();

    assert!(error.contains("exceed the population size"), "{error}");
}

#[test]
fn negative_remainder_is_rejected() {
    let error =
        initial_population(&TestEngine::with_fixed_h(), &["A"], Some("B"), &[0.95]).unwrap_err();

    assert!(error.contains("remainder 'B'"), "{error}");
}

#[test]
fn remainder_overlapping_a_calibrated_value_is_rejected() {
    let error = initial_population(&TestEngine::new(), &["A"], Some("A_cat0"), &[0.5]).unwrap_err();

    assert!(error.contains("overlaps"), "{error}");
}

#[test]
fn category_splits_apply_after_bin_values_whatever_the_parameter_order() {
    // A = 0.6 first (360 / 240), then cat1 = 0.1 splits every pair, and the
    // remainder B takes 400 at the new 450:50 split.
    let population =
        initial_population(&TestEngine::new(), &["cat1", "A"], Some("B"), &[0.1, 0.6]).unwrap();

    assert_close(&population, &[540.0, 60.0, 360.0, 40.0]);
}

#[test]
fn infeasible_initial_conditions_are_penalized() {
    let problem = CalibrationProblem::new(
        TestEngine::with_fixed_h(),
        vec![ObservedDataPoint::new(0, "A_cat0".to_string(), 300.0)],
        vec![initial_condition("A")],
        vec![],
        LossConfig::SumSquaredError,
        1000,
    )
    .unwrap()
    .with_initial_condition_remainder("B")
    .unwrap();

    let feasible = problem.cost(&vec![0.6]).expect("cost should run");
    let infeasible = problem.cost(&vec![0.95]).expect("cost should run");

    assert!(infeasible > feasible);
}

#[test]
fn aggregate_initial_conditions_keep_existing_subgroup_percentages() {
    let problem = CalibrationProblem::new(
        TestEngine::new(),
        vec![ObservedDataPoint::new(0, "A_cat0".to_string(), 390.0)],
        vec![
            CalibrationParameter::with_type_and_guess(
                "A".to_string(),
                CalibrationParameterType::InitialCondition,
                0.1,
                0.9,
                0.6,
            ),
            CalibrationParameter::with_type_and_guess(
                "B".to_string(),
                CalibrationParameterType::InitialCondition,
                0.1,
                0.9,
                0.4,
            ),
        ],
        vec![],
        LossConfig::SumSquaredError,
        1000,
    )
    .expect("aggregate bin IDs should be valid initial-condition targets");

    // Initial A is split 300/500 into A_cat0 and 200/500 into A_cat1. Setting
    // aggregate A to 0.65 should therefore set A_cat0 to 0.65 * 0.6 * 1000 = 390.
    let loss = problem.cost(&vec![0.65, 0.35]).expect("cost should run");

    assert_eq!(loss, 0.0);
}

#[test]
fn paired_category_initial_condition_redistributes_without_changing_pair_totals() {
    let problem = CalibrationProblem::new(
        TestEngine::new(),
        vec![
            ObservedDataPoint::new(0, "A_cat0".to_string(), 450.0),
            ObservedDataPoint::new(0, "A_cat1".to_string(), 50.0),
        ],
        vec![CalibrationParameter::with_type_and_guess(
            "cat1".to_string(),
            CalibrationParameterType::InitialCondition,
            0.05,
            0.1,
            0.08,
        )],
        vec![],
        LossConfig::SumSquaredError,
        1000,
    )
    .expect("paired category IDs should be valid initial-condition targets");

    // A_cat0/A_cat1 initially total 500. Setting category cat1 to 0.1 keeps
    // that pair total but redistributes it to A_cat1=50 and A_cat0=450.
    assert_eq!(problem.cost(&vec![0.1]).expect("cost should run"), 0.0);
}

#[test]
fn paired_category_initial_condition_can_target_either_binary_category() {
    let problem = CalibrationProblem::new(
        TestEngine::new(),
        vec![
            ObservedDataPoint::new(0, "A_cat0".to_string(), 100.0),
            ObservedDataPoint::new(0, "A_cat1".to_string(), 400.0),
        ],
        vec![CalibrationParameter::with_type_and_guess(
            "cat0".to_string(),
            CalibrationParameterType::InitialCondition,
            0.1,
            0.9,
            0.2,
        )],
        vec![],
        LossConfig::SumSquaredError,
        1000,
    )
    .expect("either category in a binary pair should be a valid target");

    assert_eq!(problem.cost(&vec![0.2]).expect("cost should run"), 0.0);
}

#[test]
fn paired_category_initial_condition_infers_generic_category_names() {
    let problem = CalibrationProblem::new(
        TestEngine::with_alt_categories(),
        vec![
            ObservedDataPoint::new(0, "A_seg0".to_string(), 400.0),
            ObservedDataPoint::new(0, "A_seg1".to_string(), 100.0),
        ],
        vec![CalibrationParameter::with_type_and_guess(
            "seg0".to_string(),
            CalibrationParameterType::InitialCondition,
            0.1,
            0.9,
            0.8,
        )],
        vec![],
        LossConfig::SumSquaredError,
        1000,
    )
    .expect("binary category names should be inferred from compartment names");

    assert!(problem.cost(&vec![0.8]).expect("cost should run") < 1e-20);
}

#[test]
fn exact_compartment_initial_conditions_are_still_accepted() {
    CalibrationProblem::new(
        TestEngine::new(),
        vec![ObservedDataPoint::new(0, "A_cat0".to_string(), 300.0)],
        vec![CalibrationParameter::with_type_and_guess(
            "A_cat0".to_string(),
            CalibrationParameterType::InitialCondition,
            0.1,
            0.9,
            0.3,
        )],
        vec![],
        LossConfig::SumSquaredError,
        1000,
    )
    .expect("expanded compartment IDs should remain valid initial-condition targets");
}
