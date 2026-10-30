//! What a flat profile says about the structure of a problem: the noise
//! floor of the statistic, the resets of the other parameters, and the
//! parameters that compensate, on engines whose output is a closed form of
//! the parameters.

use std::collections::HashMap;

use commol_calibration::{
    Argument, CalibrationConstraint, CalibrationParameter, CalibrationParameterType,
    CalibrationProblem, DiscrepancyConfig, LossConfig, NelderMeadConfig, ObservationModel,
    ObservedDataPoint, OptimizationConfig, optimize,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::profiles::{
    Profile, ProfileEnd, ProfileTracer, ProfileVerdict, ProfiledQuantity, StructuralReading,
    Structure, Undetermined,
};

type Formula = fn(&HashMap<String, f64>, f64) -> f64;

/// One output `x = formula(parameters, t)` at every position `t`.
#[derive(Clone, Debug)]
struct ClosedForm {
    formula: Formula,
    parameters: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for ClosedForm {
    fn check_values(&self) -> Result<(), String> {
        Ok(())
    }

    type ValuePlan = NoPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        // An engine that fails past z = 0.9, for the resets that fail.
        if self.parameters.get("z").is_some_and(|&z| z > 0.9) {
            return Err(SimulationError::Failed("z above 0.9".to_string()));
        }
        buffer.clear();
        buffer.extend(
            positions
                .iter()
                .map(|&t| vec![(self.formula)(&self.parameters, t)]),
        );
        Ok(())
    }

    fn step(&mut self) -> Result<(), SimulationError> {
        Ok(())
    }

    fn output_names(&self) -> Vec<String> {
        vec!["x".to_string()]
    }

    fn state_outputs(&self) -> usize {
        1
    }

    fn reset(&mut self) {}

    fn set_parameter(&mut self, parameter_id: &str, value: f64) -> Result<(), String> {
        self.parameters.insert(parameter_id.to_string(), value);
        Ok(())
    }

    fn current_position(&self) -> f64 {
        0.0
    }

    fn plan_values(&self, _: &[ModelValue], _: Option<&str>) -> Result<NoPlan, String> {
        Ok(NoPlan)
    }

    fn apply_values(&mut self, _: &NoPlan, _: &[f64]) -> Result<f64, String> {
        Ok(0.0)
    }
}

/// A fixed draw of noise, one per position 1, …, 10.
const NOISE: [f64; 10] = [0.3, -0.2, 0.1, -0.4, 0.2, 0.0, 0.3, -0.1, -0.3, 0.2];

/// The problem of fitting `formula` to `truth(t) + scale · noise` at
/// positions 1, …, 10.
fn problem(
    formula: Formula,
    parameters: &[(&str, f64, f64)],
    truth: fn(f64) -> f64,
    scale: f64,
    constraints: Vec<CalibrationConstraint>,
    discrepancy: DiscrepancyConfig,
) -> CalibrationProblem<ClosedForm> {
    CalibrationProblem::new(
        ClosedForm {
            formula,
            parameters: parameters
                .iter()
                .map(|(id, min, _)| (id.to_string(), *min))
                .collect(),
        },
        NOISE
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let t = (i + 1) as f64;
                ObservedDataPoint::new(t, "x".to_string(), truth(t) + scale * e)
            })
            .collect(),
        parameters
            .iter()
            .map(|(id, min, max)| CalibrationParameter::new(id.to_string(), *min, *max))
            .collect(),
        constraints,
        discrepancy,
    )
    .unwrap()
}

fn sse() -> DiscrepancyConfig {
    LossConfig::SumSquaredError.into()
}

/// Nelder-Mead run to the precision of the objective: its sd_tolerance is
/// below the noise floor of every problem here.
fn config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 1e-16,
        max_iterations: 4000,
        ..NelderMeadConfig::default()
    })
}

fn product(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] * p["s"] * t
}

/// `a · s · t` with `a` in `[0.1, 10]` and `s` in `[0.5, 2]`, fitted to
/// `t` with noise: the ridge `a · s ≈ 1` leaves the box below `a = 0.5` and
/// above `a = 2`.
fn product_problem(scale: f64) -> CalibrationProblem<ClosedForm> {
    problem(
        product,
        &[("a", 0.1, 10.0), ("s", 0.5, 2.0)],
        |t| t,
        scale,
        vec![],
        sse(),
    )
}

/// The profile of the parameter at `index` from the fit, up to `threshold`,
/// and what the tracer reads of its structure.
fn traced(
    problem: &CalibrationProblem<ClosedForm>,
    config: &OptimizationConfig,
    fit: Option<&[f64]>,
    index: usize,
    threshold: f64,
) -> (Profile, Option<ProfileVerdict>, Structure, f64) {
    let fitted;
    let fit = match fit {
        Some(fit) => fit,
        None => {
            fitted = optimize(problem, config, 0).unwrap().best_parameters;
            &fitted
        }
    };
    let tracer = ProfileTracer::new(problem, config, fit, 5).unwrap();
    let profile = tracer
        .trace(&ProfiledQuantity::Parameter(index), threshold, 200)
        .unwrap();
    let structure = tracer.read_structure(index, &profile);
    let verdict = tracer.verdict_with(&profile, Some(&structure));
    (profile, verdict, structure, tracer.tolerance())
}

#[test]
fn the_noise_floor_propagates_the_accuracy_of_a_run() {
    let problem = product_problem(1.0);
    let config = config();
    let fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    let tracer = ProfileTracer::new(&problem, &config, &fit, 0).unwrap();

    // The sum of squares has the term derivative −2 r at a mean m: the
    // error of a run, ε·|m|, moves it by 2 |r| ε |m|.
    let expected: f64 = NOISE
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let t = (i + 1) as f64;
            let mean = fit[0] * fit[1] * t;
            2.0 * (t + e - mean).abs() * f64::EPSILON * mean.abs()
        })
        .sum();
    assert!((tracer.noise_floor() - expected).abs() <= 1e-6 * expected);
    // Nelder-Mead stops when the costs of its simplex spread by less than
    // its sd_tolerance: an absolute amount.
    assert_eq!(tracer.optimizer_tolerance(), 1e-16);
    assert_eq!(
        tracer.tolerance(),
        tracer.optimizer_tolerance().max(tracer.noise_floor())
    );

    // The floor of two evaluations: 2 (ν₁ + ν₂) + 2 γₙ Σ ℓᵢ, with the
    // curvature 2 of a square, ν₂ = Σ ½ · 2 · (ε |m|)², and the squares
    // ℓᵢ = r².
    let (mut second, mut squares) = (0.0, 0.0);
    for (i, e) in NOISE.iter().enumerate() {
        let t = (i + 1) as f64;
        let mean = fit[0] * fit[1] * t;
        second += (f64::EPSILON * mean).powi(2);
        squares += (t + e - mean).powi(2);
    }
    let unit = f64::EPSILON / 2.0;
    let gamma = 10.0 * unit / (1.0 - 10.0 * unit);
    let floor = 2.0 * (expected + second) + 2.0 * gamma * squares;
    let computed = tracer.evaluation_floor().unwrap();
    assert!(
        (computed - floor).abs() <= 1e-6 * floor,
        "{computed} against {floor}"
    );
}

/// Nelder-Mead at its default sd_tolerance, 1e-6.
fn default_config() -> OptimizationConfig {
    OptimizationConfig::NelderMead(NelderMeadConfig::default())
}

#[test]
fn a_parameter_with_no_effect_is_non_identifiable_at_the_default_tolerance() {
    // u is read by nothing: every reset equals the fit within the floor,
    // whatever the refits of the profile do.
    let problem = problem(
        two_unread,
        &[("a", -10.0, 10.0), ("b", -5.0, 5.0), ("u", 0.0, 1.0)],
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let config = default_config();
    let mut fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    fit[2] = 0.4;
    let (_, verdict, structure, _) = traced(&problem, &config, Some(&fit), 2, 2.0);

    assert_eq!(structure.verdict, Ok(StructuralReading::NonIdentifiable));
    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert!(structure.largest_rise.unwrap().abs() <= 1e-12);
}

fn late(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] + p["b"] * t + 10.0 * (p["u"] - 0.95).max(0.0) * t
}

#[test]
fn the_ends_of_the_box_rule_out_non_identifiable() {
    // u acts only above 0.95; two steps from 0.4 never reach it, but the
    // reset at the upper end of its box does.
    let problem = problem(
        late,
        &[("a", -10.0, 10.0), ("b", -5.0, 5.0), ("u", 0.0, 1.0)],
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let config = config();
    let mut fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    fit[2] = 0.4;
    let tracer = ProfileTracer::new(&problem, &config, &fit, 0).unwrap();
    let profile = tracer
        .trace(&ProfiledQuantity::Parameter(2), 2.0, 2)
        .unwrap();
    let structure = tracer.read_structure(2, &profile);

    assert!(profile.points.iter().all(|p| p.value < 0.95));
    assert_eq!(structure.verdict, Err(Undetermined::NotFlat));
    assert!(structure.largest_rise.unwrap() > 1.0);
}

fn guarded(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] + p["b"] * t
}

#[test]
fn a_reset_whose_engine_fails_is_infeasible() {
    // The engine fails past z = 0.9, the upper end of the box of z is 1:
    // its reset counts as infeasible and rules out non_identifiable.
    let problem = problem(
        guarded,
        &[("a", -10.0, 10.0), ("b", -5.0, 5.0), ("z", 0.0, 1.0)],
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let config = config();
    let mut fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    fit[2] = 0.4;
    let tracer = ProfileTracer::new(&problem, &config, &fit, 0).unwrap();
    let profile = tracer
        .trace(&ProfiledQuantity::Parameter(2), 2.0, 2)
        .unwrap();
    let structure = tracer.read_structure(2, &profile);

    assert_eq!(structure.infeasible, 1);
    assert_ne!(structure.verdict, Ok(StructuralReading::NonIdentifiable));
}

#[test]
fn a_loss_with_no_floor_reads_no_structure() {
    // The root mean squared error of a perfect fit has no term derivatives.
    let problem = problem(
        product,
        &[("a", 0.1, 10.0), ("s", 0.5, 2.0)],
        |t| t,
        0.0,
        vec![],
        LossConfig::RootMeanSquaredError.into(),
    );
    let config = config();
    let (_, _, structure, _) = traced(&problem, &config, Some(&[1.0, 1.0]), 0, 0.5);

    assert_eq!(structure.verdict, Err(Undetermined::NoNoiseFloor));
}

#[test]
fn a_ridge_that_leaves_the_box_after_a_few_points_is_in_combination() {
    let (profile, verdict, structure, _) = traced(&product_problem(1.0), &config(), None, 0, 2.0);

    // The ridge leaves the box where s reaches 0.5 or 2: past a ≈ 2 and
    // below a ≈ 0.5, inside the bounds of a.
    let exits = profile.box_exits();
    assert!(
        exits
            .iter()
            .all(|exit| exit.is_some_and(|(_, index)| index == 1))
    );
    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::InCombination));
    assert_eq!(structure.compensating, vec![1]);
    assert!(structure.largest_rise.unwrap() > 2.0);
    assert_eq!(structure.joint_rise, None);
}

fn two_unread(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] + p["b"] * t
}

#[test]
fn a_parameter_with_no_effect_is_non_identifiable_even_when_another_drifts() {
    // u and v are read by nothing; a refit may move v anywhere.
    let problem = problem(
        two_unread,
        &[
            ("a", -10.0, 10.0),
            ("b", -5.0, 5.0),
            ("u", 0.0, 1.0),
            ("v", 0.0, 1.0),
        ],
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let config = config();
    let mut fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    fit[2] = 0.4;
    fit[3] = 0.6;
    let (profile, verdict, structure, _) = traced(&problem, &config, Some(&fit), 2, 2.0);

    assert!(
        profile
            .points
            .iter()
            .any(|p| (p.parameters[3] - 0.6).abs() > 1e-6),
        "v does not drift"
    );
    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::NonIdentifiable));
    assert!(structure.compensating.is_empty());
}

fn curved(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] + (p["b"] - p["c"].sin()) * t
}

#[test]
fn a_parameter_that_leaves_and_returns_is_named_where_it_moves_most() {
    // Along the profile of c, b = const + sin c: it rises to its largest
    // displacement at c = π/2 and comes back towards its fit at c = π.
    let problem = problem(
        curved,
        &[("a", -10.0, 10.0), ("b", -5.0, 5.0), ("c", 0.0, 3.0)],
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let config = config();
    let mut fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    // b = 0.5 + sin c at the fit, with c = 0.05.
    fit[1] += 0.05_f64.sin() - fit[2].sin();
    fit[2] = 0.05;
    let (profile, verdict, structure, _) = traced(&problem, &config, Some(&fit), 2, 2.0);

    let last = profile.points.last().unwrap();
    assert!(last.value > 2.9 && (last.parameters[1] - fit[1]).abs() < 0.2);
    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::InCombination));
    assert_eq!(structure.compensating, vec![1]);
}

fn scaled(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["k"] * p["q"] * t + 0.0 * p["u"]
}

/// `k · q · t` at its least-squares fit with `q = 0.5` on its lower bound,
/// and `u` read by nothing.
fn bound_problem() -> (CalibrationProblem<ClosedForm>, Vec<f64>) {
    let problem = problem(
        scaled,
        &[("k", 0.1, 2.0), ("q", 0.5, 2.0), ("u", 0.0, 1.0)],
        |t| 0.5 * t,
        0.5,
        vec![],
        sse(),
    );
    // The slope Σ t y / Σ t² of the data, all in k.
    let (ty, tt) = problem
        .observed_data()
        .iter()
        .fold((0.0, 0.0), |(ty, tt), obs| {
            (ty + obs.at * obs.value, tt + obs.at * obs.at)
        });
    (problem, vec![ty / tt / 0.5, 0.5, 0.5])
}

#[test]
fn a_parameter_on_its_bound_at_the_fit_never_gives_non_identifiable() {
    let (problem, fit) = bound_problem();
    let config = config();

    // k: above its fit, q would leave its bound, so the profile rises
    // there; below it, q rises off its bound and keeps k · q until it
    // reaches 2, where the box makes the profile cross too. That crossing
    // comes from the bound of q, so that side counts as open.
    let (k, k_verdict, k_structure, _) = traced(&problem, &config, Some(&fit), 0, 2.0);
    assert_eq!(k.ends, [ProfileEnd::Crossed; 2]);
    assert_eq!(k.box_crossings, [Some(1), None]);
    assert!(k.interval()[0].is_some());
    assert_eq!(k_verdict, Some(ProfileVerdict::PracticallyNonIdentifiable));
    assert_eq!(k_structure.verdict, Err(Undetermined::NotFlat));

    // u: flat, and resetting moves nothing, but q sits on its bound at the
    // fit, where a reset cannot show what lies beyond it.
    let (_, u_verdict, u_structure, _) = traced(&problem, &config, Some(&fit), 2, 2.0);
    assert_eq!(u_verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(u_structure.verdict, Err(Undetermined::OnBound(1)));
}

#[test]
fn a_feasible_fit_above_the_base_of_the_penalty_is_read() {
    // Values near 10⁶ and noise near 10⁵: the sum of squares of the fit is
    // above 10¹⁰, the least penalty of an infeasible point.
    let problem = problem(
        product,
        &[("a", 1e5, 1e7), ("s", 0.5, 2.0)],
        |t| 1e6 * t,
        3e5,
        vec![],
        sse(),
    );
    let config = config();
    let fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    let tracer = ProfileTracer::new(&problem, &config, &fit, 0).unwrap();
    assert!(tracer.fit_loss() > 1e10, "{}", tracer.fit_loss());
    let (_, verdict, structure, _) = traced(&problem, &config, Some(&fit), 0, 1e12);

    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::InCombination));
}

#[test]
fn one_feasible_reset_above_the_threshold_suffices() {
    // a · s ≤ 1.2: a reset of s to its fit at a large a breaks it.
    let problem = problem(
        product,
        &[("a", 0.1, 10.0), ("s", 0.5, 2.0)],
        |t| t,
        1.0,
        vec![CalibrationConstraint::new(
            "cap".to_string(),
            "1.2 - a * s".to_string(),
        )],
        sse(),
    );
    let (_, verdict, structure, _) = traced(&problem, &config(), None, 0, 2.0);

    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert!(structure.infeasible > 0);
    assert_eq!(structure.verdict, Ok(StructuralReading::InCombination));
}

#[test]
fn a_crossing_on_a_bound_is_identifiable() {
    // The slope of a line, its lower bound just past the crossing of T = 1
    // (half width √(1 / Stt)).
    let fit_problem = problem(
        two_unread,
        &[("a", -10.0, 10.0), ("b", -5.0, 5.0)],
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let fit = optimize(&fit_problem, &config(), 0)
        .unwrap()
        .best_parameters;
    let half_width = (1.0_f64 / 82.5).sqrt();
    let lower = fit[1] - 1.01 * half_width;
    let parameters = [("a", -10.0, 10.0), ("b", lower, 5.0)];
    let bounded = problem(
        two_unread,
        &parameters,
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let (profile, verdict, _, _) = traced(&bounded, &config(), Some(&fit), 1, 1.0);

    assert_eq!(profile.ends, [ProfileEnd::Crossed; 2]);
    assert_eq!(profile.points[0].value, lower);
    assert_eq!(verdict, Some(ProfileVerdict::Identifiable));
    let [low, _] = profile.interval();
    assert!((low.unwrap() - (fit[1] - half_width)).abs() < 1e-3 * half_width);
}

fn away(p: &HashMap<String, f64>, t: f64) -> f64 {
    p["a"] + p["b"] * t + p["u"] * (p["a"] - 5.0).max(0.0) * t
}

#[test]
fn an_effect_only_away_from_the_fit_is_not_seen() {
    // u acts only where a > 5; the fit has a near 1.
    let problem = problem(
        away,
        &[("a", -10.0, 10.0), ("b", -5.0, 5.0), ("u", 0.0, 1.0)],
        |t| 1.0 + 0.5 * t,
        1.0,
        vec![],
        sse(),
    );
    let (_, verdict, structure, _) = traced(&problem, &config(), None, 2, 2.0);

    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::NonIdentifiable));
}

#[test]
fn resets_between_the_tolerance_and_the_threshold_give_no_structural_verdict() {
    // A threshold far above every reset of the ridge.
    let (_, verdict, structure, tolerance) = traced(&product_problem(1.0), &config(), None, 0, 1e9);

    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Err(Undetermined::Undecided));
    let largest = structure.largest_rise.unwrap();
    assert!(largest > tolerance && largest <= 1e9, "{largest}");
}

#[test]
fn a_threshold_at_or_below_the_tolerance_reads_no_structure() {
    let loose = OptimizationConfig::NelderMead(NelderMeadConfig {
        sd_tolerance: 2.0,
        max_iterations: 4000,
        ..NelderMeadConfig::default()
    });
    let problem = product_problem(1.0);
    let fit = optimize(&problem, &config(), 0).unwrap().best_parameters;
    let (_, verdict, structure, tolerance) = traced(&problem, &loose, Some(&fit), 0, 2.0);

    assert_eq!(tolerance, 2.0);
    assert_eq!(verdict, None);
    assert_eq!(structure.verdict, Err(Undetermined::BelowTolerance));
}

#[test]
fn a_likelihood_far_from_zero_keeps_an_absolute_tolerance() {
    // A normal model of sd 10⁻⁶ on values of the product with noise of
    // that size: −log L at the fit is near 10 · log(10⁻⁶ √(2π)) ≈ −129.
    let problem = problem(
        product,
        &[("a", 0.1, 10.0), ("s", 0.5, 2.0)],
        |t| t,
        1e-6,
        vec![],
        DiscrepancyConfig::Likelihood(vec![(
            "x".to_string(),
            ObservationModel::Normal {
                sd: Argument::Value(1e-6),
            },
        )]),
    );
    let config = config();
    let fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    let tracer = ProfileTracer::new(&problem, &config, &fit, 0).unwrap();
    assert!(tracer.fit_loss() < -100.0, "{}", tracer.fit_loss());
    // The tolerance does not grow with |−log L|: it is the error of a run,
    // which the scores (r / sd²) carry.
    assert_eq!(tracer.tolerance(), tracer.noise_floor());
    assert!(tracer.tolerance() < 1e-6, "{}", tracer.tolerance());

    let (_, verdict, structure, _) = traced(&problem, &config, Some(&fit), 0, 1.92);
    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::InCombination));
}

#[test]
fn a_loss_with_no_term_derivatives_has_no_noise_floor() {
    let problem = problem(
        product,
        &[("a", 0.1, 10.0), ("s", 0.5, 2.0)],
        |t| t,
        1.0,
        vec![],
        LossConfig::MeanAbsoluteError.into(),
    );
    let config = config();
    let fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    let tracer = ProfileTracer::new(&problem, &config, &fit, 0).unwrap();

    assert_eq!(tracer.noise_floor(), 0.0);
    assert_eq!(tracer.tolerance(), 1e-16);
}

#[test]
fn a_fit_that_is_not_the_optimum_reads_no_structure() {
    // A fit off the ridge: the profile goes below it.
    let problem = product_problem(1.0);
    let (_, verdict, structure, _) = traced(&problem, &config(), Some(&[1.0, 0.9]), 0, 2.0);

    assert_ne!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Err(Undetermined::BelowFit));
}

#[test]
fn a_noise_parameter_is_never_named() {
    // A normal model whose sd is calibrated: the reset of the ridge names s,
    // and the sd joins the joint check of the parameters not named.
    let parameters = [("a", 0.1, 10.0), ("s", 0.5, 2.0)];
    let mut calibrated: Vec<CalibrationParameter> = parameters
        .iter()
        .map(|(id, min, max)| CalibrationParameter::new(id.to_string(), *min, *max))
        .collect();
    calibrated.push(CalibrationParameter::with_type(
        "sd".to_string(),
        CalibrationParameterType::Noise,
        0.01,
        5.0,
    ));
    let base = product_problem(1.0);
    let problem = CalibrationProblem::new(
        base.forward().base_engine().clone(),
        base.observed_data().to_vec(),
        calibrated,
        vec![],
        DiscrepancyConfig::Likelihood(vec![(
            "x".to_string(),
            ObservationModel::Normal {
                sd: Argument::Parameter("sd".to_string()),
            },
        )]),
    )
    .unwrap();
    let (_, verdict, structure, _) = traced(&problem, &config(), None, 0, 1.92);

    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::InCombination));
    assert_eq!(structure.compensating, vec![1]);
}

#[test]
fn a_reset_that_is_not_feasible_names_no_parameter() {
    // 0.8 ≤ a · s ≤ 1.2 holds along the ridge, but s reset to its fit where
    // s is nearest a bound, at a ≈ 0.5 or a ≈ 2, breaks it: s is not named,
    // and resetting it at the outermost points is not feasible either.
    let problem = problem(
        product,
        &[("a", 0.1, 10.0), ("s", 0.5, 2.0)],
        |t| t,
        1.0,
        vec![
            CalibrationConstraint::new("cap".to_string(), "1.2 - a * s".to_string()),
            CalibrationConstraint::new("floor".to_string(), "a * s - 0.8".to_string()),
        ],
        sse(),
    );
    let (_, verdict, structure, _) = traced(&problem, &config(), None, 0, 2.0);

    assert_eq!(verdict, Some(ProfileVerdict::StructurallyNonIdentifiable));
    assert_eq!(structure.verdict, Ok(StructuralReading::InCombination));
    assert!(structure.compensating.is_empty());
    assert_eq!(structure.joint_rise, Some(f64::INFINITY));
}

#[test]
fn the_floor_of_the_root_mean_squared_error_scales_the_squares() {
    // The root mean squared error √(S / W) has the term derivatives of the
    // sum of squares times c = 1 / (2 √(W S)), and the terms ½ S / √(W S).
    let problem = problem(
        product,
        &[("a", 0.1, 10.0), ("s", 0.5, 2.0)],
        |t| t,
        1.0,
        vec![],
        LossConfig::RootMeanSquaredError.into(),
    );
    let config = config();
    let fit = optimize(&problem, &config, 0).unwrap().best_parameters;
    let tracer = ProfileTracer::new(&problem, &config, &fit, 0).unwrap();

    let (mut squares, mut first, mut second) = (0.0, 0.0, 0.0);
    for (i, e) in NOISE.iter().enumerate() {
        let t = (i + 1) as f64;
        let mean = fit[0] * fit[1] * t;
        let error = f64::EPSILON * mean.abs();
        squares += (t + e - mean).powi(2);
        first += 2.0 * (t + e - mean).abs() * error;
        second += error * error;
    }
    let c = 0.5 / (10.0 * squares).sqrt();
    let unit = f64::EPSILON / 2.0;
    let gamma = 10.0 * unit / (1.0 - 10.0 * unit);
    let floor = 2.0 * c * (first + second) + 2.0 * gamma * 0.5 * squares / (10.0 * squares).sqrt();

    assert!((tracer.noise_floor() - c * first).abs() <= 1e-6 * c * first);
    let computed = tracer.evaluation_floor().unwrap();
    assert!(
        (computed - floor).abs() <= 1e-6 * floor,
        "{computed} against {floor}"
    );
}
