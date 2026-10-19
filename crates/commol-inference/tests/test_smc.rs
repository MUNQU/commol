//! The sequential Monte Carlo core on targets with closed forms: Gaussian
//! tempering with its evidence, a target near a bound, the rule of the moves
//! and of the levels, and ABC rejection.

use std::sync::atomic::{AtomicUsize, Ordering};

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, ObservedDataPoint, Workspace,
};
use commol_core::{ModelValue, SimulationEngine, SimulationError, ValuePlan};
use commol_inference::resampling::BlockScheme;
use commol_inference::smc::{
    AbcTarget, Distance, GibbsTarget, RandomWalk, SmcRun, SmcSampler, TemperedTarget,
    abc_rejection, bootstrap_coverage_eta, information_matching_eta, next_level,
};
use commol_inference::{Distribution, Family, Priors};
use commol_random::Generator;

/// A prior of independent distributions and a kernel `exp(−level · v)`,
/// with a value `v` that the target computes from the parameters; it counts
/// its evaluations and every point it is asked about outside its bounds.
struct Tempered {
    prior: Vec<Distribution>,
    bounds: Vec<(f64, f64)>,
    value: fn(&[f64]) -> f64,
    evaluations: AtomicUsize,
    outside: AtomicUsize,
}

impl Tempered {
    fn new(prior: Vec<Distribution>, value: fn(&[f64]) -> f64) -> Self {
        let bounds = prior.iter().map(Distribution::bounds).collect();
        Self {
            prior,
            bounds,
            value,
            evaluations: AtomicUsize::new(0),
            outside: AtomicUsize::new(0),
        }
    }

    fn check(&self, values: &[f64]) {
        let inside = values
            .iter()
            .zip(&self.bounds)
            .all(|(v, (low, high))| v >= low && v <= high);
        if !inside {
            self.outside.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl TemperedTarget for Tempered {
    fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    fn log_prior(&self, values: &[f64]) -> f64 {
        self.check(values);
        self.prior
            .iter()
            .zip(values)
            .map(|(d, &v)| d.log_density(v))
            .sum()
    }

    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64> {
        self.prior.iter().map(|d| d.sample(stream)).collect()
    }

    fn evaluate(
        &self,
        values: &[f64],
        _stream: &mut Generator,
        _workspace: &mut Workspace,
    ) -> Result<Option<f64>, String> {
        self.check(values);
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        Ok(Some((self.value)(values)))
    }

    fn log_kernel(&self, value: f64, level: f64) -> f64 {
        -level * value
    }

    fn first_level(&self, _kept: &[f64]) -> f64 {
        0.0
    }

    fn last_level(&self) -> f64 {
        1.0
    }
}

/// ABC on a distance the target computes from the parameters.
struct Abc {
    prior: Vec<Distribution>,
    bounds: Vec<(f64, f64)>,
    distance: fn(&[f64]) -> f64,
}

impl TemperedTarget for Abc {
    fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    fn log_prior(&self, values: &[f64]) -> f64 {
        self.prior
            .iter()
            .zip(values)
            .map(|(d, &v)| d.log_density(v))
            .sum()
    }

    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64> {
        self.prior.iter().map(|d| d.sample(stream)).collect()
    }

    fn evaluate(
        &self,
        values: &[f64],
        _stream: &mut Generator,
        _workspace: &mut Workspace,
    ) -> Result<Option<f64>, String> {
        Ok(Some((self.distance)(values)))
    }

    fn log_kernel(&self, value: f64, level: f64) -> f64 {
        if value <= level {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    fn first_level(&self, kept: &[f64]) -> f64 {
        kept.iter().copied().fold(0.0, f64::max)
    }

    fn last_level(&self) -> f64 {
        0.0
    }
}

fn standard_normal() -> Distribution {
    Distribution::new(
        Family::Normal { mean: 0.0, sd: 1.0 },
        f64::NEG_INFINITY,
        f64::INFINITY,
    )
    .unwrap()
}

fn unit_uniform() -> Distribution {
    Distribution::new(Family::Uniform, 0.0, 1.0).unwrap()
}

/// `A = [[4, 1.5], [1.5, 2]]` and `m = (1, −0.5)`.
const A: [[f64; 2]; 2] = [[4.0, 1.5], [1.5, 2.0]];
const M: [f64; 2] = [1.0, -0.5];

/// `½ (θ − m)ᵀ A (θ − m)`.
fn quadratic(values: &[f64]) -> f64 {
    let d = [values[0] - M[0], values[1] - M[1]];
    0.5 * (d[0] * (A[0][0] * d[0] + A[0][1] * d[1]) + d[1] * (A[1][0] * d[0] + A[1][1] * d[1]))
}

fn sampler(particles: usize, stay_probability: f64) -> SmcSampler<RandomWalk> {
    SmcSampler::new(particles, RandomWalk::new(), 0.5, 0.5, stay_probability).unwrap()
}

fn weighted_mean(run: &SmcRun, j: usize) -> f64 {
    let weights = run.weights();
    run.values.iter().zip(&weights).map(|(v, w)| v[j] * w).sum()
}

#[test]
fn gaussian_target_gives_the_closed_form_and_evidence() {
    // Prior N(0, I), kernel exp(−½ (θ − m)ᵀ A (θ − m)): the posterior has
    // precision P = I + A and mean P⁻¹ A m, and the evidence is
    // |P|^{−1/2} exp(½ (Am)ᵀ P⁻¹ (Am) − ½ mᵀ A m).
    let target = Tempered::new(vec![standard_normal(), standard_normal()], quadratic);
    let run = sampler(4000, 0.02).run(&target, 7).unwrap();

    let p = [[1.0 + A[0][0], A[0][1]], [A[1][0], 1.0 + A[1][1]]];
    let det = p[0][0] * p[1][1] - p[0][1] * p[1][0];
    let inverse = [
        [p[1][1] / det, -p[0][1] / det],
        [-p[1][0] / det, p[0][0] / det],
    ];
    let am = [
        A[0][0] * M[0] + A[0][1] * M[1],
        A[1][0] * M[0] + A[1][1] * M[1],
    ];
    let mean = [
        inverse[0][0] * am[0] + inverse[0][1] * am[1],
        inverse[1][0] * am[0] + inverse[1][1] * am[1],
    ];
    let m_a_m = M[0] * am[0] + M[1] * am[1];
    let log_evidence = -0.5 * det.ln() + 0.5 * (am[0] * mean[0] + am[1] * mean[1]) - 0.5 * m_a_m;

    for j in 0..2 {
        let sd = inverse[j][j].sqrt();
        let estimate = weighted_mean(&run, j);
        assert!(
            (estimate - mean[j]).abs() < 0.1 * sd,
            "mean {j}: {estimate} against {}",
            mean[j]
        );
    }
    let weights = run.weights();
    let covariance = |j: usize, k: usize| -> f64 {
        let (mj, mk) = (weighted_mean(&run, j), weighted_mean(&run, k));
        run.values
            .iter()
            .zip(&weights)
            .map(|(v, w)| w * (v[j] - mj) * (v[k] - mk))
            .sum()
    };
    for (j, k) in [(0, 0), (0, 1), (1, 1)] {
        let expected = inverse[j][k];
        let scale = (inverse[j][j] * inverse[k][k]).sqrt();
        assert!(
            (covariance(j, k) - expected).abs() < 0.1 * scale,
            "covariance {j}{k}: {} against {expected}",
            covariance(j, k)
        );
    }
    assert!(
        (run.log_evidence - log_evidence).abs() < 0.05,
        "log evidence {} against {log_evidence}",
        run.log_evidence
    );
    assert_eq!(run.steps.last().unwrap().level, 1.0);
}

#[test]
fn one_seed_gives_one_result() {
    let target = Tempered::new(vec![standard_normal(), standard_normal()], quadratic);
    let first = sampler(300, 0.1).run(&target, 11).unwrap();
    let second = sampler(300, 0.1).run(&target, 11).unwrap();
    let other = sampler(300, 0.1).run(&target, 12).unwrap();
    assert_eq!(first, second);
    assert_ne!(first.values, other.values);
}

#[test]
fn reweighting_runs_no_model() {
    let target = Tempered::new(vec![standard_normal(), standard_normal()], quadratic);
    let run = sampler(300, 0.1).run(&target, 3).unwrap();

    // One evaluation per particle of the prior and one per proposed move;
    // choosing the levels and reweighting add none.
    let moves: usize = run.steps.iter().map(|step| step.moves * 300).sum();
    assert_eq!(target.evaluations.load(Ordering::Relaxed), 300 + moves);
    assert_eq!(run.evaluations, 300 + moves);
    assert!(run.steps.len() > 1, "{:?}", run.steps);
}

#[test]
fn moves_on_an_unbounded_scale_stay_inside_the_bounds() {
    // Uniform prior on [0, 1] and kernel exp(−50 θ): a truncated
    // exponential of rate 50, whose mass lies against the lower bound.
    let target = Tempered::new(vec![unit_uniform()], |values| 50.0 * values[0]);
    let run = sampler(2000, 0.05).run(&target, 5).unwrap();

    assert_eq!(target.outside.load(Ordering::Relaxed), 0);
    let rate: f64 = 50.0;
    let tail = (-rate).exp();
    let mean = 1.0 / rate - tail / (1.0 - tail);
    let estimate = weighted_mean(&run, 0);
    assert!(
        (estimate - mean).abs() < 0.1 * mean,
        "mean {estimate} against {mean}"
    );
}

#[test]
fn move_count_follows_the_acceptance_rate() {
    let target = Tempered::new(vec![standard_normal(), standard_normal()], quadratic);
    for stay_probability in [0.01, 0.3] {
        let run = sampler(300, stay_probability).run(&target, 2).unwrap();
        for step in &run.steps {
            let expected = (stay_probability.ln() / (1.0 - step.acceptance_rate).ln()).ceil();
            assert_eq!(step.moves, (expected as usize).max(1), "{step:?}");
        }
    }
}

#[test]
fn a_sampler_checks_its_settings() {
    let new = |particles, threshold, cess, stay| {
        SmcSampler::new(particles, RandomWalk::new(), threshold, cess, stay).unwrap_err()
    };
    assert!(new(1, 0.5, 0.5, 0.1).contains("particles"));
    assert!(new(10, 0.0, 0.5, 0.1).contains("resample_threshold"));
    assert!(new(10, 0.5, 1.0, 0.1).contains("target_cess"));
    assert!(new(10, 0.5, 0.5, 1.0).contains("stay_probability"));
    assert!(new(10, 0.5, 0.5, 0.0).contains("stay_probability"));
}

#[test]
fn abc_rejection_keeps_the_closest_draws() {
    let target = Abc {
        prior: vec![unit_uniform()],
        bounds: vec![(0.0, 1.0)],
        distance: |values| (values[0] - 0.3).abs(),
    };
    let run = abc_rejection(&target, 1000, 0.1, 4).unwrap();

    assert_eq!(run.values.len(), 1000);
    let kept: Vec<usize> = (0..1000)
        .filter(|&i| run.log_weights[i].is_finite())
        .collect();
    assert_eq!(kept.len(), 100);
    let distance = |i: usize| run.kept[i].unwrap();
    let farthest_kept = kept.iter().map(|&i| distance(i)).fold(0.0, f64::max);
    let closest_dropped = (0..1000)
        .filter(|i| !kept.contains(i))
        .map(distance)
        .fold(f64::INFINITY, f64::min);
    assert!(farthest_kept <= closest_dropped);
    assert_eq!(run.steps.len(), 1);
    assert_eq!(run.steps[0].level, farthest_kept);
    assert_eq!(run.steps[0].moves, 0);
    assert!((run.steps[0].acceptance_rate - 0.1).abs() < 1e-12);
}

#[test]
fn abc_rejection_needs_two_draws() {
    let target = Abc {
        prior: vec![unit_uniform()],
        bounds: vec![(0.0, 1.0)],
        distance: |values| values[0],
    };
    let error = abc_rejection(&target, 1, 0.5, 1).unwrap_err();
    assert_eq!(error, "draws must be at least 2, got 1");
}

#[test]
fn abc_next_tolerance_is_the_weighted_quantile_of_the_distances() {
    let target = Abc {
        prior: vec![unit_uniform()],
        bounds: vec![(0.0, 1.0)],
        distance: |values| values[0],
    };
    let mut generator = Generator::from_seed(8);
    let distances: Vec<f64> = (0..200).map(|_| generator.uniform()).collect();
    let log_weights: Vec<f64> = (0..200).map(|_| generator.uniform().ln()).collect();
    let kept: Vec<Option<f64>> = distances.iter().copied().map(Some).collect();

    let level = next_level(&target, &kept, &log_weights, 1.0, 0.5);

    // The smallest distance whose share of the weight at or below it
    // reaches one half: the inverse of the weighted distribution function.
    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weights.iter().map(|w| (w - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut order: Vec<usize> = (0..200).collect();
    order.sort_by(|&i, &j| distances[i].total_cmp(&distances[j]));
    let mut cumulative = 0.0;
    let median = order
        .into_iter()
        .find(|&i| {
            cumulative += weights[i] / total;
            cumulative >= 0.5
        })
        .map(|i| distances[i])
        .unwrap();
    assert_eq!(level, median);
}

#[test]
fn abc_smc_stops_when_the_moves_stop_accepting() {
    let target = Abc {
        prior: vec![unit_uniform()],
        bounds: vec![(0.0, 1.0)],
        distance: |values| (values[0] - 0.3).abs(),
    };
    let run = SmcSampler::new(500, RandomWalk::new(), 1.0, 0.5, 0.05)
        .unwrap()
        .with_min_acceptance(0.2)
        .unwrap()
        .run(&target, 6)
        .unwrap();
    for pair in run.steps.windows(2) {
        assert!(pair[1].level < pair[0].level, "{:?}", run.steps);
    }
    assert!(run.steps.iter().all(|step| step.acceptance_rate >= 0.2));
    assert!(run.stopped_at.is_some());
    // Every particle with weight is inside the last tolerance.
    let last = run.steps.last().unwrap().level;
    for (kept, w) in run.kept.iter().zip(&run.log_weights) {
        if w.is_finite() {
            assert!(kept.unwrap() <= last);
        }
    }
}

/// A target whose evaluation is `rule(values, call)`, where `call` counts the
/// evaluations before this one; its kernel is `exp(−level · v)`.
struct Scripted {
    prior: Vec<Distribution>,
    bounds: Vec<(f64, f64)>,
    rule: fn(&[f64], usize) -> Result<Option<f64>, String>,
    calls: AtomicUsize,
}

impl Scripted {
    fn new(prior: Distribution, rule: fn(&[f64], usize) -> Result<Option<f64>, String>) -> Self {
        Self {
            bounds: vec![prior.bounds()],
            prior: vec![prior],
            rule,
            calls: AtomicUsize::new(0),
        }
    }
}

impl TemperedTarget for Scripted {
    fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    fn log_prior(&self, values: &[f64]) -> f64 {
        self.prior[0].log_density(values[0])
    }

    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64> {
        vec![self.prior[0].sample(stream)]
    }

    fn evaluate(
        &self,
        values: &[f64],
        _stream: &mut Generator,
        _workspace: &mut Workspace,
    ) -> Result<Option<f64>, String> {
        (self.rule)(values, self.calls.fetch_add(1, Ordering::Relaxed))
    }

    fn log_kernel(&self, value: f64, level: f64) -> f64 {
        -level * value
    }

    fn first_level(&self, _kept: &[f64]) -> f64 {
        0.0
    }

    fn last_level(&self) -> f64 {
        1.0
    }
}

const PARTICLES: usize = 200;

#[test]
fn a_prior_entirely_outside_the_support_is_an_error() {
    let target = Scripted::new(unit_uniform(), |_, _| Ok(None));
    let error = sampler(PARTICLES, 0.1).run(&target, 1).unwrap_err();
    assert!(
        error.contains("Every draw of the prior is outside the support"),
        "{error}"
    );
}

#[test]
fn moves_that_accept_nothing_are_an_error() {
    // The draws of the prior run; every proposed move is outside the support.
    let target = Scripted::new(unit_uniform(), |values, call| {
        Ok((call < PARTICLES).then(|| 50.0 * values[0]))
    });
    let error = sampler(PARTICLES, 0.1).run(&target, 2).unwrap_err();
    assert!(error.contains("accept no proposal"), "{error}");
}

#[test]
fn a_failure_of_the_model_in_a_move_stops_the_run() {
    let target = Scripted::new(unit_uniform(), |values, call| {
        if call < PARTICLES {
            Ok(Some(50.0 * values[0]))
        } else {
            Err("the engine failed".to_string())
        }
    });
    let error = sampler(PARTICLES, 0.1).run(&target, 3).unwrap_err();
    assert_eq!(error, "the engine failed");

    // The first move of the step runs; a later one fails.
    let later = Scripted::new(unit_uniform(), |values, call| {
        if call < 2 * PARTICLES {
            Ok(Some(50.0 * values[0]))
        } else {
            Err("the engine failed".to_string())
        }
    });
    let error = sampler(PARTICLES, 0.01).run(&later, 3).unwrap_err();
    assert_eq!(error, "the engine failed");
}

#[test]
fn particles_outside_the_support_keep_no_weight_and_do_not_move() {
    // Draws above 0.5 are outside the support from the start.
    let target = Scripted::new(unit_uniform(), |values, _| {
        Ok((values[0] <= 0.5).then(|| 50.0 * values[0]))
    });
    // A threshold this low never resamples, so the particles of weight 0
    // stay and are not moved.
    let run = SmcSampler::new(PARTICLES, RandomWalk::new(), 0.01, 0.5, 0.1)
        .unwrap()
        .run(&target, 4)
        .unwrap();
    assert!(run.steps.iter().all(|step| !step.resampled));
    assert!(run.weights().contains(&0.0));
    for ((values, kept), weight) in run.values.iter().zip(&run.kept).zip(run.weights()) {
        if weight > 0.0 {
            assert!(values[0] <= 0.5 && kept.is_some());
        }
    }
    assert!(run.steps.iter().all(|step| step.level > 0.0));
}

#[test]
fn a_proposal_the_prior_rules_out_runs_no_model() {
    // A log-normal on [−1, ∞) has no density at or below 0, which the walk
    // on log(θ + 1) can propose: such a proposal is rejected unrun.
    let prior = Distribution::new(
        Family::LogNormal {
            mu: -2.0,
            sigma: 1.0,
        },
        -1.0,
        f64::INFINITY,
    )
    .unwrap();
    let target = Scripted::new(prior, |values, _| Ok(Some(values[0])));
    let run = sampler(PARTICLES, 0.1).run(&target, 5).unwrap();
    let moved: usize = run.steps.iter().map(|step| step.moves * PARTICLES).sum();
    assert!(run.evaluations < PARTICLES + moved, "{}", run.evaluations);
    assert!(run.values.iter().all(|values| values[0] > 0.0));
}

/// One output `x = 10 k` at every position, `NaN` for `0.5 < k ≤ 0.7`, and
/// a failed run above; with `flat`, `x = 1` for every `k`.
#[derive(Clone, Debug)]
struct Brittle {
    k: f64,
    flat: bool,
}

#[derive(Clone, Debug)]
struct NoPlan;

impl ValuePlan for NoPlan {
    fn infeasibility(&self, excess: f64) -> String {
        format!("{excess}")
    }
}

impl SimulationEngine for Brittle {
    type ValuePlan = NoPlan;

    fn check_position(&self, _position: f64) -> Result<(), String> {
        Ok(())
    }

    fn run_at_positions_into_buffer(
        &mut self,
        positions: &[f64],
        buffer: &mut Vec<Vec<f64>>,
    ) -> Result<(), SimulationError> {
        let x = if self.flat {
            1.0
        } else if self.k > 0.7 {
            return Err(SimulationError::Failed("the run failed".to_string()));
        } else if self.k > 0.5 {
            f64::NAN
        } else {
            10.0 * self.k
        };
        buffer.clear();
        buffer.extend(positions.iter().map(|_| vec![x]));
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

    fn set_parameter(&mut self, _parameter_id: &str, value: f64) -> Result<(), String> {
        self.k = value;
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

/// `k` in `[0, 1]` of `Brittle`, observed as 2 at positions 0 to 3.
fn brittle_problem(flat: bool) -> (CalibrationProblem<Brittle>, Priors) {
    let problem = CalibrationProblem::new(
        Brittle { k: 0.2, flat },
        (0..4)
            .map(|t| ObservedDataPoint::new(f64::from(t), "x".to_string(), 2.0))
            .collect(),
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    let priors =
        Priors::uniform_on_bounds(&problem.parameter_names(), &problem.parameter_bounds()).unwrap();
    (problem, priors)
}

#[test]
fn a_gibbs_target_needs_a_finite_eta_above_zero() {
    let (problem, priors) = brittle_problem(false);
    for eta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = GibbsTarget::new(&problem, &priors, eta).err().unwrap();
        assert!(error.contains("finite number above 0"), "{error}");
    }
}

#[test]
fn abc_keeps_no_distance_outside_the_support_and_stops_on_a_failure() {
    let (problem, priors) = brittle_problem(false);
    let target = AbcTarget::new(&problem, &priors, Distance::Loss).unwrap();
    let mut stream = Generator::from_seed(1);
    let mut workspace = Workspace::default();
    let mut evaluate = |k: f64| target.evaluate(&[k], &mut stream, &mut workspace);

    // (2 − 10 k)² at four positions.
    assert_eq!(evaluate(0.1), Ok(Some(4.0)));
    // Outside the bounds, and a run that is not a number.
    assert_eq!(evaluate(1.5), Ok(None));
    assert_eq!(evaluate(0.6), Ok(None));
    assert_eq!(
        evaluate(0.8),
        Err("Simulation failed: the run failed".to_string())
    );
}

#[test]
fn bootstrap_coverage_fails_when_every_replicate_fails() {
    // The prior draws runs above 0.7 that fail, so every replicate fails.
    let (problem, priors) = brittle_problem(false);
    let blocks = BlockScheme::new(problem.observed_data(), 1).unwrap();
    let error = bootstrap_coverage_eta(
        &problem,
        &priors,
        &[0.21],
        &mut sampler(PARTICLES, 0.1),
        &blocks,
        3,
        0.8,
        7,
    )
    .unwrap_err();
    assert!(error.starts_with("Every replicate failed (3)"), "{error}");
}

#[test]
fn bootstrap_coverage_fails_when_no_eta_narrows_the_intervals() {
    // No output reads k: every posterior is the prior, whose interval holds
    // the fit at every eta.
    let (problem, priors) = brittle_problem(true);
    let blocks = BlockScheme::new(problem.observed_data(), 1).unwrap();
    let error = bootstrap_coverage_eta(
        &problem,
        &priors,
        &[0.5],
        &mut sampler(PARTICLES, 0.1),
        &blocks,
        5,
        0.8,
        8,
    )
    .unwrap_err();
    assert!(error.contains("still cover the fit"), "{error}");
}

#[test]
fn a_calibration_of_eta_needs_a_fit_inside_the_support() {
    let (problem, priors) = brittle_problem(false);
    let blocks = BlockScheme::new(problem.observed_data(), 1).unwrap();
    // The run at 0.6 is not a number.
    let error = bootstrap_coverage_eta(
        &problem,
        &priors,
        &[0.6],
        &mut sampler(PARTICLES, 0.1),
        &blocks,
        5,
        0.8,
        8,
    )
    .unwrap_err();
    assert_eq!(error, "The fit is outside the support of the problem");
    let error = information_matching_eta(&problem, &[0.6]).unwrap_err();
    assert!(error.contains("infeasible"), "{error}");
}
