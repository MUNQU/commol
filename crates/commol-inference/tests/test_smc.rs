//! The sequential Monte Carlo core on targets with closed forms: Gaussian
//! tempering with its evidence, a target near a bound, the rule of the moves
//! and of the levels, and ABC rejection.

use std::sync::atomic::{AtomicUsize, Ordering};

use commol_calibration::Workspace;
use commol_inference::smc::{
    RandomWalk, SmcRun, SmcSampler, TemperedTarget, abc_rejection, next_level,
};
use commol_inference::{Distribution, Family};
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
