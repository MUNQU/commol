//! Methods that take one seed draw from two streams of the registry of
//! `commol-random`.

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig, ObservedDataPoint,
    OptimizationConfig, Workspace, start_points,
};
use commol_core::Model;
use commol_difference::DifferenceEquations;
use commol_inference::Priors;
use commol_inference::design::InputSpace;
use commol_inference::global_sensitivity::ForwardAnalysis;
use commol_inference::importance::TemperedPosterior;
use commol_inference::local::Gaussian;
use commol_inference::resampling::{Perturbation, ReplicateRunner, Resampler};
use commol_inference::smc::{RandomWalk, SmcSampler, TemperedTarget};
use commol_inference::target::Target;
use commol_inference::transform::Transform;
use commol_random::Stream;
use commol_random::sample::normal;
use commol_random::{Generator, latin_hypercube};
use nalgebra::DMatrix;

const SEED: u64 = 11;

const DRAWS: usize = 40;

/// A → B at the proportional rate `k`, from A = 1000.
fn decay() -> Model {
    serde_json::from_str(
        r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations","settings":null},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k"}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#,
    )
    .expect("valid model JSON")
}

#[test]
fn two_methods_with_one_seed_draw_from_two_streams() {
    // A standard normal on the real line: a Laplace draw is the first normal
    // of its stream.
    let laplace = Gaussian::new(
        Transform::new(&[(f64::NEG_INFINITY, f64::INFINITY)]),
        vec![0.0],
        DMatrix::identity(1, 1),
        None,
    )
    .unwrap();
    let laplace_stream = Generator::from_seed(Stream::LocalDraws.seed(SEED));
    assert_eq!(
        laplace.draws(1, SEED)[0][0],
        normal(&mut laplace_stream.clone())
    );

    // One point of a Latin hypercube of a uniform input on [0, 1] is the
    // first uniform of the stream of the designs.
    let k = vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)];
    let inputs = Priors::uniform_on_bounds(&["k".to_string()], &[(0.0, 1.0)]).unwrap();
    let engine = DifferenceEquations::try_from_model(&decay()).unwrap();
    let analysis = ForwardAnalysis::new(
        engine,
        k,
        None,
        None,
        InputSpace::Independent(inputs),
        Some(SEED),
    )
    .unwrap();
    let design_unit = analysis.draws("PRCC", Some(1)).unwrap().values()[0][0];
    let design_stream = Generator::from_seed(Stream::SensitivityDesign.seed(SEED));
    assert_eq!(design_unit, design_stream.clone().uniform());

    assert_ne!(laplace_stream.clone().uniform(), design_unit);
}

#[test]
fn a_calibration_and_a_sensitivity_analysis_with_one_seed_draw_from_two_streams() {
    // The Latin hypercube of one row of the starts of a calibration over
    // [0, 1], and the design of one point of a sensitivity analysis of a
    // uniform input on [0, 1], with one seed.
    let k = vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)];
    let start = start_points(&k, 2, SEED)[1][0];
    let inputs = Priors::uniform_on_bounds(&["k".to_string()], &[(0.0, 1.0)]).unwrap();
    let engine = DifferenceEquations::try_from_model(&decay()).unwrap();
    let analysis = ForwardAnalysis::new(
        engine,
        k,
        None,
        None,
        InputSpace::Independent(inputs),
        Some(SEED),
    )
    .unwrap();
    let design = analysis.draws("PRCC", Some(1)).unwrap().values()[0][0];

    let starts_stream = Generator::from_seed(Stream::StartPoints.seed(SEED));
    assert_eq!(
        start,
        latin_hypercube(&mut starts_stream.clone(), 1, 1)[0][0]
    );
    let design_stream = Generator::from_seed(Stream::SensitivityDesign.seed(SEED));
    assert_eq!(
        design,
        latin_hypercube(&mut design_stream.clone(), 1, 1)[0][0]
    );
    assert_ne!(start, design);
}

/// A resampler whose replicate is the first uniform of its generator.
struct FirstUniform;

impl Resampler for FirstUniform {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        Perturbation::Weights(vec![generator.open_uniform()])
    }
}

/// A target on `[0, 1]` whose prior draw is the first uniform of the stream
/// of a particle.
struct FirstUniformPrior;

impl TemperedTarget for FirstUniformPrior {
    fn bounds(&self) -> &[(f64, f64)] {
        &[(0.0, 1.0)]
    }

    fn log_prior(&self, _values: &[f64]) -> f64 {
        0.0
    }

    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64> {
        vec![stream.open_uniform()]
    }

    fn evaluate(
        &self,
        _values: &[f64],
        _stream: &mut Generator,
        _workspace: &mut Workspace,
    ) -> Result<Option<f64>, String> {
        Ok(Some(0.0))
    }

    fn log_kernel(&self, _value: f64, _level: f64) -> f64 {
        0.0
    }

    fn first_level(&self, _kept: &[f64]) -> f64 {
        0.0
    }

    fn last_level(&self) -> f64 {
        1.0
    }
}

#[test]
fn a_bootstrap_and_an_smc_sampler_with_one_seed_share_no_generator() {
    let engine = DifferenceEquations::try_from_model(&decay()).unwrap();
    let problem = CalibrationProblem::new(
        engine,
        vec![ObservedDataPoint::new(1.0, "A".to_string(), 900.0)],
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap();
    let config = OptimizationConfig::NelderMead(NelderMeadConfig::default());
    let runner = ReplicateRunner::new(&problem, &config, &[0.1], 1, SEED).unwrap();
    let bootstrap: Vec<f64> = runner
        .perturbations(&FirstUniform, DRAWS)
        .into_iter()
        .map(|perturbation| match perturbation {
            Perturbation::Weights(first) => first[0],
            Perturbation::Values(_) => unreachable!("FirstUniform draws weights"),
        })
        .collect();

    let sampler = SmcSampler::new(DRAWS, RandomWalk::new(), 0.5, 0.5, 0.1).unwrap();
    let state = sampler.start(&FirstUniformPrior, SEED).unwrap();
    let (particles, _, _) = state.particles();

    for particle in particles {
        assert!(!bootstrap.contains(&particle[0]), "{}", particle[0]);
    }
}

/// A target on `[0, 1]` whose run keeps the first uniform of its stream, as a
/// stochastic model keeps the run of the seed it draws.
struct StochasticRun;

impl TemperedTarget for StochasticRun {
    fn bounds(&self) -> &[(f64, f64)] {
        &[(0.0, 1.0)]
    }

    fn log_prior(&self, _values: &[f64]) -> f64 {
        0.0
    }

    fn sample_prior(&self, stream: &mut Generator) -> Vec<f64> {
        vec![stream.open_uniform()]
    }

    fn evaluate(
        &self,
        _values: &[f64],
        stream: &mut Generator,
        _workspace: &mut Workspace,
    ) -> Result<Option<f64>, String> {
        Ok(Some(stream.open_uniform()))
    }

    fn log_kernel(&self, value: f64, _level: f64) -> f64 {
        value
    }

    fn first_level(&self, _kept: &[f64]) -> f64 {
        0.0
    }

    fn last_level(&self) -> f64 {
        1.0
    }
}

#[test]
fn a_tempered_posterior_runs_the_replicate_of_its_seed() {
    let mut workspace = Workspace::default();
    let density = |seed: u64, workspace: &mut Workspace| {
        TemperedPosterior::new(&StochasticRun, seed)
            .log_density(&[0.5], workspace)
            .unwrap()
    };

    let first = density(SEED, &mut workspace);
    let posterior = TemperedPosterior::new(&StochasticRun, SEED);
    assert_eq!(
        posterior.log_density(&[0.5], &mut workspace).unwrap(),
        first
    );
    assert_eq!(
        posterior.log_density(&[0.5], &mut workspace).unwrap(),
        first
    );
    assert_ne!(density(SEED + 1, &mut workspace), first);
}
