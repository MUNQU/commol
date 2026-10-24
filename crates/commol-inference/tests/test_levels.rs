//! One check of a confidence level and one check of a share: every method
//! that takes a setting strictly between 0 and 1 gives the same message.

use commol_calibration::{
    CalibrationParameter, CalibrationProblem, LossConfig, NelderMeadConfig, ObservedDataPoint,
    OptimizationConfig,
};
use commol_core::{CONFIDENCE_LEVEL, Model, Real};
use commol_difference::DifferenceEquations;
use commol_inference::conformal::{AdaptiveLevel, ConformalSettings, NonconformityScore, Pooling};
use commol_inference::identifiability::{Reference, sample_identifiability};
use commol_inference::profiles::{ProfiledQuantity, beale, bootstrap_thresholds, chi_square};
use commol_inference::resampling::{BayesianBootstrap, BlockScheme, ReplicateRunner};
use commol_inference::scores::coverage;
use commol_inference::smc::{AbcTarget, Distance, RandomWalk, SmcSampler, abc_rejection};
use commol_inference::{ObservationDraws, ParameterSample, Priors};

/// A → B at the rate `k`, from A = 1000.
fn decay() -> DifferenceEquations {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],"target":["B"],"rate":"k",
        "stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    DifferenceEquations::try_from_model(&model).unwrap()
}

fn problem() -> CalibrationProblem<DifferenceEquations> {
    CalibrationProblem::new(
        decay(),
        (1..=4)
            .map(|t| ObservedDataPoint::new(f64::from(t), "A".to_string(), 500.0))
            .collect(),
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        vec![],
        LossConfig::SumSquaredError,
    )
    .unwrap()
}

fn sample() -> ParameterSample {
    ParameterSample::new(
        vec!["p0".to_string(), "p1".to_string()],
        vec![(0.0, 1.0); 2],
        vec![vec![0.2, 0.7], vec![0.4, 0.3], vec![0.6, 0.5]],
        &[0.0; 3],
        None,
    )
    .unwrap()
}

#[test]
fn every_confidence_level_has_one_check() {
    let message =
        "confidence_level is strictly between 0 and 1, got 0; use 0.95 for a 95% interval";
    let problem = problem();
    let config = OptimizationConfig::NelderMead(NelderMeadConfig::default());
    let runner = ReplicateRunner::new(&problem, &config, &[0.1], 1, 0).unwrap();
    let blocks = BlockScheme::new(problem.observed_data(), 1).unwrap();
    let resampler = BayesianBootstrap::new(problem.observed_data(), &blocks);
    let draws = ObservationDraws {
        values: vec![vec![1.0], vec![2.0]],
        weights: vec![0.5, 0.5],
        seed: 0,
    };

    let errors = [
        CONFIDENCE_LEVEL.check(0.0).unwrap_err(),
        sample().interval(0.0).unwrap_err(),
        coverage(&draws, &[1.5], 0.0).unwrap_err(),
        ConformalSettings::new(NonconformityScore::Absolute, Pooling::All, 0.0)
            .unwrap_err()
            .to_string(),
        AdaptiveLevel::new(0.0, 0.01).unwrap_err().to_string(),
        chi_square(&problem, 0.0).unwrap_err().to_string(),
        beale(&problem, 1.0, 0.0).unwrap_err().to_string(),
        bootstrap_thresholds(
            &runner,
            &resampler,
            2,
            &[ProfiledQuantity::Parameter(0)],
            &[0.1],
            0.0,
        )
        .unwrap_err()
        .to_string(),
    ];

    for error in errors {
        assert_eq!(error, message);
    }
}

#[test]
fn every_share_has_one_check() {
    let problem = problem();
    let priors =
        Priors::uniform_on_bounds(&problem.parameter_names(), &problem.parameter_bounds()).unwrap();
    let target = AbcTarget::new(&problem, &priors, Distance::Loss).unwrap();
    let sampler = || SmcSampler::new(10, RandomWalk::new(), 0.5, 0.5, 0.1);

    let errors = [
        (
            "target_cess",
            SmcSampler::new(10, RandomWalk::new(), 0.5, 1.0, 0.1)
                .err()
                .unwrap()
                .to_string(),
        ),
        (
            "stay_probability",
            SmcSampler::new(10, RandomWalk::new(), 0.5, 0.5, 1.0)
                .err()
                .unwrap()
                .to_string(),
        ),
        (
            "min_acceptance",
            sampler()
                .unwrap()
                .with_min_acceptance(1.0)
                .err()
                .unwrap()
                .to_string(),
        ),
        (
            "quantile",
            abc_rejection(&target, 10, 1.0, 0).unwrap_err().to_string(),
        ),
        (
            "correlation_threshold",
            sample_identifiability(&sample(), Reference::Bounds, 1.0)
                .unwrap_err()
                .to_string(),
        ),
        ("share", Real::share("share").check(1.0).unwrap_err()),
    ];

    for (name, error) in errors {
        assert_eq!(error, format!("{name} is strictly between 0 and 1, got 1"));
    }
}
