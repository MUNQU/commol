//! Every setting of a calibration that has a range rejects a value outside
//! it with the message of its rule: in the check of its object of the Python
//! API ([`RANGES`]) and in the check of the engine that takes it, with no
//! Python object in between.

use commol_calibration::ranges::{self, RANGES};
use commol_calibration::{
    AccelerationStrategy, CalibrationConstraint, CalibrationParameter, ForwardModel,
    ImprovementStop, InertiaWeightStrategy, MutationStrategy, NelderMeadConfig, ObservedDataPoint,
    ParticleSwarmConfig,
};
use commol_core::{Count, Model, Real, Rule, Setting};
use commol_difference::DifferenceEquations;

/// The message of `rule` for its value outside the range.
fn real_message(rule: Real) -> String {
    rule.check(rule.outside()).unwrap_err()
}

/// The message of `rule` for its value outside the range.
fn count_message(rule: Count) -> String {
    rule.check(rule.outside()).unwrap_err()
}

/// A → B at the rate `k`, in discrete steps from A = 1000.
fn engine() -> DifferenceEquations {
    let json = r#"{"name":"m","parameters":[{"id":"k","value":0.1}],
        "engine":{"type":"DifferenceEquations"},
        "process":{"bins":[{"id":"A","name":"A"},{"id":"B","name":"B"}],
        "accumulators":[],"stratifications":[],
        "transitions":[{"id":"ab","source":["A"],
        "target":["B"],"rate":"k","stratified_rates":null,"per_compartment":null}]},
        "structure":{"kind":"aggregate","bin_amounts":{"A":1000.0,"B":0.0}}}"#;
    let model: Model = serde_json::from_str(json).expect("valid model JSON");
    DifferenceEquations::try_from_model(&model).unwrap()
}

fn point() -> ObservedDataPoint {
    ObservedDataPoint::new(1.0, "A".to_string(), 1.0)
}

fn swarm() -> ParticleSwarmConfig {
    ParticleSwarmConfig::default()
}

#[test]
fn every_setting_outside_its_range_is_rejected() {
    for ranges in RANGES {
        for rule in ranges.rules {
            let expected = match rule {
                Rule::Count(rule) => count_message(*rule),
                Rule::Real(rule) => real_message(*rule),
            };
            assert_eq!(
                ranges.check(&[(rule.name(), rule.outside())]),
                Err(expected.clone()),
                "{}",
                ranges.object
            );
            if let Rule::Count(count) = rule {
                let wording = expected.replace(&format!("got {}", count.outside()), "got -1");
                assert_eq!(
                    ranges.check(&[(rule.name(), Setting::Whole(-1))]),
                    Err(wording)
                );
            }
        }
    }

    let observation = |change: fn(&mut ObservedDataPoint)| {
        let mut point = point();
        change(&mut point);
        point.check_ranges().unwrap_err()
    };
    let constraint = |weight: f64, at: Vec<f64>| {
        CalibrationConstraint::new("c".to_string(), "k".to_string())
            .with_weight(weight)
            .with_positions(at)
            .check_ranges()
            .unwrap_err()
    };
    let simplex = |change: fn(&mut NelderMeadConfig)| {
        let mut config = NelderMeadConfig::default();
        change(&mut config);
        config.check_ranges().unwrap_err()
    };
    let particles = |change: fn(&mut ParticleSwarmConfig)| {
        let mut config = swarm();
        change(&mut config);
        config.check_ranges().unwrap_err()
    };
    let unobserved = ForwardModel::unobserved(
        engine(),
        vec![CalibrationParameter::new("k".to_string(), 0.0, 1.0)],
        None,
        Some(0),
        None,
    )
    .err()
    .unwrap();
    let cases = [
        (observation(|p| p.at = -1.0), real_message(ranges::AT)),
        (observation(|p| p.value = -1.0), real_message(ranges::VALUE)),
        (
            observation(|p| p.weight = 0.0),
            real_message(ranges::WEIGHT),
        ),
        (
            observation(|p| p.window = Some(0.0)),
            real_message(ranges::WINDOW),
        ),
        (constraint(0.0, vec![1.0]), real_message(ranges::WEIGHT)),
        (constraint(1.0, vec![-1.0]), real_message(ranges::AT)),
        (
            simplex(|c| c.max_iterations = 0),
            count_message(ranges::MAX_ITERATIONS),
        ),
        (
            simplex(|c| c.sd_tolerance = 0.0),
            real_message(ranges::SD_TOLERANCE),
        ),
        (
            simplex(|c| c.initial_simplex_size = 0.0),
            real_message(ranges::INITIAL_SIMPLEX_SIZE),
        ),
        (
            simplex(|c| c.alpha = Some(0.0)),
            real_message(ranges::ALPHA),
        ),
        (
            simplex(|c| c.gamma = Some(0.0)),
            real_message(ranges::GAMMA),
        ),
        (simplex(|c| c.rho = Some(0.0)), real_message(ranges::RHO)),
        (
            simplex(|c| c.sigma = Some(0.0)),
            real_message(ranges::SIGMA),
        ),
        (
            particles(|c| c.particles = 0),
            count_message(ranges::PARTICLES),
        ),
        (
            particles(|c| c.max_iterations = 0),
            count_message(ranges::MAX_ITERATIONS),
        ),
        (
            particles(|c| c.inertia_strategy = Some(InertiaWeightStrategy::Constant(0.0))),
            real_message(ranges::INERTIA_FACTOR),
        ),
        (
            particles(|c| {
                c.inertia_strategy = Some(InertiaWeightStrategy::Chaotic {
                    w_min: 0.0,
                    w_max: 1.0,
                })
            }),
            real_message(ranges::INERTIA_MIN),
        ),
        (
            particles(|c| {
                c.inertia_strategy = Some(InertiaWeightStrategy::Chaotic {
                    w_min: 1.0,
                    w_max: 0.0,
                })
            }),
            real_message(ranges::INERTIA_MAX),
        ),
        (
            particles(|c| {
                c.acceleration_strategy = Some(AccelerationStrategy::Constant {
                    cognitive: 0.0,
                    social: 1.0,
                })
            }),
            real_message(ranges::COGNITIVE),
        ),
        (
            particles(|c| {
                c.acceleration_strategy = Some(AccelerationStrategy::Constant {
                    cognitive: 1.0,
                    social: 0.0,
                })
            }),
            real_message(ranges::SOCIAL),
        ),
        (
            particles(|c| {
                c.acceleration_strategy = Some(AccelerationStrategy::TimeVarying {
                    c1_initial: 0.0,
                    c1_final: 1.0,
                    c2_initial: 1.0,
                    c2_final: 1.0,
                })
            }),
            real_message(ranges::COGNITIVE_INITIAL),
        ),
        (
            particles(|c| {
                c.acceleration_strategy = Some(AccelerationStrategy::TimeVarying {
                    c1_initial: 1.0,
                    c1_final: 0.0,
                    c2_initial: 1.0,
                    c2_final: 1.0,
                })
            }),
            real_message(ranges::COGNITIVE_FINAL),
        ),
        (
            particles(|c| {
                c.acceleration_strategy = Some(AccelerationStrategy::TimeVarying {
                    c1_initial: 1.0,
                    c1_final: 1.0,
                    c2_initial: 0.0,
                    c2_final: 1.0,
                })
            }),
            real_message(ranges::SOCIAL_INITIAL),
        ),
        (
            particles(|c| {
                c.acceleration_strategy = Some(AccelerationStrategy::TimeVarying {
                    c1_initial: 1.0,
                    c1_final: 1.0,
                    c2_initial: 1.0,
                    c2_final: 0.0,
                })
            }),
            real_message(ranges::SOCIAL_FINAL),
        ),
        (
            particles(|c| c.mutation_strategy = MutationStrategy::Gaussian(0.0)),
            real_message(ranges::MUTATION_SCALE),
        ),
        (
            particles(|c| c.mutation_probability = -1.0),
            real_message(ranges::MUTATION_PROBABILITY),
        ),
        (
            particles(|c| c.velocity_clamp_factor = Some(0.0)),
            real_message(ranges::VELOCITY_CLAMP_FACTOR),
        ),
        (
            particles(|c| c.velocity_mutation_threshold = Some(-1.0)),
            real_message(ranges::VELOCITY_MUTATION_THRESHOLD),
        ),
        (
            particles(|c| {
                c.improvement_stop = Some(ImprovementStop {
                    window: 0,
                    tol_abs: 0.0,
                    tol_rel: 0.0,
                })
            }),
            count_message(ranges::IMPROVEMENT_WINDOW),
        ),
        (
            particles(|c| {
                c.improvement_stop = Some(ImprovementStop {
                    window: 1,
                    tol_abs: -1.0,
                    tol_rel: 0.0,
                })
            }),
            real_message(ranges::IMPROVEMENT_TOL_ABS),
        ),
        (
            particles(|c| {
                c.improvement_stop = Some(ImprovementStop {
                    window: 1,
                    tol_abs: 0.0,
                    tol_rel: -1.0,
                })
            }),
            real_message(ranges::IMPROVEMENT_TOL_REL),
        ),
        (
            particles(|c| c.radius_stop = Some(-1.0)),
            real_message(ranges::RADIUS_TOL),
        ),
        (unobserved, count_message(ranges::REPLICATES)),
    ];

    for (error, message) in cases {
        assert!(
            error.ends_with(&message),
            "{error} does not end with {message}"
        );
    }
}
