//! Every setting of an inference method that has a range rejects a value
//! outside it with the message of its rule: in the check of its object of the
//! Python API ([`RANGES`]) and in the check of the engine that takes it, with
//! no Python object in between.

use commol_calibration::ObservedDataPoint;
use commol_core::{Count, Real, Rule, Setting};
use commol_inference::design::InputSpace;
use commol_inference::global_sensitivity::Truncation;
use commol_inference::ranges::{self, RANGES};
use commol_inference::resampling::BlockScheme;
use commol_inference::smc::{RandomWalk, SmcSampler};
use commol_inference::{Distribution, Family, Priors};

/// The message of `rule` for its value outside the range.
fn real_message(rule: Real) -> String {
    rule.check(rule.outside()).unwrap_err()
}

/// The message of `rule` for its value outside the range.
fn count_message(rule: Count) -> String {
    rule.check(rule.outside()).unwrap_err()
}

fn sampler(
    particles: usize,
    resample_threshold: f64,
    target_cess: f64,
    stay_probability: f64,
) -> String {
    SmcSampler::new(
        particles,
        RandomWalk::new(),
        resample_threshold,
        target_cess,
        stay_probability,
    )
    .err()
    .unwrap()
    .to_string()
}

fn distribution(family: Family) -> String {
    Distribution::new(family, 0.0, 1.0).unwrap_err()
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

    let space = InputSpace::Independent(
        Priors::uniform_on_bounds(&["a".to_string()], &[(0.0, 1.0)]).unwrap(),
    );
    let observations = [ObservedDataPoint::new(1.0, "x".to_string(), 1.0)];
    let cases = [
        (sampler(1, 0.5, 0.5, 0.5), count_message(ranges::PARTICLES)),
        (
            sampler(10, ranges::RESAMPLE_THRESHOLD.outside(), 0.5, 0.5),
            real_message(ranges::RESAMPLE_THRESHOLD),
        ),
        (
            sampler(10, 0.5, ranges::TARGET_CESS.outside(), 0.5),
            real_message(ranges::TARGET_CESS),
        ),
        (
            sampler(10, 0.5, 0.5, ranges::STAY_PROBABILITY.outside()),
            real_message(ranges::STAY_PROBABILITY),
        ),
        (
            SmcSampler::new(10, RandomWalk::new(), 0.5, 0.5, 0.5)
                .unwrap()
                .with_min_acceptance(ranges::MIN_ACCEPTANCE.outside())
                .err()
                .unwrap()
                .to_string(),
            real_message(ranges::MIN_ACCEPTANCE),
        ),
        (
            distribution(Family::Beta { a: 0.0, b: 1.0 }),
            real_message(ranges::A),
        ),
        (
            distribution(Family::Beta { a: 1.0, b: 0.0 }),
            real_message(ranges::B),
        ),
        (
            distribution(Family::Normal {
                mean: f64::INFINITY,
                sd: 1.0,
            }),
            real_message(ranges::MEAN),
        ),
        (
            distribution(Family::Normal { mean: 0.0, sd: 0.0 }),
            real_message(ranges::SD),
        ),
        (
            distribution(Family::LogNormal {
                mu: f64::INFINITY,
                sigma: 1.0,
            }),
            real_message(ranges::MU),
        ),
        (
            distribution(Family::LogNormal {
                mu: 0.0,
                sigma: 0.0,
            }),
            real_message(ranges::SIGMA),
        ),
        (
            distribution(Family::Gamma {
                shape: 0.0,
                rate: 1.0,
            }),
            real_message(ranges::SHAPE),
        ),
        (
            distribution(Family::Gamma {
                shape: 1.0,
                rate: 0.0,
            }),
            real_message(ranges::RATE),
        ),
        (
            Truncation::new(0.0, None).unwrap_err().to_string(),
            real_message(ranges::Q_NORM),
        ),
        (
            Truncation::new(0.5, Some(0)).unwrap_err().to_string(),
            count_message(ranges::MAX_INTERACTION),
        ),
        (
            BlockScheme::new(&observations, 0).unwrap_err().to_string(),
            count_message(ranges::BLOCK_LENGTH),
        ),
        (
            space.morris("Morris", 0, 4, 1).err().unwrap(),
            count_message(ranges::TRAJECTORIES),
        ),
        (
            space.morris("Morris", 2, 1, 1).err().unwrap(),
            count_message(ranges::LEVELS),
        ),
        (
            space.saltelli("Sobol", 1, 1).err().unwrap(),
            count_message(ranges::TWO_DRAWS),
        ),
    ];

    for (error, message) in cases {
        assert!(
            error.ends_with(&message),
            "{error} does not end with {message}"
        );
    }
}
