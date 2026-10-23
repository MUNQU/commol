//! A parameter or a seed changed between two runs of `StochasticEvents` acts
//! from the position where the first run stopped.

use commol_core::{Model, SimulationEngine};
use commol_random::replicate_seed;
use commol_stochastic::StochasticEvents;

/// `A → D` at the rate `k·A`, from `units` units of `A`, run by
/// `StochasticEvents` with `settings` (a JSON object or `null`).
fn death(settings: &str, units: f64, k: f64) -> StochasticEvents {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":{k:?}}}],
        "engine":{{"type":"StochasticEvents","settings":{settings}}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"D","name":"D"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"death","source":["A"],"target":["D"],"rate":"k"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":{units:?},"D":0.0}}}}}}"#
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    StochasticEvents::try_from_model(&model).expect("the model compiles")
}

/// The seed of replicate `index`.
fn seed(index: u64) -> u64 {
    replicate_seed(11, index)
}

/// Mean and standard error of the mean of `values`.
fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// A fixed leap of 10 from 0 with the propensity `0.01 · 1000 = 10`: 100
/// events on average, of which a run that stops at 4 owns 40.
const LEAP: &str = r#"{"method":"tau_leaping","tau":10.0}"#;

#[test]
fn a_parameter_change_acts_from_the_position() {
    // Direct method: with k = 0 up to 5 and k = 1 from 5, A(6) is a
    // binomial of 1000 units that each stay with the probability e^{-1}.
    let direct = death("null", 1000.0, 0.0);
    let left: Vec<f64> = (0..200)
        .map(|index| {
            let mut engine = direct.clone();
            engine.set_seed(seed(index));
            engine.run_at(&[5.0]).unwrap();
            engine.set_parameter("k", 1.0).unwrap();
            engine.run_at(&[6.0]).unwrap()[0][0]
        })
        .collect();
    let (mean, error) = mean_and_error(&left);
    let expected = 1000.0 * (-1.0_f64).exp();
    assert!(
        (mean - expected).abs() < 4.0 * error,
        "{mean} against {expected} ± {error}"
    );

    // Tau-leaping: k = 0 from 4 keeps only the events of the leap up to 4.
    let leaping = death(LEAP, 1000.0, 0.01);
    let moved: Vec<f64> = (0..400)
        .map(|index| {
            let mut engine = leaping.clone();
            engine.set_seed(seed(index));
            engine.run_at(&[4.0]).unwrap();
            engine.set_parameter("k", 0.0).unwrap();
            1000.0 - engine.run_at(&[10.0]).unwrap()[0][0]
        })
        .collect();
    let (mean, error) = mean_and_error(&moved);
    assert!(
        (mean - 40.0).abs() < 4.0 * error,
        "{mean} against 40 ± {error}"
    );
}

#[test]
fn a_new_seed_acts_from_the_position() {
    // Direct method: one unit that leaves at the rate 0.01 is still there at
    // 5; each copy with a new seed draws its own wait from 5, of mean 100.
    let mut engine = death("null", 1.0, 0.01);
    engine.set_seed(3);
    assert_eq!(engine.run_at(&[5.0]).unwrap()[0][0], 1.0);
    let grid: Vec<f64> = (0..=2000).map(|i| 5.0 + f64::from(i)).collect();
    let waits: Vec<f64> = (0..400)
        .map(|index| {
            let mut copy = engine.clone();
            copy.set_seed(seed(index));
            let rows = copy.run_at(&grid).unwrap();
            let first = rows.iter().position(|row| row[0] == 0.0).expect("gone");
            grid[first] - 0.5 - 5.0
        })
        .collect();
    assert!(waits.iter().any(|&wait| wait != waits[0]));
    let (mean, error) = mean_and_error(&waits);
    assert!(
        (mean - 100.0).abs() < 4.0 * error + 0.5,
        "{mean} against 100 ± {error}"
    );

    // Tau-leaping: a new seed at 4 keeps on average the 40 events of the
    // leap up to 4, and the copies draw their own leaps from there.
    let leaping = death(LEAP, 1000.0, 0.01);
    let moved: Vec<f64> = (0..400)
        .map(|index| {
            let mut engine = leaping.clone();
            engine.set_seed(seed(index));
            engine.run_at(&[4.0]).unwrap();
            engine.set_seed(seed(index + 400));
            1000.0 - engine.state()[0]
        })
        .collect();
    let (mean, error) = mean_and_error(&moved);
    assert!(
        (mean - 40.0).abs() < 4.0 * error,
        "{mean} against 40 ± {error}"
    );
    let mut one = death(LEAP, 1000.0, 0.01);
    one.run_at(&[4.0]).unwrap();
    let after: Vec<f64> = (0..20)
        .map(|index| {
            let mut copy = one.clone();
            copy.set_seed(seed(index));
            copy.run_at(&[14.0]).unwrap()[0][0]
        })
        .collect();
    assert!(after.iter().any(|&amount| amount != after[0]), "{after:?}");
}
