//! A parameter or a seed changed between two runs of `StochasticEvents` acts
//! from the position where the first run stopped, also a stop at `-0.0`,
//! which is step 0, and a run that ends next to a stop ends at that stop.

use commol_core::{Model, ParameterValue, SeriesMode, SimulationEngine};
use commol_random::replicate_seed;
use commol_stochastic::StochasticEvents;

/// `A → D` at the rate `k·A`, from `units` units of `A`, for
/// `StochasticEvents` with `settings` (a JSON object or `null`).
fn death_model(settings: &str, units: f64, k: f64) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":{k:?}}}],
        "engine":{{"type":"StochasticEvents","settings":{settings}}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"D","name":"D"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"death","source":["A"],"target":["D"],"rate":"k"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":{units:?},"D":0.0}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

/// The engine of [`death_model`].
fn death(settings: &str, units: f64, k: f64) -> StochasticEvents {
    StochasticEvents::try_from_model(&death_model(settings, units, k)).expect("the model compiles")
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

#[test]
fn a_seed_set_at_minus_zero_acts_the_pulse_of_step_zero() {
    let mut model = death_model("null", 1000.0, 0.0);
    model.parameters[0].value = Some(ParameterValue::Series {
        data: vec![(0.0, 1.0)],
        mode: SeriesMode::Pulse,
    });
    let engine = StochasticEvents::try_from_model(&model).expect("the model compiles");
    let left = |start: f64| {
        let mut engine = engine.clone();
        engine.run_at(&[start]).unwrap();
        // A seed or a parameter set at a stop settles the clock onto it, so
        // the rates of the next run read the step of `start`.
        engine.set_seed(seed(0));
        engine.run_at(&[1.0]).unwrap()[0][0]
    };

    assert!(left(0.0) < 1000.0);
    assert_eq!(left(-0.0), left(0.0));
}

/// `A → B` at `rate`, with the parameter `k` at `k` and `extra` in the
/// transition, from 1000 units of `A`, for `StochasticEvents` with
/// `settings`.
fn transfer(settings: &str, rate: &str, k: f64, extra: &str) -> StochasticEvents {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":{k:?}}}],
        "engine":{{"type":"StochasticEvents","settings":{settings}}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"ab","source":["A"],"target":["B"],"rate":"{rate}"{extra}}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":1000.0,"B":0.0}}}}}}"#
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    StochasticEvents::try_from_model(&model).expect("the model compiles")
}

/// A fixed leap of 0.05, shorter than a step.
const SHORT_LEAP: &str = r#"{"method":"tau_leaping","tau":0.05}"#;

/// The state at `end` of a run of `engine` from seed 1 that stops at `at`,
/// where `change` acts, and then goes on to `end`.
fn split(
    mut engine: StochasticEvents,
    at: f64,
    end: f64,
    change: impl Fn(&mut StochasticEvents),
) -> Vec<f64> {
    engine.set_seed(1);
    engine.run_at(&[at]).unwrap();
    change(&mut engine);
    engine.run_at(&[end]).unwrap().remove(0)
}

/// `B` at step 4 of a run from seed 1 of `A → B` at the rate 0.5 on step 3
/// alone, with a term of 1e-12 from `t >= on`.
fn step_three(on: f64) -> f64 {
    let rate = format!("if(step == 3, 0.5, 0) + if(t >= {on:?}, 1e-12, 0)");
    let mut engine = transfer("null", &rate, 0.0, "");
    engine.set_seed(1);
    engine.run(5).unwrap().rows[4][1]
}

#[test]
fn a_threshold_just_below_a_step_keeps_the_step() {
    // 0.3 / 0.1 and the others are positions that the engine calls 3.
    let at_three = step_three(3.0);
    assert!(at_three > 0.0, "the rate of step 3 acts: {at_three}");
    for on in [0.3 / 0.1, 3.0 - 1e-9, 3.0_f64.next_up(), 3.0 + 1e-9] {
        assert_eq!(step_three(on), at_three, "t >= {on:?}");
    }
}

#[test]
fn a_run_split_next_to_a_step_resumes_from_that_step() {
    // 0.3 / 0.1 and 3 - 1e-9 are in the stop 3: a run that stops there ends
    // at 3, and the next run resumes from 3 with the rate of step 3.
    let changes: [fn(&mut StochasticEvents); 2] = [
        |engine| engine.set_seed(2),
        |engine| engine.set_parameter("k", 0.5).unwrap(),
    ];
    for settings in ["null", SHORT_LEAP] {
        let engine = transfer(settings, "if(step == 3, k, 0)", 0.5, "");
        for change in changes {
            let at_three = split(engine.clone(), 3.0, 4.0, change);
            assert!(at_three[1] > 0.0, "{settings}: {at_three:?}");
            for at in [0.3 / 0.1, 3.0 - 1e-9] {
                let state = split(engine.clone(), at, 4.0, change);
                assert_eq!(state, at_three, "{settings}, split at {at:?}");
            }
        }
    }
}

#[test]
fn a_run_split_next_to_a_step_far_out_resumes_from_that_step() {
    // At 1e6 the tolerance of a position is 1e-3, so `n - 5e-4` is in the
    // stop `n`, and the rate of step `n` acts from `n`.
    let n = 1e6;
    let engine = transfer("null", &format!("if(step == {n:?}, k, 0)"), 0.0, "");
    let change = |engine: &mut StochasticEvents| engine.set_parameter("k", 1.0).unwrap();
    // The rate 1 on one step moves about 1000 (1 - e^-1) = 632 units.
    let at_n = split(engine.clone(), n, n + 1.0, change);
    assert!(at_n[1] > 500.0, "{at_n:?}");
    assert_eq!(split(engine, n - 5e-4, n + 1.0, change), at_n);
}

#[test]
fn an_impulse_next_to_the_end_of_a_run_acts_before_a_new_seed() {
    // The impulse is in the stop of 0.3 / 0.1, so the first run applies it.
    for on in [3.0, 3.0 + 1e-9] {
        let impulse = format!(r#","impulse_at":[{on:?}]"#);
        let engine = transfer("null", "0.5", 0.0, &impulse);
        let new_seed = |engine: &mut StochasticEvents| engine.set_seed(2);
        let at_three = split(engine.clone(), 3.0, 4.0, new_seed);
        assert!(at_three[1] > 0.0, "{at_three:?}");
        assert_eq!(split(engine, 0.3 / 0.1, 4.0, new_seed), at_three, "{on:?}");
    }
}

#[test]
fn a_row_next_to_an_impulse_shows_the_state_after_it() {
    let engine = transfer("null", "0.5", 0.0, r#","impulse_at":[3.0]"#);
    let rows = |positions: &[f64]| {
        let mut engine = engine.clone();
        engine.set_seed(1);
        engine.run_at(positions).unwrap()
    };
    let at_three = rows(&[3.0, 4.0]);
    assert!(at_three[0][1] > 0.0, "{at_three:?}");
    assert_eq!(rows(&[0.3 / 0.1, 4.0]), at_three);
}

/// `A → B` at `rate` with `k = 0`, and an impulse at `impulse` that moves
/// half of `A`, from seed 1.
fn with_impulse(rate: &str, impulse: f64) -> StochasticEvents {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":0.0}}],
        "engine":{{"type":"StochasticEvents","settings":null}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"B","name":"B"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"move","source":["A"],"target":["B"],"rate":"{rate}"}},
        {{"id":"half","source":["A"],"target":["B"],"rate":"0.5",
        "impulse_at":[{impulse:?}]}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":1000.0,"B":0.0}}}}}}"#
    );
    let model: Model = serde_json::from_str(&json).expect("valid model JSON");
    let mut engine = StochasticEvents::try_from_model(&model).expect("the model compiles");
    engine.set_seed(1);
    engine
}

#[test]
fn a_row_at_an_impulse_next_to_a_stop_shows_the_state_after_it() {
    // The stop 2.5 holds 2.500000002, which is the same position as the
    // impulse. The impulse is a stop of its own, where its row is.
    let rate = "k * if(t >= 2.5, 1, 0) + k * if(t >= 2.500000002, 1, 0)";
    for impulse in [2.500000004, 2.5000000035] {
        let rows = with_impulse(rate, impulse).run_at(&[impulse, 4.0]).unwrap();
        assert!(rows[0][1] > 0.0, "{impulse:?}: {rows:?}");
        assert_eq!(rows[0], rows[1], "{impulse:?}");
    }
}

#[test]
fn two_runs_that_end_next_to_each_other_and_to_an_impulse_are_the_whole_run() {
    // The first run ends at 3.5 - 5e-9, which no stop holds. The second asks
    // for 3.5 - 2.5e-9, the same position as that end and as the impulse:
    // it ends at the impulse, as the whole run does.
    let (first, second) = (3.5 - 5e-9, 3.5 - 2.5e-9);
    for rate in ["k", "k * step"] {
        let whole = with_impulse(rate, 3.5)
            .run_at(&[first, second, 5.0])
            .unwrap();
        assert!(whole[1][1] > 0.0, "{rate}: {whole:?}");
        let mut split = with_impulse(rate, 3.5);
        assert_eq!(split.run_at(&[first]).unwrap(), whole[..1], "{rate}");
        assert_eq!(split.run_at(&[second]).unwrap(), whole[1..2], "{rate}");
        assert_eq!(split.run_at(&[5.0]).unwrap(), whole[2..], "{rate}");
    }
}

#[test]
fn a_row_next_to_a_step_and_another_stop_shows_the_state_after_the_impulse_of_the_step() {
    // 2.9999999985 is the same position as the stop 2.999999996 and as the
    // step 3, where the impulse acts. Its row is at the step.
    let next = 2.999_999_998_5;
    let rate = "k * if(t >= 2.999999996, 1, 0) + k * step";
    let whole = with_impulse(rate, 3.0).run_at(&[next, 4.0]).unwrap();
    assert!(whole[0][1] > 0.0, "{whole:?}");
    assert_eq!(whole[0], whole[1]);
    let mut split = with_impulse(rate, 3.0);
    assert_eq!(split.run_at(&[next]).unwrap(), whole[..1]);
    assert_eq!(split.run_at(&[4.0]).unwrap(), whole[1..]);
}

#[test]
fn a_model_that_reads_the_step_runs_below_5e8() {
    // From 5e8 the tolerance of a position is half a step. The engine
    // refuses the position where it receives it, before a run.
    let mut steps = transfer("null", "k * step", 0.3, "");
    let error = steps.run_at(&[5e8]).unwrap_err().to_string();
    let rule = "a model that reads `step` runs only below position 5e8";
    let start = format!("The StochasticEvents engine cannot run to 500000000: {rule}");
    assert!(error.starts_with(&start), "{error}");
    assert_eq!(steps.check_position(5e8).unwrap_err(), error);
    assert!(steps.check_position(5e8 - 1.0).is_ok());
}

#[test]
fn the_engine_of_events_is_the_stochastic_entry_of_its_name() {
    let engine = death("null", 10.0, 0.1);

    assert_eq!(
        engine.engine_type(),
        Some(commol_core::EngineType::StochasticEvents)
    );
    assert!(engine.is_stochastic());
}
