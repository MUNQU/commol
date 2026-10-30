//! Each stochastic engine refuses to run a model that lacks a value, before
//! it draws an event.

use commol_core::{Model, SimulationEngine};
use commol_stochastic::{StochasticEvents, StochasticSteps};

/// `A → D` at the rate `k·A` from 100 units of `A`, run by `engine`, with
/// `k` and the amount of `D` left without a value.
fn death(engine: &str) -> Model {
    let json = format!(
        r#"{{"name":"m","parameters":[{{"id":"k","value":null}}],
        "engine":{{"type":"{engine}"}},
        "process":{{"bins":[{{"id":"A","name":"A"}},{{"id":"D","name":"D"}}],
        "accumulators":[],"stratifications":[],
        "transitions":[{{"id":"death","source":["A"],"target":["D"],"rate":"k"}}]}},
        "structure":{{"kind":"aggregate","bin_amounts":{{"A":100.0,"D":null}}}}}}"#
    );
    serde_json::from_str(&json).expect("valid model JSON")
}

const MISSING: &str = "The model cannot run: the parameters [\"k\"] and the amounts of the bins \
                       [\"D\"] have no value. Give them a value, or calibrate them and write the \
                       calibrated values back";

#[test]
fn the_engine_of_events_refuses_a_run_that_lacks_a_value() {
    let mut engine = StochasticEvents::try_from_model(&death("StochasticEvents")).unwrap();

    assert_eq!(engine.run_at(&[1.0]).unwrap_err().to_string(), MISSING);
    assert_eq!(engine.step().unwrap_err().to_string(), MISSING);
}

#[test]
fn the_engine_of_steps_refuses_a_step_that_lacks_a_value() {
    let mut engine = StochasticSteps::try_from_model(&death("StochasticSteps")).unwrap();

    assert_eq!(engine.step().unwrap_err().to_string(), MISSING);
    assert_eq!(engine.run(2).unwrap_err().to_string(), MISSING);
}
