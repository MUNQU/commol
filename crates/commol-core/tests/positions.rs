//! When two positions on the axis are the same position, and which positions
//! each engine reports at.

use commol_core::{
    EngineType, POSITION_TOLERANCE, STEP_POSITION_LIMIT, check_positions,
    check_positions_in_continuous_time, same_position,
};

#[test]
fn same_position_is_relative_to_the_larger_magnitude() {
    assert_eq!(POSITION_TOLERANCE, 1e-9);
    // Below 1 in magnitude, the tolerance is absolute.
    assert!(same_position(0.0, 0.5e-9));
    assert!(!same_position(0.0, 2e-9));
    assert!(same_position(1.0, 1.0 + 0.5e-9));
    assert!(!same_position(1.0, 1.0 + 1e-8));
    // Above 1, it grows with the larger of the two.
    assert!(same_position(1e6, 1e6 + 1e-4));
    assert!(!same_position(1e6, 1e6 + 1e-2));
    assert!(same_position(-1e6, -1e6 - 1e-4));
    assert!(same_position(1e6 + 1e-4, 1e6));
}

#[test]
fn from_the_step_limit_two_steps_can_be_the_same_position() {
    assert_eq!(POSITION_TOLERANCE * STEP_POSITION_LIMIT, 0.5);
    let below = STEP_POSITION_LIMIT - 1.0;
    assert!(!same_position(below, below + 0.5));
    assert!(same_position(
        STEP_POSITION_LIMIT,
        STEP_POSITION_LIMIT + 0.5
    ));
}

#[test]
fn only_the_engines_of_rates_per_step_report_at_integer_steps_alone() {
    for &engine in EngineType::ALL {
        let integer = matches!(
            engine,
            EngineType::DifferenceEquations | EngineType::StochasticSteps
        );
        assert_eq!(engine.reports_at_integer_steps(), integer, "{engine:?}");
        assert_eq!(engine.check_position(2.0), Ok(()), "{engine:?}");
        assert_eq!(engine.check_position(2.5).is_ok(), !integer, "{engine:?}");
        for position in [-1.0, f64::NAN, f64::INFINITY] {
            let error = engine.check_position(position).unwrap_err();
            let start = format!("The {} engine reports", engine.name());
            assert!(error.starts_with(&start), "{error}");
            assert!(error.contains(&format!("{position} is not one")), "{error}");
        }
    }
}

#[test]
fn only_an_engine_in_continuous_time_takes_a_first_position_the_same_as_the_current_one() {
    // From 1e9 two integers a step apart are the same position.
    let accept = |_: f64| Ok(());
    let error = check_positions(&[1e9 - 1.0], 1e9, accept).unwrap_err();
    assert_eq!(
        error,
        "Position 999999999 is before the current position 1000000000"
    );
    assert!(check_positions_in_continuous_time(&[3.0 - 1e-10], 3.0, accept).is_ok());
    let error = check_positions_in_continuous_time(&[3.0 - 1e-8], 3.0, accept).unwrap_err();
    assert_eq!(
        error,
        "Position 2.99999999 is before the current position 3"
    );
}
