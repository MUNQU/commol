//! When two positions on the axis are the same position, and which positions
//! each engine reports at.

use commol_core::{EngineType, POSITION_TOLERANCE, same_position};

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
