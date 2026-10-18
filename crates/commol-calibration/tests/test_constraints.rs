//! Integration tests for calibration constraints

use commol_calibration::CalibrationConstraint;
use commol_core::MathExpression;

#[test]
fn a_constraint_without_positions_reads_only_parameters() {
    let constraint = CalibrationConstraint::new("test".to_string(), "5.0 - k1/k2".to_string());
    assert!(constraint.is_parameter_only());
    assert!(!constraint.has_positions());
}

#[test]
fn a_constraint_with_positions_reads_the_outputs() {
    let constraint =
        CalibrationConstraint::new("at_positions".to_string(), "500.0 - B".to_string())
            .with_positions(vec![10.0, 20.0, 30.0]);

    assert!(constraint.has_positions());
    assert!(!constraint.is_parameter_only());
}

#[test]
fn test_constraint_compilation() {
    let constraint = CalibrationConstraint::new("ratio".to_string(), "5.0 - k1/k2".to_string());

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_invalid_constraint_expression() {
    let constraint = CalibrationConstraint::new(
        "invalid".to_string(),
        "5.0 - k1 / / k2".to_string(), // Invalid syntax
    );

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_err());
}

#[test]
fn test_linear_constraint_expression() {
    let constraint =
        CalibrationConstraint::new("sum_bound".to_string(), "0.5 - (k1 + k2)".to_string());

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_nonlinear_constraint_with_functions() {
    let constraint = CalibrationConstraint::new(
        "euclidean".to_string(),
        "1.0 - sqrt(k1^2 + k2^2)".to_string(),
    );

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_constraint_with_special_variables() {
    // Constraints can reference special variables like total, step, t, pi, e
    let constraint =
        CalibrationConstraint::new("total_bound".to_string(), "total - 1000.0".to_string())
            .with_positions(vec![10.0]);

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_constraint_with_mathematical_functions() {
    let expressions = vec![
        "max(k1, k2) - 0.5",
        "min(k1, k2) - 0.1",
        "abs(k1 - k2) - 0.05",
        "exp(k1) - 2.0",
        "ln(k1 + 1.0) - 0.5",
        "sin(k1 * pi) + 1.0",
    ];

    for expr_str in expressions {
        let constraint = CalibrationConstraint::new("test".to_string(), expr_str.to_string());
        let expr = MathExpression::new(constraint.expression.clone());
        assert!(
            expr.validate().is_ok(),
            "Expression '{}' should be valid",
            expr_str
        );
    }
}
