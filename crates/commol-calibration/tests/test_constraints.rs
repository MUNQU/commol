//! Integration tests for calibration constraints

use commol_calibration::CalibrationConstraint;
use commol_core::MathExpression;

#[test]
fn test_constraint_creation() {
    let constraint = CalibrationConstraint::new("test".to_string(), "5.0 - beta/gamma".to_string());
    assert_eq!(constraint.id, "test");
    assert_eq!(constraint.expression, "5.0 - beta/gamma");
    assert_eq!(constraint.weight, 1.0);
    assert!(constraint.is_parameter_only());
    assert!(!constraint.is_time_dependent());
}

#[test]
fn test_constraint_with_time_steps() {
    let constraint = CalibrationConstraint::new("time_test".to_string(), "500.0 - I".to_string())
        .with_time_steps(vec![10, 20, 30]);

    assert!(constraint.is_time_dependent());
    assert!(!constraint.is_parameter_only());
    assert_eq!(constraint.time_steps, Some(vec![10, 20, 30]));
}

#[test]
fn test_constraint_with_description_and_weight() {
    let constraint = CalibrationConstraint::new("weighted".to_string(), "beta - gamma".to_string())
        .with_description("Beta must be >= gamma".to_string())
        .with_weight(5.0);

    assert_eq!(
        constraint.description,
        Some("Beta must be >= gamma".to_string())
    );
    assert_eq!(constraint.weight, 5.0);
}

#[test]
fn test_constraint_compilation() {
    let constraint = CalibrationConstraint::new("r0".to_string(), "5.0 - beta/gamma".to_string());

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_invalid_constraint_expression() {
    let constraint = CalibrationConstraint::new(
        "invalid".to_string(),
        "5.0 - beta / / gamma".to_string(), // Invalid syntax
    );

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_err());
}

#[test]
fn test_linear_constraint_expression() {
    let constraint =
        CalibrationConstraint::new("sum_bound".to_string(), "0.5 - (beta + gamma)".to_string());

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_nonlinear_constraint_with_functions() {
    let constraint = CalibrationConstraint::new(
        "euclidean".to_string(),
        "1.0 - sqrt(beta^2 + gamma^2)".to_string(),
    );

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_constraint_with_special_variables() {
    // Constraints can reference special variables like N, step, t, pi, e
    let constraint =
        CalibrationConstraint::new("time_dependent".to_string(), "N - 1000.0".to_string())
            .with_time_steps(vec![10]);

    let expr = MathExpression::new(constraint.expression.clone());
    assert!(expr.validate().is_ok());
}

#[test]
fn test_constraint_with_mathematical_functions() {
    let expressions = vec![
        "max(beta, gamma) - 0.5",
        "min(beta, gamma) - 0.1",
        "abs(beta - gamma) - 0.05",
        "exp(beta) - 2.0",
        "ln(beta + 1.0) - 0.5",
        "sin(beta * pi) + 1.0",
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

#[test]
fn test_constraint_builder_pattern() {
    let constraint =
        CalibrationConstraint::new("complex".to_string(), "5.0 - beta/gamma".to_string())
            .with_description("R0 must be less than 5".to_string())
            .with_weight(2.5)
            .with_time_steps(vec![10, 20, 30]);

    assert_eq!(constraint.id, "complex");
    assert_eq!(constraint.expression, "5.0 - beta/gamma");
    assert_eq!(
        constraint.description,
        Some("R0 must be less than 5".to_string())
    );
    assert_eq!(constraint.weight, 2.5);
    assert_eq!(constraint.time_steps, Some(vec![10, 20, 30]));
    assert!(constraint.is_time_dependent());
}
