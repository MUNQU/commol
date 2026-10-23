//! Tests for JIT compilation
//!
//! This module contains integration tests for the JIT compiler.

#[cfg(test)]
mod jit_tests {
    use crate::math_expression::MathExpressionContext;
    use crate::math_expression::jit::JITCompiler;
    use crate::math_expression::preprocessing::preprocess_formula;

    /// Helper to create a test context
    fn create_test_context() -> MathExpressionContext {
        let mut ctx = MathExpressionContext::new();
        ctx.set_parameter("k1".to_string(), 0.5);
        ctx.set_parameter("k2".to_string(), 0.2);
        ctx.set_compartment("A".to_string(), 990.0);
        ctx.set_compartment("B".to_string(), 10.0);
        ctx.set_compartment("C".to_string(), 0.0);
        ctx.set_step(5.0);
        ctx
    }

    #[test]
    fn test_jit_constant() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("3.14");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!(
            (result - std::f64::consts::PI).abs() < 1e-2,
            "Expected PI, got {}",
            result
        );
    }

    #[test]
    fn test_jit_variable() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5, got {}", result);
    }

    #[test]
    fn test_jit_addition() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 + k2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.7).abs() < 1e-10, "Expected 0.7, got {}", result);
    }

    #[test]
    fn test_jit_subtraction() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 - k2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.3).abs() < 1e-10, "Expected 0.3, got {}", result);
    }

    #[test]
    fn test_jit_multiplication() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 * k2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.1).abs() < 1e-10, "Expected 0.1, got {}", result);
    }

    #[test]
    fn test_jit_division() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 / k2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 2.5).abs() < 1e-10, "Expected 2.5, got {}", result);
    }

    #[test]
    fn test_jit_negation() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("-k1");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!(
            (result - (-0.5)).abs() < 1e-10,
            "Expected -0.5, got {}",
            result
        );
    }

    #[test]
    fn test_jit_product2() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 * k2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // k1=0.5, k2=0.2, result=0.1
        assert!((result - 0.1).abs() < 1e-10, "Expected 0.1, got {}", result);
    }

    #[test]
    fn test_jit_product3() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 * A * B");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // k1=0.5, A=990, B=10, result=4950
        assert!(
            (result - 4950.0).abs() < 1e-10,
            "Expected 4950.0, got {}",
            result
        );
    }

    #[test]
    fn test_jit_product3_div() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 * A * B / total");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // k1=0.5, A=990, B=10, total=1000, result=4.95
        assert!(
            (result - 4.95).abs() < 1e-10,
            "Expected 4.95, got {}",
            result
        );
    }

    #[test]
    fn test_jit_complex_expression() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("(k1 * A + k2 * B) / total");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // (0.5 * 990 + 0.2 * 10) / 1000 = (495 + 2) / 1000 = 0.497
        assert!(
            (result - 0.497).abs() < 1e-10,
            "Expected 0.497, got {}",
            result
        );
    }

    #[test]
    fn test_jit_special_variable_step() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("step * k1");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // step=5.0, k1=0.5, result=2.5
        assert!((result - 2.5).abs() < 1e-10, "Expected 2.5, got {}", result);
    }

    #[test]
    fn test_jit_special_variable_total() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("total");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // total = A + B + C = 990 + 10 + 0 = 1000
        assert!(
            (result - 1000.0).abs() < 1e-10,
            "Expected 1000.0, got {}",
            result
        );
    }

    #[test]
    fn test_jit_special_constant_pi() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("pi");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!(
            (result - std::f64::consts::PI).abs() < 1e-10,
            "Expected PI, got {}",
            result
        );
    }

    #[test]
    fn test_jit_special_constant_e() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("e");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!(
            (result - std::f64::consts::E).abs() < 1e-10,
            "Expected E, got {}",
            result
        );
    }

    fn evaluate(formula: &str, ctx: &MathExpressionContext) -> f64 {
        let compiler = JITCompiler::new();
        let jit_fn = compiler.compile(&preprocess_formula(formula)).unwrap();
        jit_fn.call(ctx).unwrap()
    }

    #[test]
    fn at_most_and_unequal_compare_two_values() {
        let ctx = create_test_context();

        assert_eq!(evaluate("k1 <= 0.5", &ctx), 1.0);
        assert_eq!(evaluate("k1 <= 0.4", &ctx), 0.0);
        assert_eq!(evaluate("k1 != 0.5", &ctx), 0.0);
        assert_eq!(evaluate("k1 != k2", &ctx), 1.0);
    }

    #[test]
    fn a_function_of_two_arguments_needs_two() {
        let compiler = JITCompiler::new();

        let error = compiler
            .compile(&preprocess_formula("hypot(k1)"))
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Invalid expression: Function hypot takes 2 arguments, got 1"
        );
    }

    #[test]
    fn a_function_reads_the_time_apart_from_the_step() {
        let mut ctx = create_test_context();
        ctx.set_time(5.25);

        assert_eq!(evaluate("t - step", &ctx), 0.25);
    }

    #[test]
    fn a_function_of_many_variables_reads_every_one() {
        let mut ctx = MathExpressionContext::new();
        let names: Vec<String> = (0..20).map(|i| format!("v{i}")).collect();
        for (i, name) in names.iter().enumerate() {
            ctx.set_parameter(name.clone(), i as f64);
        }

        assert_eq!(evaluate(&names.join(" + "), &ctx), 190.0);
    }

    #[test]
    fn a_buffer_has_one_value_per_variable() {
        let compiler = JITCompiler::new();
        let jit_fn = compiler.compile(&preprocess_formula("k1 * A")).unwrap();

        assert_eq!(jit_fn.call_with_buffer(&[0.5, 4.0]).unwrap(), 2.0);
        assert_eq!(
            jit_fn.call_with_buffer(&[0.5]).unwrap_err().to_string(),
            "Invalid expression: JIT buffer length mismatch: expected 2, got 1"
        );
    }

    #[test]
    fn test_jit_comparison_lt() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 < k2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.5 < 0.2 is false, should return 0.0
        assert!((result - 0.0).abs() < 1e-10, "Expected 0.0, got {}", result);
    }

    #[test]
    fn test_jit_comparison_gt() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 > k2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.5 > 0.2 is true, should return 1.0
        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_comparison_eq() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k2 == 0.2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.2 == 0.2 is true, should return 1.0
        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_parentheses() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("(k1 + k2) * 10");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // (0.5 + 0.2) * 10 = 7.0
        assert!((result - 7.0).abs() < 1e-10, "Expected 7.0, got {}", result);
    }

    #[test]
    fn test_jit_operator_precedence() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("k1 + k2 * 10");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.5 + (0.2 * 10) = 0.5 + 2.0 = 2.5
        assert!((result - 2.5).abs() < 1e-10, "Expected 2.5, got {}", result);
    }

    // ===== Math Functions Tests =====

    #[test]
    fn test_jit_sin() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("sin(0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!(result.abs() < 1e-10, "Expected 0.0, got {}", result);
    }

    #[test]
    fn test_jit_cos() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("cos(0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_exp() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("exp(1)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!(
            (result - std::f64::consts::E).abs() < 1e-10,
            "Expected e, got {}",
            result
        );
    }

    #[test]
    fn test_jit_ln() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("ln(e)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_sqrt() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("sqrt(4)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 2.0).abs() < 1e-10, "Expected 2.0, got {}", result);
    }

    #[test]
    fn test_jit_pow() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("pow(2, 3)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 8.0).abs() < 1e-10, "Expected 8.0, got {}", result);
    }

    #[test]
    fn test_jit_pow_operator() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("2 ** 3");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 8.0).abs() < 1e-10, "Expected 8.0, got {}", result);
    }

    #[test]
    fn test_jit_abs() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("abs(-5)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 5.0).abs() < 1e-10, "Expected 5.0, got {}", result);
    }

    #[test]
    fn test_jit_floor() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("floor(3.7)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 3.0).abs() < 1e-10, "Expected 3.0, got {}", result);
    }

    #[test]
    fn test_jit_ceil() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("ceil(3.2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 4.0).abs() < 1e-10, "Expected 4.0, got {}", result);
    }

    #[test]
    fn test_jit_round() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("round(3.5)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 4.0).abs() < 1e-10, "Expected 4.0, got {}", result);
    }

    #[test]
    fn test_jit_min() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("min(k1, k2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // min(0.5, 0.2) = 0.2
        assert!((result - 0.2).abs() < 1e-10, "Expected 0.2, got {}", result);
    }

    #[test]
    fn test_jit_max() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("max(k1, k2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // max(0.5, 0.2) = 0.5
        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5, got {}", result);
    }

    #[test]
    fn test_jit_min_multi() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("min(5, 3, 7, 2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 2.0).abs() < 1e-10, "Expected 2.0, got {}", result);
    }

    #[test]
    fn test_jit_modulo() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("10 % 3");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_logical_and() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("1 && 1");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_logical_or() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("0 || 1");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_logical_not() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("!0");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_conditional() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("if(k1 > k2, k1, k2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // k1=0.5 > k2=0.2, so result = k1 = 0.5
        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5, got {}", result);
    }

    #[test]
    fn test_jit_complex_math_expression() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("sqrt(pow(k1, 2) + pow(k2, 2))");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // sqrt(0.5^2 + 0.2^2) = sqrt(0.25 + 0.04) = sqrt(0.29) ≈ 0.5385
        let expected = (0.5_f64.powi(2) + 0.2_f64.powi(2)).sqrt();
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_jit_atan2() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("atan2(k2, k1)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        let expected = 0.2_f64.atan2(0.5);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jit_hypot() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("hypot(k1, k2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        let expected = 0.5_f64.hypot(0.2);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jit_division_by_zero() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("1 / 0");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Division by zero should produce infinity
        assert!(result.is_infinite());
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_jit_negative_division_by_zero() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("-1 / 0");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Negative division by zero should produce negative infinity
        assert!(result.is_infinite());
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_jit_zero_divided_by_zero() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("0 / 0");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0/0 should produce NaN
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_nan_propagation() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("sqrt(-1)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // sqrt of negative number should be NaN
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_nan_in_expression() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("sqrt(-1) + 5");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // NaN should propagate through operations
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_infinity_arithmetic() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("1 / 0 + 100");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Infinity + finite = Infinity
        assert!(result.is_infinite());
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_jit_infinity_minus_infinity() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("(1 / 0) - (1 / 0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Infinity - Infinity = NaN
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_very_large_number() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("1e308");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Should handle very large numbers
        assert!((result - 1e308).abs() < 1e295);
    }

    #[test]
    fn test_jit_overflow_to_infinity() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("1e308 * 1e308");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Overflow should produce infinity
        assert!(result.is_infinite());
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_jit_very_small_number() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("1e-308");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Should handle very small numbers
        assert!((result - 1e-308).abs() < 1e-320);
    }

    #[test]
    fn test_jit_underflow_to_zero() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("1e-308 / 1e308");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Underflow should produce zero or very small number
        assert!(result.abs() < 1e-300);
    }

    #[test]
    fn what_the_jit_does_not_compile_is_an_error_that_names_it() {
        let compiler = JITCompiler::new();
        for (formula, reason) in [
            (
                "",
                "the formula, or a pair of parentheses in it, holds nothing",
            ),
            ("sin(1, 2)", "Function sin takes 1 argument, got 2"),
            ("sqrt()", "Function sqrt takes 1 argument, got 0"),
            ("min(1)", "Function min takes 2 arguments or more, got 1"),
            ("unknown_func(1)", "Unknown function: unknown_func"),
            ("pow(2, 3, 4)", "Function pow takes 2 arguments, got 3"),
        ] {
            assert_eq!(
                compiler
                    .compile(&preprocess_formula(formula))
                    .unwrap_err()
                    .to_string(),
                format!("Invalid expression: {reason}"),
                "{formula}"
            );
        }
    }

    #[test]
    fn test_jit_log_of_zero() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("ln(0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // ln(0) should produce negative infinity
        assert!(result.is_infinite());
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_jit_log_of_negative() {
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("ln(-1)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // ln of negative number should be NaN
        assert!(result.is_nan());
    }

    // ===== Tests for floor-based modulo =====

    #[test]
    fn test_jit_floor_based_modulo_periodic() {
        // step - floor(step / 7) * 7 == 0 must be 1.0 at multiples of 7 only.
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("step - floor(step / 7) * 7 == 0");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let mut ctx = MathExpressionContext::new();

        for step in 0..15_i32 {
            ctx.set_step(step as f64);
            let result = jit_fn.call(&ctx).unwrap();
            let expected = if step % 7 == 0 { 1.0 } else { 0.0 };
            assert!(
                (result - expected).abs() < 1e-10,
                "step={}: expected {}, got {}",
                step,
                expected,
                result
            );
        }
    }

    #[test]
    fn test_jit_floor_based_modulo_if_expression() {
        // Full periodic pattern: if(step - floor(step / 7) * 7 == 0, 0.1, 0).
        // Returns 0.1 exactly at multiples of 7 and 0.0 elsewhere.
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("if(step - floor(step / 7) * 7 == 0, 0.1, 0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let mut ctx = MathExpressionContext::new();

        for step in 0..15_i32 {
            ctx.set_step(step as f64);
            let result = jit_fn.call(&ctx).unwrap();
            let expected = if step % 7 == 0 { 0.1 } else { 0.0 };
            assert!(
                (result - expected).abs() < 1e-10,
                "step={}: expected {}, got {}",
                step,
                expected,
                result
            );
        }
    }

    #[test]
    fn test_jit_floor_of_negative_value() {
        // _PeriodicAxisPattern with offset>0 substitutes `step - O`, which is
        // negative for any `step < O`. The floor of a negative quotient must
        // return the next-lower integer (e.g. floor(-3.0 / 7.0) == -1).
        let compiler = JITCompiler::new();
        let preprocessed = preprocess_formula("floor(step / 7)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let mut ctx = MathExpressionContext::new();
        let cases: [(f64, f64); 6] = [
            (-1.0, -1.0),
            (-3.0, -1.0),
            (-6.0, -1.0),
            (-7.0, -1.0),
            (-8.0, -2.0),
            (-14.0, -2.0),
        ];
        for (input, expected) in cases.iter() {
            ctx.set_step(*input);
            let result = jit_fn.call(&ctx).unwrap();
            assert!(
                (result - expected).abs() < 1e-10,
                "floor({} / 7): expected {}, got {}",
                input,
                expected,
                result
            );
        }
    }

    #[test]
    fn test_jit_periodic_with_offset_negative_branch() {
        // Periodic with offset=3 evaluated for step < 3 exercises the
        // negative-input floor branch. Verifies the offset case end-to-end
        // via the JIT.
        let compiler = JITCompiler::new();
        let preprocessed =
            preprocess_formula("if((step - 3) - floor((step - 3) / 7) * 7 == 0, 1.0, 0.0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let mut ctx = MathExpressionContext::new();
        for step in 0..20_i32 {
            ctx.set_step(step as f64);
            let result = jit_fn.call(&ctx).unwrap();
            let expected = if (step - 3).rem_euclid(7) == 0 {
                1.0
            } else {
                0.0
            };
            assert!(
                (result - expected).abs() < 1e-10,
                "step={}: expected {}, got {}",
                step,
                expected,
                result
            );
        }
    }
}
