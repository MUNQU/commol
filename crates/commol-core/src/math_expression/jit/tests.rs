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
        ctx.set_parameter("beta".to_string(), 0.5);
        ctx.set_parameter("gamma".to_string(), 0.2);
        ctx.set_compartment("S".to_string(), 990.0);
        ctx.set_compartment("I".to_string(), 10.0);
        ctx.set_compartment("R".to_string(), 0.0);
        ctx.set_step(5.0);
        ctx
    }

    #[test]
    fn test_jit_constant() {
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5, got {}", result);
    }

    #[test]
    fn test_jit_addition() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta + gamma");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.7).abs() < 1e-10, "Expected 0.7, got {}", result);
    }

    #[test]
    fn test_jit_subtraction() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta - gamma");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.3).abs() < 1e-10, "Expected 0.3, got {}", result);
    }

    #[test]
    fn test_jit_multiplication() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta * gamma");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 0.1).abs() < 1e-10, "Expected 0.1, got {}", result);
    }

    #[test]
    fn test_jit_division() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta / gamma");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 2.5).abs() < 1e-10, "Expected 2.5, got {}", result);
    }

    #[test]
    fn test_jit_negation() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("-beta");
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta * gamma");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // beta=0.5, gamma=0.2, result=0.1
        assert!((result - 0.1).abs() < 1e-10, "Expected 0.1, got {}", result);
    }

    #[test]
    fn test_jit_product3() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta * S * I");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // beta=0.5, S=990, I=10, result=4950
        assert!(
            (result - 4950.0).abs() < 1e-10,
            "Expected 4950.0, got {}",
            result
        );
    }

    #[test]
    fn test_jit_product3_div() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta * S * I / N");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // beta=0.5, S=990, I=10, N=1000, result=4.95
        assert!(
            (result - 4.95).abs() < 1e-10,
            "Expected 4.95, got {}",
            result
        );
    }

    #[test]
    fn test_jit_complex_expression() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("(beta * S + gamma * I) / N");
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("step * beta");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // step=5.0, beta=0.5, result=2.5
        assert!((result - 2.5).abs() < 1e-10, "Expected 2.5, got {}", result);
    }

    #[test]
    fn test_jit_special_variable_n() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("N");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // N = S + I + R = 990 + 10 + 0 = 1000
        assert!(
            (result - 1000.0).abs() < 1e-10,
            "Expected 1000.0, got {}",
            result
        );
    }

    #[test]
    fn test_jit_special_constant_pi() {
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
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

    #[test]
    fn test_jit_comparison_lt() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta < gamma");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.5 < 0.2 is false, should return 0.0
        assert!((result - 0.0).abs() < 1e-10, "Expected 0.0, got {}", result);
    }

    #[test]
    fn test_jit_comparison_gt() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta > gamma");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.5 > 0.2 is true, should return 1.0
        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_comparison_eq() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("gamma == 0.2");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.2 == 0.2 is true, should return 1.0
        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_parentheses() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("(beta + gamma) * 10");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // (0.5 + 0.2) * 10 = 7.0
        assert!((result - 7.0).abs() < 1e-10, "Expected 7.0, got {}", result);
    }

    #[test]
    fn test_jit_operator_precedence() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("beta + gamma * 10");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0.5 + (0.2 * 10) = 0.5 + 2.0 = 2.5
        assert!((result - 2.5).abs() < 1e-10, "Expected 2.5, got {}", result);
    }

    // ===== Math Functions Tests =====

    #[test]
    fn test_jit_sin() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("sin(0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!(result.abs() < 1e-10, "Expected 0.0, got {}", result);
    }

    #[test]
    fn test_jit_cos() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("cos(0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_exp() {
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("ln(e)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_sqrt() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("sqrt(4)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 2.0).abs() < 1e-10, "Expected 2.0, got {}", result);
    }

    #[test]
    fn test_jit_pow() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("pow(2, 3)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 8.0).abs() < 1e-10, "Expected 8.0, got {}", result);
    }

    #[test]
    fn test_jit_pow_operator() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("2 ** 3");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 8.0).abs() < 1e-10, "Expected 8.0, got {}", result);
    }

    #[test]
    fn test_jit_abs() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("abs(-5)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 5.0).abs() < 1e-10, "Expected 5.0, got {}", result);
    }

    #[test]
    fn test_jit_floor() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("floor(3.7)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 3.0).abs() < 1e-10, "Expected 3.0, got {}", result);
    }

    #[test]
    fn test_jit_ceil() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("ceil(3.2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 4.0).abs() < 1e-10, "Expected 4.0, got {}", result);
    }

    #[test]
    fn test_jit_round() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("round(3.5)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 4.0).abs() < 1e-10, "Expected 4.0, got {}", result);
    }

    #[test]
    fn test_jit_min() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("min(beta, gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // min(0.5, 0.2) = 0.2
        assert!((result - 0.2).abs() < 1e-10, "Expected 0.2, got {}", result);
    }

    #[test]
    fn test_jit_max() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("max(beta, gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // max(0.5, 0.2) = 0.5
        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5, got {}", result);
    }

    #[test]
    fn test_jit_min_multi() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("min(5, 3, 7, 2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 2.0).abs() < 1e-10, "Expected 2.0, got {}", result);
    }

    #[test]
    fn test_jit_modulo() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("10 % 3");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_logical_and() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("1 && 1");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_logical_or() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("0 || 1");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_logical_not() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("!0");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        assert!((result - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_jit_conditional() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("if(beta > gamma, beta, gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // beta=0.5 > gamma=0.2, so result = beta = 0.5
        assert!((result - 0.5).abs() < 1e-10, "Expected 0.5, got {}", result);
    }

    #[test]
    fn test_jit_complex_math_expression() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("sqrt(pow(beta, 2) + pow(gamma, 2))");
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

    // ===== Tests for Exotic Math Functions =====

    #[test]
    fn test_jit_atan2() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("atan2(gamma, beta)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        let expected = 0.2_f64.atan2(0.5);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jit_hypot() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("hypot(beta, gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        let expected = 0.5_f64.hypot(0.2);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jit_copysign() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("copysign(beta, -gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        let expected = 0.5_f64.copysign(-0.2);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jit_fma() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("fma(beta, gamma, 0.1)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // fma(0.5, 0.2, 0.1) = 0.5 * 0.2 + 0.1 = 0.1 + 0.1 = 0.2
        let expected = 0.5_f64.mul_add(0.2, 0.1);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jit_trunc() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("trunc(beta + 2)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // trunc(0.5 + 2) = trunc(2.5) = 2.0
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_jit_erf() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("erf(beta)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Just verify it returns a reasonable value (erf is monotonic, erf(0.5) ≈ 0.5205)
        assert!(result > 0.5 && result < 0.53);
    }

    #[test]
    fn test_jit_fdim() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("fdim(beta, gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // fdim(0.5, 0.2) = max(0.5 - 0.2, 0) = 0.3
        assert!((result - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_jit_fmax_fmin() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("fmax(beta, gamma) + fmin(beta, gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // fmax(0.5, 0.2) + fmin(0.5, 0.2) = 0.5 + 0.2 = 0.7
        assert!((result - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_jit_remainder() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("remainder(beta, gamma)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // IEEE remainder: 0.5 rem 0.2 = 0.1
        let expected = 0.5_f64.rem_euclid(0.2);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_jit_division_by_zero() {
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("0 / 0");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // 0/0 should produce NaN
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_nan_propagation() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("sqrt(-1)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // sqrt of negative number should be NaN
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_nan_in_expression() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("sqrt(-1) + 5");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // NaN should propagate through operations
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_infinity_arithmetic() {
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("(1 / 0) - (1 / 0)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Infinity - Infinity = NaN
        assert!(result.is_nan());
    }

    #[test]
    fn test_jit_very_large_number() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("1e308");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Should handle very large numbers
        assert!((result - 1e308).abs() < 1e295);
    }

    #[test]
    fn test_jit_overflow_to_infinity() {
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("1e-308");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Should handle very small numbers
        assert!((result - 1e-308).abs() < 1e-320);
    }

    #[test]
    fn test_jit_underflow_to_zero() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("1e-308 / 1e308");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // Underflow should produce zero or very small number
        assert!(result.abs() < 1e-300);
    }

    #[test]
    fn test_jit_empty_expression_error() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("");
        let result = compiler.compile(&preprocessed);

        // Empty expression should fail to compile
        assert!(result.is_err());
    }

    #[test]
    fn test_jit_malformed_function_wrong_arg_count() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("sin(1, 2)");
        let result = compiler.compile(&preprocessed);

        // sin with 2 arguments should fail
        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = format!("{:?}", e);
            assert!(error_msg.contains("sin") || error_msg.contains("argument"));
        }
    }

    #[test]
    fn test_jit_malformed_function_no_args() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("sqrt()");
        let result = compiler.compile(&preprocessed);

        // sqrt with no arguments should fail
        assert!(result.is_err());
    }

    #[test]
    fn test_jit_unknown_function() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("unknown_func(1)");
        let result = compiler.compile(&preprocessed);

        // Unknown function should fail
        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = format!("{:?}", e);
            assert!(error_msg.contains("unknown") || error_msg.contains("Unknown"));
        }
    }

    #[test]
    fn test_jit_pow_with_too_many_args() {
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("pow(2, 3, 4)");
        let result = compiler.compile(&preprocessed);

        // pow with 3 arguments should fail
        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = format!("{:?}", e);
            assert!(error_msg.contains("pow") || error_msg.contains("argument"));
        }
    }

    #[test]
    fn test_jit_log_of_zero() {
        let compiler = JITCompiler::new().unwrap();
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
        let compiler = JITCompiler::new().unwrap();
        let preprocessed = preprocess_formula("ln(-1)");
        let jit_fn = compiler.compile(&preprocessed).unwrap();

        let ctx = create_test_context();
        let result = jit_fn.call(&ctx).unwrap();

        // ln of negative number should be NaN
        assert!(result.is_nan());
    }
}
