//! Compiled expression patterns for fast evaluation
//!
//! This module provides optimized evaluation paths for common expression patterns,
//! avoiding the overhead of AST traversal and dynamic dispatch.

use super::context::MathExpressionContext;
use super::error::MathExpressionError;

/// Compiled expression for fast evaluation without evalexpr overhead
#[derive(Debug, Clone)]
pub enum CompiledExpression {
    // ===== Basic Single Operations =====
    /// Single variable: "beta"
    SingleVar(String),

    /// Constant value: "3.14"
    Constant(f64),

    // ===== Two-Variable Operations =====
    /// Product of two variables: "beta * gamma"
    Product2(String, String),

    /// Product of variable and constant: "beta * 2.0" or "2.0 * beta"
    ProductVarConst(String, f64),

    /// Division: "A / B"
    Division(String, String),

    /// Sum of two variables: "A + B"
    Sum2(String, String),

    /// Difference of two variables: "A - B"
    Diff2(String, String),

    // ===== Three-Variable Operations =====
    /// Product of 3 variables: "beta * S * I"
    Product3(String, String, String),

    /// Product of 3 variables divided by 4th: "beta * S * I / N"
    Product3Div(String, String, String, String),

    /// Sum of three variables: "A + B + C"
    Sum3(String, String, String),

    // ===== Four-Variable Operations =====
    /// Product of 4 variables: "A * B * C * D"
    Product4(String, String, String, String),

    /// Product of 4 variables divided by 5th: "A * B * C * D / E"
    Product4Div(String, String, String, String, String),

    // ===== Mixed Operations =====
    /// Product of constant and 2 variables: "2.0 * beta * S"
    ConstProduct2(f64, String, String),

    /// Product of constant and 3 variables: "2.0 * beta * S * I"
    ConstProduct3(f64, String, String, String),

    /// Product then sum: "(A * B) + C"
    ProductSum(String, String, String),

    /// Product then difference: "(A * B) - C"
    ProductDiff(String, String, String),

    /// Division then product: "(A / B) * C"
    DivisionProduct(String, String, String),

    // ===== Mathematical Functions (Common Patterns) =====
    /// Exponential of variable: "exp(A)" or "exp(-A)"
    Exp(String, f64), // var_name, multiplier (usually 1.0 or -1.0)

    /// Natural logarithm: "ln(A)"
    Ln(String),

    /// Power: "pow(A, n)" where n is constant
    PowConst(String, f64),

    /// Sine function: "sin(A)"
    Sin(String),

    /// Cosine function: "cos(A)"
    Cos(String),

    /// Exponential decay: "exp(-rate * step)" - very common in epidemic models
    ExpDecay(String, String), // rate_var, step_var

    // ===== Complex Common Patterns =====
    /// Seasonal pattern: "base * (1 + amp * sin(2 * pi * t / period))"
    /// Stores: base_var, amp_var, t_var, period_const
    Seasonal(String, String, String, f64),

    /// SIR infection rate: "beta * S * I" (same as Product3 but semantic)
    SIRInfection(String, String, String), // beta, S, I

    /// Normalized SIR infection: "beta * S * I / N"
    SIRInfectionNormalized(String, String, String, String), // beta, S, I, N

    /// Simple recovery: "gamma * I" (common in epidemiology)
    SimpleRecovery(String, String), // gamma, I

    /// Intervention sigmoid: "1 / (1 + exp(-steepness * (t - midpoint)))"
    /// Stores: steepness_var, t_var, midpoint_var
    InterventionSigmoid(String, String, String),

    // ===== Fallback =====
    /// Fallback to evalexpr for complex expressions
    Generic,
}

impl CompiledExpression {
    /// Compile an expression to a fast-path variant if possible
    pub fn compile(preprocessed: &str) -> Self {
        // Remove whitespace for easier parsing
        let expr = preprocessed.replace(" ", "");

        // Try constant
        if let Ok(val) = expr.parse::<f64>() {
            return Self::Constant(val);
        }

        // Try single variable first (most common after constants)
        if expr.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Self::SingleVar(expr);
        }

        // Try complex patterns first (they're more specific)

        // Seasonal pattern: "base*(1+amp*sin(2*pi*t/period))"
        if let Some(compiled) = Self::try_compile_seasonal(&expr) {
            return compiled;
        }

        // Intervention sigmoid: "1/(1+exp(-steepness*(t-midpoint)))"
        if let Some(compiled) = Self::try_compile_intervention_sigmoid(&expr) {
            return compiled;
        }

        // Exponential decay: "exp(-rate*step)"
        if let Some(compiled) = Self::try_compile_exp_decay(&expr) {
            return compiled;
        }

        // Simple math functions
        if let Some(compiled) = Self::try_compile_math_function(&expr) {
            return compiled;
        }

        // N-variable products and divisions (check larger patterns first)
        if let Some(compiled) = Self::try_compile_product4_div(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_product4(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_product3_div(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_product3(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_const_product(&expr) {
            return compiled;
        }

        // Mixed operations
        if let Some(compiled) = Self::try_compile_product_sum(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_product_diff(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_division_product(&expr) {
            return compiled;
        }

        // Two-variable operations
        if let Some(compiled) = Self::try_compile_product2(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_division(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_sum2(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_sum3(&expr) {
            return compiled;
        }

        if let Some(compiled) = Self::try_compile_diff2(&expr) {
            return compiled;
        }

        // Default to generic evaluation
        Self::Generic
    }

    /// Evaluate the compiled expression with the given context
    #[inline]
    pub fn evaluate(
        &self,
        context: &mut MathExpressionContext,
    ) -> Result<f64, MathExpressionError> {
        match self {
            Self::Constant(val) => Ok(*val),

            Self::SingleVar(var) => get_var(context, var),

            Self::Product2(var1, var2) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                Ok(v1 * v2)
            }

            Self::ProductVarConst(var, constant) => {
                let v = get_var(context, var)?;
                Ok(v * constant)
            }

            Self::Division(var1, var2) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                Ok(v1 / v2)
            }

            Self::Sum2(var1, var2) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                Ok(v1 + v2)
            }

            Self::Diff2(var1, var2) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                Ok(v1 - v2)
            }

            Self::Product3(var1, var2, var3) | Self::SIRInfection(var1, var2, var3) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                Ok(v1 * v2 * v3)
            }

            Self::Product3Div(var1, var2, var3, var4)
            | Self::SIRInfectionNormalized(var1, var2, var3, var4) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                let v4 = get_var(context, var4)?;
                Ok((v1 * v2 * v3) / v4)
            }

            Self::Sum3(var1, var2, var3) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                Ok(v1 + v2 + v3)
            }

            Self::Product4(var1, var2, var3, var4) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                let v4 = get_var(context, var4)?;
                Ok(v1 * v2 * v3 * v4)
            }

            Self::Product4Div(var1, var2, var3, var4, var5) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                let v4 = get_var(context, var4)?;
                let v5 = get_var(context, var5)?;
                Ok((v1 * v2 * v3 * v4) / v5)
            }

            Self::ConstProduct2(constant, var1, var2) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                Ok(constant * v1 * v2)
            }

            Self::ConstProduct3(constant, var1, var2, var3) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                Ok(constant * v1 * v2 * v3)
            }

            Self::ProductSum(var1, var2, var3) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                Ok(v1 * v2 + v3)
            }

            Self::ProductDiff(var1, var2, var3) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                Ok(v1 * v2 - v3)
            }

            Self::DivisionProduct(var1, var2, var3) => {
                let v1 = get_var(context, var1)?;
                let v2 = get_var(context, var2)?;
                let v3 = get_var(context, var3)?;
                Ok((v1 / v2) * v3)
            }

            Self::Exp(var, multiplier) => {
                let v = get_var(context, var)?;
                Ok((v * multiplier).exp())
            }

            Self::Ln(var) => {
                let v = get_var(context, var)?;
                Ok(v.ln())
            }

            Self::PowConst(var, exponent) => {
                let v = get_var(context, var)?;
                Ok(v.powf(*exponent))
            }

            Self::Sin(var) => {
                let v = get_var(context, var)?;
                Ok(v.sin())
            }

            Self::Cos(var) => {
                let v = get_var(context, var)?;
                Ok(v.cos())
            }

            Self::ExpDecay(rate_var, step_var) => {
                let rate = get_var(context, rate_var)?;
                let step = get_var(context, step_var)?;
                Ok((-rate * step).exp())
            }

            Self::Seasonal(base_var, amp_var, t_var, period) => {
                let base = get_var(context, base_var)?;
                let amp = get_var(context, amp_var)?;
                let t = get_var(context, t_var)?;
                let two_pi = 2.0 * std::f64::consts::PI;
                Ok(base * (1.0 + amp * (two_pi * t / period).sin()))
            }

            Self::SimpleRecovery(gamma, compartment) => {
                let g = get_var(context, gamma)?;
                let c = get_var(context, compartment)?;
                Ok(g * c)
            }

            Self::InterventionSigmoid(steepness_var, t_var, midpoint_var) => {
                let steepness = get_var(context, steepness_var)?;
                let t = get_var(context, t_var)?;
                let midpoint = get_var(context, midpoint_var)?;
                Ok(1.0 / (1.0 + (-steepness * (t - midpoint)).exp()))
            }

            Self::Generic => {
                // This should never be called directly - handled by MathExpression
                Err(MathExpressionError::InvalidExpression(
                    "Generic compilation should not be evaluated directly".to_string(),
                ))
            }
        }
    }

    // ===== Pattern Compilation Methods =====

    fn try_compile_seasonal(expr: &str) -> Option<Self> {
        // Pattern: "base*(1+amp*sin(2*pi*t/period))" or variations
        // This is complex, so we'll look for key components
        if !expr.contains("math::sin") || !expr.contains("pi") {
            return None;
        }

        // Simplified: Look for pattern like "A*(1+B*math::sin(2*pi*C/D))"
        // where D is a constant (period)
        // This is a heuristic - won't catch all cases
        None // TODO: Implement full parser for this pattern
    }

    fn try_compile_intervention_sigmoid(expr: &str) -> Option<Self> {
        // Pattern: "1/(1+exp(-steepness*(t-midpoint)))"
        // Look for 1/(1+exp pattern
        if !expr.starts_with("1/(1+") || !expr.contains("math::exp") {
            return None;
        }

        // This is complex - would need proper parsing
        None // TODO: Implement full parser
    }

    fn try_compile_exp_decay(expr: &str) -> Option<Self> {
        // Pattern: "exp(-rate*step)" or "exp(-rate*t)"
        if let Some(start) = expr.find("math::exp(") {
            let inner_start = start + 10; // length of "math::exp("
            if let Some(end) = expr[inner_start..].find(')') {
                let inner = &expr[inner_start..inner_start + end];

                // Check for "-var1*var2" pattern
                if inner.starts_with('-') {
                    let parts: Vec<&str> = inner[1..].split('*').collect();
                    if parts.len() == 2 {
                        let is_var1 = parts[0].chars().all(|c| c.is_alphanumeric() || c == '_');
                        let is_var2 = parts[1].chars().all(|c| c.is_alphanumeric() || c == '_');

                        if is_var1 && is_var2 {
                            return Some(Self::ExpDecay(
                                parts[0].to_string(),
                                parts[1].to_string(),
                            ));
                        }
                    }
                }
            }
        }
        None
    }

    fn try_compile_math_function(expr: &str) -> Option<Self> {
        // Try to match single-argument math functions

        // exp(var) or exp(-var)
        if let Some(compiled) = Self::try_compile_exp(expr) {
            return Some(compiled);
        }

        // ln(var)
        if expr.starts_with("math::ln(") && expr.ends_with(')') {
            let inner = &expr[9..expr.len() - 1];
            if is_identifier(inner) {
                return Some(Self::Ln(inner.to_string()));
            }
        }

        // sin(var)
        if expr.starts_with("math::sin(") && expr.ends_with(')') {
            let inner = &expr[10..expr.len() - 1];
            if is_identifier(inner) {
                return Some(Self::Sin(inner.to_string()));
            }
        }

        // cos(var)
        if expr.starts_with("math::cos(") && expr.ends_with(')') {
            let inner = &expr[10..expr.len() - 1];
            if is_identifier(inner) {
                return Some(Self::Cos(inner.to_string()));
            }
        }

        // pow(var, const)
        if expr.starts_with("math::pow(") && expr.ends_with(')') {
            let inner = &expr[10..expr.len() - 1];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() == 2 {
                let var = parts[0].trim();
                let exp_str = parts[1].trim();
                if is_identifier(var) {
                    if let Ok(exponent) = exp_str.parse::<f64>() {
                        return Some(Self::PowConst(var.to_string(), exponent));
                    }
                }
            }
        }

        None
    }

    fn try_compile_exp(expr: &str) -> Option<Self> {
        if expr.starts_with("math::exp(") && expr.ends_with(')') {
            let inner = &expr[10..expr.len() - 1];

            // Check for negative: exp(-var)
            if inner.starts_with('-') {
                let var = &inner[1..];
                if is_identifier(var) {
                    return Some(Self::Exp(var.to_string(), -1.0));
                }
            } else if is_identifier(inner) {
                return Some(Self::Exp(inner.to_string(), 1.0));
            }
        }
        None
    }

    fn try_compile_product4_div(expr: &str) -> Option<Self> {
        if let Some(div_pos) = expr.rfind('/') {
            let numerator = &expr[..div_pos];
            let denominator = &expr[div_pos + 1..];

            let parts: Vec<&str> = numerator.split('*').collect();
            if parts.len() == 4
                && parts.iter().all(|p| is_identifier(p))
                && is_identifier(denominator)
            {
                return Some(Self::Product4Div(
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                    parts[3].to_string(),
                    denominator.to_string(),
                ));
            }
        }
        None
    }

    fn try_compile_product4(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('*').collect();
        if parts.len() == 4 && parts.iter().all(|p| is_identifier(p)) {
            return Some(Self::Product4(
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2].to_string(),
                parts[3].to_string(),
            ));
        }
        None
    }

    fn try_compile_product3_div(expr: &str) -> Option<Self> {
        if let Some(div_pos) = expr.rfind('/') {
            let numerator = &expr[..div_pos];
            let denominator = &expr[div_pos + 1..];

            let parts: Vec<&str> = numerator.split('*').collect();
            if parts.len() == 3
                && parts.iter().all(|p| is_identifier(p))
                && is_identifier(denominator)
            {
                return Some(Self::Product3Div(
                    parts[0].to_string(),
                    parts[1].to_string(),
                    parts[2].to_string(),
                    denominator.to_string(),
                ));
            }
        }
        None
    }

    fn try_compile_product3(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('*').collect();
        if parts.len() == 3 && parts.iter().all(|p| is_identifier(p)) {
            return Some(Self::Product3(
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2].to_string(),
            ));
        }
        None
    }

    fn try_compile_const_product(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('*').collect();

        // Try "const * var1 * var2"
        if parts.len() == 3 {
            if let Ok(constant) = parts[0].parse::<f64>() {
                if is_identifier(parts[1]) && is_identifier(parts[2]) {
                    return Some(Self::ConstProduct2(
                        constant,
                        parts[1].to_string(),
                        parts[2].to_string(),
                    ));
                }
            }
        }

        // Try "const * var1 * var2 * var3"
        if parts.len() == 4 {
            if let Ok(constant) = parts[0].parse::<f64>() {
                if is_identifier(parts[1]) && is_identifier(parts[2]) && is_identifier(parts[3]) {
                    return Some(Self::ConstProduct3(
                        constant,
                        parts[1].to_string(),
                        parts[2].to_string(),
                        parts[3].to_string(),
                    ));
                }
            }
        }

        None
    }

    fn try_compile_product2(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('*').collect();
        if parts.len() == 2 {
            let is_var1 = is_identifier(parts[0]);
            let is_var2 = is_identifier(parts[1]);

            // Check for variable * constant
            if is_var1 {
                if let Ok(const_val) = parts[1].parse::<f64>() {
                    return Some(Self::ProductVarConst(parts[0].to_string(), const_val));
                }
            }

            // Check for constant * variable
            if is_var2 {
                if let Ok(const_val) = parts[0].parse::<f64>() {
                    return Some(Self::ProductVarConst(parts[1].to_string(), const_val));
                }
            }

            // Both are variables
            if is_var1 && is_var2 {
                return Some(Self::Product2(parts[0].to_string(), parts[1].to_string()));
            }
        }
        None
    }

    fn try_compile_division(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('/').collect();
        if parts.len() == 2 && is_identifier(parts[0]) && is_identifier(parts[1]) {
            return Some(Self::Division(parts[0].to_string(), parts[1].to_string()));
        }
        None
    }

    fn try_compile_sum2(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('+').collect();
        if parts.len() == 2 && is_identifier(parts[0]) && is_identifier(parts[1]) {
            return Some(Self::Sum2(parts[0].to_string(), parts[1].to_string()));
        }
        None
    }

    fn try_compile_sum3(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('+').collect();
        if parts.len() == 3 && parts.iter().all(|p| is_identifier(p)) {
            return Some(Self::Sum3(
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2].to_string(),
            ));
        }
        None
    }

    fn try_compile_diff2(expr: &str) -> Option<Self> {
        let parts: Vec<&str> = expr.split('-').collect();
        if parts.len() == 2 && is_identifier(parts[0]) && is_identifier(parts[1]) {
            return Some(Self::Diff2(parts[0].to_string(), parts[1].to_string()));
        }
        None
    }

    fn try_compile_product_sum(expr: &str) -> Option<Self> {
        // Pattern: "A*B+C" (no parentheses needed due to precedence)
        let parts: Vec<&str> = expr.split('+').collect();
        if parts.len() == 2 {
            let left_parts: Vec<&str> = parts[0].split('*').collect();
            if left_parts.len() == 2
                && is_identifier(left_parts[0])
                && is_identifier(left_parts[1])
                && is_identifier(parts[1])
            {
                return Some(Self::ProductSum(
                    left_parts[0].to_string(),
                    left_parts[1].to_string(),
                    parts[1].to_string(),
                ));
            }
        }
        None
    }

    fn try_compile_product_diff(expr: &str) -> Option<Self> {
        // Pattern: "A*B-C"
        let parts: Vec<&str> = expr.split('-').collect();
        if parts.len() == 2 {
            let left_parts: Vec<&str> = parts[0].split('*').collect();
            if left_parts.len() == 2
                && is_identifier(left_parts[0])
                && is_identifier(left_parts[1])
                && is_identifier(parts[1])
            {
                return Some(Self::ProductDiff(
                    left_parts[0].to_string(),
                    left_parts[1].to_string(),
                    parts[1].to_string(),
                ));
            }
        }
        None
    }

    fn try_compile_division_product(expr: &str) -> Option<Self> {
        // Pattern: "A/B*C"
        // This is tricky because / and * have same precedence
        // We need to ensure we're parsing left-to-right

        // Find first / and first *
        if let Some(div_pos) = expr.find('/') {
            if let Some(mul_pos) = expr.find('*') {
                if div_pos < mul_pos {
                    // Pattern is A/B*C
                    let a = &expr[..div_pos];
                    let b = &expr[div_pos + 1..mul_pos];
                    let c = &expr[mul_pos + 1..];

                    if is_identifier(a) && is_identifier(b) && is_identifier(c) {
                        return Some(Self::DivisionProduct(
                            a.to_string(),
                            b.to_string(),
                            c.to_string(),
                        ));
                    }
                }
            }
        }
        None
    }
}

/// Get a variable value from context (parameters or compartments)
#[inline]
fn get_var(context: &MathExpressionContext, name: &str) -> Result<f64, MathExpressionError> {
    // Handle special variables
    if name == "N" {
        // Total population - sum of all compartments
        return Ok(context.compartments.values().sum());
    }

    if name.starts_with("N_") {
        // Stratified population (e.g., N_age_young, N_vacc_yes)
        // Sum all compartments matching the stratification pattern
        // For N_age_young, sum all compartments like S_age_young, I_age_young, R_age_young
        let strat_suffix = &name[1..]; // Remove the "N" prefix, keep "_age_young"
        let total: f64 = context
            .compartments
            .iter()
            .filter(|(comp_name, _)| comp_name.ends_with(strat_suffix))
            .map(|(_, value)| value)
            .sum();
        return Ok(total);
    }

    match name {
        "step" | "t" => {
            // Step/time variable - stored directly in context
            Ok(context.step)
        }
        "pi" => Ok(std::f64::consts::PI),
        "e" => Ok(std::f64::consts::E),
        _ => {
            // Regular parameter or compartment
            context
                .get_parameter(name)
                .or_else(|| context.compartments.get(name).copied())
                .ok_or_else(|| MathExpressionError::VariableNotFound(name.to_string()))
        }
    }
}

/// Check if a string is a valid identifier (variable name)
#[inline]
fn is_identifier(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_patterns() {
        // Test constant
        assert!(matches!(
            CompiledExpression::compile("3.14"),
            CompiledExpression::Constant(_)
        ));

        // Test single var
        assert!(matches!(
            CompiledExpression::compile("beta"),
            CompiledExpression::SingleVar(_)
        ));

        // Test product2
        assert!(matches!(
            CompiledExpression::compile("beta*gamma"),
            CompiledExpression::Product2(_, _)
        ));

        // Test product3
        assert!(matches!(
            CompiledExpression::compile("beta*S*I"),
            CompiledExpression::Product3(_, _, _)
        ));

        // Test product3div
        assert!(matches!(
            CompiledExpression::compile("beta*S*I/N"),
            CompiledExpression::Product3Div(_, _, _, _)
        ));

        // Test exp decay
        assert!(matches!(
            CompiledExpression::compile("math::exp(-rate*step)"),
            CompiledExpression::ExpDecay(_, _)
        ));
    }
}
