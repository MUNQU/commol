//! Formula preprocessing utilities
//!
//! This module handles conversion of formulas from user-friendly syntax to evalexpr-compatible syntax.

use evalexpr::{Node, build_operator_tree};
use std::collections::HashSet;

// Special variable names that are automatically available in all expressions
pub const SPECIAL_VAR_N: &str = "N";
pub const SPECIAL_VAR_STEP: &str = "step";
pub const SPECIAL_VAR_T: &str = "t";
pub const SPECIAL_VAR_PI: &str = "pi";
pub const SPECIAL_VAR_E: &str = "e";

/// Preprocesses a formula by adding `math::` prefix to mathematical functions that require it
/// and converting Python-style operators to evalexpr syntax.
///
/// Transformations:
/// - `sin(x)` → `math::sin(x)` (adds prefix for math functions)
/// - `**` → `^` (converts Python power operator to evalexpr syntax)
///
/// Note: Functions like `min`, `max`, `floor`, `ceil`, `round`, `if` are available
/// without the `math::` prefix in evalexpr and are not modified.
pub fn preprocess_formula(formula: &str) -> String {
    // Functions that need math:: prefix (not available as built-ins)
    // Order matters. Longer function names must come first to avoid
    // replacing substrings (e.g., process "asin" before "sin")
    // Note: "log" is not in this list - it's handled separately below
    const MATH_FUNCTIONS: &[&str] = &[
        "asinh", "acosh", "atanh", "asin", "acos", "atan2", "atan", "sinh", "cosh", "tanh", "sin",
        "cos", "tan", "log10", "log2", "ln", "cbrt", "sqrt", "hypot", "exp", "abs", "pow",
    ];

    // First, convert Python-style ** to evalexpr-style ^
    let mut result = formula.replace("**", "^");

    // Replace log( with ln( since evalexpr's math::log requires 2 arguments
    // Process log10 and log2 first (already in MATH_FUNCTIONS), then replace remaining log(
    result = result.replace("log(", "ln(");

    for func in MATH_FUNCTIONS {
        // Replace "func(" with "math::func(" but avoid replacing if already prefixed
        let pattern = format!("{}(", func);
        let replacement = format!("math::{}(", func);

        // We need to check if the match is a valid function call, not part of another identifier
        // Valid function call: preceded by whitespace, operator, or start of string
        let mut new_result = String::new();
        let mut remaining = result.as_str();

        while let Some(pos) = remaining.find(&pattern) {
            // Check if it's already prefixed with "math::"
            let prefix_start = pos.saturating_sub(6);
            let prefix = &remaining[prefix_start..pos];

            // Check if this is a valid function boundary (not part of another identifier)
            let is_valid_boundary = if pos == 0 {
                true // Start of string
            } else {
                let prev_char = remaining.chars().nth(pos - 1);
                match prev_char {
                    Some(c) => !c.is_alphanumeric() && c != '_',
                    None => true,
                }
            };

            if is_valid_boundary && !prefix.ends_with("math::") {
                new_result.push_str(&remaining[..pos]);
                new_result.push_str(&replacement);
                remaining = &remaining[pos + pattern.len()..];
            } else {
                new_result.push_str(&remaining[..pos + pattern.len()]);
                remaining = &remaining[pos + pattern.len()..];
            }
        }
        new_result.push_str(remaining);
        result = new_result;
    }

    result
}

/// Collects all variable identifiers from an AST node, excluding special variables.
pub fn get_variables_from_node(node: &Node, variables: &mut HashSet<String>) {
    for ident in node.iter_variable_identifiers() {
        // Exclude special variables that are automatically available
        if ident != SPECIAL_VAR_PI
            && ident != SPECIAL_VAR_E
            && ident != SPECIAL_VAR_N
            && ident != SPECIAL_VAR_STEP
            && ident != SPECIAL_VAR_T
        {
            variables.insert(ident.to_string());
        }
    }
}

/// Checks if an AST node is a single variable identifier.
pub fn is_single_identifier(node: &Node) -> bool {
    let mut var_count = 0;
    for _ in node.iter_variable_identifiers() {
        var_count += 1;
        if var_count > 1 {
            return false;
        }
    }

    var_count == 1 && node.children().is_empty()
}

/// Get all variable identifiers used in an expression.
///
/// Returns an empty vector if the expression is syntactically invalid.
pub fn get_variables(preprocessed: &str) -> Vec<String> {
    let mut variables = HashSet::new();
    if let Ok(node) = build_operator_tree(preprocessed) {
        get_variables_from_node(&node, &mut variables);
    }
    variables.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_formula() {
        assert_eq!(preprocess_formula("sin(x)"), "math::sin(x)");
        assert_eq!(preprocess_formula("2 ** 3"), "2 ^ 3");
        assert_eq!(preprocess_formula("log(x)"), "math::ln(x)");
        assert_eq!(preprocess_formula("log10(x)"), "math::log10(x)");
        assert_eq!(preprocess_formula("max(a, b)"), "max(a, b)"); // No prefix needed
    }

    #[test]
    fn test_get_variables() {
        let vars = get_variables("beta * S * I");
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&"beta".to_string()));
        assert!(vars.contains(&"S".to_string()));
        assert!(vars.contains(&"I".to_string()));

        // Special variables should be excluded
        let vars = get_variables("beta * N + pi");
        assert_eq!(vars.len(), 1);
        assert!(vars.contains(&"beta".to_string()));
    }
}
