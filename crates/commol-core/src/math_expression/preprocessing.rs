//! Formula preprocessing utilities
//!
//! This module handles conversion of formulas from user-friendly syntax to evalexpr-compatible syntax.

use evalexpr::{Node, build_operator_tree};
use std::collections::HashSet;

/// The total: the sum of every compartment at the current position,
/// accumulators excluded.
pub const SPECIAL_VAR_TOTAL: &str = "total";
/// The prefix of a subgroup total: `total_<cat>…` is the sum of the
/// compartments with the categories `<cat>…`.
pub const SUBGROUP_TOTAL_PREFIX: &str = "total_";
/// The step that holds the current position: `floor(t)`.
pub const SPECIAL_VAR_STEP: &str = "step";
/// The current position on the axis. A difference engine reads it at each
/// step, so there it equals `step`; a differential engine reads it anywhere.
pub const SPECIAL_VAR_T: &str = "t";
/// The constant π.
pub const SPECIAL_VAR_PI: &str = "pi";
/// The constant e.
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
            && ident != SPECIAL_VAR_TOTAL
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

/// Whether an expression reads the variable `name`, special variables such as
/// `step` included.
///
/// Returns `false` if the expression is syntactically invalid.
pub fn reads_variable(preprocessed: &str, name: &str) -> bool {
    build_operator_tree(preprocessed)
        .is_ok_and(|node: Node| node.iter_variable_identifiers().any(|ident| ident == name))
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
        let vars = get_variables("beta * total + pi");
        assert_eq!(vars.len(), 1);
        assert!(vars.contains(&"beta".to_string()));
    }

    #[test]
    fn reads_variable_sees_special_variables() {
        assert!(reads_variable("k * (step + 1)", "step"));
        assert!(reads_variable("if(t >= 2, k, 0)", "t"));
        assert!(!reads_variable("k * steps", "step"));
        assert!(!reads_variable("k * (", "k"));
    }
}
