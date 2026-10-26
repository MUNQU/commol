//! Error types for mathematical expression evaluation

/// The most characters of a formula that a message quotes.
pub const QUOTED_CHARACTERS: usize = 80;

/// `formula` as a message quotes it: whole when it has at most
/// [`QUOTED_CHARACTERS`] characters, else its first [`QUOTED_CHARACTERS`]
/// characters and "…", so a message never repeats a long text.
pub fn quoted(formula: &str) -> String {
    match formula.char_indices().nth(QUOTED_CHARACTERS) {
        Some((end, _)) => format!("{}…", &formula[..end]),
        None => formula.to_string(),
    }
}

/// Errors that can occur during expression evaluation
#[derive(Debug, thiserror::Error)]
pub enum MathExpressionError {
    /// The expression names a variable with no value.
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    /// The formula does not parse ([`parse_expression`](crate::math_expression::jit::parser::parse_expression)),
    /// or does not compile.
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),
    /// The formula is not in the language of formulas
    /// ([`MathExpression::check`](crate::MathExpression::check)).
    #[error("Formula '{}': {reason}", quoted(formula))]
    NotInLanguage {
        /// The formula, as it was written; the message quotes it cut
        /// ([`quoted`]).
        formula: String,
        /// The first part that is not in the language, and why.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_formula_is_quoted_whole() {
        let formula = "k".repeat(QUOTED_CHARACTERS);

        assert_eq!(quoted(&formula), formula);
    }

    #[test]
    fn a_long_formula_is_cut_after_its_first_characters() {
        let formula = format!("{}β + k", "α".repeat(QUOTED_CHARACTERS - 1));

        assert_eq!(
            quoted(&formula),
            format!("{}β…", "α".repeat(QUOTED_CHARACTERS - 1))
        );
    }

    #[test]
    fn a_message_quotes_the_formula_cut() {
        let error = MathExpressionError::NotInLanguage {
            formula: format!("{} $", "k + ".repeat(100)),
            reason: "it does not parse".to_string(),
        };

        assert_eq!(
            error.to_string(),
            format!("Formula '{}…': it does not parse", "k + ".repeat(20))
        );
    }
}
