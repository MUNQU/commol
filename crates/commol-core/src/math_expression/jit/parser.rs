//! The parser of formulas: the text of a formula into its tree ([`Expr`]).
//!
//! The grammar, from the level that binds least:
//!
//! ```text
//! formula    = chain(Or)
//! chain(L)   = operand(L) (operator of L, operand(L))*
//! operand(L) = chain(the next level) | unary, after Product
//! unary      = ("-" | "!") unary | power
//! power      = primary ("^" | "**") unary | primary
//! primary    = number | name | name "(" arguments ")" | "(" formula ")"
//! ```
//!
//! The levels are `||`, `&&`, the comparisons (`< > <= >= == !=`), `+ -` and
//! `* / %` ([`Level`]). A chain of one level is one node, read from the
//! left: `a - b + c` is `(a - b) + c`. A power binds more tightly than a
//! sign, and a chain of powers groups from the right, as in Python:
//! `-2 ** 2` is `-(2 ** 2)` and `2 ** 3 ** 2` is `2 ** (3 ** 2)`.
//!
//! The parser counts how deep the formula nests: each parenthesis, call,
//! sign and power is one level. A formula that nests deeper than
//! [`MAX_NESTING`] is rejected before the parser reads the level, so
//! neither the parser nor a walk of the tree runs out of stack. The check
//! of the language rejects a tree of more than [`MAX_NODES`] nodes.

use super::ast::{BinaryOperator, Expr, Level, UnaryOperator};
use crate::math_expression::error::MathExpressionError;
use crate::math_expression::language::MATRIX_SUM;

/// The deepest a formula may nest: parentheses, calls, signs and powers,
/// one level each. A walk of a tree at this depth fits the stack of a
/// worker thread of the parallel work (2 MiB) in a debug build.
pub const MAX_NESTING: usize = 100;

/// The most nodes the tree of a formula may have: numbers, variables,
/// operators and calls, `if` among them. The check of the language
/// ([`MathExpression::check`](crate::MathExpression::check)) counts them
/// before the JIT compiles the formula, so a formula too long to compile is
/// rejected fast, and far below the size the JIT rejects.
pub const MAX_NODES: usize = 100_000;

/// A token of a formula.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Token<'a> {
    Number(f64),
    Name(&'a str),
    Operator(BinaryOperator),
    Minus,
    Not,
    Power,
    Open,
    Close,
    Comma,
    End,
}

impl Token<'_> {
    /// The operator of a chain the token is: `-` between two operands is
    /// [`BinaryOperator::Sub`].
    fn operator(self) -> Option<BinaryOperator> {
        match self {
            Token::Operator(op) => Some(op),
            Token::Minus => Some(BinaryOperator::Sub),
            _ => None,
        }
    }
}

impl std::fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Number(value) => write!(f, "'{value}'"),
            Token::Name(name) => write!(f, "'{name}'"),
            Token::Not => write!(f, "'!'"),
            // coverage: invariant, the parser reads an operator, '-' and '^'
            // wherever they stand, or names the operands they need
            Token::Operator(_) | Token::Minus | Token::Power => {
                unreachable!("an operator is never unexpected")
            }
            Token::Open => write!(f, "'('"),
            Token::Close => write!(f, "')'"),
            Token::Comma => write!(f, "','"),
            Token::End => write!(f, "the end"),
        }
    }
}

/// Parse a formula into its tree.
///
/// # Errors
///
/// Returns [`MathExpressionError::InvalidExpression`] with the reason the
/// formula does not parse, or nests deeper than [`MAX_NESTING`].
pub fn parse_expression(formula: &str) -> Result<Expr, MathExpressionError> {
    tree(formula).map_err(MathExpressionError::InvalidExpression)
}

/// The tree of a formula.
///
/// # Errors
///
/// Returns the reason when the formula does not parse, holds nothing,
/// gives `if` other than 3 arguments or nests deeper than [`MAX_NESTING`].
pub(crate) fn tree(formula: &str) -> Result<Expr, String> {
    let mut parser = Parser {
        tokens: tokens(formula)?,
        next: 0,
        nesting: 0,
    };
    parser.formula()
}

/// The tokens of `formula`, each with the position of its first character
/// (from 1), and [`Token::End`] last.
fn tokens(formula: &str) -> Result<Vec<(Token<'_>, usize)>, String> {
    let chars: Vec<(usize, char)> = formula.char_indices().collect();
    let char_at = |index: usize| chars.get(index).map(|&(_, c)| c);
    let byte_at = |index: usize| chars.get(index).map_or(formula.len(), |&(byte, _)| byte);
    let mut tokens = Vec::new();
    let mut index = 0;
    while let Some(c) = char_at(index) {
        let position = index + 1;
        let (token, width) = match (c, char_at(index + 1)) {
            (c, _) if c.is_whitespace() => {
                index += 1;
                continue;
            }
            ('*', Some('*')) => (Token::Power, 2),
            ('&', Some('&')) => (Token::Operator(BinaryOperator::And), 2),
            ('|', Some('|')) => (Token::Operator(BinaryOperator::Or), 2),
            ('<', Some('=')) => (Token::Operator(BinaryOperator::Le), 2),
            ('>', Some('=')) => (Token::Operator(BinaryOperator::Ge), 2),
            ('=', Some('=')) => (Token::Operator(BinaryOperator::Eq), 2),
            ('!', Some('=')) => (Token::Operator(BinaryOperator::Ne), 2),
            ('+', _) => (Token::Operator(BinaryOperator::Add), 1),
            ('-', _) => (Token::Minus, 1),
            ('*', _) => (Token::Operator(BinaryOperator::Mul), 1),
            ('/', _) => (Token::Operator(BinaryOperator::Div), 1),
            ('%', _) => (Token::Operator(BinaryOperator::Mod), 1),
            ('^', _) => (Token::Power, 1),
            ('<', _) => (Token::Operator(BinaryOperator::Lt), 1),
            ('>', _) => (Token::Operator(BinaryOperator::Gt), 1),
            ('!', _) => (Token::Not, 1),
            ('(', _) => (Token::Open, 1),
            (')', _) => (Token::Close, 1),
            (',', _) => (Token::Comma, 1),
            (c, _) if c.is_ascii_digit() || c == '.' => {
                // A number runs on to the next operator, so `2x` and `1..2`
                // are one word that is not a number. A decimal exponent takes
                // its sign: `1e-5`.
                let word = |mut end: usize| {
                    while char_at(end).is_some_and(|c| c.is_alphanumeric() || c == '.' || c == '_')
                    {
                        end += 1;
                    }
                    end
                };
                let mut end = word(index);
                if radix(&formula[byte_at(index)..byte_at(end)]).is_none()
                    && matches!(char_at(end - 1), Some('e' | 'E'))
                    && matches!(char_at(end), Some('+' | '-'))
                    && char_at(end + 1).is_some_and(|c| c.is_ascii_digit())
                {
                    end = word(end + 1);
                }
                let text = &formula[byte_at(index)..byte_at(end)];
                let value = number(text).ok_or_else(|| {
                    format!("it does not parse: '{text}' at character {position} is not a number")
                })?;
                (Token::Number(value), end - index)
            }
            (c, _) if c.is_alphabetic() || c == '_' => {
                let mut end = index + 1;
                while char_at(end).is_some_and(|c| c.is_alphanumeric() || c == '_') {
                    end += 1;
                }
                (
                    Token::Name(&formula[byte_at(index)..byte_at(end)]),
                    end - index,
                )
            }
            (c, _) => {
                return Err(format!(
                    "it does not parse: the character '{c}' at character {position} is not \
                     in the language"
                ));
            }
        };
        tokens.push((token, position));
        index += width;
    }
    tokens.push((Token::End, chars.len() + 1));
    Ok(tokens)
}

/// The digits and the base of an integer written in base 16 (`0xff`), 2
/// (`0b101`) or 8 (`0o17`).
fn radix(word: &str) -> Option<(&str, u32)> {
    [("0x", 16), ("0b", 2), ("0o", 8)]
        .into_iter()
        .find_map(|(prefix, base)| word.strip_prefix(prefix).map(|digits| (digits, base)))
}

/// The value of a number: a decimal, such as `2`, `0.5` or `1e-5`, or an
/// integer in base 16, 2 or 8.
fn number(word: &str) -> Option<f64> {
    match radix(word) {
        Some((digits, base)) => i64::from_str_radix(digits, base)
            .ok()
            .map(|value| value as f64),
        None => word.parse().ok(),
    }
}

/// The parser: the tokens, the next one to read and how deep the token
/// nests.
struct Parser<'a> {
    tokens: Vec<(Token<'a>, usize)>,
    next: usize,
    nesting: usize,
}

impl<'a> Parser<'a> {
    /// The next token.
    fn peek(&self) -> Token<'a> {
        self.tokens[self.next].0
    }

    /// Read the next token.
    fn advance(&mut self) -> Token<'a> {
        let token = self.peek();
        if token != Token::End {
            self.next += 1;
        }
        token
    }

    /// Why the next token cannot come where it is.
    fn unexpected(&self, expected: &str) -> String {
        let (token, position) = self.tokens[self.next];
        format!("it does not parse: expected {expected} at character {position}, got {token}")
    }

    /// The whole formula.
    fn formula(&mut self) -> Result<Expr, String> {
        if self.peek() == Token::End {
            return Err(nothing());
        }
        let tree = self.chains()?;
        match self.peek() {
            Token::End => Ok(tree),
            _ => Err(self.unexpected("an operator or the end")),
        }
    }

    /// The chains of every level, one inside the other, with their
    /// operands. The chains that are still open wait on a stack, so the
    /// levels of a formula take no step of recursion.
    fn chains(&mut self) -> Result<Expr, String> {
        let mut open: Vec<OpenChain> = Vec::new();
        let mut operand = self.unary()?;
        while let Some(op) = self.peek().operator() {
            self.advance();
            self.operand_of(op, "two operands")?;
            let level = op.level();
            // The chains of the levels that bind more tightly end with
            // `operand`.
            while let Some(chain) = open.pop_if(|chain| chain.level > level) {
                operand = chain.end(operand);
            }
            match open.last_mut() {
                Some(chain) if chain.level == level => {
                    let pending = std::mem::replace(&mut chain.pending, op);
                    chain.rest.push((pending, operand));
                }
                _ => open.push(OpenChain {
                    level,
                    first: operand,
                    rest: Vec::new(),
                    pending: op,
                }),
            }
            operand = self.unary()?;
        }
        while let Some(chain) = open.pop() {
            operand = chain.end(operand);
        }
        Ok(operand)
    }

    /// A sign or a negation, or a power.
    fn unary(&mut self) -> Result<Expr, String> {
        let op = match self.peek() {
            Token::Minus => UnaryOperator::Neg,
            Token::Not => UnaryOperator::Not,
            _ => return self.power(),
        };
        self.advance();
        self.operand_of(op, "one operand")?;
        let operand = self.nested(Self::unary)?;
        Ok(Expr::unary(op, operand))
    }

    /// A primary, raised to a power when one follows.
    fn power(&mut self) -> Result<Expr, String> {
        let base = self.primary()?;
        if self.peek() != Token::Power {
            return Ok(base);
        }
        self.advance();
        self.operand_of("^", "two operands")?;
        let exponent = self.nested(Self::unary)?;
        Ok(Expr::power(base, exponent))
    }

    /// A number, a variable, a call or a formula in parentheses.
    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Token::Number(value) => {
                self.advance();
                Ok(Expr::Constant(value))
            }
            Token::Name(name @ ("true" | "false")) => {
                Err(format!("the boolean '{name}' is not a number"))
            }
            Token::Name(name) => {
                self.advance();
                if self.peek() != Token::Open {
                    return Ok(Expr::variable(name));
                }
                self.advance();
                let args = self.nested(Self::arguments)?;
                call(name, args)
            }
            Token::Open => {
                self.advance();
                if self.peek() == Token::Close {
                    return Err(nothing());
                }
                let inside = self.nested(|parser| parser.chains())?;
                self.close("an operator or ')'")?;
                Ok(inside)
            }
            Token::Operator(op) => Err(needs(op, "two operands")),
            Token::Power => Err(needs("^", "two operands")),
            _ => Err(self.unexpected("a number, a variable or '('")),
        }
    }

    /// The arguments of a call, after its `(`, and the `)` that closes them.
    fn arguments(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();
        if self.peek() != Token::Close {
            args.push(self.chains()?);
            while self.peek() == Token::Comma {
                self.advance();
                args.push(self.chains()?);
            }
        }
        self.close("an operator, ',' or ')'")?;
        Ok(args)
    }

    /// Whether an operand of `op` comes next, or the reason the formula
    /// does not hold the operands `op` needs.
    fn operand_of(&self, op: impl std::fmt::Display, operands: &str) -> Result<(), String> {
        match self.peek() {
            Token::Number(_) | Token::Name(_) | Token::Open | Token::Minus | Token::Not => Ok(()),
            _ => Err(needs(op, operands)),
        }
    }

    /// Read the `)` that closes a parenthesis or a call, where `expected`
    /// can come.
    fn close(&mut self, expected: &str) -> Result<(), String> {
        if self.peek() != Token::Close {
            return Err(self.unexpected(expected));
        }
        self.advance();
        Ok(())
    }

    /// `read` one level deeper, or the reason the formula nests too deep.
    fn nested<T>(
        &mut self,
        read: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.nesting == MAX_NESTING {
            return Err(format!(
                "it nests deeper than {MAX_NESTING} levels of parentheses, calls, signs and \
                 powers"
            ));
        }
        self.nesting += 1;
        let read = read(self);
        self.nesting -= 1;
        read
    }
}

/// A chain the parser reads: its operands so far, and the operator that
/// waits on the next one.
struct OpenChain {
    level: Level,
    first: Expr,
    rest: Vec<(BinaryOperator, Expr)>,
    pending: BinaryOperator,
}

impl OpenChain {
    /// The chain, with `last` as the operand of its pending operator.
    fn end(mut self, last: Expr) -> Expr {
        self.rest.push((self.pending, last));
        Expr::chain(self.first, self.rest)
    }
}

/// The reason an operator does not have the operands it needs.
fn needs(op: impl std::fmt::Display, operands: &str) -> String {
    format!("the operator '{op}' needs {operands}")
}

/// The reason a formula or a pair of parentheses holds nothing.
fn nothing() -> String {
    "the formula, or a pair of parentheses in it, holds nothing".to_string()
}

/// The call of `name` with `args`: `if` with three arguments is a
/// conditional, and `matrix_sum` with the id of a matrix and an expression
/// is a matrix sum.
fn call(name: &str, args: Vec<Expr>) -> Result<Expr, String> {
    if name == MATRIX_SUM {
        return match <[Expr; 2]>::try_from(args) {
            Ok([Expr::Variable(matrix), expression]) => Ok(Expr::matrix_sum(matrix, expression)),
            _ => Err(format!(
                "{MATRIX_SUM} takes the id of a matrix and an expression, as in \
                 {MATRIX_SUM}(M, B / total)"
            )),
        };
    }
    if name != "if" {
        return Ok(Expr::call(name, args));
    }
    match <[Expr; 3]>::try_from(args) {
        Ok([condition, true_expr, false_expr]) => {
            Ok(Expr::if_then_else(condition, true_expr, false_expr))
        }
        Err(args) => Err(format!(
            "if requires exactly 3 arguments, got {}",
            args.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use BinaryOperator::{Add, Div, Eq, Gt, Le, Mul, Sub};

    fn x() -> Expr {
        Expr::variable("x")
    }

    fn number(value: f64) -> Expr {
        Expr::Constant(value)
    }

    fn parse(formula: &str) -> Expr {
        parse_expression(formula).unwrap()
    }

    fn reason(formula: &str) -> String {
        parse_expression(formula).unwrap_err().to_string()
    }

    #[test]
    fn a_chain_of_one_level_is_one_node_read_from_the_left() {
        assert_eq!(
            parse("x + 1 - x + 2"),
            Expr::chain(
                x(),
                vec![(Add, number(1.0)), (Sub, x()), (Add, number(2.0))]
            )
        );
        assert_eq!(
            parse("x * 2 / x + 1"),
            Expr::chain(
                Expr::chain(x(), vec![(Mul, number(2.0)), (Div, x())]),
                vec![(Add, number(1.0))]
            )
        );
        assert_eq!(
            parse("(x + 1) + 2"),
            Expr::binary(Add, Expr::binary(Add, x(), number(1.0)), number(2.0))
        );
    }

    #[test]
    fn each_level_binds_more_tightly_than_the_one_before() {
        assert_eq!(
            parse("x + 1 > 2 * x"),
            Expr::binary(
                Gt,
                Expr::binary(Add, x(), number(1.0)),
                Expr::binary(Mul, number(2.0), x())
            )
        );
        assert_eq!(
            parse("x || x && x == 1"),
            Expr::binary(
                BinaryOperator::Or,
                x(),
                Expr::binary(BinaryOperator::And, x(), Expr::binary(Eq, x(), number(1.0)))
            )
        );
        assert_eq!(
            parse("-x * 2"),
            Expr::binary(Mul, Expr::unary(UnaryOperator::Neg, x()), number(2.0))
        );
        assert_eq!(
            parse("!x <= 1"),
            Expr::binary(Le, Expr::unary(UnaryOperator::Not, x()), number(1.0))
        );
    }

    #[test]
    fn a_power_binds_more_tightly_than_a_sign_and_groups_from_the_right() {
        let neg = |operand| Expr::unary(UnaryOperator::Neg, operand);
        assert_eq!(parse("-x ^ 2"), neg(Expr::power(x(), number(2.0))));
        assert_eq!(
            parse("2 ** 3 ^ x"),
            Expr::power(number(2.0), Expr::power(number(3.0), x()))
        );
        assert_eq!(
            parse("2 ^ -x ^ 2"),
            Expr::power(number(2.0), neg(Expr::power(x(), number(2.0))))
        );
        assert_eq!(
            parse("(2 ^ 3) ^ x"),
            Expr::power(Expr::power(number(2.0), number(3.0)), x())
        );
    }

    #[test]
    fn numbers_names_and_calls_are_read_whole() {
        assert_eq!(parse("1e5"), number(1e5));
        assert_eq!(
            parse("0xFF + 0b1010 + 0o777"),
            Expr::chain(
                number(255.0),
                vec![(Add, number(10.0)), (Add, number(511.0))]
            )
        );
        assert_eq!(
            parse("2.5E-1 + .5 + 3."),
            Expr::chain(number(0.25), vec![(Add, number(0.5)), (Add, number(3.0))])
        );
        assert_eq!(parse(" total_g1 "), Expr::variable("total_g1"));
        assert_eq!(parse("S_niño"), Expr::variable("S_niño"));
        assert_eq!(parse("e1"), Expr::variable("e1"));
        assert_eq!(parse("f()"), Expr::call("f", vec![]));
        assert_eq!(
            parse("max (x, -1, 2)"),
            Expr::call(
                "max",
                vec![
                    x(),
                    Expr::unary(UnaryOperator::Neg, number(1.0)),
                    number(2.0)
                ]
            )
        );
        assert_eq!(
            parse("if(x > 1, x, 0)"),
            Expr::if_then_else(Expr::binary(Gt, x(), number(1.0)), x(), number(0.0))
        );
    }

    #[test]
    fn a_parsed_expression_prints_as_a_formula_that_reads_back_as_itself() {
        for formula in [
            "x + 1 - x",
            "x * 2 / x % 3",
            "x ^ 2 ^ x",
            "x < 1",
            "x != 1",
            "x && y || x",
            "-x",
            "!x",
            "if(x > 1, 1, 2)",
            "max(x, -y, 2) <= 1",
            "1e-7 * 1e300",
            "(x + 1) + 2",
            "x - (y - z)",
            "x * (y / z)",
            "(-x) ^ 2",
            "--x",
            "-(x + 1)",
            "(x ^ 2) ^ 3",
            "2 ^ -x",
            "(x < 1) < 2",
            "k * matrix_sum(M, x / total) ^ 2",
        ] {
            let tree = parse(formula);
            assert_eq!(parse(&tree.to_string()), tree, "{formula}");
        }
        for (formula, printed) in [
            ("max(x, (-y)) <= 1", "max(x, -y) <= 1"),
            ("((x + 1)) + 2 * x", "(x + 1) + 2 * x"),
            ("(x * 2) + (-x) ^ (2)", "x * 2 + (-x) ^ 2"),
        ] {
            assert_eq!(parse(formula).to_string(), printed, "{formula}");
        }
    }

    #[test]
    fn a_formula_that_does_not_parse_says_where() {
        let expected = |what: &str, at: usize, got: &str| {
            format!("it does not parse: expected {what} at character {at}, got {got}")
        };
        let outside = |c: char, at: usize| {
            format!(
                "it does not parse: the character '{c}' at character {at} is not in the language"
            )
        };
        for (formula, reason_of) in [
            ("", nothing()),
            ("  ", nothing()),
            ("k1 * ()", nothing()),
            ("(k1 + 2", expected("an operator or ')'", 8, "the end")),
            ("k1 + 2)", expected("an operator or the end", 7, "')'")),
            ("k1 +", needs("+", "two operands")),
            ("k1 * * 2", needs("*", "two operands")),
            ("/ k1", needs("/", "two operands")),
            ("2 ^", needs("^", "two operands")),
            ("** 2", needs("^", "two operands")),
            ("x * -", needs("-", "one operand")),
            ("(!)", needs("!", "one operand")),
            ("f(, x)", expected("a number, a variable or '('", 3, "','")),
            ("f(x y)", expected("an operator, ',' or ')'", 5, "'y'")),
            ("x 2", expected("an operator or the end", 3, "'2'")),
            ("x !y", expected("an operator or the end", 3, "'!'")),
            ("2 (x)", expected("an operator or the end", 3, "'('")),
            ("x)", expected("an operator or the end", 2, "')'")),
            ("f(x,)", expected("a number, a variable or '('", 5, "')'")),
            ("1, 2", expected("an operator or the end", 2, "','")),
            ("a = 1", outside('=', 3)),
            ("\"a\"", outside('"', 1)),
            ("x & y", outside('&', 3)),
            ("x | y", outside('|', 3)),
            ("x; y", outside(';', 2)),
            (
                "1..2",
                "it does not parse: '1..2' at character 1 is not a number".to_string(),
            ),
            (
                "k * 2x",
                "it does not parse: '2x' at character 5 is not a number".to_string(),
            ),
            (
                "1e",
                "it does not parse: '1e' at character 1 is not a number".to_string(),
            ),
            (
                "0xFG",
                "it does not parse: '0xFG' at character 1 is not a number".to_string(),
            ),
            ("k * true", "the boolean 'true' is not a number".to_string()),
            (
                "if(false, 1, 2)",
                "the boolean 'false' is not a number".to_string(),
            ),
            (
                "if(1, 2)",
                "if requires exactly 3 arguments, got 2".to_string(),
            ),
        ] {
            assert_eq!(
                reason(formula),
                format!("Invalid expression: {reason_of}"),
                "{formula}"
            );
        }
    }

    #[test]
    fn the_limit_counts_parentheses_calls_signs_and_powers() {
        let levels = MAX_NESTING / 4;
        let at_limit = format!("{}x{}", "(-sin(x ^ ".repeat(levels), "))".repeat(levels));
        assert!(parse_expression(&at_limit).is_ok());
        assert_eq!(
            reason(&format!("-{at_limit}")),
            format!(
                "Invalid expression: it nests deeper than {MAX_NESTING} levels of parentheses, \
                 calls, signs and powers"
            )
        );
    }
}
