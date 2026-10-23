//! The language of formulas: what `MathExpression::check` accepts and
//! rejects, and the value of each function of the guide.

use std::collections::BTreeSet;

use commol_core::math_expression::MAX_NESTING;
use commol_core::{FUNCTIONS, MathExpression, MathExpressionContext};

/// The message of the check of `formula`.
fn rejection(formula: &str) -> String {
    MathExpression::new(formula.to_string())
        .check()
        .expect_err(formula)
        .to_string()
}

/// The value of `formula` with `x` at 2.5, `k1` at 0.3 and the step at
/// `step`.
fn value(formula: &str, step: f64) -> f64 {
    let expression = MathExpression::new(formula.to_string());
    expression.check().expect(formula);
    let mut context = MathExpressionContext::new();
    context.set_parameter("x".to_string(), 2.5);
    context.set_parameter("k1".to_string(), 0.3);
    context.set_step(step);
    context.set_time(step);
    expression.evaluate(&mut context).expect(formula)
}

#[test]
fn a_function_outside_the_guide_is_rejected() {
    let functions = FUNCTIONS
        .iter()
        .map(|function| function.name)
        .collect::<Vec<_>>()
        .join(", ");
    for (formula, name) in [
        ("erf(x)", "erf"),
        ("fma(x, x, x)", "fma"),
        ("k1 * trunc(x)", "trunc"),
        ("fmax(x, x)", "fmax"),
        ("typeof(x)", "typeof"),
    ] {
        assert_eq!(
            rejection(formula),
            format!(
                "Formula '{formula}': '{name}' is not a function of the language, whose \
                 functions are {functions}, and if(condition, a, b)"
            )
        );
    }
}

#[test]
fn a_comparison_outside_a_condition_is_rejected() {
    let only_in_a_condition = "is only allowed in the condition of if(condition, a, b)";
    for (formula, reason) in [
        (
            "step > 1",
            format!("the comparison '>' {only_in_a_condition}"),
        ),
        (
            "k1 * (step > 30)",
            format!("the comparison '>' {only_in_a_condition}"),
        ),
        ("!x", format!("the operator '!' {only_in_a_condition}")),
        (
            "if(x && k1, 1, 0)",
            "the operator '&&' is not in the language; nest if(condition, a, b) calls instead"
                .to_string(),
        ),
        (
            "x || k1",
            "the operator '||' is not in the language; nest if(condition, a, b) calls instead"
                .to_string(),
        ),
        (
            "if(x, 1, 0)",
            "the condition of if(condition, a, b) is a comparison, such as 'step > 30', or \
             '!' of one; got 'x'"
                .to_string(),
        ),
        (
            "if(x < 1 < 2, 1, 0)",
            format!("the comparison '<' {only_in_a_condition}"),
        ),
        (
            "if(true, 1, 0)",
            "the boolean 'true' is not a number".to_string(),
        ),
    ] {
        assert_eq!(rejection(formula), format!("Formula '{formula}': {reason}"));
    }
}

#[test]
fn a_comparison_in_a_condition_is_accepted() {
    assert_eq!(value("if(!(step > 30), k1, 0)", 15.0), 0.3);
    assert_eq!(value("if(!(step > 30), k1, 0)", 31.0), 0.0);
    let window = "if(step >= 10, if(step <= 20, 0.05, 0), 0)";
    assert_eq!(value(window, 15.0), 0.05);
    assert_eq!(value(window, 25.0), 0.0);
}

#[test]
fn every_function_of_the_guide_has_its_value() {
    let x = 2.5_f64;
    let cases: [(&str, &str, f64); 28] = [
        ("sin", "sin(x)", x.sin()),
        ("cos", "cos(x)", x.cos()),
        ("tan", "tan(x)", x.tan()),
        ("asin", "asin(x / 5)", (x / 5.0).asin()),
        ("acos", "acos(x / 5)", (x / 5.0).acos()),
        ("atan", "atan(x)", x.atan()),
        ("atan2", "atan2(x, 2)", x.atan2(2.0)),
        ("sinh", "sinh(x)", x.sinh()),
        ("cosh", "cosh(x)", x.cosh()),
        ("tanh", "tanh(x)", x.tanh()),
        ("asinh", "asinh(x)", x.asinh()),
        ("acosh", "acosh(x)", x.acosh()),
        ("atanh", "atanh(x / 5)", (x / 5.0).atanh()),
        ("exp", "exp(x)", x.exp()),
        ("log", "log(x)", x.ln()),
        ("ln", "ln(x)", x.ln()),
        ("log10", "log10(x)", x.log10()),
        ("log2", "log2(x)", x.log2()),
        ("sqrt", "sqrt(x)", x.sqrt()),
        ("cbrt", "cbrt(x)", x.cbrt()),
        ("pow", "pow(x, 1.5)", x.powf(1.5)),
        ("hypot", "hypot(x, 6)", x.hypot(6.0)),
        ("min", "min(x, 1.5, 4)", 1.5),
        ("max", "max(x, 1.5, 4)", 4.0),
        ("abs", "abs(-x)", 2.5),
        ("floor", "floor(x)", 2.0),
        ("ceil", "ceil(x)", 3.0),
        ("round", "round(x) + round(-x)", 3.0 - 3.0),
    ];
    for (_, formula, expected) in cases {
        assert_eq!(value(formula, 0.0), expected, "{formula}");
    }
    assert_eq!(value("round(x)", 0.0), 3.0);
    let tested: BTreeSet<&str> = cases.iter().map(|(name, _, _)| *name).collect();
    let listed: BTreeSet<&str> = FUNCTIONS.iter().map(|function| function.name).collect();
    assert_eq!(tested, listed);
}

#[test]
fn a_call_with_the_wrong_arguments_names_the_function() {
    for (formula, reason) in [
        ("sqrt()", "'sqrt' takes 1 argument, got 0"),
        ("sin(x, 1)", "'sin' takes 1 argument, got 2"),
        ("pow(2, 3, 4)", "'pow' takes 2 arguments, got 3"),
        ("min(x)", "'min' takes 2 arguments or more, got 1"),
        ("if(x > 1, 2)", "if requires exactly 3 arguments, got 2"),
        (
            "k1 * ()",
            "the formula, or a pair of parentheses in it, holds nothing",
        ),
    ] {
        assert_eq!(rejection(formula), format!("Formula '{formula}': {reason}"));
    }
    let unknown = rejection("unknown_func(1)");
    assert!(
        unknown.starts_with(
            "Formula 'unknown_func(1)': 'unknown_func' is not a function of the language"
        ),
        "{unknown}"
    );
}

#[test]
fn the_check_counts_the_nodes_of_the_tree() {
    let nodes = |formula: &str| MathExpression::new(formula.to_string()).check().unwrap();

    assert_eq!(nodes("k1"), 1);
    assert_eq!(nodes("k1 + k2 + k3 + k4"), 7);
    assert_eq!(nodes("-sin(x)"), 3);
    assert_eq!(nodes("if(!(step > 30), k1, 0)"), 7);
}

#[test]
fn the_variables_of_a_formula_come_from_its_tree() {
    let expression = MathExpression::new("k1 * exp(-e1 * step) + 1e5 + k1 * pi".to_string());

    assert_eq!(expression.get_variables(), ["k1", "e1"]);
    assert!(expression.reads_variable("step"));
    assert!(!expression.reads_variable("t"));
}

/// The stack of a worker thread of the parallel work, the smallest stack a
/// formula is read on.
const WORKER_STACK: usize = 2 << 20;

/// `run` on a worker thread of a rayon pool, with the stack of
/// `WORKER_STACK`.
fn on_a_worker<T: Send>(run: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .stack_size(WORKER_STACK)
        .build()
        .unwrap()
        .install(run)
}

#[test]
fn a_formula_deeper_than_the_limit_is_rejected() {
    let shapes: [fn(usize) -> String; 5] = [
        |levels| format!("{}x{}", "(".repeat(levels), ")".repeat(levels)),
        |levels| format!("{}x{}", "sin(".repeat(levels), ")".repeat(levels)),
        |levels| format!("{}x", "-".repeat(levels)),
        |levels| vec!["x"; levels + 1].join(" ^ "),
        |levels| format!("{}x{}", "if(x > 1, ".repeat(levels), ", 0)".repeat(levels)),
    ];
    on_a_worker(|| {
        for shape in shapes {
            let deepest = shape(MAX_NESTING);
            let expression = MathExpression::new(deepest.clone());
            expression.check().expect(&deepest);
            let mut context = MathExpressionContext::new();
            context.set_parameter("x".to_string(), 0.5);
            assert!(expression.evaluate(&mut context).is_ok(), "{deepest}");
            assert_eq!(expression.get_variables(), ["x"]);

            let deeper = shape(MAX_NESTING + 1);
            assert_eq!(
                rejection(&deeper),
                format!(
                    "Formula '{deeper}': it nests deeper than {MAX_NESTING} levels of \
                     parentheses, calls, signs and powers"
                )
            );
        }
    });
}

#[test]
fn every_level_of_operators_at_the_limit_reads_on_a_worker_thread() {
    // Each pair of parentheses holds a chain of every level, one inside the
    // other: the deepest tree a formula at the limit has.
    let formula = format!(
        "{}x{}",
        "x || x && x < x + x * (".repeat(MAX_NESTING),
        ")".repeat(MAX_NESTING)
    );
    on_a_worker(|| {
        let expression = MathExpression::new(formula.clone());
        assert_eq!(expression.get_variables(), ["x"]);
        assert!(expression.jit_variable_names().is_some());
        assert!(
            rejection(&formula).ends_with(
                "the operator '||' is not in the language; nest if(condition, a, b) calls \
                 instead"
            ),
            "{formula}"
        );
        let tree = expression.tree().unwrap();
        assert_eq!(
            MathExpression::new(tree.to_string()).tree().unwrap(),
            tree.clone()
        );
    });
}

#[test]
fn a_chain_of_terms_is_one_level_and_adds_from_the_left() {
    let terms = 20_000;
    let sum = vec!["x"; terms].join(" + ");
    on_a_worker(|| {
        assert_eq!(
            MathExpression::new(sum.clone()).check().unwrap(),
            2 * terms - 1
        );
        assert_eq!(value(&sum, 0.0), 2.5 * terms as f64);
    });
    // 1 + 1e16 rounds to 1e16, so the order of the terms shows.
    assert_eq!(value("1 + 1e16 - 1e16", 0.0), 0.0);
    assert_eq!(value("1 + (1e16 - 1e16)", 0.0), 1.0);
    assert_eq!(value("2 / 4 * 8 % 3", 0.0), 1.0);
}

#[test]
fn a_chain_of_powers_groups_from_the_right() {
    assert_eq!(value("2 ** 3 ** 2", 0.0), 512.0);
    assert_eq!(value("2 ^ 3 ^ 2", 0.0), 512.0);
    assert_eq!(value("(2 ** 3) ** 2", 0.0), 64.0);
    assert_eq!(value("-2 ** 2", 0.0), -4.0);
    assert_eq!(value("2 ** -1", 0.0), 0.5);
    assert_eq!(value("2 ** -1 ** 2", 0.0), 0.5);
    assert_eq!(value("2 * 3 ** 4", 0.0), 162.0);
    let tree = |formula: &str| MathExpression::new(formula.to_string()).tree().unwrap();
    assert_eq!(tree("2 ** 3 ^ 2"), tree("2 ^ (3 ^ 2)"));
    assert_ne!(tree("2 ** 3 ^ 2"), tree("(2 ^ 3) ^ 2"));
    assert_eq!(tree("-x ^ 2"), tree("-(x ^ 2)"));
    assert_eq!(tree("(2 ^ 3) ^ 2").to_string(), "(2 ^ 3) ^ 2");
    assert_eq!(tree("2 ^ (3 ^ 2)").to_string(), "2 ^ 3 ^ 2");
}
