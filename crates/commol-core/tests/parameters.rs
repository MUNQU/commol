//! The parameters of an expression context: a value set again replaces the
//! one before, also for a slot reserved before any value.

use commol_core::MathExpressionContext;

#[test]
fn a_parameter_set_again_takes_the_new_value() {
    let mut context = MathExpressionContext::new();
    context.set_parameter("k1".to_string(), 1.0);
    context.set_parameter("k1".to_string(), 2.0);
    assert_eq!(context.get_parameter("k1"), Some(2.0));
}

#[test]
fn a_reserved_parameter_has_no_value_until_it_is_set() {
    let mut context = MathExpressionContext::new();
    context.reserve_parameters(["k1".to_string()]);
    assert_eq!(context.get_parameter("k1"), None);
    context.set_parameter("k1".to_string(), 3.0);
    assert_eq!(context.get_parameter("k1"), Some(3.0));
}
