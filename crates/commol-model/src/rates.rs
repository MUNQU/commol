//! Evaluation of the compiled rate of a flow.
//!
//! A rate takes the fastest path it supports: the specialized fast-path
//! interpreter, or the JIT with pre-resolved slots.

use crate::types::{FAST_STACK_DEPTH, FastRateExpression, FastRateOp, TransitionFlow, VarSlot};
use commol_core::{MathExpression, MathExpressionContext, RateMathExpression};

/// The step and the time a rate is evaluated at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Clock {
    /// Value of `step`.
    pub(crate) step: f64,
    /// Value of `t`.
    pub(crate) time: f64,
}

/// Evaluate a transition flow's rate, choosing the fastest path the rate
/// supports: the specialized fast-path interpreter, or the JIT with
/// pre-resolved slots.
#[inline]
pub(crate) fn evaluate_rate(
    flow: &TransitionFlow,
    amounts: &[f64],
    ctx: &mut MathExpressionContext,
    clock: Clock,
    jit_buf: &mut [f64; 16],
) -> Result<f64, String> {
    let rate_result = match (
        &flow.fast_rate_expression,
        &flow.rate_expression,
        &flow.resolved_slots,
    ) {
        (Some(fast_expr), _, _) => eval_fast_rate(fast_expr, amounts, ctx, clock),
        (None, RateMathExpression::Formula(expr), Some(slots)) => {
            evaluate_jit_rate(expr, slots, amounts, ctx, clock, jit_buf)
        }
        // coverage: invariant, the builder rejects a formula that does not compile
        _ => unreachable!("the builder gives every rate a fast path or the slots of its JIT"),
    };

    rate_result.map_err(|error| {
        format!(
            "Failed to evaluate rate for transition from {:?} to {:?}: {}",
            flow.source_index, flow.target_index, error
        )
    })
}

/// Resolve every variable slot the JIT-compiled rate needs, then invoke the
/// compiled function. The 16-slot stack buffer covers the vast majority of
/// rate expressions; wider expressions fall back to a one-off heap buffer.
#[inline]
fn evaluate_jit_rate(
    expr: &MathExpression,
    slots: &[VarSlot],
    amounts: &[f64],
    ctx: &MathExpressionContext,
    clock: Clock,
    jit_buf: &mut [f64; 16],
) -> Result<f64, String> {
    let n = slots.len();
    if n <= jit_buf.len() {
        let values = &mut jit_buf[..n];
        for (i, slot) in slots.iter().enumerate() {
            values[i] = resolve_slot(slot, amounts, ctx, clock)?;
        }
        expr.call_jit_with_buffer(values)
            .map_err(|error| error.to_string())
    } else {
        let mut heap = vec![0.0; n];
        for (i, slot) in slots.iter().enumerate() {
            heap[i] = resolve_slot(slot, amounts, ctx, clock)?;
        }
        expr.call_jit_with_buffer(&heap)
            .map_err(|error| error.to_string())
    }
}

/// Resolve a pre-computed `VarSlot` to its current numeric value. Compartments
/// and the step alias are O(1); parameters fall back to a single HashMap probe.
#[inline]
fn resolve_slot(
    slot: &VarSlot,
    amounts: &[f64],
    ctx: &MathExpressionContext,
    clock: Clock,
) -> Result<f64, String> {
    match slot {
        VarSlot::Compartment(idx) => Ok(amounts[*idx]),
        VarSlot::Step => Ok(clock.step),
        VarSlot::Time => Ok(clock.time),
        VarSlot::Constant(value) => Ok(*value),
        VarSlot::ParameterIndex(idx) => ctx
            .get_parameter_by_index(*idx)
            .ok_or_else(|| format!("Parameter index '{}' not found", idx)),
        VarSlot::Parameter(name) => ctx
            .get_parameter(name)
            .ok_or_else(|| format!("Variable '{}' not found", name)),
    }
}

#[inline]
fn eval_fast_rate(
    expr: &FastRateExpression,
    amounts: &[f64],
    ctx: &MathExpressionContext,
    clock: Clock,
) -> Result<f64, String> {
    Ok(match expr {
        FastRateExpression::Constant(value) => *value,
        FastRateExpression::Slot(slot) => resolve_slot(slot, amounts, ctx, clock)?,
        FastRateExpression::Mul2(a, b) => {
            resolve_slot(a, amounts, ctx, clock)? * resolve_slot(b, amounts, ctx, clock)?
        }
        FastRateExpression::Mul3(a, b, c) => {
            resolve_slot(a, amounts, ctx, clock)?
                * resolve_slot(b, amounts, ctx, clock)?
                * resolve_slot(c, amounts, ctx, clock)?
        }
        FastRateExpression::OneMinusMul3 {
            subtract,
            factor,
            value,
        } => {
            (1.0 - resolve_slot(subtract, amounts, ctx, clock)?)
                * resolve_slot(factor, amounts, ctx, clock)?
                * resolve_slot(value, amounts, ctx, clock)?
        }
        FastRateExpression::MulAddDiv {
            first,
            second,
            add_left,
            add_right,
            denominator,
        } => {
            resolve_slot(first, amounts, ctx, clock)?
                * resolve_slot(second, amounts, ctx, clock)?
                * (resolve_slot(add_left, amounts, ctx, clock)?
                    + resolve_slot(add_right, amounts, ctx, clock)?)
                / resolve_slot(denominator, amounts, ctx, clock)?
        }
        FastRateExpression::Program(ops) => eval_program(ops, amounts, ctx, clock)?,
    })
}

/// Run a stack program of a rate.
#[inline]
fn eval_program(
    ops: &[FastRateOp],
    amounts: &[f64],
    ctx: &MathExpressionContext,
    clock: Clock,
) -> Result<f64, String> {
    // The builder gives a program only when it is well formed and fits the
    // stack (`FAST_STACK_DEPTH`).
    let mut stack = [0.0_f64; FAST_STACK_DEPTH];
    let mut sp = 0usize;
    for op in ops {
        let value = match op {
            FastRateOp::Constant(value) => *value,
            FastRateOp::Slot(slot) => resolve_slot(slot, amounts, ctx, clock)?,
            FastRateOp::Neg => {
                stack[sp - 1] = -stack[sp - 1];
                continue;
            }
            binary => {
                sp -= 1;
                let (lhs, rhs) = (stack[sp - 1], stack[sp]);
                stack[sp - 1] = match binary {
                    FastRateOp::Add => lhs + rhs,
                    FastRateOp::Sub => lhs - rhs,
                    FastRateOp::Mul => lhs * rhs,
                    _ => lhs / rhs,
                };
                continue;
            }
        };
        stack[sp] = value;
        sp += 1;
    }
    debug_assert_eq!(sp, 1, "a program leaves one value");
    Ok(stack[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: Clock = Clock {
        step: 0.0,
        time: 0.0,
    };
    const FIVE: Clock = Clock {
        step: 5.0,
        time: 5.5,
    };

    fn ctx_with_params(pairs: &[(&str, f64)]) -> MathExpressionContext {
        let mut ctx = MathExpressionContext::new();
        for (name, value) in pairs {
            ctx.set_parameter((*name).to_string(), *value);
        }
        ctx
    }

    fn param_slot(ctx: &MathExpressionContext, name: &str) -> VarSlot {
        VarSlot::ParameterIndex(
            ctx.parameter_index(name)
                .expect("parameter must be registered"),
        )
    }

    #[test]
    fn fast_rate_constant_returns_value() {
        let ctx = MathExpressionContext::new();
        let expr = FastRateExpression::Constant(0.42);
        let result = eval_fast_rate(&expr, &[], &ctx, ZERO).unwrap();
        assert_eq!(result, 0.42);
    }

    #[test]
    fn fast_rate_slot_resolves_compartment_step_and_time() {
        let ctx = MathExpressionContext::new();
        let amounts = vec![10.0, 20.0];

        let compartment = FastRateExpression::Slot(VarSlot::Compartment(1));
        assert_eq!(
            eval_fast_rate(&compartment, &amounts, &ctx, FIVE).unwrap(),
            20.0
        );

        let step = FastRateExpression::Slot(VarSlot::Step);
        assert_eq!(eval_fast_rate(&step, &amounts, &ctx, FIVE).unwrap(), 5.0);

        let time = FastRateExpression::Slot(VarSlot::Time);
        assert_eq!(eval_fast_rate(&time, &amounts, &ctx, FIVE).unwrap(), 5.5);
    }

    #[test]
    fn fast_rate_mul2_and_mul3_compute_products() {
        let ctx = ctx_with_params(&[("k1", 2.0), ("k2", 3.0), ("k3", 4.0)]);
        let mul2 = FastRateExpression::Mul2(param_slot(&ctx, "k1"), param_slot(&ctx, "k2"));
        assert_eq!(eval_fast_rate(&mul2, &[], &ctx, ZERO).unwrap(), 6.0);

        let mul3 = FastRateExpression::Mul3(
            param_slot(&ctx, "k1"),
            param_slot(&ctx, "k2"),
            param_slot(&ctx, "k3"),
        );
        assert_eq!(eval_fast_rate(&mul3, &[], &ctx, ZERO).unwrap(), 24.0);
    }

    #[test]
    fn fast_rate_one_minus_mul3_and_mul_add_div() {
        let ctx = ctx_with_params(&[
            ("k1", 0.25),
            ("k2", 0.5),
            ("k3", 80.0),
            ("k4", 20.0),
            ("k5", 2.0),
            ("k6", 3.0),
            ("k7", 100.0),
        ]);

        let one_minus = FastRateExpression::OneMinusMul3 {
            subtract: param_slot(&ctx, "k1"),
            factor: param_slot(&ctx, "k2"),
            value: param_slot(&ctx, "k3"),
        };
        let expected = (1.0 - 0.25) * 0.5 * 80.0;
        assert!((eval_fast_rate(&one_minus, &[], &ctx, ZERO).unwrap() - expected).abs() < 1e-12);

        let mul_add_div = FastRateExpression::MulAddDiv {
            first: param_slot(&ctx, "k2"),
            second: param_slot(&ctx, "k3"),
            add_left: param_slot(&ctx, "k5"),
            add_right: param_slot(&ctx, "k6"),
            denominator: param_slot(&ctx, "k7"),
        };
        let expected = 0.5 * 80.0 * (2.0 + 3.0) / 100.0;
        assert!((eval_fast_rate(&mul_add_div, &[], &ctx, ZERO).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn fast_rate_program_evaluates_in_postfix_order() {
        let ctx = ctx_with_params(&[("k1", 5.0), ("k2", 3.0)]);
        let program = FastRateExpression::Program(vec![
            FastRateOp::Slot(param_slot(&ctx, "k1")),
            FastRateOp::Slot(param_slot(&ctx, "k2")),
            FastRateOp::Sub,
            FastRateOp::Constant(2.0),
            FastRateOp::Mul,
        ]);
        let expected = (5.0 - 3.0) * 2.0;
        assert!((eval_fast_rate(&program, &[], &ctx, ZERO).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn fast_rate_program_neg_unary_op() {
        let ctx = ctx_with_params(&[("k1", 7.0)]);
        let program = FastRateExpression::Program(vec![
            FastRateOp::Slot(param_slot(&ctx, "k1")),
            FastRateOp::Neg,
        ]);
        assert_eq!(eval_fast_rate(&program, &[], &ctx, ZERO).unwrap(), -7.0);
    }
}
