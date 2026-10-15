//! Evaluation of the compiled rate of a flow.
//!
//! A rate takes the fastest path it supports: the specialized fast-path
//! interpreter, the JIT with pre-resolved slots, or the evalexpr fallback.

use crate::types::{FastRateExpression, FastRateOp, TransitionFlow, VarSlot};
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
/// supports: the specialized fast-path interpreter, the JIT with pre-resolved
/// slots, or the evalexpr fallback.
#[inline]
pub(crate) fn evaluate_rate(
    flow: &TransitionFlow,
    amounts: &[f64],
    ctx: &mut MathExpressionContext,
    clock: Clock,
    jit_buf: &mut [f64; 16],
) -> Result<f64, String> {
    let rate_result = if let Some(fast_expr) = &flow.fast_rate_expression {
        eval_fast_rate(fast_expr, amounts, ctx, clock)
    } else if let (RateMathExpression::Formula(expr), Some(slots)) =
        (&flow.rate_expression, &flow.resolved_slots)
    {
        evaluate_jit_rate(expr, slots, amounts, ctx, clock, jit_buf)
    } else {
        flow.rate_expression
            .evaluate(ctx)
            .map_err(|error| error.to_string())
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
    match expr {
        FastRateExpression::Constant(value) => return Ok(*value),
        FastRateExpression::Slot(slot) => {
            return resolve_slot(slot, amounts, ctx, clock);
        }
        FastRateExpression::Mul2(a, b) => {
            return Ok(
                resolve_slot(a, amounts, ctx, clock)? * resolve_slot(b, amounts, ctx, clock)?
            );
        }
        FastRateExpression::Mul3(a, b, c) => {
            return Ok(resolve_slot(a, amounts, ctx, clock)?
                * resolve_slot(b, amounts, ctx, clock)?
                * resolve_slot(c, amounts, ctx, clock)?);
        }
        FastRateExpression::OneMinusMul3 {
            subtract,
            factor,
            value,
        } => {
            return Ok((1.0 - resolve_slot(subtract, amounts, ctx, clock)?)
                * resolve_slot(factor, amounts, ctx, clock)?
                * resolve_slot(value, amounts, ctx, clock)?);
        }
        FastRateExpression::MulAddDiv {
            first,
            second,
            add_left,
            add_right,
            denominator,
        } => {
            return Ok(resolve_slot(first, amounts, ctx, clock)?
                * resolve_slot(second, amounts, ctx, clock)?
                * (resolve_slot(add_left, amounts, ctx, clock)?
                    + resolve_slot(add_right, amounts, ctx, clock)?)
                / resolve_slot(denominator, amounts, ctx, clock)?);
        }
        FastRateExpression::Program(_) => {}
    }

    let FastRateExpression::Program(ops) = expr else {
        unreachable!("specialized fast rate variants return above");
    };
    let mut stack = [0.0_f64; 32];
    let mut sp = 0usize;

    for op in ops {
        match op {
            FastRateOp::Constant(value) => push_stack(&mut stack, &mut sp, *value)?,
            FastRateOp::Slot(slot) => push_stack(
                &mut stack,
                &mut sp,
                resolve_slot(slot, amounts, ctx, clock)?,
            )?,
            FastRateOp::Add => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs + rhs)?;
            }
            FastRateOp::Sub => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs - rhs)?;
            }
            FastRateOp::Mul => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs * rhs)?;
            }
            FastRateOp::Div => {
                let rhs = pop_stack(&stack, &mut sp)?;
                let lhs = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, lhs / rhs)?;
            }
            FastRateOp::Neg => {
                let value = pop_stack(&stack, &mut sp)?;
                push_stack(&mut stack, &mut sp, -value)?;
            }
        }
    }

    if sp == 1 {
        pop_stack(&stack, &mut sp)
    } else {
        Err("fast rate expression left an invalid stack state".to_string())
    }
}

#[inline]
fn push_stack(stack: &mut [f64; 32], sp: &mut usize, value: f64) -> Result<(), String> {
    if *sp >= stack.len() {
        return Err("fast rate expression stack overflow".to_string());
    }
    stack[*sp] = value;
    *sp += 1;
    Ok(())
}

#[inline]
fn pop_stack(stack: &[f64; 32], sp: &mut usize) -> Result<f64, String> {
    if *sp == 0 {
        return Err("fast rate expression stack underflow".to_string());
    }
    *sp -= 1;
    Ok(stack[*sp])
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

    #[test]
    fn fast_rate_program_detects_invalid_stack_state() {
        let ctx = MathExpressionContext::new();
        let program =
            FastRateExpression::Program(vec![FastRateOp::Constant(1.0), FastRateOp::Constant(2.0)]);
        let error = eval_fast_rate(&program, &[], &ctx, ZERO).unwrap_err();
        assert!(error.contains("invalid stack state"));
    }

    #[test]
    fn fast_rate_program_underflow_is_reported() {
        let ctx = MathExpressionContext::new();
        let program = FastRateExpression::Program(vec![FastRateOp::Add]);
        let error = eval_fast_rate(&program, &[], &ctx, ZERO).unwrap_err();
        assert!(error.contains("underflow"));
    }

    #[test]
    fn fast_rate_program_overflow_is_reported() {
        let ctx = MathExpressionContext::new();
        let ops: Vec<FastRateOp> = (0..33).map(|_| FastRateOp::Constant(1.0)).collect();
        let program = FastRateExpression::Program(ops);
        let error = eval_fast_rate(&program, &[], &ctx, ZERO).unwrap_err();
        assert!(error.contains("overflow"));
    }
}
