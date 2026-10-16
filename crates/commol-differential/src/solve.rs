//! Integration of a compiled model with `diffsol`.
//!
//! The right-hand side is [`CompiledModel::derivatives`]. The engine splits the
//! integration at breakpoints: the impulse positions, the values a rate
//! compares `t` with, the points of a series read at `t` and, when a rate
//! reads `step` or a pulse series, every integer. Each piece is integrated on
//! its own, so a discontinuity of the right-hand side is always the end of a
//! piece, and the solver restarts from the state after the jumps. On a piece
//! `[start, end)` the right-hand side sees `t` below `end`, so a value that
//! jumps at `end` acts only from the next piece on.

use std::cell::{Cell, RefCell};

use commol_core::{OdeMethod, OdeSolver};
use commol_model::CompiledModel;
use diffsol::{
    FaerSparseLU, FaerSparseMat, NalgebraLU, NalgebraMat, OdeBuilder, OdeEquations,
    OdeSolverMethod, OdeSolverStopReason, Vector,
};

/// Number of state components from which `bdf` uses a sparse Jacobian and a
/// sparse LU factorization instead of dense ones.
///
/// Measured by `benchmark_dense_against_sparse_bdf` (`src/tests.rs`): sparse
/// wins on a sparse pattern from about 32 components (5 times at 128, 17 at
/// 256) and loses about 3 times from 192 components on a pattern where every
/// row reads a total. At 128 the worst loss of either choice is about 5 times.
pub const SPARSE_THRESHOLD: usize = 128;

/// Relative tolerance with which two positions on the axis are the same.
const POSITION_TOLERANCE: f64 = 1e-9;

fn same_position(a: f64, b: f64) -> bool {
    (a - b).abs() <= POSITION_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

/// The right-hand side and its Jacobian-vector product, shared by the solver
/// closures. `diffsol` takes `Fn` closures, so the mutable model lives in a
/// `RefCell`.
struct RightHandSide<'m> {
    model: RefCell<&'m mut CompiledModel>,
    /// Value of `step` for the piece being integrated.
    step: Cell<f64>,
    /// Largest `t` the right-hand side sees on the piece being integrated:
    /// just below its end, so a value that jumps at the end acts only on the
    /// next piece.
    last_t: Cell<f64>,
    /// First evaluation error; the closures cannot return it.
    error: RefCell<Option<String>>,
    /// Last `(t, x, f(t, x))`, reused by the finite-difference `J·v`.
    cache: RefCell<(f64, Vec<f64>, Vec<f64>)>,
}

impl RightHandSide<'_> {
    fn eval(&self, x: &[f64], t: f64, y: &mut [f64]) {
        let t = t.min(self.last_t.get());
        let result = self
            .model
            .borrow_mut()
            .derivatives(self.step.get(), t, x, y);
        if let Err(error) = result {
            y.fill(f64::NAN);
            self.error.borrow_mut().get_or_insert(error);
        }
    }

    /// `J·v ≈ (f(x + εv) − f(x)) / ε`, with `f(x)` reused from the cache when
    /// `(t, x)` did not change.
    fn jac_mul(&self, x: &[f64], t: f64, v: &[f64], y: &mut [f64]) {
        {
            let mut cache = self.cache.borrow_mut();
            let (cached_t, cached_x, cached_f) = &mut *cache;
            if *cached_t != t || cached_x.as_slice() != x || cached_f.len() != x.len() {
                *cached_t = t;
                cached_x.clear();
                cached_x.extend_from_slice(x);
                cached_f.resize(x.len(), 0.0);
                self.eval(x, t, cached_f);
            }
        }
        let x_norm = x.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let v_norm = v.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let mut epsilon = f64::EPSILON.sqrt() * (1.0 + x_norm);
        // `diffsol` finds the sparsity pattern with NaN entries in `v`; then
        // the norm is NaN and the step must not depend on it.
        if v_norm.is_finite() && v_norm > 0.0 {
            epsilon /= v_norm;
        }
        let shifted: Vec<f64> = x
            .iter()
            .zip(v)
            .map(|(value, direction)| value + epsilon * direction)
            .collect();
        self.eval(&shifted, t, y);
        let cache = self.cache.borrow();
        for (out, base) in y.iter_mut().zip(&cache.2) {
            *out = (*out - base) / epsilon;
        }
    }

    fn take_error(&self) -> Option<String> {
        self.error.borrow_mut().take()
    }
}

/// Integrate `model` from `(t0, state)` to the last of `positions`, recording
/// the state at each of them.
///
/// `state` is the state at `t0` after its impulses. `positions` are sorted,
/// unique and greater than `t0`. On return `state` holds the state at the last
/// position, after its impulses, and `rows[i]` the state at `positions[i]`.
pub(crate) fn integrate(
    model: &mut CompiledModel,
    settings: &OdeSolver,
    t0: f64,
    state: &mut Vec<f64>,
    positions: &[f64],
    rows: &mut [Vec<f64>],
) -> Result<(), String> {
    let sparse = state.len() >= SPARSE_THRESHOLD;
    integrate_with(model, settings, sparse, t0, state, positions, rows)
}

/// [`integrate`] with the matrix type of `bdf` chosen by `sparse`.
pub(crate) fn integrate_with(
    model: &mut CompiledModel,
    settings: &OdeSolver,
    sparse: bool,
    t0: f64,
    state: &mut Vec<f64>,
    positions: &[f64],
    rows: &mut [Vec<f64>],
) -> Result<(), String> {
    let Some(&end) = positions.last() else {
        return Ok(());
    };
    let breakpoints = model.breakpoints(t0, end)?;
    let len = state.len();
    let rhs = RightHandSide {
        model: RefCell::new(model),
        step: Cell::new(t0.floor()),
        last_t: Cell::new(f64::INFINITY),
        error: RefCell::new(None),
        cache: RefCell::new((f64::NAN, Vec::new(), Vec::new())),
    };
    let plan = Plan {
        t0,
        breakpoints: &breakpoints,
        positions,
    };
    let initial = state.clone();
    macro_rules! problem {
        ($matrix:ty) => {
            OdeBuilder::<$matrix>::new()
                .t0(t0)
                .rtol(settings.rtol)
                .atol([settings.atol])
                .rhs_implicit(
                    |x: &[f64], _p: &[f64], t: f64, y: &mut [f64]| rhs.eval(x, t, y),
                    |x: &[f64], _p: &[f64], t: f64, v: &[f64], y: &mut [f64]| {
                        rhs.jac_mul(x, t, v, y)
                    },
                )
                .init(
                    |_p: &[f64], _t: f64, y: &mut [f64]| y.copy_from_slice(&initial),
                    len,
                )
                .build()
                .map_err(|error| format!("Cannot build the ODE problem: {error}"))?
        };
    }
    let solver_error =
        |error: diffsol::DiffsolError| format!("Cannot start the ODE solver: {error}");
    let result = match settings.method {
        OdeMethod::Tsit45 => {
            let problem = problem!(NalgebraMat<f64>);
            drive(
                problem.tsit45().map_err(solver_error)?,
                &rhs,
                &plan,
                state,
                rows,
            )
        }
        OdeMethod::Bdf if !sparse => {
            let problem = problem!(NalgebraMat<f64>);
            let solver = problem.bdf::<NalgebraLU<f64>>().map_err(solver_error)?;
            drive(solver, &rhs, &plan, state, rows)
        }
        OdeMethod::Bdf => {
            let problem = problem!(FaerSparseMat<f64>);
            let solver = problem.bdf::<FaerSparseLU<f64>>().map_err(solver_error)?;
            drive(solver, &rhs, &plan, state, rows)
        }
    };
    match rhs.take_error() {
        Some(error) => Err(error),
        None => result,
    }
}

/// Where the pieces of an integration end, and where it records.
struct Plan<'p> {
    t0: f64,
    /// Ends of the pieces, ascending; the last one is the last position.
    breakpoints: &'p [f64],
    positions: &'p [f64],
}

/// Integrate every piece of `plan` with `solver`, restarting it at the start
/// of each piece from the state after the jumps.
fn drive<'a, Eqn, S>(
    mut solver: S,
    rhs: &RightHandSide<'_>,
    plan: &Plan<'_>,
    state: &mut Vec<f64>,
    rows: &mut [Vec<f64>],
) -> Result<(), String>
where
    Eqn: OdeEquations<T = f64> + 'a,
    S: OdeSolverMethod<'a, Eqn>,
{
    let positions = plan.positions;
    let ctx = solver.state().y.context().clone();
    let mut y = state.clone();
    let mut dy = vec![0.0; y.len()];
    let mut start = plan.t0;
    let mut next = 0;

    for &end in plan.breakpoints {
        // `step` is constant on the piece: the step that holds its start.
        rhs.step.set(start.floor());
        rhs.last_t.set(end.next_down());
        // A cached `f(t, x)` of the previous piece saw another `t` at its end.
        rhs.cache.borrow_mut().0 = f64::NAN;
        rhs.eval(&y, start, &mut dy);
        if let Some(error) = rhs.take_error() {
            return Err(error);
        }
        {
            let restart = solver.state_mut();
            restart.y.copy_from(&Eqn::V::from_slice(&y, ctx.clone()));
            restart.dy.copy_from(&Eqn::V::from_slice(&dy, ctx.clone()));
            *restart.t = start;
        }
        solver
            .set_stop_time(end)
            .map_err(|error| format!("Cannot stop the ODE solver at {end}: {error}"))?;
        loop {
            let reason = solver.step();
            if let Some(error) = rhs.take_error() {
                return Err(error);
            }
            let reason = reason.map_err(|error| {
                format!(
                    "The ODE solver failed after t = {}: {error}",
                    solver.state().t
                )
            })?;
            let reached = solver.state().t;
            while next < positions.len()
                && positions[next] <= reached
                && !same_position(positions[next], end)
            {
                let row = solver.interpolate(positions[next]).map_err(|error| {
                    format!("Cannot interpolate at {}: {error}", positions[next])
                })?;
                rows[next] = row.clone_as_vec();
                next += 1;
            }
            if matches!(reason, OdeSolverStopReason::TstopReached) {
                break;
            }
        }
        y = solver.state().y.clone_as_vec();
        rhs.model.borrow_mut().apply_impulses(end, &mut y)?;
        while next < positions.len() && same_position(positions[next], end) {
            rows[next].clone_from(&y);
            next += 1;
        }
        start = end;
    }
    *state = y;
    Ok(())
}
