//! The profile tracer: one quantity held at values that step away from the
//! fit in both directions, the others fitted again at each value.
//!
//! The step is the adaptive step of Raue et al. (2009): it doubles while the
//! profile rises by less than half the target rise `q · T` (`q = 0.1`, `T`
//! the threshold) and halves when a step rises by more than twice it. Each
//! point starts from a secant prediction of every parameter, extrapolated
//! from the two points before it (from the point before at the first step),
//! as the profile likelihood of Data2Dynamics does. The tracer knows no
//! optimizer and no threshold rule: a subproblem
//! ([`commol_calibration::Subproblem`]) runs the problem's optimizer, and the
//! caller gives the threshold.

use commol_calibration::{
    CalibrationProblem, ConstraintSettings, ObservedDataPoint, OptimizationConfig,
    OptimizationObjective, Subproblem, SubproblemResult, optimize_subproblem, predict,
};
use commol_core::SimulationEngine;
use commol_random::replicate_seed;
use rayon::prelude::*;

/// The share of the threshold a step aims to rise by.
const RISE_FRACTION: f64 = 0.1;
/// The first step, as a share of the scale of the quantity.
const INITIAL_STEP: f64 = 1e-2;
/// The smallest step, as a share of the scale of the quantity.
const SMALLEST_STEP: f64 = 1e-8;
/// The largest step of a parameter, as a share of the width of its bounds.
const LARGEST_STEP: f64 = 0.25;
/// The tolerance of a held prediction, as a share of the step.
const CONSTRAINT_TOLERANCE: f64 = 1e-4;
/// How many times in a row a held prediction may be missed before the
/// profile ends.
const RETRIES: usize = 4;
/// The outer iterations of a held prediction.
const OUTER_ITERATIONS: u64 = 50;

/// What a profile holds fixed.
#[derive(Debug, Clone)]
pub enum ProfiledQuantity {
    /// The parameter at this index of the problem.
    Parameter(usize),
    /// The prediction of this observation, read as an observation is (its
    /// outputs summed, its window, its factor); its value is not read.
    Prediction(ObservedDataPoint),
}

/// Why one side of a profile ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileEnd {
    /// The profile rose past the threshold.
    Crossed,
    /// The parameter reached its bound below the threshold.
    Bound,
    /// The side took `max_steps` points below the threshold.
    MaxSteps,
    /// No values within the bounds give the prediction: the constraint
    /// could not be met.
    Unreachable,
}

/// One point of a profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfilePoint {
    /// The value the quantity is held at.
    pub value: f64,
    /// The profile statistic: the objective of the refit less that of the
    /// fit.
    pub statistic: f64,
    /// Every parameter of the refit, in problem order.
    pub parameters: Vec<f64>,
    /// The multiplier of a held prediction: minus the slope of the profile.
    pub multiplier: Option<f64>,
    /// Whether the refit converged.
    pub converged: bool,
}

/// The profile of one quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    /// The value of the quantity at the fit.
    pub center: f64,
    /// The threshold of the statistic.
    pub threshold: f64,
    /// The points, by increasing value; the fit is one of them, with
    /// statistic 0.
    pub points: Vec<ProfilePoint>,
    /// Why the side below and the side above the fit end.
    pub ends: [ProfileEnd; 2],
    /// Where the statistic crosses the threshold below and above the fit;
    /// `None` on a side that does not cross. The quadratic through the last
    /// three points of a side gives the crossing, and the line through the
    /// last two when it has no root between them.
    pub interval: [Option<f64>; 2],
    /// The evaluations of the objective, rejected steps included.
    pub evaluations: usize,
}

impl Profile {
    /// The interval of the quantity: where the statistic crosses the
    /// threshold below and above the fit.
    pub fn interval(&self) -> [Option<f64>; 2] {
        self.interval
    }
}

/// How each point of a prediction profile sets the initial penalty of its
/// augmented Lagrangian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyRule {
    /// The scaled initial penalty of Algencan, from the objective and the
    /// constraint at the start.
    Algencan,
    /// `factor · |Δλ / Δc|`: the factor times the curvature of the profile
    /// from the multipliers of the two points before; Algencan's at the
    /// first step.
    Curvature(f64),
}

/// One side of a profile, outward from the fit.
struct Side {
    points: Vec<ProfilePoint>,
    end: ProfileEnd,
    evaluations: usize,
}

/// Traces the profiles of a problem from a fit.
pub struct ProfileTracer<'a, E: SimulationEngine> {
    problem: &'a CalibrationProblem<E>,
    config: &'a OptimizationConfig,
    fit: Vec<f64>,
    fit_loss: f64,
    seed: u64,
    penalty: PenaltyRule,
}

impl<'a, E> ProfileTracer<'a, E>
where
    E: SimulationEngine + Send + Sync,
{
    /// A tracer of `problem` from the parameter values `fit`, whose
    /// subproblems run the optimizer of `config`. A particle swarm takes the
    /// seed of point `k` of side `d` of quantity `i` from
    /// `replicate_seed(replicate_seed(replicate_seed(seed, i), d), k)`.
    ///
    /// # Errors
    ///
    /// Returns a message when `fit` has the wrong length or is not feasible.
    pub fn new(
        problem: &'a CalibrationProblem<E>,
        config: &'a OptimizationConfig,
        fit: &[f64],
        seed: u64,
    ) -> Result<Self, String> {
        if fit.len() != problem.num_parameters() {
            return Err(format!(
                "The fit has {} values for {} parameters",
                fit.len(),
                problem.num_parameters()
            ));
        }
        let fit = problem.applied_parameter_values(fit);
        let score = OptimizationObjective::new(problem)
            .score(&fit)
            .map_err(|error| error.to_string())?;
        if !score.feasible {
            return Err("The fit a profile starts from is not feasible".to_string());
        }
        Ok(Self {
            problem,
            config,
            fit,
            fit_loss: score.cost,
            seed,
            penalty: PenaltyRule::Algencan,
        })
    }

    /// The same tracer with another rule for the penalty of a held
    /// prediction.
    pub fn with_penalty_rule(mut self, rule: PenaltyRule) -> Self {
        self.penalty = rule;
        self
    }

    /// The problem.
    pub fn problem(&self) -> &CalibrationProblem<E> {
        self.problem
    }

    /// The fit, clamped to the bounds.
    pub fn fit(&self) -> &[f64] {
        &self.fit
    }

    /// The objective at the fit: the discrepancy, a loss or `−log L`.
    pub fn fit_loss(&self) -> f64 {
        self.fit_loss
    }

    /// The numerical tolerance of a profile value: the `sd_tolerance` of
    /// Nelder-Mead, which bounds the spread of the objective over its
    /// simplex at the end, or `1e-6` for a particle swarm, which has none;
    /// at least `1e-9`, both relative to `max(1, |objective at the fit|)`.
    pub fn tolerance(&self) -> f64 {
        let relative = match self.config {
            OptimizationConfig::NelderMead(config) => config.sd_tolerance.max(1e-9),
            OptimizationConfig::ParticleSwarm(_) => 1e-6,
        };
        relative * self.fit_loss.abs().max(1.0)
    }

    /// The value of `quantity` at the fit.
    ///
    /// # Errors
    ///
    /// Returns a message when the index is not a parameter of the problem,
    /// or the prediction is not valid for it.
    pub fn center(&self, quantity: &ProfiledQuantity) -> Result<f64, String> {
        match quantity {
            ProfiledQuantity::Parameter(index) => self.fit.get(*index).copied().ok_or_else(|| {
                format!(
                    "Parameter {index} does not exist; the problem has {}",
                    self.problem.num_parameters()
                )
            }),
            ProfiledQuantity::Prediction(spec) => predict(self.problem, spec, &self.fit),
        }
    }

    /// The profile of `quantity` up to `threshold` on the statistic, with at
    /// most `max_steps` points on each side.
    ///
    /// # Errors
    ///
    /// As [`Self::center`], and the message of a subproblem that fails for a
    /// reason no point avoids.
    pub fn trace(
        &self,
        quantity: &ProfiledQuantity,
        threshold: f64,
        max_steps: usize,
    ) -> Result<Profile, String> {
        self.trace_all(std::slice::from_ref(quantity), &[threshold], max_steps)
            .pop()
            .expect("one profile per quantity")
    }

    /// The profile of each of `quantities` up to its threshold, every side
    /// of every quantity in parallel.
    ///
    /// # Panics
    ///
    /// Panics when `thresholds` does not hold one threshold per quantity.
    pub fn trace_all(
        &self,
        quantities: &[ProfiledQuantity],
        thresholds: &[f64],
        max_steps: usize,
    ) -> Vec<Result<Profile, String>> {
        assert_eq!(
            quantities.len(),
            thresholds.len(),
            "one threshold per quantity"
        );
        let centers: Vec<Result<f64, String>> = quantities.iter().map(|q| self.center(q)).collect();
        let sides: Vec<Result<Side, String>> = (0..2 * quantities.len())
            .into_par_iter()
            .map(|task| {
                let index = task / 2;
                let center = centers[index].clone()?;
                self.walk(
                    &quantities[index],
                    index as u64,
                    task % 2,
                    center,
                    thresholds[index],
                    max_steps,
                )
            })
            .collect();
        let mut sides = sides.into_iter();
        centers
            .into_iter()
            .zip(thresholds)
            .map(|(center, &threshold)| {
                let below = sides.next().expect("two sides per quantity");
                let above = sides.next().expect("two sides per quantity");
                let (center, below, above) = (center?, below?, above?);
                Ok(self.join(center, threshold, below, above))
            })
            .collect()
    }

    /// The point of the fit, and both sides, as one profile.
    fn join(&self, center: f64, threshold: f64, below: Side, above: Side) -> Profile {
        let fit = ProfilePoint {
            value: center,
            statistic: 0.0,
            parameters: self.fit.clone(),
            multiplier: None,
            converged: true,
        };
        let interval = [&below, &above].map(|side| {
            (side.end == ProfileEnd::Crossed).then(|| crossing(&fit, &side.points, threshold))
        });
        let ends = [below.end, above.end];
        let evaluations = below.evaluations + above.evaluations;
        let mut points: Vec<ProfilePoint> = below.points.into_iter().rev().collect();
        points.push(fit);
        points.extend(above.points);
        Profile {
            center,
            threshold,
            points,
            ends,
            interval,
            evaluations,
        }
    }

    /// Walk one side of `quantity` from `center`: below it for `side` 0,
    /// above it for 1.
    fn walk(
        &self,
        quantity: &ProfiledQuantity,
        stream: u64,
        side: usize,
        center: f64,
        threshold: f64,
        max_steps: usize,
    ) -> Result<Side, String> {
        let direction = if side == 0 { -1.0 } else { 1.0 };
        let (scale, limit, largest) = match quantity {
            ProfiledQuantity::Parameter(index) => {
                let (lo, hi) = self.problem.parameter_bounds()[*index];
                let width = hi - lo;
                (width, if side == 0 { lo } else { hi }, LARGEST_STEP * width)
            }
            ProfiledQuantity::Prediction(_) => {
                let scale = if center == 0.0 { 1.0 } else { center.abs() };
                (scale, direction * f64::INFINITY, f64::INFINITY)
            }
        };
        let control = threshold.max(self.tolerance());
        let smallest = SMALLEST_STEP * scale;
        let mut step = INITIAL_STEP * scale;
        let mut previous: Option<ProfilePoint> = None;
        let mut current = ProfilePoint {
            value: center,
            statistic: 0.0,
            parameters: self.fit.clone(),
            multiplier: matches!(quantity, ProfiledQuantity::Prediction(_)).then_some(0.0),
            converged: true,
        };
        let mut points = Vec::new();
        let mut evaluations = 0;
        let mut attempt = 0;
        let mut misses = 0;
        let stream_seed = replicate_seed(replicate_seed(self.seed, stream), side as u64);
        let end = loop {
            if points.len() >= max_steps {
                break ProfileEnd::MaxSteps;
            }
            let mut value = current.value + direction * step;
            let at_limit = direction * (value - limit) >= 0.0;
            if at_limit {
                value = limit;
            }
            if value == current.value {
                break ProfileEnd::Bound;
            }
            let start = self.secant(quantity, previous.as_ref(), &current, value);
            let seed = replicate_seed(stream_seed, attempt);
            attempt += 1;
            let result = match quantity {
                ProfiledQuantity::Parameter(index) => {
                    let subproblem = Subproblem::fix_parameter(self.problem, *index, value)?;
                    optimize_subproblem(&subproblem, self.config, &start.parameters, Some(seed))?
                }
                ProfiledQuantity::Prediction(spec) => {
                    let settings = ConstraintSettings {
                        tolerance: CONSTRAINT_TOLERANCE * step,
                        max_outer_iterations: OUTER_ITERATIONS,
                        multiplier: start.multiplier,
                        penalty: self.penalty(previous.as_ref(), &current),
                    };
                    let subproblem = Subproblem::fix_prediction(self.problem, spec, value)?
                        .with_settings(settings)?;
                    optimize_subproblem(&subproblem, self.config, &start.parameters, Some(seed))?
                }
            };
            evaluations += result.evaluations;
            if !held(&result, value, step) {
                misses += 1;
                if misses > RETRIES || step <= smallest {
                    break ProfileEnd::Unreachable;
                }
                step = (step / 2.0).max(smallest);
                continue;
            }
            misses = 0;
            let statistic = result.fit.final_loss - self.fit_loss;
            let rise = statistic - current.statistic;
            if rise > 2.0 * RISE_FRACTION * control && step > smallest {
                step = (step / 2.0).max(smallest);
                continue;
            }
            let point = ProfilePoint {
                value,
                statistic,
                parameters: result.fit.best_parameters,
                multiplier: result.multiplier,
                converged: result.fit.converged,
            };
            points.push(point.clone());
            previous = Some(std::mem::replace(&mut current, point));
            if statistic > threshold {
                break ProfileEnd::Crossed;
            }
            if at_limit {
                break ProfileEnd::Bound;
            }
            if rise < RISE_FRACTION * control / 2.0 {
                step = (2.0 * step).min(largest);
            }
        };
        Ok(Side {
            points,
            end,
            evaluations,
        })
    }

    /// The start of the point at `value`: every parameter and the multiplier
    /// extrapolated along the line through the two points before, or those
    /// of the point before at the first step; parameters clamped to their
    /// bounds, and a held parameter at `value`.
    fn secant(
        &self,
        quantity: &ProfiledQuantity,
        previous: Option<&ProfilePoint>,
        current: &ProfilePoint,
        value: f64,
    ) -> ProfilePoint {
        let ratio = previous.map_or(0.0, |p| (value - current.value) / (current.value - p.value));
        let extrapolate = |now: f64, before: Option<f64>| match before {
            Some(before) => now + (now - before) * ratio,
            None => now,
        };
        let parameters: Vec<f64> = current
            .parameters
            .iter()
            .enumerate()
            .map(|(i, &now)| extrapolate(now, previous.map(|p| p.parameters[i])))
            .collect();
        let mut parameters = self.problem.applied_parameter_values(&parameters);
        if let ProfiledQuantity::Parameter(index) = quantity {
            parameters[*index] = value;
        }
        ProfilePoint {
            value,
            statistic: 0.0,
            parameters,
            multiplier: current
                .multiplier
                .map(|now| extrapolate(now, previous.and_then(|p| p.multiplier))),
            converged: false,
        }
    }

    /// The initial penalty of the next point of a prediction profile.
    fn penalty(&self, previous: Option<&ProfilePoint>, current: &ProfilePoint) -> Option<f64> {
        match self.penalty {
            PenaltyRule::Algencan => None,
            PenaltyRule::Curvature(factor) => {
                let previous = previous?;
                let slope =
                    (current.multiplier? - previous.multiplier?) / (current.value - previous.value);
                let penalty = factor * slope.abs();
                (penalty.is_finite() && penalty > 0.0).then_some(penalty)
            }
        }
    }
}

/// Whether a refit holds its quantity: always for a parameter; for a
/// prediction, when it is within a thousandth of the step of its value.
fn held(result: &SubproblemResult, value: f64, step: f64) -> bool {
    match result.multiplier {
        None => true,
        Some(_) => result
            .prediction
            .is_some_and(|g| (g - value).abs() <= 10.0 * CONSTRAINT_TOLERANCE * step),
    }
}

/// Where a side crosses `threshold`: the root, between its last two points,
/// of the quadratic through its last three (the fit counts as the first),
/// or of the line through its last two.
fn crossing(fit: &ProfilePoint, side: &[ProfilePoint], threshold: f64) -> f64 {
    let mut outward: Vec<&ProfilePoint> = vec![fit];
    outward.extend(side);
    let n = outward.len();
    let (inside, outside) = (outward[n - 2], outward[n - 1]);
    let line = || {
        inside.value
            + (threshold - inside.statistic) * (outside.value - inside.value)
                / (outside.statistic - inside.statistic)
    };
    if n < 3 {
        return line();
    }
    let first = outward[n - 3];
    quadratic_root(
        [first.value, inside.value, outside.value],
        [first.statistic, inside.statistic, outside.statistic],
        threshold,
    )
    .unwrap_or_else(line)
}

/// The root of `q(x) = level` between `x[1]` and `x[2]`, where `q` is the
/// quadratic through the three points; `None` when it has none there.
fn quadratic_root(x: [f64; 3], y: [f64; 3], level: f64) -> Option<f64> {
    let d1 = (y[1] - y[0]) / (x[1] - x[0]);
    let d2 = ((y[2] - y[1]) / (x[2] - x[1]) - d1) / (x[2] - x[0]);
    // q(x) = y0 + d1 (x − x0) + d2 (x − x0)(x − x1) = a x² + b x + c.
    let a = d2;
    let b = d1 - d2 * (x[0] + x[1]);
    let c = y[0] - d1 * x[0] + d2 * x[0] * x[1] - level;
    let (lo, hi) = (x[1].min(x[2]), x[1].max(x[2]));
    let inside = |root: f64| (lo..=hi).contains(&root).then_some(root);
    if a.abs() <= f64::EPSILON * (b.abs() + c.abs()) {
        return if b == 0.0 { None } else { inside(-c / b) };
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    // The stable pair of roots.
    let q = -0.5 * (b + b.signum() * root);
    let roots = [q / a, if q == 0.0 { f64::NAN } else { c / q }];
    roots.into_iter().find_map(inside)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_of_a_quadratic_is_found_between_its_last_two_points() {
        // y = (x − 1)², crossing 4 at x = 3 (and at −1, outside).
        let root = quadratic_root([1.0, 2.0, 3.5], [0.0, 1.0, 6.25], 4.0).unwrap();
        assert!((root - 3.0).abs() < 1e-12, "{root}");
        // The same profile walked downwards crosses at −1.
        let root = quadratic_root([1.0, 0.0, -1.5], [0.0, 1.0, 6.25], 4.0).unwrap();
        assert!((root + 1.0).abs() < 1e-12, "{root}");
        // A line has its one root.
        let root = quadratic_root([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1.5).unwrap();
        assert!((root - 1.5).abs() < 1e-12, "{root}");
        // No root between the last two points.
        assert_eq!(quadratic_root([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 3.0), None);
        assert_eq!(quadratic_root([0.0, 1.0, 2.0], [5.0, 5.0, 5.0], 3.0), None);
    }
}
