//! The unbounded scale a move kernel proposes on.
//!
//! Each parameter maps to the whole real line by the bounds it has: the
//! logit of its position between two finite bounds, `log(θ − a)` with a
//! lower bound `a` only, `log(b − θ)` with an upper bound `b` only, and the
//! identity with none. A random walk on that scale never proposes a point
//! outside the bounds, and its acceptance adds the log Jacobian
//! [`Transform::log_jacobian`] of the map back.
//!
//! [`on_bound`] is the one rule of a value on a bound, which the curvature,
//! the Laplace mixture and the profiles read.

/// The largest magnitude of a value on the unbounded scale: a parameter on
/// its bound maps to it instead of to an infinity.
const LIMIT: f64 = 700.0;

/// How close to a bound a value is on it ([`on_bound`]): this share of the
/// width of two finite bounds, or of the larger of 1 and the magnitude of
/// the bound when only one is finite. It is about the square root of the
/// machine epsilon, where an optimizer that the bound stops ends.
pub const ON_BOUND: f64 = 1e-8;

/// The bound of `(lower, upper)` that `value` is on, the lower first: within
/// [`ON_BOUND`] of it, or beyond it. An infinite bound is no bound, so a
/// value with none is on no bound.
pub fn on_bound(value: f64, (lower, upper): (f64, f64)) -> Option<f64> {
    let scale = match (lower.is_finite(), upper.is_finite()) {
        (true, true) => upper - lower,
        (true, false) => lower.abs().max(1.0),
        (false, true) => upper.abs().max(1.0),
        (false, false) => return None,
    };
    let tolerance = ON_BOUND * scale;
    if lower.is_finite() && value - lower <= tolerance {
        Some(lower)
    } else if upper.is_finite() && upper - value <= tolerance {
        Some(upper)
    } else {
        None
    }
}

/// How one parameter maps to the real line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scale {
    /// `u = log((θ − a) / (b − θ))`.
    Logit {
        /// The lower bound `a`.
        lower: f64,
        /// The upper bound `b`.
        upper: f64,
    },
    /// `u = log(θ − a)`.
    Log {
        /// The lower bound `a`.
        lower: f64,
    },
    /// `u = log(b − θ)`.
    LogFromAbove {
        /// The upper bound `b`.
        upper: f64,
    },
    /// `u = θ`.
    Identity,
}

impl Scale {
    /// The bounds of the parameter, an infinity where it has none.
    pub fn bounds(&self) -> (f64, f64) {
        match *self {
            Self::Logit { lower, upper } => (lower, upper),
            Self::Log { lower } => (lower, f64::INFINITY),
            Self::LogFromAbove { upper } => (f64::NEG_INFINITY, upper),
            Self::Identity => (f64::NEG_INFINITY, f64::INFINITY),
        }
    }
}

/// The scale of every parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    scales: Vec<Scale>,
}

impl Transform {
    /// The transform of parameters with `bounds`; an infinite bound is no
    /// bound.
    pub fn new(bounds: &[(f64, f64)]) -> Self {
        let scales = bounds
            .iter()
            .map(
                |&(lower, upper)| match (lower.is_finite(), upper.is_finite()) {
                    (true, true) => Scale::Logit { lower, upper },
                    (true, false) => Scale::Log { lower },
                    (false, true) => Scale::LogFromAbove { upper },
                    (false, false) => Scale::Identity,
                },
            )
            .collect();
        Self { scales }
    }

    /// The scale of each parameter.
    pub fn scales(&self) -> &[Scale] {
        &self.scales
    }

    /// The first parameter of `values` on one of its bounds ([`on_bound`]),
    /// and that bound.
    pub fn on_bound(&self, values: &[f64]) -> Option<(usize, f64)> {
        self.scales
            .iter()
            .zip(values)
            .enumerate()
            .find_map(|(index, (scale, &value))| {
                on_bound(value, scale.bounds()).map(|bound| (index, bound))
            })
    }

    /// The values on the unbounded scale; a value on its bound maps to
    /// `±700`.
    pub fn to_unbounded(&self, values: &[f64]) -> Vec<f64> {
        self.scales
            .iter()
            .zip(values)
            .map(|(scale, &theta)| {
                let u = match *scale {
                    Scale::Logit { lower, upper } => {
                        libm::log(theta - lower) - libm::log(upper - theta)
                    }
                    Scale::Log { lower } => libm::log(theta - lower),
                    Scale::LogFromAbove { upper } => libm::log(upper - theta),
                    Scale::Identity => theta,
                };
                u.clamp(-LIMIT, LIMIT)
            })
            .collect()
    }

    /// The values of points on the unbounded scale, kept inside the bounds.
    pub fn from_unbounded(&self, unbounded: &[f64]) -> Vec<f64> {
        self.scales
            .iter()
            .zip(unbounded)
            .map(|(scale, &u)| match *scale {
                Scale::Logit { lower, upper } => {
                    let width = upper - lower;
                    // From the nearer bound, so that a value near either
                    // keeps its digits.
                    let theta = if u < 0.0 {
                        lower + width / (1.0 + libm::exp(-u))
                    } else {
                        upper - width / (1.0 + libm::exp(u))
                    };
                    theta.clamp(lower, upper)
                }
                Scale::Log { lower } => (lower + libm::exp(u)).max(lower),
                Scale::LogFromAbove { upper } => (upper - libm::exp(u)).min(upper),
                Scale::Identity => u,
            })
            .collect()
    }

    /// `dθ/du` of each parameter at `values`: `(θ − a)(b − θ)/(b − a)` for a
    /// logit, `θ − a` or `b − θ` for a log, 1 for the identity; 0 on a
    /// bound.
    pub fn derivatives(&self, values: &[f64]) -> Vec<f64> {
        self.scales
            .iter()
            .zip(values)
            .map(|(scale, &theta)| match *scale {
                Scale::Logit { lower, upper } => {
                    ((theta - lower) * (upper - theta) / (upper - lower)).max(0.0)
                }
                Scale::Log { lower } => (theta - lower).max(0.0),
                Scale::LogFromAbove { upper } => (upper - theta).max(0.0),
                Scale::Identity => 1.0,
            })
            .collect()
    }

    /// `log |dθ/du|` at `values`: the sum over the parameters, `−∞` when a
    /// value sits on a bound.
    pub fn log_jacobian(&self, values: &[f64]) -> f64 {
        self.scales
            .iter()
            .zip(values)
            .map(|(scale, &theta)| match *scale {
                Scale::Logit { lower, upper } => {
                    libm::log(theta - lower) + libm::log(upper - theta) - libm::log(upper - lower)
                }
                Scale::Log { lower } => libm::log(theta - lower),
                Scale::LogFromAbove { upper } => libm::log(upper - theta),
                Scale::Identity => 0.0,
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOUNDS: [(f64, f64); 4] = [
        (2.0, 6.0),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 3.0),
        (f64::NEG_INFINITY, f64::INFINITY),
    ];

    #[test]
    fn each_parameter_maps_by_the_bounds_it_has() {
        assert_eq!(
            Transform::new(&BOUNDS).scales(),
            [
                Scale::Logit {
                    lower: 2.0,
                    upper: 6.0
                },
                Scale::Log { lower: 1.0 },
                Scale::LogFromAbove { upper: 3.0 },
                Scale::Identity,
            ]
        );
    }

    #[test]
    fn the_map_back_inverts_the_map() {
        let transform = Transform::new(&BOUNDS);
        let values = [5.9, 1.5, -10.0, 4.0];
        let back = transform.from_unbounded(&transform.to_unbounded(&values));
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() < 1e-12, "{a} {b}");
        }
    }

    #[test]
    fn the_jacobian_is_the_derivative_of_the_map_back() {
        let transform = Transform::new(&BOUNDS);
        let values = [3.0, 2.5, 0.0, -1.0];
        let u = transform.to_unbounded(&values);
        let h = 1e-6;
        let mut expected = 0.0;
        for j in 0..4 {
            let mut up = u.clone();
            let mut down = u.clone();
            up[j] += h;
            down[j] -= h;
            let derivative =
                (transform.from_unbounded(&up)[j] - transform.from_unbounded(&down)[j]) / (2.0 * h);
            expected += derivative.abs().ln();
        }
        assert!((transform.log_jacobian(&values) - expected).abs() < 1e-6);
        let product: f64 = transform.derivatives(&values).iter().map(|d| d.ln()).sum();
        assert!((transform.log_jacobian(&values) - product).abs() < 1e-12);
    }

    #[test]
    fn far_points_stay_inside_the_bounds() {
        let transform = Transform::new(&BOUNDS[..1]);
        assert_eq!(transform.from_unbounded(&[-1e6]), vec![2.0]);
        assert_eq!(transform.from_unbounded(&[1e6]), vec![6.0]);
        assert_eq!(transform.to_unbounded(&[2.0]), vec![-LIMIT]);
        assert_eq!(transform.log_jacobian(&[2.0]), f64::NEG_INFINITY);
    }
}
