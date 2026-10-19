//! The unbounded scale a move kernel proposes on.
//!
//! Each parameter maps to the whole real line by the bounds it has: the
//! logit of its position between two finite bounds, `log(θ − a)` with a
//! lower bound `a` only, `log(b − θ)` with an upper bound `b` only, and the
//! identity with none. A random walk on that scale never proposes a point
//! outside the bounds, and its acceptance adds the log Jacobian
//! [`Transform::log_jacobian`] of the map back.

/// The largest magnitude of a value on the unbounded scale: a parameter on
/// its bound maps to it instead of to an infinity.
const LIMIT: f64 = 700.0;

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
