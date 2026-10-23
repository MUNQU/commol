//! When two positions on the axis are the same position.
//!
//! A position that an engine computes, such as the end of a piece of an
//! integration or a point of a schedule `offset + k × period`, carries the
//! rounding of its arithmetic. Every crate that compares such positions uses
//! [`same_position`], so they all agree on which positions are one.

/// Relative tolerance with which two positions on the axis are the same
/// ([`same_position`]).
pub const POSITION_TOLERANCE: f64 = 1e-9;

/// Whether `a` and `b` are the same position: they differ by at most
/// [`POSITION_TOLERANCE`] times the larger of their magnitudes and 1.
pub fn same_position(a: f64, b: f64) -> bool {
    (a - b).abs() <= POSITION_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}
