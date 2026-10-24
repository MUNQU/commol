//! The rules of the values of a model: an initial amount, the fractions of a
//! stratification, an impulse and the points of a series.
//!
//! Each rule is declared once, here. The compilation of a model checks each
//! value with it, and the Python object that holds the value calls the same
//! check when it is built (`commol_rs.model`), so a value outside its rule
//! gets one message, whoever reads it first.

use commol_core::{Bound, Real};

/// The initial amount of a bin or of a compartment.
pub const INITIAL_AMOUNT: Real = Real::non_negative("initial amount");
/// The fraction of a category of a stratification.
pub const FRACTION: Real = Real::new("fraction", Bound::Closed(0.0), Bound::Closed(1.0));
/// How far from 1 the fractions of a stratification may sum.
pub const FRACTION_SUM_TOLERANCE: f64 = 1e-6;
/// A position of `impulse_at`.
pub const IMPULSE_POSITION: Real = Real::non_negative("impulse position");
/// The period of `impulse_every`.
pub const IMPULSE_PERIOD: Real = Real::positive("impulse period");
/// The offset of `impulse_every`.
pub const IMPULSE_OFFSET: Real = Real::non_negative("impulse offset");
/// The position of a point of a series.
pub const SERIES_POSITION: Real = Real::non_negative("series position");
/// The value of a point of a series.
pub const SERIES_VALUE: Real = Real::finite("series value");

/// Check the initial amount of the bin `bin`.
///
/// # Errors
///
/// Returns the message of [`INITIAL_AMOUNT`] after the bin, such as "Bin
/// 'B': initial amount is a finite number of at least 0, got -1".
pub fn check_bin_amount(bin: &str, amount: f64) -> Result<(), String> {
    INITIAL_AMOUNT
        .check(amount)
        .map_err(|error| format!("Bin '{bin}': {error}"))
}

/// Check the initial amount that overrides the compartment `compartment`.
///
/// # Errors
///
/// Returns the message of [`INITIAL_AMOUNT`] after the compartment.
pub fn check_compartment_amount(compartment: &str, amount: f64) -> Result<(), String> {
    INITIAL_AMOUNT
        .check(amount)
        .map_err(|error| format!("Compartment '{compartment}': {error}"))
}

/// Check the `(category, fraction)` pairs of the stratification
/// `stratification`: each fraction of [`FRACTION`], and a sum of 1 within
/// [`FRACTION_SUM_TOLERANCE`].
///
/// # Errors
///
/// Returns the first rule that the fractions break, such as "Stratification
/// 'g', category 'g1': fraction is a number from 0 to 1, got 1.5".
pub fn check_fractions<'a>(
    stratification: &str,
    fractions: impl IntoIterator<Item = (&'a str, f64)>,
) -> Result<(), String> {
    let mut sum = 0.0;
    for (category, fraction) in fractions {
        FRACTION.check(fraction).map_err(|error| {
            format!("Stratification '{stratification}', category '{category}': {error}")
        })?;
        sum += fraction;
    }
    if (1.0 - FRACTION_SUM_TOLERANCE..=1.0 + FRACTION_SUM_TOLERANCE).contains(&sum) {
        Ok(())
    } else {
        Err(format!(
            "Stratification '{stratification}': the fractions of its categories sum to 1 \
             within {FRACTION_SUM_TOLERANCE:e}, got {sum}"
        ))
    }
}

/// Check the impulse of the transition `transition`: at most one of its
/// positions `at` and its schedule `every`, `(period, offset)`; at least one
/// position, each on the axis ([`IMPULSE_POSITION`]); a period of
/// [`IMPULSE_PERIOD`] and an offset of [`IMPULSE_OFFSET`].
///
/// # Errors
///
/// Returns the first rule that the impulse breaks, after the transition.
pub fn check_impulse(
    transition: &str,
    at: Option<&[f64]>,
    every: Option<(f64, f64)>,
) -> Result<(), String> {
    let checked = match (at, every) {
        (None, None) => Ok(()),
        (Some(_), Some(_)) => Err("set impulse_at or impulse_every, not both".to_string()),
        (Some([]), None) => Err("impulse_at has no position; give at least one".to_string()),
        (Some(positions), None) => positions
            .iter()
            .try_for_each(|&position| IMPULSE_POSITION.check(position)),
        (None, Some((period, offset))) => IMPULSE_PERIOD
            .check(period)
            .and_then(|()| IMPULSE_OFFSET.check(offset)),
    };
    checked.map_err(|error| format!("Transition '{transition}': {error}"))
}

/// Check the `(position, value)` points of a series: at least one point,
/// each position on the axis ([`SERIES_POSITION`]) and once, each value
/// finite ([`SERIES_VALUE`]).
///
/// # Errors
///
/// Returns the first rule that the points break.
pub fn check_series(data: &[(f64, f64)]) -> Result<(), String> {
    if data.is_empty() {
        return Err("a series has at least one point, got none".to_string());
    }
    data.iter().try_for_each(|&(position, value)| {
        SERIES_POSITION
            .check(position)
            .and_then(|()| SERIES_VALUE.check(value))
    })?;
    let mut positions: Vec<f64> = data.iter().map(|&(position, _)| position).collect();
    positions.sort_unstable_by(f64::total_cmp);
    match positions.windows(2).find(|pair| pair[0] == pair[1]) {
        Some(pair) => Err(format!(
            "a series has one point per position, got two at {}",
            pair[0]
        )),
        None => Ok(()),
    }
}
