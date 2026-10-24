//! The one check of a setting that is a share: a number strictly between 0
//! and 1.
//!
//! The settings of an engine, such as the `epsilon` of an adaptive leap, and
//! the settings of the inference methods, such as a confidence level, read
//! [`check_open_unit`], so every share has one message.

/// Check that the setting `name` is strictly between 0 and 1.
///
/// # Errors
///
/// Returns "`name` is strictly between 0 and 1, got `value`" when it is
/// not, NaN included.
pub fn check_open_unit(name: &str, value: f64) -> Result<(), String> {
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(format!("{name} is strictly between 0 and 1, got {value}"))
    }
}
