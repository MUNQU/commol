//! The one check of a setting that is a share: a number strictly between 0
//! and 1, and the one check of a confidence level.
//!
//! The settings of an engine, such as the `epsilon` of an adaptive leap, and
//! the settings of the inference methods read [`check_open_unit`], so every
//! share has one message; a confidence level reads
//! [`check_confidence_level`], which adds its example.

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

/// Check that `confidence_level` is a confidence level, strictly between 0
/// and 1: the one rule of a confidence level, which each caller checks at its
/// boundary before it computes with the level.
///
/// # Errors
///
/// Returns the message of [`check_open_unit`] for `confidence_level`, with
/// the example 0.95, when it is not strictly between 0 and 1.
pub fn check_confidence_level(confidence_level: f64) -> Result<(), String> {
    check_open_unit("confidence_level", confidence_level)
        .map_err(|error| format!("{error}; use 0.95 for a 95% interval"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_confidence_level_is_strictly_between_0_and_1() {
        assert_eq!(check_confidence_level(0.95), Ok(()));
        for level in [0.0, 1.0, 1.5, f64::NAN] {
            assert_eq!(
                check_confidence_level(level),
                Err(format!(
                    "confidence_level is strictly between 0 and 1, got {level}; \
                     use 0.95 for a 95% interval"
                ))
            );
        }
    }
}
