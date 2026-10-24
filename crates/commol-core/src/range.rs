//! The range rules of the settings: each setting declares its range once, as
//! data, and one check of that rule gives one message for every value
//! outside it.
//!
//! - [`Count`]: a whole number, such as `Count::at_least("replicates", 1)`.
//!   [`Count::check`] reads a whole number of any type, a negative one
//!   included, so `replicates = -1` and `replicates = 0` get one message.
//! - [`Real`]: a number in an interval, such as a share
//!   (`Real::share("target_cess")`), a positive length
//!   (`Real::positive("window")`) or a level ([`CONFIDENCE_LEVEL`]).
//! - [`Ranges`]: the rules of the settings of one object, as the Python API
//!   names it. [`Ranges::check`] checks the settings of the object, each a
//!   [`Setting`] of its own type, so the Python model of the object calls
//!   one check of the engine when it is built.
//!
//! The function or the constructor of the engine that takes a setting checks
//! it with the same rule, so a Rust caller and a binding get the same
//! message.

use std::fmt::Display;

/// The rule of a setting that is a whole number from `min` to `max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Count {
    name: &'static str,
    min: i128,
    max: Option<i128>,
}

impl Count {
    /// The rule of the setting `name`: a whole number of at least `min`.
    pub const fn at_least(name: &'static str, min: i128) -> Self {
        Self {
            name,
            min,
            max: None,
        }
    }

    /// The rule of the setting `name`: a whole number from `min` to `max`.
    pub const fn between(name: &'static str, min: i128, max: i128) -> Self {
        Self {
            name,
            min,
            max: Some(max),
        }
    }

    /// The rule with its largest value lowered to `max`, for a setting whose
    /// largest value depends on its input, such as a subset of the
    /// parameters.
    pub fn at_most(self, max: usize) -> Self {
        let max = i128::try_from(max).unwrap_or(i128::MAX);
        Self {
            max: Some(self.max.map_or(max, |current| current.min(max))),
            ..self
        }
    }

    /// The name of the setting.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Check the whole number `value`, of any type.
    ///
    /// # Errors
    ///
    /// Returns "`name` is a whole number of at least `min`, got `value`" (or
    /// "from `min` to `max`") for every value outside the range, a negative
    /// one included.
    pub fn check<V: Copy + Display + TryInto<i128>>(&self, value: V) -> Result<(), String> {
        let inside = value
            .try_into()
            .is_ok_and(|whole: i128| whole >= self.min && self.max.is_none_or(|max| whole <= max));
        if inside {
            Ok(())
        } else {
            Err(self.message(value))
        }
    }

    /// Read the signed whole number `value` as the type of the setting, such
    /// as `usize`, after [`Count::check`].
    ///
    /// # Errors
    ///
    /// Returns the message of [`Count::check`] when `value` is outside the
    /// range.
    pub fn read<T: TryFrom<i128>>(&self, value: impl Into<i128>) -> Result<T, String> {
        let value = value.into();
        self.check(value)?;
        T::try_from(value).map_err(|_| unreachable!("the range of {} fits its type", self.name))
    }

    /// A value outside the range: the whole number just below it.
    pub fn outside(&self) -> i128 {
        self.min - 1
    }

    fn message(&self, value: impl Display) -> String {
        let range = match self.max {
            None => format!("of at least {}", self.min),
            Some(max) => format!("from {} to {max}", self.min),
        };
        format!("{} is a whole number {range}, got {value}", self.name)
    }

    fn check_setting(&self, setting: &Setting) -> Result<(), String> {
        match setting {
            Setting::Whole(whole) => self.check(*whole),
            Setting::Number(number) => Err(self.message(number)),
            Setting::List(_) | Setting::Other => Ok(()),
        }
    }
}

/// An end of the interval of a [`Real`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bound {
    /// The end is not in the interval. `Open(f64::INFINITY)` as the upper
    /// end makes the interval finite.
    Open(f64),
    /// The end is in the interval. `Closed(f64::INFINITY)` as the upper end
    /// lets the setting be infinite.
    Closed(f64),
}

/// The rule of a setting that is a number in an interval. NaN is never in
/// it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Real {
    name: &'static str,
    low: Bound,
    high: Bound,
    hint: Option<&'static str>,
}

impl Real {
    /// The rule of the setting `name`: a number from `low` to `high`.
    pub const fn new(name: &'static str, low: Bound, high: Bound) -> Self {
        Self {
            name,
            low,
            high,
            hint: None,
        }
    }

    /// A finite number.
    pub const fn finite(name: &'static str) -> Self {
        Self::new(
            name,
            Bound::Open(f64::NEG_INFINITY),
            Bound::Open(f64::INFINITY),
        )
    }

    /// A finite number greater than 0, such as a length.
    pub const fn positive(name: &'static str) -> Self {
        Self::new(name, Bound::Open(0.0), Bound::Open(f64::INFINITY))
    }

    /// A finite number of at least 0, such as a position on the axis.
    pub const fn non_negative(name: &'static str) -> Self {
        Self::new(name, Bound::Closed(0.0), Bound::Open(f64::INFINITY))
    }

    /// A number greater than 0, infinity included, such as a weight.
    pub const fn positive_or_infinite(name: &'static str) -> Self {
        Self::new(name, Bound::Open(0.0), Bound::Closed(f64::INFINITY))
    }

    /// A number of at least 0, infinity included, such as a tolerance.
    pub const fn non_negative_or_infinite(name: &'static str) -> Self {
        Self::new(name, Bound::Closed(0.0), Bound::Closed(f64::INFINITY))
    }

    /// A number strictly between 0 and 1, such as a share or a probability.
    pub const fn share(name: &'static str) -> Self {
        Self::new(name, Bound::Open(0.0), Bound::Open(1.0))
    }

    /// The same rule, with a hint after its message.
    pub const fn with_hint(self, hint: &'static str) -> Self {
        Self {
            hint: Some(hint),
            ..self
        }
    }

    /// The name of the setting.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Check the number `value`.
    ///
    /// # Errors
    ///
    /// Returns "`name` is `range`, got `value`", and the hint of the rule,
    /// for every value outside the interval, NaN included.
    pub fn check(&self, value: f64) -> Result<(), String> {
        let above = match self.low {
            Bound::Open(low) => value > low,
            Bound::Closed(low) => value >= low,
        };
        let below = match self.high {
            Bound::Open(high) => value < high,
            Bound::Closed(high) => value <= high,
        };
        if above && below {
            Ok(())
        } else {
            Err(self.message(value))
        }
    }

    /// A value outside the interval: its lower end, or the number below it,
    /// when the end is finite; else infinity.
    pub fn outside(&self) -> f64 {
        match (self.low, self.high) {
            (Bound::Open(low), _) if low.is_finite() => low,
            (Bound::Closed(low), _) if low.is_finite() => low - 1.0,
            _ => f64::INFINITY,
        }
    }

    /// The interval in words, such as "a finite number greater than 0" or
    /// "strictly between 0 and 1".
    pub fn range(&self) -> String {
        match (self.low, self.high) {
            (Bound::Open(low), Bound::Open(high)) if low.is_finite() && high.is_finite() => {
                format!("strictly between {low} and {high}")
            }
            (Bound::Closed(low), Bound::Closed(high)) if low.is_finite() && high.is_finite() => {
                format!("a number from {low} to {high}")
            }
            (low, high) => {
                let finite = high == Bound::Open(f64::INFINITY);
                let ends: Vec<String> = [
                    match low {
                        Bound::Open(low) if low.is_finite() => Some(format!("greater than {low}")),
                        Bound::Closed(low) if low.is_finite() => Some(format!("of at least {low}")),
                        _ => None,
                    },
                    match high {
                        Bound::Open(high) if high.is_finite() => Some(format!("less than {high}")),
                        Bound::Closed(high) if high.is_finite() => Some(format!("at most {high}")),
                        _ => None,
                    },
                ]
                .into_iter()
                .flatten()
                .collect();
                let number = if finite {
                    "a finite number"
                } else {
                    "a number"
                };
                [number.to_string(), ends.join(" and ")]
                    .into_iter()
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }

    fn message(&self, value: f64) -> String {
        let hint = self
            .hint
            .map(|hint| format!("; {hint}"))
            .unwrap_or_default();
        format!("{} is {}, got {value}{hint}", self.name, self.range())
    }

    fn check_setting(&self, setting: &Setting) -> Result<(), String> {
        match setting {
            // A whole number beyond 2^53 reads as the nearest number.
            Setting::Whole(whole) => self.check(*whole as f64),
            Setting::Number(number) => self.check(*number),
            Setting::List(_) | Setting::Other => Ok(()),
        }
    }
}

/// The rule of a confidence level: strictly between 0 and 1, with the example
/// 0.95. Each method checks its level with it before it computes with the
/// level.
pub const CONFIDENCE_LEVEL: Real =
    Real::share("confidence_level").with_hint("use 0.95 for a 95% interval");

/// The value of a setting, of the type its caller holds it in.
#[derive(Clone, Debug, PartialEq)]
pub enum Setting {
    /// A whole number, such as a count; a rule of a number reads it as a
    /// number.
    Whole(i128),
    /// A number, NaN and infinity included; a rule of a count rejects it.
    Number(f64),
    /// A list: the rule checks each of its settings.
    List(Vec<Setting>),
    /// Not a number, such as a choice (`"auto"`), a parameter name or a
    /// setting not given: no rule applies to it.
    Other,
}

/// The rule of one setting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// A whole number.
    Count(Count),
    /// A number in an interval.
    Real(Real),
}

impl Rule {
    /// The name of the setting.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Count(rule) => rule.name(),
            Self::Real(rule) => rule.name(),
        }
    }

    /// Check the setting: a number, and each number of a list. A setting
    /// that is not a number ([`Setting::Other`]) is not checked.
    ///
    /// # Errors
    ///
    /// Returns the message of the rule for the first number outside the
    /// range.
    pub fn check(&self, setting: &Setting) -> Result<(), String> {
        match (setting, self) {
            (Setting::List(settings), _) => settings.iter().try_for_each(|item| self.check(item)),
            (setting, Self::Count(rule)) => rule.check_setting(setting),
            (setting, Self::Real(rule)) => rule.check_setting(setting),
        }
    }

    /// A value outside the range.
    pub fn outside(&self) -> Setting {
        match self {
            Self::Count(rule) => Setting::Whole(rule.outside()),
            Self::Real(rule) => Setting::Number(rule.outside()),
        }
    }
}

/// The rules of the settings of one object, such as a method of the Python
/// API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ranges {
    /// The object, as the Python API names its class.
    pub object: &'static str,
    /// The rule of each of its settings that has a range.
    pub rules: &'static [Rule],
}

impl Ranges {
    /// Check the settings of the object, each by its name. A setting that is
    /// not in `settings` is not checked.
    ///
    /// # Errors
    ///
    /// Returns the message of the first rule that a setting breaks.
    pub fn check(&self, settings: &[(impl AsRef<str>, Setting)]) -> Result<(), String> {
        self.rules.iter().try_for_each(|rule| {
            settings
                .iter()
                .filter(|(name, _)| name.as_ref() == rule.name())
                .try_for_each(|(_, setting)| rule.check(setting))
        })
    }
}

/// Check the settings of the object `object` of one of `tables`, as
/// [`Ranges::check`] does.
///
/// # Errors
///
/// Returns the message of [`Ranges::check`], or a message when no table has
/// rules for `object`.
pub fn check_object(
    tables: &[&[Ranges]],
    object: &str,
    settings: &[(impl AsRef<str>, Setting)],
) -> Result<(), String> {
    tables
        .iter()
        .flat_map(|table| table.iter())
        .find(|ranges| ranges.object == object)
        .ok_or_else(|| format!("No range rules for the object {object}"))?
        .check(settings)
}

/// The rules of the replicates of a simulation (`ReplicateRuns`) and of the
/// axis patterns of `commol.AxisPattern`, which build rate expressions.
pub const RANGES: &[Ranges] = &[
    Ranges {
        object: "ReplicateRuns",
        rules: &[
            Rule::Count(crate::REPLICATE_COUNT),
            Rule::Count(crate::SEED),
        ],
    },
    Ranges {
        object: "_PulseAxisPattern",
        rules: &[Rule::Count(Count::at_least("at", 0))],
    },
    Ranges {
        object: "_PulsesAxisPattern",
        rules: &[Rule::Count(Count::at_least("steps", 0))],
    },
    Ranges {
        object: "_PeriodicAxisPattern",
        rules: &[
            Rule::Count(Count::at_least("period", 1)),
            Rule::Count(Count::at_least("offset", 0)),
        ],
    },
    Ranges {
        object: "_IntervalAxisPattern",
        rules: &[
            Rule::Real(Real::finite("start")),
            Rule::Real(Real::finite("end")),
        ],
    },
    Ranges {
        object: "_GaussianPulseAxisPattern",
        rules: &[Rule::Real(Real::new(
            "width",
            Bound::Open(0.0),
            Bound::Closed(f64::INFINITY),
        ))],
    },
    Ranges {
        object: "_LinearRampAxisPattern",
        rules: &[
            Rule::Real(Real::finite("start")),
            Rule::Real(Real::finite("end")),
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_gives_one_message_for_every_value_outside_its_range() {
        let replicates = Count::at_least("replicates", 1);
        assert_eq!(replicates.check(1_usize), Ok(()));
        for value in [0_i64, -1, i64::MIN] {
            assert_eq!(
                replicates.check(value),
                Err(format!(
                    "replicates is a whole number of at least 1, got {value}"
                ))
            );
        }
        let subset = Count::at_least("max_subset_size", 2).at_most(3);
        assert_eq!(subset.check(3_usize), Ok(()));
        for value in [1_usize, 4] {
            assert_eq!(
                subset.check(value),
                Err(format!(
                    "max_subset_size is a whole number from 2 to 3, got {value}"
                ))
            );
        }
        let steps = Count::between("num_steps", 0, i128::from(u32::MAX));
        assert_eq!(steps.read::<u32>(7), Ok(7));
        assert_eq!(
            steps.read::<u32>(-1),
            Err("num_steps is a whole number from 0 to 4294967295, got -1".to_string())
        );
        assert_eq!(
            steps.check(u64::MAX),
            Err(format!(
                "num_steps is a whole number from 0 to 4294967295, got {}",
                u64::MAX
            ))
        );
    }

    #[test]
    fn a_real_names_its_interval() {
        let cases = [
            (Real::finite("x"), "x is a finite number, got inf"),
            (
                Real::positive("x"),
                "x is a finite number greater than 0, got 0",
            ),
            (
                Real::non_negative("x"),
                "x is a finite number of at least 0, got -1",
            ),
            (Real::share("x"), "x is strictly between 0 and 1, got 0"),
            (
                Real::new("x", Bound::Closed(0.0), Bound::Closed(1.0)),
                "x is a number from 0 to 1, got -1",
            ),
            (
                Real::new("x", Bound::Open(0.0), Bound::Closed(1.0)),
                "x is a number greater than 0 and at most 1, got 0",
            ),
            (
                Real::new("x", Bound::Open(1.0), Bound::Closed(f64::INFINITY)),
                "x is a number greater than 1, got 1",
            ),
            (
                Real::non_negative_or_infinite("x"),
                "x is a number of at least 0, got -1",
            ),
            (
                Real::positive_or_infinite("x").with_hint("use 1 for no weight"),
                "x is a number greater than 0, got 0; use 1 for no weight",
            ),
            (
                Real::new("x", Bound::Closed(0.0), Bound::Open(1.0)),
                "x is a number of at least 0 and less than 1, got -1",
            ),
        ];
        for (rule, message) in cases {
            assert_eq!(rule.check(rule.outside()), Err(message.to_string()));
            assert!(rule.check(f64::NAN).is_err());
        }
        assert_eq!(
            Real::new("x", Bound::Open(0.0), Bound::Closed(f64::INFINITY)).check(f64::INFINITY),
            Ok(())
        );
    }

    #[test]
    fn a_confidence_level_is_strictly_between_0_and_1() {
        assert_eq!(CONFIDENCE_LEVEL.check(0.95), Ok(()));
        for level in [0.0, 1.0, 1.5, f64::NAN] {
            assert_eq!(
                CONFIDENCE_LEVEL.check(level),
                Err(format!(
                    "confidence_level is strictly between 0 and 1, got {level}; \
                     use 0.95 for a 95% interval"
                ))
            );
        }
    }

    #[test]
    fn an_object_checks_each_number_of_its_settings() {
        const METHOD: Ranges = Ranges {
            object: "Method",
            rules: &[
                Rule::Count(Count::at_least("block_length", 1)),
                Rule::Real(Real::non_negative("at")),
            ],
        };
        let at = |settings: Vec<Setting>| [("at", Setting::List(settings))];
        assert_eq!(
            METHOD.check(&at(vec![Setting::Whole(0), Setting::Number(2.5)])),
            Ok(())
        );
        for setting in [
            Setting::Other,
            Setting::Whole(3),
            Setting::Whole(i128::from(u64::MAX)),
        ] {
            assert_eq!(METHOD.check(&[("block_length", setting.clone())]), Ok(()));
        }
        assert_eq!(METHOD.check(&[("other", Setting::Whole(-1))]), Ok(()));
        for (settings, message) in [
            (
                [("block_length", Setting::Whole(-1))],
                "block_length is a whole number of at least 1, got -1",
            ),
            (
                [("block_length", Setting::Number(1.5))],
                "block_length is a whole number of at least 1, got 1.5",
            ),
            (
                [(
                    "at",
                    Setting::List(vec![Setting::Whole(1), Setting::Number(f64::NAN)]),
                )],
                "at is a finite number of at least 0, got NaN",
            ),
            (
                [("at", Setting::Number(f64::INFINITY))],
                "at is a finite number of at least 0, got inf",
            ),
            (
                [("at", Setting::Whole(-1))],
                "at is a finite number of at least 0, got -1",
            ),
        ] {
            assert_eq!(METHOD.check(&settings), Err(message.to_string()));
        }
        let none: [(&str, Setting); 0] = [];
        assert_eq!(
            check_object(&[RANGES], "Unknown", &none),
            Err("No range rules for the object Unknown".to_string())
        );
        assert_eq!(
            check_object(
                &[RANGES],
                "_PeriodicAxisPattern",
                &[("period", Setting::Whole(0))]
            ),
            Err("period is a whole number of at least 1, got 0".to_string())
        );
    }
}
