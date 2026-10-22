//! Quantities of interest: the scalars a method reads from each run.
//!
//! A quantity reads a [`Series`] of a run and reduces it. The series is the
//! level of some outputs (their sum at each position), or their increments
//! over windows of one width. The reduction is the value at a position
//! ([`Quantity::ValueAt`]), the largest value or its first position over
//! given positions ([`Quantity::Peak`], [`Quantity::PeakPosition`]), or the
//! value at each position ([`Quantity::Trajectory`]). [`Quantity::Gain`] is
//! one window given by its start and its end. [`Quantity::Ratio`] divides
//! two quantities of one value each.
//!
//! Every value is read through the [`ObservationOperator`], the one code
//! that turns a run into observed values: a level is an observation, an
//! increment an observation with a window. The rules:
//!
//! - a quantity reads only the positions it names, with no interpolation;
//! - outputs are summed at each row before any reduction;
//! - a peak position is the first of the positions where the value is
//!   largest;
//! - a NaN in the values a quantity reads makes it NaN, its peak and the
//!   position of its peak too;
//! - with a stochastic engine, a quantity is computed for each replicate;
//! - a ratio of a replicate whose denominator is 0 is NaN, also when its
//!   numerator is not 0.
//!
//! A method that reduces the points of a design reads one value per point:
//! [`Quantities::combine`] combines the replicates of the point, the mean of
//! each quantity, and for a ratio the rule its [`ReplicateRule`] names.
//!
//! A quantity crosses from Python as JSON, with `kind` as the tag of each
//! enum. The quantities of a design are evaluated in Rust, run after run, so
//! a design of thousands of runs never returns to Python. The quantities of
//! a run that is already done ([`Quantities::evaluate_run`]) read its rows
//! at the positions of the schedule, found with the tolerance of the axis
//! ([`index_at`]), and reduce them with the same code.

use std::ops::Range;

use commol_calibration::{ObservationOperator, ObservedDataPoint, index_at};
use serde::{Deserialize, Serialize};

/// What a quantity reads at each position of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Series {
    /// The sum of `outputs` at the position.
    Level {
        /// The outputs summed.
        outputs: Vec<String>,
    },
    /// The gain of the sum of `outputs` over `(end − window, end]`, at each
    /// end: an observation with the window `window`.
    Increments {
        /// The outputs summed.
        outputs: Vec<String>,
        /// The width of each window, above 0.
        window: f64,
    },
}

impl Series {
    /// The outputs summed.
    pub fn outputs(&self) -> &[String] {
        match self {
            Self::Level { outputs } | Self::Increments { outputs, .. } => outputs,
        }
    }
}

/// A series of a run and how it is reduced to scalars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Quantity {
    /// The value of `series` at `at`.
    ValueAt {
        /// The series read.
        series: Series,
        /// The position.
        at: f64,
    },
    /// The value at `end` minus the value at `start`, as an observation with
    /// the window `end − start` that starts at exactly `start`: what entered
    /// a cumulative output over `(start, end]`.
    Gain {
        /// The outputs summed.
        outputs: Vec<String>,
        /// The start of the window.
        start: f64,
        /// The end of the window.
        end: f64,
    },
    /// The largest value of `series` at `positions`.
    Peak {
        /// The series read.
        series: Series,
        /// The positions compared.
        positions: Vec<f64>,
    },
    /// The first of `positions` where the value of `series` is largest.
    PeakPosition {
        /// The series read.
        series: Series,
        /// The positions compared.
        positions: Vec<f64>,
    },
    /// The value of `series` at each of `positions`: one scalar per
    /// position.
    Trajectory {
        /// The series read.
        series: Series,
        /// The positions.
        positions: Vec<f64>,
    },
    /// `numerator` divided by `denominator`, two quantities of one value
    /// each ([`Quantity::ValueAt`], [`Quantity::Gain`], [`Quantity::Peak`]).
    /// The ratio of a run whose denominator is 0 is NaN.
    Ratio {
        /// The quantity divided.
        numerator: Box<Quantity>,
        /// The quantity it is divided by.
        denominator: Box<Quantity>,
        /// How the replicates of a stochastic run combine.
        replicate_rule: ReplicateRule,
    },
}

/// How the replicates of a point combine into the value of a ratio of
/// numerator N and denominator D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicateRule {
    /// The mean of the ratio of each replicate, `E[N/D]`: a replicate with
    /// D = 0 leaves the point with no value.
    Within,
    /// The mean of the ratio of the replicates with D > 0, `E[N/D | D > 0]`: a
    /// point with no such replicate has no value.
    WithinPositive,
    /// The ratio of the means, `E[N]/E[D]`: a point with `E[D] = 0` has no
    /// value.
    Pooled,
}

/// The numerator and the denominator of a scalar of one run: the two
/// quantities of a ratio, or the value of any other scalar over 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Term {
    /// The numerator, or the value of a scalar that is not a ratio.
    pub numerator: f64,
    /// The denominator; 1 for a scalar that is not a ratio.
    pub denominator: f64,
}

impl Term {
    /// The term of a scalar that is not a ratio: `value` over 1.
    fn of(value: f64) -> Self {
        Self {
            numerator: value,
            denominator: 1.0,
        }
    }

    /// The value of the scalar in its run: the numerator over the
    /// denominator, NaN when the denominator is 0.
    pub fn value(self) -> f64 {
        if self.denominator == 0.0 {
            f64::NAN
        } else {
            self.numerator / self.denominator
        }
    }
}

/// Why the replicates of a point give a scalar no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Undefined {
    /// The value of this replicate is NaN, or one of the terms of its ratio
    /// is.
    Replicate(usize),
    /// No replicate is NaN, but a ratio has no value at the point: no
    /// replicate has D > 0 under [`ReplicateRule::WithinPositive`], or `E[D] = 0`
    /// under [`ReplicateRule::Pooled`].
    Point,
}

impl Quantity {
    /// The number of scalars the quantity gives: one per position of a
    /// trajectory, one for every other quantity.
    pub fn len(&self) -> usize {
        match self {
            Self::Trajectory { positions, .. } => positions.len(),
            _ => 1,
        }
    }

    /// Whether the quantity gives no scalar: a trajectory of no positions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The outputs the quantity sums; for a ratio, those of its numerator,
    /// then those of its denominator.
    pub fn outputs(&self) -> Vec<&str> {
        match self {
            Self::Gain { outputs, .. } => outputs.iter().map(String::as_str).collect(),
            Self::ValueAt { series, .. }
            | Self::Peak { series, .. }
            | Self::PeakPosition { series, .. }
            | Self::Trajectory { series, .. } => {
                series.outputs().iter().map(String::as_str).collect()
            }
            Self::Ratio {
                numerator,
                denominator,
                ..
            } => {
                let mut outputs = numerator.outputs();
                outputs.extend(denominator.outputs());
                outputs
            }
        }
    }
}

/// The positions of the schedule that a run lacks, for one quantity that
/// reads them.
#[derive(Debug, Clone, PartialEq)]
pub struct MissingPositions {
    /// The index of the quantity, in the order it was given.
    pub quantity: usize,
    /// The positions it reads that the run lacks, ascending.
    pub positions: Vec<f64>,
}

/// How a quantity of one value reduces its observations.
#[derive(Debug, Clone)]
enum Scalar {
    /// The value of one observation.
    One(usize),
    /// The largest value of a range of observations.
    Max(Range<usize>),
    /// The position of the first largest value of a range of observations.
    ArgMax(Range<usize>, Vec<f64>),
}

impl Scalar {
    /// The value in the observations of a run.
    fn value(&self, observed: &[f64]) -> f64 {
        match self {
            Self::One(index) => observed[*index],
            Self::Max(range) => {
                first_largest(&observed[range.clone()]).map_or(f64::NAN, |(_, v)| v)
            }
            Self::ArgMax(range, positions) => {
                first_largest(&observed[range.clone()]).map_or(f64::NAN, |(i, _)| positions[i])
            }
        }
    }
}

/// How a quantity reduces its observations.
#[derive(Debug, Clone)]
enum Reduction {
    /// One value.
    Scalar(Scalar),
    /// The value of each observation of a range.
    Each(Range<usize>),
    /// The value of one quantity over the value of another.
    Ratio(Scalar, Scalar),
}

/// The quantities of a design, compiled for runs at their schedule.
#[derive(Debug, Clone)]
pub struct Quantities {
    schedule: Vec<f64>,
    operator: ObservationOperator,
    reductions: Vec<Reduction>,
    /// How the replicates of each scalar combine: [`ReplicateRule::Within`]
    /// for a scalar that is not a ratio, whose denominator is 1.
    rules: Vec<ReplicateRule>,
    count: usize,
    /// The positions each quantity reads, ascending: its positions and the
    /// starts of its windows.
    reads: Vec<Vec<f64>>,
}

/// The observations of a quantity, one per position, and the start of the
/// window of each.
struct Points<'a> {
    index: usize,
    points: &'a mut Vec<ObservedDataPoint>,
    starts: &'a mut Vec<Option<f64>>,
}

impl Points<'_> {
    /// The observations of `series` at `positions`, as a range of the
    /// observations of the design.
    fn read(&mut self, series: &Series, positions: &[f64]) -> Result<Range<usize>, String> {
        let index = self.index;
        if series.outputs().is_empty() {
            return Err(format!("Quantity {index} has no outputs"));
        }
        let window = match series {
            Series::Level { .. } => None,
            Series::Increments { window, .. } => {
                if !(window.is_finite() && *window > 0.0) {
                    return Err(format!(
                        "Quantity {index}: the window of increments is finite and above 0, got \
                         {window}"
                    ));
                }
                Some(*window)
            }
        };
        let first = self.points.len();
        for &at in positions {
            check_position(index, at)?;
            if let Some(window) = window
                && at < window
            {
                return Err(format!(
                    "Quantity {index}: increments over windows of {window} start at 0 at the \
                     earliest, so each end is at least {window}; got the end {at}"
                ));
            }
            self.points.push(ObservedDataPoint::of_sum(
                at,
                format!("quantity {index}"),
                series.outputs().to_vec(),
                0.0,
            ));
            self.starts.push(window.map(|window| at - window));
        }
        Ok(first..self.points.len())
    }

    /// The reduction of `quantity`, its observations read.
    fn reduction(&mut self, quantity: &Quantity) -> Result<Reduction, String> {
        let index = self.index;
        Ok(match quantity {
            Quantity::ValueAt { series, at } => {
                Reduction::Scalar(Scalar::One(self.read(series, &[*at])?.start))
            }
            Quantity::Gain {
                outputs,
                start,
                end,
            } => {
                check_position(index, *start)?;
                if end <= start {
                    return Err(format!(
                        "Quantity {index}: a gain ends after it starts, got start {start} and \
                         end {end}"
                    ));
                }
                let series = Series::Level {
                    outputs: outputs.clone(),
                };
                let row = self.read(&series, &[*end])?.start;
                self.starts[row] = Some(*start);
                Reduction::Scalar(Scalar::One(row))
            }
            Quantity::Peak { series, positions }
            | Quantity::PeakPosition { series, positions }
            | Quantity::Trajectory { series, positions } => {
                if positions.is_empty() {
                    return Err(format!(
                        "Quantity {index}: a peak or a trajectory needs positions"
                    ));
                }
                let range = self.read(series, positions)?;
                match quantity {
                    Quantity::Peak { .. } => Reduction::Scalar(Scalar::Max(range)),
                    Quantity::PeakPosition { .. } => {
                        Reduction::Scalar(Scalar::ArgMax(range, positions.clone()))
                    }
                    _ => Reduction::Each(range),
                }
            }
            Quantity::Ratio {
                numerator,
                denominator,
                ..
            } => Reduction::Ratio(self.operand(numerator)?, self.operand(denominator)?),
        })
    }

    /// The reduction of an operand of a ratio: a value, a gain or a peak.
    fn operand(&mut self, quantity: &Quantity) -> Result<Scalar, String> {
        match (quantity, self.reduction(quantity)?) {
            (
                Quantity::ValueAt { .. } | Quantity::Gain { .. } | Quantity::Peak { .. },
                Reduction::Scalar(scalar),
            ) => Ok(scalar),
            _ => {
                let kind = match quantity {
                    Quantity::PeakPosition { .. } => "peak_position",
                    Quantity::Trajectory { .. } => "trajectory",
                    _ => "ratio",
                };
                Err(format!(
                    "Quantity {}: a ratio divides quantities of one value each (value_at, gain, \
                     peak), got {kind}",
                    self.index
                ))
            }
        }
    }
}

impl Quantities {
    /// Compile `quantities` for rows of `output_names`.
    ///
    /// # Errors
    ///
    /// Returns a message when there is no quantity, a quantity has no
    /// outputs or names an unknown one, a position is not finite, a gain
    /// does not end after it starts, the window of increments is not finite
    /// and above 0 or longer than an end, or a peak or a trajectory has no
    /// positions.
    pub fn new(quantities: &[Quantity], output_names: &[String]) -> Result<Self, String> {
        if quantities.is_empty() {
            return Err("No quantities of interest given".to_string());
        }
        let mut points = Vec::new();
        let mut starts = Vec::new();
        let mut reductions = Vec::with_capacity(quantities.len());
        let mut reads = Vec::with_capacity(quantities.len());
        let mut rules = Vec::with_capacity(quantities.len());
        for (index, quantity) in quantities.iter().enumerate() {
            let first = points.len();
            let mut read = Points {
                index,
                points: &mut points,
                starts: &mut starts,
            };
            reductions.push(read.reduction(quantity)?);
            let rule = match quantity {
                Quantity::Ratio { replicate_rule, .. } => *replicate_rule,
                _ => ReplicateRule::Within,
            };
            rules.extend(std::iter::repeat_n(rule, quantity.len()));
            let mut positions: Vec<f64> = points[first..]
                .iter()
                .map(|point| point.at)
                .chain(starts[first..].iter().flatten().copied())
                .collect();
            positions.sort_by(f64::total_cmp);
            positions.dedup();
            reads.push(positions);
        }
        let mut schedule: Vec<f64> = points
            .iter()
            .map(|point| point.at)
            .chain(starts.iter().flatten().copied())
            .collect();
        schedule.sort_by(f64::total_cmp);
        schedule.dedup();
        let operator =
            ObservationOperator::with_starts(&points, &starts, &[], output_names, &schedule)?;
        Ok(Self {
            schedule,
            operator,
            count: quantities.iter().map(Quantity::len).sum(),
            reductions,
            rules,
            reads,
        })
    }

    /// The schedule of `quantities` alone: the sorted, deduplicated
    /// positions a run must report at, compiled for the outputs the
    /// quantities name.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], but for an unknown output.
    pub fn schedule_of(quantities: &[Quantity]) -> Result<Vec<f64>, String> {
        let mut outputs: Vec<String> = quantities
            .iter()
            .flat_map(Quantity::outputs)
            .map(str::to_string)
            .collect();
        outputs.sort();
        outputs.dedup();
        Ok(Self::new(quantities, &outputs)?.schedule)
    }

    /// The sorted, deduplicated positions a run must report at.
    pub fn schedule(&self) -> &[f64] {
        &self.schedule
    }

    /// The number of scalars: one per quantity, and one per position of a
    /// trajectory.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether there is no scalar; never, for compiled quantities.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The value of each scalar of a run at `positions`, ascending, with one
    /// row of `rows` per position: the rows at the positions of
    /// [`Self::schedule`], each found within the tolerance of the axis
    /// ([`index_at`]), reduced by [`Self::evaluate`]. The run may hold more
    /// positions than the schedule.
    ///
    /// # Errors
    ///
    /// Every quantity that reads a position the run lacks, with those
    /// positions, in the order of the quantities.
    ///
    /// # Panics
    ///
    /// Panics when `rows` does not hold one row per position.
    pub fn evaluate_run(
        &self,
        positions: &[f64],
        rows: &[Vec<f64>],
    ) -> Result<Vec<f64>, Vec<MissingPositions>> {
        assert_eq!(rows.len(), positions.len(), "one row per position");
        let missing: Vec<MissingPositions> = self
            .reads
            .iter()
            .enumerate()
            .filter_map(|(quantity, reads)| {
                let positions: Vec<f64> = reads
                    .iter()
                    .copied()
                    .filter(|&at| index_at(positions, at).is_none())
                    .collect();
                (!positions.is_empty()).then_some(MissingPositions {
                    quantity,
                    positions,
                })
            })
            .collect();
        if !missing.is_empty() {
            return Err(missing);
        }
        let selected: Vec<Vec<f64>> = self
            .schedule
            .iter()
            .map(|&at| {
                let index = index_at(positions, at).expect("every read position was found");
                rows[index].clone()
            })
            .collect();
        Ok(self.evaluate(&selected))
    }

    /// The value of each scalar in the rows of a run at [`Self::schedule`]:
    /// the [`Term::value`] of each of its [`Self::terms`].
    pub fn evaluate(&self, rows: &[Vec<f64>]) -> Vec<f64> {
        self.terms(rows).into_iter().map(Term::value).collect()
    }

    /// The numerator and the denominator of each scalar in the rows of a
    /// run at [`Self::schedule`].
    pub fn terms(&self, rows: &[Vec<f64>]) -> Vec<Term> {
        let mut observed = vec![0.0; self.operator.count()];
        self.operator.apply(rows, &[], &mut observed);
        let mut terms = Vec::with_capacity(self.count);
        for reduction in &self.reductions {
            match reduction {
                Reduction::Scalar(scalar) => terms.push(Term::of(scalar.value(&observed))),
                Reduction::Each(range) => {
                    terms.extend(observed[range.clone()].iter().copied().map(Term::of));
                }
                Reduction::Ratio(numerator, denominator) => terms.push(Term {
                    numerator: numerator.value(&observed),
                    denominator: denominator.value(&observed),
                }),
            }
        }
        terms
    }

    /// The value of each scalar at a point, from the [`Self::terms`] of
    /// each of its replicates, `[replicate][scalar]`: the mean over the
    /// replicates, and for a ratio the rule of its [`ReplicateRule`].
    ///
    /// # Errors
    ///
    /// Each scalar with no value at the point holds why: the first
    /// replicate that is NaN, or a ratio whose rule gives no value.
    pub fn combine(&self, replicates: &[Vec<Term>]) -> Vec<Result<f64, Undefined>> {
        self.rules
            .iter()
            .enumerate()
            .map(|(scalar, rule)| {
                let terms = replicates.iter().map(move |terms| terms[scalar]);
                combine(terms, *rule)
            })
            .collect()
    }
}

/// The value of a scalar from its term in each replicate, combined by
/// `rule`.
fn combine(
    terms: impl Iterator<Item = Term> + Clone,
    rule: ReplicateRule,
) -> Result<f64, Undefined> {
    let nan = |(replicate, term): (usize, Term)| {
        match rule {
            ReplicateRule::Within => term.value().is_nan(),
            ReplicateRule::WithinPositive | ReplicateRule::Pooled => {
                term.numerator.is_nan() || term.denominator.is_nan()
            }
        }
        .then_some(Undefined::Replicate(replicate))
    };
    if let Some(undefined) = terms.clone().enumerate().find_map(nan) {
        return Err(undefined);
    }
    let mean = |values: &[f64]| {
        let count = values.len() as f64;
        values.iter().map(|value| value / count).sum::<f64>()
    };
    let value = match rule {
        ReplicateRule::Within => mean(&terms.map(Term::value).collect::<Vec<_>>()),
        ReplicateRule::WithinPositive => {
            let ratios: Vec<f64> = terms
                .filter(|term| term.denominator > 0.0)
                .map(Term::value)
                .collect();
            if ratios.is_empty() {
                return Err(Undefined::Point);
            }
            mean(&ratios)
        }
        ReplicateRule::Pooled => {
            let (numerators, denominators): (Vec<f64>, Vec<f64>) =
                terms.map(|term| (term.numerator, term.denominator)).unzip();
            Term {
                numerator: mean(&numerators),
                denominator: mean(&denominators),
            }
            .value()
        }
    };
    if value.is_nan() {
        Err(Undefined::Point)
    } else {
        Ok(value)
    }
}

/// The index and the value of the first largest of `values`, or None when
/// one is NaN.
fn first_largest(values: &[f64]) -> Option<(usize, f64)> {
    let mut best = (0, values[0]);
    for (i, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return None;
        }
        if value > best.1 {
            best = (i, value);
        }
    }
    Some(best)
}

fn check_position(index: usize, at: f64) -> Result<(), String> {
    if at.is_finite() {
        Ok(())
    } else {
        Err(format!("Quantity {index}: a position is finite, got {at}"))
    }
}
