//! Quantities of interest: the scalars a method reads from each run.
//!
//! A quantity reads one output, or a sum of outputs, through the
//! [`ObservationOperator`], the one code that turns a run into observed
//! values: [`Quantity::ValueAt`] is an observation at a position,
//! [`Quantity::Gain`] an observation with a window, and [`Quantity::Peak`]
//! and [`Quantity::PeakPosition`] reduce the observations at their positions.
//! The quantities of a design are evaluated in Rust, run after run, so a
//! design of thousands of runs never returns to Python.

use std::ops::Range;

use commol_calibration::{ObservationOperator, ObservedDataPoint};

/// A scalar read from a run. `outputs` are summed, as the outputs of an
/// observation are.
#[derive(Debug, Clone, PartialEq)]
pub enum Quantity {
    /// The value at `at`.
    ValueAt {
        /// The outputs summed.
        outputs: Vec<String>,
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
    /// The largest value at `positions`.
    Peak {
        /// The outputs summed.
        outputs: Vec<String>,
        /// The positions compared.
        positions: Vec<f64>,
    },
    /// The first of `positions` where the value is largest.
    PeakPosition {
        /// The outputs summed.
        outputs: Vec<String>,
        /// The positions compared.
        positions: Vec<f64>,
    },
}

/// How a quantity reduces its observations.
#[derive(Debug, Clone)]
enum Reduction {
    /// The value of one observation.
    One(usize),
    /// The largest value of a range of observations.
    Max(Range<usize>),
    /// The position of the first largest value of a range of observations.
    ArgMax(Range<usize>, Vec<f64>),
}

/// The quantities of a design, compiled for runs at their schedule.
#[derive(Debug, Clone)]
pub struct Quantities {
    schedule: Vec<f64>,
    operator: ObservationOperator,
    reductions: Vec<Reduction>,
}

impl Quantities {
    /// Compile `quantities` for rows of `output_names`.
    ///
    /// # Errors
    ///
    /// Returns a message when there is no quantity, a quantity has no
    /// outputs or names an unknown one, a position is not finite, a gain
    /// does not end after it starts, or a peak has no positions.
    pub fn new(quantities: &[Quantity], output_names: &[String]) -> Result<Self, String> {
        if quantities.is_empty() {
            return Err("No quantities of interest given".to_string());
        }
        let mut points = Vec::new();
        let mut starts = Vec::new();
        let mut reductions = Vec::with_capacity(quantities.len());
        for (index, quantity) in quantities.iter().enumerate() {
            if outputs(quantity).is_empty() {
                return Err(format!("Quantity {index} has no outputs"));
            }
            let point = |at: f64, outputs: &[String]| {
                ObservedDataPoint::of_sum(at, format!("quantity {index}"), outputs.to_vec(), 0.0)
            };
            match quantity {
                Quantity::ValueAt { outputs, at } => {
                    check_position(index, *at)?;
                    reductions.push(Reduction::One(points.len()));
                    points.push(point(*at, outputs));
                    starts.push(None);
                }
                Quantity::Gain {
                    outputs,
                    start,
                    end,
                } => {
                    check_position(index, *start)?;
                    check_position(index, *end)?;
                    if end <= start {
                        return Err(format!(
                            "Quantity {index}: a gain ends after it starts, got start {start} \
                             and end {end}"
                        ));
                    }
                    reductions.push(Reduction::One(points.len()));
                    points.push(point(*end, outputs));
                    starts.push(Some(*start));
                }
                Quantity::Peak { outputs, positions }
                | Quantity::PeakPosition { outputs, positions } => {
                    if positions.is_empty() {
                        return Err(format!("Quantity {index}: a peak needs positions"));
                    }
                    for &at in positions {
                        check_position(index, at)?;
                    }
                    let range = points.len()..points.len() + positions.len();
                    points.extend(positions.iter().map(|&at| point(at, outputs)));
                    starts.resize(points.len(), None);
                    reductions.push(match quantity {
                        Quantity::Peak { .. } => Reduction::Max(range),
                        _ => Reduction::ArgMax(range, positions.clone()),
                    });
                }
            }
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
            reductions,
        })
    }

    /// The sorted, deduplicated positions a run must report at.
    pub fn schedule(&self) -> &[f64] {
        &self.schedule
    }

    /// The number of quantities.
    pub fn len(&self) -> usize {
        self.reductions.len()
    }

    /// Whether there is no quantity; never, for compiled quantities.
    pub fn is_empty(&self) -> bool {
        self.reductions.is_empty()
    }

    /// The value of each quantity in the rows of a run at [`Self::schedule`].
    pub fn evaluate(&self, rows: &[Vec<f64>]) -> Vec<f64> {
        let mut observed = vec![0.0; self.operator.count()];
        self.operator.apply(rows, &[], &mut observed);
        self.reductions
            .iter()
            .map(|reduction| match reduction {
                Reduction::One(index) => observed[*index],
                Reduction::Max(range) => observed[range.clone()]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
                Reduction::ArgMax(range, positions) => {
                    let values = &observed[range.clone()];
                    let first = values
                        .iter()
                        .enumerate()
                        .fold(0, |best, (i, &v)| if v > values[best] { i } else { best });
                    positions[first]
                }
            })
            .collect()
    }
}

fn outputs(quantity: &Quantity) -> &[String] {
    match quantity {
        Quantity::ValueAt { outputs, .. }
        | Quantity::Gain { outputs, .. }
        | Quantity::Peak { outputs, .. }
        | Quantity::PeakPosition { outputs, .. } => outputs,
    }
}

fn check_position(index: usize, at: f64) -> Result<(), String> {
    if at.is_finite() {
        Ok(())
    } else {
        Err(format!("Quantity {index}: a position is finite, got {at}"))
    }
}
