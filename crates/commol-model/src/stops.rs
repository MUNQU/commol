//! Where an engine in continuous time stops: the points of a model that
//! [`same_position`] calls one are one stop.

use commol_core::same_position;

/// A position where an engine in continuous time stops, since the
/// right-hand side can jump there ([`crate::CompiledModel::breakpoints`]).
///
/// A stop holds the points of the model that are the same position as it.
/// When a rate reads `step`, every integer is a stop, and it holds every point
/// that is the same position as it. Another stop is the earliest of its
/// points, and it holds the later points that are the same position as it.
#[derive(Debug)]
pub struct Stop {
    /// Where an engine stops, applies the impulses and reports.
    pub at: f64,
    /// The earliest point of the stop. The rates of the piece before the stop
    /// see `t` below it, so a value that jumps at a point of the stop acts
    /// only from the stop.
    pub cut: f64,
    /// The latest point of the stop.
    last: f64,
    /// The impulses that act at the stop, each once, in the order of
    /// [`crate::CompiledModel::impulse_changes`].
    pub impulses: Vec<usize>,
}

impl Stop {
    /// A stop at `at` that holds no point yet.
    fn new(at: f64) -> Self {
        Self {
            at,
            cut: at,
            last: at,
            impulses: Vec::new(),
        }
    }

    /// Hold `point`, a point of `impulse` when it has one.
    fn hold(&mut self, point: f64, impulse: Option<usize>) {
        self.cut = self.cut.min(point);
        self.last = self.last.max(point);
        if let Some(index) = impulse
            && !self.impulses.contains(&index)
        {
            self.impulses.push(index);
        }
    }

    /// Whether `position` is the same position as a point of the stop.
    pub fn holds(&self, position: f64) -> bool {
        (self.cut..=self.last).contains(&position)
            || same_position(position, self.cut)
            || same_position(position, self.last)
    }
}

/// The stops of a run from a position to a position
/// ([`crate::CompiledModel::breakpoints`]).
#[derive(Debug)]
pub struct Breakpoints {
    /// Where the run starts.
    from: f64,
    /// The stop at `from`, when the start is one: when it holds a point of
    /// the model, or is a step. Its points are behind the run. A start that
    /// is not a stop, such as the end of a run that no stop held, takes no
    /// position from a later stop, so the position goes where the whole run
    /// puts it.
    start: Option<Stop>,
    /// The stops after the start, ascending, up to the end of the run.
    pub stops: Vec<Stop>,
    /// Where the run ends: the stop that holds the last position of the run,
    /// or that position when no stop holds it.
    pub end: f64,
    /// Whether a rate reads `step`, so that every integer is a stop.
    steps: bool,
}

impl Breakpoints {
    /// The stops of a run from `from` to `to`, from `points`, ascending,
    /// each with its impulse when it has one. When `steps` is true every
    /// integer is a stop too.
    ///
    /// `points` start at `from` and reach past `to`, so the stop that holds
    /// `to` holds all of its points. A point that is the same position as
    /// `from` is behind the run, as is a point of the integer `from` is.
    pub(crate) fn new(from: f64, to: f64, points: &[(f64, Option<usize>)], steps: bool) -> Self {
        let reach = points.last().map_or(to, |&(point, _)| point.max(to));
        let mut start = Stop::new(from);
        let mut start_is_stop = steps && same_position(from, nearest_step(from));
        let mut integers = Vec::new();
        if steps {
            let mut n = from.floor() + 1.0;
            while n <= reach + 1.0 {
                if !same_position(n, from) {
                    integers.push(Stop::new(n));
                }
                n += 1.0;
            }
        }
        let mut others: Vec<Stop> = Vec::new();
        for &(point, impulse) in points {
            let nearest = nearest_step(point);
            let stop = if steps && same_position(point, nearest) {
                match integers.binary_search_by(|stop| stop.at.total_cmp(&nearest)) {
                    Ok(index) => &mut integers[index],
                    Err(_) => &mut start,
                }
            } else {
                let anchor = others.last().map_or(from, |stop| stop.at);
                if !same_position(point, anchor) {
                    others.push(Stop::new(point));
                }
                match others.last_mut() {
                    Some(stop) => stop,
                    None => {
                        start_is_stop = true;
                        &mut start
                    }
                }
            };
            stop.hold(point, impulse);
        }
        let mut stops = integers;
        stops.append(&mut others);
        stops.sort_by(|a, b| a.at.total_cmp(&b.at));
        for stop in &mut stops {
            stop.impulses.sort_unstable();
        }
        let mut plan = Self {
            from,
            start: start_is_stop.then_some(start),
            stops,
            end: f64::INFINITY,
            steps,
        };
        plan.end = plan.place(to);
        let kept = plan.stops.partition_point(|stop| stop.at <= plan.end);
        plan.stops.truncate(kept);
        plan
    }

    /// Where an engine reports at `position`, never past the end of the run:
    /// at the start for a position up to it, at the step that is the same
    /// position as it when a rate reads `step`, at the start, when it is a
    /// stop, or at the stop that is the same position as it, or else at the
    /// first stop that holds it, or at the position itself.
    ///
    /// A position can be the same position as a point of one stop and as the
    /// next stop itself: it goes to the next stop, so a run that ends there
    /// keeps that stop. A step comes before any other stop, since `step`
    /// reads it there.
    pub fn place(&self, position: f64) -> f64 {
        if position <= self.from {
            return self.from;
        }
        let step = nearest_step(position);
        let start = self.start.as_ref();
        let placed = if self.steps && same_position(position, step) {
            // The start is a step, or it holds the step.
            if same_position(step, self.from) {
                self.from
            } else {
                step
            }
        } else if start.is_some_and(|start| same_position(position, start.at)) {
            self.from
        } else {
            // The stops hold ranges that follow one another along the axis.
            let index = self.stops.partition_point(|stop| {
                stop.last < position && !same_position(stop.last, position)
            });
            let near = &self.stops[index..self.stops.len().min(index + 2)];
            if let Some(stop) = near.iter().find(|stop| same_position(position, stop.at)) {
                stop.at
            } else if start.is_some_and(|start| start.holds(position)) {
                self.from
            } else {
                near.first()
                    .filter(|stop| stop.holds(position))
                    .map_or(position, |stop| stop.at)
            }
        };
        placed.min(self.end)
    }
}

/// The integer nearest `position`, with 0 as `+0.0`, so that a point just
/// below 0 finds the stop 0.
fn nearest_step(position: f64) -> f64 {
    position.round() + 0.0
}
