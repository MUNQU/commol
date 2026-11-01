//! The steps of a long method, as events that a loop sends when a stage
//! begins and when a step of it finishes.
//!
//! A loop of a method sends each event to a [`Steps`] it is given, from the
//! thread that ran the step. The events of a loop are an enum of the crate
//! that owns the loop, such as the starts of a calibration or the
//! replicates of a bootstrap; an event borrows the outcome of its step, so
//! a loop that sends one clones nothing. The receiver decides what an event
//! means to the application: the Python extension writes a log record of
//! each, and [`NoSteps`] drops them.
//!
//! A loop holds its receiver as `&(dyn for<'e> Steps<Event<'e>> + '_)`, so
//! it can send an event that borrows a value of its own closure.
//!
//! The receiver also tells a loop to stop ([`Steps::stopped`]): a loop
//! reads it before each step, and once it is set starts no step. A step
//! that sees it ends at once with the error of its method, since whoever
//! stopped the job drops its result.

/// The reason of a step that a stop kept from running.
pub const STOPPED: &str = "Stopped before it ran";

/// The receiver of the events `E` of the loops of a method.
///
/// The loops run in parallel, so a receiver is shared between threads and
/// takes the events of many steps at once, in the order they finish.
pub trait Steps<E>: Sync {
    /// Take `event`. The default drops it.
    fn send(&self, _event: E) {}

    /// Whether the loops stop: once it is `true`, a loop starts no step.
    /// The default never stops.
    fn stopped(&self) -> bool {
        false
    }
}

/// The receiver that drops every event: the receiver of a loop that nobody
/// follows.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSteps;

impl<E> Steps<E> for NoSteps {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The events of a loop of numbers, kept in the order they come.
    struct Kept(Mutex<Vec<usize>>);

    impl Steps<&usize> for Kept {
        fn send(&self, event: &usize) {
            self.0.lock().unwrap().push(*event);
        }
    }

    /// Send each of `values` to `steps`, as a loop does.
    fn send_all(steps: &(dyn for<'e> Steps<&'e usize> + '_), values: &[usize]) {
        for value in values {
            let local = *value;
            steps.send(&local);
        }
    }

    #[test]
    fn a_receiver_takes_events_that_borrow_a_value_of_the_loop() {
        let kept = Kept(Mutex::new(Vec::new()));
        send_all(&kept, &[3, 1, 2]);
        assert_eq!(kept.0.into_inner().unwrap(), vec![3, 1, 2]);
    }

    #[test]
    fn no_steps_drops_every_event_and_never_stops() {
        send_all(&NoSteps, &[3, 1, 2]);
        assert!(!Steps::<&usize>::stopped(&NoSteps));
    }
}
