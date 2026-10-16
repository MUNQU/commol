//! The rates of a set of events, to pick the next event in `O(log n)`.

/// A binary tree of sums over the rates of `len` events.
///
/// Each leaf holds the rate of one event, and each inner node the sum of its
/// children, so the total, a change of one rate and the pick of the event
/// that holds a point of the total all cost `O(log n)`. An engine of events
/// keeps one tree and changes only the rates an event touches.
#[derive(Clone, Debug)]
pub struct SumTree {
    /// The number of events.
    len: usize,
    /// The number of leaves, a power of two of at least `len`.
    leaves: usize,
    /// The nodes: the root at 1, the children of `i` at `2i` and `2i + 1`,
    /// the leaves from `leaves`.
    nodes: Vec<f64>,
}

impl SumTree {
    /// A tree of `len` events, all of rate 0.
    pub fn new(len: usize) -> Self {
        let leaves = len.max(1).next_power_of_two();
        Self {
            len,
            leaves,
            nodes: vec![0.0; 2 * leaves],
        }
    }

    /// The number of events.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tree has no events.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The rate of event `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not an event of the tree.
    pub fn rate(&self, index: usize) -> f64 {
        assert!(index < self.len, "event {index} of {}", self.len);
        self.nodes[self.leaves + index]
    }

    /// The sum of every rate.
    pub fn total(&self) -> f64 {
        self.nodes[1]
    }

    /// Set the rate of event `index`, and the sums above it.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not an event of the tree, and in debug builds
    /// when the rate is negative or not finite.
    pub fn set(&mut self, index: usize, rate: f64) {
        assert!(index < self.len, "event {index} of {}", self.len);
        debug_assert!(rate.is_finite() && rate >= 0.0, "rate {rate}");
        let mut node = self.leaves + index;
        self.nodes[node] = rate;
        while node > 1 {
            node /= 2;
            self.nodes[node] = self.nodes[2 * node] + self.nodes[2 * node + 1];
        }
    }

    /// The event whose share of the total holds `point`, a number in
    /// `[0, total)`: the events take consecutive intervals, in index order,
    /// each as long as its rate.
    ///
    /// Returns `None` when the total is 0. A point at or past the total picks
    /// the last event of positive rate.
    pub fn find(&self, point: f64) -> Option<usize> {
        if self.total() <= 0.0 || self.total().is_nan() {
            return None;
        }
        let mut node = 1;
        let mut point = point.max(0.0);
        while node < self.leaves {
            let left = self.nodes[2 * node];
            let right = self.nodes[2 * node + 1];
            if point < left || right <= 0.0 {
                node *= 2;
            } else {
                point -= left;
                node = 2 * node + 1;
            }
        }
        Some(node - self.leaves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Generator;

    #[test]
    fn the_total_follows_every_change() {
        let mut tree = SumTree::new(5);
        for (index, rate) in [1.0, 2.0, 0.5, 0.0, 4.0].into_iter().enumerate() {
            tree.set(index, rate);
        }
        assert_eq!(tree.total(), 7.5);
        tree.set(1, 0.25);
        assert_eq!(tree.total(), 5.75);
        assert_eq!(tree.rate(1), 0.25);
        assert_eq!(tree.len(), 5);
    }

    #[test]
    fn a_point_picks_the_event_whose_interval_holds_it() {
        let mut tree = SumTree::new(4);
        for (index, rate) in [1.0, 0.0, 2.0, 1.0].into_iter().enumerate() {
            tree.set(index, rate);
        }
        assert_eq!(tree.find(0.0), Some(0));
        assert_eq!(tree.find(0.999), Some(0));
        assert_eq!(tree.find(1.0), Some(2));
        assert_eq!(tree.find(2.999), Some(2));
        assert_eq!(tree.find(3.0), Some(3));
        assert_eq!(tree.find(10.0), Some(3));
    }

    #[test]
    fn an_empty_total_picks_nothing() {
        let tree = SumTree::new(3);
        assert_eq!(tree.find(0.0), None);
        assert!(SumTree::new(0).is_empty());
        assert_eq!(SumTree::new(0).find(0.0), None);
    }

    #[test]
    fn events_are_picked_in_proportion_to_their_rates() {
        let rates = [0.5, 1.5, 3.0, 0.0, 5.0];
        let mut tree = SumTree::new(rates.len());
        for (index, rate) in rates.into_iter().enumerate() {
            tree.set(index, rate);
        }
        let mut generator = Generator::from_seed(1);
        let mut picks = [0usize; 5];
        let draws = 100_000;
        for _ in 0..draws {
            let event = tree.find(generator.uniform() * tree.total()).unwrap();
            picks[event] += 1;
        }
        for (picked, rate) in picks.iter().zip(rates) {
            let share = *picked as f64 / draws as f64;
            assert!((share - rate / 10.0).abs() < 0.005, "{share} for {rate}");
        }
    }
}
