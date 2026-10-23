//! The series of a set of observations: the observations that share a
//! label.
//!
//! Every method that reads the observations series by series takes its
//! series from [`SeriesIndex`]: the labels in order of first appearance, and
//! the observations of each label in position order.

use std::collections::HashMap;

use crate::types::ObservedDataPoint;

/// The series of a set of entries, each named by a label.
///
/// The series are in order of the first appearance of their label. Each
/// holds the indices of its entries in position order, and entries at one
/// position keep their order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeriesIndex {
    labels: Vec<String>,
    indices: Vec<Vec<usize>>,
    series_of: Vec<usize>,
}

impl SeriesIndex {
    /// The series of `observations`, each in position order.
    pub fn new(observations: &[ObservedDataPoint]) -> Self {
        let mut index = Self::grouped(observations.iter().map(|obs| obs.label.as_str()));
        for indices in &mut index.indices {
            indices.sort_by(|&a, &b| observations[a].at.total_cmp(&observations[b].at));
        }
        index
    }

    /// The series of the entries that `labels` names, each in the order of
    /// `labels`: for values that have a label and no position.
    pub fn of_labels(labels: &[String]) -> Self {
        Self::grouped(labels.iter().map(String::as_str))
    }

    fn grouped<'a>(labels: impl Iterator<Item = &'a str>) -> Self {
        let mut slots: HashMap<&str, usize> = HashMap::new();
        let mut index = Self {
            labels: Vec::new(),
            indices: Vec::new(),
            series_of: Vec::new(),
        };
        for (entry, label) in labels.enumerate() {
            let slot = *slots.entry(label).or_insert_with(|| {
                index.labels.push(label.to_string());
                index.indices.push(Vec::new());
                index.indices.len() - 1
            });
            index.indices[slot].push(entry);
            index.series_of.push(slot);
        }
        index
    }

    /// The label of each series, in order of first appearance.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The indices of the entries of each series, as [`Self::labels`].
    pub fn indices(&self) -> &[Vec<usize>] {
        &self.indices
    }

    /// The series of each entry, as an index of [`Self::labels`].
    pub fn series_of(&self) -> &[usize] {
        &self.series_of
    }

    /// The index of the series `label`, if an entry has it.
    pub fn series(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|name| name == label)
    }

    /// The number of series.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether there is no series.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}
