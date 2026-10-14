//! Structured index of the compartments of a model.
//!
//! Compartments are identified by their bin and the categories applied to them,
//! never by parsing their names. Every part of the engine that needs to find a
//! compartment — transition targets, subpopulation totals, initial conditions —
//! goes through this index.

use std::collections::HashMap;

use commol_core::CompartmentalStructure;

use crate::labels::OutputLabels;

use crate::builder::expand_compartments;

/// Compartments of a model, in engine order, with their labels.
pub(crate) struct CompartmentLayout {
    names: Vec<String>,
    labels: Vec<OutputLabels>,
    index: HashMap<(String, Vec<(String, String)>), usize>,
}

impl CompartmentLayout {
    pub(crate) fn new(structure: &CompartmentalStructure) -> Self {
        let mut names = Vec::new();
        let mut labels = Vec::new();
        for output in expand_compartments(structure) {
            names.push(output.name);
            labels.push(OutputLabels {
                source: output.source,
                categories: output.categories,
            });
        }
        let index = labels
            .iter()
            .enumerate()
            .map(|(idx, label)| ((label.source.clone(), label.categories.clone()), idx))
            .collect();
        Self {
            names,
            labels,
            index,
        }
    }

    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }

    pub(crate) fn labels(&self) -> &[OutputLabels] {
        &self.labels
    }

    pub(crate) fn name(&self, idx: usize) -> &str {
        &self.names[idx]
    }

    /// Categories applied to a compartment, keyed by stratification id.
    pub(crate) fn applied(&self, idx: usize) -> HashMap<String, String> {
        self.labels[idx].categories.iter().cloned().collect()
    }

    /// Compartment of a bin with exactly these categories, in stratification
    /// declaration order.
    pub(crate) fn find(&self, bin: &str, categories: &[(String, String)]) -> Option<usize> {
        self.index
            .get(&(bin.to_string(), categories.to_vec()))
            .copied()
    }

    /// Indices of the compartments of a bin.
    pub(crate) fn of_bin<'a>(&'a self, bin: &'a str) -> impl Iterator<Item = usize> + 'a {
        self.labels
            .iter()
            .enumerate()
            .filter(move |(_, label)| label.source == bin)
            .map(|(idx, _)| idx)
    }

    /// Indices of the compartments that carry a category of a stratification.
    pub(crate) fn with_category<'a>(
        &'a self,
        stratification: &'a str,
        category: &'a str,
    ) -> impl Iterator<Item = usize> + 'a {
        self.labels
            .iter()
            .enumerate()
            .filter(move |(_, label)| label.category(stratification) == Some(category))
            .map(|(idx, _)| idx)
    }
}
