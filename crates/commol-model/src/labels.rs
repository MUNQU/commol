//! Labels of the outputs of a compartmental model, and the sums they name.

/// Source and applied stratification categories of one simulation output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputLabels {
    /// Bin or accumulator the output expands from.
    pub source: String,
    /// `(stratification id, category)` pairs applied to the output, in
    /// stratification declaration order.
    pub categories: Vec<(String, String)>,
}

impl OutputLabels {
    /// Names the output answers to as a prefix: its source, then its source
    /// followed by each leading category (`A`, `A_g1`, `A_g1_s2`). The last
    /// one is the output's own name.
    pub fn prefixes(&self) -> Vec<String> {
        let mut prefix = self.source.clone();
        let mut prefixes = vec![prefix.clone()];
        for (_, category) in &self.categories {
            prefix.push('_');
            prefix.push_str(category);
            prefixes.push(prefix.clone());
        }
        prefixes
    }

    /// Category applied to the output for a stratification, if any.
    pub fn category(&self, stratification: &str) -> Option<&str> {
        self.categories
            .iter()
            .find(|(id, _)| id == stratification)
            .map(|(_, category)| category.as_str())
    }
}

/// Every name that sums several outputs, with the indices of the outputs it
/// sums, sorted by name.
///
/// A name is an output's source followed by any number of its leading
/// categories (`A`, `A_g1`), short of its full name. It is built from the
/// labels rather than by splitting names, so a bin `A` never covers the
/// outputs of a bin `A_x`. A name is kept only when two or more outputs share
/// it. This covers bin totals over every category and partial totals over the
/// categories left unnamed.
pub fn output_sums(labels: &[OutputLabels]) -> Vec<(String, Vec<usize>)> {
    let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (idx, label) in labels.iter().enumerate() {
        let mut prefixes = label.prefixes();
        prefixes.pop();
        for prefix in prefixes {
            map.entry(prefix).or_default().push(idx);
        }
    }
    let mut result: Vec<(String, Vec<usize>)> =
        map.into_iter().filter(|(_, v)| v.len() > 1).collect();
    result.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(source: &str, categories: &[(&str, &str)]) -> OutputLabels {
        OutputLabels {
            source: source.to_string(),
            categories: categories
                .iter()
                .map(|(id, category)| (id.to_string(), category.to_string()))
                .collect(),
        }
    }

    #[test]
    fn sums_follow_labels_not_name_prefixes() {
        let labels = vec![
            label("A", &[("group", "g1"), ("sub", "s1")]),
            label("A", &[("group", "g1"), ("sub", "s2")]),
            label("A", &[("group", "g2"), ("sub", "s1")]),
            label("A_x", &[("group", "g1"), ("sub", "s1")]),
            label("A_x", &[("group", "g1"), ("sub", "s2")]),
        ];

        let sums: std::collections::HashMap<String, Vec<usize>> =
            output_sums(&labels).into_iter().collect();

        assert_eq!(sums["A"], vec![0, 1, 2]);
        assert_eq!(sums["A_g1"], vec![0, 1]);
        assert_eq!(sums["A_x"], vec![3, 4]);
        assert_eq!(sums["A_x_g1"], vec![3, 4]);
        assert!(!sums.contains_key("A_g2"));
    }
}
