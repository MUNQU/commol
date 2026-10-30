//! The process of a model: what changes and how, written once for every
//! structure.
//!
//! A [`Process`] holds the bins, the accumulators, the stratifications, the
//! matrices between the categories of a stratification and the
//! transitions. The structure of the model
//! ([`ModelStructure`](super::ModelStructure)) says how the process is laid
//! out, such as amounts per compartment ([`Aggregate`](super::Aggregate)).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::transitions::{StratificationCondition, Transition};

/// Prefix of the rate variables that read the neighbors of a unit:
/// `neighbors_<bin>` is the amount of a bin among them.
pub const NEIGHBORS_PREFIX: &str = "neighbors_";

/// The rate variable that reads the total of the neighbors of a unit.
pub const NEIGHBORS: &str = "neighbors";

/// A state of the model. Stratifications expand it into compartments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bin {
    /// Identifier used in rates, initial conditions and outputs.
    pub id: String,
    /// Display name.
    pub name: String,
}

/// A cumulative event counter tracked by the engine but excluded from the total.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Accumulator {
    /// Identifier, and the output name before expansion.
    pub id: String,
    /// Display name.
    pub name: String,
}

/// A stratification dimension with its categories
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stratification {
    /// Identifier used in conditions and initial conditions.
    pub id: String,
    /// Categories, in the order their compartments are expanded.
    pub categories: Vec<String>,
    /// Conditions that must be satisfied (by already-applied stratifications) for
    /// this stratification to expand a compartment. When `None`, it always applies
    /// (standard full Cartesian product). When set, only compartments whose
    /// already-applied categories satisfy ALL conditions are further expanded;
    /// others are kept as-is without this stratification's categories appended.
    ///
    /// Conditions may only reference stratifications declared before this one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditions: Option<Vec<StratificationCondition>>,
}

/// The suffix of the variable that reads the spectral radius of a matrix:
/// `<M>_radius`.
pub const MATRIX_RADIUS_SUFFIX: &str = "_radius";

/// A matrix between the categories of a stratification: one value for each
/// pair of its categories, and a parameter that may scale each row and each
/// column.
///
/// A rate and a calibration constraint read the scaled matrix
/// `diag(a) · values · diag(b)`, where `a_i` is the value of the parameter
/// that scales row `i` and `b_j` of the one that scales column `j` (1 for a
/// row or a column without one). A rate reads it through
/// `matrix_sum(M, e)`: in a compartment of category `i`,
/// `Σ_j M[i, j] · e_j`. A rate and a constraint read the entry `M[i, j]` as
/// the variable `<M>_<i>_<j>` ([`Self::entry_name`]) and the spectral
/// radius, the largest modulus of the eigenvalues, as `<M>_radius`
/// ([`Self::radius_name`]); [`crate::ScaledMatrix::variables`] computes
/// them at the values of the scales.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CategoryMatrix {
    /// Identifier used in `matrix_sum` and in the names of its variables.
    pub id: String,
    /// Id of the stratification whose categories are its rows and columns.
    pub stratification: String,
    /// `values[i][j]`, the value of row `i` and column `j`, by category.
    pub values: IndexMap<String, IndexMap<String, f64>>,
    /// The id of the parameter that multiplies the row of a category, by
    /// category; a category without one keeps its row.
    #[serde(default)]
    pub row_scales: IndexMap<String, String>,
    /// The id of the parameter that multiplies the column of a category, by
    /// category; a category without one keeps its column.
    #[serde(default)]
    pub column_scales: IndexMap<String, String>,
}

impl CategoryMatrix {
    /// Each scale, as `("row" or "column", category, parameter id)`: the
    /// rows first, then the columns, each in the order the matrix gives
    /// them.
    pub fn scales(&self) -> impl Iterator<Item = (&'static str, &str, &str)> {
        fn of<'a>(
            side: &'static str,
            scales: &'a IndexMap<String, String>,
        ) -> impl Iterator<Item = (&'static str, &'a str, &'a str)> {
            scales
                .iter()
                .map(move |(category, parameter)| (side, category.as_str(), parameter.as_str()))
        }
        of("row", &self.row_scales).chain(of("column", &self.column_scales))
    }

    /// The value of row `row` and column `column`; 0 when the matrix has
    /// none, which [`Self::check`] rejects.
    pub fn value(&self, row: &str, column: &str) -> f64 {
        self.values
            .get(row)
            .and_then(|values| values.get(column))
            .copied()
            .unwrap_or(0.0)
    }

    /// The name of the variable that reads the entry of row `row` and
    /// column `column`: `<M>_<row>_<column>`.
    pub fn entry_name(&self, row: &str, column: &str) -> String {
        format!("{}_{row}_{column}", self.id)
    }

    /// The name of the variable that reads the spectral radius:
    /// `<M>_radius`.
    pub fn radius_name(&self) -> String {
        format!("{}{MATRIX_RADIUS_SUFFIX}", self.id)
    }

    /// The categories of its stratification in `process`, in their order;
    /// empty when the process does not have the stratification.
    pub fn categories<'a>(&self, process: &'a Process) -> &'a [String] {
        process
            .stratifications
            .iter()
            .find(|stratification| stratification.id == self.stratification)
            .map(|stratification| stratification.categories.as_slice())
            .unwrap_or_default()
    }

    /// The values row by row, in the order of the categories of its
    /// stratification in `process`.
    pub fn rows(&self, process: &Process) -> Vec<f64> {
        let categories = self.categories(process);
        categories
            .iter()
            .flat_map(|row| categories.iter().map(move |column| self.value(row, column)))
            .collect()
    }

    /// Check the matrix against the process.
    ///
    /// # Errors
    ///
    /// Returns a message that names the matrix when its stratification is
    /// not one of the process, and that also names the category when a row
    /// or a value of a category is missing, a row, a column or a scale names
    /// no category of the stratification, or a value is not finite. The
    /// parameter of a scale is checked against the model
    /// (`check_matrices` of `commol-model`).
    pub fn check(&self, process: &Process) -> Result<(), String> {
        let id = &self.id;
        let stratification = &self.stratification;
        if !process
            .stratifications
            .iter()
            .any(|candidate| &candidate.id == stratification)
        {
            let ids: Vec<&str> = process
                .stratifications
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect();
            return Err(format!(
                "Matrix '{id}' is between the categories of '{stratification}', which the \
                 process does not have; its stratifications are {ids:?}"
            ));
        }
        let categories = self.categories(process);
        let foreign = |name: &String| {
            format!(
                "Matrix '{id}' names '{name}', which is not a category of \
                 '{stratification}' ({categories:?})"
            )
        };
        for (row, values) in &self.values {
            if !categories.contains(row) {
                return Err(foreign(row));
            }
            if let Some(column) = values.keys().find(|column| !categories.contains(column)) {
                return Err(foreign(column));
            }
        }
        if let Some((side, category, parameter)) = self
            .scales()
            .find(|(_, category, _)| !categories.iter().any(|known| known == category))
        {
            return Err(format!(
                "Matrix '{id}' scales the {side} of '{category}' by '{parameter}', but \
                 '{category}' is not a category of '{stratification}' ({categories:?})"
            ));
        }
        for row in categories {
            let Some(values) = self.values.get(row) else {
                return Err(format!(
                    "Matrix '{id}' has no row for the category '{row}' of '{stratification}'; \
                     it needs one value for each pair of its categories ({categories:?})"
                ));
            };
            for column in categories {
                match values.get(column) {
                    None => {
                        return Err(format!(
                            "Matrix '{id}' has no value in row '{row}' for the category \
                             '{column}' of '{stratification}'; it needs one value for each \
                             pair of its categories ({categories:?})"
                        ));
                    }
                    Some(value) if !value.is_finite() => {
                        return Err(format!(
                            "Matrix '{id}' has {value} in row '{row}' and column '{column}'; \
                             each value must be a finite number"
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(())
    }
}

/// What changes in a model and how: bins, accumulators, stratifications,
/// matrices and transitions, written once for every structure.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Process {
    /// Bins, in declaration order.
    #[serde(default)]
    pub bins: Vec<Bin>,
    /// Accumulators, in declaration order.
    #[serde(default)]
    pub accumulators: Vec<Accumulator>,
    /// Stratifications, in declaration order. The order defines the expansion.
    #[serde(default)]
    pub stratifications: Vec<Stratification>,
    /// Matrices between the categories of a stratification, in declaration
    /// order.
    #[serde(default)]
    pub matrices: Vec<CategoryMatrix>,
    /// Transitions, in declaration order.
    #[serde(default)]
    pub transitions: Vec<Transition>,
}

impl Process {
    /// Whether the process holds nothing.
    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
            && self.accumulators.is_empty()
            && self.stratifications.is_empty()
            && self.matrices.is_empty()
            && self.transitions.is_empty()
    }

    /// The matrix with the id `id`.
    pub fn matrix(&self, id: &str) -> Option<&CategoryMatrix> {
        self.matrices.iter().find(|matrix| matrix.id == id)
    }

    /// Check the conditions of each stratification and of each stratified
    /// rate ([`StratificationCondition::check`]). A condition of a
    /// stratification names a stratification declared before it, and a
    /// condition of a stratified rate one of the process.
    ///
    /// # Errors
    ///
    /// Returns the message of the first condition that does not fit the
    /// process, which names the stratification or the transition that holds
    /// it.
    pub fn check_conditions(&self) -> Result<(), String> {
        let stratifications = &self.stratifications;
        for (position, stratification) in stratifications.iter().enumerate() {
            let owner = format!("Stratification '{}'", stratification.id);
            for condition in stratification.conditions.iter().flatten() {
                let earlier = &stratifications[..position];
                let Some(named) = earlier.iter().find(|s| s.id == condition.stratification) else {
                    return Err(format!(
                        "{owner}: a condition names '{}', which is not a stratification \
                         declared before it",
                        condition.stratification
                    ));
                };
                condition.check(&owner, named)?;
            }
        }
        for transition in &self.transitions {
            for (index, stratified_rate) in transition.stratified_rates.iter().flatten().enumerate()
            {
                let owner = format!("Transition '{}', stratified rate {index}", transition.id);
                for condition in &stratified_rate.conditions {
                    let Some(named) = stratifications
                        .iter()
                        .find(|s| s.id == condition.stratification)
                    else {
                        let ids: Vec<&str> =
                            stratifications.iter().map(|s| s.id.as_str()).collect();
                        return Err(format!(
                            "{owner}: a condition names '{}', which is not a stratification \
                             of the process; its stratifications are {ids:?}",
                            condition.stratification
                        ));
                    };
                    condition.check(&owner, named)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process() -> Process {
        Process {
            stratifications: vec![Stratification {
                id: "group".to_string(),
                categories: vec!["g1".to_string(), "g2".to_string()],
                conditions: None,
            }],
            ..Process::default()
        }
    }

    fn matrix(values: &[(&str, &[(&str, f64)])]) -> CategoryMatrix {
        CategoryMatrix {
            id: "M".to_string(),
            stratification: "group".to_string(),
            values: values
                .iter()
                .map(|(row, columns)| {
                    (
                        row.to_string(),
                        columns.iter().map(|(c, v)| (c.to_string(), *v)).collect(),
                    )
                })
                .collect(),
            row_scales: IndexMap::new(),
            column_scales: IndexMap::new(),
        }
    }

    const FULL: [(&str, &[(&str, f64)]); 2] = [
        ("g2", &[("g2", 3.0), ("g1", 1.0)]),
        ("g1", &[("g1", 2.0), ("g2", 1.0)]),
    ];

    #[test]
    fn a_matrix_reads_its_rows_in_the_order_of_the_categories() {
        let mut process = process();
        let matrix = matrix(&FULL);
        process.matrices.push(matrix.clone());

        assert_eq!(matrix.check(&process), Ok(()));
        assert_eq!(matrix.rows(&process), [2.0, 1.0, 1.0, 3.0]);
        assert_eq!(matrix.entry_name("g1", "g2"), "M_g1_g2");
        assert_eq!(matrix.radius_name(), "M_radius");
        assert!(process.matrix("M").is_some() && process.matrix("N").is_none());
        assert!(!process.is_empty());
    }

    #[test]
    fn a_matrix_that_does_not_fit_its_stratification_names_the_category() {
        let categories = r#"["g1", "g2"]"#;
        let mut other = matrix(&FULL);
        other.stratification = "other".to_string();
        let infinite = matrix(&[FULL[0], ("g1", &[("g1", f64::INFINITY), ("g2", 1.0)])]);
        for (matrix, message) in [
            (
                other,
                r#"Matrix 'M' is between the categories of 'other', which the process does not have; its stratifications are ["group"]"#.to_string(),
            ),
            (
                matrix(&[FULL[1]]),
                format!(
                    "Matrix 'M' has no row for the category 'g2' of 'group'; it needs one \
                     value for each pair of its categories ({categories})"
                ),
            ),
            (
                matrix(&[FULL[0], FULL[1], ("g3", &[])]),
                format!("Matrix 'M' names 'g3', which is not a category of 'group' ({categories})"),
            ),
            (
                matrix(&[FULL[0], ("g1", &[("g1", 2.0), ("g3", 1.0)])]),
                format!("Matrix 'M' names 'g3', which is not a category of 'group' ({categories})"),
            ),
            (
                scaled(matrix(&FULL), &[], &[("g3", "b3")]),
                format!(
                    "Matrix 'M' scales the column of 'g3' by 'b3', but 'g3' is not a \
                     category of 'group' ({categories})"
                ),
            ),
            (
                infinite,
                "Matrix 'M' has inf in row 'g1' and column 'g1'; each value must be a finite \
                 number"
                    .to_string(),
            ),
        ] {
            assert_eq!(matrix.check(&process()), Err(message));
        }
    }

    fn scaled(
        mut matrix: CategoryMatrix,
        rows: &[(&str, &str)],
        columns: &[(&str, &str)],
    ) -> CategoryMatrix {
        let map = |scales: &[(&str, &str)]| {
            scales
                .iter()
                .map(|(category, parameter)| (category.to_string(), parameter.to_string()))
                .collect()
        };
        matrix.row_scales = map(rows);
        matrix.column_scales = map(columns);
        matrix
    }

    #[test]
    fn a_matrix_lists_its_scales() {
        let matrix = scaled(matrix(&FULL), &[("g1", "a1")], &[("g2", "b2")]);
        let mut process = process();
        process.matrices.push(matrix.clone());
        assert_eq!(matrix.check(&process), Ok(()));
        assert_eq!(
            matrix.scales().collect::<Vec<_>>(),
            [("row", "g1", "a1"), ("column", "g2", "b2")]
        );
    }

    #[test]
    fn a_matrix_reads_without_scales() {
        let json = r#"{"id": "M", "stratification": "group", "values": {}}"#;
        let matrix: CategoryMatrix = serde_json::from_str(json).unwrap();
        assert!(matrix.row_scales.is_empty() && matrix.column_scales.is_empty());
    }
}
