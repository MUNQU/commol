//! A matrix between categories as an engine and a calibration constraint
//! read it: its values, each scaled by the parameters of its row and its
//! column, and the spectral radius of the scaled matrix.
//!
//! The process holds the matrix as data ([`CategoryMatrix`]). An engine keeps
//! one [`ScaledMatrix`] per matrix and changes the value of a scale when a
//! caller sets its parameter; a calibration constraint reads its variables at
//! the values of the point it evaluates ([`ScaledMatrix::variables`]).

use commol_statistics::linear_algebra::spectral_radius;
use indexmap::IndexMap;

use crate::types::{CategoryMatrix, Process};

/// The scale of a row or a column of a matrix: the parameter that
/// multiplies it, and the value of that parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixScale {
    /// The id of the parameter.
    pub parameter: String,
    /// Its value; `None` while the parameter has none, such as a parameter
    /// to calibrate before a value is set.
    pub value: Option<f64>,
}

/// A matrix between categories with the value of each of its scales: what a
/// rate and a calibration constraint read.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaledMatrix {
    /// The id of the matrix.
    pub id: String,
    /// The name of each entry, `<M>_<i>_<j>`, row by row in the order of the
    /// categories.
    pub entry_names: Vec<String>,
    /// The name of the spectral radius, `<M>_radius`.
    pub radius_name: String,
    /// The values before the scales, row by row.
    values: Vec<f64>,
    /// The scale of each row, in the order of the categories.
    rows: Vec<Option<MatrixScale>>,
    /// The scale of each column, in the order of the categories.
    columns: Vec<Option<MatrixScale>>,
}

impl ScaledMatrix {
    /// `matrix` of `process`, with each scale at `value(parameter)`.
    pub fn new(
        matrix: &CategoryMatrix,
        process: &Process,
        value: impl Fn(&str) -> Option<f64>,
    ) -> Self {
        let categories = matrix.categories(process);
        let scales = |scales: &IndexMap<String, String>| -> Vec<Option<MatrixScale>> {
            categories
                .iter()
                .map(|category| {
                    scales.get(category).map(|parameter| MatrixScale {
                        parameter: parameter.clone(),
                        value: value(parameter),
                    })
                })
                .collect()
        };
        Self {
            id: matrix.id.clone(),
            entry_names: categories
                .iter()
                .flat_map(|row| {
                    categories
                        .iter()
                        .map(move |column| matrix.entry_name(row, column))
                })
                .collect(),
            radius_name: matrix.radius_name(),
            values: matrix.rows(process),
            rows: scales(&matrix.row_scales),
            columns: scales(&matrix.column_scales),
        }
    }

    /// The number of its rows, which is the number of its columns.
    pub fn size(&self) -> usize {
        self.rows.len()
    }

    /// The names of its variables: its entries row by row, then its
    /// spectral radius.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entry_names
            .iter()
            .chain(std::iter::once(&self.radius_name))
            .map(String::as_str)
    }

    /// Whether `parameter` scales a row or a column.
    pub fn scaled_by(&self, parameter: &str) -> bool {
        self.scales().any(|scale| scale.parameter == parameter)
    }

    /// Give every scale of `parameter` the value `value`; whether the matrix
    /// has one.
    pub fn set_scale(&mut self, parameter: &str, value: f64) -> bool {
        let mut found = false;
        for scale in self.rows.iter_mut().chain(&mut self.columns).flatten() {
            if scale.parameter == parameter {
                scale.value = Some(value);
                found = true;
            }
        }
        found
    }

    /// The entries row by row, `a_i · values[i][j] · b_j`: each scale at
    /// `value(parameter)` when that gives one, else at its own value.
    ///
    /// # Errors
    ///
    /// Returns the scale, the first row then column, that has no value.
    pub fn entries<'a>(
        &'a self,
        value: impl Fn(&str) -> Option<f64>,
    ) -> Result<Vec<f64>, &'a MatrixScale> {
        let factors = |scales: &'a [Option<MatrixScale>]| -> Result<Vec<f64>, &'a MatrixScale> {
            scales
                .iter()
                .map(|scale| match scale {
                    None => Ok(1.0),
                    Some(scale) => value(&scale.parameter).or(scale.value).ok_or(scale),
                })
                .collect()
        };
        let rows = factors(&self.rows)?;
        let columns = factors(&self.columns)?;
        let size = self.size();
        Ok(self
            .values
            .iter()
            .enumerate()
            .map(|(index, entry)| rows[index / size] * entry * columns[index % size])
            .collect())
    }

    /// The values of its variables, parallel to [`Self::names`]: the
    /// entries ([`Self::entries`]), then the spectral radius of those
    /// entries (`commol_statistics::linear_algebra::spectral_radius`).
    ///
    /// # Errors
    ///
    /// Returns the scale, the first row then column, that has no value.
    pub fn variables(&self, value: impl Fn(&str) -> Option<f64>) -> Result<Vec<f64>, &MatrixScale> {
        let mut values = self.entries(value)?;
        values.push(spectral_radius(&values, self.size()));
        Ok(values)
    }

    fn scales(&self) -> impl Iterator<Item = &MatrixScale> {
        self.rows.iter().chain(&self.columns).flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Stratification;

    /// The matrix `M` = [[2, 1], [1, 3]] of `g1`, `g2`, whose row of `g1`
    /// `a1` scales and whose column of `g2` `b2` scales.
    fn matrix() -> (CategoryMatrix, Process) {
        let matrix: CategoryMatrix = serde_json::from_str(
            r#"{"id": "M", "stratification": "group",
            "values": {"g1": {"g1": 2.0, "g2": 1.0}, "g2": {"g1": 1.0, "g2": 3.0}},
            "row_scales": {"g1": "a1"}, "column_scales": {"g2": "b2"}}"#,
        )
        .unwrap();
        let process = Process {
            stratifications: vec![Stratification {
                id: "group".to_string(),
                categories: vec!["g1".to_string(), "g2".to_string()],
                conditions: None,
            }],
            matrices: vec![matrix.clone()],
            ..Process::default()
        };
        (matrix, process)
    }

    #[test]
    fn a_scaled_matrix_multiplies_each_row_and_column_by_its_parameter() {
        let (matrix, process) = matrix();
        let mut scaled = ScaledMatrix::new(&matrix, &process, |parameter| {
            (parameter == "a1").then_some(2.0)
        });
        assert_eq!(scaled.size(), 2);
        assert_eq!(
            scaled.names().collect::<Vec<_>>(),
            ["M_g1_g1", "M_g1_g2", "M_g2_g1", "M_g2_g2", "M_radius"]
        );
        assert!(scaled.scaled_by("b2") && !scaled.scaled_by("k"));
        // `b2` has no value: the entries need one, from the caller or a set.
        let unset = scaled.entries(|_| None).unwrap_err();
        assert_eq!((unset.parameter.as_str(), unset.value), ("b2", None));
        assert_eq!(scaled.variables(|_| None).unwrap_err().parameter, "b2");
        assert_eq!(
            scaled.entries(|parameter| (parameter == "b2").then_some(3.0)),
            Ok(vec![4.0, 6.0, 1.0, 9.0])
        );

        assert!(scaled.set_scale("b2", 0.5));
        assert!(!scaled.set_scale("k", 0.5));
        assert_eq!(scaled.entries(|_| None), Ok(vec![4.0, 1.0, 1.0, 1.5]));
        // A value from the caller wins over the value of the scale.
        assert_eq!(
            scaled.entries(|parameter| (parameter == "a1").then_some(1.0)),
            Ok(vec![2.0, 0.5, 1.0, 1.5])
        );
    }

    #[test]
    fn the_variables_are_the_entries_then_the_radius() {
        let (matrix, process) = matrix();
        let scaled = ScaledMatrix::new(&matrix, &process, |_| Some(1.0));

        let variables = scaled.variables(|_| None).unwrap();
        assert_eq!(variables[..4], [2.0, 1.0, 1.0, 3.0]);
        assert!((variables[4] - (5.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
        // [[4, 2], [1, 3]]: the eigenvalues 5 and 2.
        let doubled = scaled
            .variables(|parameter| (parameter == "a1").then_some(2.0))
            .unwrap();
        assert!((doubled[4] - 5.0).abs() < 1e-12);
    }
}
