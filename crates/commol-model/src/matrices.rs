//! Matrices between the categories of a stratification: the variables of a
//! matrix, and the matrix sums of a rate written as the sums they are.
//!
//! A matrix `M` gives a rate one variable per entry, `<M>_<i>_<j>`, and its
//! spectral radius, `<M>_radius` ([`matrix_variables`]), both of the matrix
//! scaled by the parameters of its rows and columns ([`ScaledMatrix`]), so
//! the engine writes them again when a caller sets a scale
//! ([`variable_values`]). In the rate of a
//! compartment of category `i`, `matrix_sum(M, e)` is
//! `Σ_j <M>_<i>_<j> * (e_j)`, where `e_j` is `e` with each bin read as its
//! amount in category `j` and `total` as the total of category `j`
//! ([`MatrixSums::resolve`]). A term whose entry is 0 is left out. A row of
//! zeros is `if(<M>_<i>_<j> != 0, <M>_<i>_<j> * (e_j), 0)` of its first
//! column: 0, in the unit of the sum, even where `e_j` is not a number. The
//! amounts are found by label: one compartment is its name, several are the
//! subgroup total that sums exactly them, or their sum.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use commol_core::math_expression::jit::ast::{BinaryOperator, Expr};
use commol_core::math_expression::jit::parser::parse_expression;
use commol_core::math_expression::language::variables;
use commol_core::{
    CategoryMatrix, MATRIX_SUM, Model, NEIGHBORS, NEIGHBORS_PREFIX, ParameterValue, Process,
    SPECIAL_VAR_TOTAL, ScaledMatrix,
};
use commol_statistics::linear_algebra::spectral_radius;

use crate::builder::subgroup_totals_by_compartments;
use crate::layout::CompartmentLayout;

/// A variable that a matrix gives the rates and the formula parameters of a
/// model.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixVariable {
    /// The name a formula reads: `<M>_<i>_<j>` or `<M>_radius`.
    pub name: String,
    /// The id of the matrix.
    pub matrix: String,
    /// Its value at the values of the scales of the matrix: the entry, or
    /// the spectral radius; `None` while a scale has no value, such as a
    /// parameter to calibrate.
    pub value: Option<f64>,
    /// Whether it is the spectral radius rather than an entry.
    pub radius: bool,
}

/// The matrices of a model, each scale at the value of its parameter: the
/// constant, or none for a parameter to calibrate.
pub fn scaled_matrices(model: &Model) -> Vec<ScaledMatrix> {
    let constant = |id: &str| {
        model
            .parameters
            .iter()
            .find(|parameter| parameter.id == id)
            .and_then(|parameter| match parameter.value {
                Some(ParameterValue::Constant(value)) => Some(value),
                _ => None,
            })
    };
    model
        .process
        .matrices
        .iter()
        .map(|matrix| ScaledMatrix::new(matrix, &model.process, constant))
        .collect()
}

/// The values of the variables of `matrix` at the values of its scales,
/// parallel to [`ScaledMatrix::names`]: its entries row by row, then the
/// spectral radius of those entries. `None` while a scale has no value.
pub fn variable_values(matrix: &ScaledMatrix) -> Option<Vec<f64>> {
    let mut values = matrix.entries(|_| None).ok()?;
    values.push(spectral_radius(&values, matrix.size()));
    Some(values)
}

/// The variables of the matrices of a model: for each matrix, its entries
/// row by row in the order of the categories, then its spectral radius, at
/// the values of its scales ([`scaled_matrices`]).
pub fn matrix_variables(model: &Model) -> Vec<MatrixVariable> {
    let mut variables = Vec::new();
    for matrix in scaled_matrices(model) {
        let values = variable_values(&matrix);
        let count = matrix.entry_names.len();
        for (index, name) in matrix.names().enumerate() {
            variables.push(MatrixVariable {
                name: name.to_string(),
                matrix: matrix.id.clone(),
                value: values.as_ref().map(|values| values[index]),
                radius: index == count,
            });
        }
    }
    variables
}

/// Check the matrices of a model, their scales and the matrix sums of its
/// formulas.
///
/// # Errors
///
/// Returns a message when two matrices share an id, a matrix does not fit
/// the process ([`CategoryMatrix::check`]), a scale names a parameter the
/// model does not have or a formula or a series parameter, the id or a
/// variable of a matrix has the name of a parameter, a bin, an accumulator,
/// a compartment, a subgroup total or an engine variable (`taken`, by name
/// and what it is), two matrices give one variable, a variable of a matrix
/// is the id of a matrix, a rate reads a matrix the process does not have or
/// reads the neighbors inside a matrix sum, or a formula parameter reads a
/// matrix sum. So the engine, a unit and a description never read one name
/// as two things.
pub(crate) fn check_matrices(
    model: &Model,
    rates: &[(&str, &str)],
    formula_parameters: &[(&str, &str)],
    taken: &HashMap<&str, &str>,
) -> Result<(), String> {
    let process = &model.process;
    let mut ids = HashSet::new();
    for matrix in &process.matrices {
        if !ids.insert(matrix.id.as_str()) {
            return Err(format!("Two matrices have the id '{}'", matrix.id));
        }
        if let Some(kind) = taken.get(matrix.id.as_str()) {
            return Err(format!(
                "Matrix '{}' has the name of {kind}; rename one of them",
                matrix.id
            ));
        }
        matrix.check(process)?;
        check_scales(model, matrix)?;
    }
    let mut given: HashMap<String, String> = HashMap::new();
    for variable in matrix_variables(model) {
        if ids.contains(variable.name.as_str()) {
            return Err(format!(
                "Matrix '{}' gives the variable '{}', which is also the id of a matrix; \
                 rename one of them",
                variable.matrix, variable.name
            ));
        }
        if let Some(other) = given.insert(variable.name.clone(), variable.matrix.clone()) {
            return Err(format!(
                "Matrices '{other}' and '{}' both give the variable '{}'; rename one of them",
                variable.matrix, variable.name
            ));
        }
        if let Some(kind) = taken.get(variable.name.as_str()) {
            return Err(format!(
                "Matrix '{}' gives the variable '{}', which is also the name of {kind}; \
                 rename one of them",
                variable.matrix, variable.name
            ));
        }
    }
    for &(transition, rate) in rates {
        for (matrix, expression) in matrix_sums(rate) {
            if process.matrix(&matrix).is_none() {
                let ids: Vec<&str> = process.matrices.iter().map(|m| m.id.as_str()).collect();
                return Err(format!(
                    "Transition '{transition}' reads {MATRIX_SUM}({matrix}, …), but '{matrix}' \
                     is not a matrix of the process; its matrices are {ids:?}"
                ));
            }
            if let Some(term) = variables(&expression)
                .into_iter()
                .find(|name| name == NEIGHBORS || name.starts_with(NEIGHBORS_PREFIX))
            {
                return Err(format!(
                    "Transition '{transition}' reads '{term}' inside {MATRIX_SUM}({matrix}, …); \
                     the expression of a matrix sum reads bins, totals and parameters, not the \
                     neighbors of a unit"
                ));
            }
        }
    }
    for &(parameter, formula) in formula_parameters {
        if !matrix_sums(formula).is_empty() {
            return Err(format!(
                "Parameter '{parameter}' reads {MATRIX_SUM}, which reads the category of the \
                 compartment of a rate; only the rate of a transition reads it"
            ));
        }
    }
    Ok(())
}

/// Check that each scale of `matrix` names a constant or a parameter to
/// calibrate of `model`: a value that does not change during a run.
fn check_scales(model: &Model, matrix: &CategoryMatrix) -> Result<(), String> {
    for (side, category, parameter) in matrix.scales() {
        let what = match model.parameters.iter().find(|p| p.id == parameter) {
            None => "which is not a parameter of the model",
            Some(found) => match found.value {
                Some(ParameterValue::Formula(_)) => "a formula parameter",
                Some(ParameterValue::Series { .. }) => "a series parameter",
                Some(ParameterValue::Constant(_)) | None => continue,
            },
        };
        return Err(format!(
            "Matrix '{}' scales the {side} of '{category}' by '{parameter}', {what}; a scale \
             is a constant or a parameter to calibrate, which does not change during a run",
            matrix.id
        ));
    }
    Ok(())
}

/// The matrix sums of a formula, each as the id of its matrix and its
/// expression; none when the formula does not parse.
fn matrix_sums(formula: &str) -> Vec<(String, Expr)> {
    fn collect(expr: &Expr, out: &mut Vec<(String, Expr)>) {
        if let Expr::MatrixSum { matrix, expression } = expr {
            out.push((matrix.clone(), expression.as_ref().clone()));
        }
        for operand in expr.operands() {
            collect(operand, out);
        }
    }
    let mut out = Vec::new();
    if let Ok(tree) = parse_expression(formula) {
        collect(&tree, &mut out);
    }
    out
}

/// The variables the tree of a rate reads outside its matrix sums, which
/// decide whether the rate is absolute: the expression of a matrix sum reads
/// the amounts of other categories, as the neighbors of a unit do.
pub(crate) fn variables_outside_matrix_sums(rate: &Expr) -> Vec<String> {
    let Ok(outside) = rate.try_map(&mut |node| match node {
        Expr::MatrixSum { .. } => Some(Ok::<_, Infallible>(Expr::Constant(0.0))),
        _ => None,
    });
    variables(&outside)
}

/// Writes the matrix sums of the rates of a process as the sums they are.
pub(crate) struct MatrixSums<'a> {
    process: &'a Process,
    layout: &'a CompartmentLayout,
    /// The subgroup total of each set of compartments, sorted, that one sums.
    totals: HashMap<Vec<usize>, String>,
}

impl<'a> MatrixSums<'a> {
    /// The matrix sums of `process`, laid out in `layout`.
    pub(crate) fn new(process: &'a Process, layout: &'a CompartmentLayout) -> Self {
        let totals = if process.matrices.is_empty() {
            HashMap::new()
        } else {
            subgroup_totals_by_compartments(layout, process)
        };
        Self {
            process,
            layout,
            totals,
        }
    }

    /// `rate`, whose tree is `tree` (None when it does not parse), with each
    /// of its matrix sums written as the sum it is in the compartment
    /// `start`; `rate` itself when it reads none.
    ///
    /// # Errors
    ///
    /// Returns a message when a matrix sum names a matrix the process does
    /// not have, or `start` carries no category of the stratification of
    /// the matrix.
    pub(crate) fn resolve(
        &self,
        rate: &str,
        tree: Option<&Expr>,
        start: usize,
        transition_id: &str,
    ) -> Result<String, String> {
        match tree {
            Some(tree) if tree.reads_matrix_sum() => Ok(tree
                .try_map(&mut |node| match node {
                    Expr::MatrixSum { matrix, expression } => {
                        Some(self.sum(matrix, expression, start, transition_id))
                    }
                    _ => None,
                })?
                .to_string()),
            _ => Ok(rate.to_string()),
        }
    }

    /// `Σ_j <M>_<i>_<j> * (e_j)` in the compartment `start`, of category `i`,
    /// without the terms of a zero entry; for a row of zeros, the term of its
    /// first column when its entry is not 0, else 0.
    fn sum(
        &self,
        matrix: &str,
        expression: &Expr,
        start: usize,
        transition_id: &str,
    ) -> Result<Expr, String> {
        let Some(matrix) = self.process.matrix(matrix) else {
            return Err(format!(
                "Transition '{transition_id}' reads the matrix '{matrix}', which the process \
                 does not have"
            ));
        };
        let stratification = &matrix.stratification;
        let Some(own) = self.layout.labels()[start].category(stratification) else {
            return Err(format!(
                "Transition '{transition_id}' reads {MATRIX_SUM}({}, …) from '{}', which has \
                 no category of '{stratification}', the stratification of the matrix",
                matrix.id,
                self.layout.name(start)
            ));
        };
        let term = |column: &String| {
            Expr::binary(
                BinaryOperator::Mul,
                Expr::variable(matrix.entry_name(own, column)),
                self.in_category(expression, matrix, column),
            )
        };
        // `own` is a category of the stratification, so the row has a column.
        let categories = matrix.categories(self.process);
        let mut terms = categories
            .iter()
            .filter(|column| matrix.value(own, column) != 0.0)
            .map(term);
        let Some(first) = terms.next() else {
            // A row of zeros: `if(<M>_<i>_<j> != 0, <M>_<i>_<j> * (e_j), 0)` of
            // the first column has the unit of the sum, and it is 0 even
            // where `e_j` is not a number.
            let column = &categories[0];
            let entry = Expr::variable(matrix.entry_name(own, column));
            return Ok(Expr::if_then_else(
                Expr::binary(BinaryOperator::Ne, entry, Expr::Constant(0.0)),
                term(column),
                Expr::Constant(0.0),
            ));
        };
        let rest: Vec<_> = terms.map(|term| (BinaryOperator::Add, term)).collect();
        Ok(if rest.is_empty() {
            first
        } else {
            Expr::chain(first, rest)
        })
    }

    /// `expression` read in `category` of the stratification of `matrix`:
    /// each bin as its amount there, and `total` as the total there.
    fn in_category(&self, expression: &Expr, matrix: &CategoryMatrix, category: &str) -> Expr {
        let amount = |bin: Option<&str>| {
            let compartments: Vec<usize> = self
                .layout
                .with_category(&matrix.stratification, category)
                .filter(|&index| bin.is_none_or(|bin| self.layout.labels()[index].source == bin))
                .collect();
            self.amount(compartments)
        };
        let Ok(read) = expression.try_map(&mut |node| match node {
            Expr::Variable(name) if name == SPECIAL_VAR_TOTAL => {
                Some(Ok::<_, Infallible>(amount(None)))
            }
            Expr::Variable(name) if self.process.bins.iter().any(|bin| &bin.id == name) => {
                Some(Ok(amount(Some(name))))
            }
            _ => None,
        });
        read
    }

    /// The amount of a set of compartments: 0 for none, the name of one, the
    /// subgroup total that sums exactly them, or their sum.
    fn amount(&self, mut compartments: Vec<usize>) -> Expr {
        compartments.sort_unstable();
        match compartments.as_slice() {
            [] => Expr::Constant(0.0),
            [one] => Expr::variable(self.layout.name(*one)),
            many => match self.totals.get(many) {
                Some(total) => Expr::variable(total.clone()),
                None => Expr::chain(
                    Expr::variable(self.layout.name(many[0])),
                    many[1..]
                        .iter()
                        .map(|&index| {
                            (BinaryOperator::Add, Expr::variable(self.layout.name(index)))
                        })
                        .collect(),
                ),
            },
        }
    }
}
