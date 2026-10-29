//! Matrices between the categories of a stratification: the variables of a
//! matrix, and the matrix sums of a rate written as the sums they are.
//!
//! A matrix `M` gives a rate one variable per entry, `<M>_<i>_<j>`, and its
//! spectral radius, `<M>_radius` ([`matrix_variables`]). In the rate of a
//! compartment of category `i`, `matrix_sum(M, e)` is
//! `Σ_j <M>_<i>_<j> * (e_j)`, where `e_j` is `e` with each bin read as its
//! amount in category `j` and `total` as the total of category `j`
//! ([`MatrixSums::resolve`]). A term whose entry is 0 is left out, unless
//! every entry of the row is 0: then the first term stays, so the rate keeps
//! its unit and its value is 0. The amounts are found by label: one compartment is its name, several are the
//! subgroup total that sums exactly them, or their sum.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use commol_core::math_expression::jit::ast::{BinaryOperator, Expr};
use commol_core::math_expression::jit::parser::parse_expression;
use commol_core::math_expression::language::variables;
use commol_core::{
    CategoryMatrix, MATRIX_SUM, NEIGHBORS, NEIGHBORS_PREFIX, Process, SPECIAL_VAR_TOTAL,
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
    /// Its value: the entry, or the spectral radius.
    pub value: f64,
    /// Whether it is the spectral radius rather than an entry.
    pub radius: bool,
}

/// The variables of the matrices of a process: for each matrix, its entries
/// row by row in the order of the categories, then its spectral radius.
pub fn matrix_variables(process: &Process) -> Vec<MatrixVariable> {
    let mut variables = Vec::new();
    for matrix in &process.matrices {
        let categories = matrix.categories(process);
        for row in categories {
            for column in categories {
                variables.push(MatrixVariable {
                    name: matrix.entry_name(row, column),
                    matrix: matrix.id.clone(),
                    value: matrix.value(row, column),
                    radius: false,
                });
            }
        }
        variables.push(MatrixVariable {
            name: matrix.radius_name(),
            matrix: matrix.id.clone(),
            value: spectral_radius(&matrix.rows(process), categories.len()),
            radius: true,
        });
    }
    variables
}

/// Check the matrices of a process and the matrix sums of its formulas.
///
/// # Errors
///
/// Returns a message when two matrices share an id, a matrix does not fit
/// the process ([`CategoryMatrix::check`]), the id or a variable of a matrix
/// has the name of a parameter, a bin, an accumulator, a compartment, a
/// subgroup total or an engine variable (`taken`, by name and what it is),
/// two matrices give one variable, a rate reads a matrix the process does
/// not have or reads the neighbors inside a matrix sum, or a formula
/// parameter reads a matrix sum. So the engine, a unit and a description
/// never read one name as two things.
pub(crate) fn check_matrices(
    process: &Process,
    rates: &[(&str, &str)],
    formula_parameters: &[(&str, &str)],
    taken: &HashMap<&str, &str>,
) -> Result<(), String> {
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
    }
    let mut given: HashMap<String, String> = HashMap::new();
    for variable in matrix_variables(process) {
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
    /// without the terms of a zero entry; the first term alone when every
    /// entry of the row is 0.
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
        // `own` is a category of the stratification, so the row has a column.
        let categories = matrix.categories(self.process);
        let mut columns: Vec<&String> = categories
            .iter()
            .filter(|column| matrix.value(own, column) != 0.0)
            .collect();
        if columns.is_empty() {
            columns.push(&categories[0]);
        }
        let mut terms = columns.into_iter().map(|column| {
            Expr::binary(
                BinaryOperator::Mul,
                Expr::variable(matrix.entry_name(own, column)),
                self.in_category(expression, matrix, column),
            )
        });
        let first = terms.next().expect("the row has a column");
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
