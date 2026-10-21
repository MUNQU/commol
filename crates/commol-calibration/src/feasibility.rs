//! Feasibility: bounds, constraints, infeasible starts and infeasible runs.
//!
//! A point of parameter space is feasible, or it violates the problem by an
//! amount ([`Violation`]). This module never turns a violation into a
//! penalty: the optimization objective does that
//! ([`crate::objective::OptimizationObjective`]), and an inference method sees
//! a violation as a point outside the support.

use std::collections::HashSet;

use commol_core::{MathExpression, MathExpressionContext};

use crate::types::{CalibrationConstraint, CalibrationParameter};

/// Why a point is not feasible.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    /// A value lies outside its bounds; `amount` is the sum of the distances
    /// to the bounds.
    OutOfBounds {
        /// Sum of the distances of the values to their bounds.
        amount: f64,
    },
    /// Constraints evaluate below zero; `amount` is the sum, over the
    /// violated constraints and positions, of the weight times the
    /// violation.
    Constraint {
        /// Sum of weight × violation.
        amount: f64,
    },
    /// The values make the starting state infeasible; `excess` is how far
    /// the state is from feasible, as the engine measures it.
    InfeasibleStart {
        /// The engine's measure of the excess.
        excess: f64,
        /// What the values exceed, as the plan of values reports it.
        message: String,
    },
    /// The run left the values the model allows.
    InfeasibleRun {
        /// What left them, as the engine reports it: the compartment, the
        /// position and the flows.
        message: String,
    },
    /// The run or the discrepancy is not a finite number.
    NonFinite,
    /// A predicted mean lies outside the means its observation model
    /// allows; `amount` is the sum of the distances to them.
    MeanOutsideModel {
        /// Sum of the distances of the means to the means the models allow.
        amount: f64,
    },
}

impl Violation {
    /// The size of the violation on the scale of the objective's penalty:
    /// the constraint, bound or mean amount, 100 times the excess of an
    /// infeasible start, and 0 for an infeasible or non-finite run.
    pub fn amount(&self) -> f64 {
        match *self {
            Self::OutOfBounds { amount }
            | Self::Constraint { amount }
            | Self::MeanOutsideModel { amount } => amount,
            Self::InfeasibleStart { excess, .. } => 100.0 * excess,
            Self::InfeasibleRun { .. } | Self::NonFinite => 0.0,
        }
    }
}

/// Why an evaluation has no discrepancy: the point is not feasible, or the
/// evaluation failed for a reason that no point avoids.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// The point is not feasible.
    Violation(Violation),
    /// The engine or a constraint could not be evaluated; the message says
    /// why. Calibration stops on it.
    Failed(String),
}

impl From<Violation> for EvaluationError {
    fn from(violation: Violation) -> Self {
        Self::Violation(violation)
    }
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Violation(violation) => write!(f, "Infeasible point: {violation:?}"),
            Self::Failed(message) => f.write_str(message),
        }
    }
}

/// The bounds and constraints of a calibration problem.
#[derive(Debug, Clone)]
pub struct Feasibility {
    /// `(id, min, max)` of each parameter, in problem order.
    parameters: Vec<(String, f64, f64)>,
    /// Constraints that read only parameters.
    parameter_constraints: Vec<CompiledConstraint>,
    /// Constraints that read outputs at positions of the run.
    position_constraints: Vec<CompiledConstraint>,
    /// Names of the state outputs, which `total` sums.
    state_names: Vec<String>,
    /// Outputs after the state, such as accumulators, as `(name, index)`.
    non_state_outputs: Vec<(String, usize)>,
    /// Named sums of outputs a constraint may read.
    output_sums: Vec<(String, Vec<usize>)>,
}

#[derive(Debug, Clone)]
struct CompiledConstraint {
    weight: f64,
    expression: MathExpression,
    /// `(position, row)` pairs of a constraint at positions.
    rows: Vec<(f64, usize)>,
}

/// The positions at which a constraint is evaluated, from its definition.
pub(crate) fn constraint_positions(constraints: &[CalibrationConstraint]) -> Vec<f64> {
    constraints
        .iter()
        .flat_map(|constraint| constraint.at.iter().flatten().copied())
        .collect()
}

impl Feasibility {
    /// The feasibility of `parameters` under `constraints`, on a run at
    /// `positions` whose rows have the outputs `output_names`.
    ///
    /// # Errors
    ///
    /// Returns a message when a constraint expression is invalid, names an
    /// unknown variable, or reads an output without positions.
    pub fn new(
        parameters: &[CalibrationParameter],
        constraints: &[CalibrationConstraint],
        output_names: &[String],
        state_outputs: usize,
        output_sums: Vec<(String, Vec<usize>)>,
        positions: &[f64],
    ) -> Result<Self, String> {
        let parameter_ids: HashSet<&str> = parameters.iter().map(|p| p.id.as_str()).collect();
        let outputs: HashSet<&str> = output_names
            .iter()
            .map(String::as_str)
            .chain(output_sums.iter().map(|(name, _)| name.as_str()))
            .collect();
        let row_of = |position: f64| {
            positions
                .binary_search_by(|candidate| candidate.total_cmp(&position))
                .expect("the schedule holds every constraint position")
        };

        let mut parameter_constraints = Vec::new();
        let mut position_constraints = Vec::new();
        for constraint in constraints {
            let expression = MathExpression::new(constraint.expression.clone());
            if let Err(e) = expression.validate().and_then(|()| expression.compiles()) {
                return Err(format!(
                    "Constraint '{}' has invalid expression: {:?}",
                    constraint.id, e
                ));
            }
            // The engine variables (`total`, `t`, ...) are not among them.
            for variable in expression.get_variables() {
                let is_parameter = parameter_ids.contains(variable.as_str());
                let is_output = outputs.contains(variable.as_str());
                if !is_parameter && !is_output {
                    return Err(format!(
                        "Constraint '{}' references unknown variable '{}' (not a parameter or \
                         simulation output)",
                        constraint.id, variable
                    ));
                }
                if is_output && constraint.at.is_none() {
                    return Err(format!(
                        "Constraint '{}' references output '{}' but has no positions (`at`)",
                        constraint.id, variable
                    ));
                }
            }
            let compiled = CompiledConstraint {
                weight: constraint.weight,
                expression,
                rows: constraint
                    .at
                    .iter()
                    .flatten()
                    .map(|&position| (position, row_of(position)))
                    .collect(),
            };
            if constraint.at.is_some() {
                position_constraints.push(compiled);
            } else {
                parameter_constraints.push(compiled);
            }
        }

        Ok(Self {
            parameters: parameters
                .iter()
                .map(|p| (p.id.clone(), p.min_bound, p.max_bound))
                .collect(),
            parameter_constraints,
            position_constraints,
            state_names: output_names[..state_outputs].to_vec(),
            non_state_outputs: output_names
                .iter()
                .enumerate()
                .skip(state_outputs)
                .map(|(index, name)| (name.clone(), index))
                .collect(),
            output_sums,
        })
    }

    /// The same feasibility on a run at `positions`, a schedule that holds
    /// every position of the constraints.
    pub(crate) fn on_schedule(&self, positions: &[f64]) -> Self {
        let mut feasibility = self.clone();
        for constraint in &mut feasibility.position_constraints {
            for (position, row) in &mut constraint.rows {
                *row = positions
                    .binary_search_by(|candidate| candidate.total_cmp(position))
                    .expect("the schedule holds every constraint position");
            }
        }
        feasibility
    }

    /// Whether some constraint reads the rows of a run.
    pub fn reads_rows(&self) -> bool {
        !self.position_constraints.is_empty()
    }

    /// Check that every value is inside its bounds.
    pub fn check_bounds(&self, values: &[f64]) -> Result<(), Violation> {
        let amount: f64 = self
            .parameters
            .iter()
            .zip(values)
            .map(|((_, min, max), &value)| (min - value).max(0.0) + (value - max).max(0.0))
            .sum();
        if amount > 0.0 {
            Err(Violation::OutOfBounds { amount })
        } else {
            Ok(())
        }
    }

    fn context(&self, values: &[f64]) -> MathExpressionContext {
        let mut context = MathExpressionContext::new();
        for ((id, _, _), &value) in self.parameters.iter().zip(values) {
            context.set_parameter(id.clone(), value);
        }
        context
    }

    /// Check the constraints that read only parameters.
    ///
    /// # Errors
    ///
    /// A [`Violation::Constraint`] when one evaluates below zero, or a
    /// failure when one cannot be evaluated.
    pub fn check_parameters(&self, values: &[f64]) -> Result<(), EvaluationError> {
        if self.parameter_constraints.is_empty() {
            return Ok(());
        }
        let mut context = self.context(values);
        let mut amount = 0.0;
        for constraint in &self.parameter_constraints {
            let value = constraint
                .expression
                .evaluate(&mut context)
                .expect("Feasibility::new checks that a constraint compiles and reads parameters");
            if value < 0.0 {
                amount += constraint.weight * -value;
            }
        }
        violation(amount)
    }

    /// The constraint amount of the rows of one run: 0 when every
    /// constraint at positions holds.
    pub(crate) fn row_violation(&self, values: &[f64], rows: &[Vec<f64>]) -> f64 {
        let mut context = self.context(values);
        // Only the state outputs are registered, so `total` stays the sum of
        // the state. Outputs after it are injected by name, with the sums.
        context.init_compartments(self.state_names.clone());
        let mut amount = 0.0;
        for constraint in &self.position_constraints {
            for &(position, row) in &constraint.rows {
                // The rows are those of the schedule, which holds every
                // constraint position.
                let values = &rows[row];
                context.set_compartments_by_index(values);
                for (name, index) in &self.non_state_outputs {
                    context.set_parameter_str(name, values[*index]);
                }
                for (name, indices) in &self.output_sums {
                    let total: f64 = indices.iter().map(|&i| values[i]).sum();
                    context.set_parameter_str(name, total);
                }
                context.set_step(position);
                let value = constraint.expression.evaluate(&mut context).expect(
                    "Feasibility::new checks that a constraint compiles and reads parameters and outputs",
                );
                if value < 0.0 {
                    amount += constraint.weight * -value;
                }
            }
        }
        amount
    }
}

fn violation(amount: f64) -> Result<(), EvaluationError> {
    if amount > 0.0 {
        Err(Violation::Constraint { amount }.into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feasibility(constraints: &[CalibrationConstraint]) -> Feasibility {
        let parameters = vec![
            CalibrationParameter::new("k1".to_string(), 0.0, 1.0),
            CalibrationParameter::new("k2".to_string(), 0.0, 2.0),
        ];
        let names = vec!["A".to_string(), "B".to_string()];
        Feasibility::new(&parameters, constraints, &names, 2, vec![], &[1.0, 2.0]).unwrap()
    }

    #[test]
    fn bounds_report_the_distance_outside_them() {
        let feasibility = feasibility(&[]);
        assert_eq!(feasibility.check_bounds(&[0.5, 2.0]), Ok(()));
        assert_eq!(
            feasibility.check_bounds(&[-0.5, 2.5]),
            Err(Violation::OutOfBounds { amount: 1.0 })
        );
    }

    #[test]
    fn a_violated_constraint_reports_its_weighted_amount() {
        let constraints = [
            CalibrationConstraint::new("order".to_string(), "k1 - k2".to_string()).with_weight(2.0),
            CalibrationConstraint::new("cap".to_string(), "3 - B".to_string())
                .with_positions(vec![2.0]),
        ];
        let feasibility = feasibility(&constraints);

        assert_eq!(feasibility.check_parameters(&[0.75, 0.5]), Ok(()));
        assert_eq!(
            feasibility.check_parameters(&[0.25, 0.5]),
            Err(Violation::Constraint { amount: 0.5 }.into())
        );
        let rows = vec![vec![0.0, 9.0], vec![1.0, 4.0]];
        assert_eq!(feasibility.row_violation(&[0.25, 0.5], &rows), 1.0);
    }

    #[test]
    fn the_amount_of_a_bound_violation_is_its_distance_outside_them() {
        assert_eq!(Violation::OutOfBounds { amount: 1.5 }.amount(), 1.5);
    }

    #[test]
    fn the_amount_of_an_infeasible_start_is_a_hundred_times_its_excess() {
        let start = Violation::InfeasibleStart {
            excess: 0.5,
            message: "A above its limit".to_string(),
        };
        assert_eq!(start.amount(), 50.0);
        let run = Violation::InfeasibleRun {
            message: "A below zero".to_string(),
        };
        assert_eq!(run.amount(), 0.0);
    }
}
