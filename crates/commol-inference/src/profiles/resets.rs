//! What a profile says about the structure of a problem, by resetting the
//! other parameters to their fitted values.
//!
//! A profile of a parameter is flat for one of two reasons: other
//! parameters compensate it along the profile, or nothing the data read
//! depends on it. Resetting the other parameters tells them apart, with no
//! optimizer: the objective is evaluated with the parameter at a value of its
//! profile, or at either end of its box, and every other parameter at its
//! fitted value, and compared with the objective at the fit, `f̂`.
//!
//! - Every reset feasible and within the evaluation floor of `f̂`
//!   ([`ProfileTracer::evaluation_floor`]): nothing the data read depends on
//!   the parameter, `NonIdentifiable`. Since the minimum of the profile is
//!   at most each reset, this also proves the profile flat from above.
//! - A flat profile (within the tolerance `ε`,
//!   [`ProfileTracer::tolerance`]) whose reset at a point inside the box
//!   rises above the threshold `T`: other parameters compensate it,
//!   `InCombination`. The ends of the box only rule out `NonIdentifiable`.
//!
//! An evaluation that fails counts as an infeasible reset. Every verdict
//! holds with the other parameters at their fitted values.

use commol_calibration::{CalibrationParameterType, OptimizationObjective};
use commol_core::SimulationEngine;

use crate::profiles::tracer::{ON_BOUND, Profile, ProfilePoint, ProfileTracer};
use crate::profiles::verdict::ProfileVerdict;
use crate::transform::Transform;

/// The structural verdict of a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralReading {
    /// The profile is flat and resetting the other parameters raises it
    /// above the threshold: they compensate the parameter along it.
    InCombination,
    /// Every reset is feasible and within the evaluation floor of the fit:
    /// nothing the data read depends on the parameter.
    NonIdentifiable,
}

/// Why a profile gives no structural verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Undetermined {
    /// The threshold is at or below the tolerance.
    BelowTolerance,
    /// The profile is not flat, and the resets do not prove it flat.
    NotFlat,
    /// The profile goes below the fit by more than the evaluation floor:
    /// the fit is not the optimum.
    BelowFit,
    /// The engine is stochastic.
    Stochastic,
    /// Every reset stays within the evaluation floor, but this free
    /// parameter, by index, sits on its bound at the fit: what lies past the
    /// bound is not seen.
    OnBound(usize),
    /// The resets neither rise above the threshold nor stay within the
    /// evaluation floor.
    Undecided,
    /// The evaluation floor cannot be computed, as for a loss with no term
    /// derivatives.
    NoNoiseFloor,
}

impl Undetermined {
    /// The name of the reason: `"below_tolerance"`, `"not_flat"`,
    /// `"below_fit"`, `"stochastic"`, `"on_bound"`, `"undecided"` or
    /// `"no_noise_floor"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::BelowTolerance => "below_tolerance",
            Self::NotFlat => "not_flat",
            Self::BelowFit => "below_fit",
            Self::Stochastic => "stochastic",
            Self::OnBound(_) => "on_bound",
            Self::Undecided => "undecided",
            Self::NoNoiseFloor => "no_noise_floor",
        }
    }
}

/// What the resets of a profile say.
#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    /// The structural verdict, or why there is none.
    pub verdict: Result<StructuralReading, Undetermined>,
    /// The largest rise of a feasible reset over the objective at the fit;
    /// `None` when none ran or none was feasible.
    pub largest_rise: Option<f64>,
    /// The free parameters, by index, that compensate the parameter:
    /// resetting one alone, at its point of largest displacement, raises
    /// the objective above the threshold. Noise parameters of observation
    /// models are never named.
    pub compensating: Vec<usize>,
    /// The largest rise of resetting every free parameter not named, on each
    /// side together, when it is above the evaluation floor: the list of
    /// compensating parameters may be incomplete. Infinite for a reset that
    /// is not feasible.
    pub joint_rise: Option<f64>,
    /// The resets that were not feasible, or whose evaluation failed.
    pub infeasible: usize,
}

impl Structure {
    fn undetermined(why: Undetermined) -> Self {
        Self {
            verdict: Err(why),
            largest_rise: None,
            compensating: Vec::new(),
            joint_rise: None,
            infeasible: 0,
        }
    }

    /// Whether the resets prove the parameter non-identifiable.
    pub fn non_identifiable(&self) -> bool {
        self.verdict == Ok(StructuralReading::NonIdentifiable)
    }
}

impl<E> ProfileTracer<'_, E>
where
    E: SimulationEngine + Send + Sync,
{
    /// The objective at `values`, or `None` when the point is not feasible
    /// or its evaluation fails.
    fn evaluate(&self, objective: &OptimizationObjective<'_, E>, values: &[f64]) -> Option<f64> {
        objective
            .score(values)
            .ok()
            .and_then(|score| score.feasible.then_some(score.cost))
    }

    /// What the profile of the parameter at `index` says about the
    /// structure of the problem (module documentation).
    ///
    /// The resets run at each point of the profile inside the box but the
    /// fit, and at both ends of the box of the parameter. A compensating
    /// parameter is named when resetting it alone, at the point where it
    /// moves most on the unbounded scale of [`Transform`], raises the
    /// objective above the threshold; the parameters not named are then
    /// reset together at the outermost point inside the box of each side,
    /// compared with the evaluation floor.
    pub fn read_structure(&self, index: usize, profile: &Profile) -> Structure {
        let tolerance = self.tolerance();
        let threshold = profile.threshold;
        if threshold <= tolerance {
            return Structure::undetermined(Undetermined::BelowTolerance);
        }
        let shape = profile.verdict(tolerance);
        let flat = shape == Some(ProfileVerdict::StructurallyNonIdentifiable);
        if shape == Some(ProfileVerdict::Identifiable) {
            return Structure::undetermined(Undetermined::NotFlat);
        }
        let not_read =
            |why| Structure::undetermined(if flat { why } else { Undetermined::NotFlat });
        if self.stochastic {
            return not_read(Undetermined::Stochastic);
        }
        let Some(floor) = self.evaluation_floor() else {
            return Structure::undetermined(Undetermined::NoNoiseFloor);
        };

        let objective = OptimizationObjective::new(self.problem);
        let bounds = self.problem.parameter_bounds();
        let inside: Vec<&ProfilePoint> = profile
            .points
            .iter()
            .filter(|p| p.on_bound.is_none() && !(p.value == profile.center && p.statistic == 0.0))
            .collect();
        let (lo, hi) = bounds[index];
        let reset_at = |value: f64| {
            let mut values = self.fit.clone();
            values[index] = value;
            self.evaluate(&objective, &values)
                .map(|cost| cost - self.fit_loss)
        };
        let inside_rises: Vec<Option<f64>> = inside.iter().map(|p| reset_at(p.value)).collect();
        let end_rises = [reset_at(lo), reset_at(hi)];
        let all = inside_rises.iter().chain(&end_rises);
        let infeasible = all.clone().filter(|rise| rise.is_none()).count();
        let largest_rise = all.clone().flatten().copied().reduce(f64::max);
        let within_floor = infeasible == 0 && all.flatten().all(|rise| rise.abs() <= floor);

        let free = |i: usize| i != index;
        let on_bound_at_fit = (0..self.fit.len()).find(|&i| {
            let (lo, hi) = bounds[i];
            let tolerance = ON_BOUND * (hi - lo);
            free(i) && (self.fit[i] - lo <= tolerance || hi - self.fit[i] <= tolerance)
        });
        let rises_above = inside_rises.iter().flatten().any(|&rise| rise > threshold);
        let verdict = if within_floor {
            match on_bound_at_fit {
                Some(i) => Err(Undetermined::OnBound(i)),
                None => Ok(StructuralReading::NonIdentifiable),
            }
        } else if profile.lowest() < -floor {
            Err(Undetermined::BelowFit)
        } else if !flat {
            Err(Undetermined::NotFlat)
        } else if rises_above {
            Ok(StructuralReading::InCombination)
        } else {
            Err(Undetermined::Undecided)
        };
        let mut structure = Structure {
            verdict,
            largest_rise,
            compensating: Vec::new(),
            joint_rise: None,
            infeasible,
        };
        if structure.verdict == Ok(StructuralReading::InCombination) {
            self.name_compensating(index, profile, &inside, threshold, floor, &mut structure);
        }
        structure
    }

    /// Name the parameters that compensate the one at `index`, and check
    /// that resetting the others together keeps the objective of the point,
    /// within the evaluation floor `floor`.
    fn name_compensating(
        &self,
        index: usize,
        profile: &Profile,
        inside: &[&ProfilePoint],
        threshold: f64,
        floor: f64,
        structure: &mut Structure,
    ) {
        let objective = OptimizationObjective::new(self.problem);
        // The objective with the parameters `reset` at their fitted values,
        // less the objective of the point: two evaluations, no optimizer.
        let rise = |at: &ProfilePoint, reset: &[usize]| {
            let mut values = at.parameters.clone();
            for &i in reset {
                values[i] = self.fit[i];
            }
            match (
                self.evaluate(&objective, &values),
                self.evaluate(&objective, &at.parameters),
            ) {
                (Some(reset), Some(point)) => reset - point,
                _ => f64::INFINITY,
            }
        };
        let transform = Transform::new(&self.problem.parameter_bounds());
        let fit = transform.to_unbounded(&self.fit);
        let parameters = self.problem.parameters();
        for i in (0..self.fit.len()).filter(|&i| i != index) {
            if parameters[i].parameter_type == CalibrationParameterType::Noise {
                continue;
            }
            let displacement =
                |p: &&&ProfilePoint| (transform.to_unbounded(&p.parameters)[i] - fit[i]).abs();
            let point = inside
                .iter()
                .max_by(|a, b| displacement(a).total_cmp(&displacement(b)))
                .expect("a reset inside the box rises above the threshold");
            let reset = rise(point, &[i]);
            if reset.is_finite() && reset > threshold {
                structure.compensating.push(i);
            }
        }
        let unnamed: Vec<usize> = (0..self.fit.len())
            .filter(|&i| i != index && !structure.compensating.contains(&i))
            .collect();
        if unnamed.is_empty() {
            return;
        }
        let joint = profile
            .sides()
            .iter()
            .filter_map(|side| side.iter().rev().find(|p| p.on_bound.is_none()))
            .map(|outermost| rise(outermost, &unnamed))
            .fold(f64::NEG_INFINITY, f64::max);
        if joint > floor {
            structure.joint_rise = Some(joint);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_reason_has_a_name() {
        let names = [
            Undetermined::BelowTolerance,
            Undetermined::NotFlat,
            Undetermined::BelowFit,
            Undetermined::Stochastic,
            Undetermined::OnBound(2),
            Undetermined::Undecided,
            Undetermined::NoNoiseFloor,
        ]
        .map(Undetermined::name);
        assert_eq!(
            names,
            [
                "below_tolerance",
                "not_flat",
                "below_fit",
                "stochastic",
                "on_bound",
                "undecided",
                "no_noise_floor"
            ]
        );
    }
}
