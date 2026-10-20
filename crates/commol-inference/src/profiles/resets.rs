//! What a flat profile says about the structure of a problem.
//!
//! A profile of a parameter is flat for one of two reasons: other
//! parameters compensate it along the profile, or nothing the data read
//! depends on it. Resetting the other parameters to their fitted values
//! tells them apart: where they compensate, the reset raises the statistic
//! above the threshold `T`; where nothing depends on the parameter, it
//! raises it by no more than the tolerance `ε`
//! ([`ProfileTracer::tolerance`]). Each rise is measured against the profile
//! at the same point. Every verdict holds with the other parameters at their
//! fitted values.

use commol_calibration::{CalibrationParameterType, OptimizationObjective};
use commol_core::SimulationEngine;

use crate::profiles::tracer::{ON_BOUND, Profile, ProfilePoint, ProfileTracer};
use crate::profiles::verdict::ProfileVerdict;
use crate::transform::Transform;

/// The structural verdict of a flat profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralReading {
    /// Resetting the other parameters raises the profile above the
    /// threshold: they compensate the parameter along its profile.
    InCombination,
    /// Every reset is feasible and raises the profile by no more than the
    /// tolerance: nothing the data read depends on the parameter.
    NonIdentifiable,
}

/// Why a profile gives no structural verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Undetermined {
    /// The threshold is at or below the tolerance.
    BelowTolerance,
    /// The profile is not flat.
    NotFlat,
    /// The profile goes below the fit by more than the tolerance: the fit is
    /// not the optimum.
    BelowFit,
    /// The engine is stochastic.
    Stochastic,
    /// Every reset stays within the tolerance, but this free parameter, by
    /// index, sits on its bound at the fit: what lies past the bound is not
    /// seen.
    OnBound(usize),
    /// The resets neither rise above the threshold nor stay feasible and
    /// within the tolerance.
    Undecided,
}

impl Undetermined {
    /// The name of the reason: `"below_tolerance"`, `"not_flat"`,
    /// `"below_fit"`, `"stochastic"`, `"on_bound"` or `"undecided"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::BelowTolerance => "below_tolerance",
            Self::NotFlat => "not_flat",
            Self::BelowFit => "below_fit",
            Self::Stochastic => "stochastic",
            Self::OnBound(_) => "on_bound",
            Self::Undecided => "undecided",
        }
    }
}

/// What the resets of a profile say.
#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    /// The structural verdict, or why there is none.
    pub verdict: Result<StructuralReading, Undetermined>,
    /// The largest rise of a feasible reset; `None` when none ran or none
    /// was feasible.
    pub largest_rise: Option<f64>,
    /// The free parameters, by index, that compensate the parameter: resetting
    /// one alone, at its point of largest displacement, raises the profile
    /// above the threshold. Noise parameters of observation models are never
    /// named.
    pub compensating: Vec<usize>,
    /// The largest rise of resetting every free parameter not named, on each
    /// side together, when it is above the tolerance: the list of
    /// compensating parameters may be incomplete. Infinite for a reset that
    /// is not feasible.
    pub joint_rise: Option<f64>,
    /// The resets that were not feasible.
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
}

impl<E> ProfileTracer<'_, E>
where
    E: SimulationEngine + Send + Sync,
{
    /// What the profile of the parameter at `index` says about the
    /// structure of the problem.
    ///
    /// At each point of the profile inside the box, and at each end of the
    /// box of the parameter where the profile has a point, the other
    /// parameters are reset to their fitted values and the objective is
    /// evaluated once; the rise is the reset value less the profile at that
    /// point. A feasible rise above the threshold gives `InCombination`;
    /// feasible rises all within the tolerance give `NonIdentifiable`,
    /// unless a free parameter sits on its bound at the fit. A compensating
    /// parameter is named when resetting it alone, at the point where it
    /// moves most on the unbounded scale of [`Transform`], raises the
    /// profile above the threshold; the parameters not named are then reset
    /// together at the outermost point of each side.
    ///
    /// # Errors
    ///
    /// Returns the message of an evaluation that fails for a reason no point
    /// avoids.
    pub fn read_structure(&self, index: usize, profile: &Profile) -> Result<Structure, String> {
        let tolerance = self.tolerance();
        let threshold = profile.threshold;
        if threshold <= tolerance {
            return Ok(Structure::undetermined(Undetermined::BelowTolerance));
        }
        let below_fit = profile.lowest() < -tolerance;
        if profile.verdict(tolerance) != Some(ProfileVerdict::StructurallyNonIdentifiable) {
            return Ok(Structure::undetermined(if below_fit {
                Undetermined::BelowFit
            } else {
                Undetermined::NotFlat
            }));
        }
        if self.stochastic {
            return Ok(Structure::undetermined(Undetermined::Stochastic));
        }
        if below_fit {
            return Ok(Structure::undetermined(Undetermined::BelowFit));
        }

        let objective = OptimizationObjective::new(self.problem);
        let reset = |values: &[f64], at: &ProfilePoint| -> Result<Option<f64>, String> {
            let score = objective.score(values).map_err(|error| error.to_string())?;
            Ok(score
                .feasible
                .then_some(score.cost - (self.fit_loss + at.statistic)))
        };
        let bounds = self.problem.parameter_bounds();
        let inside: Vec<&ProfilePoint> = profile
            .points
            .iter()
            .filter(|p| p.on_bound.is_none() && !(p.value == profile.center && p.statistic == 0.0))
            .collect();
        let (lo, hi) = bounds[index];
        let ends: Vec<(f64, &ProfilePoint)> = [lo, hi]
            .into_iter()
            .filter_map(|end| {
                profile
                    .points
                    .iter()
                    .find(|p| (p.value - end).abs() <= ON_BOUND * (hi - lo))
                    .map(|p| (end, p))
            })
            .collect();
        let mut rises = Vec::new();
        let mut infeasible = 0;
        for (value, at) in inside.iter().map(|p| (p.value, *p)).chain(ends) {
            let mut values = self.fit.clone();
            values[index] = value;
            match reset(&values, at)? {
                Some(rise) => rises.push(rise),
                None => infeasible += 1,
            }
        }
        let largest_rise = rises.iter().copied().reduce(f64::max);
        let largest = largest_rise.unwrap_or(f64::NEG_INFINITY);
        let free = |i: usize| i != index;
        let on_bound_at_fit = (0..self.fit.len()).find(|&i| {
            let (lo, hi) = bounds[i];
            let tolerance = ON_BOUND * (hi - lo);
            free(i) && (self.fit[i] - lo <= tolerance || hi - self.fit[i] <= tolerance)
        });
        let verdict = if largest > threshold {
            Ok(StructuralReading::InCombination)
        } else if infeasible == 0 && largest <= tolerance {
            match on_bound_at_fit {
                Some(i) => Err(Undetermined::OnBound(i)),
                None => Ok(StructuralReading::NonIdentifiable),
            }
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
        if verdict == Ok(StructuralReading::InCombination) {
            self.name_compensating(
                index,
                profile,
                &inside,
                threshold,
                tolerance,
                &mut structure,
            )?;
        }
        Ok(structure)
    }

    /// Name the parameters that compensate the one at `index`, and check
    /// that resetting the others together keeps the profile.
    fn name_compensating(
        &self,
        index: usize,
        profile: &Profile,
        inside: &[&ProfilePoint],
        threshold: f64,
        tolerance: f64,
        structure: &mut Structure,
    ) -> Result<(), String> {
        let objective = OptimizationObjective::new(self.problem);
        let rise = |values: &[f64], at: &ProfilePoint| -> Result<f64, String> {
            let score = objective.score(values).map_err(|error| error.to_string())?;
            Ok(if score.feasible {
                score.cost - (self.fit_loss + at.statistic)
            } else {
                f64::INFINITY
            })
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
            let Some(point) = inside
                .iter()
                .max_by(|a, b| displacement(a).total_cmp(&displacement(b)))
            else {
                continue;
            };
            if displacement(&point) == 0.0 {
                continue;
            }
            let mut values = point.parameters.clone();
            values[i] = self.fit[i];
            let reset = rise(&values, point)?;
            if reset.is_finite() && reset > threshold {
                structure.compensating.push(i);
            }
        }
        let unnamed: Vec<usize> = (0..self.fit.len())
            .filter(|&i| i != index && !structure.compensating.contains(&i))
            .collect();
        if unnamed.is_empty() {
            return Ok(());
        }
        let mut joint = f64::NEG_INFINITY;
        for side in profile.sides() {
            let Some(outermost) = side.iter().rev().find(|p| p.on_bound.is_none()) else {
                continue;
            };
            let mut values = outermost.parameters.clone();
            for &i in &unnamed {
                values[i] = self.fit[i];
            }
            joint = joint.max(rise(&values, outermost)?);
        }
        if joint > tolerance {
            structure.joint_rise = Some(joint);
        }
        Ok(())
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
                "undecided"
            ]
        );
    }
}
