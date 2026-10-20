//! The verdict of a profile: what its shape says about its quantity.

use crate::profiles::tracer::{Profile, ProfileEnd};

/// What the shape of a profile says about its quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileVerdict {
    /// The profile crosses the threshold on both sides.
    Identifiable,
    /// The profile stays below the threshold on at least one side, up to a
    /// bound, the end of the reachable predictions or `max_steps`.
    PracticallyNonIdentifiable,
    /// The profile is flat within the numerical tolerance over its whole
    /// range: both sides reach a bound, or the end of the reachable
    /// predictions.
    StructurallyNonIdentifiable,
}

impl ProfileVerdict {
    /// The name of the verdict: `"identifiable"`,
    /// `"practically_non_identifiable"` or `"structurally_non_identifiable"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Identifiable => "identifiable",
            Self::PracticallyNonIdentifiable => "practically_non_identifiable",
            Self::StructurallyNonIdentifiable => "structurally_non_identifiable",
        }
    }
}

impl Profile {
    /// The verdict of the profile, whose statistic is known to `tolerance`;
    /// `None` when the threshold is below the tolerance, where a rise
    /// cannot be told from numerical error.
    pub fn verdict(&self, tolerance: f64) -> Option<ProfileVerdict> {
        if self.threshold < tolerance {
            return None;
        }
        let whole_range = self
            .ends
            .iter()
            .all(|end| matches!(end, ProfileEnd::Bound | ProfileEnd::Unreachable));
        let flat = self
            .points
            .iter()
            .all(|point| point.statistic.abs() <= tolerance);
        Some(if self.ends == [ProfileEnd::Crossed; 2] {
            ProfileVerdict::Identifiable
        } else if whole_range && flat {
            ProfileVerdict::StructurallyNonIdentifiable
        } else {
            ProfileVerdict::PracticallyNonIdentifiable
        })
    }

    /// The lowest statistic of the profile: below `−tolerance`, a refit
    /// found a lower objective than the fit, which is then not the optimum.
    pub fn lowest(&self) -> f64 {
        self.points
            .iter()
            .map(|point| point.statistic)
            .fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::tracer::ProfilePoint;

    fn profile(statistics: &[f64], ends: [ProfileEnd; 2], threshold: f64) -> Profile {
        Profile {
            center: 0.0,
            threshold,
            points: statistics
                .iter()
                .enumerate()
                .map(|(i, &statistic)| ProfilePoint {
                    value: i as f64,
                    statistic,
                    parameters: vec![i as f64],
                    multiplier: None,
                    converged: true,
                })
                .collect(),
            ends,
            interval: [None, None],
            evaluations: 0,
        }
    }

    #[test]
    fn the_ends_and_the_flatness_of_a_profile_give_its_verdict() {
        use ProfileEnd::{Bound, Crossed, MaxSteps, Unreachable};
        let verdict = |statistics: &[f64], ends| profile(statistics, ends, 2.0).verdict(1e-6);

        assert_eq!(
            verdict(&[3.0, 0.0, 2.5], [Crossed, Crossed]),
            Some(ProfileVerdict::Identifiable)
        );
        assert_eq!(
            verdict(&[3.0, 0.0, 0.5], [Crossed, Bound]),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
        assert_eq!(
            verdict(&[0.5, 0.0, 0.5], [MaxSteps, Bound]),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
        assert_eq!(
            verdict(&[1e-7, 0.0, -1e-7], [Bound, Unreachable]),
            Some(ProfileVerdict::StructurallyNonIdentifiable)
        );
        // Flat, but not over the whole range.
        assert_eq!(
            verdict(&[1e-7, 0.0, 1e-7], [MaxSteps, Bound]),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
        // A rise above the tolerance up to both bounds.
        assert_eq!(
            verdict(&[1e-3, 0.0, 1e-7], [Bound, Bound]),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
    }

    #[test]
    fn a_threshold_below_the_tolerance_gives_no_verdict() {
        let flat = profile(&[0.0, 0.0, 0.0], [ProfileEnd::Bound; 2], 1e-9);
        assert_eq!(flat.verdict(1e-6), None);
        assert_eq!(
            flat.verdict(1e-12),
            Some(ProfileVerdict::StructurallyNonIdentifiable)
        );
    }

    #[test]
    fn the_lowest_statistic_shows_a_refit_below_the_fit() {
        let profile = profile(&[0.5, 0.0, -0.25], [ProfileEnd::Crossed; 2], 2.0);
        assert_eq!(profile.lowest(), -0.25);
        assert_eq!(
            [
                ProfileVerdict::Identifiable,
                ProfileVerdict::PracticallyNonIdentifiable,
                ProfileVerdict::StructurallyNonIdentifiable
            ]
            .map(ProfileVerdict::name),
            [
                "identifiable",
                "practically_non_identifiable",
                "structurally_non_identifiable"
            ]
        );
    }
}
