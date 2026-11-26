//! The verdict of a profile: what its shape says about its quantity.

use crate::profiles::tracer::{Profile, ProfileEnd, ProfilePoint};

/// What the shape of a profile says about its quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileVerdict {
    /// The profile crosses the threshold on both sides.
    Identifiable,
    /// The profile stays below the threshold on at least one side, up to a
    /// bound or the end of the reachable predictions, and its points do not
    /// show it flat over its whole range. With a stochastic engine, a flat
    /// profile too
    /// ([`ProfileTracer::verdict`](crate::profiles::ProfileTracer::verdict)).
    PracticallyNonIdentifiable,
    /// The profile is flat within the numerical tolerance over its whole
    /// range inside the box of the parameters: each side reaches a bound or
    /// the end of the reachable predictions, or rises only where a free
    /// parameter sits on its bound.
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

/// Why a profile gives no verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoVerdict {
    /// The threshold is at or below the tolerance, where a rise cannot be
    /// told from numerical error.
    BelowTolerance,
    /// A side ends at `max_steps`, and a longer side could change the
    /// verdict.
    MaxSteps,
}

impl NoVerdict {
    /// The name of the reason: `"below_tolerance"` or `"max_steps"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::BelowTolerance => "below_tolerance",
            Self::MaxSteps => "max_steps",
        }
    }
}

impl Profile {
    /// The verdict of the profile, whose statistic is known to `tolerance`;
    /// `None` when it gives none ([`Self::reading`] says why).
    pub fn verdict(&self, tolerance: f64) -> Option<ProfileVerdict> {
        self.reading(tolerance).ok()
    }

    /// The verdict of the profile, whose statistic is known to `tolerance`,
    /// or why it gives none.
    ///
    /// A point where a free parameter sits on its bound
    /// ([`ProfilePoint::on_bound`]) is outside the ridge the data leave
    /// flat: its rise comes from the box of the parameters. The profile is
    /// structurally non-identifiable when each side is flat over its points
    /// inside the box, has one past the fit, and ends at a bound, at the end
    /// of the reachable predictions, or on a point outside the box. A side
    /// whose crossing comes from the box ([`Profile::box_crossings`]) counts
    /// as ending at a bound, so it is never `identifiable`.
    ///
    /// A profile with a side that ends at `max_steps` has a verdict only
    /// when no longer side can change it: the other side ends at a bound or
    /// at the end of the reachable predictions, and the profile can no
    /// longer be flat, as when a side already rises above the tolerance.
    /// That verdict is `practically_non_identifiable`.
    pub fn reading(&self, tolerance: f64) -> Result<ProfileVerdict, NoVerdict> {
        if self.threshold <= tolerance {
            return Err(NoVerdict::BelowTolerance);
        }
        let ends = self.effective_ends();
        // A side at `max_steps` can still be flat to its end when it is
        // flat so far.
        let can_be_flat = self.sides().iter().zip(ends).all(|(side, end)| {
            let inside: Vec<&ProfilePoint> = side
                .iter()
                .copied()
                .filter(|p| p.on_bound.is_none())
                .collect();
            let ends_outside = match end {
                ProfileEnd::Bound | ProfileEnd::Unreachable => true,
                ProfileEnd::Crossed => side.last().is_some_and(|p| p.on_bound.is_some()),
                ProfileEnd::MaxSteps => true,
            };
            let reaches_out = !inside.is_empty()
                || end == ProfileEnd::MaxSteps
                || (side.is_empty() && end == ProfileEnd::Bound);
            ends_outside && reaches_out && inside.iter().all(|p| p.statistic.abs() <= tolerance)
        });
        if ends.contains(&ProfileEnd::MaxSteps) {
            // A longer side may cross, rise, or stay flat to its end.
            let open = ends
                .iter()
                .any(|end| matches!(end, ProfileEnd::Bound | ProfileEnd::Unreachable));
            return if open && !can_be_flat {
                Ok(ProfileVerdict::PracticallyNonIdentifiable)
            } else {
                Err(NoVerdict::MaxSteps)
            };
        }
        Ok(if can_be_flat {
            ProfileVerdict::StructurallyNonIdentifiable
        } else if ends == [ProfileEnd::Crossed; 2] {
            ProfileVerdict::Identifiable
        } else {
            ProfileVerdict::PracticallyNonIdentifiable
        })
    }

    /// The ends of the sides, a crossing that comes from the box read as a
    /// bound.
    pub fn effective_ends(&self) -> [ProfileEnd; 2] {
        [0, 1].map(|side| match self.box_crossings[side] {
            Some(_) => ProfileEnd::Bound,
            None => self.ends[side],
        })
    }

    /// Where each side leaves the box of the parameters: the value of its
    /// first point, outward from the fit, with a free parameter on its
    /// bound, and that parameter; `None` on a side that stays inside.
    pub fn box_exits(&self) -> [Option<(f64, usize)>; 2] {
        self.sides().map(|side| {
            side.iter()
                .find_map(|p| p.on_bound.map(|index| (p.value, index)))
        })
    }

    /// The points of the side below and of the side above the fit, outward
    /// from it.
    pub(crate) fn sides(&self) -> [Vec<&ProfilePoint>; 2] {
        let fit = self
            .points
            .iter()
            .position(|p| p.value == self.center && p.statistic == 0.0)
            .unwrap_or(0);
        [
            self.points[..fit].iter().rev().collect(),
            self.points[fit + 1..].iter().collect(),
        ]
    }

    /// The lowest statistic of the profile: below `−tolerance`, a refit
    /// found a lower objective than the fit, which is then not the optimum.
    pub fn lowest(&self) -> f64 {
        self.points
            .iter()
            .map(|point| point.statistic)
            .fold(f64::INFINITY, f64::min)
    }

    /// Whether a refit is below the fit by more than `tolerance`
    /// ([`Self::lowest`] below `−tolerance`): the fit is not the optimum.
    pub fn below_fit(&self, tolerance: f64) -> bool {
        self.lowest() < -tolerance
    }

    /// Whether a refit at the fit lands farther from it than `tolerance`
    /// ([`Profile::refit_error`] above it): the refits are less accurate
    /// than the tolerance.
    pub fn inaccurate_refit(&self, tolerance: f64) -> bool {
        self.refit_error.is_some_and(|error| error > tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A profile whose fit is its first point of statistic 0.
    fn profile(statistics: &[f64], ends: [ProfileEnd; 2], threshold: f64) -> Profile {
        Profile {
            center: statistics.iter().position(|&s| s == 0.0).unwrap_or(0) as f64,
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
                    on_bound: None,
                })
                .collect(),
            ends,
            interval: [None, None],
            evaluations: 0,
            box_crossings: [None, None],
            refit_error: None,
        }
    }

    /// A profile whose fit is the point at `center`, and whose free
    /// parameter `on_bound[i]` is on its bound at point `i`.
    fn boxed(
        statistics: &[f64],
        on_bound: &[Option<usize>],
        center: usize,
        ends: [ProfileEnd; 2],
    ) -> Profile {
        let mut profile = profile(statistics, ends, 2.0);
        profile.center = center as f64;
        for (point, &bound) in profile.points.iter_mut().zip(on_bound) {
            point.on_bound = bound;
        }
        profile
    }

    #[test]
    fn a_profile_flat_until_the_ridge_leaves_the_box_is_structural() {
        use ProfileEnd::{Bound, Crossed};
        let edges = [Some(1), None, None, None, Some(1)];
        // The rise at both ends comes from parameter 1 on its bound.
        let below_threshold = boxed(&[0.03, 1e-9, 0.0, -1e-9, 0.03], &edges, 2, [Bound; 2]);
        assert_eq!(
            below_threshold.verdict(1e-6),
            Some(ProfileVerdict::StructurallyNonIdentifiable)
        );
        assert_eq!(
            below_threshold.box_exits(),
            [Some((0.0, 1)), Some((4.0, 1))]
        );
        let past_threshold = boxed(&[3.0, 0.0, 0.0, 0.0, 3.0], &edges, 2, [Crossed; 2]);
        assert_eq!(
            past_threshold.verdict(1e-6),
            Some(ProfileVerdict::StructurallyNonIdentifiable)
        );
    }

    #[test]
    fn a_profile_that_rises_inside_the_box_keeps_its_verdict() {
        use ProfileEnd::{Bound, Crossed};
        // A rise before the ridge leaves the box.
        let rises = boxed(
            &[0.04, 0.02, 0.0, 1e-9, 0.04],
            &[Some(1), None, None, None, Some(1)],
            2,
            [Bound; 2],
        );
        assert_eq!(
            rises.verdict(1e-6),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
        // No point inside the box below the fit.
        let no_inside = boxed(
            &[0.03, 0.0, 1e-9, 0.03],
            &[Some(1), None, None, Some(1)],
            1,
            [Bound; 2],
        );
        assert_eq!(
            no_inside.verdict(1e-6),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
        // Crossing inside the box on both sides.
        let crosses = boxed(&[3.0, 0.0, 2.5], &[None; 3], 1, [Crossed; 2]);
        assert_eq!(crosses.verdict(1e-6), Some(ProfileVerdict::Identifiable));
        assert_eq!(crosses.box_exits(), [None, None]);
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
            verdict(&[1e-7, 0.0, -1e-7], [Bound, Unreachable]),
            Some(ProfileVerdict::StructurallyNonIdentifiable)
        );
        // A side at `max_steps` that a longer side cannot change: the other
        // side does not cross, and a side already rises.
        assert_eq!(
            verdict(&[0.5, 0.0, 0.5], [MaxSteps, Bound]),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
        assert_eq!(
            verdict(&[0.5, 0.0, 1e-7], [MaxSteps, Unreachable]),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
        // A side at `max_steps` that a longer side can change: it may cross,
        // or stay flat to its end.
        for (statistics, ends) in [
            ([0.5, 0.0, 3.0], [MaxSteps, Crossed]),
            ([0.5, 0.0, 0.5], [MaxSteps, MaxSteps]),
            ([1e-7, 0.0, 1e-7], [MaxSteps, Bound]),
        ] {
            let profile = profile(&statistics, ends, 2.0);
            assert_eq!(profile.verdict(1e-6), None, "{ends:?}");
            assert_eq!(profile.reading(1e-6), Err(NoVerdict::MaxSteps));
        }
        // A rise above the tolerance up to both bounds.
        assert_eq!(
            verdict(&[1e-3, 0.0, 1e-7], [Bound, Bound]),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
    }

    #[test]
    fn a_side_at_max_steps_beside_a_side_outside_the_box_is_practical() {
        use ProfileEnd::{Bound, MaxSteps, Unreachable};
        // The side below has no point inside the box, so the profile can no
        // longer be flat, though the side above is flat so far.
        for end in [Bound, Unreachable] {
            let profile = boxed(
                &[1e-9, 0.0, 1e-9],
                &[Some(1), None, None],
                1,
                [end, MaxSteps],
            );
            assert_eq!(
                profile.verdict(1e-6),
                Some(ProfileVerdict::PracticallyNonIdentifiable),
                "{end:?}"
            );
            assert_eq!(profile.reading(1e-6).err(), None);
        }
    }

    #[test]
    fn a_crossing_made_by_the_box_counts_as_a_bound() {
        let mut crossed = profile(&[3.0, 0.0, 2.5], [ProfileEnd::Crossed; 2], 2.0);
        assert_eq!(crossed.verdict(1e-6), Some(ProfileVerdict::Identifiable));
        crossed.box_crossings = [Some(1), None];
        assert_eq!(
            crossed.effective_ends(),
            [ProfileEnd::Bound, ProfileEnd::Crossed]
        );
        assert_eq!(
            crossed.verdict(1e-6),
            Some(ProfileVerdict::PracticallyNonIdentifiable)
        );
    }

    #[test]
    fn a_threshold_below_the_tolerance_gives_no_verdict() {
        let flat = profile(&[0.0, 0.0, 0.0], [ProfileEnd::Bound; 2], 1e-9);
        assert_eq!(flat.verdict(1e-6), None);
        assert_eq!(flat.verdict(1e-9), None);
        assert_eq!(flat.reading(1e-9), Err(NoVerdict::BelowTolerance));
        assert_eq!(
            flat.verdict(1e-12),
            Some(ProfileVerdict::StructurallyNonIdentifiable)
        );
        assert_eq!(flat.reading(1e-12).err(), None);
        assert_eq!(
            [NoVerdict::BelowTolerance, NoVerdict::MaxSteps].map(NoVerdict::name),
            ["below_tolerance", "max_steps"]
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
