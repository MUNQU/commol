//! Pareto-front preference selection.

use super::ParetoSolution;

pub(super) struct ObjectivePoint {
    pub ci_width: f64,
    pub coverage: f64,
}

pub(super) fn select_pareto_solution_by_preference(
    pareto_front: &[ParetoSolution],
    preference: f64,
    confidence_level: f64,
) -> usize {
    let points: Vec<_> = pareto_front
        .iter()
        .map(|solution| ObjectivePoint {
            ci_width: solution.ci_width,
            coverage: solution.coverage,
        })
        .collect();
    select_objective_point_by_preference(&points, preference, confidence_level)
}

pub(super) fn select_objective_point_by_preference(
    points: &[ObjectivePoint],
    preference: f64,
    confidence_level: f64,
) -> usize {
    if points.is_empty() || points.len() == 1 {
        return 0;
    }

    let preference = preference.clamp(0.0, 1.0);
    if preference <= 0.05 {
        return points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.ci_width
                    .partial_cmp(&b.ci_width)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    if preference >= 0.95 {
        return points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.coverage
                    .partial_cmp(&b.coverage)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    let weight = 1.0 - preference;
    let max_coverage = points
        .iter()
        .map(|point| point.coverage)
        .fold(f64::NEG_INFINITY, f64::max);
    let coverage_floor = confidence_level.min(max_coverage);
    let eligible_indices: Vec<usize> = points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| (point.coverage + f64::EPSILON >= coverage_floor).then_some(idx))
        .collect();

    let (min_ci, max_ci) = points.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_ci, max_ci), point| (min_ci.min(point.ci_width), max_ci.max(point.ci_width)),
    );
    let (min_neg_coverage, max_neg_coverage) = points.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_cov, max_cov), point| {
            let neg_coverage = 1.0 - point.coverage;
            (min_cov.min(neg_coverage), max_cov.max(neg_coverage))
        },
    );
    let ci_range = max_ci - min_ci;
    let neg_coverage_range = max_neg_coverage - min_neg_coverage;

    eligible_indices
        .iter()
        .copied()
        .map(|idx| {
            let point = &points[idx];
            let normalized_ci = if ci_range > f64::EPSILON {
                (point.ci_width - min_ci) / ci_range
            } else {
                0.0
            };
            let neg_coverage = 1.0 - point.coverage;
            let normalized_neg_coverage = if neg_coverage_range > f64::EPSILON {
                (neg_coverage - min_neg_coverage) / neg_coverage_range
            } else {
                0.0
            };

            let score = weight * normalized_ci + (1.0 - weight) * normalized_neg_coverage;
            (idx, score)
        })
        .min_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
