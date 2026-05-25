//! Incremental subset-cost evaluation for greedy ensemble selection.

use argmin::core::CostFunction;
use argmin::solver::greedysubsetselection::SubsetSelectionCost;
use std::cell::RefCell;
use std::cmp::Ordering;

use super::EnsembleSelectionProblem;

thread_local! {
    static INCREMENTAL_CACHE: RefCell<IncrementalCache> =
        const { RefCell::new(IncrementalCache::new()) };
}

/// Sentinel meaning "this candidate is not in the current selection."
const POSITION_NONE: u32 = u32::MAX;

/// Per-thread cache mapping a problem + selection to the per-point sorted
/// candidate indices plus reverse rank lookups.
struct IncrementalCache {
    problem_id: u64,
    selection: Vec<usize>,
    ci_sorted: Vec<Vec<usize>>,
    ci_positions: Vec<Vec<u32>>,
    coverage_sorted: Vec<Vec<usize>>,
    coverage_positions: Vec<Vec<u32>>,
}

impl IncrementalCache {
    const fn new() -> Self {
        Self {
            problem_id: 0,
            selection: Vec::new(),
            ci_sorted: Vec::new(),
            ci_positions: Vec::new(),
            coverage_sorted: Vec::new(),
            coverage_positions: Vec::new(),
        }
    }
}

impl SubsetSelectionCost for EnsembleSelectionProblem {
    fn n_items(&self) -> usize {
        self.candidates.len()
    }

    fn cost_with_added(
        &self,
        current: &[usize],
        candidate: usize,
    ) -> Result<f64, argmin::core::Error> {
        // Greedy adds when `current` is empty / a single item. The cache + virtual
        // percentile path requires at least two existing selections to be meaningful.
        if current.len() < 2 {
            return self.fallback_cost_with_added(current, candidate);
        }
        Ok(self.incremental_scalar_cost(current, IncrementalOp::Add { candidate }))
    }

    fn cost_with_removed(
        &self,
        current: &[usize],
        removed: usize,
    ) -> Result<f64, argmin::core::Error> {
        if current.len() <= 2 {
            return self.fallback_cost_with_removed(current, removed);
        }
        Ok(self.incremental_scalar_cost(current, IncrementalOp::Remove { removed }))
    }

    fn cost_with_swapped(
        &self,
        current: &[usize],
        removed: usize,
        added: usize,
    ) -> Result<f64, argmin::core::Error> {
        if current.len() < 2 {
            return self.fallback_cost_with_swapped(current, removed, added);
        }
        Ok(self.incremental_scalar_cost(current, IncrementalOp::Swap { removed, added }))
    }
}

/// Local modification used by the incremental percentile path.
#[derive(Copy, Clone)]
enum IncrementalOp {
    Add { candidate: usize },
    Remove { removed: usize },
    Swap { removed: usize, added: usize },
}

impl IncrementalOp {
    fn new_len(self, current_len: usize) -> usize {
        match self {
            IncrementalOp::Add { .. } => current_len + 1,
            IncrementalOp::Remove { .. } => current_len - 1,
            IncrementalOp::Swap { .. } => current_len,
        }
    }
}

impl EnsembleSelectionProblem {
    fn fallback_cost_with_added(
        &self,
        current: &[usize],
        candidate: usize,
    ) -> Result<f64, argmin::core::Error> {
        let mut next = Vec::with_capacity(current.len() + 1);
        next.extend_from_slice(current);
        let insert_at = next.partition_point(|&i| i < candidate);
        next.insert(insert_at, candidate);
        self.cost(&next)
    }

    fn fallback_cost_with_removed(
        &self,
        current: &[usize],
        removed: usize,
    ) -> Result<f64, argmin::core::Error> {
        let next: Vec<usize> = current.iter().copied().filter(|&i| i != removed).collect();
        self.cost(&next)
    }

    fn fallback_cost_with_swapped(
        &self,
        current: &[usize],
        removed: usize,
        added: usize,
    ) -> Result<f64, argmin::core::Error> {
        let mut next: Vec<usize> = current.iter().copied().filter(|&i| i != removed).collect();
        let insert_at = next.partition_point(|&i| i < added);
        next.insert(insert_at, added);
        self.cost(&next)
    }

    /// Scalarized cost for an incremental modification. Caller must ensure the
    /// post-modification size is at least 2.
    fn incremental_scalar_cost(&self, current: &[usize], op: IncrementalOp) -> f64 {
        self.ensure_incremental_cache(current);
        INCREMENTAL_CACHE.with(|cell| {
            let cache = cell.borrow();
            let new_len = op.new_len(current.len());

            let (total_width, count) = cache
                .ci_sorted
                .iter()
                .zip(cache.ci_positions.iter())
                .zip(&self.ci_point_predictions)
                .fold(
                    (0.0_f64, 0_usize),
                    |(total, count), ((sorted, positions), point)| match virtual_ci_width(
                        sorted,
                        positions,
                        point,
                        op,
                        new_len,
                        self.lower_percentile,
                        self.upper_percentile,
                    ) {
                        Some(width) => (total + width, count + 1),
                        None => (total, count),
                    },
                );
            let ci_width_raw = if count > 0 {
                total_width / count as f64
            } else {
                f64::MAX
            };
            let normalized_ci_width = if self.max_ci_width > self.min_ci_width {
                ((ci_width_raw - self.min_ci_width) / (self.max_ci_width - self.min_ci_width))
                    .clamp(0.0, 1.0)
            } else {
                0.5
            };

            let coverage = if self.observed_values.is_empty() {
                0.0
            } else {
                let covered: usize = cache
                    .coverage_sorted
                    .iter()
                    .zip(cache.coverage_positions.iter())
                    .zip(&self.coverage_point_predictions)
                    .zip(&self.observed_values)
                    .map(|(((sorted, positions), point), &observed_value)| {
                        usize::from(virtual_point_covers(
                            sorted,
                            positions,
                            point,
                            observed_value,
                            op,
                            new_len,
                            self.lower_percentile,
                            self.upper_percentile,
                        ))
                    })
                    .sum();
                covered as f64 / self.observed_values.len() as f64
            };

            normalized_ci_width + (1.0 - coverage)
        })
    }

    /// Populate the thread-local cache so it matches `(self.problem_id, selection)`.
    fn ensure_incremental_cache(&self, selection: &[usize]) {
        INCREMENTAL_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if cache.problem_id == self.problem_id && cache.selection == selection {
                return;
            }

            let n_candidates = self.candidates.len();
            cache.problem_id = self.problem_id;
            cache.selection.clear();
            cache.selection.extend_from_slice(selection);

            cache
                .ci_sorted
                .resize_with(self.ci_point_predictions.len(), Vec::new);
            cache
                .ci_positions
                .resize_with(self.ci_point_predictions.len(), Vec::new);
            cache
                .coverage_sorted
                .resize_with(self.coverage_point_predictions.len(), Vec::new);
            cache
                .coverage_positions
                .resize_with(self.coverage_point_predictions.len(), Vec::new);

            let IncrementalCache {
                ci_sorted,
                ci_positions,
                coverage_sorted,
                coverage_positions,
                ..
            } = &mut *cache;

            rebuild_point_caches(
                &self.ci_point_predictions,
                ci_sorted,
                ci_positions,
                selection,
                n_candidates,
            );
            rebuild_point_caches(
                &self.coverage_point_predictions,
                coverage_sorted,
                coverage_positions,
                selection,
                n_candidates,
            );
        });
    }
}

fn rebuild_point_caches(
    point_predictions: &[Vec<f64>],
    sorted_per_point: &mut [Vec<usize>],
    positions_per_point: &mut [Vec<u32>],
    selection: &[usize],
    n_candidates: usize,
) {
    for ((point, sorted), positions) in point_predictions
        .iter()
        .zip(sorted_per_point.iter_mut())
        .zip(positions_per_point.iter_mut())
    {
        sorted.clear();
        sorted.extend_from_slice(selection);
        sorted.sort_by(|&a, &b| point[a].partial_cmp(&point[b]).unwrap_or(Ordering::Equal));

        positions.clear();
        positions.resize(n_candidates, POSITION_NONE);
        for (rank, &candidate_idx) in sorted.iter().enumerate() {
            positions[candidate_idx] = rank as u32;
        }
    }
}

#[derive(Copy, Clone)]
enum VirtualPlan {
    Add {
        added_value: f64,
        insert_pos: usize,
    },
    Remove {
        remove_pos: usize,
    },
    Swap {
        added_value: f64,
        remove_pos: usize,
        insert_pos_after_remove: usize,
    },
}

fn plan_virtual_modification(
    sorted: &[usize],
    positions: &[u32],
    point: &[f64],
    op: IncrementalOp,
) -> VirtualPlan {
    match op {
        IncrementalOp::Add { candidate } => {
            let added_value = point[candidate];
            VirtualPlan::Add {
                added_value,
                insert_pos: insertion_position(sorted, point, added_value),
            }
        }
        IncrementalOp::Remove { removed } => VirtualPlan::Remove {
            remove_pos: lookup_rank(positions, removed),
        },
        IncrementalOp::Swap { removed, added } => {
            let added_value = point[added];
            let remove_pos = lookup_rank(positions, removed);
            let raw_insert = insertion_position(sorted, point, added_value);
            let insert_pos_after_remove = if remove_pos < raw_insert {
                raw_insert - 1
            } else {
                raw_insert
            };
            VirtualPlan::Swap {
                added_value,
                remove_pos,
                insert_pos_after_remove,
            }
        }
    }
}

fn virtual_percentile(
    sorted: &[usize],
    point: &[f64],
    plan: VirtualPlan,
    new_len: usize,
    percentile: f64,
) -> f64 {
    debug_assert!(new_len >= 1);

    if new_len == 1 {
        return value_at_virtual_rank(sorted, point, plan, 0);
    }

    let position = (percentile / 100.0) * (new_len - 1) as f64;
    let lower_idx = position.floor() as usize;
    let upper_idx = position.ceil() as usize;
    let lower_value = value_at_virtual_rank(sorted, point, plan, lower_idx);
    if lower_idx == upper_idx {
        lower_value
    } else {
        let upper_value = value_at_virtual_rank(sorted, point, plan, upper_idx);
        let weight = position - lower_idx as f64;
        lower_value * (1.0 - weight) + upper_value * weight
    }
}

fn virtual_ci_width(
    sorted: &[usize],
    positions: &[u32],
    point: &[f64],
    op: IncrementalOp,
    new_len: usize,
    lower_percentile: f64,
    upper_percentile: f64,
) -> Option<f64> {
    if new_len < 2 || sorted.is_empty() {
        return None;
    }
    let plan = plan_virtual_modification(sorted, positions, point, op);
    let lower = virtual_percentile(sorted, point, plan, new_len, lower_percentile);
    let upper = virtual_percentile(sorted, point, plan, new_len, upper_percentile);
    Some(upper - lower)
}

#[allow(clippy::too_many_arguments)]
fn virtual_point_covers(
    sorted: &[usize],
    positions: &[u32],
    point: &[f64],
    observed_value: f64,
    op: IncrementalOp,
    new_len: usize,
    lower_percentile: f64,
    upper_percentile: f64,
) -> bool {
    if new_len < 2 || sorted.is_empty() {
        return false;
    }
    let plan = plan_virtual_modification(sorted, positions, point, op);
    let lower = virtual_percentile(sorted, point, plan, new_len, lower_percentile);
    let upper = virtual_percentile(sorted, point, plan, new_len, upper_percentile);
    observed_value >= lower && observed_value <= upper
}

fn value_at_virtual_rank(sorted: &[usize], point: &[f64], plan: VirtualPlan, rank: usize) -> f64 {
    match plan {
        VirtualPlan::Add {
            added_value,
            insert_pos,
        } => value_at_rank_after_add(sorted, point, added_value, insert_pos, rank),
        VirtualPlan::Remove { remove_pos } => {
            value_at_rank_after_remove(sorted, point, remove_pos, rank)
        }
        VirtualPlan::Swap {
            added_value,
            remove_pos,
            insert_pos_after_remove,
        } => value_at_rank_after_swap(
            sorted,
            point,
            added_value,
            remove_pos,
            insert_pos_after_remove,
            rank,
        ),
    }
}

fn insertion_position(sorted: &[usize], point: &[f64], inserted_value: f64) -> usize {
    sorted.partition_point(|&candidate_idx| {
        point[candidate_idx]
            .partial_cmp(&inserted_value)
            .unwrap_or(Ordering::Equal)
            == Ordering::Less
    })
}

fn lookup_rank(positions: &[u32], candidate: usize) -> usize {
    let rank = positions[candidate];
    debug_assert!(
        rank != POSITION_NONE,
        "remove/swap target must be in current selection"
    );
    rank as usize
}

fn value_at_rank_after_add(
    sorted: &[usize],
    point: &[f64],
    added_value: f64,
    insert_pos: usize,
    rank: usize,
) -> f64 {
    match rank.cmp(&insert_pos) {
        Ordering::Less => point[sorted[rank]],
        Ordering::Equal => added_value,
        Ordering::Greater => point[sorted[rank - 1]],
    }
}

fn value_at_rank_after_remove(
    sorted: &[usize],
    point: &[f64],
    remove_pos: usize,
    rank: usize,
) -> f64 {
    let original_idx = if rank < remove_pos { rank } else { rank + 1 };
    point[sorted[original_idx]]
}

fn value_at_rank_after_swap(
    sorted: &[usize],
    point: &[f64],
    added_value: f64,
    remove_pos: usize,
    insert_pos_after_remove: usize,
    rank: usize,
) -> f64 {
    match rank.cmp(&insert_pos_after_remove) {
        Ordering::Less => value_at_rank_after_remove(sorted, point, remove_pos, rank),
        Ordering::Equal => added_value,
        Ordering::Greater => value_at_rank_after_remove(sorted, point, remove_pos, rank - 1),
    }
}
