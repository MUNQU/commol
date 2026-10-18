//! The weighted parameter sample every method with draws returns.

use commol_random::Generator;

use crate::statistics;

/// Draws of the parameters of a calibration problem, each with a weight.
///
/// Invariants, checked at construction: one name per column, every value
/// finite and inside its bounds, every log weight finite or `−∞` with at
/// least one finite. The weights are normalized to sum to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSample {
    names: Vec<String>,
    bounds: Vec<(f64, f64)>,
    values: Vec<Vec<f64>>,
    weights: Vec<f64>,
    groups: Option<Vec<usize>>,
}

impl ParameterSample {
    /// A sample of `values` (one row per draw, one column per name) with
    /// `log_weights` of any scale, and optionally the group (chain,
    /// replicate or wave) of each draw.
    ///
    /// # Errors
    ///
    /// Returns a message when an invariant does not hold.
    pub fn new(
        names: Vec<String>,
        bounds: Vec<(f64, f64)>,
        values: Vec<Vec<f64>>,
        log_weights: &[f64],
        groups: Option<Vec<usize>>,
    ) -> Result<Self, String> {
        if names.len() != bounds.len() {
            return Err(format!(
                "A sample has one bound per name: {} names, {} bounds",
                names.len(),
                bounds.len()
            ));
        }
        for (index, name) in names.iter().enumerate() {
            if names[..index].contains(name) {
                return Err(format!(
                    "A sample names '{name}' twice; one name per column"
                ));
            }
        }
        if values.is_empty() {
            return Err("A sample needs at least one draw".to_string());
        }
        if log_weights.len() != values.len() {
            return Err(format!(
                "A sample has one log weight per draw: {} draws, {} log weights",
                values.len(),
                log_weights.len()
            ));
        }
        if let Some(groups) = &groups
            && groups.len() != values.len()
        {
            return Err(format!(
                "A sample has one group per draw: {} draws, {} groups",
                values.len(),
                groups.len()
            ));
        }
        for (index, draw) in values.iter().enumerate() {
            if draw.len() != names.len() {
                return Err(format!(
                    "Draw {index} has {} values for {} names",
                    draw.len(),
                    names.len()
                ));
            }
            for ((name, (min, max)), &value) in names.iter().zip(&bounds).zip(draw) {
                if !value.is_finite() || value < *min || value > *max {
                    return Err(format!(
                        "Draw {index} gives '{name}' the value {value}, outside its bounds \
                         [{min}, {max}]"
                    ));
                }
            }
        }
        if log_weights
            .iter()
            .any(|w| w.is_nan() || *w == f64::INFINITY)
        {
            return Err("A log weight is finite or −∞".to_string());
        }
        if !log_weights.iter().any(|w| w.is_finite()) {
            return Err("A sample needs one draw with a finite log weight".to_string());
        }
        Ok(Self {
            names,
            bounds,
            values,
            weights: statistics::normalized_weights(log_weights),
            groups,
        })
    }

    /// The parameter names, one per column.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The bounds of each parameter.
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// The draws, one row each.
    pub fn values(&self) -> &[Vec<f64>] {
        &self.values
    }

    /// The weights, which sum to 1.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The natural logarithms of the weights.
    pub fn log_weights(&self) -> Vec<f64> {
        self.weights.iter().map(|w| w.ln()).collect()
    }

    /// The group of each draw, if the method records one.
    pub fn groups(&self) -> Option<&[usize]> {
        self.groups.as_deref()
    }

    /// Number of draws.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the sample has no draws; never true for a built sample.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn column(&self, index: usize) -> Vec<f64> {
        self.values.iter().map(|draw| draw[index]).collect()
    }

    /// The weighted mean of each parameter.
    pub fn mean(&self) -> Vec<f64> {
        (0..self.names.len())
            .map(|i| statistics::mean(&self.column(i), &self.weights))
            .collect()
    }

    /// The weighted covariance matrix ([`statistics::covariance`]).
    pub fn covariance(&self) -> Vec<Vec<f64>> {
        let columns: Vec<Vec<f64>> = (0..self.names.len()).map(|i| self.column(i)).collect();
        columns
            .iter()
            .map(|x| {
                columns
                    .iter()
                    .map(|y| statistics::covariance(x, y, &self.weights))
                    .collect()
            })
            .collect()
    }

    /// The weighted correlation matrix. An entry of a parameter with no
    /// spread is NaN.
    pub fn correlation(&self) -> Vec<Vec<f64>> {
        let covariance = self.covariance();
        let sd: Vec<f64> = (0..covariance.len())
            .map(|i| covariance[i][i].sqrt())
            .collect();
        covariance
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, c)| c / (sd[i] * sd[j]))
                    .collect()
            })
            .collect()
    }

    /// The weighted quantiles of each parameter: one row per level, one
    /// column per parameter.
    ///
    /// # Errors
    ///
    /// Returns a message when a level is not in `[0, 1]`.
    pub fn quantiles(&self, levels: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        statistics::check_levels(levels)?;
        let per_parameter: Vec<Vec<f64>> = (0..self.names.len())
            .map(|i| statistics::quantiles(&self.column(i), &self.weights, levels))
            .collect();
        Ok((0..levels.len())
            .map(|l| per_parameter.iter().map(|q| q[l]).collect())
            .collect())
    }

    /// The central interval of `confidence_level` of each parameter, as
    /// `(lower, upper)` rows.
    ///
    /// # Errors
    ///
    /// Returns a message when `confidence_level` is not strictly between 0
    /// and 1.
    pub fn interval(&self, confidence_level: f64) -> Result<[Vec<f64>; 2], String> {
        let levels = statistics::interval_levels(confidence_level)?;
        let mut rows = self.quantiles(&levels)?.into_iter();
        Ok([
            rows.next().expect("two levels"),
            rows.next().expect("two levels"),
        ])
    }

    /// Kish's effective sample size of the weights.
    pub fn effective_size(&self) -> f64 {
        statistics::effective_size(&self.weights)
    }

    /// The indices of `count` draws taken with replacement, each with
    /// probability its weight, from the generator seeded with `seed`.
    pub fn resample(&self, count: usize, seed: u64) -> Vec<usize> {
        let mut cumulative = Vec::with_capacity(self.weights.len());
        let mut total = 0.0;
        for w in &self.weights {
            total += w;
            cumulative.push(total);
        }
        let mut generator = Generator::from_seed(seed);
        (0..count)
            .map(|_| {
                let point = generator.uniform() * total;
                cumulative
                    .partition_point(|&c| c <= point)
                    .min(self.weights.len() - 1)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(log_weights: &[f64]) -> ParameterSample {
        ParameterSample::new(
            vec!["k1".to_string(), "k2".to_string()],
            vec![(0.0, 10.0), (0.0, 10.0)],
            vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![4.0, 0.0]],
            log_weights,
            None,
        )
        .unwrap()
    }

    #[test]
    fn equal_weights_give_the_unweighted_statistics() {
        let sample = sample(&[0.0; 3]);
        let mean = sample.mean();
        assert!((mean[0] - 7.0 / 3.0).abs() < 1e-12 && (mean[1] - 1.0).abs() < 1e-12);
        let covariance = sample.covariance();
        assert!((covariance[0][1] + 1.5).abs() < 1e-12);
        assert!((covariance[1][1] - 1.0).abs() < 1e-12);
        assert!((sample.effective_size() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_weighted_mean_is_the_closed_form() {
        let sample = sample(&[0.5f64.ln(), 0.25f64.ln(), 0.25f64.ln()]);
        // 0.5·1 + 0.25·2 + 0.25·4 = 2.
        assert!((sample.mean()[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_draw_outside_its_bounds_is_rejected() {
        let error = ParameterSample::new(
            vec!["k".to_string()],
            vec![(0.0, 1.0)],
            vec![vec![2.0]],
            &[0.0],
            None,
        )
        .unwrap_err();
        assert!(error.contains("outside its bounds"), "{error}");
    }

    #[test]
    fn quantiles_have_one_row_per_level_and_one_column_per_parameter() {
        let sample = sample(&[0.0; 3]);
        assert_eq!(
            sample.quantiles(&[0.0, 0.5, 1.0]),
            Ok(vec![vec![1.0, 0.0], vec![2.0, 1.0], vec![4.0, 2.0]])
        );
        assert!(sample.quantiles(&[1.5]).is_err());
    }

    #[test]
    fn resampling_never_picks_a_draw_of_zero_weight() {
        let sample = sample(&[0.0, f64::NEG_INFINITY, 0.0]);
        assert!(!sample.is_empty());
        let picks = sample.resample(1000, 3);
        assert!(picks.iter().all(|&i| i != 1));
        // Draws 0 and 2 share the weight: each is picked about 500 times.
        let first = picks.iter().filter(|&&i| i == 0).count();
        assert!((400..600).contains(&first), "{first}");
        assert_eq!(picks, sample.resample(1000, 3));
    }
}
