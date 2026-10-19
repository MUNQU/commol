//! What both stochastic engines share: the generator, whole units and their
//! start, and impulses that move whole units.

use commol_core::{EngineType, Rounding, SimulationError};
use commol_model::{CompiledModel, as_whole};
use commol_random::{Generator, sample};

/// `value` to six significant digits, for a message.
pub(crate) fn rounded(value: f64) -> f64 {
    format!("{value:.5e}").parse().unwrap_or(value)
}

/// The compiled model of a stochastic engine, its generator and its state of
/// whole units.
///
/// After a reset the state holds the initial amounts of the model, as they
/// are written ([`Self::set_initial_condition`]); the first run makes them
/// whole ([`Self::begin`]), with the rounding of the settings, so a rounding
/// that draws reads the generator after the values calibration writes. The
/// rounding makes the compartments of each bin whole and keeps the amount of
/// the bin, so no unit moves to a bin of another unit.
#[derive(Clone)]
pub(crate) struct Units {
    pub(crate) model: CompiledModel,
    engine: EngineType,
    rounding: Option<Rounding>,
    /// Indices of the compartments of each bin, in engine order.
    bins: Vec<Vec<usize>>,
    seed: u64,
    pub(crate) generator: Generator,
    /// Current state: compartments, then accumulators.
    pub(crate) state: Vec<f64>,
    /// Whether the compartments of the state hold whole units.
    started: bool,
    /// Rates of the impulses that act at a position, reused.
    impulse_rates: Vec<(usize, f64)>,
}

impl Units {
    /// The units of a compiled model, with the seed 0.
    pub(crate) fn new(
        model: CompiledModel,
        engine: EngineType,
        rounding: Option<Rounding>,
    ) -> Self {
        let state = model.initial_state().to_vec();
        let mut bins: Vec<(String, Vec<usize>)> = Vec::new();
        for (index, label) in model.output_labels()[..model.num_compartments()]
            .iter()
            .enumerate()
        {
            match bins.iter_mut().find(|(bin, _)| *bin == label.source) {
                Some((_, indices)) => indices.push(index),
                None => bins.push((label.source.clone(), vec![index])),
            }
        }
        Self {
            model,
            engine,
            rounding,
            bins: bins.into_iter().map(|(_, indices)| indices).collect(),
            seed: 0,
            generator: Generator::from_seed(0),
            state,
            started: false,
            impulse_rates: Vec::new(),
        }
    }

    /// Seed the generator, and restore it.
    pub(crate) fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.generator = Generator::from_seed(seed);
    }

    /// Restore the generator of the seed and the initial amounts of the
    /// model.
    pub(crate) fn reset(&mut self) {
        self.generator = Generator::from_seed(self.seed);
        self.state.copy_from_slice(self.model.initial_state());
        self.started = false;
    }

    /// Make the initial amounts whole, once per run.
    pub(crate) fn begin(&mut self) {
        if self.started {
            return;
        }
        let num_compartments = self.model.num_compartments();
        let amounts = &self.model.initial_state()[..num_compartments];
        let whole = match self.rounding {
            // The model check made sure every amount is whole.
            None => amounts
                .iter()
                .map(|&amount| as_whole(amount).unwrap_or_else(|| amount.round()))
                .collect(),
            Some(rounding) => round_bins(amounts, &self.bins, rounding, &mut self.generator),
        };
        self.state[..num_compartments].copy_from_slice(&whole);
        self.started = true;
    }

    /// Change the initial amount of one compartment.
    ///
    /// # Errors
    ///
    /// Returns a message when the index names no compartment, the value is
    /// negative, or the value is not whole and the settings name no rounding.
    pub(crate) fn set_initial_condition(&mut self, index: usize, value: f64) -> Result<(), String> {
        if self.rounding.is_none() && value >= 0.0 && as_whole(value).is_none() {
            return Err(format!(
                "The {} engine moves whole units, and {value} is not whole. Name a \
                 rounding in the settings of the engine: \"{}\" or \"{}\"",
                self.engine.name(),
                Rounding::LargestRemainder.name(),
                Rounding::Random.name(),
            ));
        }
        self.model.set_initial_condition(index, value)?;
        if !self.started {
            self.state[index] = value;
        }
        Ok(())
    }

    /// Apply the impulses that act at position `t`.
    ///
    /// Every impulse reads the state before any of them, as in the
    /// deterministic engines. A proportional impulse moves a binomial draw of
    /// the units of its source, its rate being the probability that one unit
    /// moves; an absolute impulse moves its rate, a whole number of units.
    /// Each moves at most what its sources still hold, in the order of the
    /// model.
    ///
    /// Returns whether any impulse acted.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Infeasible`] that names the impulse when a
    /// proportional rate is not a probability or an absolute rate is not a
    /// whole number of units, and [`SimulationError::Failed`] when a rate
    /// cannot be evaluated.
    pub(crate) fn apply_impulses(&mut self, t: f64) -> Result<bool, SimulationError> {
        let mut rates = std::mem::take(&mut self.impulse_rates);
        let result = self.model.impulse_rates(t, &self.state, &mut rates);
        let applied = result.map_err(SimulationError::from).and_then(|()| {
            let before = self.state.clone();
            for &(index, rate) in &rates {
                let change = &self.model.impulse_changes()[index];
                let name = change.transition_id();
                let mut count = match change.proportional_to() {
                    Some(source) => {
                        if !(0.0..=1.0).contains(&rate) {
                            return Err(SimulationError::Infeasible {
                                position: t,
                                message: format!(
                                    "Impulse '{name}' at {t} moves a fraction {} of \
                                     compartment '{}'. The {} engine moves each unit with \
                                     that probability, so it must be between 0 and 1",
                                    rounded(rate),
                                    self.model.compartments()[source],
                                    self.engine.name(),
                                ),
                            });
                        }
                        sample::binomial(&mut self.generator, before[source] as u64, rate) as f64
                    }
                    None => match as_whole(rate) {
                        Some(units) if units >= 0.0 => units,
                        _ => {
                            return Err(SimulationError::Infeasible {
                                position: t,
                                message: format!(
                                    "Impulse '{name}' at {t} moves {} units. The {} engine \
                                     moves whole units, so an absolute impulse must be a \
                                     whole number that is not negative",
                                    rounded(rate),
                                    self.engine.name(),
                                ),
                            });
                        }
                    },
                };
                for &(compartment, units) in change.takes() {
                    count = count.min((self.state[compartment] / f64::from(units)).floor());
                }
                for &(state_index, sign) in change.changes() {
                    self.state[state_index] += sign * count;
                }
            }
            Ok(!rates.is_empty())
        });
        self.impulse_rates = rates;
        applied
    }

    /// Copy the state into an output row.
    pub(crate) fn copy_row(&self, row: &mut Vec<f64>) {
        row.clear();
        row.extend_from_slice(&self.state);
    }
}

/// Whole amounts in which the compartments of each bin sum to the rounded
/// amount of the bin, with `rounding` applied to each bin in turn.
pub(crate) fn round_bins(
    amounts: &[f64],
    bins: &[Vec<usize>],
    rounding: Rounding,
    generator: &mut Generator,
) -> Vec<f64> {
    let mut whole = amounts.to_vec();
    for bin in bins {
        let of_bin: Vec<f64> = bin.iter().map(|&index| amounts[index]).collect();
        let rounded = match rounding {
            Rounding::LargestRemainder => largest_remainder(&of_bin),
            Rounding::Random => random_rounding(&of_bin, generator),
        };
        for (&index, value) in bin.iter().zip(rounded) {
            whole[index] = value;
        }
    }
    whole
}

/// Whole amounts that sum to the rounded sum of `amounts`: each amount
/// rounded down, and one more unit to each of the largest remainders, the
/// first compartment first on a tie.
pub(crate) fn largest_remainder(amounts: &[f64]) -> Vec<f64> {
    let snapped: Vec<f64> = amounts
        .iter()
        .map(|&amount| as_whole(amount).unwrap_or(amount))
        .collect();
    let mut whole: Vec<f64> = snapped.iter().map(|amount| amount.floor()).collect();
    let target = snapped.iter().sum::<f64>().round();
    let left = (target - whole.iter().sum::<f64>()).max(0.0) as usize;
    let mut order: Vec<usize> = (0..snapped.len()).collect();
    order.sort_by(|&a, &b| {
        let remainder = |i: usize| snapped[i] - whole[i];
        remainder(b).total_cmp(&remainder(a)).then(a.cmp(&b))
    });
    for &index in order.iter().take(left) {
        whole[index] += 1.0;
    }
    whole
}

/// Whole amounts drawn from a multinomial of the rounded sum of `amounts`,
/// each compartment with its share of the sum, so the mean of a draw is
/// `amounts`.
pub(crate) fn random_rounding(amounts: &[f64], generator: &mut Generator) -> Vec<f64> {
    let sum: f64 = amounts.iter().sum();
    let trials = sum.round() as u64;
    if sum <= 0.0 || trials == 0 {
        return vec![0.0; amounts.len()];
    }
    let probabilities: Vec<f64> = amounts.iter().map(|amount| amount / sum).collect();
    let mut counts = vec![0u64; amounts.len()];
    sample::multinomial(generator, trials, &probabilities, &mut counts);
    // The units the rounding of the shares leaves go to the last compartment
    // that holds any.
    let drawn: u64 = counts.iter().sum();
    if let Some(last) = amounts.iter().rposition(|&amount| amount > 0.0) {
        counts[last] += trials - drawn;
    }
    counts.into_iter().map(|count| count as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_largest_remainders_take_the_units_left() {
        assert_eq!(
            largest_remainder(&[0.5, 0.5, 0.5, 0.5, 2.0]),
            vec![1.0, 1.0, 0.0, 0.0, 2.0]
        );
        assert_eq!(largest_remainder(&[3.2, 3.7, 3.1]), vec![3.0, 4.0, 3.0]);
        assert_eq!(
            largest_remainder(&[699.999_999_999_9, 300.0]),
            vec![700.0, 300.0]
        );
    }

    #[test]
    fn each_bin_keeps_its_amount_when_rounded() {
        // Over all four compartments, the largest remainders would give both
        // units left to the first bin.
        let amounts = [0.5, 0.5, 0.5, 0.5];
        let bins = [vec![0, 1], vec![2, 3]];
        let mut generator = Generator::from_seed(5);

        assert_eq!(
            round_bins(&amounts, &bins, Rounding::LargestRemainder, &mut generator),
            [1.0, 0.0, 1.0, 0.0]
        );
        for _ in 0..100 {
            let whole = round_bins(&amounts, &bins, Rounding::Random, &mut generator);
            assert_eq!(whole[0] + whole[1], 1.0, "{whole:?}");
            assert_eq!(whole[2] + whole[3], 1.0, "{whole:?}");
        }
    }

    #[test]
    fn a_random_rounding_of_less_than_half_a_unit_is_zero() {
        let mut generator = Generator::from_seed(1);

        assert_eq!(random_rounding(&[0.2, 0.1], &mut generator), [0.0, 0.0]);
        assert_eq!(random_rounding(&[0.0, 0.0], &mut generator), [0.0, 0.0]);
    }

    #[test]
    fn a_random_rounding_keeps_the_total_and_the_mean() {
        let amounts = [0.5, 1.5, 3.0, 0.0];
        let mut generator = Generator::from_seed(3);
        let draws = 20_000;
        let mut sums = [0.0; 4];
        for _ in 0..draws {
            let whole = random_rounding(&amounts, &mut generator);
            assert_eq!(whole.iter().sum::<f64>(), 5.0);
            assert_eq!(whole[3], 0.0);
            for (sum, value) in sums.iter_mut().zip(whole) {
                *sum += value;
            }
        }
        for (sum, amount) in sums.iter().zip(amounts) {
            assert!(
                (sum / draws as f64 - amount).abs() < 0.03,
                "{sum} for {amount}"
            );
        }
    }
}
