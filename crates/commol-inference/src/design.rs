//! Designs: the inputs a method varies and the points it runs them at.
//!
//! An [`InputSpace`] is independent distributions, one per input, or a
//! weighted [`ParameterSample`], whose inputs may depend on each other. A
//! design of independent inputs is built on the unit cube and mapped to each
//! input by the quantile of its distribution; a sample is its own design.

use commol_random::sample::index;
use commol_random::{Generator, latin_hypercube};

use crate::priors::Priors;
use crate::sample::ParameterSample;

/// What a method varies.
#[derive(Debug, Clone)]
pub enum InputSpace {
    /// One distribution per input, independent of the others.
    Independent(Priors),
    /// The draws of a sample, with their weights; the inputs may depend on
    /// each other.
    Sample(ParameterSample),
}

impl InputSpace {
    /// The names of the inputs, in order.
    pub fn names(&self) -> &[String] {
        match self {
            Self::Independent(priors) => priors.names(),
            Self::Sample(sample) => sample.names(),
        }
    }

    /// The bounds of each input.
    pub fn bounds(&self) -> Vec<(f64, f64)> {
        match self {
            Self::Independent(priors) => priors.bounds(),
            Self::Sample(sample) => sample.bounds().to_vec(),
        }
    }

    /// The distributions of independent inputs.
    ///
    /// # Errors
    ///
    /// Returns a message that names `method` when the inputs come from a
    /// sample: the method needs independent inputs, and PRCC and
    /// propagation take a sample.
    pub fn independent(&self, method: &str) -> Result<&Priors, String> {
        match self {
            Self::Independent(priors) => Ok(priors),
            Self::Sample(_) => Err(format!(
                "{method} needs independent inputs, and inputs from a sample may depend on \
                 each other; PRCC and propagation take a sample"
            )),
        }
    }

    /// A Latin hypercube of `draws` points of independent inputs, seeded
    /// with `seed`: each input puts one point in each of `draws` strata of
    /// equal probability.
    ///
    /// # Errors
    ///
    /// As [`Self::independent`].
    pub fn latin_hypercube(
        &self,
        method: &str,
        draws: usize,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>, String> {
        let priors = self.independent(method)?;
        let unit = latin_hypercube(
            &mut Generator::from_seed(seed),
            draws,
            priors.distributions().len(),
        );
        Ok(to_inputs(priors, unit))
    }

    /// Each point of independent inputs on the scale of probabilities:
    /// `u = F(x)` by the distribution function of each input.
    ///
    /// # Errors
    ///
    /// As [`Self::independent`], or when a point has not one value per
    /// input or a value lies outside the bounds of its input, where `F`
    /// would clamp it to 0 or 1.
    pub fn to_unit(&self, method: &str, points: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let priors = self.independent(method)?;
        let distributions = priors.distributions();
        points
            .iter()
            .map(|point| {
                if point.len() != distributions.len() {
                    return Err(format!(
                        "A point has {} values; the inputs are {}",
                        point.len(),
                        distributions.len()
                    ));
                }
                point
                    .iter()
                    .zip(distributions)
                    .zip(priors.names())
                    .map(|((x, distribution), name)| {
                        let (lower, upper) = distribution.bounds();
                        if x.is_nan() || *x < lower || *x > upper {
                            return Err(format!(
                                "'{name}' is {x}, outside its bounds [{lower:?}, {upper:?}]"
                            ));
                        }
                        Ok(distribution.cdf(*x))
                    })
                    .collect()
            })
            .collect()
    }

    /// The Morris trajectories of independent inputs, seeded with `seed`
    /// ([`MorrisDesign`]).
    ///
    /// # Errors
    ///
    /// As [`Self::independent`], or when `levels` is not even and at least 2
    /// or `trajectories` is 0.
    pub fn morris(
        &self,
        method: &str,
        trajectories: usize,
        levels: usize,
        seed: u64,
    ) -> Result<MorrisDesign, String> {
        let priors = self.independent(method)?;
        if levels < 2 || !levels.is_multiple_of(2) {
            return Err(format!(
                "Morris takes an even number of levels, at least 2, got {levels}"
            ));
        }
        if trajectories == 0 {
            return Err("Morris needs at least one trajectory".to_string());
        }
        let inputs = priors.distributions().len();
        let half = levels / 2;
        let mut generator = Generator::from_seed(seed);
        let mut draw = |count: usize| index(&mut generator, count);
        let mut unit = Vec::with_capacity(trajectories * (inputs + 1));
        let mut steps = Vec::with_capacity(trajectories);
        for _ in 0..trajectories {
            // A level from the lower half and a direction for each input,
            // then the inputs in a random order.
            let mut level: Vec<usize> = (0..inputs).map(|_| draw(half)).collect();
            let up: Vec<bool> = (0..inputs).map(|_| draw(2) == 1).collect();
            for (index, up) in level.iter_mut().zip(&up) {
                if !up {
                    *index += half;
                }
            }
            let mut order: Vec<usize> = (0..inputs).collect();
            for i in (1..inputs).rev() {
                order.swap(i, draw(i + 1));
            }
            let at = |level: &[usize]| -> Vec<f64> {
                level
                    .iter()
                    .map(|&l| (l as f64 + 0.5) / levels as f64)
                    .collect()
            };
            unit.push(at(&level));
            let mut trajectory = Vec::with_capacity(inputs);
            for &input in &order {
                if up[input] {
                    level[input] += half;
                } else {
                    level[input] -= half;
                }
                unit.push(at(&level));
                trajectory.push(MorrisStep {
                    input,
                    change: if up[input] { 0.5 } else { -0.5 },
                });
            }
            steps.push(trajectory);
        }
        Ok(MorrisDesign {
            points: to_inputs(priors, unit),
            steps,
        })
    }
    /// The design of Saltelli (2010) on `base` rows of independent inputs,
    /// seeded with `seed` ([`SaltelliDesign`]).
    ///
    /// # Errors
    ///
    /// As [`Self::independent`], or when `base` is below 2.
    pub fn saltelli(&self, method: &str, base: usize, seed: u64) -> Result<SaltelliDesign, String> {
        let priors = self.independent(method)?;
        if base < 2 {
            return Err(format!("{method} needs at least 2 draws, got {base}"));
        }
        let inputs = priors.distributions().len();
        let mut generator = Generator::from_seed(seed);
        let mut matrix = || -> Vec<Vec<f64>> {
            (0..base)
                .map(|_| (0..inputs).map(|_| generator.uniform()).collect())
                .collect()
        };
        let a = matrix();
        let b = matrix();
        let mut unit = Vec::with_capacity(base * (inputs + 2));
        unit.extend(a.iter().cloned());
        unit.extend(b.iter().cloned());
        for i in 0..inputs {
            unit.extend(a.iter().zip(&b).map(|(row_a, row_b)| {
                let mut row = row_a.clone();
                row[i] = row_b[i];
                row
            }));
        }
        Ok(SaltelliDesign {
            base,
            inputs,
            points: to_inputs(priors, unit),
        })
    }
}

/// The design of Saltelli (2010) for Sobol indices.
///
/// Two matrices `A` and `B` of `base` rows hold independent Monte Carlo
/// draws of the inputs; matrix `A_B⁽ⁱ⁾` is `A` with the column of input `i`
/// taken from `B`. The rows are independent, so a bootstrap of the rows
/// gives valid intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct SaltelliDesign {
    /// The rows of each matrix.
    pub base: usize,
    /// The number of inputs.
    pub inputs: usize,
    /// The points to run: the rows of `A`, of `B`, then of `A_B⁽ⁱ⁾` for each
    /// input `i`, `base · (inputs + 2)` in all.
    pub points: Vec<Vec<f64>>,
}

/// One step of a Morris trajectory: the input it moves and by how much, on
/// the scale of probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorrisStep {
    /// The input that moves.
    pub input: usize,
    /// The change of `F(x)` of the input: `+½` or `−½`.
    pub change: f64,
}

/// The trajectories of a Morris screening (Morris 1991).
///
/// Each input has `p` levels, the centres of `p` strata of equal
/// probability (`F(x) = (j + ½) / p`), so a level is finite for every
/// distribution, unbounded ones too. A trajectory starts from a random level
/// of each input and moves the inputs one at a time, in a random order, by
/// `p / 2` levels, half the range of probabilities, up from the lower half
/// or down from the upper half: with `p` even each level is as likely as any
/// other, as in the design of Morris with `Δ = p / (2 (p − 1))`.
#[derive(Debug, Clone, PartialEq)]
pub struct MorrisDesign {
    /// The points to run, trajectory after trajectory: the start, then one
    /// point per step.
    pub points: Vec<Vec<f64>>,
    /// The steps of each trajectory, `[trajectory][step]`.
    pub steps: Vec<Vec<MorrisStep>>,
}

/// Points on the unit cube mapped to independent inputs by the quantile of
/// each distribution.
fn to_inputs(priors: &Priors, unit: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    unit.into_iter()
        .map(|point| {
            point
                .into_iter()
                .zip(priors.distributions())
                .map(|(p, distribution)| distribution.quantile(p))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Family;

    fn priors() -> Priors {
        Priors::new(
            &["a".to_string(), "b".to_string()],
            &[(0.0, 1.0), (1.0, 100.0)],
            &[
                ("a".to_string(), Family::Uniform),
                ("b".to_string(), Family::LogUniform),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_latin_hypercube_puts_one_point_in_each_stratum_of_each_distribution() {
        let space = InputSpace::Independent(priors());
        let points = space.latin_hypercube("PRCC", 10, 4).unwrap();

        for (column, distribution) in priors().distributions().iter().enumerate() {
            let mut strata: Vec<usize> = points
                .iter()
                .map(|point| (distribution.cdf(point[column]) * 10.0) as usize)
                .collect();
            strata.sort_unstable();
            assert_eq!(strata, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn a_morris_trajectory_moves_one_input_at_a_time_by_half_the_levels() {
        let space = InputSpace::Independent(priors());
        let design = space.morris("Morris", 5, 4, 2).unwrap();

        assert_eq!(design.points.len(), 5 * 3);
        assert_eq!(design.steps.len(), 5);
        let distributions = priors().distributions().to_vec();
        let unit = |point: &[f64]| -> Vec<f64> {
            point
                .iter()
                .zip(&distributions)
                .map(|(x, d)| d.cdf(*x))
                .collect()
        };
        for (t, steps) in design.steps.iter().enumerate() {
            let mut moved: Vec<usize> = steps.iter().map(|step| step.input).collect();
            moved.sort_unstable();
            assert_eq!(moved, vec![0, 1]);
            for (s, step) in steps.iter().enumerate() {
                let before = unit(&design.points[t * 3 + s]);
                let after = unit(&design.points[t * 3 + s + 1]);
                for input in 0..2 {
                    let change = after[input] - before[input];
                    let expected = if input == step.input {
                        step.change
                    } else {
                        0.0
                    };
                    assert!((change - expected).abs() < 1e-9, "{change} ≠ {expected}");
                }
                // Levels are the centres of four strata.
                for u in after {
                    assert!(
                        [0.125, 0.375, 0.625, 0.875]
                            .iter()
                            .any(|l| (u - l).abs() < 1e-9)
                    );
                }
            }
        }
    }

    #[test]
    fn morris_takes_an_even_number_of_levels_and_some_trajectories() {
        let space = InputSpace::Independent(priors());
        assert!(
            space
                .morris("Morris", 2, 3, 1)
                .unwrap_err()
                .contains("even")
        );
        assert!(
            space
                .morris("Morris", 2, 0, 1)
                .unwrap_err()
                .contains("even")
        );
        assert!(
            space
                .morris("Morris", 0, 4, 1)
                .unwrap_err()
                .contains("trajectory")
        );
    }

    #[test]
    fn a_value_outside_its_bounds_has_no_probability() {
        let space = InputSpace::Independent(priors());
        let error = space.to_unit("Chaos", &[vec![0.5, 200.0]]).unwrap_err();
        assert!(error.contains("'b' is 200"), "{error}");
        assert!(
            space
                .to_unit("Chaos", &[vec![0.5]])
                .unwrap_err()
                .contains("2")
        );
        let unit = space.to_unit("Chaos", &[vec![0.25, 10.0]]).unwrap();
        assert!((unit[0][0] - 0.25).abs() < 1e-12 && (unit[0][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_sample_is_not_independent() {
        let sample = ParameterSample::new(
            vec!["a".to_string()],
            vec![(0.0, 1.0)],
            vec![vec![0.5]],
            &[0.0],
            None,
        )
        .unwrap();
        let space = InputSpace::Sample(sample);

        let error = space.latin_hypercube("Sobol", 4, 1).unwrap_err();

        assert!(
            error.starts_with("Sobol needs independent inputs"),
            "{error}"
        );
        assert!(error.contains("PRCC and propagation"), "{error}");
        assert_eq!(space.names(), ["a".to_string()]);
        assert_eq!(space.bounds(), vec![(0.0, 1.0)]);
    }
}
