//! The priors of a sampling method: one [`Distribution`] per parameter,
//! truncated to its bounds.

use commol_random::Generator;

use crate::distribution::{Distribution, Family, kinds_that_fit};

/// One prior per parameter, in problem order.
#[derive(Debug, Clone, PartialEq)]
pub struct Priors {
    names: Vec<String>,
    distributions: Vec<Distribution>,
}

impl Priors {
    /// The priors of the parameters `names` with `bounds`, from the family
    /// given to each name.
    ///
    /// # Errors
    ///
    /// Returns a message that names the parameter when one has no prior, a
    /// prior names no parameter, or a family does not fit the bounds of its
    /// parameter (the message lists the kinds that fit them).
    pub fn new(
        names: &[String],
        bounds: &[(f64, f64)],
        families: &[(String, Family)],
    ) -> Result<Self, String> {
        for (name, _) in families {
            if !names.contains(name) {
                return Err(format!(
                    "'{name}' is not a calibrated parameter; the parameters are {names:?}"
                ));
            }
        }
        let distributions = names
            .iter()
            .zip(bounds)
            .map(|(name, &(lower, upper))| {
                let family = families
                    .iter()
                    .find(|(given, _)| given == name)
                    .map(|(_, family)| *family)
                    .ok_or_else(|| {
                        format!(
                            "'{name}' has no prior; give one per calibrated parameter, or use \
                             Priors.uniform_on_bounds()"
                        )
                    })?;
                if let Some(problem) = family.bound_problem(lower, upper) {
                    return Err(format!(
                        "The prior {}() of '{name}' {problem}, and '{name}' has the bounds \
                         [{lower:?}, {upper:?}]; the distributions that fit them: {}",
                        family.kind(),
                        kinds_that_fit(lower, upper).join(", ")
                    ));
                }
                Distribution::new(family, lower, upper)
                    .map_err(|error| format!("The prior of '{name}': {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            names: names.to_vec(),
            distributions,
        })
    }

    /// `Uniform` on the bounds for every parameter.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], when a bound is infinite.
    pub fn uniform_on_bounds(names: &[String], bounds: &[(f64, f64)]) -> Result<Self, String> {
        let families: Vec<(String, Family)> = names
            .iter()
            .map(|name| (name.clone(), Family::Uniform))
            .collect();
        Self::new(names, bounds, &families)
    }

    /// The parameter names, in problem order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The prior of each parameter, in problem order.
    pub fn distributions(&self) -> &[Distribution] {
        &self.distributions
    }

    /// The bounds of each parameter.
    pub fn bounds(&self) -> Vec<(f64, f64)> {
        self.distributions
            .iter()
            .map(Distribution::bounds)
            .collect()
    }

    /// The joint log density at `values`: `−∞` outside the bounds.
    pub fn log_density(&self, values: &[f64]) -> f64 {
        self.distributions
            .iter()
            .zip(values)
            .map(|(distribution, &value)| distribution.log_density(value))
            .sum()
    }

    /// One draw of every parameter, independently.
    pub fn sample(&self, generator: &mut Generator) -> Vec<f64> {
        self.distributions
            .iter()
            .map(|distribution| distribution.sample(generator))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["a".to_string(), "b".to_string()]
    }

    #[test]
    fn every_parameter_needs_a_prior_and_every_prior_a_parameter() {
        let bounds = [(0.0, 1.0), (0.0, 2.0)];
        let missing = Priors::new(&names(), &bounds, &[("a".to_string(), Family::Uniform)]);
        assert_eq!(
            missing.unwrap_err(),
            "'b' has no prior; give one per calibrated parameter, or use \
             Priors.uniform_on_bounds()"
        );
        let extra = Priors::new(
            &names(),
            &bounds,
            &[
                ("a".to_string(), Family::Uniform),
                ("c".to_string(), Family::Uniform),
            ],
        );
        assert!(
            extra
                .unwrap_err()
                .starts_with("'c' is not a calibrated parameter")
        );
    }

    #[test]
    fn the_joint_density_is_the_sum_of_the_marginals() {
        let priors = Priors::uniform_on_bounds(&names(), &[(0.0, 1.0), (0.0, 2.0)]).unwrap();
        assert!((priors.log_density(&[0.5, 1.0]) + 2f64.ln()).abs() < 1e-15);
        assert_eq!(priors.log_density(&[0.5, 3.0]), f64::NEG_INFINITY);
        let draw = priors.sample(&mut Generator::from_seed(1));
        assert!(draw[0] < 1.0 && draw[1] < 2.0);
    }

    #[test]
    fn a_prior_with_no_mass_on_its_bounds_names_its_parameter() {
        let error = Priors::new(
            &names()[..1],
            &[(0.0, 1.0)],
            &[(
                "a".to_string(),
                Family::Normal {
                    mean: 100.0,
                    sd: 1.0,
                },
            )],
        )
        .unwrap_err();
        assert!(error.starts_with("The prior of 'a': "), "{error}");
        assert!(error.contains("no mass"), "{error}");
    }

    #[test]
    fn a_family_that_does_not_fit_the_bounds_names_those_that_do() {
        let error =
            Priors::uniform_on_bounds(&names(), &[(0.0, 1.0), (0.0, f64::INFINITY)]).unwrap_err();
        assert_eq!(
            error,
            "The prior Uniform() of 'b' needs finite bounds, and 'b' has the bounds [0.0, inf]; \
             the distributions that fit them: Normal, LogNormal, Gamma"
        );
    }
}
