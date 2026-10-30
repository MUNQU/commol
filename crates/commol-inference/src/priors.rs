//! One [`Distribution`] per parameter, truncated to its bounds: the priors of
//! a sampling method, or the inputs of a sensitivity analysis.

use commol_random::Generator;

use crate::distribution::{Distribution, Family, kinds_that_fit};

/// One distribution per parameter, in problem order.
#[derive(Debug, Clone, PartialEq)]
pub struct Priors {
    names: Vec<String>,
    distributions: Vec<Distribution>,
}

/// What a set of distributions is of, which sets the words of the message
/// of a parameter with no distribution or a distribution of no parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// The priors of a sampling method, by calibrated parameter.
    Priors,
    /// The inputs of a sensitivity analysis.
    Inputs,
}

impl Role {
    /// The message of `name`, which has no distribution.
    fn missing(self, name: &str) -> String {
        match self {
            Self::Priors => format!(
                "'{name}' has no prior; give one per calibrated parameter, or use \
                 Priors.uniform_on_bounds()"
            ),
            Self::Inputs => format!(
                "The input '{name}' has no distribution; give one per input, or use \
                 InputSpace.uniform_on_bounds(parameters)"
            ),
        }
    }

    /// The message of the distribution of `name`, which is none of `names`.
    fn extra(self, name: &str, names: &[String]) -> String {
        match self {
            Self::Priors => {
                format!("'{name}' is not a calibrated parameter; the parameters are {names:?}")
            }
            Self::Inputs => {
                format!("The distribution of '{name}' names no input; the inputs are {names:?}")
            }
        }
    }
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
        Self::of(Role::Priors, names, bounds, families)
    }

    /// The distributions of the inputs `names` of a sensitivity analysis,
    /// with `bounds`, from the family given to each name.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], in the words of the inputs.
    pub fn of_inputs(
        names: &[String],
        bounds: &[(f64, f64)],
        families: &[(String, Family)],
    ) -> Result<Self, String> {
        Self::of(Role::Inputs, names, bounds, families)
    }

    fn of(
        role: Role,
        names: &[String],
        bounds: &[(f64, f64)],
        families: &[(String, Family)],
    ) -> Result<Self, String> {
        if let Some((name, _)) = families.iter().find(|(name, _)| !names.contains(name)) {
            return Err(role.extra(name, names));
        }
        let distributions = names
            .iter()
            .zip(bounds)
            .map(|(name, &(lower, upper))| {
                let family = families
                    .iter()
                    .find(|(given, _)| given == name)
                    .map(|(_, family)| *family)
                    .ok_or_else(|| role.missing(name))?;
                if let Some(problem) = family.bound_problem(lower, upper) {
                    return Err(format!(
                        "The distribution {}() of '{name}' {problem}, and '{name}' has the bounds \
                         [{lower:?}, {upper:?}]; the distributions that fit them: {}",
                        family.kind(),
                        kinds_that_fit(lower, upper).join(", ")
                    ));
                }
                Distribution::new(family, lower, upper)
                    .map_err(|error| format!("The distribution of '{name}': {error}"))
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
    fn the_distributions_of_inputs_name_inputs() {
        let bounds = [(0.0, 1.0), (0.0, 2.0)];
        let missing = Priors::of_inputs(&names(), &bounds, &[("a".to_string(), Family::Uniform)]);
        assert_eq!(
            missing.unwrap_err(),
            "The input 'b' has no distribution; give one per input, or use \
             InputSpace.uniform_on_bounds(parameters)"
        );
        let extra = Priors::of_inputs(
            &names()[..1],
            &bounds[..1],
            &[
                ("a".to_string(), Family::Uniform),
                ("c".to_string(), Family::Uniform),
            ],
        );
        assert_eq!(
            extra.unwrap_err(),
            "The distribution of 'c' names no input; the inputs are [\"a\"]"
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
        assert!(error.starts_with("The distribution of 'a': "), "{error}");
        assert!(error.contains("no mass"), "{error}");
    }

    #[test]
    fn a_family_that_does_not_fit_the_bounds_names_those_that_do() {
        let error =
            Priors::uniform_on_bounds(&names(), &[(0.0, 1.0), (0.0, f64::INFINITY)]).unwrap_err();
        assert_eq!(
            error,
            "The distribution Uniform() of 'b' needs finite bounds, and 'b' has the bounds [0.0, inf]; \
             the distributions that fit them: Normal, LogNormal, Gamma"
        );
    }
}
