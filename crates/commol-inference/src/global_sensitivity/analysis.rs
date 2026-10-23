//! A forward analysis: a model with no observations, the inputs it varies
//! and one seed, and every global sensitivity method on them.
//!
//! Each method builds its design from the [`InputSpace`], runs it with the
//! [`BatchRunner`], and hands the quantities to its estimator, all here, so a
//! caller of Rust or Python runs a method in one call. The seed of the
//! analysis splits into one stream per purpose ([`Stream`]): the designs,
//! the replicates of a stochastic engine, the bootstrap of Sobol and the runs
//! of a propagation never draw from one sequence.

use commol_calibration::{CalibrationParameter, ForwardModel};
use commol_core::SimulationEngine;

use crate::batch::BatchRunner;
use crate::design::InputSpace;
use crate::error::MethodError;
use crate::global_sensitivity::{
    ChaosExpansion, Morris, Prcc, Sobol, Truncation, morris, prcc, sobol,
};
use crate::predictive::{draw_seeds, run_sets, share_among_replicates};
use crate::quantity::{Quantities, Quantity};
use crate::sample::ParameterSample;
use crate::streams::Stream;

/// The draws of a propagation and what each run gives.
#[derive(Debug, Clone, PartialEq)]
pub struct Propagation {
    /// The draws of the inputs.
    pub sample: ParameterSample,
    /// The quantities of each run, `[run][quantity]`: one run per draw, or
    /// with a stochastic engine one per replicate of each draw.
    pub values: Vec<Vec<f64>>,
    /// The weight of each run; a replicate has an equal share of its draw's
    /// weight.
    pub weights: Vec<f64>,
    /// The seed of each draw of `sample` ([`crate::predictive::draw_seeds`]):
    /// with a stochastic engine, replicate `i` of draw `j` runs with
    /// `replicate_seed(draw_seeds[j], i)`.
    pub draw_seeds: Vec<u64>,
}

/// A forward model with no observations, the inputs it varies and the seed
/// of every random number.
#[derive(Debug, Clone)]
pub struct ForwardAnalysis<E: SimulationEngine> {
    forward: ForwardModel<E>,
    inputs: InputSpace,
    seed: u64,
}

impl<E: SimulationEngine + Send + Sync> ForwardAnalysis<E> {
    /// The analysis of `inputs`, the values `parameters` of `engine`, with
    /// `remainder` for the initial conditions and, for a stochastic engine,
    /// `replicates` replicates at each point of a design.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when the inputs are not the parameters, in
    /// order and with their bounds, or as [`ForwardModel::unobserved`].
    pub fn new(
        engine: E,
        parameters: Vec<CalibrationParameter>,
        remainder: Option<&str>,
        replicates: Option<usize>,
        inputs: InputSpace,
        seed: u64,
    ) -> Result<Self, MethodError> {
        let names: Vec<String> = parameters.iter().map(|p| p.id.clone()).collect();
        let bounds: Vec<(f64, f64)> = parameters
            .iter()
            .map(|p| (p.min_bound, p.max_bound))
            .collect();
        if inputs.names() != names.as_slice() || inputs.bounds() != bounds {
            return Err(MethodError::Invalid(format!(
                "The inputs are {:?} with the bounds {:?}; the parameters are {names:?} with \
                 the bounds {bounds:?}, in this order",
                inputs.names(),
                inputs.bounds()
            )));
        }
        let forward = ForwardModel::unobserved(
            engine,
            parameters,
            remainder,
            replicates,
            Stream::SensitivityReplicates.seed(seed),
        )
        .map_err(MethodError::Invalid)?;
        Ok(Self {
            forward,
            inputs,
            seed,
        })
    }

    /// The forward model.
    pub fn forward(&self) -> &ForwardModel<E> {
        &self.forward
    }

    /// The seed of the analysis; a predictive of its draws runs from it.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The draws a method of independent inputs or of a sample runs: a
    /// Latin hypercube of `draws` points of independent inputs, or the
    /// sample of the inputs, whose weights it keeps.
    ///
    /// # Errors
    ///
    /// [`MethodError::Invalid`] when independent inputs have no
    /// `draws`, or inputs from a sample have `draws`.
    pub fn draws(
        &self,
        method: &str,
        draws: Option<usize>,
    ) -> Result<ParameterSample, MethodError> {
        let invalid = MethodError::Invalid;
        match (&self.inputs, draws) {
            (InputSpace::Sample(sample), None) => Ok(sample.clone()),
            (InputSpace::Sample(_), Some(_)) => Err(invalid(
                "With inputs from a sample, the sample is the design: leave draws unset"
                    .to_string(),
            )),
            (InputSpace::Independent(_), None) => Err(invalid(format!(
                "{method} of independent inputs needs draws: the number of points of the \
                 Latin hypercube to run"
            ))),
            (InputSpace::Independent(_), Some(count)) => {
                let points = self
                    .inputs
                    .latin_hypercube(method, count, self.design_seed())
                    .map_err(invalid)?;
                ParameterSample::new(
                    self.inputs.names().to_vec(),
                    self.inputs.bounds(),
                    points,
                    &vec![0.0; count],
                    None,
                )
                .map_err(invalid)
            }
        }
    }

    /// Monte Carlo propagation: the quantities of every run of the draws,
    /// each replicate of a stochastic engine with a seed of its own, as a
    /// predictive of the draws runs them ([`run_sets`]).
    ///
    /// # Errors
    ///
    /// As [`Self::draws`], when a quantity does not fit the model, or when a
    /// draw has no run.
    pub fn propagate(
        &self,
        draws: Option<usize>,
        quantities: &[Quantity],
    ) -> Result<Propagation, MethodError> {
        let sample = self.draws("Propagation", draws)?;
        let (values, weights) = if quantities.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let compiled = self.compile(quantities)?;
            let runs = run_sets(
                &self.forward,
                sample.values(),
                compiled.schedule(),
                Some(self.seed),
            )?;
            let evaluated = runs.iter().map(|replicates| {
                replicates
                    .iter()
                    .map(|rows| compiled.evaluate(rows))
                    .collect::<Vec<_>>()
            });
            share_among_replicates(evaluated, sample.weights())
        };
        Ok(Propagation {
            draw_seeds: draw_seeds(self.seed, sample.values().len()),
            sample,
            values,
            weights,
        })
    }

    /// The PRCC of every input for every quantity, on a Latin hypercube of
    /// `draws` points or on the sample of the inputs with its weights.
    ///
    /// # Errors
    ///
    /// As [`Self::draws`], when a quantity does not fit the model, a point
    /// has no run, or as [`prcc()`].
    pub fn prcc(
        &self,
        draws: Option<usize>,
        quantities: &[Quantity],
    ) -> Result<(ParameterSample, Prcc), MethodError> {
        let sample = self.draws("PRCC", draws)?;
        let outputs = self.means(sample.values(), quantities)?;
        let result = prcc(sample.values(), &outputs, sample.weights())?;
        Ok((sample, result))
    }

    /// Morris screening on `trajectories` trajectories of `levels` levels,
    /// and the number of points run.
    ///
    /// # Errors
    ///
    /// As [`InputSpace::morris`], when a quantity does not fit the model, or
    /// a point has no run.
    pub fn morris(
        &self,
        trajectories: usize,
        levels: usize,
        quantities: &[Quantity],
    ) -> Result<(Morris, usize), MethodError> {
        let design = self
            .inputs
            .morris("Morris", trajectories, levels, self.design_seed())
            .map_err(MethodError::Invalid)?;
        let outputs = self.means(&design.points, quantities)?;
        let result = morris(&design, &outputs)?;
        Ok((result, design.points.len()))
    }

    /// Sobol indices on the design of Saltelli with `draws` rows, with
    /// `replicates` bootstrap replicates for intervals at `confidence_level`,
    /// and the number of points run.
    ///
    /// # Errors
    ///
    /// As [`InputSpace::saltelli`] and [`sobol()`], when a quantity does not
    /// fit the model, or a point has no run.
    pub fn sobol(
        &self,
        draws: usize,
        replicates: usize,
        confidence_level: f64,
        quantities: &[Quantity],
    ) -> Result<(Sobol, usize), MethodError> {
        if replicates < 2 {
            return Err(MethodError::Invalid(format!(
                "Sobol needs at least 2 bootstrap replicates, got {replicates}"
            )));
        }
        crate::statistics::interval_levels(confidence_level).map_err(MethodError::Invalid)?;
        let design = self
            .inputs
            .saltelli("Sobol", draws, self.design_seed())
            .map_err(MethodError::Invalid)?;
        let outputs = self.means(&design.points, quantities)?;
        sobol(
            &design,
            &outputs,
            replicates,
            confidence_level,
            Stream::SobolBootstrap.seed(self.seed),
        )
        .map(|result| (result, design.points.len()))
    }

    /// A polynomial chaos expansion of `degree`, truncated by `truncation`,
    /// fitted on a Latin hypercube of `draws` points, and the number of
    /// points run. The number of terms is checked before any run.
    ///
    /// # Errors
    ///
    /// As [`InputSpace::independent`], when `draws` is not above the number
    /// of terms, a quantity does not fit the model, a point has no run, or as
    /// [`ChaosExpansion::fit`].
    pub fn chaos(
        &self,
        degree: usize,
        truncation: Truncation,
        draws: usize,
        quantities: &[Quantity],
    ) -> Result<(ChaosExpansion, usize), MethodError> {
        let method = "PolynomialChaos";
        self.inputs
            .independent(method)
            .map_err(MethodError::Invalid)?;
        let inputs = self.inputs.names().len();
        let terms = ChaosExpansion::terms(inputs, degree, truncation);
        if draws <= terms {
            return Err(MethodError::Invalid(format!(
                "Polynomial chaos of degree {degree} in {inputs} inputs has {terms} terms and \
                 needs more points than terms, got {draws}"
            )));
        }
        let points = self
            .inputs
            .latin_hypercube(method, draws, self.design_seed())
            .map_err(MethodError::Invalid)?;
        let unit = self
            .inputs
            .to_unit(method, &points)
            .map_err(MethodError::Invalid)?;
        let outputs = self.means(&points, quantities)?;
        let expansion = ChaosExpansion::fit(&unit, &outputs, degree, truncation)?;
        Ok((expansion, points.len()))
    }

    /// The quantities, compiled for the engine and checked against its
    /// positions.
    fn compile(&self, quantities: &[Quantity]) -> Result<Quantities, MethodError> {
        let compiled = Quantities::new(quantities, &self.forward.base_engine().output_names())
            .map_err(MethodError::Invalid)?;
        BatchRunner::new(&self.forward, &compiled)?;
        Ok(compiled)
    }

    /// The quantities of every point, the mean of its replicates on common
    /// seeds.
    fn means(
        &self,
        points: &[Vec<f64>],
        quantities: &[Quantity],
    ) -> Result<Vec<Vec<f64>>, MethodError> {
        let compiled = self.compile(quantities)?;
        Ok(BatchRunner::new(&self.forward, &compiled)?.means(points)?)
    }

    fn design_seed(&self) -> u64 {
        Stream::SensitivityDesign.seed(self.seed)
    }
}
