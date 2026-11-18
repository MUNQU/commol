//! The dependence of the residuals of a predictive: the quantile residuals
//! of the observation models, integrated over randomizations common to every
//! draw, or the additive residuals of a loss, against a reference that holds
//! at a fit.

use commol_calibration::Term;
use commol_calibration::{
    CalibrationProblem, Likelihood, ObservedDataPoint, OptimizationConfig, ReplicateSummary,
};
use commol_core::{CONFIDENCE_LEVEL, SimulationEngine};
use commol_random::{Generator, Stream, replicate_seed};
use rayon::prelude::*;

use super::grid::Layout;
use super::law::{Portmanteau, Tests, result, spread, test};
use super::of_values::{lags_of, wild_band};
use super::sets::{SetStatistics, Summary, statistics, summarize};
use super::{MaxLag, SeriesDependence};
use crate::error::MethodError;
use crate::predictive::{
    DistributionNoise, ObservationNoise, ObservedMeans, check_names, observe_sets,
};
use crate::ranges::RANDOMIZATIONS;
use crate::resampling::{
    FitUnit, Multipliers, ParametricResampler, Perturbation, ReplicateRunner, Resampler,
    RunnerSteps,
};
use crate::sample::ParameterSample;
use crate::scores::quantile_residual;
use crate::settings::dependence::Reference;

/// The residuals an analysis reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualKind {
    /// The randomized quantile residuals of the observation models
    /// (`DistributionNoise()`).
    Quantile,
    /// The additive residuals `y − ŷ`, `ŷ` the prediction the loss scores
    /// (`NoNoise()`).
    Additive,
}

/// The settings of an analysis of residuals.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualSettings {
    /// The largest lag of each series.
    pub max_lag: MaxLag,
    /// The randomizations of the quantile residuals; 1 for additive ones.
    pub randomizations: usize,
    /// The reference of the tests and the bands.
    pub reference: Reference,
    /// The level of the band of `τ`; the share of the p-values of the draws
    /// below `1 − confidence_level`.
    pub confidence_level: f64,
}

/// The data of a refit of additive residuals: the fit plus its residuals,
/// each multiplied by a sign ±1.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SignFlips {
    /// The fitted value of each observation.
    pub fitted: Vec<f64>,
    /// The residual of each observation.
    pub residuals: Vec<f64>,
}

impl Resampler for SignFlips {
    fn replicate(&self, generator: &mut Generator) -> Perturbation {
        Perturbation::Values(
            self.fitted
                .iter()
                .zip(&self.residuals)
                .map(|(fitted, residual)| {
                    fitted + Multipliers::Rademacher.draw(generator) * residual
                })
                .collect(),
        )
    }
}

/// How the residuals of a draw are read.
enum Reader {
    /// Quantile residuals, with the places of each randomization.
    Quantile {
        likelihood: Likelihood,
        terms: Vec<Term>,
        places: Vec<Vec<f64>>,
    },
    /// Additive residuals of the prediction of `summary`.
    Additive { summary: Option<ReplicateSummary> },
}

impl Reader {
    /// The residual sets of `observations` under the draw of parameter
    /// `values` whose runs gave `means`: one per randomization.
    fn sets(
        &self,
        observations: &[ObservedDataPoint],
        values: &[f64],
        means: &[ObservedMeans],
    ) -> Vec<Vec<f64>> {
        match self {
            Self::Quantile {
                likelihood,
                terms,
                places,
            } => {
                let share = 1.0 / means.len() as f64;
                let mut tails = vec![commol_calibration::Tails::default(); observations.len()];
                for run in means {
                    for (sum, run_tails) in tails.iter_mut().zip(likelihood.tails_of(
                        observations,
                        terms,
                        values,
                        &run.parts,
                    )) {
                        sum.add_weighted(share, run_tails);
                    }
                }
                places
                    .iter()
                    .map(|places| {
                        tails
                            .iter()
                            .zip(places)
                            .map(|(tails, &v)| quantile_residual(tails, v))
                            .collect()
                    })
                    .collect()
            }
            Self::Additive { summary } => {
                let residuals = observations
                    .iter()
                    .enumerate()
                    .map(|(i, obs)| {
                        let predicted: Vec<f64> =
                            means.iter().map(|run| run.predicted[i]).collect();
                        let fitted = match summary {
                            Some(
                                summary
                                @ (ReplicateSummary::Median | ReplicateSummary::Quantile(_)),
                            ) => summary.summarize(&predicted),
                            _ => predicted.iter().sum::<f64>() / predicted.len() as f64,
                        };
                        obs.value - fitted
                    })
                    .collect();
                vec![residuals]
            }
        }
    }
}

/// `observations` with the observed values `values`.
fn with_values(observations: &[ObservedDataPoint], values: &[f64]) -> Vec<ObservedDataPoint> {
    observations
        .iter()
        .zip(values)
        .map(|(obs, &value)| ObservedDataPoint {
            value,
            ..obs.clone()
        })
        .collect()
}

/// The statistics of each set of `sets`.
fn statistics_of(layout: &Layout, lags: &[usize], sets: &[Vec<f64>]) -> Vec<SetStatistics> {
    sets.iter()
        .map(|set| statistics(layout, lags, set))
        .collect()
}

/// The summary of the statistics of the randomizations of one draw.
fn summary_of(sets: &[SetStatistics]) -> Summary {
    let share = 1.0 / sets.len() as f64;
    let weighted: Vec<(f64, &SetStatistics)> = sets.iter().map(|set| (share, set)).collect();
    summarize(&weighted)
}

/// One draw of the sample: its parameter values, its runs, its residual
/// sets and their statistics.
struct Draw<'a> {
    values: &'a [f64],
    means: Vec<ObservedMeans>,
    residuals: Vec<Vec<f64>>,
    statistics: Vec<SetStatistics>,
}

/// The summaries of the refits of a reference.
struct Refits {
    /// The summary of the residuals of each refit that succeeded.
    summaries: Vec<Summary>,
    /// The index and reason of each refit that failed.
    failed: Vec<(usize, String)>,
    /// The refits kept although their optimizer did not converge.
    not_converged: usize,
}

/// The parts of an analysis that every draw reads.
struct Analysis<'a, E: SimulationEngine> {
    problem: &'a CalibrationProblem<E>,
    observations: &'a [ObservedDataPoint],
    layout: Layout,
    lags: Vec<usize>,
    reader: Reader,
    noise: Option<DistributionNoise>,
    seed: u64,
}

impl<E: SimulationEngine + Send + Sync> Analysis<'_, E> {
    /// The reference of `draw`, number `index`, at its parameters: data sets
    /// drawn from the observation models with no refit, or wild sets of
    /// its additive residuals.
    fn at_parameters(
        &self,
        index: usize,
        draw: &Draw<'_>,
        replicates: usize,
    ) -> (Vec<Summary>, Portmanteau) {
        let Some(noise) = &self.noise else {
            let seed = Stream::DependenceSigns.replicate(self.seed, index as u64);
            let band = wild_band(
                &self.layout,
                &self.lags,
                &draw.residuals[0],
                replicates,
                seed,
            );
            return (band, Portmanteau::ChiSquare);
        };
        let seed = Stream::DependenceReference.replicate(self.seed, index as u64);
        let summaries = (0..replicates)
            .into_par_iter()
            .map(|b| {
                let mut generator = Generator::from_seed(replicate_seed(seed, b as u64));
                let run = &draw.means[b % draw.means.len()];
                let mut values = vec![0.0; self.observations.len()];
                noise.observe(
                    self.observations,
                    draw.values,
                    &run.predicted,
                    &run.parts,
                    &mut generator,
                    &mut values,
                );
                let data = with_values(self.observations, &values);
                let sets = self.reader.sets(&data, draw.values, &draw.means);
                summary_of(&statistics_of(&self.layout, &self.lags, &sets))
            })
            .collect();
        (summaries, Portmanteau::Replicates)
    }

    /// The reference of refits of the one draw `draw`: data sets drawn at it,
    /// each fitted again from `starts` starts, and the statistics of their
    /// residuals; with the index and reason of each refit that failed and
    /// the refits that did not converge.
    fn refitted(
        &self,
        draw: &Draw<'_>,
        replicates: usize,
        starts: usize,
        config: &OptimizationConfig,
        steps: &RunnerSteps<'_>,
    ) -> Result<Refits, MethodError> {
        let runner = ReplicateRunner::new(self.problem, config, draw.values, starts, self.seed)?
            .with_steps(steps);
        let resampler: Box<dyn Resampler> = if self.noise.is_some() {
            Box::new(ParametricResampler::new(self.problem, draw.values)?)
        } else {
            let residuals = draw.residuals[0].clone();
            let fitted = self
                .observations
                .iter()
                .zip(&residuals)
                .map(|(obs, residual)| obs.value - residual)
                .collect();
            Box::new(SignFlips { fitted, residuals })
        };
        let perturbations = runner.perturbations(resampler.as_ref(), replicates);
        let fits = runner.fit_all(&perturbations);
        fits.require(FitUnit::Replicate, 1)?;
        let mut failed = fits.failed.clone();
        let outcomes: Vec<(usize, Result<Summary, MethodError>)> = fits
            .fits
            .par_iter()
            .map(|(index, fit)| {
                let outcome =
                    runner
                        .replicate_problem(&perturbations[*index])
                        .and_then(|problem| {
                            let values = problem.applied_parameter_values(&fit.best_parameters);
                            observe_sets(
                                problem.forward(),
                                std::slice::from_ref(&values),
                                problem.observed_data(),
                                None,
                            )
                            .map_err(MethodError::from)
                            .map(|mut means| {
                                let means = means.swap_remove(0);
                                let sets =
                                    self.reader.sets(problem.observed_data(), &values, &means);
                                summary_of(&statistics_of(&self.layout, &self.lags, &sets))
                            })
                        });
                (*index, outcome)
            })
            .collect();
        let mut summaries = Vec::new();
        for (index, outcome) in outcomes {
            match outcome {
                Ok(summary) => summaries.push(summary),
                // coverage: invariant, the refit ran this problem at these values
                Err(error) => failed.push((index, error.to_string())),
            }
        }
        failed.sort_by_key(|(index, _)| *index);
        FitUnit::Replicate.require(summaries.len(), &failed, 1)?;
        Ok(Refits {
            summaries,
            failed,
            not_converged: fits.not_converged,
        })
    }
}

/// Check the settings and the inputs of an analysis.
fn check<E: SimulationEngine>(
    problem: &CalibrationProblem<E>,
    sample: &ParameterSample,
    observations: &[ObservedDataPoint],
    kind: ResidualKind,
    settings: &ResidualSettings,
) -> Result<(), MethodError> {
    settings.max_lag.check().map_err(MethodError::Invalid)?;
    RANDOMIZATIONS
        .check(settings.randomizations)
        .map_err(MethodError::Invalid)?;
    CONFIDENCE_LEVEL
        .check(settings.confidence_level)
        .map_err(MethodError::Invalid)?;
    if kind == ResidualKind::Additive && settings.randomizations != 1 {
        return Err(MethodError::Invalid(format!(
            "NoNoise() reads the additive residuals, which have no randomization, so \
             randomizations must be 1; got {}",
            settings.randomizations
        )));
    }
    check_names(problem.forward(), sample)?;
    if let Reference::Refitted(_) = settings.reference {
        if sample.len() != 1 {
            return Err(MethodError::Invalid(format!(
                "Refitted(...) refits data drawn at one point, and the sample has {} draws. \
                 Give the sample of a fit, Calibrator.point_sample(result), or use \
                 AtParameters(...)",
                sample.len()
            )));
        }
        if observations != problem.observed_data() {
            return Err(MethodError::Invalid(
                "Refitted(...) refits the data of the problem, and these observations are not \
                 the problem's. Give the observations of the problem, or use AtParameters(...) \
                 for held-out data"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// The dependence of the residuals of `sample` at `observations`: of each
/// draw, the residuals `kind` reads, the quantile residuals of
/// `settings.randomizations` randomizations, whose places `v` come from the
/// stream [`Stream::DependenceRandomizations`] of `seed` and are common to
/// every draw, or the additive residuals; their statistics averaged over the
/// randomizations and, with the weights of the draws, over the draws (`τ`
/// as the ratio of the mean long-run variance to the mean variance).
///
/// With one draw, the tests and the band of `τ` come from
/// `settings.reference`: [`Reference::AtParameters`] draws data sets from
/// the observation models at the draw (the stream
/// [`Stream::DependenceReference`]) and reads `Q̃` by the law of their
/// statistics, or with additive residuals tests `Q̃` against `χ²(h)` and
/// takes the band from wild sets of the residuals ([`Stream::DependenceSigns`]);
/// [`Reference::Refitted`] draws data sets at the draw (the observation
/// models, or the fit plus its residuals with signs ±1), fits each again
/// with `config` from its starts, sending the events of the refits to
/// `steps`, and reads every statistic by the law of the residuals of the
/// refits, with Monte Carlo p-values `(1 + #{Q_b ≥ Q}) / (B + 1)`. With many
/// draws, the result gives the spread of the p-values of the draws, each
/// against its own reference at its parameters, and no test of its own.
///
/// # Errors
///
/// [`MethodError::Invalid`] when a setting is outside its range, the
/// sample's parameters are not the problem's, quantile residuals have no
/// observation models or an observed value outside the support of its
/// model, additive residuals have more than one randomization, or a refit
/// reference reads many draws or observations that are not the problem's;
/// a prediction error when a draw has no run; [`MethodError::Failed`] when
/// every refit fails.
#[allow(clippy::too_many_arguments)]
pub fn residual_dependence<E>(
    problem: &CalibrationProblem<E>,
    config: &OptimizationConfig,
    sample: &ParameterSample,
    observations: &[ObservedDataPoint],
    kind: ResidualKind,
    settings: &ResidualSettings,
    seed: u64,
    steps: &RunnerSteps<'_>,
) -> Result<SeriesDependence, MethodError>
where
    E: SimulationEngine + Send + Sync,
{
    check(problem, sample, observations, kind, settings)?;
    let (noise, reader) = match kind {
        ResidualKind::Quantile => {
            let noise = DistributionNoise::new(problem).map_err(MethodError::Invalid)?;
            let terms = noise
                .likelihood()
                .check_values(observations)
                .map_err(MethodError::Invalid)?;
            let places = (0..settings.randomizations as u64)
                .map(|r| {
                    let mut generator =
                        Generator::from_seed(Stream::DependenceRandomizations.replicate(seed, r));
                    observations.iter().map(|_| generator.uniform()).collect()
                })
                .collect();
            let reader = Reader::Quantile {
                likelihood: noise.likelihood().clone(),
                terms,
                places,
            };
            (Some(noise), reader)
        }
        ResidualKind::Additive => (
            None,
            Reader::Additive {
                summary: problem.discrepancy().summary(),
            },
        ),
    };
    let layout = Layout::new(observations);
    let lags = lags_of(&layout, settings.max_lag);
    let analysis = Analysis {
        problem,
        observations,
        layout,
        lags,
        reader,
        noise,
        seed,
    };
    let runs = observe_sets(problem.forward(), sample.values(), observations, None)?;
    let draws: Vec<Draw> = runs
        .into_par_iter()
        .zip(sample.values())
        .map(|(means, values)| {
            let residuals = analysis.reader.sets(observations, values, &means);
            let statistics = statistics_of(&analysis.layout, &analysis.lags, &residuals);
            Draw {
                values,
                means,
                residuals,
                statistics,
            }
        })
        .collect();
    let weights = sample.weights();
    let mut weighted: Vec<(f64, &SetStatistics)> = Vec::new();
    for (draw, &weight) in draws.iter().zip(weights) {
        let share = weight / draw.statistics.len() as f64;
        weighted.extend(draw.statistics.iter().map(|set| (share, set)));
    }
    let summary = summarize(&weighted);
    let mut failed = Vec::new();
    let mut not_converged = 0;
    let tests = if let [draw] = draws.as_slice() {
        let observed = summary_of(&draw.statistics);
        let (replicates, portmanteau) = match &settings.reference {
            Reference::AtParameters(reference) => {
                analysis.at_parameters(0, draw, reference.replicates)
            }
            Reference::Refitted(reference) => {
                let refits = analysis.refitted(
                    draw,
                    reference.replicates,
                    reference.starts,
                    config,
                    steps,
                )?;
                failed = refits.failed;
                not_converged = refits.not_converged;
                (refits.summaries, Portmanteau::Replicates)
            }
        };
        Tests::One(test(
            &observed,
            &replicates,
            portmanteau,
            settings.confidence_level,
        ))
    } else {
        let replicates = settings.reference.replicates();
        let per_draw: Vec<_> = draws
            .par_iter()
            .enumerate()
            .map(|(index, draw)| {
                let (reference, portmanteau) = analysis.at_parameters(index, draw, replicates);
                test(
                    &summary_of(&draw.statistics),
                    &reference,
                    portmanteau,
                    settings.confidence_level,
                )
            })
            .collect();
        let series = (0..analysis.layout.series.len())
            .map(|s| {
                let p_values: Option<Vec<f64>> = per_draw
                    .iter()
                    .map(|test| test.series[s].and_then(|series| series.p_value))
                    .collect();
                p_values.map(|p| spread(&p, weights, settings.confidence_level))
            })
            .collect();
        let factors: Option<Vec<f64>> = per_draw.iter().map(|test| test.problem).collect();
        Tests::Draws {
            series,
            problem: factors.map(|f| spread(&f, weights, settings.confidence_level)),
        }
    };
    let mut dependence = result(
        &analysis.layout,
        &analysis.lags,
        &summary,
        &tests,
        draws.len(),
    );
    dependence.failed = failed;
    dependence.not_converged = not_converged;
    Ok(dependence)
}
