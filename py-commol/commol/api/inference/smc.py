"""The runners of the sequential Monte Carlo methods, registered for
``Calibrator.uncertainty``.

Each runner hands the priors and the settings to the Rust sampler, which
draws, runs and moves every particle in parallel, and builds the result from
the run.
"""

import logging
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.api.inference.resampling import fit_values, rust_block_length
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.calibration import CalibrationResult
from commol.context.inference.sample import ParameterSample, SampleMethod
from commol.context.inference.smc import (
    ABCDistance,
    ABCRejection,
    ABCSMC,
    BayesianPosterior,
    BootstrapCoverage,
    GibbsPosterior,
    GibbsResult,
    InformationMatching,
    LossDistance,
    SMCDiagnostics,
    SMCResult,
)

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator
    from commol.commol_rs._commol_rs import SMCRunProtocol

logger = logging.getLogger(__name__)

rust = commol_rs.inference

ONE_REPLICATE = (
    "Each distance reads one fresh replicate of the stochastic engine, seeded "
    "from its particle: the problem's replicates and replicate_summary do not "
    "apply."
)


def _sample(
    calibrator: "Calibrator", run: "SMCRunProtocol", method: SampleMethod
) -> ParameterSample:
    parameters = calibrator.problem.parameters
    return ParameterSample(
        names=[p.id for p in parameters],
        bounds=[(p.min_bound, p.max_bound) for p in parameters],
        values=run.values,
        log_weights=run.log_weights,
        method=method,
    )


def _diagnostics(run: "SMCRunProtocol", notes: list[str]) -> SMCDiagnostics:
    if run.stop_reason is not None:
        logger.info(run.stop_reason)
    return SMCDiagnostics(
        effective_sizes=run.effective_sizes,
        resampled=run.resampled,
        acceptance_rates=run.acceptance_rates,
        move_counts=run.move_counts,
        evaluations=run.evaluations,
        stop_reason=run.stop_reason,
        notes=notes,
    )


def _metric(distance: ABCDistance) -> str | None:
    """The metric of the series, or None for the problem's loss."""
    return None if isinstance(distance, LossDistance) else distance.metric


def run_gibbs_posterior(
    calibrator: "Calibrator", method: GibbsPosterior
) -> GibbsResult:
    """
    Sample a Gibbs posterior, after calibrating its `η` if asked.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : GibbsPosterior
        The settings.

    Returns
    -------
    GibbsResult
        The sample, the levels, the log evidence, the diagnostics and `η`.
    """
    problem = calibrator.rust_problem
    priors = method.priors.rust_priors()
    sampler = (
        method.particles,
        method.resample_threshold,
        method.target_cess,
        method.stay_probability,
    )
    fit: CalibrationResult | None = None
    coverage: float | None = None
    notes: list[str] = []
    if isinstance(method.eta, InformationMatching):
        fit = calibrator.run()
        eta = rust.information_matching_eta(
            problem,
            fit_values(calibrator, fit),
            rust_block_length(method.eta.block_length),
        )
    elif isinstance(method.eta, BootstrapCoverage):
        fit = calibrator.run()
        calibration = method.eta
        found = rust.bootstrap_coverage_eta(
            problem,
            priors,
            fit_values(calibrator, fit),
            calibration.replicates,
            rust_block_length(calibration.block_length),
            calibration.confidence_level,
            *sampler,
            calibrator.seed,
        )
        eta, coverage = found.eta, found.coverage
        for index, reason in found.failed:
            notes.append(f"Bootstrap replicate {index} failed: {reason}")
            logger.warning(notes[-1])
    else:
        eta = method.eta
    run = rust.gibbs_posterior(problem, priors, eta, *sampler, calibrator.seed)
    return GibbsResult(
        sample=_sample(
            calibrator,
            run,
            SampleMethod(
                name="gibbs_posterior", settings=method.model_dump(mode="json")
            ),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, notes),
        eta=eta,
        fit=fit,
        coverage=coverage,
    )


def run_bayesian_posterior(
    calibrator: "Calibrator", method: BayesianPosterior
) -> SMCResult:
    """
    Sample the posterior of the problem's observation models.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem, with observation models.
    method : BayesianPosterior
        The settings.

    Returns
    -------
    SMCResult
        The sample, the temperatures, the log evidence and the diagnostics.
    """
    run = rust.bayesian_posterior(
        calibrator.rust_problem,
        method.priors.rust_priors(),
        method.particles,
        method.resample_threshold,
        method.target_cess,
        method.stay_probability,
        calibrator.seed,
    )
    return SMCResult(
        sample=_sample(
            calibrator,
            run,
            SampleMethod(
                name="bayesian_posterior", settings=method.model_dump(mode="json")
            ),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, []),
    )


def run_abc_smc(calibrator: "Calibrator", method: ABCSMC) -> SMCResult:
    """
    Sample ABC by sequential Monte Carlo.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : ABCSMC
        The settings.

    Returns
    -------
    SMCResult
        The sample, the tolerances, the log evidence and the diagnostics.
    """
    run, stochastic = rust.abc_smc(
        calibrator.rust_problem,
        method.priors.rust_priors(),
        _metric(method.distance),
        method.particles,
        method.target_cess,
        method.min_acceptance,
        method.stay_probability,
        calibrator.seed,
    )
    return SMCResult(
        sample=_sample(
            calibrator,
            run,
            SampleMethod(name="abc_smc", settings=method.model_dump(mode="json")),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, [ONE_REPLICATE] if stochastic else []),
    )


def run_abc_rejection(calibrator: "Calibrator", method: ABCRejection) -> SMCResult:
    """
    Run ABC rejection.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : ABCRejection
        The settings.

    Returns
    -------
    SMCResult
        The kept draws, the tolerance, the log evidence and the diagnostics.
    """
    run, stochastic = rust.abc_rejection(
        calibrator.rust_problem,
        method.priors.rust_priors(),
        _metric(method.distance),
        method.draws,
        method.quantile,
        calibrator.seed,
    )
    return SMCResult(
        sample=_sample(
            calibrator,
            run,
            SampleMethod(name="abc_rejection", settings=method.model_dump(mode="json")),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, [ONE_REPLICATE] if stochastic else []),
    )


register(GibbsPosterior, run_gibbs_posterior)
register(BayesianPosterior, run_bayesian_posterior)
register(ABCSMC, run_abc_smc)
register(ABCRejection, run_abc_rejection)
