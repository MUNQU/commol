"""The runners of the sequential Monte Carlo methods, which
``Calibrator.uncertainty`` calls through the ``_runner`` of each method.

Each runner sends its method whole to the engine, which fits the problem and
calibrates `η` when the method asks, then draws, runs and moves every
particle in parallel; the runner builds the result from the run.
"""

import logging
from typing import TYPE_CHECKING

from commol._calibration.method import CalibrationPort
from commol._data.engine_json import engine_json
from commol._fit.starts import fit_result
from commol.commol_rs import _commol_rs as commol_rs
from commol.resampling.runners import log_failed, sample_method
from commol.smc.results import GibbsResult, SMCDiagnostics, SMCResult
from commol.uncertainty.sample import ParameterSample, SampleMethod

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import EtaCalibrationProtocol, SMCRunProtocol
    from commol.smc.methods import (
        ABCSMC,
        ABCRejection,
        BayesianPosterior,
        GibbsPosterior,
    )

logger = logging.getLogger("commol.smc")

ONE_REPLICATE = (
    "Each distance reads one fresh replicate of the stochastic engine, seeded "
    "from its particle: the problem's replicates and replicate_summary do not "
    "apply."
)


def _sample(
    calibrator: CalibrationPort, run: "SMCRunProtocol", method: SampleMethod
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


def _eta_notes(eta: "EtaCalibrationProtocol") -> list[str]:
    """The notes of a calibration of `η`, logged as warnings: few degrees of
    freedom."""
    if eta.few_coverage is None:
        return []
    level = commol_rs.inference.FEW_COVERAGE_LEVEL
    note = (
        f"η rests on few degrees of freedom, ν = {eta.degrees_of_freedom:.3g} "
        f"(block_length {eta.block_length}): {level:.0%} intervals are expected to "
        f"cover about {eta.few_coverage:.0%}, with flat priors and noise of one scale "
        "that is independent between blocks and between series."
    )
    if eta.block_length != 1:
        note += (
            " Shorter blocks can give more, but must still hold the dependence "
            "of the residuals."
        )
    logger.warning(note)
    return [note]


def run_gibbs_posterior(
    calibrator: CalibrationPort, method: "GibbsPosterior"
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
        The sample, the levels, the log evidence, the diagnostics, `η`
        and, with a calibration of `η`, its block length and the degrees
        of freedom of `η`.
    """
    starts, run, eta = commol_rs.inference.gibbs_posterior(
        calibrator._engine_problem, engine_json(method), logger
    )
    fit = (
        None
        if starts is None
        else fit_result(starts, calibrator.parameter_names, calibrator.seed, logger)
    )
    return GibbsResult(
        sample=_sample(
            calibrator,
            run,
            sample_method(calibrator, "gibbs_posterior", method),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, _eta_notes(eta)),
        eta=eta.eta,
        fit=fit,
        coverage=eta.coverage,
        failed=log_failed(eta.failed, logger),
        block_length=eta.block_length,
        degrees_of_freedom=eta.degrees_of_freedom,
    )


def run_bayesian_posterior(
    calibrator: CalibrationPort, method: "BayesianPosterior"
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
    run = commol_rs.inference.bayesian_posterior(
        calibrator._engine_problem, engine_json(method), logger
    )
    return SMCResult(
        sample=_sample(
            calibrator,
            run,
            sample_method(calibrator, "bayesian_posterior", method),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, []),
    )


def run_abc_smc(calibrator: CalibrationPort, method: "ABCSMC") -> SMCResult:
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
    run, stochastic = commol_rs.inference.abc_smc(
        calibrator._engine_problem, engine_json(method), logger
    )
    return SMCResult(
        sample=_sample(
            calibrator,
            run,
            sample_method(calibrator, "abc_smc", method),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, [ONE_REPLICATE] if stochastic else []),
    )


def run_abc_rejection(calibrator: CalibrationPort, method: "ABCRejection") -> SMCResult:
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
    run, stochastic = commol_rs.inference.abc_rejection(
        calibrator._engine_problem, engine_json(method), logger
    )
    return SMCResult(
        sample=_sample(
            calibrator,
            run,
            sample_method(calibrator, "abc_rejection", method),
        ),
        levels=run.levels,
        kept=run.kept,
        log_evidence=run.log_evidence,
        diagnostics=_diagnostics(run, [ONE_REPLICATE] if stochastic else []),
    )
