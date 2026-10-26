"""The runners of the local approximations, which
``Calibrator.uncertainty`` calls through the ``_runner`` of each method.

Each runner sends its method whole to the engine, which fits the problem,
differentiates it at the fit, or at the end of each start, and draws on the
unbounded scale.
"""

import logging
from typing import TYPE_CHECKING

from commol._calibration.method import CalibrationPort
from commol._data.engine_json import engine_json
from commol._fit.starts import fit_result
from commol.commol_rs import _commol_rs as commol_rs
from commol.local.results import (
    ImportanceResult,
    LaplaceResult,
    MixtureComponent,
    SandwichResult,
)
from commol.resampling.runners import block_length_estimates, sample_method
from commol.uncertainty.sample import ParameterSample, SampleMethod

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import LocalRunProtocol
    from commol.local.methods import Laplace, LaplaceMixture, Sandwich

logger = logging.getLogger("commol.local")


def _sample(
    calibrator: CalibrationPort, run: "LocalRunProtocol", method: SampleMethod
) -> ParameterSample:
    parameters = calibrator.problem.parameters
    return ParameterSample.equally_weighted(
        [p.id for p in parameters],
        [(p.min_bound, p.max_bound) for p in parameters],
        run.values,
        method,
    )


def _notes(run: "LocalRunProtocol") -> list[str]:
    for note in run.notes:
        logger.warning(note)
    return list(run.notes)


def run_laplace(calibrator: CalibrationPort, method: "Laplace") -> LaplaceResult:
    """
    Run the Laplace approximation at the fit.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem, with observation models.
    method : Laplace
        The settings.

    Returns
    -------
    LaplaceResult
        The fit, the covariance on the unbounded scale and the draws.
    """
    starts, run = commol_rs.inference.laplace(
        calibrator._engine_problem, engine_json(method)
    )
    return LaplaceResult(
        fit=fit_result(starts, calibrator.parameter_names, calibrator.seed, logger),
        covariance=run.covariance,
        sample=_sample(
            calibrator,
            run,
            sample_method(calibrator, "laplace", method),
        ),
        notes=_notes(run),
    )


def run_sandwich(calibrator: CalibrationPort, method: "Sandwich") -> SandwichResult:
    """
    Run the sandwich at the fit.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : Sandwich
        The settings.

    Returns
    -------
    SandwichResult
        The fit, the sandwich covariance on the unbounded scale, the draws
        and the blocks.
    """
    starts, run, blocks, block_length, estimates = commol_rs.inference.sandwich(
        calibrator._engine_problem, engine_json(method)
    )
    return SandwichResult(
        fit=fit_result(starts, calibrator.parameter_names, calibrator.seed, logger),
        covariance=run.covariance,
        sample=_sample(
            calibrator,
            run,
            sample_method(calibrator, "sandwich", method),
        ),
        notes=_notes(run),
        blocks=blocks,
        block_length=block_length,
        block_length_estimates=block_length_estimates(estimates),
    )


def run_laplace_mixture(
    calibrator: CalibrationPort, method: "LaplaceMixture"
) -> ImportanceResult:
    """
    Fit from several starts and sample the target with the Laplace mixture of
    their ends.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : LaplaceMixture
        The settings.

    Returns
    -------
    ImportanceResult
        The fit, the weighted draws, the components, ``k_hat`` and the
        effective size.
    """
    starts, run = commol_rs.inference.laplace_mixture(
        calibrator._engine_problem, engine_json(method)
    )
    names = calibrator.parameter_names
    fit = fit_result(starts, names, calibrator.seed, logger)
    for note in run.notes:
        logger.warning(note)
    parameters = calibrator.problem.parameters
    return ImportanceResult(
        fit=fit,
        sample=ParameterSample(
            names=names,
            bounds=[(p.min_bound, p.max_bound) for p in parameters],
            values=run.values,
            log_weights=run.log_weights,
            method=sample_method(calibrator, "laplace_mixture", method),
        ),
        components=[
            MixtureComponent(
                center=dict(zip(names, center, strict=True)),
                weight=weight,
                covariance=covariance,
                log_evidence=log_evidence,
            )
            for center, weight, covariance, log_evidence in zip(
                run.centers,
                run.weights,
                run.covariances,
                run.log_evidences,
                strict=True,
            )
        ],
        k_hat=run.k_hat,
        effective_size=run.effective_size,
        reliable=run.reliable,
        notes=list(run.notes),
    )
