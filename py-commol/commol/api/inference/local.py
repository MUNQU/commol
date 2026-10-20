"""The runners of the local approximations, registered for
``Calibrator.uncertainty``.

Each runner fits the problem with ``Calibrator.run`` and hands the fit, or
the ends of its starts, to the Rust runner, which differentiates the problem
there and draws on the unbounded scale.
"""

import logging
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.api.inference.resampling import fit_values, rust_block_length
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.inference.local import (
    GibbsTarget,
    ImportanceResult,
    Laplace,
    LaplaceMixture,
    LaplaceResult,
    MixtureComponent,
    Sandwich,
    SandwichResult,
)
from commol.context.inference.sample import ParameterSample, SampleMethod

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator
    from commol.commol_rs._commol_rs import LocalRunProtocol

logger = logging.getLogger(__name__)

rust = commol_rs.inference


def _sample(
    calibrator: "Calibrator", run: "LocalRunProtocol", method: SampleMethod
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


def run_laplace(calibrator: "Calibrator", method: Laplace) -> LaplaceResult:
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
    fit = calibrator.run()
    run = rust.laplace(
        calibrator.rust_problem,
        fit_values(calibrator, fit),
        method.curvature,
        method.draws,
        calibrator.seed,
    )
    return LaplaceResult(
        fit=fit,
        covariance=run.covariance,
        sample=_sample(
            calibrator,
            run,
            SampleMethod(name="laplace", settings=method.model_dump(mode="json")),
        ),
        notes=_notes(run),
    )


def run_sandwich(calibrator: "Calibrator", method: Sandwich) -> SandwichResult:
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
    fit = calibrator.run()
    run, blocks, block_length, estimates = rust.sandwich(
        calibrator.rust_problem,
        fit_values(calibrator, fit),
        method.curvature,
        rust_block_length(method.block_length),
        method.draws,
        calibrator.seed,
    )
    return SandwichResult(
        fit=fit,
        covariance=run.covariance,
        sample=_sample(
            calibrator,
            run,
            SampleMethod(name="sandwich", settings=method.model_dump(mode="json")),
        ),
        notes=_notes(run),
        blocks=blocks,
        block_length=block_length,
        block_length_estimates=dict(estimates) if estimates is not None else None,
    )


def run_laplace_mixture(
    calibrator: "Calibrator", method: LaplaceMixture
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
    fit = calibrator.run(starts=method.starts)
    names = calibrator.parameter_names
    target = method.target
    run = rust.laplace_mixture(
        calibrator.rust_problem,
        target.priors.rust_priors(),
        target.eta if isinstance(target, GibbsTarget) else None,
        [[start.end[name] for name in names] for start in fit.starts],
        method.draws,
        method.curvature,
        calibrator.seed,
    )
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
            method=SampleMethod(
                name="laplace_mixture", settings=method.model_dump(mode="json")
            ),
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


register(Laplace, run_laplace)
register(LaplaceMixture, run_laplace_mixture)
register(Sandwich, run_sandwich)
