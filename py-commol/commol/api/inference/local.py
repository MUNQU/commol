"""The runners of the local approximations, registered for
``Calibrator.uncertainty``.

Each runner fits the problem with ``Calibrator.run`` and hands the fit to the
Rust runner, which differentiates the problem there and draws on the
unbounded scale.
"""

import logging
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.api.inference.resampling import fit_values
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.inference.local import Laplace, LaplaceResult
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


register(Laplace, run_laplace)
