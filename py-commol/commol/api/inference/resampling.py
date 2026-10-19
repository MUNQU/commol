"""The runners of the resampling methods, registered for
``Calibrator.uncertainty``.

Each runner fits the problem with ``Calibrator.run`` and hands the fit to the
Rust runner, which draws and fits every replicate in parallel.
"""

import logging
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.context.calibration import CalibrationResult
from commol.context.inference.resampling import (
    BayesianBootstrap,
    BlockJackknife,
    BlockLength,
    BootstrapResult,
    JackknifeResult,
    ResamplingDiagnostics,
    ResidualBootstrap,
    WildBootstrap,
)
from commol.context.inference.sample import ParameterSample, SampleMethod

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator
    from commol.commol_rs._commol_rs import BootstrapRunProtocol

logger = logging.getLogger(__name__)


def fit_values(calibrator: "Calibrator", fit: CalibrationResult) -> list[float]:
    """
    The values of a fit in the order of the problem's parameters.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    fit : CalibrationResult
        A fit of that problem.

    Returns
    -------
    list[float]
        One value per parameter.
    """
    return [fit.best_parameters[name] for name in calibrator.parameter_names]


def rust_block_length(block_length: BlockLength) -> int | None:
    """
    A block length as the Rust runners take it.

    Parameters
    ----------
    block_length : int | "auto"
        Positions per block.

    Returns
    -------
    int | None
        The length, or None for ``"auto"``.
    """
    return None if block_length == "auto" else block_length


def _diagnostics(
    block_length: int,
    estimates: list[tuple[str, float]] | None,
    failed: list[str],
    not_converged: int,
    negative_values: int,
) -> ResamplingDiagnostics:
    for reason in failed:
        logger.warning(reason)
    return ResamplingDiagnostics(
        block_length=block_length,
        block_length_estimates=dict(estimates) if estimates is not None else None,
        failed=failed,
        not_converged=not_converged,
        negative_values=negative_values,
    )


def _bootstrap_result(
    calibrator: "Calibrator",
    fit: CalibrationResult,
    run: "BootstrapRunProtocol",
    method: SampleMethod,
) -> BootstrapResult:
    parameters = calibrator.problem.parameters
    return BootstrapResult(
        fit=fit,
        sample=ParameterSample.equally_weighted(
            [p.id for p in parameters],
            [(p.min_bound, p.max_bound) for p in parameters],
            run.values,
            method,
            groups=run.groups,
        ),
        diagnostics=_diagnostics(
            run.block_length,
            run.block_length_estimates,
            [f"Replicate {index} failed: {reason}" for index, reason in run.failed],
            run.not_converged,
            run.negative_values,
        ),
    )


def run_bayesian_bootstrap(
    calibrator: "Calibrator", method: BayesianBootstrap
) -> BootstrapResult:
    """
    Run a Bayesian (weighted) bootstrap.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : BayesianBootstrap
        The settings.

    Returns
    -------
    BootstrapResult
        The fit, one draw per replicate and the diagnostics.
    """
    fit = calibrator.run(starts=method.starts)
    run = calibrator.rust_problem.bayesian_bootstrap(
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        method.replicates,
        rust_block_length(method.block_length),
        method.starts,
        calibrator.seed,
    )
    return _bootstrap_result(
        calibrator,
        fit,
        run,
        SampleMethod(
            name="bayesian_bootstrap", settings=method.model_dump(mode="json")
        ),
    )


def run_residual_bootstrap(
    calibrator: "Calibrator", method: ResidualBootstrap
) -> BootstrapResult:
    """
    Run a residual block bootstrap.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : ResidualBootstrap
        The settings.

    Returns
    -------
    BootstrapResult
        The fit, one draw per replicate and the diagnostics.
    """
    fit = calibrator.run(starts=method.starts)
    run = calibrator.rust_problem.residual_bootstrap(
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        method.replicates,
        rust_block_length(method.block_length),
        method.starts,
        calibrator.seed,
        method.residual_scale,
        method.scheme,
    )
    return _bootstrap_result(
        calibrator,
        fit,
        run,
        SampleMethod(
            name="residual_bootstrap", settings=method.model_dump(mode="json")
        ),
    )


def run_wild_bootstrap(
    calibrator: "Calibrator", method: WildBootstrap
) -> BootstrapResult:
    """
    Run a wild block bootstrap.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : WildBootstrap
        The settings.

    Returns
    -------
    BootstrapResult
        The fit, one draw per replicate and the diagnostics.
    """
    fit = calibrator.run(starts=method.starts)
    run = calibrator.rust_problem.wild_bootstrap(
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        method.replicates,
        rust_block_length(method.block_length),
        method.starts,
        calibrator.seed,
        method.multipliers,
    )
    return _bootstrap_result(
        calibrator,
        fit,
        run,
        SampleMethod(name="wild_bootstrap", settings=method.model_dump(mode="json")),
    )


def run_block_jackknife(
    calibrator: "Calibrator", method: BlockJackknife
) -> JackknifeResult:
    """
    Run a block jackknife.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : BlockJackknife
        The settings.

    Returns
    -------
    JackknifeResult
        The fit, the jackknife statistics and the diagnostics.
    """
    fit = calibrator.run(starts=method.starts)
    run = calibrator.rust_problem.jackknife(
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        rust_block_length(method.block_length),
        method.starts,
        calibrator.seed,
    )
    names = calibrator.parameter_names

    def by_parameter(rows: list[list[float]]) -> dict[str, list[float]]:
        return {name: [row[j] for row in rows] for j, name in enumerate(names)}

    return JackknifeResult(
        fit=fit,
        estimate=dict(zip(names, run.estimate, strict=True)),
        bias=dict(zip(names, run.bias, strict=True)),
        standard_error=dict(zip(names, run.standard_error, strict=True)),
        blocks=run.spans,
        pseudo_values=by_parameter(run.pseudo_values),
        influence=by_parameter(run.influence),
        diagnostics=_diagnostics(
            run.block_length,
            run.block_length_estimates,
            [
                f"Block [{first}, {last}] failed: {reason}"
                for (first, last), reason in run.failed
            ],
            run.not_converged,
            0,
        ),
    )


register(BayesianBootstrap, run_bayesian_bootstrap)
register(ResidualBootstrap, run_residual_bootstrap)
register(WildBootstrap, run_wild_bootstrap)
register(BlockJackknife, run_block_jackknife)
