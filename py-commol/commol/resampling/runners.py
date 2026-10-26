"""The runners of the resampling methods, registered for
``Calibrator.uncertainty``.

Each runner sends its method whole to the engine, which fits the problem and
then draws and fits every replicate in parallel.
"""

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from commol._calibration.method import CalibrationPort
from commol._data.engine_json import engine_json
from commol._fit.result import CalibrationResult
from commol._fit.starts import fit_result
from commol.commol_rs import _commol_rs as commol_rs
from commol.resampling.results import (
    BootstrapResult,
    JackknifeResult,
    ResamplingDiagnostics,
)
from commol.uncertainty.sample import ParameterSample, SampleMethod

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import BootstrapRunProtocol
    from commol.resampling.methods import (
        BayesianBootstrap,
        BlockJackknife,
        ParametricBootstrap,
        ResidualBootstrap,
        WildBootstrap,
    )

    type _Bootstrap = (
        BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap
    )

logger = logging.getLogger("commol.resampling")


def sample_method(
    calibrator: CalibrationPort, name: str, method: BaseModel
) -> SampleMethod:
    """
    What made the sample of a method: its name, its settings and the seed.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    name : str
        The name of the method.
    method : BaseModel
        The settings of the method.

    Returns
    -------
    SampleMethod
        The settings of `method`, with ``"seed"`` set to ``calibrator.seed``,
        which repeats the sample when set as ``problem.seed``.
    """
    settings = method.model_dump(mode="json")
    settings["seed"] = calibrator.seed
    return SampleMethod(name=name, settings=settings)


def block_length_estimates(
    estimates: list[tuple[str, float]] | None,
) -> dict[str, float] | None:
    """
    The estimates of an automatic block length as a result gives them.

    Parameters
    ----------
    estimates : list[tuple[str, float]] | None
        The estimate of each series, as the Rust runners give it; None for a
        length given.

    Returns
    -------
    dict[str, float] | None
        The estimate of each series, by label; None for a length given.
    """
    return dict(estimates) if estimates is not None else None


def log_failed(failed: list[str], log: logging.Logger = logger) -> list[str]:
    """
    Log each fit that failed as a warning.

    Parameters
    ----------
    failed : list[str]
        Each fit that failed, as the Rust runner names it.
    log : logging.Logger
        The logger of the method's runner.

    Returns
    -------
    list[str]
        `failed`, for the result.
    """
    for entry in failed:
        log.warning(entry)
    return failed


def _diagnostics(
    block_length: int | None,
    estimates: list[tuple[str, float]] | None,
    negative_values: int,
) -> ResamplingDiagnostics:
    return ResamplingDiagnostics(
        block_length=block_length,
        block_length_estimates=block_length_estimates(estimates),
        negative_values=negative_values,
    )


def _bootstrap_result(
    calibrator: CalibrationPort,
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
        failed=log_failed(run.failed),
        not_converged=run.not_converged,
        diagnostics=_diagnostics(
            run.block_length, run.block_length_estimates, run.negative_values
        ),
    )


def run_bootstrap(
    calibrator: CalibrationPort,
    method: "_Bootstrap",
) -> BootstrapResult:
    """
    Run a bootstrap: Bayesian, parametric, residual or wild, by its ``kind``.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem; a parametric bootstrap needs observation
        models.
    method : BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap
        The settings.

    Returns
    -------
    BootstrapResult
        The fit, one draw per replicate and the diagnostics.
    """
    starts, run = commol_rs.inference.bootstrap(
        calibrator._engine_problem, engine_json(method)
    )
    return _bootstrap_result(
        calibrator,
        fit_result(starts, calibrator.parameter_names, calibrator.seed, logger),
        run,
        sample_method(calibrator, f"{method.kind}_bootstrap", method),
    )


def run_block_jackknife(
    calibrator: CalibrationPort, method: "BlockJackknife"
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
    starts, run = commol_rs.inference.block_jackknife(
        calibrator._engine_problem, engine_json(method)
    )
    names = calibrator.parameter_names
    fit = fit_result(starts, names, calibrator.seed, logger)

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
        failed=log_failed(run.failed),
        not_converged=run.not_converged,
        diagnostics=_diagnostics(run.block_length, run.block_length_estimates, 0),
    )
