"""The `CalibrationResult` of the starts of a fit that the engine ran: of
`Calibrator.run`, and of each method that fits the problem before it runs."""

import logging
from typing import TYPE_CHECKING

from commol._fit.result import CalibrationResult, StartSummary

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import StartsProtocol


def failed_starts(run: "StartsProtocol", log: logging.Logger) -> list[str]:
    """
    Each start of a fit that failed, logged as a warning.

    Parameters
    ----------
    run : StartsProtocol
        The outcome of each start, as the engine returns them.
    log : logging.Logger
        The logger of the caller.

    Returns
    -------
    list[str]
        ``"Start <index> failed: <reason>"`` for each start that failed.
    """
    failed = [
        f"Start {index} failed: {outcome.error}"
        for index, outcome in enumerate(run.outcomes)
        if outcome.result is None
    ]
    for reason in failed:
        log.warning(reason)
    return failed


def fit_result(
    run: "StartsProtocol", names: list[str], seed: int, log: logging.Logger
) -> CalibrationResult:
    """
    The result of the starts of a fit: the best of them, and every start.

    Each start that failed is logged as a warning.

    Parameters
    ----------
    run : StartsProtocol
        The outcome of each start, as the engine returns them.
    names : list[str]
        The names of the parameters, in the order of the problem.
    seed : int
        The seed of the calibration.
    log : logging.Logger
        The logger of the caller, which logs each start that failed.

    Returns
    -------
    CalibrationResult
        The best start, every start that finished, each start that failed and
        the seed.
    """
    summaries = [
        StartSummary(
            start=dict(zip(names, outcome.start, strict=True)),
            end=dict(outcome.result.best_parameters),
            loss=outcome.result.final_loss,
            converged=outcome.result.converged,
        )
        for outcome in run.outcomes
        if outcome.result is not None
    ]
    best = run.outcomes[run.best].result
    assert best is not None, "the best start did not fail"
    return CalibrationResult(
        best_parameters=dict(best.best_parameters),
        final_loss=best.final_loss,
        iterations=best.iterations,
        converged=best.converged,
        termination_reason=best.termination_reason,
        starts=summaries,
        failed=failed_starts(run, log),
        seed=seed,
    )
