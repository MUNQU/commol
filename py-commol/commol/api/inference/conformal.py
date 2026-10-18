"""The runners of the conformal methods, registered for
``Calibrator.uncertainty``.

Each runner fits the problem with ``Calibrator.run`` when its predictor is a
fit and hands the rest to Rust, which predicts, scores, refits and takes the
quantiles.
"""

import logging
from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.api.inference.resampling import fit_values, rust_block_length
from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from commol.context.inference.conformal import (
    ConformalResult,
    EnbPI,
    JackknifePlus,
    PredictionIntervals,
    RollingConformal,
    SeriesIntervals,
    SplitConformal,
)

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator
    from commol.commol_rs._commol_rs import ConformalRunProtocol

logger = logging.getLogger(__name__)


def _log_failed(run: "ConformalRunProtocol") -> None:
    for index, reason in run.failed:
        logger.warning("Refit %d failed: %s", index, reason)


def _result(
    run: "ConformalRunProtocol", level: float, guarantee: str
) -> ConformalResult:
    series: dict[str, dict[str, list[float]]] = {}
    unbounded: dict[str, list[float]] = {}
    for label, at, prediction, lower, upper in zip(
        run.labels, run.positions, run.prediction, run.lower, run.upper, strict=True
    ):
        entry = series.setdefault(
            label, {"positions": [], "prediction": [], "lower": [], "upper": []}
        )
        entry["positions"].append(at)
        entry["prediction"].append(prediction)
        entry["lower"].append(lower)
        entry["upper"].append(upper)
        if not (abs(lower) < float("inf") and abs(upper) < float("inf")):
            unbounded.setdefault(label, []).append(at)
    scores: dict[str, list[float]] = {}
    for label, score in run.scores:
        scores.setdefault(label, []).append(score)
    return ConformalResult(
        intervals=PredictionIntervals(
            level=level,
            series={
                label: SeriesIntervals(**values) for label, values in series.items()
            },
            unbounded=unbounded,
        ),
        scores=scores,
        guarantee=guarantee,
    )


def run_split_conformal(
    calibrator: "Calibrator", method: SplitConformal
) -> ConformalResult:
    """
    Run split conformal prediction.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : SplitConformal
        The settings.

    Returns
    -------
    ConformalResult
        The intervals at the positions, the scores of the held-out
        observations and the guarantee.
    """
    fit = None
    if method.sample is None:
        fit = fit_values(calibrator, calibrator.run(starts=method.starts))
    run = commol_rs.inference.split_conformal(
        calibrator.rust_problem,
        fit,
        None if method.sample is None else method.sample.rust_sample,
        engine_state.rust_observed_data(method.calibration_data),
        method.positions,
        method.score,
        method.by,
        method.level,
    )
    return _result(
        run,
        method.level,
        f"Each interval covers a new observation with probability at least "
        f"{method.level} when its score and the scores of the held-out "
        f"observations are exchangeable.",
    )


def run_rolling_conformal(
    calibrator: "Calibrator", method: RollingConformal
) -> ConformalResult:
    """
    Run rolling-origin conformal prediction.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : RollingConformal
        The settings.

    Returns
    -------
    ConformalResult
        The intervals at the positions, around the fit of every observation;
        the scores of every origin and horizon; and the guarantee.
    """
    fit = calibrator.run(starts=method.starts)
    run = commol_rs.inference.rolling_conformal(
        calibrator.rust_problem,
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        method.origins,
        method.positions,
        method.score,
        method.by,
        method.level,
        method.starts,
        calibrator.seed,
    )
    _log_failed(run)
    return _result(
        run,
        method.level,
        f"Each interval covers a new observation at its horizon with probability "
        f"at least {method.level} when the scores of that horizon are "
        f"exchangeable across origins.",
    )


def run_jackknife_plus(
    calibrator: "Calibrator", method: JackknifePlus
) -> ConformalResult:
    """
    Run jackknife+ with blocks.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : JackknifePlus
        The settings.

    Returns
    -------
    ConformalResult
        The intervals at the positions, the left-out residuals and the
        guarantee, with the CV+ bound of its blocks.
    """
    fit = calibrator.run(starts=method.starts)
    run = commol_rs.inference.jackknife_plus(
        calibrator.rust_problem,
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        rust_block_length(method.block_length),
        method.positions,
        method.score,
        method.by,
        method.level,
        method.starts,
        calibrator.seed,
    )
    _log_failed(run)
    return _result(
        run,
        method.level,
        f"Each interval covers a new observation with probability at least "
        f"{run.bound:.4g} when the blocks are exchangeable: 1 − 2α with one "
        f"observation per block, and the CV+ bound of Barber et al. (2021) with "
        f"K = {run.blocks} blocks of {run.block_length} positions otherwise.",
    )


def run_enbpi(calibrator: "Calibrator", method: EnbPI) -> ConformalResult:
    """
    Run ensemble batch prediction intervals.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : EnbPI
        The settings.

    Returns
    -------
    ConformalResult
        The intervals at the positions, the leave-one-out scores and the
        guarantee.
    """
    fit = calibrator.run(starts=method.starts)
    run = commol_rs.inference.enbpi(
        calibrator.rust_problem,
        calibrator.rust_optimizer,
        fit_values(calibrator, fit),
        method.replicates,
        rust_block_length(method.block_length),
        method.positions,
        method.score,
        method.by,
        method.level,
        method.starts,
        calibrator.seed,
    )
    _log_failed(run)
    return _result(
        run,
        method.level,
        f"Each interval covers a new observation with probability near "
        f"{method.level}, approximately, when the errors are stationary and "
        f"strongly mixing (Xu and Xie 2021); blocks of {run.block_length} "
        f"positions.",
    )


register(SplitConformal, run_split_conformal)
register(RollingConformal, run_rolling_conformal)
register(JackknifePlus, run_jackknife_plus)
register(EnbPI, run_enbpi)
