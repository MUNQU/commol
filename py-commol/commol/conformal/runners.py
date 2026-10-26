"""The runners of the conformal methods, registered for
``Calibrator.uncertainty``.

Each runner sends its method whole to the engine, with the handle of its
sample when its predictor is a sample; the engine fits the problem when the
method needs a fit, predicts, scores, refits and takes the quantiles.
"""

import logging
from typing import TYPE_CHECKING

from commol._fit.starts import failed_starts
from commol._calibration.registry import register
from commol.resampling.runners import block_length_estimates, log_failed
from commol.commol_rs import _commol_rs as commol_rs
from commol._data.engine_json import engine_handles, engine_json
from commol.conformal.methods import (
    AdaptiveConformal,
    ConformalResult,
    EnbPI,
    JackknifePlus,
    PredictionIntervals,
    RollingConformal,
    SeriesIntervals,
    SplitConformal,
)

if TYPE_CHECKING:
    from commol._calibration.calibrator import Calibrator
    from commol.commol_rs._commol_rs import ConformalRunProtocol, StartsProtocol

logger = logging.getLogger("commol.conformal")


def _result(
    starts: "StartsProtocol | None",
    run: "ConformalRunProtocol",
    confidence_level: float,
    guarantee: str,
) -> ConformalResult:
    if starts is not None:
        failed_starts(starts, logger)
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
            confidence_level=confidence_level,
            series={
                label: SeriesIntervals(**values) for label, values in series.items()
            },
            unbounded=unbounded,
        ),
        scores=scores,
        guarantee=guarantee,
        failed=log_failed(run.failed, logger),
        not_converged=run.not_converged,
        block_length=run.block_length,
        block_length_estimates=block_length_estimates(run.block_length_estimates),
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
    starts, run = commol_rs.inference.split_conformal(
        calibrator._rust_problem, engine_json(method), **engine_handles(method)
    )
    return _result(
        starts,
        run,
        method.confidence_level,
        f"Each interval covers a new observation with probability at least "
        f"{method.confidence_level} when its score and the scores of the held-out "
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
    starts, run = commol_rs.inference.rolling_conformal(
        calibrator._rust_problem, engine_json(method)
    )
    return _result(
        starts,
        run,
        method.confidence_level,
        f"Each interval covers a new observation at its horizon with probability "
        f"at least {method.confidence_level} when the scores of that horizon are "
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
    starts, run = commol_rs.inference.jackknife_plus(
        calibrator._rust_problem, engine_json(method)
    )
    return _result(
        starts,
        run,
        method.confidence_level,
        f"Each interval covers a new observation with probability at least "
        f"{run.bound:.4g} when the blocks are exchangeable: the CV+ bound of "
        f"Barber et al. (2021) with K = {run.blocks} blocks of "
        f"{run.block_length} positions, which is 1 − 2α with one observation "
        f"per block.",
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
    starts, run = commol_rs.inference.enbpi(
        calibrator._rust_problem, engine_json(method)
    )
    return _result(
        starts,
        run,
        method.confidence_level,
        f"Each interval covers a new observation with probability near "
        f"{method.confidence_level}, approximately, when the errors are stationary and "
        f"strongly mixing (Xu and Xie 2021); blocks of {run.block_length} "
        f"positions.",
    )


def run_adaptive_conformal(
    calibrator: "Calibrator", method: AdaptiveConformal
) -> ConformalResult:
    """
    Run adaptive conformal inference over the stream.

    Parameters
    ----------
    calibrator : Calibrator
        The calibration problem.
    method : AdaptiveConformal
        The settings.

    Returns
    -------
    ConformalResult
        The interval given at each point of the stream before it was seen,
        the scores of the stream and the guarantee.
    """
    starts, run = commol_rs.inference.adaptive_conformal(
        calibrator._rust_problem, engine_json(method), **engine_handles(method)
    )
    return _result(
        starts,
        run,
        method.confidence_level,
        f"The long-run share of the stream outside its intervals tends to "
        f"{1 - method.confidence_level:.4g}, with no assumption on the data (Gibbs and "
        f"Candès 2021); it says nothing about one interval.",
    )


register(SplitConformal, run_split_conformal)
register(RollingConformal, run_rolling_conformal)
register(JackknifePlus, run_jackknife_plus)
register(EnbPI, run_enbpi)
register(AdaptiveConformal, run_adaptive_conformal)
