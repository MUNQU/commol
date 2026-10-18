"""The runners of the conformal methods, registered for
``Calibrator.uncertainty``.

Each runner fits the problem with ``Calibrator.run`` when its predictor is a
fit and hands the rest to Rust, which predicts, scores, refits and takes the
quantiles.
"""

from typing import TYPE_CHECKING

from commol.api.inference.registry import register
from commol.api.inference.resampling import fit_values
from commol.commol_rs import _commol_rs as commol_rs
from commol.context._model import engine_state
from commol.context.inference.conformal import (
    ConformalResult,
    PredictionIntervals,
    SeriesIntervals,
    SplitConformal,
)

if TYPE_CHECKING:
    from commol.api.calibrator import Calibrator
    from commol.commol_rs._commol_rs import ConformalRunProtocol


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


register(SplitConformal, run_split_conformal)
