"""Per-series observation normalization shared across the probabilistic pipeline.

The factors computed here must stay consistent with the Rust-side
``compute_observation_normalization`` in ``crates/commol-calibration`` so that the
optimization loss and the fit-gated ensemble selection measure error in the same
normalized space.
"""

from collections.abc import Sequence

import numpy as np

from commol.commol_rs import _commol_rs as commol_rs
from commol.context.calibration import ObservedDataPoint
from commol.context.constants import LossFunction


def series_normalization_factors(
    observed_data: list[ObservedDataPoint],
    enabled: bool,
) -> dict[str, float]:
    """Return a ``{series: factor}`` map of per-series normalization factors.

    Each factor is ``1 / rms`` of the observed values of its series (grouped by
    ``compartment`` label), so a residual equal to a series' typical magnitude
    contributes comparably regardless of the series' scale. Series whose observed
    values are all (near) zero get a factor of ``1.0``.

    When ``enabled`` is ``False`` every factor is ``1.0``, so callers can multiply
    unconditionally and leave the loss unchanged.
    """
    series = {observation.label for observation in observed_data}
    if not enabled:
        return {name: 1.0 for name in series}

    sum_squares: dict[str, float] = {name: 0.0 for name in series}
    counts: dict[str, int] = {name: 0 for name in series}
    for observation in observed_data:
        sum_squares[observation.label] += observation.value**2
        counts[observation.label] += 1

    factors: dict[str, float] = {}
    for name in series:
        count = counts[name]
        rms = (sum_squares[name] / count) ** 0.5 if count > 0 else 0.0
        factors[name] = 1.0 / rms if rms > np.finfo(float).eps else 1.0
    return factors


def central_fit_loss(
    residuals: Sequence[float],
    weights: Sequence[float],
    normalization: Sequence[float],
    loss_function: str,
) -> float:
    """Aggregate ensemble-median residuals with the optimizer's loss function.

    Each residual is scaled by its per-series normalization factor, then the
    engine's ``LossConfig`` applies the observation weights, as it does for
    the optimizer.

    Raises
    ------
    ValueError
        If ``loss_function`` is not a ``LossFunction``.
    """
    loss_configs = {
        LossFunction.SSE: commol_rs.calibration.LossConfig.sse,
        LossFunction.RMSE: commol_rs.calibration.LossConfig.rmse,
        LossFunction.MAE: commol_rs.calibration.LossConfig.mae,
    }
    if loss_function not in loss_configs:
        raise ValueError(
            f"Unsupported loss function for central loss: '{loss_function}'"
        )
    scaled = [
        factor * residual
        for residual, factor in zip(residuals, normalization, strict=True)
    ]
    return loss_configs[LossFunction(loss_function)]().aggregate(scaled, list(weights))
