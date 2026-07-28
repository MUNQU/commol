"""Per-series observation normalization shared across the probabilistic pipeline.

The factors computed here must stay consistent with the Rust-side
``compute_observation_normalization`` in ``crates/commol-calibration`` so that the
optimization loss and the fit-gated ensemble selection measure error in the same
normalized space.
"""

from collections.abc import Sequence

import numpy as np

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
    series = {observation.compartment for observation in observed_data}
    if not enabled:
        return {name: 1.0 for name in series}

    sum_squares: dict[str, float] = {name: 0.0 for name in series}
    counts: dict[str, int] = {name: 0 for name in series}
    for observation in observed_data:
        sum_squares[observation.compartment] += observation.value**2
        counts[observation.compartment] += 1

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

    This mirrors the Rust ``CentralLossMetric`` (and the ``LossConfig`` formulas
    in ``calibration_problem.rs``) so the reported ``central_loss`` — and the
    fit-gate it is compared against — measure error the same way the members
    were fit. Each residual is first scaled by its per-series normalization
    factor; the sum-of-squares family then additionally weights residuals, while
    RMSE and MAE ignore weights, exactly as the optimizer does.
    """
    if not residuals:
        return 0.0

    scaled = [factor * residual for residual, factor in zip(residuals, normalization)]
    if loss_function in (LossFunction.SSE, LossFunction.WEIGHTED_SSE):
        return sum((weight * value) ** 2 for value, weight in zip(scaled, weights))
    if loss_function == LossFunction.RMSE:
        return (sum(value * value for value in scaled) / len(scaled)) ** 0.5
    if loss_function == LossFunction.MAE:
        return sum(abs(value) for value in scaled) / len(scaled)
    raise ValueError(f"Unsupported loss function for central loss: '{loss_function}'")
