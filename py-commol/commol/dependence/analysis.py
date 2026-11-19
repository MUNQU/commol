"""The two analyses of dependence: of any values, one per observation, and
of the residuals of a predictive."""

import logging
from collections.abc import Sequence
from typing import Literal

from commol._data.engine_json import engine_json
from commol._problem.problem import ObservedDataPoint
from commol.commol_rs import _commol_rs as commol_rs
from commol.dependence.references import AtParameters, Refitted
from commol.dependence.results import DependenceSettings, SeriesDependence
from commol.observations.noise import DistributionNoise, NoNoise, _ObservationNoise
from commol.uncertainty.predictive import Predictive

logger = logging.getLogger("commol.dependence")


def series_dependence(
    values: Sequence[float],
    observations: Sequence[ObservedDataPoint],
    max_lag: int | Literal["auto"],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> SeriesDependence:
    """
    The dependence of one value per observation, series by series.

    The values of each label, in position order, are read on the grid of the
    smallest spacing of the series, and each lag over the pairs of values
    that exist at that distance (Parzen 1963; Stoffer and Toloi 1992). For
    each series: the autocorrelation up to `max_lag`; the robust portmanteau
    ``Q̃ = Σ_k (Σ e_t e_{t−k})² / Σ e_t² e_{t−k}²`` of the values centered
    per series (Dalla, Giraitis and Phillips 2022), which holds under
    heteroscedasticity, over the ``h`` lags of 4 pairs or more with a
    product other than 0, with its p-value against ``χ²(h)``; and the
    integrated autocorrelation time ``τ`` by the initial monotone sequence
    of Geyer (1992), with its band under independence from `replicates`
    sets of the centered values with signs ±1 (a wild band). For the series
    together: their correlations at lag 0, and over the standardized series
    ``z`` the factors ``Var(Σz) / ΣVar(z)`` and ``τ`` of ``Σz``, whose product
    ``LRV(Σz) / ΣVar(z)`` is the factor by which the dependence multiplies the
    information, with its p-value against the wild sets.

    Parameters
    ----------
    values : Sequence[float]
        One finite value per observation, in the order of `observations`: a
        quantile residual of held-out data, an additive residual, any score.
    observations : Sequence[ObservedDataPoint]
        The observations the values belong to.
    max_lag : int | "auto"
        The largest lag, in steps of the grid of each series; ``"auto"`` is
        ``min(10, n / 5)``, at least 1, for a series of ``n`` values.
    replicates : int
        The wild sets of the band and of the test of the factor, at least 1.
    confidence_level : float
        The level of the band of ``τ``.
    seed : int
        The seed of the signs, such as ``Calibrator.seed``.

    Returns
    -------
    SeriesDependence
        The statistics, with one draw.

    Raises
    ------
    TypeError
        If `values` holds lists: one set of values only.
    ValueError
        If a setting is outside its range, or `values` is not one finite
        value per observation.

    References
    ----------
    Dalla, V., Giraitis, L., & Phillips, P. C. B. (2022). Robust tests for
    white noise and cross-correlation. Econometric Theory, 38(5), 913–941.

    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. Statistical
    Science, 7(4), 473–483.
    """
    if any(isinstance(value, Sequence) for value in values):
        raise TypeError(
            "series_dependence reads one set of values, one per observation; for "
            "the residuals of several draws or randomizations, use "
            "residual_dependence"
        )
    raw = commol_rs.inference.series_dependence(
        engine_json(list(observations)),
        list(values),
        max_lag,
        replicates,
        confidence_level,
        seed,
    )
    settings = DependenceSettings(
        max_lag=max_lag,
        confidence_level=confidence_level,
        seed=seed,
        replicates=replicates,
    )
    return SeriesDependence._of(raw, settings)


def residual_dependence(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
    max_lag: int | Literal["auto"],
    randomizations: int,
    reference: AtParameters | Refitted,
    confidence_level: float,
) -> SeriesDependence:
    """
    The dependence of the residuals of a predictive, series by series.

    With `DistributionNoise`, the residuals of each draw are its randomized
    quantile residuals (Dunn and Smyth 1996), standard normal and
    independent for independent data under the right model; a count model
    randomizes them, so the statistics are averaged over `randomizations`
    randomizations, common to every draw, and the result does not depend on
    one of them. With `NoNoise`, for a loss or a stochastic engine, they are
    the additive residuals ``y − ŷ``, ``ŷ`` the prediction the loss scores,
    and `randomizations` is 1. The statistics are those of
    `series_dependence`, averaged over the randomizations and, with their
    weights, over the draws (``τ`` as the ratio of the mean long-run variance
    to the mean variance).

    With one draw, the tests and the band of ``τ`` come from `reference`:
    `Refitted` holds at a fit, `AtParameters` at the parameters the data
    were drawn from. With many draws, the result gives the weighted
    statistics, and the median and the weighted share below the level of
    the p-values of the draws, with no test of its own: test a fit with
    ``Calibrator.point_sample(fit)``. Every random number comes from the seed
    of the source of the predictive, so the same predictive gives the same
    result.

    Parameters
    ----------
    predictive : Predictive
        The predictive of a `Calibrator`.
    observations : Sequence[ObservedDataPoint]
        The observations whose residuals are read: the problem's, for
        `Refitted`, or held-out data.
    noise : NoNoise | DistributionNoise
        `DistributionNoise()` for the quantile residuals of observation
        models, `NoNoise()` for the additive residuals of a loss.
    max_lag : int | "auto"
        The largest lag, as in `series_dependence`.
    randomizations : int
        The randomizations of the quantile residuals, at least 1; 1 with
        `NoNoise`.
    reference : AtParameters | Refitted
        The reference of the tests and the band of ``τ``.
    confidence_level : float
        The level of the band of ``τ``; a p-value of a draw is below the
        level at or below ``1 − confidence_level``.

    Returns
    -------
    SeriesDependence
        The statistics and their tests, with ``failed`` and ``not_converged``
        of the refits.

    Raises
    ------
    TypeError
        If `noise` is `ResidualNoise`.
    ValueError
        If a setting is outside its range, the predictive is of a
        sensitivity analysis, quantile residuals have a loss or an observed
        value outside the support of its model, additive residuals have more
        than one randomization, or `Refitted` reads many draws or
        observations other than the problem's.
    RuntimeError
        If every refit fails.
    InfeasibleStateError
        If a draw leaves the values its model allows.

    References
    ----------
    Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals.
    Journal of Computational and Graphical Statistics, 5(3), 236–244.
    """
    raw = predictive.source._residual_dependence(
        predictive.sample,
        list(observations),
        noise,
        max_lag,
        randomizations,
        engine_json(reference),
        confidence_level,
    )
    for entry in raw.failed:
        logger.warning(entry)
    settings = DependenceSettings(
        max_lag=max_lag,
        confidence_level=confidence_level,
        seed=predictive.source.seed,
        noise=noise if isinstance(noise, (NoNoise, DistributionNoise)) else None,
        randomizations=randomizations,
        reference=reference,
    )
    return SeriesDependence._of(raw, settings)
