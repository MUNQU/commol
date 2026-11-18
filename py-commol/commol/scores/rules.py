"""Scores of a predictive against observations.

Every score is computed in Rust, per observation and per series: from each
draw's value of the observations (`Predictive.at_observations`); for the PIT
and the quantile residuals, from the tails of each draw at each observed value,
which the distribution functions of the observation models give exactly; for
the log score, from the density of the observation models at each draw's
prediction. Which observations to score, for example the ones held out of the
calibration, and the observation noise are arguments with no default: a band
of the model (`NoNoise`) compared with data covers less than its level.
"""

from collections.abc import Sequence

from pydantic import ConfigDict, Field

from commol._data.result_model import ResultModel
from commol._problem.problem import ObservedDataPoint
from commol.commol_rs import _commol_rs as commol_rs
from commol.observations.noise import DistributionNoise, _ObservationNoise
from commol.uncertainty.predictive import ObservationDraws, Predictive


class Score(ResultModel):
    """
    One score of a predictive at a set of observations.

    Attributes
    ----------
    per_observation : list[float]
        The score of each observation, in the order given.
    per_series : dict[str, float]
        The mean score of each series (observation label), in order of first
        appearance.
    mean : float
        The mean over every observation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    per_observation: list[float] = Field(description="Score of each observation")
    per_series: dict[str, float] = Field(description="Mean score of each series")
    mean: float = Field(description="Mean over every observation")


def _score(draws: ObservationDraws, values: list[float]) -> Score:
    return _score_of(draws.observations, values)


def _score_of(observations: Sequence[ObservedDataPoint], values: list[float]) -> Score:
    labels = [point.label for point in observations]
    per_series = dict(commol_rs.inference.by_series(labels, values))
    overall = commol_rs.inference.by_series(["all"] * len(values), values)
    return Score(
        per_observation=values,
        per_series=per_series,
        mean=overall[0][1] if overall else float("nan"),
    )


def coverage(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
    confidence_level: float,
) -> Score:
    """
    The empirical coverage of the central interval of `confidence_level`.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise | DistributionNoise
        The observation noise of the draws (`Predictive.at_observations`).
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    Score
        1 for each observation inside its interval, 0 outside; the means are
        the coverage. A calibrated interval covers about `confidence_level`.
    """
    draws = predictive.at_observations(observations, noise)
    return _score(draws, draws._draws.coverage(draws.observed, confidence_level))


def interval_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
    confidence_level: float,
) -> Score:
    """
    The interval score: ``(u − l) + (2/α)(l − y)₊ + (2/α)(y − u)₊``.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise | DistributionNoise
        The observation noise of the draws (`Predictive.at_observations`).
    confidence_level : float
        The level ``c`` of the central interval; ``α = 1 − c``.

    Returns
    -------
    Score
        Lower is better.
    """
    draws = predictive.at_observations(observations, noise)
    return _score(draws, draws._draws.interval_score(draws.observed, confidence_level))


def weighted_interval_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
    confidence_levels: Sequence[float],
) -> Score:
    """
    The weighted interval score of Bracher et al. (2021).

    ``(½|y − m| + Σₖ (αₖ/2) IS_αₖ) / (K + ½)``, with ``m`` the median and
    ``αₖ = 1 − cₖ`` for each confidence level ``cₖ``.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise | DistributionNoise
        The observation noise of the draws (`Predictive.at_observations`).
    confidence_levels : Sequence[float]
        The levels of the central intervals.

    Returns
    -------
    Score
        Lower is better.

    References
    ----------
    Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating
    epidemic forecasts in an interval format. PLOS Computational Biology,
    17(2), e1008618. https://doi.org/10.1371/journal.pcbi.1008618
    """
    draws = predictive.at_observations(observations, noise)
    return _score(
        draws,
        draws._draws.weighted_interval_score(
            draws.observed, [float(c) for c in confidence_levels]
        ),
    )


def crps(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
) -> Score:
    """
    The continuous ranked probability score of the weighted draws.

    ``Σ wᵢ|xᵢ − y| − ½ Σᵢⱼ wᵢwⱼ|xᵢ − xⱼ|``, the function the CRPS loss of a
    stochastic calibration uses.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise | DistributionNoise
        The observation noise of the draws (`Predictive.at_observations`).

    Returns
    -------
    Score
        Lower is better; in the unit of the observations.
    """
    draws = predictive.at_observations(observations, noise)
    return _score(draws, draws._draws.crps(draws.observed))


def pit(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: _ObservationNoise,
) -> Score:
    """
    The probability integral transform of each observation.

    ``F(y⁻) + v · p(y)`` of the predictive: the mixture over the draws of the
    distribution of an observation, with ``v`` uniform in ``[0, 1]``, so a
    discrete predictive gives uniform values too. With `DistributionNoise`
    the distribution of each draw is its observation model, read exactly:
    a sample of one point gives its PIT, and a model with no atoms gives
    ``F(y)`` whatever the seed. With the other noises it is the mass at the
    draw's value, so ``F`` is that of the weighted draws and ``p(y)`` the
    weight of the draws that tie with ``y``. The values ``v`` come from the
    seed of the source of the predictive (``Calibrator.seed`` or
    ``SensitivityAnalysis.seed``), so the same predictive gives the same
    values; `quantile_residuals` reads the same ``v``.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : NoNoise | ResidualNoise | DistributionNoise
        The observation noise of the predictive.

    Returns
    -------
    Score
        Values in ``[0, 1]``; calibrated predictions give uniform values.

    Raises
    ------
    ValueError
        As `Predictive.at_observations`, and if an observed value is outside
        the support of its observation model with `DistributionNoise`.
    """
    bounds = predictive.source._observation_bounds(
        predictive.sample, list(observations), noise
    )
    return _score_of(observations, bounds.pit())


def quantile_residuals(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: DistributionNoise,
) -> Score:
    """
    The randomized quantile residual of each observation (Dunn and Smyth 1996).

    ``Φ⁻¹(u)`` of the PIT ``u`` of `pit`, with the same ``v``, read from the
    nearer tail: ``−Φ⁻¹(P(Y > y) + (1 − v) · p(y))`` above ``u = ½``, so a
    value far in the upper tail has a finite residual. Standard normal for
    data the predictive did not see; in the data a predictive was fitted to,
    the residuals shrink.

    Parameters
    ----------
    predictive : Predictive
        The predictive, of a problem with ``observation_models``.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : DistributionNoise
        The observation models of the problem.

    Returns
    -------
    Score
        The quantile residual of each observation.

    Raises
    ------
    TypeError
        If `noise` is not `DistributionNoise`.
    ValueError
        If the problem has a loss, an observation names an unknown output or
        factor or a series with no observation model, an observed value is
        outside the support of its model, or the predictive is of a
        sensitivity analysis, which has no observation models.
    InfeasibleStateError
        If a draw leaves the values its model allows, as `Predictive.at_observations`.

    References
    ----------
    Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals.
    Journal of Computational and Graphical Statistics, 5(3), 236–244.
    https://doi.org/10.1080/10618600.1996.10474708
    """
    if not isinstance(noise, DistributionNoise):
        raise TypeError(
            f"Quantile residuals need the distribution functions of the "
            f"observation models, so they take DistributionNoise(); "
            f"{type(noise).__name__} has none."
        )
    bounds = predictive.source._observation_bounds(
        predictive.sample, list(observations), noise
    )
    return _score_of(observations, bounds.quantile_residuals())


def log_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    noise: DistributionNoise,
) -> Score:
    """
    The log score of each observation: the log density of its value under
    the predictive.

    The predictive density is the mixture over the draws of the observation
    models, ``log Σ_d w_d · p(y | μ_d)``, with ``μ_d`` the prediction of draw
    ``d`` and its noise parameters. It needs the densities of the models, so
    only `DistributionNoise` has one. A proper scoring rule; higher is
    better, and an observation the predictive gives no mass scores ``-inf``.

    Parameters
    ----------
    predictive : Predictive
        The predictive, of a problem with ``observation_models``.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    noise : DistributionNoise
        The observation models of the problem.

    Returns
    -------
    Score
        The log score of each observation.

    Raises
    ------
    TypeError
        If `noise` is not `DistributionNoise`.
    ValueError
        If the problem has a loss, an observation names an unknown output or
        factor or a series with no observation model, or the predictive is of
        a sensitivity analysis, which has no observation models.
    """
    if not isinstance(noise, DistributionNoise):
        raise TypeError(
            f"The log score needs the density of the observation models, so it "
            f"takes DistributionNoise(); {type(noise).__name__} has no density."
        )
    values = predictive.source._log_score(predictive.sample, list(observations))
    return _score_of(observations, values)
