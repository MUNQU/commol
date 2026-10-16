"""Scores of a predictive against observations.

Every score is computed in Rust from each draw's value of the observations
(`Predictive.at_observations`), per observation and per series. Which
observations to score, for example the ones held out of the calibration, is
an argument with no default.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from commol.api.inference.predictive import ObservationDraws, Predictive
from commol.commol_rs import _commol_rs as commol_rs
from commol.context.calibration import ObservedDataPoint


class Score(BaseModel):
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
    labels = [point.label for point in draws.observations]
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
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    Score
        1 for each observation inside its interval, 0 outside; the means are
        the coverage. A calibrated interval covers about `confidence_level`.
    """
    draws = predictive.at_observations(observations)
    return _score(draws, draws.rust_draws.coverage(draws.observed, confidence_level))


def interval_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
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
    confidence_level : float
        The level ``c`` of the central interval; ``α = 1 − c``.

    Returns
    -------
    Score
        Lower is better.
    """
    draws = predictive.at_observations(observations)
    return _score(
        draws, draws.rust_draws.interval_score(draws.observed, confidence_level)
    )


def weighted_interval_score(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
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
    draws = predictive.at_observations(observations)
    return _score(
        draws,
        draws.rust_draws.weighted_interval_score(
            draws.observed, [float(c) for c in confidence_levels]
        ),
    )


def crps(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
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

    Returns
    -------
    Score
        Lower is better; in the unit of the observations.
    """
    draws = predictive.at_observations(observations)
    return _score(draws, draws.rust_draws.crps(draws.observed))


def pit(
    predictive: Predictive,
    observations: Sequence[ObservedDataPoint],
    seed: int,
) -> Score:
    """
    The probability integral transform of each observation.

    ``F(y)`` of the weighted draws, randomized between ``F(y⁻)`` and ``F(y)``
    where draws tie with ``y``, so a discrete predictive gives uniform values
    too.

    Parameters
    ----------
    predictive : Predictive
        The predictive.
    observations : Sequence[ObservedDataPoint]
        The observations to score.
    seed : int
        Seed of the randomization of ties.

    Returns
    -------
    Score
        Values in ``[0, 1]``; calibrated predictions give uniform values.
    """
    draws = predictive.at_observations(observations)
    return _score(draws, draws.rust_draws.pit(draws.observed, seed))
