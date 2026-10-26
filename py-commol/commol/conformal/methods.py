"""The conformal prediction methods: split, rolling, jackknife+, EnbPI and
adaptive conformal, their settings and their result.

Each gives intervals for observations, with a coverage guarantee that needs
no noise model, no prior and no correct model. A nonconformity score turns a
predicted and an observed value into a score; each method chooses which
predictions and which held-out values feed it, and a quantile of the scores
sets the width of every interval.
"""

from collections.abc import Mapping
from typing import ClassVar, Literal, Self

from pydantic import Field, JsonValue, model_validator

from commol._calibration.method import UncertaintyMethod
from commol._data.engine_json import engine_json
from commol._data.ranges import RangeChecked
from commol._problem.problem import ObservedDataPoint
from commol.commol_rs import _commol_rs as commol_rs
from commol.conformal.results import (
    ConformalResult,
)
from commol.conformal.runners import (
    run_adaptive_conformal,
    run_enbpi,
    run_jackknife_plus,
    run_rolling_conformal,
    run_split_conformal,
)
from commol.uncertainty.sample import ParameterSample


class ConformalMethod(UncertaintyMethod[ConformalResult]):
    """
    What every conformal method takes: a score, the pooling of its
    quantiles, and a confidence level, none of them with a default.

    Attributes
    ----------
    score : {"absolute", "normalized", "spread"}
        The nonconformity score. ``"normalized"`` needs ``by="all"``: per
        series its factor is one constant, so its intervals would equal those
        of ``"absolute"``.
    by : {"series", "all"}
        Whether each series gets its own quantile, or all series share one.
    confidence_level : float
        The level of the intervals, strictly between 0 and 1.
    """

    score: Literal["absolute", "normalized", "spread"] = Field(
        description="The nonconformity score."
    )
    by: Literal["series", "all"] = Field(
        description="Which scores each quantile reads."
    )
    confidence_level: float = Field(description="The level of the intervals.")

    @model_validator(mode="before")
    @classmethod
    def _check_score_and_by(cls, data: JsonValue) -> JsonValue:
        """Check, in Rust, that `score` and `by` are given and fit together."""
        if isinstance(data, Mapping):
            given = {
                key: data[key] for key in ("score", "by") if data.get(key) is not None
            }
            commol_rs.inference.check_conformal_settings(engine_json(given))
        return data


def _no_spread_without_sample(method: "SplitConformal | AdaptiveConformal") -> None:
    if method.score == "spread" and method.sample is None:
        raise ValueError(
            'score="spread" divides by the spread of the predictions, and a fit has '
            "none: give a sample, whose predictive gives the prediction and its spread"
        )


class SplitConformal(ConformalMethod, RangeChecked):
    """
    Split conformal prediction.

    It fits the problem's observations and scores `calibration_data`,
    observations held out of the fit. When the scores of the held-out and of
    new observations are exchangeable, each interval covers a new
    observation with probability at least `confidence_level`.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

    Attributes
    ----------
    calibration_data : list[ObservedDataPoint]
        Held-out observations; each of their series gets intervals.
    positions : list[float]
        Where the intervals are, for each series of `calibration_data`.
    sample : ParameterSample | None
        None: the prediction is the fit. A sample: the prediction is the
        weighted median of its predictive, and its weighted standard
        deviation is the spread of ``score="spread"``.
    starts : int
        Starts of the fit (default 1).
    """

    _runner = staticmethod(run_split_conformal)

    _engine_handles: ClassVar[tuple[str, ...]] = ("sample",)

    calibration_data: list[ObservedDataPoint] = Field(
        min_length=1, description="Held-out observations."
    )
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    sample: ParameterSample | None = Field(
        default=None, description="The sample whose predictive predicts."
    )
    starts: int = Field(default=1, description="Starts of the fit.")

    @model_validator(mode="after")
    def _check_spread(self) -> Self:
        """Reject ``score="spread"`` without a sample."""
        _no_spread_without_sample(self)
        return self


class RollingConformal(ConformalMethod, RangeChecked):
    """
    Rolling-origin conformal prediction: intervals ahead of the data, one
    quantile per horizon.

    Each origin fits the observations at or before it and scores the later
    ones. Horizon ``k`` is the ``k``-th observation of a series after the
    origin: a count, not a distance on the axis.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

    Attributes
    ----------
    origins : list[float]
        The positions of the origins.
    positions : list[float]
        Where the intervals are, all after the data; the ``k``-th position
        is horizon ``k``.
    starts : int
        Starts of each fit (default 1).
    """

    _runner = staticmethod(run_rolling_conformal)

    origins: list[float] = Field(min_length=1, description="The origins.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, description="Starts of each fit.")

    @model_validator(mode="after")
    def _check_spread(self) -> Self:
        """Reject ``score="spread"``: one fit per origin has no spread."""
        if self.score == "spread":
            raise ValueError(
                "RollingConformal fits once per origin, so its predictions have no "
                'spread for score="spread"; use "absolute" or "normalized"'
            )
        return self


class JackknifePlus(ConformalMethod, RangeChecked):
    """
    Jackknife+ (Barber et al. 2021) with blocks: one refit per block left
    out, and intervals from the left-out residuals and predictions.

    With exchangeable blocks it covers at least ``1 − 2α`` when each block
    holds one observation, and the CV+ bound with ``K`` blocks otherwise.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; it must leave at least two blocks.
    positions : list[float]
        Where the intervals are, for each series of the problem.
    starts : int
        Starts of the fit and of each refit (default 1).
    """

    _runner = staticmethod(run_jackknife_plus)

    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, description="Starts per fit.")


class EnbPI(ConformalMethod, RangeChecked):
    """
    Ensemble batch prediction intervals (Xu and Xie 2021).

    Each replicate draws blocks with replacement and fits. The replicates
    that left out the block of an observation give its leave-one-out
    prediction and score; their mean over every observation is the point
    prediction.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

    Attributes
    ----------
    replicates : int
        Number of replicates, at least 2.
    block_length : int | "auto"
        Positions per block.
    positions : list[float]
        Where the intervals are, for each series of the problem.
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    _runner = staticmethod(run_enbpi)

    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    positions: list[float] = Field(min_length=1, description="Where the intervals are.")
    starts: int = Field(default=1, description="Starts per fit.")


class AdaptiveConformal(ConformalMethod, RangeChecked):
    """
    Adaptive conformal inference (Gibbs and Candès 2021).

    It fits the problem's observations and walks `stream` in order of
    position. The interval of each point is the quantile, at the working
    level ``1 − α_t``, of the scores of the earlier points of the stream;
    then the working level moves by ``α_{t+1} = α_t + γ (α − err_t)``. The
    long-run share of misses tends to ``α``, with no assumption on the data.

    It takes `score`, `by` and `confidence_level`, as every `ConformalMethod`.

    Attributes
    ----------
    step : float
        The step ``γ`` of the working level, positive.
    stream : list[ObservedDataPoint]
        The observations that arrive after the data.
    sample : ParameterSample | None
        As in `SplitConformal`.
    starts : int
        Starts of the fit (default 1).
    """

    _runner = staticmethod(run_adaptive_conformal)

    step: float = Field(description="The step of the working level.")
    stream: list[ObservedDataPoint] = Field(
        min_length=1, description="The observations that arrive."
    )
    _engine_handles: ClassVar[tuple[str, ...]] = ("sample",)

    sample: ParameterSample | None = Field(
        default=None, description="The sample whose predictive predicts."
    )
    starts: int = Field(default=1, description="Starts of the fit.")

    @model_validator(mode="after")
    def _check_spread(self) -> Self:
        """Reject ``score="spread"`` without a sample."""
        _no_spread_without_sample(self)
        return self
