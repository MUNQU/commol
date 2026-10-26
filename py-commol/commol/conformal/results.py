"""The results of the methods of `commol.conformal`."""

from pydantic import ConfigDict, Field

from commol._data.result_model import ResultModel


class SeriesIntervals(ResultModel):
    """
    The prediction intervals of one series.

    Attributes
    ----------
    positions : list[float]
        Where the intervals are.
    prediction : list[float]
        The point prediction at each position.
    lower : list[float]
        The lower end of each interval; ``-inf`` when unbounded.
    upper : list[float]
        The upper end of each interval; ``inf`` when unbounded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions: list[float]
    prediction: list[float]
    lower: list[float]
    upper: list[float]


class PredictionIntervals(ResultModel):
    """
    Prediction intervals for observations: where new observations of each
    series fall, at a confidence level.

    Attributes
    ----------
    confidence_level : float
        The level of the intervals, ``1 − α``.
    series : dict[str, SeriesIntervals]
        The intervals of each series, by label.
    unbounded : dict[str, list[float]]
        The positions of each series whose interval is unbounded, because
        its quantile had fewer than ``1/α − 1`` scores. A series with none
        is not a key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_level: float
    series: dict[str, SeriesIntervals]
    unbounded: dict[str, list[float]]


class ConformalResult(ResultModel):
    """
    The result of a conformal method.

    Attributes
    ----------
    intervals : PredictionIntervals
        The intervals at the positions the method was given.
    scores : dict[str, list[float]]
        The scores of each series, by label, in the order the method took
        them.
    guarantee : str
        The coverage the method guarantees, and what the guarantee assumes.
    failed : list[str]
        Each refit that failed, which gives no score: ``"Origin <index>
        failed: <reason>"`` for `RollingConformal`, ``"Block <index> failed:
        <reason>"`` for `JackknifePlus`, ``"Replicate <index> failed:
        <reason>"`` for `EnbPI`; also logged as a warning. Empty for a method
        that does not refit.
    not_converged : int
        Refits kept although their optimizer did not converge; 0 for a
        method that does not refit.
    block_length : int | None
        With `JackknifePlus` and `EnbPI`, the positions per block they ran
        with; None for a method without blocks.
    block_length_estimates : dict[str, float] | None
        With ``block_length="auto"``, the estimate of each series, by label;
        otherwise None.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intervals: PredictionIntervals
    scores: dict[str, list[float]]
    guarantee: str
    failed: list[str] = Field(description="Each refit that failed.")
    not_converged: int = Field(description="Refits that did not converge.")
    block_length: int | None = Field(description="The block length used.")
    block_length_estimates: dict[str, float] | None = Field(
        description="The estimate of each series with the automatic length."
    )
