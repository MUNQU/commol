"""The references of a dependence analysis of residuals: the law of its
statistics under independent observations, at the parameters or with refits."""

from typing import Literal

from pydantic import ConfigDict, Field

from commol._data.ranges import RangeChecked


class AtParameters(RangeChecked):
    """
    A reference at the parameters of each draw, with no refit.

    With `DistributionNoise`, `replicates` data sets drawn from the
    observation models at the predictions of the draw give the law of the
    integrated statistics: the p-values of ``Q̃`` and of the information
    factor, and the band of ``τ``. No model runs again. With `NoNoise`,
    ``Q̃`` is read against ``χ²(h)`` and the band comes from `replicates`
    sets of the residuals with signs ±1. A fit leaves its residuals
    correlated with each other, the more so the more parameters it has, so
    with the data of the fit this reference rejects more often than its
    level (8 % at 5 % with 3 parameters and 24 to 26 % with 10, at 35
    positions); `Refitted` holds there. It is the reference of held-out data.

    Attributes
    ----------
    kind : "at_parameters"
        The reference, which the engine reads.
    replicates : int
        The data sets of the reference, at least 1.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["at_parameters"] = Field(
        default="at_parameters", description="The reference."
    )
    replicates: int = Field(description="The data sets of the reference.")


class Refitted(RangeChecked):
    """
    A reference of data sets drawn at the fit and fitted again.

    Each of `replicates` data sets is drawn from the observation models at
    the fit (`DistributionNoise`), or is the fit plus its residuals with
    signs ±1 (`NoNoise`), and is fitted again by the optimizer of the
    problem from `starts` starts. The statistics of the residuals of the
    refits are the law of the statistics of the fit, so the test holds at a
    fit whatever its number of parameters: the p-value of ``Q̃`` and of the
    information factor is ``(1 + #{Q_b ≥ Q}) / (B + 1)``. It needs the sample
    of one point and the observations of the problem; the refits run in
    parallel and log a record of each.

    Attributes
    ----------
    kind : "refitted"
        The reference, which the engine reads.
    replicates : int
        The refits, at least 1.
    starts : int
        Starts of each refit (default 1): the fit, then rows of a Latin
        hypercube over the bounds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["refitted"] = Field(default="refitted", description="The reference.")
    replicates: int = Field(description="The refits.")
    starts: int = Field(default=1, description="Starts per refit.")
