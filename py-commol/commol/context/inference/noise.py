"""What an observation shows around the prediction of a draw: the observation
noise a predictive adds before it reduces the draws."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from commol.context.calibration import CalibrationResult
from commol.context.ranges import RangeChecked


class NoNoise(BaseModel):
    """
    No observation noise: an observation shows the prediction of each draw.

    The band of a predictive with `NoNoise` is a band of the model. It says
    where the model curve lies, not where the next observation falls, so its
    coverage of data is below its level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ResidualNoise(RangeChecked):
    """
    Observation noise made of the residuals of a fit, with no noise model.

    Each draw adds moving blocks of `block_length` positions of residuals of
    the fit, drawn with replacement across every series at once, to its
    predictions, on `residual_scale`. The band is then a prediction interval.
    It serves the series the fit has residuals of, at any position: the
    residual of a series at a drawn position is the one at the nearest
    position where the series was observed.

    Attributes
    ----------
    fit : CalibrationResult
        The fit whose residuals are added.
    block_length : int
        Positions per block, at least 1 and at most the number of distinct
        positions of the problem's observations. A bootstrap reports the
        length it used in its diagnostics.
    residual_scale : {"additive", "multiplicative"}
        ``"additive"`` adds ``r = y − ŷ``; ``"multiplicative"`` multiplies by
        ``exp(r)`` with ``r = log(y / ŷ)`` and needs positive observed and
        fitted values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult = Field(description="The fit whose residuals are added.")
    block_length: int = Field(description="Positions per block.")
    residual_scale: Literal["additive", "multiplicative"] = Field(
        description="The scale of the residuals."
    )


class DistributionNoise(BaseModel):
    """
    Observation noise of the problem's observation models.

    Each draw draws every observation from the model of its series around
    the draw's prediction, with the draw's noise parameters, so the band is a
    prediction interval of the posterior or bootstrap predictive. It needs a
    problem with ``observation_models``, and serves the series they name. A
    prediction outside the means a model allows, such as a negative count,
    draws at the nearest allowed one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


type ObservationNoise = NoNoise | ResidualNoise | DistributionNoise
