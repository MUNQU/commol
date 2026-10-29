"""The profile likelihood of parameters and predictions: the settings, the
thresholds and the result.

A profile holds one quantity, a parameter or a prediction, at values that
step away from the fit, and fits the other parameters again at each value
with the problem's own optimizer. The statistic of a point is the objective
of its refit less that of the fit; the interval of the quantity is where the
statistic stays below a threshold (Raue et al. 2009; Kreutz et al. 2012).
Every interval is pointwise: it holds one quantity, not several at once.
"""

from typing import TYPE_CHECKING, Self

from pydantic import ConfigDict, Field, model_validator

from commol._calibration.method import UncertaintyMethod
from commol._data.ranges import RangeChecked
from commol._problem.problem import ObservedDataPoint
from commol.profiles.runners import run_prediction_profile, run_profile_likelihood

if TYPE_CHECKING:
    pass

from commol.profiles.results import (
    ProfileResult,
    Threshold,
)


class PredictionSpec(RangeChecked):
    """
    A prediction to profile: an output, or a sum of outputs, at a position,
    read as an observation is.

    Attributes
    ----------
    at : float
        The position.
    output : str | None
        The output. Give either `output` or `outputs`.
    outputs : list[str] | None
        Outputs summed. Needs a `label`.
    label : str
        The name of the prediction in the result; defaults to `output`.
    window : float | None
        A window length: the prediction is the output at `at` less the
        output at ``at - window``.
    factor_id : str | None
        A factor parameter that multiplies the prediction.
    scale : float
        A known positive number that multiplies the prediction, with the
        factor, as the scale of an observation does (default: 1.0).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    at: float = Field()
    output: str | None = Field(default=None, min_length=1)
    outputs: list[str] | None = None
    label: str = ""
    window: float | None = Field(
        default=None,
    )
    factor_id: str | None = None
    scale: float = 1.0

    def as_observation(self) -> ObservedDataPoint:
        """
        The observation that reads this prediction; its value is not read.

        Returns
        -------
        ObservedDataPoint
            The observation.

        Raises
        ------
        ValueError
            If the outputs or the label are not valid.
        """
        return ObservedDataPoint(
            at=self.at,
            output=self.output,
            outputs=self.outputs,
            label=self.label,
            value=0.0,
            window=self.window,
            factor_id=self.factor_id,
            scale=self.scale,
        )


class ProfileLikelihood(UncertaintyMethod[ProfileResult], RangeChecked):
    """
    The profile likelihood of parameters (Raue et al. 2009).

    Each parameter is held at values that step away from the fit, in both
    directions, and the others are fitted again with the problem's own
    optimizer. The step grows while the profile rises slowly and shrinks
    when it jumps, and each point starts from a line through the two points
    before it. A side ends past the threshold, at a bound, or after
    `max_steps` points. Parameters run in parallel.

    Attributes
    ----------
    parameters : list[str]
        The parameters to profile, at least one.
    threshold : Threshold
        The threshold of the statistic.
    max_steps : int
        The most points on each side of a profile (default 100).
    starts : int
        Starts of the fit (default 1): the fit, then rows of a Latin
        hypercube over the bounds.
    """

    _runner = staticmethod(run_profile_likelihood)

    parameters: list[str] = Field(min_length=1)
    threshold: Threshold
    max_steps: int = Field(
        default=100,
    )
    starts: int = Field(
        default=1,
    )

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        """Each parameter is profiled once."""
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError(f"A parameter is listed twice: {self.parameters}.")
        return self


class PredictionProfile(UncertaintyMethod[ProfileResult], RangeChecked):
    """
    The profile likelihood of predictions (Kreutz et al. 2012).

    Each prediction is held at values that step away from its value at the
    fit, as an equality constraint met by an augmented Lagrangian whose
    inner solver is the problem's own optimizer, and the parameters are
    fitted again. The interval of a prediction is the set of values the data
    do not reject; several specs at several positions give a band.

    Attributes
    ----------
    specs : list[PredictionSpec]
        The predictions, at least one, with distinct labels.
    threshold : Threshold
        The threshold of the statistic.
    max_steps : int
        The most points on each side of a profile (default 100).
    starts : int
        Starts of the fit (default 1).
    """

    _runner = staticmethod(run_prediction_profile)

    specs: list[PredictionSpec] = Field(min_length=1)
    threshold: Threshold
    max_steps: int = Field(
        default=100,
    )
    starts: int = Field(
        default=1,
    )

    @model_validator(mode="after")
    def _check_labels(self) -> Self:
        """Each prediction has a label of its own."""
        labels = [spec.as_observation().label for spec in self.specs]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"Each prediction needs a label of its own, got {labels}; set label."
            )
        return self
