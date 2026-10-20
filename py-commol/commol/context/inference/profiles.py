"""The profile likelihood of parameters and predictions: the settings, the
thresholds and the result.

A profile holds one quantity, a parameter or a prediction, at values that
step away from the fit, and fits the other parameters again at each value
with the problem's own optimizer. The statistic of a point is the objective
of its refit less that of the fit; the interval of the quantity is where the
statistic stays below a threshold (Raue et al. 2009; Kreutz et al. 2012).
Every interval is pointwise: it holds one quantity, not several at once.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from commol.context.calibration import CalibrationResult, ObservedDataPoint
from commol.context.inference.method import UncertaintyMethod
from commol.context.inference.resampling import (
    BayesianBootstrap,
    ParametricBootstrap,
    ResidualBootstrap,
    WildBootstrap,
)

ProfileEndName = Literal["crossed", "bound", "max_steps", "unreachable"]
"""Why one side of a profile ends: it rose past the threshold, reached a
bound of the parameter, took `max_steps` points, or met a prediction that no
values within the bounds give."""

ProfileVerdictName = Literal[
    "identifiable", "practically_non_identifiable", "structurally_non_identifiable"
]
"""What the shape of a profile says about its quantity."""

type ProfileBootstrap = (
    BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap
)
"""A bootstrap that calibrates a threshold."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Threshold(_Frozen):
    """
    How far the statistic of a profile may rise above the fit inside an
    interval of level `level`.

    Build one with `chi_square`, `beale` or `bootstrap_calibrated`.

    Attributes
    ----------
    kind : {"chi_square", "beale", "bootstrap_calibrated"}
        The rule.
    level : float
        The level of each pointwise interval, strictly between 0 and 1.
    bootstrap : BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap | None
        The bootstrap of ``"bootstrap_calibrated"``; None otherwise.
    """  # noqa: E501

    kind: Literal["chi_square", "beale", "bootstrap_calibrated"]
    level: float = Field(gt=0.0, lt=1.0)
    bootstrap: ProfileBootstrap | None = None

    @model_validator(mode="after")
    def check_bootstrap(self) -> Self:
        """A bootstrap goes with ``"bootstrap_calibrated"``, and only with it."""
        if (self.kind == "bootstrap_calibrated") != (self.bootstrap is not None):
            raise ValueError(
                "A bootstrap calibrates the threshold 'bootstrap_calibrated' and no "
                "other; build thresholds with Threshold.chi_square, Threshold.beale "
                "or Threshold.bootstrap_calibrated."
            )
        return self

    @classmethod
    def chi_square(cls, level: float) -> Self:
        """
        ``½ χ²₁(level)`` on ``−log L`` (Wilks): the threshold of a problem
        with observation models.

        It assumes that the models hold and that the statistic is near its
        large-sample distribution.

        Parameters
        ----------
        level : float
            The level of each interval, strictly between 0 and 1.

        Returns
        -------
        Threshold
            The threshold.

        Raises
        ------
        ValueError
            If the level is not strictly between 0 and 1.
        """
        return cls(kind="chi_square", level=_level(level))

    @classmethod
    def beale(cls, level: float) -> Self:
        """
        Beale's threshold of a sum of squares: ``S(θ̂) · F(1, n − p; level) /
        (n − p)``, with ``n`` observations of positive weight and ``p``
        parameters.

        It assumes independent Gaussian errors of one variance, and is exact
        for a model linear in its parameters. It needs the loss ``"sse"``.

        Parameters
        ----------
        level : float
            The level of each interval, strictly between 0 and 1.

        Returns
        -------
        Threshold
            The threshold.

        Raises
        ------
        ValueError
            If the level is not strictly between 0 and 1.
        """
        return cls(kind="beale", level=_level(level))

    @classmethod
    def bootstrap_calibrated(cls, bootstrap: ProfileBootstrap, level: float) -> Self:
        """
        The threshold of each quantity calibrated by `bootstrap`
        (prepivoting, Beran 1988), for any loss.

        Each replicate of the bootstrap, rebuilt from the problem's seed as
        ``Calibrator.uncertainty(bootstrap)`` builds it, is fitted
        (``θ̂_b``) and then fitted again with the quantity held at its value
        at the fit of the original data: the rise of the replicate is
        ``PL_b(θ̂ⱼ) − L_b(θ̂_b)``. The threshold is the quantile `level` of
        the rises. It costs one fit and one refit per quantity for each
        replicate.

        Parameters
        ----------
        bootstrap : BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap
            The bootstrap, with its replicates and starts.
        level : float
            The level of each interval, strictly between 0 and 1.

        Returns
        -------
        Threshold
            The threshold.

        Raises
        ------
        ValueError
            If the level is not strictly between 0 and 1.
        """  # noqa: E501
        return cls(
            kind="bootstrap_calibrated", level=_level(level), bootstrap=bootstrap
        )


def _level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"A level is strictly between 0 and 1, got {level}.")
    return level


class PredictionSpec(_Frozen):
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
    """

    at: float = Field(ge=0, allow_inf_nan=False)
    output: str | None = Field(default=None, min_length=1)
    outputs: list[str] | None = None
    label: str = ""
    window: float | None = Field(default=None, gt=0)
    factor_id: str | None = None

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
        )


class ProfileLikelihood(UncertaintyMethod["ProfileResult"]):
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

    parameters: list[str] = Field(min_length=1)
    threshold: Threshold
    max_steps: int = Field(default=100, ge=1)
    starts: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def check_unique(self) -> Self:
        """Each parameter is profiled once."""
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError(f"A parameter is listed twice: {self.parameters}.")
        return self


class PredictionProfile(UncertaintyMethod["ProfileResult"]):
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

    specs: list[PredictionSpec] = Field(min_length=1)
    threshold: Threshold
    max_steps: int = Field(default=100, ge=1)
    starts: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def check_labels(self) -> Self:
        """Each prediction has a label of its own."""
        labels = [spec.as_observation().label for spec in self.specs]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"Each prediction needs a label of its own, got {labels}; set label."
            )
        return self


class Profile(_Frozen):
    """
    The profile of one parameter or prediction.

    Attributes
    ----------
    center : float
        The value at the fit.
    threshold : float
        The threshold of the statistic.
    values : list[float]
        The value of each point, increasing; the fit is one of them.
    statistics : list[float]
        The statistic of each point: the objective of its refit less that of
        the fit.
    parameters : list[dict[str, float]]
        Every parameter of each refit.
    multipliers : list[float | None]
        For a prediction, the multiplier of each point: minus the slope of
        the profile there.
    converged : list[bool]
        Whether each refit converged.
    ends : tuple[ProfileEndName, ProfileEndName]
        Why the side below and the side above the fit end.
    evaluations : int
        Evaluations of the objective.
    replicate_rises : list[float] | None
        With a bootstrap threshold, the rise of each replicate kept.
    """

    center: float
    threshold: float
    values: list[float]
    statistics: list[float]
    parameters: list[dict[str, float]]
    multipliers: list[float | None]
    converged: list[bool]
    ends: tuple[ProfileEndName, ProfileEndName]
    evaluations: int
    replicate_rises: list[float] | None = None


class ProfileInterval(_Frozen):
    """
    The pointwise interval of a quantity: where its profile crosses the
    threshold.

    Attributes
    ----------
    lower, upper : float | None
        The ends; None on a side that does not cross.
    """

    lower: float | None
    upper: float | None


class ProfileResult(_Frozen):
    """
    The result of a profile likelihood.

    Attributes
    ----------
    fit : CalibrationResult
        The fit every profile starts from.
    threshold : Threshold
        The threshold the profiles ran with.
    profiles : dict[str, Profile]
        The profile of each parameter or prediction, by name.
    intervals : dict[str, ProfileInterval]
        The interval of each.
    verdicts : dict[str, ProfileVerdictName | None]
        What the shape of each profile says: ``identifiable`` (it crosses the
        threshold on both sides), ``practically_non_identifiable`` (it stays
        below on a side, up to a bound or `max_steps`) or
        ``structurally_non_identifiable`` (flat within `tolerance` over the
        whole range). None when the threshold is below `tolerance`: the
        profile cannot tell a rise from numerical error.
    tolerance : float
        The numerical tolerance of the statistic.
    notes : list[str]
        What a reader should know, such as a threshold below the tolerance or
        a refit below the fit.
    failed_replicates : list[str]
        With a bootstrap threshold, each replicate that failed.
    """

    fit: CalibrationResult
    threshold: Threshold
    profiles: dict[str, Profile]
    intervals: dict[str, ProfileInterval]
    verdicts: dict[str, ProfileVerdictName | None]
    tolerance: float
    notes: list[str] = Field(default_factory=list)
    failed_replicates: list[str] = Field(default_factory=list)
