"""How the observed values of a series scatter around the model: the
observation models of a calibration problem.

A problem with ``observation_models`` fits by maximum likelihood instead of a
loss. Each model takes the prediction of an observation (its sums, window and
factor) as its mean and adds how the observed value scatters around it. A
noise argument is a number, or the id of a calibration parameter of type
``"noise"``, which is calibrated with the others and never reaches the
engine.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from commol._data.ranges import RangeChecked


class _ObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NormalObservations(_ObservationModel, RangeChecked):
    """
    Normal noise: ``y ~ N(μ, sd²)``, with ``μ`` the prediction.

    For continuous values with errors of the same size everywhere. With a
    known ``sd`` the maximum-likelihood fit is the least-squares fit.

    Attributes
    ----------
    sd : float | str
        The standard deviation, above 0, or the id of a noise parameter.
    """

    kind: Literal["normal"] = "normal"
    sd: float | str = Field(description="The standard deviation.")


class LogNormalObservations(_ObservationModel, RangeChecked):
    """
    Log-normal noise: ``log y`` is normal with standard deviation ``sigma``.

    For positive values with errors that grow with the value. ``center``
    says what the prediction ``μ`` is, and has no default: ``"mean"`` makes
    ``E[y] = μ`` (``log y ~ N(log μ − σ²/2, σ²)``), ``"median"`` makes the
    median ``μ`` (``log y ~ N(log μ, σ²)``, the proportional error
    ``y = μ·e^ε``). Every observed value must be above 0.

    Attributes
    ----------
    sigma : float | str
        The standard deviation of ``log y``, above 0, or the id of a noise
        parameter.
    center : {"mean", "median"}
        What the prediction is.
    """

    kind: Literal["log_normal"] = "log_normal"
    sigma: float | str = Field(description="The standard deviation of log y.")
    center: Literal["mean", "median"] = Field(description="What the prediction is.")


class StudentTObservations(_ObservationModel, RangeChecked):
    """
    Student-t noise of center ``μ``, the prediction: normal noise with heavier
    tails, so an outlier pulls the fit less.

    Attributes
    ----------
    df : float | str
        The degrees of freedom, above 0, or the id of a noise parameter. The
        noise has a mean when ``df > 1`` and a variance when ``df > 2``.
    scale : float | str
        The scale, above 0, or the id of a noise parameter.
    """

    kind: Literal["student_t"] = "student_t"
    df: float | str = Field(description="The degrees of freedom.")
    scale: float | str = Field(description="The scale.")


class PoissonObservations(_ObservationModel):
    """
    Poisson counts of mean ``μ``, the prediction: variance equal to the mean.

    Every observed value must be a whole number, at least 0.
    """

    kind: Literal["poisson"] = "poisson"


class NegativeBinomialObservations(_ObservationModel, RangeChecked):
    """
    Negative binomial counts of mean ``μ``, the prediction, and variance
    ``μ + φμ²``: counts that scatter more than a Poisson.

    ``φ = 0`` is the Poisson, so bounds ``[0, …]`` of a noise parameter hold
    it. The dispersion ``k`` (``size`` in R) is ``1/φ``.
    Every observed value must be a whole number, at least 0.

    Attributes
    ----------
    overdispersion : float | str
        ``φ``, at least 0, or the id of a noise parameter.
    """

    kind: Literal["negative_binomial"] = "negative_binomial"
    overdispersion: float | str = Field(description="The overdispersion φ.")


class BinomialObservations(_ObservationModel, RangeChecked):
    """
    Binomial counts of ``n`` trials and mean ``μ``, the prediction: the
    probability is ``μ / n``, so the prediction is an expected count, not a
    proportion.

    ``trials`` gives ``n`` for every observation of the series; without it,
    each observation of the series gives its own (``ObservedDataPoint.trials``),
    for trials that change from one observation to the next. One source per
    series. Every observed value must be a whole number between 0 and its
    trials.

    Attributes
    ----------
    trials : int | None
        The trials of every observation, or None (default) to read the
        trials of each observation.
    """

    kind: Literal["binomial"] = "binomial"
    trials: int | None = Field(default=None, description="The trials.")


type _ObservationModel = Annotated[
    NormalObservations
    | LogNormalObservations
    | StudentTObservations
    | PoissonObservations
    | NegativeBinomialObservations
    | BinomialObservations,
    Field(discriminator="kind"),
]
"""One observation model of a series."""
