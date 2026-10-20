"""The local approximations around a fit: the Laplace approximation, its
settings and its result.

Each reads the curvature of the objective at the fit on the unbounded scale
of the parameters (the logit of a parameter between two bounds, a log with
one bound, the identity with none), so every draw maps back inside the
bounds.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from commol.context.calibration import CalibrationResult
from commol.context.inference.method import UncertaintyMethod
from commol.context.inference.sample import ParameterSample

type CurvatureName = Literal["gauss_newton"]
"""The curvature a local approximation reads. ``"gauss_newton"``: the
expected curvature, ``Jᵀ W J`` with the Fisher information ``W`` of each
observation, and the Fisher information of each noise parameter."""


class LaplaceResult(BaseModel):
    """
    The result of a Laplace approximation.

    Attributes
    ----------
    fit : CalibrationResult
        The fit the approximation is centred on: ``Calibrator.run()``.
    covariance : list[list[float]]
        The covariance ``Σ = H⁻¹`` of the normal on the unbounded scale of
        the parameters, in the order of the problem's parameters.
    sample : ParameterSample
        Draws of the normal, mapped back to the parameters, with equal
        weights.
    notes : list[str]
        The entries of the curvature whose estimated error is not small
        against their scale; each is also logged as a warning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    covariance: list[list[float]]
    sample: ParameterSample
    notes: list[str]


class Laplace(UncertaintyMethod[LaplaceResult]):
    """
    The Laplace approximation of the problem's observation models: the normal
    ``N(θ̂, H⁻¹)`` on the unbounded scale of the parameters, centred on the
    fit, with ``H`` the curvature of ``−log L`` there.

    It needs ``observation_models``: a loss has no noise scale, so its
    curvature is no inverse variance (use `Sandwich`). A fit on a bound of
    its parameter is refused: on the unbounded scale its variance is
    infinite there.

    Attributes
    ----------
    draws : int
        Number of draws of the normal, at least 1.
    curvature : "gauss_newton"
        The curvature (default ``"gauss_newton"``).
    """

    draws: int = Field(ge=1, description="Number of draws.")
    curvature: CurvatureName = Field(
        default="gauss_newton", description="The curvature."
    )
