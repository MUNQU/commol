"""The local approximations around a fit: the Laplace approximation, its
settings and its result.

Each reads the curvature of the objective at the fit on the unbounded scale
of the parameters (the logit of a parameter between two bounds, a log with
one bound, the identity with none), so every draw maps back inside the
bounds.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from commol.commol_rs import _commol_rs as commol_rs
from commol.context.calibration import CalibrationResult
from commol.context.inference.method import UncertaintyMethod
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import CalibrationProblemProtocol

type CurvatureName = Literal["gauss_newton"]
"""The curvature a local approximation reads. ``"gauss_newton"``: the
expected curvature, ``Jᵀ W J`` with the Fisher information ``W`` of each
observation, and the Fisher information of each noise parameter."""


class DeltaBand(BaseModel):
    """
    The band of every output by the delta method: the prediction at the fit
    plus and minus a quantile times the standard deviation of the
    linearized model, ``J Σ Jᵀ``, with no run of the draws.

    Attributes
    ----------
    confidence_level : float
        The level of the central interval.
    positions : list[float]
        Where the values are.
    center, lower, upper : dict[str, list[float]]
        For each output, one value per position: the prediction at the fit
        and the ends of the interval.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    confidence_level: float
    positions: list[float]
    center: dict[str, list[float]] = Field(description="Prediction at the fit")
    lower: dict[str, list[float]] = Field(description="Lower end")
    upper: dict[str, list[float]] = Field(description="Upper end")


class LocalResult(BaseModel):
    """
    What every local approximation returns: a distribution on the unbounded
    scale of the parameters, centred on the fit.

    Attributes
    ----------
    fit : CalibrationResult
        The fit the approximation is centred on: ``Calibrator.run()``.
    covariance : list[list[float]]
        The covariance of the distribution on the unbounded scale of the
        parameters, in the order of the problem's parameters.
    sample : ParameterSample
        Draws of the distribution, mapped back to the parameters, with equal
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

    _problem: "CalibrationProblemProtocol | None" = PrivateAttr(default=None)

    def attach(self, problem: "CalibrationProblemProtocol") -> None:
        """
        Keep the calibration problem the result was made from, which
        `delta_band` differentiates.

        Parameters
        ----------
        problem : CalibrationProblemProtocol
            The problem of the calibrator (``Calibrator.rust_problem``).
        """
        self._problem = problem

    def degrees_of_freedom(self) -> float | None:
        """
        The degrees of freedom of the distribution.

        Returns
        -------
        float | None
            Those of a Student t, or None for a normal.
        """
        return None

    def delta_band(
        self, positions: Sequence[float], confidence_level: float
    ) -> DeltaBand:
        """
        The band of every output by the delta method.

        The model is linearized at the fit: the variance of an output is
        ``J Σ Jᵀ``, with ``J`` its derivatives on the unbounded scale. It
        runs the model a few times per parameter, not once per draw.

        Parameters
        ----------
        positions : Sequence[float]
            Positions to report at.
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        DeltaBand
            The prediction at the fit and the ends of the interval of each
            output at each position.

        Raises
        ------
        ValueError
            If the result has no problem (it was read from JSON), the level
            is not strictly between 0 and 1, or the engine rejects a
            position.
        """
        if self._problem is None:
            raise ValueError(
                "delta_band needs the calibration problem of the run; a result "
                "read from JSON has none."
            )
        positions = [float(position) for position in positions]
        outputs, center, lower, upper = commol_rs.inference.delta_band(
            self._problem,
            [self.fit.best_parameters[name] for name in self.sample.names],
            self.covariance,
            self.degrees_of_freedom(),
            positions,
            confidence_level,
        )
        return DeltaBand(
            confidence_level=confidence_level,
            positions=positions,
            center=dict(zip(outputs, center, strict=True)),
            lower=dict(zip(outputs, lower, strict=True)),
            upper=dict(zip(outputs, upper, strict=True)),
        )


class LaplaceResult(LocalResult):
    """
    The result of a Laplace approximation: the normal ``N(θ̂, Σ)`` on the
    unbounded scale, ``Σ = H⁻¹``.

    Attributes
    ----------
    fit : CalibrationResult
        The fit the approximation is centred on: ``Calibrator.run()``.
    covariance : list[list[float]]
        ``Σ = H⁻¹`` on the unbounded scale of the parameters.
    sample : ParameterSample
        Draws of the normal, mapped back to the parameters, with equal
        weights.
    notes : list[str]
        The entries of the curvature whose estimated error is not small
        against their scale.
    """


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
