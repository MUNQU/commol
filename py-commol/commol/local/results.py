"""The results of the methods of `commol.local`."""

from collections.abc import Sequence

from pydantic import ConfigDict, Field

from commol._calibration.method import CalibrationPort
from commol._data.result_model import ResultModel
from commol._fit.result import CalibrationResult
from commol.commol_rs import _commol_rs as commol_rs
from commol.uncertainty.predictive import by_output
from commol.uncertainty.sample import ParameterSample


class DeltaBand(ResultModel):
    """
    The band of every output by the delta method (`LocalResult.delta_band`):
    the prediction at the fit
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


class LocalResult(ResultModel):
    """
    What every local approximation returns: a distribution on the unbounded
    scale of the parameters, centred on the fit.

    Attributes
    ----------
    fit : CalibrationResult
        The fit the approximation is centred on: ``Calibrator.run(starts)``.
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
        self,
        calibrator: CalibrationPort,
        positions: Sequence[float],
        confidence_level: float,
    ) -> DeltaBand:
        """
        The band of every output of this approximation by the delta method.

        The model is linearized at the fit: the variance of an output is
        ``J Σ Jᵀ``, with ``J`` its derivatives on the unbounded scale and
        ``Σ`` the covariance of this result. The band is the prediction at
        the fit plus and minus the quantile of a normal, or of the Student t
        of a `SandwichResult`, times its standard deviation. It runs the
        model a few times per parameter, not once per draw, and reads only
        the fields of this result, so a result read back from JSON works.

        Parameters
        ----------
        calibrator : Calibrator
            The calibrator of the problem this approximation is of.
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
            If the parameters of this result are not those of the problem,
            the level is not strictly between 0 and 1, the engine is
            stochastic, or the engine rejects a position.
        RuntimeError
            If the derivatives fail at the fit.
        """
        points = [float(position) for position in positions]
        outputs, center, lower, upper = commol_rs.inference.delta_band(
            calibrator._engine_problem,
            self.sample.names,
            list(self.fit.best_parameters.items()),
            self.covariance,
            self.degrees_of_freedom(),
            points,
            confidence_level,
        )
        return DeltaBand(
            confidence_level=confidence_level,
            positions=points,
            center=by_output(outputs, center),
            lower=by_output(outputs, lower),
            upper=by_output(outputs, upper),
        )


class LaplaceResult(LocalResult):
    """
    The result of a Laplace approximation: the normal ``N(θ̂, Σ)`` on the
    unbounded scale, ``Σ = H⁻¹``.

    Attributes
    ----------
    fit : CalibrationResult
        The fit the approximation is centred on: ``Calibrator.run(starts)``.
    covariance : list[list[float]]
        ``Σ = H⁻¹`` on the unbounded scale of the parameters.
    sample : ParameterSample
        Draws of the normal, mapped back to the parameters, with equal
        weights.
    notes : list[str]
        The entries of the curvature whose estimated error is not small
        against their scale.
    """


class SandwichResult(LocalResult):
    """
    The result of a sandwich: a Student t on the unbounded scale of the
    parameters, centred on the fit, of covariance ``H⁻¹ B H⁻¹`` and
    ``blocks − 1`` degrees of freedom.

    Attributes
    ----------
    fit : CalibrationResult
        The fit the sandwich is centred on: ``Calibrator.run(starts)``.
    covariance : list[list[float]]
        ``H⁻¹ B H⁻¹`` on the unbounded scale of the parameters.
    sample : ParameterSample
        Draws of the Student t, mapped back to the parameters, with equal
        weights.
    notes : list[str]
        The entries of the curvature whose estimated error is not small
        against their scale.
    blocks : int
        The number of blocks ``G``; intervals read a Student t with
        ``G − 1`` degrees of freedom.
    block_length : int
        The positions per block.
    block_length_estimates : dict[str, float] | None
        With ``block_length="auto"``, the estimate of each series, by label;
        otherwise None.
    """

    blocks: int
    block_length: int
    block_length_estimates: dict[str, float] | None

    def degrees_of_freedom(self) -> float | None:
        """
        The degrees of freedom of the Student t.

        Returns
        -------
        float | None
            ``blocks − 1``.
        """
        return float(self.blocks - 1)


class MixtureComponent(ResultModel):
    """
    One component of a Laplace mixture: the Laplace approximation at one
    optimum of the starts.

    Attributes
    ----------
    center : dict[str, float]
        The optimum, by parameter.
    weight : float
        Its weight in the mixture: its Laplace evidence over the sum.
    covariance : list[list[float]]
        Its covariance on the unbounded scale of the parameters.
    log_evidence : float
        The log of its Laplace evidence,
        ``log p(θ̂) + log |dθ/du| + (d/2) log 2π + ½ log |Σ|``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    center: dict[str, float]
    weight: float
    covariance: list[list[float]]
    log_evidence: float


class ImportanceResult(ResultModel):
    """
    The result of a Laplace mixture: draws of the mixture weighted by the
    target over the mixture, smoothed by Pareto-smoothed importance
    sampling.

    Attributes
    ----------
    fit : CalibrationResult
        The multi-start fit whose ends made the components:
        ``Calibrator.run(starts)``.
    sample : ParameterSample
        The draws with their smoothed weights.
    components : list[MixtureComponent]
        The components, in order of their objective.
    k_hat : float
        The Pareto shape of the tail of the weights. Up to 0.5 the weights
        have a finite variance; above 0.7 the estimate is unreliable.
    effective_size : float
        Kish's effective size of the smoothed weights.
    reliable : bool
        Whether ``k_hat`` is at most 0.7. An unreliable sample is also
        logged as a warning.
    notes : list[str]
        Ends that gave no component, and an unreliable sample, in words.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    sample: ParameterSample
    components: list[MixtureComponent]
    k_hat: float
    effective_size: float
    reliable: bool
    notes: list[str]
