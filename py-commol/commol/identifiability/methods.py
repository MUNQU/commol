"""The identifiability methods and the report they return.

Each method answers, from one kind of evidence, which parameters the model and
the data determine:

- `StructuralRank`: the rank of the sensitivities at random points, on the
  design of the observations or a dense one. Structural: what any data from
  that design can determine.
- `Collinearity`: Brun's indices at a fit or over a sample. Practical.
- `SampleIdentifiability`: the contraction, correlations and modes of a
  sample. Practical.
- `StartsIdentifiability`: starts of equal loss that end apart. Practical.
- `ProfileIdentifiability`: the shape of each profile of a
  `ProfileLikelihood`. Structural when a profile is flat, practical
  otherwise.

Each returns an `IdentifiabilityReport`: one structural and one practical
verdict per parameter, and the evidence of its section.
`IdentifiabilityReport.merge` joins the reports of several methods.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import Field, JsonValue, model_validator

from commol._calibration.method import IdentifiabilityMethod
from commol._data.engine_json import EngineChecked
from commol._data.ranges import RangeChecked
from commol._fit.result import CalibrationResult
from commol.distributions.distributions import Priors, coerce_priors
from commol.identifiability.runners import (
    run_collinearity,
    run_profile_identifiability,
    run_sample_identifiability,
    run_starts_identifiability,
    run_structural_rank,
)
from commol.profiles.results import ProfileResult
from commol.uncertainty.sample import ParameterSample

if TYPE_CHECKING:
    pass

from commol.identifiability.results import (
    IdentifiabilityReport,
    SampleReference,
    StructuralDesign,
)


class StructuralRank(IdentifiabilityMethod[IdentifiabilityReport], RangeChecked):
    """
    Structural identifiability, local and numerical: the rank of the scaled
    sensitivities at random points inside the bounds.

    The sensitivities come from finite differences, with a step and an error
    set by the accuracy of the engine; a singular value counts when it is
    above the norm of that error. The engine must be deterministic. The
    points come from the seed of the calibrator, ``Calibrator.seed``. A noise
    parameter of an observation model (``parameter_type="noise"``) changes
    no prediction: it is left out of the rank and gets the verdict ``None``.

    With an engine that integrates with tolerances, such as
    ``DifferentialEquations``, each point whose rank is below the number of
    parameters differentiated runs again at a hundredth of ``ODESolver.rtol``
    and ``ODESolver.atol``, no lower than 1e-14, which costs a second, slower
    evaluation of that point; a parameter whose verdict changes there gets
    the verdict ``None``, with a reason. See "When the solver tolerance sets
    the rank" in the guide page Identifiability.

    Attributes
    ----------
    points : int
        How many rows of a Latin hypercube over the bounds, at least 1.
    design : str
        ``"observations"``: the positions and outputs of the observations,
        their values ignored (can this design determine the parameters?).
        ``"dense"``: every observed series at every unit position up to the
        last observation (can denser data of these series?).
    """

    _runner = staticmethod(run_structural_rank)

    points: int = Field()
    design: StructuralDesign


class Collinearity(IdentifiabilityMethod[IdentifiabilityReport], RangeChecked):
    """
    Collinearity analysis after Brun et al. (2001): the sensitivity measure
    of each parameter and the collinearity index of each subset.

    The index of a subset is 1 when its sensitivities are orthogonal and grows
    without bound as they approach linear dependence. The literature puts the
    threshold between 5 and 20. The engine must be deterministic. A noise
    parameter of an observation model (``parameter_type="noise"``) changes
    no prediction: it is left out, and gets no verdict.

    Attributes
    ----------
    at : CalibrationResult | ParameterSample
        Where to evaluate: the best point of a fit, or each draw of a sample,
        summarized by the weighted median.
    max_subset_size : int
        The largest subset, at least 2 and at most the number of parameters
        that are not noise parameters, of which there must be at least 2.
        Every subset up to 6 parameters is evaluated; a greedy search adds
        parameters above.
    threshold : float
        The index below which a subset can be fitted together. No default:
        it is a modelling judgment.
    """

    _runner = staticmethod(run_collinearity)

    _engine_handles: ClassVar[tuple[str, ...]] = ("at",)

    at: CalibrationResult | ParameterSample
    max_subset_size: int
    threshold: float = Field()


class SampleIdentifiability(
    IdentifiabilityMethod[IdentifiabilityReport], EngineChecked, RangeChecked
):
    """
    Practical identifiability from any parameter sample: how much it contracts
    each parameter, which parameters move together, and which marginals have
    several modes.

    Attributes
    ----------
    sample : ParameterSample
        The sample, such as the draws of an uncertainty method.
    reference : {"bounds", "prior"}
        The distribution the contraction compares with: ``"bounds"``, the
        uniform distribution on the bounds, or ``"prior"``, the prior of each
        parameter in `priors`, such as the priors the sample was drawn with.
    correlation_threshold : float
        The absolute correlation above which a pair is listed and its
        parameters are poorly determined, strictly between 0 and 1. No
        default: it is a modelling judgment.
    priors : Priors | dict[str, distribution] | None
        The prior of each parameter, with ``reference="prior"`` only.
    """

    _runner = staticmethod(run_sample_identifiability)

    _engine_handles: ClassVar[tuple[str, ...]] = ("sample",)

    sample: ParameterSample
    reference: SampleReference
    correlation_threshold: float
    priors: Priors | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_priors(cls, data: JsonValue) -> JsonValue:
        """Read a mapping of parameter to distribution as `Priors`."""
        if isinstance(data, Mapping) and data.get("priors") is not None:
            return {**data, "priors": coerce_priors(data["priors"])}
        return data


class StartsIdentifiability(IdentifiabilityMethod[IdentifiabilityReport], RangeChecked):
    """
    Practical identifiability from the starts of a calibration
    (``Calibrator.run(starts=n)``): converged starts of equal loss that end
    apart flag the parameters that differ.

    Attributes
    ----------
    result : CalibrationResult
        The result, with at least two converged starts.
    parameter_tolerance : float
        The share of its bound width above which a parameter differs.
    loss_tolerance : float
        The share of the best loss within which two losses are equal.
    """

    _runner = staticmethod(run_starts_identifiability)

    _engine_handles: ClassVar[tuple[str, ...]] = ("result",)

    result: CalibrationResult
    parameter_tolerance: float = Field()
    loss_tolerance: float = Field()


class ProfileIdentifiability(IdentifiabilityMethod[IdentifiabilityReport]):
    """
    Identifiability from the profiles of a `ProfileLikelihood`: the verdict
    of the shape of each profile, with a ``profile`` section.

    ``identifiable`` gives the practical verdict ``determined`` and
    ``practically_non_identifiable`` the practical verdict
    ``poorly_determined``. ``structurally_non_identifiable`` gives the
    structural verdict of its `structure`, ``in_combination`` or
    ``non_identifiable``, with no combination: the reason names the
    compensating parameters. A flat profile with no structural verdict gives
    the practical verdict ``poorly_determined``. Every reason says that the
    verdict holds with the other parameters at their fitted values. A
    parameter with no profile, or no verdict, gets neither.

    Attributes
    ----------
    result : ProfileResult
        The profiles of the parameters of this calibration problem.

    Raises
    ------
    ValueError
        If the result holds prediction profiles, which have no verdict on a
        parameter.
    """

    _runner = staticmethod(run_profile_identifiability)

    result: ProfileResult

    @model_validator(mode="after")
    def _parameter_profiles(self) -> Self:
        if self.result.quantity != "parameter":
            raise ValueError(
                "This result holds the profiles of predictions, and predictions "
                "have no verdict on a parameter; profile the parameters "
                "(ProfileLikelihood) for their identifiability."
            )
        return self
