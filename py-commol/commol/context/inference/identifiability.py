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
- `ProfileResult.report` (`commol.context.inference.profiles`): the shape of
  each profile of a `ProfileLikelihood`. Structural when a profile is flat,
  practical otherwise.

Each returns an `IdentifiabilityReport`: one structural and one practical
verdict per parameter, and the evidence of its section.
`IdentifiabilityReport.merge` joins the reports of several methods.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from commol.context.calibration import CalibrationResult
from commol.context.inference.distributions import Priors, coerce_priors
from commol.context.inference.method import IdentifiabilityMethod
from commol.context.inference.sample import ParameterSample

if TYPE_CHECKING:
    from commol.api.reporting import IdentifiabilityReportData

StructuralVerdictName = Literal[
    "identifiable", "in_combination", "non_identifiable", "conflicting"
]
"""What the structure of the model and the design say about a parameter."""

PracticalVerdictName = Literal["determined", "poorly_determined", "conflicting"]
"""What these data say about a parameter."""

ProfileVerdictName = Literal[
    "identifiable", "practically_non_identifiable", "structurally_non_identifiable"
]
"""What the shape of a profile says about its quantity."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StructuralVerdict(_Frozen):
    """
    The structural verdict of one parameter.

    Attributes
    ----------
    verdict : StructuralVerdictName
        ``identifiable``: the null space of the sensitivities has no
        component on it, or a profile crosses its threshold. ``in_combination``:
        the data determine it only together with other parameters.
        ``non_identifiable``: it lies in the null space, or nothing the data
        read depends on it. ``conflicting``: two reports disagree.
    reasons : list[str]
        Why, one per piece of evidence.
    combinations : list[str]
        The algebraic combinations that hold the parameter, such as
        ``"k1 · s"``, from the methods that compute them (the structural
        rank); empty for the verdict of a profile, whose reason names the
        compensating parameters.
    """

    verdict: StructuralVerdictName
    reasons: list[str] = Field(default_factory=list)
    combinations: list[str] = Field(default_factory=list)


class PracticalVerdict(_Frozen):
    """
    The practical verdict of one parameter.

    Attributes
    ----------
    verdict : PracticalVerdictName
        ``determined`` or ``poorly_determined`` by these data, or
        ``conflicting`` when two reports disagree.
    reasons : list[str]
        Why, one per piece of evidence.
    """

    verdict: PracticalVerdictName
    reasons: list[str] = Field(default_factory=list)


class ParameterVerdicts(_Frozen):
    """
    The two verdicts of one parameter; ``None`` when no evidence gives one.

    Attributes
    ----------
    structural : StructuralVerdict | None
        From the structural rank on the design of the observations.
    practical : PracticalVerdict | None
        From collinearity, a sample or the starts.
    """

    structural: StructuralVerdict | None = None
    practical: PracticalVerdict | None = None


class Combination(_Frozen):
    """
    A combination of parameters the data determine: the product of each
    parameter raised to its exponent.

    Attributes
    ----------
    exponents : dict[str, float]
        The exponent of each parameter, in problem order.
    text : str
        The product, such as ``"k1 · s"`` or ``"k1 / s^2"``; a parameter whose
        bounds allow negative values reads as its absolute value.
    """

    exponents: dict[str, float]
    text: str


class PointRank(_Frozen):
    """
    The rank of the scaled sensitivities at one point.

    Attributes
    ----------
    values : dict[str, float]
        The point.
    singular_values : list[float]
        The singular values, largest first.
    threshold : float
        The norm of the estimated errors of the sensitivities, which a
        singular value must pass to count.
    rank : int
        How many pass it.
    """

    values: dict[str, float]
    singular_values: list[float]
    threshold: float
    rank: int


class StructuralSection(_Frozen):
    """
    The evidence of `StructuralRank`.

    Attributes
    ----------
    design : str
        The design the sensitivities were evaluated on: ``"observations"`` or
        ``"dense"``.
    points : list[PointRank]
        Each feasible point.
    skipped : list[str]
        Why each infeasible point was skipped.
    generic_rank : int
        The rank at most points.
    projections : dict[str, float]
        The norm of the projection of each parameter on the null space,
        averaged over the points of generic rank.
    verdicts : dict[str, StructuralVerdictName]
        The verdict of each parameter on this design.
    combinations : list[Combination]
        The identifiable combinations.
    """

    design: Literal["observations", "dense"]
    points: list[PointRank]
    skipped: list[str]
    generic_rank: int
    projections: dict[str, float]
    verdicts: dict[str, StructuralVerdictName]
    combinations: list[Combination]


class SubsetIndex(_Frozen):
    """
    A subset of parameters and its collinearity index.

    Attributes
    ----------
    parameters : list[str]
        The parameters, in problem order.
    gamma : float
        The weighted median of the index over the points.
    share_below : float
        The share of the weight of the points where the index is below the
        threshold.
    """

    parameters: list[str]
    gamma: float
    share_below: float


class CollinearitySection(_Frozen):
    """
    The evidence of `Collinearity`.

    Attributes
    ----------
    threshold : float
        The index below which a subset can be fitted together.
    points : int
        How many points were used: one for a fit, one per feasible draw of a
        sample.
    skipped : list[str]
        Why each infeasible point was skipped.
    sensitivity : dict[str, float]
        The weighted median of Brun's sensitivity measure of each parameter.
    pair_gamma : dict[str, dict[str, float]]
        The weighted median of the index of each pair.
    best_by_size : list[SubsetIndex]
        The subset of lowest index of each size.
    admissible : list[SubsetIndex]
        The largest subsets whose index is below the threshold.
    """

    threshold: float
    points: int
    skipped: list[str]
    sensitivity: dict[str, float]
    pair_gamma: dict[str, dict[str, float]]
    best_by_size: list[SubsetIndex]
    admissible: list[SubsetIndex]


class CorrelatedPair(_Frozen):
    """
    Two parameters whose absolute correlation is above the threshold.

    Attributes
    ----------
    first, second : str
        The parameters, in problem order.
    correlation : float
        Their weighted correlation.
    """

    first: str
    second: str
    correlation: float


class SampleSection(_Frozen):
    """
    The evidence of `SampleIdentifiability`.

    Attributes
    ----------
    reference : {"bounds", "prior"}
        The distribution the contraction compares with: ``"bounds"``, the
        uniform distribution on the bounds, or ``"prior"``, the prior of each
        parameter.
    correlation_threshold : float
        The absolute correlation above which a pair is listed.
    contraction : dict[str, float]
        ``1 − Var(sample) / Var(reference)`` of each parameter: near 0 when
        the sample spreads as the reference, near 1 when it sits on a value.
    correlation : dict[str, dict[str, float]]
        The weighted correlation matrix.
    correlated_pairs : list[CorrelatedPair]
        The pairs above the threshold.
    modes : dict[str, list[float]]
        The modes of the marginal of each parameter.
    """

    reference: Literal["bounds", "prior"]
    correlation_threshold: float
    contraction: dict[str, float]
    correlation: dict[str, dict[str, float]]
    correlated_pairs: list[CorrelatedPair]
    modes: dict[str, list[float]]


class StartsSection(_Frozen):
    """
    The evidence of `StartsIdentifiability`.

    Attributes
    ----------
    parameter_tolerance : float
        The share of its bound width above which a parameter differs.
    loss_tolerance : float
        The share of the best loss within which two losses are equal.
    near_best : list[int]
        The converged starts within the loss tolerance of the best.
    spread : dict[str, float]
        The largest difference of each parameter between them, as a share of
        its bound width.
    flagged : list[str]
        The parameters whose spread is above the tolerance.
    """

    parameter_tolerance: float
    loss_tolerance: float
    near_best: list[int]
    spread: dict[str, float]
    flagged: list[str]


class ProfileSection(_Frozen):
    """
    The evidence of a `ProfileLikelihood`.

    Attributes
    ----------
    threshold : {"chi_square", "beale", "bootstrap_calibrated"}
        The rule of the threshold.
    confidence_level : float
        The level of each pointwise interval.
    tolerance : float
        The numerical tolerance of the statistic.
    verdicts : dict[str, ProfileVerdictName | None]
        The verdict of each profiled parameter; None when its threshold is
        below the tolerance.
    intervals : dict[str, tuple[float | None, float | None]]
        The interval of each profiled parameter; None on a side that does not
        cross the threshold.
    """

    threshold: Literal["chi_square", "beale", "bootstrap_calibrated"]
    confidence_level: float
    tolerance: float
    verdicts: dict[str, ProfileVerdictName | None]
    intervals: dict[str, tuple[float | None, float | None]]


def _merge_structural(
    verdicts: list[StructuralVerdict],
) -> StructuralVerdict | None:
    if not verdicts:
        return None
    names = {verdict.verdict for verdict in verdicts}
    reasons = [reason for verdict in verdicts for reason in verdict.reasons]
    combinations = list(
        dict.fromkeys(c for verdict in verdicts for c in verdict.combinations)
    )
    return StructuralVerdict(
        verdict=names.pop() if len(names) == 1 else "conflicting",
        reasons=reasons,
        combinations=combinations,
    )


def _merge_practical(verdicts: list[PracticalVerdict]) -> PracticalVerdict | None:
    if not verdicts:
        return None
    names = {verdict.verdict for verdict in verdicts}
    return PracticalVerdict(
        verdict=names.pop() if len(names) == 1 else "conflicting",
        reasons=[reason for verdict in verdicts for reason in verdict.reasons],
    )


class IdentifiabilityReport(_Frozen):
    """
    Which parameters the model and the data determine: one structural and one
    practical verdict per parameter, and the evidence of each method.

    The structural verdict comes from `StructuralRank` on the design of the
    observations, and from a flat profile; a dense design answers what
    denser data would determine, so its verdicts stay in its section. The
    practical verdict comes from `Collinearity`, `SampleIdentifiability`,
    `StartsIdentifiability` and the other profiles.

    Attributes
    ----------
    parameters : list[str]
        The parameters, in problem order.
    verdicts : dict[str, ParameterVerdicts]
        The verdicts of each parameter.
    structural, collinearity, sample, starts, profile : list
        The sections of the methods that ran, one per run.
    """

    parameters: list[str]
    verdicts: dict[str, ParameterVerdicts]
    structural: list[StructuralSection] = Field(default_factory=list)
    collinearity: list[CollinearitySection] = Field(default_factory=list)
    sample: list[SampleSection] = Field(default_factory=list)
    starts: list[StartsSection] = Field(default_factory=list)
    profile: list[ProfileSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        """Check that the verdicts hold the parameters, in order."""
        if list(self.verdicts) != self.parameters:
            raise ValueError(
                f"The verdicts must hold the parameters {self.parameters} in "
                f"order, got {list(self.verdicts)}."
            )
        return self

    def report(self) -> "IdentifiabilityReportData":
        """
        Report the identifiability of each parameter as plain data.

        Returns
        -------
        IdentifiabilityReportData
            ``parameters``: each parameter, in problem order, with its
            structural and practical verdicts; ``sections``: the evidence of
            each method that ran.
        """
        from commol.api.reporting import _identifiability_report

        return _identifiability_report(self)

    def merge(self, *others: "IdentifiabilityReport") -> "IdentifiabilityReport":
        """
        Join this report with others of the same parameters.

        The sections are kept side by side. A verdict that the reports agree
        on stays, with the reasons of each; one they disagree on is
        ``conflicting``, with every reason.

        Parameters
        ----------
        *others : IdentifiabilityReport
            The reports to join.

        Returns
        -------
        IdentifiabilityReport
            The joined report.

        Raises
        ------
        ValueError
            If a report has other parameters.
        """
        reports = [self, *others]
        for other in others:
            if other.parameters != self.parameters:
                raise ValueError(
                    f"Reports of different parameters cannot merge: "
                    f"{self.parameters} and {other.parameters}."
                )
        verdicts = {
            name: ParameterVerdicts(
                structural=_merge_structural(
                    [
                        v
                        for r in reports
                        if (v := r.verdicts[name].structural) is not None
                    ]
                ),
                practical=_merge_practical(
                    [
                        v
                        for r in reports
                        if (v := r.verdicts[name].practical) is not None
                    ]
                ),
            )
            for name in self.parameters
        }
        return IdentifiabilityReport(
            parameters=self.parameters,
            verdicts=verdicts,
            structural=[s for r in reports for s in r.structural],
            collinearity=[s for r in reports for s in r.collinearity],
            sample=[s for r in reports for s in r.sample],
            starts=[s for r in reports for s in r.starts],
            profile=[s for r in reports for s in r.profile],
        )


class StructuralRank(IdentifiabilityMethod[IdentifiabilityReport]):
    """
    Structural identifiability, local and numerical: the rank of the scaled
    sensitivities at random points inside the bounds.

    The sensitivities come from finite differences, with a step and an error
    set by the accuracy of the engine; a singular value counts when it is
    above the norm of that error. The engine must be deterministic. The
    points come from the seed of the calibrator, ``Calibrator.seed``.

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

    points: int = Field(ge=1)
    design: Literal["observations", "dense"]


class Collinearity(IdentifiabilityMethod[IdentifiabilityReport]):
    """
    Collinearity analysis after Brun et al. (2001): the sensitivity measure
    of each parameter and the collinearity index of each subset.

    The index of a subset is 1 when its sensitivities are orthogonal and grows
    without bound as they approach linear dependence. The literature puts the
    threshold between 5 and 20. The engine must be deterministic.

    Attributes
    ----------
    at : CalibrationResult | ParameterSample
        Where to evaluate: the best point of a fit, or each draw of a sample,
        summarized by the weighted median.
    max_subset_size : int
        The largest subset, at least 2 and at most the number of parameters.
        Every subset up to 6 parameters is evaluated; a greedy search adds
        parameters above.
    threshold : float
        The index below which a subset can be fitted together. No default:
        it is a modelling judgment.
    """

    at: CalibrationResult | ParameterSample
    max_subset_size: int = Field(ge=2)
    threshold: float = Field(gt=1.0)


class SampleIdentifiability(IdentifiabilityMethod[IdentifiabilityReport]):
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
    priors : Priors | dict[str, Distribution] | None
        The prior of each parameter, with ``reference="prior"`` only.
    """

    sample: ParameterSample
    reference: Literal["bounds", "prior"]
    correlation_threshold: float = Field(gt=0.0, lt=1.0)
    priors: Priors | None = None

    @model_validator(mode="before")
    @classmethod
    def read_priors(cls, data: object) -> object:
        """Read a mapping of parameter to distribution as `Priors`."""
        if isinstance(data, Mapping) and data.get("priors") is not None:
            return {**data, "priors": coerce_priors(data["priors"])}
        return data

    @model_validator(mode="after")
    def check_priors(self) -> Self:
        """The priors go with ``reference="prior"``, and only with it."""
        if self.reference == "prior" and self.priors is None:
            raise ValueError(
                "reference='prior' needs priors: the prior of each parameter."
            )
        if self.reference == "bounds" and self.priors is not None:
            raise ValueError(
                "priors are read only with reference='prior'; with "
                "reference='bounds' leave them out."
            )
        return self


class StartsIdentifiability(IdentifiabilityMethod[IdentifiabilityReport]):
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

    result: CalibrationResult
    parameter_tolerance: float = Field(ge=0.0)
    loss_tolerance: float = Field(ge=0.0)
