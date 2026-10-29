"""The results of the methods of `commol.identifiability`."""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from commol._data.result_model import ResultModel
from commol.identifiability.report import (
    IdentifiabilityReportData,
    identifiability_report,
)
from commol.profiles.results import ProfileVerdictName

StructuralVerdictName = Literal[
    "identifiable", "in_combination", "non_identifiable", "conflicting"
]
"""What the structure of the model and the design say about a parameter."""


PracticalVerdictName = Literal["determined", "poorly_determined", "conflicting"]
"""What these data say about a parameter."""


StructuralDesign = Literal["observations", "dense"]
"""The positions where a structural rank reads the sensitivities."""


SampleReference = Literal["bounds", "prior"]
"""What the contraction of a sample is measured against."""


ProfileThresholdName = Literal["chi_square", "beale", "bootstrap_calibrated"]
"""The threshold of a profile."""


class _FrozenResult(ResultModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StructuralVerdict(_FrozenResult):
    """
    The structural verdict of one parameter.

    Attributes
    ----------
    verdict : StructuralVerdictName | None
        ``identifiable``: the null space of the sensitivities has no
        component on it, or a profile crosses its threshold. ``in_combination``:
        the data determine it only together with other parameters.
        ``non_identifiable``: it lies in the null space, or nothing the data
        read depends on it. ``conflicting``: two reports disagree. ``None``:
        the evidence gives no verdict, as when the accuracy of the runs, not
        the structure, may set the structural rank; the reasons say why.
    reasons : list[str]
        Why, one per piece of evidence.
    combinations : list[str]
        The algebraic combinations that hold the parameter, such as
        ``"k1 · s"``, from the methods that compute them (the structural
        rank); empty for the verdict of a profile, whose reason names the
        compensating parameters.
    """

    verdict: StructuralVerdictName | None
    reasons: list[str] = Field(default_factory=list)
    combinations: list[str] = Field(default_factory=list)


class PracticalVerdict(_FrozenResult):
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


class ParameterVerdicts(_FrozenResult):
    """
    The two verdicts of one parameter; ``None`` when no evidence gives one.

    Attributes
    ----------
    structural : StructuralVerdict | None
        From the structural rank on the design of the observations, and from
        a flat profile. ``None`` when no method that ran speaks to the
        structure of the parameter. A `StructuralVerdict` whose ``verdict``
        is ``None`` is evidence with no verdict: the structural rank ran, but
        the accuracy of the runs may set its rank; its reasons say why.
    practical : PracticalVerdict | None
        From collinearity, a sample or the starts.
    """

    structural: StructuralVerdict | None = None
    practical: PracticalVerdict | None = None


class Combination(_FrozenResult):
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


class Accuracy(_FrozenResult):
    """
    The accuracy of the runs the sensitivities come from.

    Attributes
    ----------
    relative : float
        The relative error a run may carry: ``ODESolver.rtol`` for
        ``DifferentialEquations``, machine epsilon for an engine whose runs
        are exact up to rounding.
    absolute : float
        The error a run may carry on an output near 0: ``ODESolver.atol``, or
        0.
    """

    relative: float
    absolute: float


class TighterRank(_FrozenResult):
    """
    The rank of the scaled sensitivities at a point evaluated again at a
    tighter accuracy.

    Attributes
    ----------
    accuracy : Accuracy
        The accuracy of the runs.
    singular_values : list[float]
        The singular values, largest first.
    threshold : float
        The norm of the estimated errors of the sensitivities, which a
        singular value must pass to count.
    rank : int
        How many pass it.
    """

    accuracy: Accuracy
    singular_values: list[float]
    threshold: float
    rank: int


class PointRank(_FrozenResult):
    """
    The rank of the scaled sensitivities at one point.

    Attributes
    ----------
    values : dict[str, float]
        The point.
    accuracy : Accuracy
        The accuracy of the runs of the engine.
    singular_values : list[float]
        The singular values, largest first.
    threshold : float
        The norm of the estimated errors of the sensitivities, which a
        singular value must pass to count.
    rank : int
        How many pass it.
    tighter : TighterRank | None
        The rank at a tighter accuracy, when the rank is below the number of
        parameters and a setting of the engine sets its accuracy. ``None``
        when the rank is that number, when no setting sets the accuracy (an
        engine exact up to rounding), when the accuracy is already the
        tightest (1e-14), or when the point could not run at the tighter
        accuracy (``StructuralSection.unchecked`` says why).
    """

    values: dict[str, float]
    accuracy: Accuracy
    singular_values: list[float]
    threshold: float
    rank: int
    tighter: TighterRank | None


class StructuralSection(_FrozenResult):
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
        The rank at most points, at the accuracy of the engine.
    projections : dict[str, float]
        The norm of the projection of each parameter on the null space,
        averaged over the points of generic rank, at the accuracy of the
        engine.
    verdicts : dict[str, StructuralVerdictName | None]
        The verdict of each parameter on this design; ``None`` when the
        verdict changes at the tighter accuracy, or a point could not run
        there and the parameter is not identifiable.
    combinations : list[Combination]
        The identifiable combinations.
    accuracy : Accuracy
        The accuracy of the runs of the engine.
    tighter : Accuracy | None
        The accuracy a point of rank below the number of parameters is
        evaluated again at: each accuracy of the engine divided by 100, no
        lower than 1e-14. ``None`` when no setting sets the accuracy of the
        engine, or it is already the tightest.
    tighter_generic_rank : int | None
        The rank at most points with each point evaluated again at the
        tighter accuracy in place of its rank at the accuracy of the engine;
        ``None`` when no point was evaluated again.
    unchecked : list[str]
        Why each point that the tighter accuracy should check could not run
        there.
    """

    design: StructuralDesign
    points: list[PointRank]
    skipped: list[str]
    generic_rank: int
    projections: dict[str, float]
    verdicts: dict[str, StructuralVerdictName | None]
    combinations: list[Combination]
    accuracy: Accuracy
    tighter: Accuracy | None
    tighter_generic_rank: int | None
    unchecked: list[str]


class SubsetIndex(_FrozenResult):
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


class CollinearitySection(_FrozenResult):
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


class CorrelatedPair(_FrozenResult):
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


class SampleSection(_FrozenResult):
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

    reference: SampleReference
    correlation_threshold: float
    contraction: dict[str, float]
    correlation: dict[str, dict[str, float]]
    correlated_pairs: list[CorrelatedPair]
    modes: dict[str, list[float]]


class StartsSection(_FrozenResult):
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


class ProfileSection(_FrozenResult):
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

    threshold: ProfileThresholdName
    confidence_level: float
    tolerance: float
    verdicts: dict[str, ProfileVerdictName | None]
    intervals: dict[str, tuple[float | None, float | None]]


def _merge_structural(
    verdicts: list[StructuralVerdict],
) -> StructuralVerdict | None:
    if not verdicts:
        return None
    names = {verdict.verdict for verdict in verdicts if verdict.verdict is not None}
    reasons = [reason for verdict in verdicts for reason in verdict.reasons]
    combinations = list(
        dict.fromkeys(c for verdict in verdicts for c in verdict.combinations)
    )
    return StructuralVerdict(
        verdict=(
            None if not names else names.pop() if len(names) == 1 else "conflicting"
        ),
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


class IdentifiabilityReport(_FrozenResult):
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
    def _check_parameters(self) -> Self:
        """Check that the verdicts hold the parameters, in order."""
        if list(self.verdicts) != self.parameters:
            raise ValueError(
                f"The verdicts must hold the parameters {self.parameters} in "
                f"order, got {list(self.verdicts)}."
            )
        return self

    def report(self) -> IdentifiabilityReportData:
        """
        Report the identifiability of each parameter as plain data.

        Returns
        -------
        IdentifiabilityReportData
            ``parameters``: each parameter, in problem order, with its
            structural and practical verdicts; ``sections``: the evidence of
            each method that ran.
        """
        return identifiability_report(self)

    def merge(self, *others: "IdentifiabilityReport") -> "IdentifiabilityReport":
        """
        Join this report with others of the same parameters.

        The sections are kept side by side. A verdict that the reports agree
        on stays, with the reasons of each; one they disagree on is
        ``conflicting``, with every reason. A structural verdict of ``None``
        adds its reasons and leaves the verdicts of the other reports.

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
