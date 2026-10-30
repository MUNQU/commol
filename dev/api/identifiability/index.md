# Identifiability API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.identifiability`: the identifiability methods, the report they return, its parts, and the types of its plain-dictionary report (`IdentifiabilityReport.report`). Guide: [Identifiability](https://munqu.github.io/commol/latest/guide/uncertainty/identifiability/index.md).

## StructuralRank

Bases: `IdentifiabilityMethod[IdentifiabilityReport]`, `RangeChecked`

Structural identifiability, local and numerical: the rank of the scaled sensitivities at random points inside the bounds.

The sensitivities come from finite differences, with a step and an error set by the accuracy of the engine; a singular value counts when it is above the norm of that error. The engine must be deterministic. The points come from the seed of the calibrator, `Calibrator.seed`. A noise parameter of an observation model (`parameter_type="noise"`) changes no prediction: it is left out of the rank and gets the verdict `None`.

With an engine that integrates with tolerances, such as `DifferentialEquations`, each point whose rank is below the number of parameters differentiated runs again at a hundredth of `ODESolver.rtol` and `ODESolver.atol`, no lower than 1e-14, which costs a second, slower evaluation of that point; a parameter whose verdict changes there gets the verdict `None`, with a reason. See "When the solver tolerance sets the rank" in the guide page Identifiability.

Attributes:

| Name     | Type  | Description                                                                                                                                                                                                                                           |
| -------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `points` | `int` | How many rows of a Latin hypercube over the bounds, at least 1.                                                                                                                                                                                       |
| `design` | `str` | "observations": the positions and outputs of the observations, their values ignored (can this design determine the parameters?). "dense": every observed series at every unit position up to the last observation (can denser data of these series?). |

Source code in `commol/identifiability/methods.py`

```python
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
```

## Collinearity

Bases: `IdentifiabilityMethod[IdentifiabilityReport]`, `RangeChecked`

Collinearity analysis after Brun et al. (2001): the sensitivity measure of each parameter and the collinearity index of each subset.

The index of a subset is 1 when its sensitivities are orthogonal and grows without bound as they approach linear dependence. The literature puts the threshold between 5 and 20. The engine must be deterministic. A noise parameter of an observation model (`parameter_type="noise"`) changes no prediction: it is left out, and gets no verdict.

Attributes:

| Name              | Type                | Description                                                                                                                                                                                                                |
| ----------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `at`              | \`CalibrationResult | ParameterSample\`                                                                                                                                                                                                          |
| `max_subset_size` | `int`               | The largest subset, at least 2 and at most the number of parameters that are not noise parameters, of which there must be at least 2. Every subset up to 6 parameters is evaluated; a greedy search adds parameters above. |
| `threshold`       | `float`             | The index below which a subset can be fitted together. No default: it is a modelling judgment.                                                                                                                             |

Source code in `commol/identifiability/methods.py`

```python
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
```

## SampleIdentifiability

Bases: `IdentifiabilityMethod[IdentifiabilityReport]`, `RangeChecked`

Practical identifiability from any parameter sample: how much it contracts each parameter, which parameters move together, and which marginals have several modes.

Attributes:

| Name                    | Type              | Description                                                                                                                                                                                        |
| ----------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample`                | `ParameterSample` | The sample, such as the draws of an uncertainty method.                                                                                                                                            |
| `reference`             | `{bounds, prior}` | The distribution the contraction compares with: "bounds", the uniform distribution on the bounds, or "prior", the prior of each parameter in priors, such as the priors the sample was drawn with. |
| `correlation_threshold` | `float`           | The absolute correlation above which a pair is listed and its parameters are poorly determined, strictly between 0 and 1. No default: it is a modelling judgment.                                  |
| `priors`                | \`Priors          | dict[str, distribution]                                                                                                                                                                            |

Source code in `commol/identifiability/methods.py`

```python
class SampleIdentifiability(IdentifiabilityMethod[IdentifiabilityReport], RangeChecked):
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

    @model_validator(mode="after")
    def _check_priors(self) -> Self:
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
```

## StartsIdentifiability

Bases: `IdentifiabilityMethod[IdentifiabilityReport]`, `RangeChecked`

Practical identifiability from the starts of a calibration (`Calibrator.run(starts=n)`): converged starts of equal loss that end apart flag the parameters that differ.

Attributes:

| Name                  | Type                | Description                                                   |
| --------------------- | ------------------- | ------------------------------------------------------------- |
| `result`              | `CalibrationResult` | The result, with at least two converged starts.               |
| `parameter_tolerance` | `float`             | The share of its bound width above which a parameter differs. |
| `loss_tolerance`      | `float`             | The share of the best loss within which two losses are equal. |

Source code in `commol/identifiability/methods.py`

```python
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
```

## ProfileIdentifiability

Bases: `IdentifiabilityMethod[IdentifiabilityReport]`

Identifiability from the profiles of a `ProfileLikelihood`: the verdict of the shape of each profile, with a `profile` section.

`identifiable` gives the practical verdict `determined` and `practically_non_identifiable` the practical verdict `poorly_determined`. `structurally_non_identifiable` gives the structural verdict of its `structure`, `in_combination` or `non_identifiable`, with no combination: the reason names the compensating parameters. A flat profile with no structural verdict gives the practical verdict `poorly_determined`. Every reason says that the verdict holds with the other parameters at their fitted values. A parameter with no profile, or no verdict, gets neither.

Attributes:

| Name     | Type            | Description                                                 |
| -------- | --------------- | ----------------------------------------------------------- |
| `result` | `ProfileResult` | The profiles of the parameters of this calibration problem. |

Raises:

| Type         | Description                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| `ValueError` | If the result holds prediction profiles, which have no verdict on a parameter. |

Source code in `commol/identifiability/methods.py`

```python
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
```

## IdentifiabilityReport

Bases: `_FrozenResult`

Which parameters the model and the data determine: one structural and one practical verdict per parameter, and the evidence of each method.

The structural verdict comes from `StructuralRank` on the design of the observations, and from a flat profile; a dense design answers what denser data would determine, so its verdicts stay in its section. The practical verdict comes from `Collinearity`, `SampleIdentifiability`, `StartsIdentifiability` and the other profiles.

Attributes:

| Name                                                | Type                           | Description                                        |
| --------------------------------------------------- | ------------------------------ | -------------------------------------------------- |
| `parameters`                                        | `list[str]`                    | The parameters, in problem order.                  |
| `verdicts`                                          | `dict[str, ParameterVerdicts]` | The verdicts of each parameter.                    |
| `structural, collinearity, sample, starts, profile` | `list`                         | The sections of the methods that ran, one per run. |

Source code in `commol/identifiability/results.py`

```python
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
```

### Methods:

#### report

```python
report() -> IdentifiabilityReportData
```

Report the identifiability of each parameter as plain data.

Returns:

| Type                        | Description                                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `IdentifiabilityReportData` | parameters: each parameter, in problem order, with its structural and practical verdicts; sections: the evidence of each method that ran. |

Source code in `commol/identifiability/results.py`

```python
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
```

#### merge

```python
merge(*others: IdentifiabilityReport) -> IdentifiabilityReport
```

Join this report with others of the same parameters.

The sections are kept side by side. A verdict that the reports agree on stays, with the reasons of each; one they disagree on is `conflicting`, with every reason. A structural verdict of `None` adds its reasons and leaves the verdicts of the other reports.

Parameters:

| Name      | Type                    | Description          | Default |
| --------- | ----------------------- | -------------------- | ------- |
| `*others` | `IdentifiabilityReport` | The reports to join. | `()`    |

Returns:

| Type                    | Description        |
| ----------------------- | ------------------ |
| `IdentifiabilityReport` | The joined report. |

Raises:

| Type         | Description                       |
| ------------ | --------------------------------- |
| `ValueError` | If a report has other parameters. |

Source code in `commol/identifiability/results.py`

```python
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
```

## ParameterVerdicts

Bases: `_FrozenResult`

The two verdicts of one parameter; `None` when no evidence gives one.

Attributes:

| Name         | Type                | Description |
| ------------ | ------------------- | ----------- |
| `structural` | \`StructuralVerdict | None\`      |
| `practical`  | \`PracticalVerdict  | None\`      |

Source code in `commol/identifiability/results.py`

```python
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
        the accuracy of the runs may set its rank, or the parameter is a
        noise parameter, which it leaves out; its reasons say why.
    practical : PracticalVerdict | None
        From collinearity, a sample or the starts. Collinearity gives none to
        a noise parameter.
    """

    structural: StructuralVerdict | None = None
    practical: PracticalVerdict | None = None
```

## StructuralVerdict

Bases: `_FrozenResult`

The structural verdict of one parameter.

Attributes:

| Name           | Type                    | Description                                                                                                                                                                                                         |
| -------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `verdict`      | \`StructuralVerdictName | None\`                                                                                                                                                                                                              |
| `reasons`      | `list[str]`             | Why, one per piece of evidence.                                                                                                                                                                                     |
| `combinations` | `list[str]`             | The algebraic combinations that hold the parameter, such as "k1 · s", from the methods that compute them (the structural rank); empty for the verdict of a profile, whose reason names the compensating parameters. |

Source code in `commol/identifiability/results.py`

```python
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
        the structure, may set the structural rank, or for a noise parameter,
        which the structural rank leaves out; the reasons say why.
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
```

## PracticalVerdict

Bases: `_FrozenResult`

The practical verdict of one parameter.

Attributes:

| Name      | Type                   | Description                                                                              |
| --------- | ---------------------- | ---------------------------------------------------------------------------------------- |
| `verdict` | `PracticalVerdictName` | determined or poorly_determined by these data, or conflicting when two reports disagree. |
| `reasons` | `list[str]`            | Why, one per piece of evidence.                                                          |

Source code in `commol/identifiability/results.py`

```python
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
```

## StructuralSection

Bases: `_FrozenResult`

The evidence of `StructuralRank`.

Attributes:

| Name                   | Type                               | Description                                                                                                                                                   |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `design`               | `str`                              | The design the sensitivities were evaluated on: "observations" or "dense".                                                                                    |
| `points`               | `list[PointRank]`                  | Each feasible point.                                                                                                                                          |
| `skipped`              | `list[str]`                        | Why each infeasible point was skipped.                                                                                                                        |
| `left_out`             | `dict[str, str]`                   | The parameters left out, each with the reason: a noise parameter of an observation model, which no prediction reads. Every other parameter is differentiated. |
| `generic_rank`         | `int`                              | The rank at most points, at the accuracy of the engine; at most the number of parameters differentiated.                                                      |
| `projections`          | `dict[str, float]`                 | The norm of the projection of each parameter differentiated on the null space, averaged over the points of generic rank, at the accuracy of the engine.       |
| `verdicts`             | \`dict\[str, StructuralVerdictName | None\]\`                                                                                                                                                      |
| `combinations`         | `list[Combination]`                | The identifiable combinations.                                                                                                                                |
| `accuracy`             | `Accuracy`                         | The accuracy of the runs of the engine.                                                                                                                       |
| `tighter`              | \`Accuracy                         | None\`                                                                                                                                                        |
| `tighter_generic_rank` | \`int                              | None\`                                                                                                                                                        |
| `unchecked`            | `list[str]`                        | Why each point that the tighter accuracy should check could not run there.                                                                                    |

Source code in `commol/identifiability/results.py`

```python
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
    left_out : dict[str, str]
        The parameters left out, each with the reason: a noise parameter of
        an observation model, which no prediction reads. Every other
        parameter is differentiated.
    generic_rank : int
        The rank at most points, at the accuracy of the engine; at most the
        number of parameters differentiated.
    projections : dict[str, float]
        The norm of the projection of each parameter differentiated on the
        null space, averaged over the points of generic rank, at the accuracy
        of the engine.
    verdicts : dict[str, StructuralVerdictName | None]
        The verdict of each parameter differentiated on this design; ``None``
        when the verdict changes at the tighter accuracy, or a point could
        not run there and the parameter is not identifiable.
    combinations : list[Combination]
        The identifiable combinations.
    accuracy : Accuracy
        The accuracy of the runs of the engine.
    tighter : Accuracy | None
        The accuracy a point of rank below the number of parameters
        differentiated is evaluated again at: each accuracy of the engine
        divided by 100, no lower than 1e-14. ``None`` when no setting sets the
        accuracy of the engine, or it is already the tightest.
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
    left_out: dict[str, str]
    generic_rank: int
    projections: dict[str, float]
    verdicts: dict[str, StructuralVerdictName | None]
    combinations: list[Combination]
    accuracy: Accuracy
    tighter: Accuracy | None
    tighter_generic_rank: int | None
    unchecked: list[str]
```

## PointRank

Bases: `_FrozenResult`

The rank of the scaled sensitivities at one point.

Attributes:

| Name              | Type               | Description                                                                                       |
| ----------------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| `values`          | `dict[str, float]` | The point, with a value of every parameter.                                                       |
| `accuracy`        | `Accuracy`         | The accuracy of the runs of the engine.                                                           |
| `singular_values` | `list[float]`      | The singular values, largest first, one per parameter differentiated.                             |
| `threshold`       | `float`            | The norm of the estimated errors of the sensitivities, which a singular value must pass to count. |
| `rank`            | `int`              | How many pass it.                                                                                 |
| `tighter`         | \`TighterRank      | None\`                                                                                            |

Source code in `commol/identifiability/results.py`

```python
class PointRank(_FrozenResult):
    """
    The rank of the scaled sensitivities at one point.

    Attributes
    ----------
    values : dict[str, float]
        The point, with a value of every parameter.
    accuracy : Accuracy
        The accuracy of the runs of the engine.
    singular_values : list[float]
        The singular values, largest first, one per parameter differentiated.
    threshold : float
        The norm of the estimated errors of the sensitivities, which a
        singular value must pass to count.
    rank : int
        How many pass it.
    tighter : TighterRank | None
        The rank at a tighter accuracy, when the rank is below the number of
        parameters differentiated and a setting of the engine sets its
        accuracy. ``None`` when the rank is that number, when no setting sets
        the accuracy (an engine exact up to rounding), when the accuracy is
        already the tightest (1e-14), or when the point could not run at the
        tighter accuracy (``StructuralSection.unchecked`` says why).
    """

    values: dict[str, float]
    accuracy: Accuracy
    singular_values: list[float]
    threshold: float
    rank: int
    tighter: TighterRank | None
```

## TighterRank

Bases: `_FrozenResult`

The rank of the scaled sensitivities at a point evaluated again at a tighter accuracy.

Attributes:

| Name              | Type          | Description                                                                                       |
| ----------------- | ------------- | ------------------------------------------------------------------------------------------------- |
| `accuracy`        | `Accuracy`    | The accuracy of the runs.                                                                         |
| `singular_values` | `list[float]` | The singular values, largest first, one per parameter differentiated.                             |
| `threshold`       | `float`       | The norm of the estimated errors of the sensitivities, which a singular value must pass to count. |
| `rank`            | `int`         | How many pass it.                                                                                 |

Source code in `commol/identifiability/results.py`

```python
class TighterRank(_FrozenResult):
    """
    The rank of the scaled sensitivities at a point evaluated again at a
    tighter accuracy.

    Attributes
    ----------
    accuracy : Accuracy
        The accuracy of the runs.
    singular_values : list[float]
        The singular values, largest first, one per parameter differentiated.
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
```

## Accuracy

Bases: `_FrozenResult`

The accuracy of the runs the sensitivities come from.

Attributes:

| Name       | Type    | Description                                                                                                                                      |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `relative` | `float` | The relative error a run may carry: ODESolver.rtol for DifferentialEquations, machine epsilon for an engine whose runs are exact up to rounding. |
| `absolute` | `float` | The error a run may carry on an output near 0: ODESolver.atol, or 0.                                                                             |

Source code in `commol/identifiability/results.py`

```python
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
```

## Combination

Bases: `_FrozenResult`

A combination of parameters the data determine: the product of each parameter raised to its exponent.

Attributes:

| Name        | Type               | Description                                                                                                              |
| ----------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `exponents` | `dict[str, float]` | The exponent of each parameter, in problem order.                                                                        |
| `text`      | `str`              | The product, such as "k1 · s" or "k1 / s^2"; a parameter whose bounds allow negative values reads as its absolute value. |

Source code in `commol/identifiability/results.py`

```python
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
```

## CollinearitySection

Bases: `_FrozenResult`

The evidence of `Collinearity`.

Attributes:

| Name           | Type                          | Description                                                                                                                                                   |
| -------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `threshold`    | `float`                       | The index below which a subset can be fitted together.                                                                                                        |
| `points`       | `int`                         | How many points were used: one for a fit, one per feasible draw of a sample.                                                                                  |
| `skipped`      | `list[str]`                   | Why each infeasible point was skipped.                                                                                                                        |
| `left_out`     | `dict[str, str]`              | The parameters left out, each with the reason: a noise parameter of an observation model, which no prediction reads. Every other parameter is differentiated. |
| `sensitivity`  | `dict[str, float]`            | The weighted median of Brun's sensitivity measure of each parameter differentiated.                                                                           |
| `pair_gamma`   | `dict[str, dict[str, float]]` | The weighted median of the index of each pair of parameters differentiated.                                                                                   |
| `best_by_size` | `list[SubsetIndex]`           | The subset of lowest index of each size.                                                                                                                      |
| `admissible`   | `list[SubsetIndex]`           | The largest subsets whose index is below the threshold.                                                                                                       |

Source code in `commol/identifiability/results.py`

```python
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
    left_out : dict[str, str]
        The parameters left out, each with the reason: a noise parameter of
        an observation model, which no prediction reads. Every other
        parameter is differentiated.
    sensitivity : dict[str, float]
        The weighted median of Brun's sensitivity measure of each parameter
        differentiated.
    pair_gamma : dict[str, dict[str, float]]
        The weighted median of the index of each pair of parameters
        differentiated.
    best_by_size : list[SubsetIndex]
        The subset of lowest index of each size.
    admissible : list[SubsetIndex]
        The largest subsets whose index is below the threshold.
    """

    threshold: float
    points: int
    skipped: list[str]
    left_out: dict[str, str]
    sensitivity: dict[str, float]
    pair_gamma: dict[str, dict[str, float]]
    best_by_size: list[SubsetIndex]
    admissible: list[SubsetIndex]
```

## SubsetIndex

Bases: `_FrozenResult`

A subset of parameters and its collinearity index.

Attributes:

| Name          | Type        | Description                                                                   |
| ------------- | ----------- | ----------------------------------------------------------------------------- |
| `parameters`  | `list[str]` | The parameters, in problem order.                                             |
| `gamma`       | `float`     | The weighted median of the index over the points.                             |
| `share_below` | `float`     | The share of the weight of the points where the index is below the threshold. |

Source code in `commol/identifiability/results.py`

```python
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
```

## SampleSection

Bases: `_FrozenResult`

The evidence of `SampleIdentifiability`.

Attributes:

| Name                    | Type                          | Description                                                                                                                                |
| ----------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `reference`             | `{bounds, prior}`             | The distribution the contraction compares with: "bounds", the uniform distribution on the bounds, or "prior", the prior of each parameter. |
| `correlation_threshold` | `float`                       | The absolute correlation above which a pair is listed.                                                                                     |
| `contraction`           | `dict[str, float]`            | 1 − Var(sample) / Var(reference) of each parameter: near 0 when the sample spreads as the reference, near 1 when it sits on a value.       |
| `correlation`           | `dict[str, dict[str, float]]` | The weighted correlation matrix.                                                                                                           |
| `correlated_pairs`      | `list[CorrelatedPair]`        | The pairs above the threshold.                                                                                                             |
| `modes`                 | `dict[str, list[float]]`      | The modes of the marginal of each parameter.                                                                                               |

Source code in `commol/identifiability/results.py`

```python
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
```

## CorrelatedPair

Bases: `_FrozenResult`

Two parameters whose absolute correlation is above the threshold.

Attributes:

| Name            | Type    | Description                       |
| --------------- | ------- | --------------------------------- |
| `first, second` | `str`   | The parameters, in problem order. |
| `correlation`   | `float` | Their weighted correlation.       |

Source code in `commol/identifiability/results.py`

```python
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
```

## StartsSection

Bases: `_FrozenResult`

The evidence of `StartsIdentifiability`.

Attributes:

| Name                  | Type               | Description                                                                           |
| --------------------- | ------------------ | ------------------------------------------------------------------------------------- |
| `parameter_tolerance` | `float`            | The share of its bound width above which a parameter differs.                         |
| `loss_tolerance`      | `float`            | The share of the best loss within which two losses are equal.                         |
| `near_best`           | `list[int]`        | The converged starts within the loss tolerance of the best.                           |
| `spread`              | `dict[str, float]` | The largest difference of each parameter between them, as a share of its bound width. |
| `flagged`             | `list[str]`        | The parameters whose spread is above the tolerance.                                   |

Source code in `commol/identifiability/results.py`

```python
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
```

## ProfileSection

Bases: `_FrozenResult`

The evidence of a `ProfileLikelihood`.

Attributes:

| Name               | Type                                        | Description                               |
| ------------------ | ------------------------------------------- | ----------------------------------------- |
| `threshold`        | `{chi_square, beale, bootstrap_calibrated}` | The rule of the threshold.                |
| `confidence_level` | `float`                                     | The level of each pointwise interval.     |
| `tolerance`        | `float`                                     | The numerical tolerance of the statistic. |
| `verdicts`         | \`dict\[str, ProfileVerdictName             | None\]\`                                  |
| `intervals`        | \`dict\[str, tuple\[float                   | None, float                               |

Source code in `commol/identifiability/results.py`

```python
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
```

## The types of the report

`IdentifiabilityReport.report` returns an `IdentifiabilityReportData`, a `TypedDict` that holds the others.

## IdentifiabilityReportData

Bases: `TypedDict`

The identifiability of each parameter, and the evidence of each method.

`sections` holds the sections of each method, as plain data (`IdentifiabilitySections`).

Source code in `commol/identifiability/report.py`

```python
class IdentifiabilityReportData(TypedDict):
    """
    The identifiability of each parameter, and the evidence of each method.

    ``sections`` holds the sections of each method, as plain data
    (`IdentifiabilitySections`).
    """

    parameters: dict[str, VerdictsEntry]
    sections: IdentifiabilitySections
```

## VerdictsEntry

Bases: `TypedDict`

The two verdicts of a parameter; None when no evidence gives one.

Source code in `commol/identifiability/report.py`

```python
class VerdictsEntry(TypedDict):
    """The two verdicts of a parameter; None when no evidence gives one."""

    structural: StructuralVerdictEntry | None
    practical: PracticalVerdictEntry | None
```

## StructuralVerdictEntry

Bases: `TypedDict`

The structural verdict of a parameter (None when the evidence gives none), its reasons and combinations.

Source code in `commol/identifiability/report.py`

```python
class StructuralVerdictEntry(TypedDict):
    """The structural verdict of a parameter (None when the evidence gives
    none), its reasons and combinations."""

    verdict: str | None
    reasons: list[str]
    combinations: list[str]
```

## PracticalVerdictEntry

Bases: `TypedDict`

The practical verdict of a parameter and its reasons.

Source code in `commol/identifiability/report.py`

```python
class PracticalVerdictEntry(TypedDict):
    """The practical verdict of a parameter and its reasons."""

    verdict: str
    reasons: list[str]
```

## IdentifiabilitySections

Bases: `TypedDict`

The sections of each method of an identifiability report.

Source code in `commol/identifiability/report.py`

```python
class IdentifiabilitySections(TypedDict):
    """The sections of each method of an identifiability report."""

    structural: list[StructuralSectionData]
    collinearity: list[CollinearitySectionData]
    sample: list[SampleSectionData]
    starts: list[StartsSectionData]
    profile: list[ProfileSectionData]
```

## StructuralSectionData

Bases: `TypedDict`

`StructuralSection` as plain data.

Source code in `commol/identifiability/report.py`

```python
class StructuralSectionData(TypedDict):
    """`StructuralSection` as plain data."""

    design: "StructuralDesign"
    points: list[PointRankData]
    skipped: list[str]
    left_out: dict[str, str]
    generic_rank: int
    projections: dict[str, float]
    verdicts: "dict[str, StructuralVerdictName | None]"
    combinations: list[CombinationData]
    accuracy: AccuracyData
    tighter: AccuracyData | None
    tighter_generic_rank: int | None
    unchecked: list[str]
```

## PointRankData

Bases: `TypedDict`

`PointRank` as plain data.

Source code in `commol/identifiability/report.py`

```python
class PointRankData(TypedDict):
    """`PointRank` as plain data."""

    values: dict[str, float]
    accuracy: AccuracyData
    singular_values: list[float]
    threshold: float
    rank: int
    tighter: TighterRankData | None
```

## TighterRankData

Bases: `TypedDict`

`TighterRank` as plain data.

Source code in `commol/identifiability/report.py`

```python
class TighterRankData(TypedDict):
    """`TighterRank` as plain data."""

    accuracy: AccuracyData
    singular_values: list[float]
    threshold: float
    rank: int
```

## AccuracyData

Bases: `TypedDict`

`Accuracy` as plain data.

Source code in `commol/identifiability/report.py`

```python
class AccuracyData(TypedDict):
    """`Accuracy` as plain data."""

    relative: float
    absolute: float
```

## CombinationData

Bases: `TypedDict`

`Combination` as plain data.

Source code in `commol/identifiability/report.py`

```python
class CombinationData(TypedDict):
    """`Combination` as plain data."""

    exponents: dict[str, float]
    text: str
```

## CollinearitySectionData

Bases: `TypedDict`

`CollinearitySection` as plain data.

Source code in `commol/identifiability/report.py`

```python
class CollinearitySectionData(TypedDict):
    """`CollinearitySection` as plain data."""

    threshold: float
    points: int
    skipped: list[str]
    left_out: dict[str, str]
    sensitivity: dict[str, float]
    pair_gamma: dict[str, dict[str, float]]
    best_by_size: list[SubsetIndexData]
    admissible: list[SubsetIndexData]
```

## SubsetIndexData

Bases: `TypedDict`

`SubsetIndex` as plain data.

Source code in `commol/identifiability/report.py`

```python
class SubsetIndexData(TypedDict):
    """`SubsetIndex` as plain data."""

    parameters: list[str]
    gamma: float
    share_below: float
```

## SampleSectionData

Bases: `TypedDict`

`SampleSection` as plain data.

Source code in `commol/identifiability/report.py`

```python
class SampleSectionData(TypedDict):
    """`SampleSection` as plain data."""

    reference: "SampleReference"
    correlation_threshold: float
    contraction: dict[str, float]
    correlation: dict[str, dict[str, float]]
    correlated_pairs: list[CorrelatedPairData]
    modes: dict[str, list[float]]
```

## CorrelatedPairData

Bases: `TypedDict`

`CorrelatedPair` as plain data.

Source code in `commol/identifiability/report.py`

```python
class CorrelatedPairData(TypedDict):
    """`CorrelatedPair` as plain data."""

    first: str
    second: str
    correlation: float
```

## StartsSectionData

Bases: `TypedDict`

`StartsSection` as plain data.

Source code in `commol/identifiability/report.py`

```python
class StartsSectionData(TypedDict):
    """`StartsSection` as plain data."""

    parameter_tolerance: float
    loss_tolerance: float
    near_best: list[int]
    spread: dict[str, float]
    flagged: list[str]
```

## ProfileSectionData

Bases: `TypedDict`

`ProfileSection` as plain data: each interval is a pair of ends.

Source code in `commol/identifiability/report.py`

```python
class ProfileSectionData(TypedDict):
    """`ProfileSection` as plain data: each interval is a pair of ends."""

    threshold: "ProfileThresholdName"
    confidence_level: float
    tolerance: float
    verdicts: dict[str, ProfileVerdictName | None]
    intervals: dict[str, tuple[float | None, float | None]]
```
