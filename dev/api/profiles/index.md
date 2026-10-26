# Profiles API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.profiles`: the profile likelihood of parameters and predictions, and its result. Guide: [Profile Likelihood](https://munqu.github.io/commol/latest/guide/uncertainty/profiles/index.md).

## ProfileLikelihood

Bases: `UncertaintyMethod[ProfileResult]`, `RangeChecked`

The profile likelihood of parameters (Raue et al. 2009).

Each parameter is held at values that step away from the fit, in both directions, and the others are fitted again with the problem's own optimizer. The step grows while the profile rises slowly and shrinks when it jumps, and each point starts from a line through the two points before it. A side ends past the threshold, at a bound, or after `max_steps` points. Parameters run in parallel.

Attributes:

| Name         | Type        | Description                                                                             |
| ------------ | ----------- | --------------------------------------------------------------------------------------- |
| `parameters` | `list[str]` | The parameters to profile, at least one.                                                |
| `threshold`  | `Threshold` | The threshold of the statistic.                                                         |
| `max_steps`  | `int`       | The most points on each side of a profile (default 100).                                |
| `starts`     | `int`       | Starts of the fit (default 1): the fit, then rows of a Latin hypercube over the bounds. |

Source code in `commol/profiles/methods.py`

```python
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
```

## PredictionProfile

Bases: `UncertaintyMethod[ProfileResult]`, `RangeChecked`

The profile likelihood of predictions (Kreutz et al. 2012).

Each prediction is held at values that step away from its value at the fit, as an equality constraint met by an augmented Lagrangian whose inner solver is the problem's own optimizer, and the parameters are fitted again. The interval of a prediction is the set of values the data do not reject; several specs at several positions give a band.

Attributes:

| Name        | Type                   | Description                                              |
| ----------- | ---------------------- | -------------------------------------------------------- |
| `specs`     | `list[PredictionSpec]` | The predictions, at least one, with distinct labels.     |
| `threshold` | `Threshold`            | The threshold of the statistic.                          |
| `max_steps` | `int`                  | The most points on each side of a profile (default 100). |
| `starts`    | `int`                  | Starts of the fit (default 1).                           |

Source code in `commol/profiles/methods.py`

```python
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
```

## PredictionSpec

Bases: `RangeChecked`

A prediction to profile: an output, or a sum of outputs, at a position, read as an observation is.

Attributes:

| Name        | Type        | Description                                                   |
| ----------- | ----------- | ------------------------------------------------------------- |
| `at`        | `float`     | The position.                                                 |
| `output`    | \`str       | None\`                                                        |
| `outputs`   | \`list[str] | None\`                                                        |
| `label`     | `str`       | The name of the prediction in the result; defaults to output. |
| `window`    | \`float     | None\`                                                        |
| `factor_id` | \`str       | None\`                                                        |

Source code in `commol/profiles/methods.py`

```python
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
```

### Methods:

#### as_observation

```python
as_observation() -> ObservedDataPoint
```

The observation that reads this prediction; its value is not read.

Returns:

| Type                | Description      |
| ------------------- | ---------------- |
| `ObservedDataPoint` | The observation. |

Raises:

| Type         | Description                                |
| ------------ | ------------------------------------------ |
| `ValueError` | If the outputs or the label are not valid. |

Source code in `commol/profiles/methods.py`

```python
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
```

## Threshold

Bases: `RangeChecked`

How far the statistic of a profile may rise above the fit inside an interval of `confidence_level`.

Build one with `chi_square`, `beale` or `bootstrap_calibrated`.

Attributes:

| Name               | Type                                        | Description                                                     |
| ------------------ | ------------------------------------------- | --------------------------------------------------------------- |
| `kind`             | `{chi_square, beale, bootstrap_calibrated}` | The rule.                                                       |
| `confidence_level` | `float`                                     | The level of each pointwise interval, strictly between 0 and 1. |
| `bootstrap`        | \`BayesianBootstrap                         | ParametricBootstrap                                             |

Source code in `commol/profiles/results.py`

```python
class Threshold(RangeChecked):
    """
    How far the statistic of a profile may rise above the fit inside an
    interval of `confidence_level`.

    Build one with `chi_square`, `beale` or `bootstrap_calibrated`.

    Attributes
    ----------
    kind : {"chi_square", "beale", "bootstrap_calibrated"}
        The rule.
    confidence_level : float
        The level of each pointwise interval, strictly between 0 and 1.
    bootstrap : BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap | None
        The bootstrap of ``"bootstrap_calibrated"``; None otherwise.
    """  # noqa: E501

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["chi_square", "beale", "bootstrap_calibrated"]
    confidence_level: float
    bootstrap: _ProfileBootstrap | None = None

    @model_validator(mode="after")
    def _check_bootstrap(self) -> Self:
        """A bootstrap goes with ``"bootstrap_calibrated"``, and only with it."""
        if (self.kind == "bootstrap_calibrated") != (self.bootstrap is not None):
            raise ValueError(
                "A bootstrap calibrates the threshold 'bootstrap_calibrated' and no "
                "other; build thresholds with Threshold.chi_square, Threshold.beale "
                "or Threshold.bootstrap_calibrated."
            )
        return self

    @classmethod
    def chi_square(cls, confidence_level: float) -> Self:
        """
        ``½ χ²₁(confidence_level)`` on ``−log L`` (Wilks): the threshold of a
        problem with observation models.

        It assumes that the models hold and that the statistic is near its
        large-sample distribution.

        Parameters
        ----------
        confidence_level : float
            The level of each interval, strictly between 0 and 1.

        Returns
        -------
        Threshold
            The threshold.

        Raises
        ------
        ValueError
            If ``confidence_level`` is not strictly between 0 and 1.
        """
        return cls(kind="chi_square", confidence_level=confidence_level)

    @classmethod
    def beale(cls, confidence_level: float) -> Self:
        """
        Beale's threshold of a sum of squares:
        ``S(θ̂) · F(1, n − p; confidence_level) / (n − p)``, with ``n``
        observations of positive weight and ``p`` parameters.

        It assumes independent Gaussian errors of one variance, and is exact
        for a model linear in its parameters. It needs the loss ``"sse"``.

        Parameters
        ----------
        confidence_level : float
            The level of each interval, strictly between 0 and 1.

        Returns
        -------
        Threshold
            The threshold.

        Raises
        ------
        ValueError
            If ``confidence_level`` is not strictly between 0 and 1.
        """
        return cls(kind="beale", confidence_level=confidence_level)

    @classmethod
    def bootstrap_calibrated(
        cls, bootstrap: _ProfileBootstrap, confidence_level: float
    ) -> Self:
        """
        The threshold of each quantity calibrated by `bootstrap`
        (prepivoting, Beran 1988), for any loss.

        Each replicate of the bootstrap, rebuilt from the problem's seed as
        ``Calibrator.uncertainty(bootstrap)`` builds it, is fitted
        (``θ̂_b``) and then fitted again with the quantity held at its value
        at the fit of the original data: the rise of the replicate is
        ``PL_b(θ̂ⱼ) − L_b(θ̂_b)``. The threshold is the quantile
        `confidence_level` of the rises. It costs one fit and one refit per
        quantity for each replicate.

        Parameters
        ----------
        bootstrap : BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap
            The bootstrap, with its replicates and starts.
        confidence_level : float
            The level of each interval, strictly between 0 and 1.

        Returns
        -------
        Threshold
            The threshold.

        Raises
        ------
        ValueError
            If ``confidence_level`` is not strictly between 0 and 1.
        """  # noqa: E501
        return cls(
            kind="bootstrap_calibrated",
            confidence_level=confidence_level,
            bootstrap=bootstrap,
        )
```

### Methods:

#### chi_square

```python
chi_square(confidence_level: float) -> Self
```

`½ χ²₁(confidence_level)` on `−log L` (Wilks): the threshold of a problem with observation models.

It assumes that the models hold and that the statistic is near its large-sample distribution.

Parameters:

| Name               | Type    | Description                                           | Default    |
| ------------------ | ------- | ----------------------------------------------------- | ---------- |
| `confidence_level` | `float` | The level of each interval, strictly between 0 and 1. | *required* |

Returns:

| Type        | Description    |
| ----------- | -------------- |
| `Threshold` | The threshold. |

Raises:

| Type         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `ValueError` | If confidence_level is not strictly between 0 and 1. |

Source code in `commol/profiles/results.py`

```python
@classmethod
def chi_square(cls, confidence_level: float) -> Self:
    """
    ``½ χ²₁(confidence_level)`` on ``−log L`` (Wilks): the threshold of a
    problem with observation models.

    It assumes that the models hold and that the statistic is near its
    large-sample distribution.

    Parameters
    ----------
    confidence_level : float
        The level of each interval, strictly between 0 and 1.

    Returns
    -------
    Threshold
        The threshold.

    Raises
    ------
    ValueError
        If ``confidence_level`` is not strictly between 0 and 1.
    """
    return cls(kind="chi_square", confidence_level=confidence_level)
```

#### beale

```python
beale(confidence_level: float) -> Self
```

Beale's threshold of a sum of squares: `S(θ̂) · F(1, n − p; confidence_level) / (n − p)`, with `n` observations of positive weight and `p` parameters.

It assumes independent Gaussian errors of one variance, and is exact for a model linear in its parameters. It needs the loss `"sse"`.

Parameters:

| Name               | Type    | Description                                           | Default    |
| ------------------ | ------- | ----------------------------------------------------- | ---------- |
| `confidence_level` | `float` | The level of each interval, strictly between 0 and 1. | *required* |

Returns:

| Type        | Description    |
| ----------- | -------------- |
| `Threshold` | The threshold. |

Raises:

| Type         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `ValueError` | If confidence_level is not strictly between 0 and 1. |

Source code in `commol/profiles/results.py`

```python
@classmethod
def beale(cls, confidence_level: float) -> Self:
    """
    Beale's threshold of a sum of squares:
    ``S(θ̂) · F(1, n − p; confidence_level) / (n − p)``, with ``n``
    observations of positive weight and ``p`` parameters.

    It assumes independent Gaussian errors of one variance, and is exact
    for a model linear in its parameters. It needs the loss ``"sse"``.

    Parameters
    ----------
    confidence_level : float
        The level of each interval, strictly between 0 and 1.

    Returns
    -------
    Threshold
        The threshold.

    Raises
    ------
    ValueError
        If ``confidence_level`` is not strictly between 0 and 1.
    """
    return cls(kind="beale", confidence_level=confidence_level)
```

#### bootstrap_calibrated

```python
bootstrap_calibrated(bootstrap: _ProfileBootstrap, confidence_level: float) -> Self
```

The threshold of each quantity calibrated by `bootstrap` (prepivoting, Beran 1988), for any loss.

Each replicate of the bootstrap, rebuilt from the problem's seed as `Calibrator.uncertainty(bootstrap)` builds it, is fitted (`θ̂_b`) and then fitted again with the quantity held at its value at the fit of the original data: the rise of the replicate is `PL_b(θ̂ⱼ) − L_b(θ̂_b)`. The threshold is the quantile `confidence_level` of the rises. It costs one fit and one refit per quantity for each replicate.

Parameters:

| Name               | Type                | Description                                           | Default           |
| ------------------ | ------------------- | ----------------------------------------------------- | ----------------- |
| `bootstrap`        | \`BayesianBootstrap | ParametricBootstrap                                   | ResidualBootstrap |
| `confidence_level` | `float`             | The level of each interval, strictly between 0 and 1. | *required*        |

Returns:

| Type        | Description    |
| ----------- | -------------- |
| `Threshold` | The threshold. |

Raises:

| Type         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `ValueError` | If confidence_level is not strictly between 0 and 1. |

Source code in `commol/profiles/results.py`

```python
@classmethod
def bootstrap_calibrated(
    cls, bootstrap: _ProfileBootstrap, confidence_level: float
) -> Self:
    """
    The threshold of each quantity calibrated by `bootstrap`
    (prepivoting, Beran 1988), for any loss.

    Each replicate of the bootstrap, rebuilt from the problem's seed as
    ``Calibrator.uncertainty(bootstrap)`` builds it, is fitted
    (``θ̂_b``) and then fitted again with the quantity held at its value
    at the fit of the original data: the rise of the replicate is
    ``PL_b(θ̂ⱼ) − L_b(θ̂_b)``. The threshold is the quantile
    `confidence_level` of the rises. It costs one fit and one refit per
    quantity for each replicate.

    Parameters
    ----------
    bootstrap : BayesianBootstrap | ParametricBootstrap | ResidualBootstrap | WildBootstrap
        The bootstrap, with its replicates and starts.
    confidence_level : float
        The level of each interval, strictly between 0 and 1.

    Returns
    -------
    Threshold
        The threshold.

    Raises
    ------
    ValueError
        If ``confidence_level`` is not strictly between 0 and 1.
    """  # noqa: E501
    return cls(
        kind="bootstrap_calibrated",
        confidence_level=confidence_level,
        bootstrap=bootstrap,
    )
```

## ProfileResult

Bases: `_FrozenResult`

The result of a profile likelihood.

Attributes:

| Name               | Type                            | Description                                                                                                                                             |
| ------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit`              | `CalibrationResult`             | The fit every profile starts from.                                                                                                                      |
| `quantity`         | `{parameter, prediction}`       | What the profiles hold fixed: parameters (ProfileLikelihood) or predictions (PredictionProfile).                                                        |
| `threshold`        | `Threshold`                     | The threshold the profiles ran with.                                                                                                                    |
| `profiles`         | `dict[str, Profile]`            | The profile of each parameter or prediction, by name.                                                                                                   |
| `intervals`        | `dict[str, ProfileInterval]`    | The interval of each.                                                                                                                                   |
| `verdicts`         | \`dict\[str, ProfileVerdictName | None\]\`                                                                                                                                                |
| `structure`        | `dict[str, ProfileStructure]`   | For each profile of a parameter that does not cross on both sides, what resetting the other parameters says.                                            |
| `tolerance`        | `float`                         | The numerical tolerance of the statistic: max(τ, ν), the tolerance of the optimizer and the noise floor of the objective at the fit.                    |
| `evaluation_floor` | \`float                         | None\`                                                                                                                                                  |
| `notes`            | `list[str]`                     | What a reader should know, such as a threshold below the tolerance or a refit below the fit.                                                            |
| `failed`           | `list[str]`                     | With a bootstrap threshold, each replicate that failed: "Replicate <index> failed: <reason>", also logged as a warning. Empty with any other threshold. |
| `not_converged`    | `int`                           | With a bootstrap threshold, the replicates kept although their optimizer did not converge; 0 with any other threshold.                                  |

Source code in `commol/profiles/results.py`

```python
class ProfileResult(_FrozenResult):
    """
    The result of a profile likelihood.

    Attributes
    ----------
    fit : CalibrationResult
        The fit every profile starts from.
    quantity : {"parameter", "prediction"}
        What the profiles hold fixed: parameters (`ProfileLikelihood`) or
        predictions (`PredictionProfile`).
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
    structure : dict[str, ProfileStructure]
        For each profile of a parameter that does not cross on both sides,
        what resetting the other parameters says.
    tolerance : float
        The numerical tolerance of the statistic: ``max(τ, ν)``, the
        tolerance of the optimizer and the noise floor of the objective at
        the fit.
    evaluation_floor : float | None
        The floor of a comparison of two evaluations of the objective; None
        when it cannot be computed.
    notes : list[str]
        What a reader should know, such as a threshold below the tolerance or
        a refit below the fit.
    failed : list[str]
        With a bootstrap threshold, each replicate that failed:
        ``"Replicate <index> failed: <reason>"``, also logged as a warning.
        Empty with any other threshold.
    not_converged : int
        With a bootstrap threshold, the replicates kept although their
        optimizer did not converge; 0 with any other threshold.
    """

    fit: CalibrationResult
    quantity: Literal["parameter", "prediction"]
    threshold: Threshold
    profiles: dict[str, Profile]
    intervals: dict[str, ProfileInterval]
    verdicts: dict[str, ProfileVerdictName | None]
    tolerance: float
    evaluation_floor: float | None = None
    structure: dict[str, ProfileStructure] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    not_converged: int = 0

    def plot(
        self,
        output_file: str | None = None,
        config: "PlotConfig | None" = None,
        axes: "Figure | None" = None,
    ) -> "Figure":
        """
        Draw each profile: one panel per parameter or prediction, with the
        statistic of each point, the threshold, and the ends of the interval.

        A profile that stays under the threshold up to a bound is open on that
        side; a flat one stays near 0 over its whole range.

        Parameters
        ----------
        output_file : str | None
            Path to save the figure to. If None, the figure is only returned.
        config : PlotConfig | None
            Layout and style. If None, the defaults.
        axes : Figure | None
            A figure of commol to draw on: each panel is the `Axes` whose
            label is the name of its profile. If None, a new figure.

        Returns
        -------
        Figure
            The matplotlib figure.

        Raises
        ------
        ValueError
            If ``axes`` has no panel of a profile.
        ImportError
            If matplotlib is not installed (``pip install 'commol[plot]'``).
        """
        from commol.profiles.plots import profiles_figure

        return profiles_figure(self, output_file=output_file, config=config, axes=axes)
```

### Methods:

#### plot

```python
plot(output_file: str | None = None, config: PlotConfig | None = None, axes: Figure | None = None) -> Figure
```

Draw each profile: one panel per parameter or prediction, with the statistic of each point, the threshold, and the ends of the interval.

A profile that stays under the threshold up to a bound is open on that side; a flat one stays near 0 over its whole range.

Parameters:

| Name          | Type         | Description | Default                                                                                                              |
| ------------- | ------------ | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| `output_file` | \`str        | None\`      | Path to save the figure to. If None, the figure is only returned.                                                    |
| `config`      | \`PlotConfig | None\`      | Layout and style. If None, the defaults.                                                                             |
| `axes`        | \`Figure     | None\`      | A figure of commol to draw on: each panel is the Axes whose label is the name of its profile. If None, a new figure. |

Returns:

| Type     | Description            |
| -------- | ---------------------- |
| `Figure` | The matplotlib figure. |

Raises:

| Type          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `ValueError`  | If axes has no panel of a profile.                           |
| `ImportError` | If matplotlib is not installed (pip install 'commol[plot]'). |

Source code in `commol/profiles/results.py`

```python
def plot(
    self,
    output_file: str | None = None,
    config: "PlotConfig | None" = None,
    axes: "Figure | None" = None,
) -> "Figure":
    """
    Draw each profile: one panel per parameter or prediction, with the
    statistic of each point, the threshold, and the ends of the interval.

    A profile that stays under the threshold up to a bound is open on that
    side; a flat one stays near 0 over its whole range.

    Parameters
    ----------
    output_file : str | None
        Path to save the figure to. If None, the figure is only returned.
    config : PlotConfig | None
        Layout and style. If None, the defaults.
    axes : Figure | None
        A figure of commol to draw on: each panel is the `Axes` whose
        label is the name of its profile. If None, a new figure.

    Returns
    -------
    Figure
        The matplotlib figure.

    Raises
    ------
    ValueError
        If ``axes`` has no panel of a profile.
    ImportError
        If matplotlib is not installed (``pip install 'commol[plot]'``).
    """
    from commol.profiles.plots import profiles_figure

    return profiles_figure(self, output_file=output_file, config=config, axes=axes)
```

## Profile

Bases: `_FrozenResult`

The profile of one parameter or prediction.

Attributes:

| Name              | Type                                    | Description                                                                   |
| ----------------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| `center`          | `float`                                 | The value at the fit.                                                         |
| `threshold`       | `float`                                 | The threshold of the statistic.                                               |
| `values`          | `list[float]`                           | The value of each point, increasing; the fit is one of them.                  |
| `statistics`      | `list[float]`                           | The statistic of each point: the objective of its refit less that of the fit. |
| `parameters`      | `list[dict[str, float]]`                | Every parameter of each refit.                                                |
| `multipliers`     | \`list\[float                           | None\]\`                                                                      |
| `converged`       | `list[bool]`                            | Whether each refit converged.                                                 |
| `ends`            | `tuple[ProfileEndName, ProfileEndName]` | Why the side below and the side above the fit end.                            |
| `evaluations`     | `int`                                   | Evaluations of the objective.                                                 |
| `replicate_rises` | \`list[float]                           | None\`                                                                        |
| `box_crossings`   | \`tuple\[str                            | None, str                                                                     |
| `refit_error`     | \`float                                 | None\`                                                                        |

Source code in `commol/profiles/results.py`

```python
class Profile(_FrozenResult):
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
    box_crossings : tuple[str | None, str | None]
        For the side below and the side above, the free parameter whose bound
        makes the profile cross the threshold: it is on a bound at the
        crossing point that it is not on over the whole side from the fit.
        Such a side counts as open; its interval end stays, flagged in
        `ProfileResult.notes`. A parameter that stays on its bound from the
        fit to the crossing is not seen.
    refit_error : float | None
        For a parameter, ``|δ|``: the largest distance from the objective at
        the fit of a refit with the parameter at its fitted value, from each
        neighbour of the fit. A diagnostic of the optimizer.
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
    box_crossings: tuple[str | None, str | None] = (None, None)
    refit_error: float | None = None
```

## ProfileInterval

Bases: `_FrozenResult`

The pointwise interval of a quantity: where its profile crosses the threshold.

Attributes:

| Name           | Type    | Description |
| -------------- | ------- | ----------- |
| `lower, upper` | \`float | None\`      |

Source code in `commol/profiles/results.py`

```python
class ProfileInterval(_FrozenResult):
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
```

## ProfileStructure

Bases: `_FrozenResult`

What resetting the other parameters says about a profile.

A profile is flat because other parameters compensate the parameter along it, or because nothing the data read depends on it. At each point of the profile inside the box, and at both ends of the box of the parameter, the other parameters are reset to their fitted values and the objective is evaluated once, with no optimizer; each reset is compared with the objective at the fit.

Attributes:

| Name             | Type                                 | Description                                                                                                                                                       |
| ---------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `verdict`        | \`{in_combination, non_identifiable} | None\`                                                                                                                                                            |
| `largest_rise`   | \`float                              | None\`                                                                                                                                                            |
| `compensated_by` | `list[str]`                          | The parameters whose reset alone, where each moves most along the profile, raises it above the threshold. Noise parameters of observation models are never named. |
| `joint_rise`     | \`float                              | None\`                                                                                                                                                            |
| `undetermined`   | \`UndeterminedName                   | None\`                                                                                                                                                            |
| `on_bound`       | \`str                                | None\`                                                                                                                                                            |

Source code in `commol/profiles/results.py`

```python
class ProfileStructure(_FrozenResult):
    """
    What resetting the other parameters says about a profile.

    A profile is flat because other parameters compensate the parameter
    along it, or because nothing the data read depends on it. At each point
    of the profile inside the box, and at both ends of the box of the
    parameter, the other parameters are reset to their fitted values and the
    objective is evaluated once, with no optimizer; each reset is compared
    with the objective at the fit.

    Attributes
    ----------
    verdict : {"in_combination", "non_identifiable"} | None
        ``non_identifiable``: every reset is feasible and within the
        evaluation floor of the fit. ``in_combination``: the profile is flat
        and a feasible reset inside the box rises above the threshold. None
        otherwise, and `undetermined` says why.
    largest_rise : float | None
        The largest rise of a feasible reset over the objective at the fit;
        None when none was feasible.
    compensated_by : list[str]
        The parameters whose reset alone, where each moves most along the
        profile, raises it above the threshold. Noise parameters of
        observation models are never named.
    joint_rise : float | None
        The largest rise of resetting the parameters not named together, at
        the outermost point of each side, over the objective of that point,
        when it is above the evaluation floor: `compensated_by` may be
        incomplete. Infinite when such a reset is not feasible.
    undetermined : UndeterminedName | None
        Why there is no verdict.
    on_bound : str | None
        With ``undetermined="on_bound"``, the parameter on its bound at the
        fit.
    """

    verdict: Literal["in_combination", "non_identifiable"] | None
    largest_rise: float | None
    compensated_by: list[str]
    joint_rise: float | None
    undetermined: UndeterminedName | None = None
    on_bound: str | None = None

    def reason(self, name: str, threshold: float, floor: float | None) -> str:
        """
        The reason of the verdict of parameter `name` in a report.

        Parameters
        ----------
        name : str
            The parameter.
        threshold : float
            The threshold of its profile.
        floor : float | None
            The evaluation floor of the result.

        Returns
        -------
        str
            The reason, which says that the verdict holds with the other
            parameters at their fitted values.
        """
        rise = (
            "no feasible reset"
            if self.largest_rise is None
            else (f"a largest rise of {self.largest_rise:.3g}")
        )
        if self.verdict == "in_combination":
            text = (
                f"profile likelihood: flat, and resetting the other parameters "
                f"gives {rise}, above the threshold {threshold:.3g}"
            )
            if self.compensated_by:
                text += (
                    f"; {name}, compensated by {', '.join(self.compensated_by)} "
                    "along its profile"
                )
            if self.joint_rise is not None:
                text += (
                    f"; the list may be incomplete (largest rise {self.joint_rise:.3g})"
                )
        elif self.verdict == "non_identifiable":
            text = (
                "profile likelihood: resetting the other parameters leaves the "
                f"objective within the evaluation floor {floor:.3g} of the fit "
                f"along the profile and at both ends of the box ({rise}): nothing "
                f"the data read depends on {name}"
            )
        else:
            text = (
                f"profile likelihood: flat, with no structural verdict "
                f"({self.undetermined}; {rise}, threshold {threshold:.3g})"
            )
        return f"{text}. {RESET_REASON}"
```

### Methods:

#### reason

```python
reason(name: str, threshold: float, floor: float | None) -> str
```

The reason of the verdict of parameter `name` in a report.

Parameters:

| Name        | Type    | Description                   | Default                             |
| ----------- | ------- | ----------------------------- | ----------------------------------- |
| `name`      | `str`   | The parameter.                | *required*                          |
| `threshold` | `float` | The threshold of its profile. | *required*                          |
| `floor`     | \`float | None\`                        | The evaluation floor of the result. |

Returns:

| Type  | Description                                                                                     |
| ----- | ----------------------------------------------------------------------------------------------- |
| `str` | The reason, which says that the verdict holds with the other parameters at their fitted values. |

Source code in `commol/profiles/results.py`

```python
def reason(self, name: str, threshold: float, floor: float | None) -> str:
    """
    The reason of the verdict of parameter `name` in a report.

    Parameters
    ----------
    name : str
        The parameter.
    threshold : float
        The threshold of its profile.
    floor : float | None
        The evaluation floor of the result.

    Returns
    -------
    str
        The reason, which says that the verdict holds with the other
        parameters at their fitted values.
    """
    rise = (
        "no feasible reset"
        if self.largest_rise is None
        else (f"a largest rise of {self.largest_rise:.3g}")
    )
    if self.verdict == "in_combination":
        text = (
            f"profile likelihood: flat, and resetting the other parameters "
            f"gives {rise}, above the threshold {threshold:.3g}"
        )
        if self.compensated_by:
            text += (
                f"; {name}, compensated by {', '.join(self.compensated_by)} "
                "along its profile"
            )
        if self.joint_rise is not None:
            text += (
                f"; the list may be incomplete (largest rise {self.joint_rise:.3g})"
            )
    elif self.verdict == "non_identifiable":
        text = (
            "profile likelihood: resetting the other parameters leaves the "
            f"objective within the evaluation floor {floor:.3g} of the fit "
            f"along the profile and at both ends of the box ({rise}): nothing "
            f"the data read depends on {name}"
        )
    else:
        text = (
            f"profile likelihood: flat, with no structural verdict "
            f"({self.undetermined}; {rise}, threshold {threshold:.3g})"
        )
    return f"{text}. {RESET_REASON}"
```
