# Local Approximations API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.local`: the Laplace approximation, the sandwich, the Laplace mixture, and their results. Guide: [Local Approximations](https://munqu.github.io/commol/latest/guide/uncertainty/local/index.md).

## Laplace

Bases: `UncertaintyMethod[LaplaceResult]`, `RangeChecked`

The Laplace approximation of the problem's observation models: the normal `N(θ̂, H⁻¹)` on the unbounded scale of the parameters, centred on the fit, with `H` the curvature of `−log L` there.

It needs `observation_models`: a loss has no noise scale, so its curvature is no inverse variance (use `Sandwich`). A fit on a bound of its parameter is refused: on the unbounded scale its variance is infinite there.

Attributes:

| Name        | Type                      | Description                                                                                       |
| ----------- | ------------------------- | ------------------------------------------------------------------------------------------------- |
| `draws`     | `int`                     | Number of draws of the normal, at least 1.                                                        |
| `curvature` | `{gauss_newton, hessian}` | The curvature (default "gauss_newton").                                                           |
| `starts`    | `int`                     | Starts of the fit (default 1): the initial guess, then rows of a Latin hypercube over the bounds. |

Source code in `commol/local/methods.py`

```python
class Laplace(UncertaintyMethod[LaplaceResult], RangeChecked):
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
    curvature : {"gauss_newton", "hessian"}
        The curvature (default ``"gauss_newton"``).
    starts : int
        Starts of the fit (default 1): the initial guess, then rows of a Latin
        hypercube over the bounds.
    """

    _runner = staticmethod(run_laplace)

    draws: int = Field(description="Number of draws.")
    curvature: Literal["gauss_newton", "hessian"] = Field(
        default="gauss_newton", description="The curvature."
    )
    starts: int = Field(default=1, description="Starts of the fit.")
```

## Sandwich

Bases: `UncertaintyMethod[SandwichResult]`, `RangeChecked`

The sandwich covariance `H⁻¹ B H⁻¹` at the fit, with the meat `B` summed over blocks of positions (the clustered estimator CR1): `B = G/(G − 1) · Σ_b s_b s_bᵀ`, with `s_b` the sum of the gradients of the terms of the observations of block `b`.

It needs no noise model: with a loss, `H` is the curvature of the loss; with observation models, of `−log L`. It holds when the noise depends on its neighbours inside a block, when its size changes, and when the noise model is wrong but the mean is right. Intervals and draws read a Student t with `G − 1` degrees of freedom; with `G` blocks not above the parameters it is refused, since `B` would have rank below them. `loss_function="mae"` is refused: its sandwich needs a density of the residuals at 0 and a bandwidth (use a bootstrap).

Attributes:

| Name           | Type                      | Description                                                                                                 |
| -------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `block_length` | \`int                     | auto\`                                                                                                      |
| `draws`        | `int`                     | Number of draws of the Student t, at least 1.                                                               |
| `curvature`    | `{gauss_newton, hessian}` | The curvature H (default "gauss_newton"). With the Hessian, the sandwich also holds when the mean is wrong. |
| `starts`       | `int`                     | Starts of the fit (default 1): the initial guess, then rows of a Latin hypercube over the bounds.           |

Source code in `commol/local/methods.py`

```python
class Sandwich(UncertaintyMethod[SandwichResult], RangeChecked):
    """
    The sandwich covariance ``H⁻¹ B H⁻¹`` at the fit, with the meat ``B``
    summed over blocks of positions (the clustered estimator CR1):
    ``B = G/(G − 1) · Σ_b s_b s_bᵀ``, with ``s_b`` the sum of the gradients
    of the terms of the observations of block ``b``.

    It needs no noise model: with a loss, ``H`` is the curvature of the loss;
    with observation models, of ``−log L``. It holds when the noise depends
    on its neighbours inside a block, when its size changes, and when the
    noise model is wrong but the mean is right. Intervals and draws read a
    Student t with ``G − 1`` degrees of freedom; with ``G`` blocks not above
    the parameters it is refused, since ``B`` would have rank below them.
    ``loss_function="mae"`` is refused: its sandwich needs a density of the
    residuals at 0 and a bandwidth (use a bootstrap).

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; ``"auto"`` is the rule of Politis and White
        (2004) for non-overlapping blocks on the residuals of the fit.
    draws : int
        Number of draws of the Student t, at least 1.
    curvature : {"gauss_newton", "hessian"}
        The curvature ``H`` (default ``"gauss_newton"``). With the Hessian,
        the sandwich also holds when the mean is wrong.
    starts : int
        Starts of the fit (default 1): the initial guess, then rows of a Latin
        hypercube over the bounds.
    """

    _runner = staticmethod(run_sandwich)

    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    draws: int = Field(description="Number of draws.")
    curvature: Literal["gauss_newton", "hessian"] = Field(
        default="gauss_newton", description="The curvature."
    )
    starts: int = Field(default=1, description="Starts of the fit.")
```

## LaplaceMixture

Bases: `UncertaintyMethod[ImportanceResult]`, `RangeChecked`

The Laplace mixture: a Laplace approximation at each distinct optimum of a multi-start fit, weighted by its Laplace evidence, as the proposal of importance sampling of a posterior, with the largest weights smoothed (Pareto-smoothed importance sampling, Vehtari et al. 2024).

The ends of the starts merge into a component when they are closer than one Mahalanobis unit to it. Every draw is weighted by the target over the mixture, on the unbounded scale. `k_hat` says whether the mixture covers the target: above 0.7 the sample is marked unreliable.

Attributes:

| Name        | Type                      | Description                                                                                       |
| ----------- | ------------------------- | ------------------------------------------------------------------------------------------------- |
| `starts`    | `int`                     | Starts of the fit (default 1): the initial guess, then rows of a Latin hypercube over the bounds. |
| `draws`     | `int`                     | Number of draws of the mixture, at least 25, so that the tail of the weights has 5.               |
| `target`    | \`LikelihoodTarget        | GibbsTarget\`                                                                                     |
| `curvature` | `{gauss_newton, hessian}` | The curvature of each component (default "gauss_newton").                                         |

Source code in `commol/local/methods.py`

```python
class LaplaceMixture(UncertaintyMethod[ImportanceResult], RangeChecked):
    """
    The Laplace mixture: a Laplace approximation at each distinct optimum of
    a multi-start fit, weighted by its Laplace evidence, as the proposal of
    importance sampling of a posterior, with the largest weights smoothed
    (Pareto-smoothed importance sampling, Vehtari et al. 2024).

    The ends of the starts merge into a component when they are closer than
    one Mahalanobis unit to it. Every draw is weighted by the target over
    the mixture, on the unbounded scale. ``k_hat`` says whether the mixture
    covers the target: above 0.7 the sample is marked unreliable.

    Attributes
    ----------
    starts : int
        Starts of the fit (default 1): the initial guess, then rows of a Latin
        hypercube over the bounds.
    draws : int
        Number of draws of the mixture, at least 25, so that the tail of the
        weights has 5.
    target : LikelihoodTarget | GibbsTarget
        The posterior of the observation models with priors, or the Gibbs
        posterior of the loss with priors and ``eta``.
    curvature : {"gauss_newton", "hessian"}
        The curvature of each component (default ``"gauss_newton"``).
    """

    _runner = staticmethod(run_laplace_mixture)

    starts: int = Field(default=1, description="Starts of the fit.")
    draws: int = Field(description="Number of draws.")
    target: _MixtureTarget = Field(description="The posterior to sample.")
    curvature: Literal["gauss_newton", "hessian"] = Field(
        default="gauss_newton", description="The curvature."
    )
```

## LikelihoodTarget

Bases: `WithPriors`

The posterior of the problem's observation models, `π(θ) ∝ prior(θ) · L(θ)`, as the target of a `LaplaceMixture`.

Attributes:

| Name     | Type     | Description               |
| -------- | -------- | ------------------------- |
| `priors` | \`Priors | dict[str, distribution]\` |

Source code in `commol/local/methods.py`

```python
class LikelihoodTarget(WithPriors):
    """
    The posterior of the problem's observation models,
    ``π(θ) ∝ prior(θ) · L(θ)``, as the target of a `LaplaceMixture`.

    Attributes
    ----------
    priors : Priors | dict[str, distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    """

    kind: Literal["likelihood"] = "likelihood"
```

## GibbsTarget

Bases: `WithPriors`, `RangeChecked`

The Gibbs posterior of the problem's loss, `π(θ) ∝ prior(θ) · exp(−η · loss(θ))`, as the target of a `LaplaceMixture`.

Attributes:

| Name     | Type     | Description                     |
| -------- | -------- | ------------------------------- |
| `priors` | \`Priors | dict[str, distribution]\`       |
| `eta`    | `float`  | The scale of the loss, above 0. |

Source code in `commol/local/methods.py`

```python
class GibbsTarget(WithPriors, RangeChecked):
    """
    The Gibbs posterior of the problem's loss,
    ``π(θ) ∝ prior(θ) · exp(−η · loss(θ))``, as the target of a
    `LaplaceMixture`.

    Attributes
    ----------
    priors : Priors | dict[str, distribution]
        The prior of each calibrated parameter, or
        ``Priors.uniform_on_bounds()``.
    eta : float
        The scale of the loss, above 0.
    """

    kind: Literal["gibbs"] = "gibbs"
    eta: float = Field(description="Scale of the loss.")
```

## LocalResult

Bases: `ResultModel`

What every local approximation returns: a distribution on the unbounded scale of the parameters, centred on the fit.

Attributes:

| Name         | Type                | Description                                                                                                            |
| ------------ | ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `fit`        | `CalibrationResult` | The fit the approximation is centred on: Calibrator.run(starts).                                                       |
| `covariance` | `list[list[float]]` | The covariance of the distribution on the unbounded scale of the parameters, in the order of the problem's parameters. |
| `sample`     | `ParameterSample`   | Draws of the distribution, mapped back to the parameters, with equal weights.                                          |
| `notes`      | `list[str]`         | The entries of the curvature whose estimated error is not small against their scale; each is also logged as a warning. |

Source code in `commol/local/results.py`

```python
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
        names = calibrator.parameter_names
        if self.sample.names != names:
            raise ValueError(
                f"The result has the parameters {self.sample.names}; the problem "
                f"has {names}."
            )
        points = [float(position) for position in positions]
        outputs, center, lower, upper = commol_rs.inference.delta_band(
            calibrator._engine_problem,
            [self.fit.best_parameters[name] for name in names],
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
```

### Methods:

#### degrees_of_freedom

```python
degrees_of_freedom() -> float | None
```

The degrees of freedom of the distribution.

Returns:

| Type    | Description |
| ------- | ----------- |
| \`float | None\`      |

Source code in `commol/local/results.py`

```python
def degrees_of_freedom(self) -> float | None:
    """
    The degrees of freedom of the distribution.

    Returns
    -------
    float | None
        Those of a Student t, or None for a normal.
    """
    return None
```

#### delta_band

```python
delta_band(calibrator: CalibrationPort, positions: Sequence[float], confidence_level: float) -> DeltaBand
```

The band of every output of this approximation by the delta method.

The model is linearized at the fit: the variance of an output is `J Σ Jᵀ`, with `J` its derivatives on the unbounded scale and `Σ` the covariance of this result. The band is the prediction at the fit plus and minus the quantile of a normal, or of the Student t of a `SandwichResult`, times its standard deviation. It runs the model a few times per parameter, not once per draw, and reads only the fields of this result, so a result read back from JSON works.

Parameters:

| Name               | Type              | Description                                             | Default    |
| ------------------ | ----------------- | ------------------------------------------------------- | ---------- |
| `calibrator`       | `Calibrator`      | The calibrator of the problem this approximation is of. | *required* |
| `positions`        | `Sequence[float]` | Positions to report at.                                 | *required* |
| `confidence_level` | `float`           | Strictly between 0 and 1.                               | *required* |

Returns:

| Type        | Description                                                                             |
| ----------- | --------------------------------------------------------------------------------------- |
| `DeltaBand` | The prediction at the fit and the ends of the interval of each output at each position. |

Raises:

| Type           | Description                                                                                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`   | If the parameters of this result are not those of the problem, the level is not strictly between 0 and 1, the engine is stochastic, or the engine rejects a position. |
| `RuntimeError` | If the derivatives fail at the fit.                                                                                                                                   |

Source code in `commol/local/results.py`

```python
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
    names = calibrator.parameter_names
    if self.sample.names != names:
        raise ValueError(
            f"The result has the parameters {self.sample.names}; the problem "
            f"has {names}."
        )
    points = [float(position) for position in positions]
    outputs, center, lower, upper = commol_rs.inference.delta_band(
        calibrator._engine_problem,
        [self.fit.best_parameters[name] for name in names],
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
```

## LaplaceResult

Bases: `LocalResult`

The result of a Laplace approximation: the normal `N(θ̂, Σ)` on the unbounded scale, `Σ = H⁻¹`.

Attributes:

| Name         | Type                | Description                                                                          |
| ------------ | ------------------- | ------------------------------------------------------------------------------------ |
| `fit`        | `CalibrationResult` | The fit the approximation is centred on: Calibrator.run(starts).                     |
| `covariance` | `list[list[float]]` | Σ = H⁻¹ on the unbounded scale of the parameters.                                    |
| `sample`     | `ParameterSample`   | Draws of the normal, mapped back to the parameters, with equal weights.              |
| `notes`      | `list[str]`         | The entries of the curvature whose estimated error is not small against their scale. |

Source code in `commol/local/results.py`

```python
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
```

## SandwichResult

Bases: `LocalResult`

The result of a sandwich: a Student t on the unbounded scale of the parameters, centred on the fit, of covariance `H⁻¹ B H⁻¹` and `blocks − 1` degrees of freedom.

Attributes:

| Name                     | Type                | Description                                                                          |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------------ |
| `fit`                    | `CalibrationResult` | The fit the sandwich is centred on: Calibrator.run(starts).                          |
| `covariance`             | `list[list[float]]` | H⁻¹ B H⁻¹ on the unbounded scale of the parameters.                                  |
| `sample`                 | `ParameterSample`   | Draws of the Student t, mapped back to the parameters, with equal weights.           |
| `notes`                  | `list[str]`         | The entries of the curvature whose estimated error is not small against their scale. |
| `blocks`                 | `int`               | The number of blocks G; intervals read a Student t with G − 1 degrees of freedom.    |
| `block_length`           | `int`               | The positions per block.                                                             |
| `block_length_estimates` | \`dict[str, float]  | None\`                                                                               |

Source code in `commol/local/results.py`

```python
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
```

### Methods:

#### degrees_of_freedom

```python
degrees_of_freedom() -> float | None
```

The degrees of freedom of the Student t.

Returns:

| Type    | Description |
| ------- | ----------- |
| \`float | None\`      |

Source code in `commol/local/results.py`

```python
def degrees_of_freedom(self) -> float | None:
    """
    The degrees of freedom of the Student t.

    Returns
    -------
    float | None
        ``blocks − 1``.
    """
    return float(self.blocks - 1)
```

## ImportanceResult

Bases: `ResultModel`

The result of a Laplace mixture: draws of the mixture weighted by the target over the mixture, smoothed by Pareto-smoothed importance sampling.

Attributes:

| Name             | Type                     | Description                                                                                                                      |
| ---------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `fit`            | `CalibrationResult`      | The multi-start fit whose ends made the components: Calibrator.run(starts).                                                      |
| `sample`         | `ParameterSample`        | The draws with their smoothed weights.                                                                                           |
| `components`     | `list[MixtureComponent]` | The components, in order of their objective.                                                                                     |
| `k_hat`          | `float`                  | The Pareto shape of the tail of the weights. Up to 0.5 the weights have a finite variance; above 0.7 the estimate is unreliable. |
| `effective_size` | `float`                  | Kish's effective size of the smoothed weights.                                                                                   |
| `reliable`       | `bool`                   | Whether k_hat is at most 0.7. An unreliable sample is also logged as a warning.                                                  |
| `notes`          | `list[str]`              | Ends that gave no component, and an unreliable sample, in words.                                                                 |

Source code in `commol/local/results.py`

```python
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
```

## MixtureComponent

Bases: `ResultModel`

One component of a Laplace mixture: the Laplace approximation at one optimum of the starts.

Attributes:

| Name           | Type                | Description                                                   |
| -------------- | ------------------- | ------------------------------------------------------------- |
| `center`       | `dict[str, float]`  | The optimum, by parameter.                                    |
| `weight`       | `float`             | Its weight in the mixture: its Laplace evidence over the sum. |
| `covariance`   | `list[list[float]]` | Its covariance on the unbounded scale of the parameters.      |
| `log_evidence` | `float`             | The log of its Laplace evidence, log p(θ̂) + log               |

Source code in `commol/local/results.py`

```python
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
```

## DeltaBand

Bases: `ResultModel`

The band of every output by the delta method (`LocalResult.delta_band`): the prediction at the fit plus and minus a quantile times the standard deviation of the linearized model, `J Σ Jᵀ`, with no run of the draws.

Attributes:

| Name                   | Type                     | Description                                                                                      |
| ---------------------- | ------------------------ | ------------------------------------------------------------------------------------------------ |
| `confidence_level`     | `float`                  | The level of the central interval.                                                               |
| `positions`            | `list[float]`            | Where the values are.                                                                            |
| `center, lower, upper` | `dict[str, list[float]]` | For each output, one value per position: the prediction at the fit and the ends of the interval. |

Source code in `commol/local/results.py`

```python
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
```
