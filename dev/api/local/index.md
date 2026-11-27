# Local Approximations API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.local`: the Laplace approximation, the sandwich, the Laplace mixture, and their results. Guide: [Local Approximations](https://munqu.github.io/commol/latest/guide/uncertainty/local/index.md).

## Laplace

Bases: `UncertaintyMethod[LaplaceResult]`, `RangeChecked`

The Laplace approximation of the problem's observation models: the normal `N(θ̂, H⁻¹)` on the unbounded scale of the parameters, centred on the fit, with `H` the curvature of `−log L` there.

It needs `observation_models`: a loss has no noise scale, so its curvature is no inverse variance (use `Sandwich`). A fit on a bound of its parameter is refused: on the unbounded scale its variance is infinite there. A bound less than 3 sd from the fit is a note: the draws, mapped back, are skewed by it.

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
    infinite there. A bound less than 3 sd from the fit is a note: the draws,
    mapped back, are skewed by it.

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

The sandwich covariance at the fit, with the meat summed over blocks of positions: `√(f_j f_k) · (H⁻¹ Σ_b s̃_b s̃_bᵀ H⁻¹)_jk`, with `s_b` the sum of the gradients of the terms of the observations of block `b`, taken after one Gauss–Newton step to the zero of their sum `g`: `s̃_b = s_b − J_b J⁻¹ g`. The factor `f_j` and the degrees of freedom `ν_j` of each parameter come from the share `J_b` of its Gauss–Newton curvature `J` that each block holds (Bell and McCaffrey, 2002): with equal shares, `f_j = G/(G − 1)` and `ν_j = G − 1` for `G` blocks; a parameter that few blocks read has fewer, and a block with no curvature counts for nothing.

It needs no noise model: with a loss, `H` is the curvature of the loss; with observation models, of `−log L`. It holds when the noise depends on its neighbours inside a block, when its size changes, and when the noise model is wrong but the mean is right. Intervals read the Student t of each parameter, and draws a grouped t whose margins are those t. With no more blocks with curvature than parameters it is refused, since the meat would have rank below them; so it is when the blocks each hold all the curvature of a parameter. `loss_function="mae"` is refused: its sandwich needs a density of the residuals at 0 and a bandwidth (use a bootstrap). A bound less than 3 sd from the fit is a note: the draws, mapped back, are skewed by it. The distance of its Student t is that of a normal with the same mass past the bound, so with few degrees of freedom a bound beyond 3 units of the scale of the t can be noted.

Attributes:

| Name           | Type                      | Description                                                                                                 |
| -------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `block_length` | `int`                     | Positions per block of the meat; they must make more blocks with curvature than parameters.                 |
| `draws`        | `int`                     | Number of draws of the grouped t, at least 1.                                                               |
| `curvature`    | `{gauss_newton, hessian}` | The curvature H (default "gauss_newton"). With the Hessian, the sandwich also holds when the mean is wrong. |
| `starts`       | `int`                     | Starts of the fit (default 1): the initial guess, then rows of a Latin hypercube over the bounds.           |

Source code in `commol/local/methods.py`

```python
class Sandwich(UncertaintyMethod[SandwichResult], RangeChecked):
    """
    The sandwich covariance at the fit, with the meat summed over blocks of
    positions: ``√(f_j f_k) · (H⁻¹ Σ_b s̃_b s̃_bᵀ H⁻¹)_jk``, with ``s_b`` the
    sum of the gradients of the terms of the observations of block ``b``,
    taken after one Gauss–Newton step to the zero of their sum ``g``:
    ``s̃_b = s_b − J_b J⁻¹ g``. The factor ``f_j`` and the degrees of freedom
    ``ν_j`` of each parameter come from the share ``J_b`` of its Gauss–Newton
    curvature ``J`` that each block holds (Bell and McCaffrey, 2002): with
    equal shares, ``f_j = G/(G − 1)`` and ``ν_j = G − 1`` for ``G`` blocks; a
    parameter that few blocks read has fewer, and a block with no curvature
    counts for nothing.

    It needs no noise model: with a loss, ``H`` is the curvature of the loss;
    with observation models, of ``−log L``. It holds when the noise depends
    on its neighbours inside a block, when its size changes, and when the
    noise model is wrong but the mean is right. Intervals read the Student t
    of each parameter, and draws a grouped t whose margins are those t. With
    no more blocks with curvature than parameters it is refused, since the
    meat would have rank below them; so it is when the blocks each hold all
    the curvature of a parameter. ``loss_function="mae"`` is refused: its
    sandwich needs a density of the residuals at 0 and a bandwidth (use a
    bootstrap). A bound less than 3 sd from the fit is a note: the draws,
    mapped back, are skewed by it. The distance of its Student t is that of
    a normal with the same mass past the bound, so with few degrees of
    freedom a bound beyond 3 units of the scale of the t can be noted.

    Attributes
    ----------
    block_length : int
        Positions per block of the meat; they must make more blocks with
        curvature than parameters.
    draws : int
        Number of draws of the grouped t, at least 1.
    curvature : {"gauss_newton", "hessian"}
        The curvature ``H`` (default ``"gauss_newton"``). With the Hessian,
        the sandwich also holds when the mean is wrong.
    starts : int
        Starts of the fit (default 1): the initial guess, then rows of a Latin
        hypercube over the bounds.
    """

    _runner = staticmethod(run_sandwich)

    block_length: int = Field(description="Positions per block.")
    draws: int = Field(description="Number of draws.")
    curvature: Literal["gauss_newton", "hessian"] = Field(
        default="gauss_newton", description="The curvature."
    )
    starts: int = Field(default=1, description="Starts of the fit.")
```

## LaplaceMixture

Bases: `UncertaintyMethod[ImportanceResult]`, `RangeChecked`

The Laplace mixture: a Laplace approximation at each distinct optimum of a multi-start fit, weighted by its Laplace evidence, as the proposal of importance sampling of a posterior, with the largest weights smoothed (Pareto-smoothed importance sampling, Vehtari et al. 2024).

The ends of the starts merge into a component when they are closer than one Mahalanobis unit to it. Every draw is weighted by the target over the mixture, on the unbounded scale. `k_hat` says whether the mixture covers the target: above 0.7 the sample is marked unreliable. A bound less than 3 sd from the centre of a component is a note: it skews the component, which the importance weights correct as far as `k_hat` says.

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
    covers the target: above 0.7 the sample is marked unreliable. A bound
    less than 3 sd from the centre of a component is a note: it skews the
    component, which the importance weights correct as far as ``k_hat``
    says.

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

| Name         | Type                | Description                                                                                                                                                                                                                      |
| ------------ | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit`        | `CalibrationResult` | The fit the approximation is centred on: Calibrator.run(starts).                                                                                                                                                                 |
| `covariance` | `list[list[float]]` | The covariance of the distribution on the unbounded scale of the parameters, in the order of the problem's parameters.                                                                                                           |
| `sample`     | `ParameterSample`   | Draws of the distribution, mapped back to the parameters, with equal weights.                                                                                                                                                    |
| `notes`      | `list[str]`         | The entries of the curvature whose estimated error is not small against their scale, then each bound less than 3 sd from the fit, on the scale of the parameters, whose draws the bound skews; each is also logged as a warning. |

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
        against their scale, then each bound less than 3 sd from the fit, on
        the scale of the parameters, whose draws the bound skews; each is also
        logged as a warning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    covariance: list[list[float]]
    sample: ParameterSample
    notes: list[str]

    def _block_length(self) -> int | None:
        """The positions per block of the Student t of a sandwich, or None
        for a normal."""
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
        the fit plus and minus the quantile of a normal, or for a
        `SandwichResult` of a Student t, times its standard deviation. The t
        of each output at each position has the degrees of freedom that the
        share of the curvature of each block gives its direction ``J``, read
        at the fit as for the parameters (`SandwichResult`). It runs the
        model a few times per parameter, not once per draw. It reads the
        fields of this result, so a result read back from JSON works; for a
        sandwich, the degrees of freedom also read the observations of the
        problem, in blocks of `SandwichResult.block_length`, which must be
        the observations of the fit.

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
            stochastic, the engine rejects a position, or, for a sandwich,
            its block length does not fit the positions of the problem or
            the Gauss–Newton curvature of its blocks at the fit is
            singular.
        RuntimeError
            If the derivatives fail at the fit.
        """
        points = [float(position) for position in positions]
        outputs, center, lower, upper = commol_rs.inference.delta_band(
            calibrator._engine_problem,
            self.sample.names,
            list(self.fit.best_parameters.items()),
            self.covariance,
            self._block_length(),
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

#### delta_band

```python
delta_band(calibrator: CalibrationPort, positions: Sequence[float], confidence_level: float) -> DeltaBand
```

The band of every output of this approximation by the delta method.

The model is linearized at the fit: the variance of an output is `J Σ Jᵀ`, with `J` its derivatives on the unbounded scale and `Σ` the covariance of this result. The band is the prediction at the fit plus and minus the quantile of a normal, or for a `SandwichResult` of a Student t, times its standard deviation. The t of each output at each position has the degrees of freedom that the share of the curvature of each block gives its direction `J`, read at the fit as for the parameters (`SandwichResult`). It runs the model a few times per parameter, not once per draw. It reads the fields of this result, so a result read back from JSON works; for a sandwich, the degrees of freedom also read the observations of the problem, in blocks of `SandwichResult.block_length`, which must be the observations of the fit.

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

| Type           | Description                                                                                                                                                                                                                                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError`   | If the parameters of this result are not those of the problem, the level is not strictly between 0 and 1, the engine is stochastic, the engine rejects a position, or, for a sandwich, its block length does not fit the positions of the problem or the Gauss–Newton curvature of its blocks at the fit is singular. |
| `RuntimeError` | If the derivatives fail at the fit.                                                                                                                                                                                                                                                                                   |

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
    the fit plus and minus the quantile of a normal, or for a
    `SandwichResult` of a Student t, times its standard deviation. The t
    of each output at each position has the degrees of freedom that the
    share of the curvature of each block gives its direction ``J``, read
    at the fit as for the parameters (`SandwichResult`). It runs the
    model a few times per parameter, not once per draw. It reads the
    fields of this result, so a result read back from JSON works; for a
    sandwich, the degrees of freedom also read the observations of the
    problem, in blocks of `SandwichResult.block_length`, which must be
    the observations of the fit.

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
        stochastic, the engine rejects a position, or, for a sandwich,
        its block length does not fit the positions of the problem or
        the Gauss–Newton curvature of its blocks at the fit is
        singular.
    RuntimeError
        If the derivatives fail at the fit.
    """
    points = [float(position) for position in positions]
    outputs, center, lower, upper = commol_rs.inference.delta_band(
        calibrator._engine_problem,
        self.sample.names,
        list(self.fit.best_parameters.items()),
        self.covariance,
        self._block_length(),
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

| Name         | Type                | Description                                                                                                                       |
| ------------ | ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `fit`        | `CalibrationResult` | The fit the approximation is centred on: Calibrator.run(starts).                                                                  |
| `covariance` | `list[list[float]]` | Σ = H⁻¹ on the unbounded scale of the parameters.                                                                                 |
| `sample`     | `ParameterSample`   | Draws of the normal, mapped back to the parameters, with equal weights.                                                           |
| `notes`      | `list[str]`         | The entries of the curvature whose estimated error is not small against their scale, then each bound less than 3 sd from the fit. |

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
        against their scale, then each bound less than 3 sd from the fit.
    """
```

## SandwichResult

Bases: `LocalResult`

The result of a sandwich: a Student t on the unbounded scale of the parameters, centred on the fit, of covariance `√(f_j f_k) · (H⁻¹ Σ_b s̃_b s̃_bᵀ H⁻¹)_jk`, with the scores `s_b` of each block taken after one Gauss–Newton step to the zero of their sum `g`, `s̃_b = s_b − J_b J⁻¹ g`, and the factor `f_j` and the degrees of freedom `ν_j` of each parameter read from the share `J_b` of its Gauss–Newton curvature `J` that each block holds.

Attributes:

| Name                 | Type                | Description                                                                                                                                                                                       |
| -------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit`                | `CalibrationResult` | The fit the sandwich is centred on: Calibrator.run(starts).                                                                                                                                       |
| `covariance`         | `list[list[float]]` | The sandwich on the unbounded scale of the parameters.                                                                                                                                            |
| `sample`             | `ParameterSample`   | Draws of a grouped t, mapped back to the parameters, with equal weights: the margin of each parameter is a Student t of its degrees of freedom.                                                   |
| `notes`              | `list[str]`         | The entries of the curvature whose estimated error is not small against their scale, then each bound less than 3 sd from the fit, read as a normal with the mass of the Student t past the bound. |
| `degrees_of_freedom` | `dict[str, float]`  | The degrees of freedom ν_j of each parameter: blocks − 1 when every block holds the same share of its curvature, fewer when few blocks hold it.                                                   |
| `blocks`             | `int`               | The number of blocks G, those with no curvature included.                                                                                                                                         |
| `block_length`       | `int`               | The positions per block.                                                                                                                                                                          |

Source code in `commol/local/results.py`

```python
class SandwichResult(LocalResult):
    """
    The result of a sandwich: a Student t on the unbounded scale of the
    parameters, centred on the fit, of covariance
    ``√(f_j f_k) · (H⁻¹ Σ_b s̃_b s̃_bᵀ H⁻¹)_jk``, with the scores ``s_b`` of
    each block taken after one Gauss–Newton step to the zero of their sum
    ``g``, ``s̃_b = s_b − J_b J⁻¹ g``, and the factor ``f_j`` and the degrees
    of freedom ``ν_j`` of each parameter read from the share ``J_b`` of its
    Gauss–Newton curvature ``J`` that each block holds.

    Attributes
    ----------
    fit : CalibrationResult
        The fit the sandwich is centred on: ``Calibrator.run(starts)``.
    covariance : list[list[float]]
        The sandwich on the unbounded scale of the parameters.
    sample : ParameterSample
        Draws of a grouped t, mapped back to the parameters, with equal
        weights: the margin of each parameter is a Student t of its
        degrees of freedom.
    notes : list[str]
        The entries of the curvature whose estimated error is not small
        against their scale, then each bound less than 3 sd from the fit,
        read as a normal with the mass of the Student t past the bound.
    degrees_of_freedom : dict[str, float]
        The degrees of freedom ``ν_j`` of each parameter: ``blocks − 1`` when
        every block holds the same share of its curvature, fewer when few
        blocks hold it.
    blocks : int
        The number of blocks ``G``, those with no curvature included.
    block_length : int
        The positions per block.
    """

    degrees_of_freedom: dict[str, float]
    blocks: int
    block_length: int

    def _block_length(self) -> int | None:
        """The positions per block, whose curvature gives the degrees of
        freedom of the band."""
        return self.block_length
```

## ImportanceResult

Bases: `ResultModel`

The result of a Laplace mixture: draws of the mixture weighted by the target over the mixture, smoothed by Pareto-smoothed importance sampling.

Attributes:

| Name             | Type                     | Description                                                                                                                                                                                                                  |
| ---------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit`            | `CalibrationResult`      | The multi-start fit whose ends made the components: Calibrator.run(starts).                                                                                                                                                  |
| `sample`         | `ParameterSample`        | The draws with their smoothed weights.                                                                                                                                                                                       |
| `components`     | `list[MixtureComponent]` | The components, in order of their objective.                                                                                                                                                                                 |
| `k_hat`          | `float`                  | The Pareto shape of the tail of the weights. Up to 0.5 the weights have a finite variance; above 0.7 the estimate is unreliable.                                                                                             |
| `effective_size` | `float`                  | Kish's effective size of the smoothed weights.                                                                                                                                                                               |
| `reliable`       | `bool`                   | Whether k_hat is at most 0.7 and no end that gives no component fits at least as well as the worst component. An unreliable sample is also logged as a warning.                                                              |
| `notes`          | `list[str]`              | Ends that gave no component, each bound less than 3 sd from the centre of a component, whose skew the importance weights correct as far as k_hat says, and an unreliable sample, in words; each is also logged as a warning. |

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
        Whether ``k_hat`` is at most 0.7 and no end that gives no component
        fits at least as well as the worst component. An unreliable sample is
        also logged as a warning.
    notes : list[str]
        Ends that gave no component, each bound less than 3 sd from the
        centre of a component, whose skew the importance weights correct as
        far as ``k_hat`` says, and an unreliable sample, in words; each is
        also logged as a warning.
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
