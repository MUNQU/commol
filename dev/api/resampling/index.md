# Resampling API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.resampling`: the bootstraps and the block jackknife, and their results. Guide: [Resampling](https://munqu.github.io/commol/latest/guide/uncertainty/resampling/index.md).

## BayesianBootstrap

Bases: `UncertaintyMethod[BootstrapResult]`, `RangeChecked`

The Bayesian (weighted) bootstrap by blocks.

Each replicate draws one Dirichlet(1, …, 1) weight per block, scaled to mean 1, multiplies the weight of every observation of the block by it, and fits. With a loss it is the loss-likelihood bootstrap of Lyddon, Holmes and Walker (2019); with one position per block, the Bayesian bootstrap of Rubin (1981) and the weighted likelihood bootstrap of Newton and Raftery (1994). The observed values never change.

Attributes:

| Name           | Type       | Description                                                                                                   |
| -------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| `kind`         | `bayesian` | The bootstrap, which the engine reads.                                                                        |
| `replicates`   | `int`      | Number of replicates, at least 1.                                                                             |
| `block_length` | \`int      | auto\`                                                                                                        |
| `starts`       | `int`      | Starts of the fit and of each replicate (default 1): the fit, then rows of a Latin hypercube over the bounds. |

Source code in `commol/resampling/methods.py`

```python
class BayesianBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The Bayesian (weighted) bootstrap by blocks.

    Each replicate draws one Dirichlet(1, …, 1) weight per block, scaled to
    mean 1, multiplies the weight of every observation of the block by it,
    and fits. With a loss it is the loss-likelihood bootstrap of Lyddon,
    Holmes and Walker (2019); with one position per block, the Bayesian
    bootstrap of Rubin (1981) and the weighted likelihood bootstrap of Newton
    and Raftery (1994). The observed values never change.

    Attributes
    ----------
    kind : "bayesian"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    starts : int
        Starts of the fit and of each replicate (default 1): the fit, then
        rows of a Latin hypercube over the bounds.
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["bayesian"] = Field(default="bayesian", description="The bootstrap.")
    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    starts: int = Field(default=1, description="Starts per fit.")
```

## ResidualBootstrap

Bases: `UncertaintyMethod[BootstrapResult]`, `RangeChecked`

The residual block bootstrap.

Each replicate joins blocks of residuals of the fit, drawn with replacement across every series at once, adds them to the fitted values on `residual_scale`, and fits the synthetic data.

Attributes:

| Name             | Type                         | Description                                                                                                                                                                              |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`           | `residual`                   | The bootstrap, which the engine reads.                                                                                                                                                   |
| `replicates`     | `int`                        | Number of replicates, at least 1.                                                                                                                                                        |
| `block_length`   | \`int                        | auto\`                                                                                                                                                                                   |
| `residual_scale` | `{additive, multiplicative}` | "additive": r = y − ŷ and ŷ + r\*. "multiplicative": r = log(y / ŷ) and ŷ · exp(r\*); it needs positive observed and fitted values and keeps synthetic values positive.                  |
| `scheme`         | `{moving, stationary}`       | "moving": blocks of block_length consecutive positions (Künsch 1989). "stationary": blocks of geometric length with mean block_length, wrapped around the end (Politis and Romano 1994). |
| `starts`         | `int`                        | Starts of the fit and of each replicate (default 1).                                                                                                                                     |

Source code in `commol/resampling/methods.py`

```python
class ResidualBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The residual block bootstrap.

    Each replicate joins blocks of residuals of the fit, drawn with
    replacement across every series at once, adds them to the fitted values
    on `residual_scale`, and fits the synthetic data.

    Attributes
    ----------
    kind : "residual"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block; with ``scheme="stationary"`` the mean length.
    residual_scale : {"additive", "multiplicative"}
        ``"additive"``: ``r = y − ŷ`` and ``ŷ + r*``. ``"multiplicative"``:
        ``r = log(y / ŷ)`` and ``ŷ · exp(r*)``; it needs positive observed and
        fitted values and keeps synthetic values positive.
    scheme : {"moving", "stationary"}
        ``"moving"``: blocks of `block_length` consecutive positions (Künsch
        1989). ``"stationary"``: blocks of geometric length with mean
        `block_length`, wrapped around the end (Politis and Romano 1994).
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["residual"] = Field(default="residual", description="The bootstrap.")
    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    residual_scale: Literal["additive", "multiplicative"] = Field(
        description="The scale of the residuals."
    )
    scheme: Literal["moving", "stationary"] = Field(
        description="How blocks of residuals are drawn."
    )
    starts: int = Field(default=1, description="Starts per fit.")
```

## WildBootstrap

Bases: `UncertaintyMethod[BootstrapResult]`, `RangeChecked`

The wild block bootstrap.

Each replicate multiplies the additive residuals of each block by one draw of `multipliers` and fits `ŷ + r · v`. Every residual stays at its own position, so it holds under noise whose size changes along the series.

Attributes:

| Name           | Type                   | Description                                                                                                                                                      |
| -------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`         | `wild`                 | The bootstrap, which the engine reads.                                                                                                                           |
| `replicates`   | `int`                  | Number of replicates, at least 1.                                                                                                                                |
| `block_length` | \`int                  | auto\`                                                                                                                                                           |
| `multipliers`  | `{rademacher, mammen}` | "rademacher": ±1 with probability 1/2. "mammen": two values with mean 0, variance 1 and third moment 1 (Mammen 1993), which keeps the skewness of the residuals. |
| `starts`       | `int`                  | Starts of the fit and of each replicate (default 1).                                                                                                             |

Source code in `commol/resampling/methods.py`

```python
class WildBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The wild block bootstrap.

    Each replicate multiplies the additive residuals of each block by one
    draw of `multipliers` and fits ``ŷ + r · v``. Every residual stays at its
    own position, so it holds under noise whose size changes along the
    series.

    Attributes
    ----------
    kind : "wild"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    block_length : int | "auto"
        Positions per block.
    multipliers : {"rademacher", "mammen"}
        ``"rademacher"``: ±1 with probability 1/2. ``"mammen"``: two values
        with mean 0, variance 1 and third moment 1 (Mammen 1993), which keeps
        the skewness of the residuals.
    starts : int
        Starts of the fit and of each replicate (default 1).
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["wild"] = Field(default="wild", description="The bootstrap.")
    replicates: int = Field(description="Number of replicates.")
    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    multipliers: Literal["rademacher", "mammen"] = Field(
        description="The distribution of the multipliers."
    )
    starts: int = Field(default=1, description="Starts per fit.")
```

## BlockJackknife

Bases: `UncertaintyMethod[JackknifeResult]`, `RangeChecked`

The block jackknife (Künsch 1989): one fit per block left out.

It gives a standard error and a bias of each parameter, and the influence of each block, with as many fits as blocks. A block is left out by giving its observations weight 0.

Attributes:

| Name           | Type  | Description                                              |
| -------------- | ----- | -------------------------------------------------------- |
| `block_length` | \`int | auto\`                                                   |
| `starts`       | `int` | Starts of the fit and of each leave-out fit (default 1). |

Source code in `commol/resampling/methods.py`

```python
class BlockJackknife(UncertaintyMethod[JackknifeResult], RangeChecked):
    """
    The block jackknife (Künsch 1989): one fit per block left out.

    It gives a standard error and a bias of each parameter, and the influence
    of each block, with as many fits as blocks. A block is left out by giving
    its observations weight 0.

    Attributes
    ----------
    block_length : int | "auto"
        Positions per block; it must leave at least two blocks.
    starts : int
        Starts of the fit and of each leave-out fit (default 1).
    """

    _runner = staticmethod(run_block_jackknife)

    block_length: int | Literal["auto"] = Field(description="Positions per block.")
    starts: int = Field(default=1, description="Starts per fit.")
```

## ParametricBootstrap

Bases: `UncertaintyMethod[BootstrapResult]`, `RangeChecked`

The parametric bootstrap of the problem's observation models.

Each replicate draws every observed value from the model of its series at the fit, around the fitted prediction and with the fitted noise parameters, and fits the new data. It needs a problem with `observation_models`; the observations of a replicate are independent given the fit, so it assumes the models hold.

Attributes:

| Name         | Type         | Description                                                                                                   |
| ------------ | ------------ | ------------------------------------------------------------------------------------------------------------- |
| `kind`       | `parametric` | The bootstrap, which the engine reads.                                                                        |
| `replicates` | `int`        | Number of replicates, at least 1.                                                                             |
| `starts`     | `int`        | Starts of the fit and of each replicate (default 1): the fit, then rows of a Latin hypercube over the bounds. |

Source code in `commol/resampling/methods.py`

```python
class ParametricBootstrap(UncertaintyMethod[BootstrapResult], RangeChecked):
    """
    The parametric bootstrap of the problem's observation models.

    Each replicate draws every observed value from the model of its series
    at the fit, around the fitted prediction and with the fitted noise
    parameters, and fits the new data. It needs a problem with
    ``observation_models``; the observations of a replicate are independent
    given the fit, so it assumes the models hold.

    Attributes
    ----------
    kind : "parametric"
        The bootstrap, which the engine reads.
    replicates : int
        Number of replicates, at least 1.
    starts : int
        Starts of the fit and of each replicate (default 1): the fit, then
        rows of a Latin hypercube over the bounds.
    """

    _runner = staticmethod(run_bootstrap)

    kind: Literal["parametric"] = Field(
        default="parametric", description="The bootstrap."
    )
    replicates: int = Field(description="Number of replicates.")
    starts: int = Field(default=1, description="Starts per fit.")
```

## BootstrapResult

Bases: `ResultModel`

The result of a bootstrap.

Attributes:

| Name            | Type                    | Description                                                                                                                          |
| --------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `fit`           | `CalibrationResult`     | The fit every replicate starts from: Calibrator.run(starts).                                                                         |
| `sample`        | `ParameterSample`       | One draw per replicate that did not fail, with equal weights; the group of a draw is its replicate.                                  |
| `failed`        | `list[str]`             | Each replicate that failed: "Replicate <index> failed: <reason>", also logged as a warning. A failed replicate is not in the sample. |
| `not_converged` | `int`                   | Replicates kept although their optimizer did not converge.                                                                           |
| `diagnostics`   | `ResamplingDiagnostics` | The block length, and the synthetic values below zero.                                                                               |

Source code in `commol/resampling/results.py`

```python
class BootstrapResult(ResultModel):
    """
    The result of a bootstrap.

    Attributes
    ----------
    fit : CalibrationResult
        The fit every replicate starts from: ``Calibrator.run(starts)``.
    sample : ParameterSample
        One draw per replicate that did not fail, with equal weights; the
        group of a draw is its replicate.
    failed : list[str]
        Each replicate that failed: ``"Replicate <index> failed: <reason>"``,
        also logged as a warning. A failed replicate is not in the sample.
    not_converged : int
        Replicates kept although their optimizer did not converge.
    diagnostics : ResamplingDiagnostics
        The block length, and the synthetic values below zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    sample: ParameterSample
    failed: list[str] = Field(description="Each replicate that failed.")
    not_converged: int = Field(description="Replicates that did not converge.")
    diagnostics: ResamplingDiagnostics

    def interval_error(self, confidence_level: float) -> dict[str, BoundError]:
        """
        The Monte Carlo error of the central interval of each parameter.

        More replicates make it smaller; when it is wide against the interval,
        run more.

        Parameters
        ----------
        confidence_level : float
            Strictly between 0 and 1.

        Returns
        -------
        dict[str, BoundError]
            The error of each bound, by parameter.

        Raises
        ------
        ValueError
            If the confidence level is not strictly between 0 and 1.
        """
        low, high = commol_rs.inference.interval_levels(confidence_level)
        errors: dict[str, BoundError] = {}
        for index, name in enumerate(self.sample.names):
            column = [row[index] for row in self.sample.values]
            errors[name] = BoundError(
                lower=commol_rs.inference.quantile_error(column, low),
                upper=commol_rs.inference.quantile_error(column, high),
            )
        return errors
```

### Methods:

#### interval_error

```python
interval_error(confidence_level: float) -> dict[str, BoundError]
```

The Monte Carlo error of the central interval of each parameter.

More replicates make it smaller; when it is wide against the interval, run more.

Parameters:

| Name               | Type    | Description               | Default    |
| ------------------ | ------- | ------------------------- | ---------- |
| `confidence_level` | `float` | Strictly between 0 and 1. | *required* |

Returns:

| Type                    | Description                            |
| ----------------------- | -------------------------------------- |
| `dict[str, BoundError]` | The error of each bound, by parameter. |

Raises:

| Type         | Description                                              |
| ------------ | -------------------------------------------------------- |
| `ValueError` | If the confidence level is not strictly between 0 and 1. |

Source code in `commol/resampling/results.py`

```python
def interval_error(self, confidence_level: float) -> dict[str, BoundError]:
    """
    The Monte Carlo error of the central interval of each parameter.

    More replicates make it smaller; when it is wide against the interval,
    run more.

    Parameters
    ----------
    confidence_level : float
        Strictly between 0 and 1.

    Returns
    -------
    dict[str, BoundError]
        The error of each bound, by parameter.

    Raises
    ------
    ValueError
        If the confidence level is not strictly between 0 and 1.
    """
    low, high = commol_rs.inference.interval_levels(confidence_level)
    errors: dict[str, BoundError] = {}
    for index, name in enumerate(self.sample.names):
        column = [row[index] for row in self.sample.values]
        errors[name] = BoundError(
            lower=commol_rs.inference.quantile_error(column, low),
            upper=commol_rs.inference.quantile_error(column, high),
        )
    return errors
```

## JackknifeResult

Bases: `ResultModel`

The result of a block jackknife.

With `B` fits `θ̂₍₋ᵦ₎` that leave one block out, around the fit `θ̂`, and their mean `θ̄`: the bias is `(B − 1)(θ̄ − θ̂)`, the standard error `√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)`, and the pseudo-value of block `b` is `B θ̂ − (B − 1) θ̂₍₋ᵦ₎`. The pseudo-values are not draws: their percentiles are not an interval.

Attributes:

| Name             | Type                        | Description                                                                                                                                                                      |
| ---------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit`            | `CalibrationResult`         | The fit with every observation.                                                                                                                                                  |
| `estimate`       | `dict[str, float]`          | The bias-corrected estimate of each parameter.                                                                                                                                   |
| `bias`           | `dict[str, float]`          | The jackknife bias of each parameter.                                                                                                                                            |
| `standard_error` | `dict[str, float]`          | The jackknife standard error of each parameter.                                                                                                                                  |
| `blocks`         | `list[tuple[float, float]]` | The first and last position of each block that was left out and fitted, in order.                                                                                                |
| `pseudo_values`  | `dict[str, list[float]]`    | The pseudo-values of each parameter, one per block.                                                                                                                              |
| `influence`      | `dict[str, list[float]]`    | How far leaving each block out moves each parameter, in standard errors: (θ̂₍₋ᵦ₎ − θ̂) / se, one per block. NaN or infinite when a standard error is 0.                            |
| `failed`         | `list[str]`                 | Each block whose fit failed: "Block <index> failed: <reason>", with blocks numbered from 0 in order of position; also logged as a warning. A block that failed is not in blocks. |
| `not_converged`  | `int`                       | Fits kept although their optimizer did not converge.                                                                                                                             |
| `diagnostics`    | `ResamplingDiagnostics`     | The block length.                                                                                                                                                                |

Source code in `commol/resampling/results.py`

```python
class JackknifeResult(ResultModel):
    """
    The result of a block jackknife.

    With ``B`` fits ``θ̂₍₋ᵦ₎`` that leave one block out, around the fit
    ``θ̂``, and their mean ``θ̄``: the bias is ``(B − 1)(θ̄ − θ̂)``, the
    standard error ``√((B − 1)/B · Σ (θ̂₍₋ᵦ₎ − θ̄)²)``, and the pseudo-value of
    block ``b`` is ``B θ̂ − (B − 1) θ̂₍₋ᵦ₎``. The pseudo-values are not draws:
    their percentiles are not an interval.

    Attributes
    ----------
    fit : CalibrationResult
        The fit with every observation.
    estimate : dict[str, float]
        The bias-corrected estimate of each parameter.
    bias : dict[str, float]
        The jackknife bias of each parameter.
    standard_error : dict[str, float]
        The jackknife standard error of each parameter.
    blocks : list[tuple[float, float]]
        The first and last position of each block that was left out and
        fitted, in order.
    pseudo_values : dict[str, list[float]]
        The pseudo-values of each parameter, one per block.
    influence : dict[str, list[float]]
        How far leaving each block out moves each parameter, in standard
        errors: ``(θ̂₍₋ᵦ₎ − θ̂) / se``, one per block. NaN or infinite when a
        standard error is 0.
    failed : list[str]
        Each block whose fit failed: ``"Block <index> failed: <reason>"``,
        with blocks numbered from 0 in order of position; also logged as a
        warning. A block that failed is not in `blocks`.
    not_converged : int
        Fits kept although their optimizer did not converge.
    diagnostics : ResamplingDiagnostics
        The block length.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fit: CalibrationResult
    estimate: dict[str, float]
    bias: dict[str, float]
    standard_error: dict[str, float]
    blocks: list[tuple[float, float]]
    pseudo_values: dict[str, list[float]]
    influence: dict[str, list[float]]
    failed: list[str] = Field(description="Each block whose fit failed.")
    not_converged: int = Field(description="Fits that did not converge.")
    diagnostics: ResamplingDiagnostics
```

## ResamplingDiagnostics

Bases: `ResultModel`

The blocks of a resampling method, and the synthetic values below zero.

Attributes:

| Name                     | Type               | Description                                                                                                                       |
| ------------------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `block_length`           | \`int              | None\`                                                                                                                            |
| `block_length_estimates` | \`dict[str, float] | None\`                                                                                                                            |
| `negative_values`        | `int`              | Synthetic observed values below zero, over every replicate. They are kept; many of them point to residual_scale="multiplicative". |

Source code in `commol/resampling/results.py`

```python
class ResamplingDiagnostics(ResultModel):
    """
    The blocks of a resampling method, and the synthetic values below zero.

    Attributes
    ----------
    block_length : int | None
        The block length the method ran with; None for the parametric
        bootstrap, which draws no blocks.
    block_length_estimates : dict[str, float] | None
        With ``block_length="auto"``, the estimate of each series, by label;
        otherwise None.
    negative_values : int
        Synthetic observed values below zero, over every replicate. They are
        kept; many of them point to ``residual_scale="multiplicative"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_length: int | None = Field(description="The block length used.")
    block_length_estimates: dict[str, float] | None = Field(
        description="The estimate of each series with the automatic length."
    )
    negative_values: int = Field(description="Synthetic values below zero.")
```

## BoundError

Bases: `ResultModel`

The Monte Carlo error of the bounds of an interval: for each bound, an order-statistic interval that holds the bound of infinitely many replicates with about 95 % probability.

Attributes:

| Name    | Type                  | Description                            |
| ------- | --------------------- | -------------------------------------- |
| `lower` | `tuple[float, float]` | The error interval of the lower bound. |
| `upper` | `tuple[float, float]` | The error interval of the upper bound. |

Source code in `commol/resampling/results.py`

```python
class BoundError(ResultModel):
    """
    The Monte Carlo error of the bounds of an interval: for each bound, an
    order-statistic interval that holds the bound of infinitely many
    replicates with about 95 % probability.

    Attributes
    ----------
    lower : tuple[float, float]
        The error interval of the lower bound.
    upper : tuple[float, float]
        The error interval of the upper bound.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: tuple[float, float]
    upper: tuple[float, float]
```
