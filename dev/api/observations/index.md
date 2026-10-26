# Observations API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.observations`: how observed values scatter around the model. The observation models of a calibration problem, and the observation noise that a predictive adds to its draws. Guide: [Observation Models](https://munqu.github.io/commol/latest/guide/uncertainty/observation-models/index.md).

## NormalObservations

Bases: `_ObservationModel`, `RangeChecked`

Normal noise: `y ~ N(μ, sd²)`, with `μ` the prediction.

For continuous values with errors of the same size everywhere. With a known `sd` the maximum-likelihood fit is the least-squares fit.

Attributes:

| Name | Type    | Description |
| ---- | ------- | ----------- |
| `sd` | \`float | str\`       |

Source code in `commol/observations/models.py`

```python
class NormalObservations(_ObservationModel, RangeChecked):
    """
    Normal noise: ``y ~ N(μ, sd²)``, with ``μ`` the prediction.

    For continuous values with errors of the same size everywhere. With a
    known ``sd`` the maximum-likelihood fit is the least-squares fit.

    Attributes
    ----------
    sd : float | str
        The standard deviation, above 0, or the id of a noise parameter.
    """

    kind: Literal["normal"] = "normal"
    sd: float | str = Field(description="The standard deviation.")
```

## LogNormalObservations

Bases: `_ObservationModel`, `RangeChecked`

Log-normal noise: `log y` is normal with standard deviation `sigma`.

For positive values with errors that grow with the value. `center` says what the prediction `μ` is, and has no default: `"mean"` makes `E[y] = μ` (`log y ~ N(log μ − σ²/2, σ²)`), `"median"` makes the median `μ` (`log y ~ N(log μ, σ²)`, the proportional error `y = μ·e^ε`). Every observed value must be above 0.

Attributes:

| Name     | Type             | Description             |
| -------- | ---------------- | ----------------------- |
| `sigma`  | \`float          | str\`                   |
| `center` | `{mean, median}` | What the prediction is. |

Source code in `commol/observations/models.py`

```python
class LogNormalObservations(_ObservationModel, RangeChecked):
    """
    Log-normal noise: ``log y`` is normal with standard deviation ``sigma``.

    For positive values with errors that grow with the value. ``center``
    says what the prediction ``μ`` is, and has no default: ``"mean"`` makes
    ``E[y] = μ`` (``log y ~ N(log μ − σ²/2, σ²)``), ``"median"`` makes the
    median ``μ`` (``log y ~ N(log μ, σ²)``, the proportional error
    ``y = μ·e^ε``). Every observed value must be above 0.

    Attributes
    ----------
    sigma : float | str
        The standard deviation of ``log y``, above 0, or the id of a noise
        parameter.
    center : {"mean", "median"}
        What the prediction is.
    """

    kind: Literal["log_normal"] = "log_normal"
    sigma: float | str = Field(description="The standard deviation of log y.")
    center: Literal["mean", "median"] = Field(description="What the prediction is.")
```

## StudentTObservations

Bases: `_ObservationModel`, `RangeChecked`

Student-t noise of center `μ`, the prediction: normal noise with heavier tails, so an outlier pulls the fit less.

Attributes:

| Name    | Type    | Description |
| ------- | ------- | ----------- |
| `df`    | \`float | str\`       |
| `scale` | \`float | str\`       |

Source code in `commol/observations/models.py`

```python
class StudentTObservations(_ObservationModel, RangeChecked):
    """
    Student-t noise of center ``μ``, the prediction: normal noise with heavier
    tails, so an outlier pulls the fit less.

    Attributes
    ----------
    df : float | str
        The degrees of freedom, above 0, or the id of a noise parameter. The
        noise has a mean when ``df > 1`` and a variance when ``df > 2``.
    scale : float | str
        The scale, above 0, or the id of a noise parameter.
    """

    kind: Literal["student_t"] = "student_t"
    df: float | str = Field(description="The degrees of freedom.")
    scale: float | str = Field(description="The scale.")
```

## PoissonObservations

Bases: `_ObservationModel`

Poisson counts of mean `μ`, the prediction: variance equal to the mean.

Every observed value must be a whole number, at least 0.

Source code in `commol/observations/models.py`

```python
class PoissonObservations(_ObservationModel):
    """
    Poisson counts of mean ``μ``, the prediction: variance equal to the mean.

    Every observed value must be a whole number, at least 0.
    """

    kind: Literal["poisson"] = "poisson"
```

## NegativeBinomialObservations

Bases: `_ObservationModel`, `RangeChecked`

Negative binomial counts of mean `μ`, the prediction, and variance `μ + φμ²`: counts that scatter more than a Poisson.

`φ = 0` is the Poisson, so bounds `[0, …]` of a noise parameter hold it. The dispersion `k` (`size` in R) is `1/φ`. Every observed value must be a whole number, at least 0.

Attributes:

| Name             | Type    | Description |
| ---------------- | ------- | ----------- |
| `overdispersion` | \`float | str\`       |

Source code in `commol/observations/models.py`

```python
class NegativeBinomialObservations(_ObservationModel, RangeChecked):
    """
    Negative binomial counts of mean ``μ``, the prediction, and variance
    ``μ + φμ²``: counts that scatter more than a Poisson.

    ``φ = 0`` is the Poisson, so bounds ``[0, …]`` of a noise parameter hold
    it. The dispersion ``k`` (``size`` in R) is ``1/φ``.
    Every observed value must be a whole number, at least 0.

    Attributes
    ----------
    overdispersion : float | str
        ``φ``, at least 0, or the id of a noise parameter.
    """

    kind: Literal["negative_binomial"] = "negative_binomial"
    overdispersion: float | str = Field(description="The overdispersion φ.")
```

## BinomialObservations

Bases: `_ObservationModel`, `RangeChecked`

Binomial counts of `n` trials and mean `μ`, the prediction: the probability is `μ / n`, so the prediction is an expected count, not a proportion.

`trials` gives `n` for every observation of the series; without it, each observation of the series gives its own (`ObservedDataPoint.trials`), for trials that change from one observation to the next. One source per series. Every observed value must be a whole number between 0 and its trials.

Attributes:

| Name     | Type  | Description |
| -------- | ----- | ----------- |
| `trials` | \`int | None\`      |

Source code in `commol/observations/models.py`

```python
class BinomialObservations(_ObservationModel, RangeChecked):
    """
    Binomial counts of ``n`` trials and mean ``μ``, the prediction: the
    probability is ``μ / n``, so the prediction is an expected count, not a
    proportion.

    ``trials`` gives ``n`` for every observation of the series; without it,
    each observation of the series gives its own (``ObservedDataPoint.trials``),
    for trials that change from one observation to the next. One source per
    series. Every observed value must be a whole number between 0 and its
    trials.

    Attributes
    ----------
    trials : int | None
        The trials of every observation, or None (default) to read the
        trials of each observation.
    """

    kind: Literal["binomial"] = "binomial"
    trials: int | None = Field(default=None, description="The trials.")
```

## NoNoise

Bases: `BaseModel`

No observation noise: an observation shows the prediction of each draw.

The band of a predictive with `NoNoise` is a band of the model. It says where the model curve lies, not where the next observation falls, so its coverage of data is below its level.

Attributes:

| Name   | Type   | Description                        |
| ------ | ------ | ---------------------------------- |
| `kind` | `none` | The noise, which the engine reads. |

Source code in `commol/observations/noise.py`

```python
class NoNoise(BaseModel):
    """
    No observation noise: an observation shows the prediction of each draw.

    The band of a predictive with `NoNoise` is a band of the model. It says
    where the model curve lies, not where the next observation falls, so its
    coverage of data is below its level.

    Attributes
    ----------
    kind : "none"
        The noise, which the engine reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["none"] = Field(default="none", description="The noise.")
```

## ResidualNoise

Bases: `RangeChecked`

Observation noise made of the residuals of a fit, with no noise model.

Each draw adds moving blocks of `block_length` positions of residuals of the fit, drawn with replacement across every series at once, to its predictions, on `residual_scale`. The band is then a prediction interval. It serves the series the fit has residuals of, at any position: the residual of a series at a drawn position is the one at the nearest position where the series was observed.

Attributes:

| Name             | Type                         | Description                                                                                                                                                            |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`           | `residual`                   | The noise, which the engine reads.                                                                                                                                     |
| `fit`            | `CalibrationResult`          | The fit whose residuals are added.                                                                                                                                     |
| `block_length`   | `int`                        | Positions per block, at least 1 and at most the number of distinct positions of the problem's observations. A bootstrap reports the length it used in its diagnostics. |
| `residual_scale` | `{additive, multiplicative}` | "additive" adds r = y − ŷ; "multiplicative" multiplies by exp(r) with r = log(y / ŷ) and needs positive observed and fitted values.                                    |

Source code in `commol/observations/noise.py`

```python
class ResidualNoise(RangeChecked):
    """
    Observation noise made of the residuals of a fit, with no noise model.

    Each draw adds moving blocks of `block_length` positions of residuals of
    the fit, drawn with replacement across every series at once, to its
    predictions, on `residual_scale`. The band is then a prediction interval.
    It serves the series the fit has residuals of, at any position: the
    residual of a series at a drawn position is the one at the nearest
    position where the series was observed.

    Attributes
    ----------
    kind : "residual"
        The noise, which the engine reads.
    fit : CalibrationResult
        The fit whose residuals are added.
    block_length : int
        Positions per block, at least 1 and at most the number of distinct
        positions of the problem's observations. A bootstrap reports the
        length it used in its diagnostics.
    residual_scale : {"additive", "multiplicative"}
        ``"additive"`` adds ``r = y − ŷ``; ``"multiplicative"`` multiplies by
        ``exp(r)`` with ``r = log(y / ŷ)`` and needs positive observed and
        fitted values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["residual"] = Field(default="residual", description="The noise.")
    fit: CalibrationResult = Field(description="The fit whose residuals are added.")
    block_length: int = Field(description="Positions per block.")
    residual_scale: Literal["additive", "multiplicative"] = Field(
        description="The scale of the residuals."
    )
```

## DistributionNoise

Bases: `BaseModel`

Observation noise of the problem's observation models.

Each draw draws every observation from the model of its series around the draw's prediction, with the draw's noise parameters, so the band is a prediction interval of the posterior or bootstrap predictive. It needs a problem with `observation_models`, and serves the series they name. A prediction outside the means a model allows, such as a negative count, draws at the nearest allowed one.

Attributes:

| Name   | Type           | Description                        |
| ------ | -------------- | ---------------------------------- |
| `kind` | `distribution` | The noise, which the engine reads. |

Source code in `commol/observations/noise.py`

```python
class DistributionNoise(BaseModel):
    """
    Observation noise of the problem's observation models.

    Each draw draws every observation from the model of its series around
    the draw's prediction, with the draw's noise parameters, so the band is a
    prediction interval of the posterior or bootstrap predictive. It needs a
    problem with ``observation_models``, and serves the series they name. A
    prediction outside the means a model allows, such as a negative count,
    draws at the nearest allowed one.

    Attributes
    ----------
    kind : "distribution"
        The noise, which the engine reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["distribution"] = Field(
        default="distribution", description="The noise."
    )
```
