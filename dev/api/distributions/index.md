# Distributions API

[Back to API Reference](https://munqu.github.io/commol/latest/api/index.md)

The module `commol.distributions`: the distributions of commol, and the priors of a sampling method. Guide: [Sequential Monte Carlo](https://munqu.github.io/commol/latest/guide/uncertainty/smc/index.md).

## Priors

Bases: `BaseModel`

One prior per calibrated parameter.

A sampling method takes its priors with no default: what is known about each parameter before the data is a modelling choice. Give a mapping of parameter to distribution, or `uniform_on_bounds` for `Uniform()` on every parameter.

Attributes:

| Name            | Type                      | Description |
| --------------- | ------------------------- | ----------- |
| `distributions` | \`dict[str, distribution] | None\`      |

Source code in `commol/distributions/distributions.py`

```python
class Priors(BaseModel):
    """
    One prior per calibrated parameter.

    A sampling method takes its priors with no default: what is known about
    each parameter before the data is a modelling choice. Give a mapping of
    parameter to distribution, or `uniform_on_bounds` for ``Uniform()`` on
    every parameter.

    Attributes
    ----------
    distributions : dict[str, distribution] | None
        The prior of each parameter; None gives ``Uniform()`` to every
        parameter (`uniform_on_bounds`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    distributions: dict[str, _Distribution] | None = Field(
        description="The prior of each parameter, or None for uniform on the bounds."
    )

    @classmethod
    def uniform_on_bounds(cls) -> Self:
        """
        ``Uniform()`` on the bounds of every parameter.

        Returns
        -------
        Priors
            The priors; every parameter needs finite bounds.
        """
        return cls(distributions=None)
```

### Methods:

#### uniform_on_bounds

```python
uniform_on_bounds() -> Self
```

`Uniform()` on the bounds of every parameter.

Returns:

| Type     | Description                                      |
| -------- | ------------------------------------------------ |
| `Priors` | The priors; every parameter needs finite bounds. |

Source code in `commol/distributions/distributions.py`

```python
@classmethod
def uniform_on_bounds(cls) -> Self:
    """
    ``Uniform()`` on the bounds of every parameter.

    Returns
    -------
    Priors
        The priors; every parameter needs finite bounds.
    """
    return cls(distributions=None)
```

## Uniform

Bases: `DistributionBase`

The uniform distribution on the bounds; they must be finite.

Source code in `commol/distributions/distributions.py`

```python
class Uniform(DistributionBase):
    """The uniform distribution on the bounds; they must be finite."""

    kind: Literal["uniform"] = "uniform"
```

## LogUniform

Bases: `DistributionBase`

The log-uniform distribution on the bounds: `log x` is uniform, the density is proportional to `1/x`. The bounds must be finite and the lower one above 0.

Source code in `commol/distributions/distributions.py`

```python
class LogUniform(DistributionBase):
    """
    The log-uniform distribution on the bounds: ``log x`` is uniform, the
    density is proportional to ``1/x``. The bounds must be finite and the
    lower one above 0.
    """

    kind: Literal["log_uniform"] = "log_uniform"
```

## Beta

Bases: `DistributionBase`, `RangeChecked`

The beta distribution of shapes `a` and `b`, scaled to the bounds; they must be finite.

Attributes:

| Name | Type    | Description                |
| ---- | ------- | -------------------------- |
| `a`  | `float` | The first shape, above 0.  |
| `b`  | `float` | The second shape, above 0. |

Source code in `commol/distributions/distributions.py`

```python
class Beta(DistributionBase, RangeChecked):
    """
    The beta distribution of shapes `a` and `b`, scaled to the bounds; they
    must be finite.

    Attributes
    ----------
    a : float
        The first shape, above 0.
    b : float
        The second shape, above 0.
    """

    kind: Literal["beta"] = "beta"
    a: float = Field(description="The first shape.")
    b: float = Field(description="The second shape.")
```

## Normal

Bases: `DistributionBase`, `RangeChecked`

The normal distribution, truncated to the bounds.

Attributes:

| Name   | Type    | Description                                        |
| ------ | ------- | -------------------------------------------------- |
| `mean` | `float` | The mean before truncation.                        |
| `sd`   | `float` | The standard deviation before truncation, above 0. |

Source code in `commol/distributions/distributions.py`

```python
class Normal(DistributionBase, RangeChecked):
    """
    The normal distribution, truncated to the bounds.

    Attributes
    ----------
    mean : float
        The mean before truncation.
    sd : float
        The standard deviation before truncation, above 0.
    """

    kind: Literal["normal"] = "normal"
    mean: float = Field(description="The mean.")
    sd: float = Field(description="The standard deviation.")
```

## LogNormal

Bases: `DistributionBase`, `RangeChecked`

The log-normal distribution, truncated to the bounds and to `(0, ∞)`: `log x` is normal. The upper bound must be above 0.

Attributes:

| Name    | Type    | Description                               |
| ------- | ------- | ----------------------------------------- |
| `mu`    | `float` | The mean of log x.                        |
| `sigma` | `float` | The standard deviation of log x, above 0. |

Source code in `commol/distributions/distributions.py`

```python
class LogNormal(DistributionBase, RangeChecked):
    """
    The log-normal distribution, truncated to the bounds and to ``(0, ∞)``:
    ``log x`` is normal. The upper bound must be above 0.

    Attributes
    ----------
    mu : float
        The mean of ``log x``.
    sigma : float
        The standard deviation of ``log x``, above 0.
    """

    kind: Literal["log_normal"] = "log_normal"
    mu: float = Field(description="The mean of log x.")
    sigma: float = Field(description="The standard deviation of log x.")
```

## Gamma

Bases: `DistributionBase`, `RangeChecked`

The gamma distribution of shape `shape` and rate `rate` (mean `shape / rate`), truncated to the bounds and to `(0, ∞)`. The upper bound must be above 0.

Attributes:

| Name    | Type    | Description         |
| ------- | ------- | ------------------- |
| `shape` | `float` | The shape, above 0. |
| `rate`  | `float` | The rate, above 0.  |

Source code in `commol/distributions/distributions.py`

```python
class Gamma(DistributionBase, RangeChecked):
    """
    The gamma distribution of shape `shape` and rate `rate` (mean
    ``shape / rate``), truncated to the bounds and to ``(0, ∞)``. The upper
    bound must be above 0.

    Attributes
    ----------
    shape : float
        The shape, above 0.
    rate : float
        The rate, above 0.
    """

    kind: Literal["gamma"] = "gamma"
    shape: float = Field(description="The shape.")
    rate: float = Field(description="The rate.")
```
